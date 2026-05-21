#!/usr/bin/env python3
"""bpd_yolo_infer.py — BPD substrate YOLOv5n end-to-end inference on real images.

The final spike for our first model runner, written by us.

Pipeline:
  1. Load JPG image (any aspect ratio)
  2. Letterbox preprocess: resize maintaining aspect ratio + pad with (114,114,114) to 640x640
  3. Normalize /255.0, transpose HWC→CHW, add batch dim (BCHW float32)
  4. Forward through all 24 YOLOv5n layers via substrate primitives (bit-identical with PyTorch)
  5. Detect head: anchor decoding + sigmoid + box scaling
  6. Decode predictions: score = obj_conf × cls_conf; argmax over classes; threshold filter
  7. NMS via torchvision.ops.nms (deterministic reference)
  8. Scale boxes from letterbox-640 back to original image coordinates
  9. Save .npz in canonical format: boxes (N,4) xyxy float32, confidence (N,), class_ids (N,)

This module supports both substrate (bpd) and reference (pytorch) backends so we can
generate the proper YOLOv5 reference vectors AND the BPD vectors with the SAME pipeline.
The only difference between paths is whether the forward+detect use substrate primitives
or inline PyTorch nn.Modules.

Per Medayek's spec and Heath's framing: "We are about to have a model runner, written by us."
"""
import argparse
import ctypes
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nms(boxes, scores, iou_threshold):
    """Greedy NMS — same algorithm as torchvision.ops.nms.
    
    boxes: (N, 4) tensor [x1, y1, x2, y2]
    scores: (N,) tensor
    Returns indices of kept boxes, sorted by score descending.
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.int64)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64)

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent))

from yolo_forward import (
    load_yolov5n_weights,
    run_cbs, run_cN, run_sppf, run_upsample, run_concat, run_maxpool2d, run_detect,
    get_layer_weights, get_cN_weights,
)


# ── Inline PyTorch reference modules ──
# Same as in verify_yolo_composition_sweep.py — the bit-identity ground truth.

class _CBS(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        c_ = c2
        self.cv1 = _CBS(c1, c_, 1, 1, 0)
        self.cv2 = _CBS(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class _C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = _CBS(c1, c_, 1, 1, 0)
        self.cv2 = _CBS(c1, c_, 1, 1, 0)
        self.cv3 = _CBS(2 * c_, c2, 1, 1, 0)
        self.m = nn.Sequential(*[_Bottleneck(c_, c_, shortcut) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class _SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = _CBS(c1, c_, 1, 1, 0)
        self.cv2 = _CBS(c_ * 4, c2, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x); y2 = self.m(y1); y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class _Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.stride = None

    def _make_grid(self, nx, ny, i):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y = torch.arange(ny, device=d, dtype=t)
        x = torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand(shape)
        return grid, anchor_grid

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]
            wh = (wh * 2) ** 2 * self.anchor_grid[i]
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))
        return torch.cat(z, 1), x


# ── YOLOv5n architecture spec ──

YOLOV5N_LAYERS = [
    (0, 'cbs', {'c1':3, 'c2':16, 'k':6, 's':2, 'p':2}),
    (1, 'cbs', {'c1':16, 'c2':32, 'k':3, 's':2, 'p':1}),
    (2, 'c3', {'c1':32, 'c2':32, 'n':1, 'shortcut':True}),
    (3, 'cbs', {'c1':32, 'c2':64, 'k':3, 's':2, 'p':1}),
    (4, 'c3', {'c1':64, 'c2':64, 'n':2, 'shortcut':True}),
    (5, 'cbs', {'c1':64, 'c2':128, 'k':3, 's':2, 'p':1}),
    (6, 'c3', {'c1':128, 'c2':128, 'n':3, 'shortcut':True}),
    (7, 'cbs', {'c1':128, 'c2':256, 'k':3, 's':2, 'p':1}),
    (8, 'c3', {'c1':256, 'c2':256, 'n':1, 'shortcut':True}),
    (9, 'sppf', {'c1':256, 'c2':256, 'k':5}),
    (10, 'cbs', {'c1':256, 'c2':128, 'k':1, 's':1, 'p':0}),
    (11, 'upsample', {}),
    (12, 'concat', {'from': [6]}),
    (13, 'c3', {'c1':256, 'c2':128, 'n':1, 'shortcut':False}),
    (14, 'cbs', {'c1':128, 'c2':64, 'k':1, 's':1, 'p':0}),
    (15, 'upsample', {}),
    (16, 'concat', {'from': [4]}),
    (17, 'c3', {'c1':128, 'c2':64, 'n':1, 'shortcut':False}),
    (18, 'cbs', {'c1':64, 'c2':64, 'k':3, 's':2, 'p':1}),
    (19, 'concat', {'from': [14]}),
    (20, 'c3', {'c1':128, 'c2':128, 'n':1, 'shortcut':False}),
    (21, 'cbs', {'c1':128, 'c2':128, 'k':3, 's':2, 'p':1}),
    (22, 'concat', {'from': [10]}),
    (23, 'c3', {'c1':256, 'c2':256, 'n':1, 'shortcut':False}),
]

YOLOV5_ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]
YOLOV5_STRIDES = [8.0, 16.0, 32.0]
YOLOV5_NC = 80
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def setup_lib():
    so = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")
    lib = ctypes.CDLL(so)
    lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
    lib.bpd_conv2d_full_cpu.restype = None
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
    lib.bpd_conv2d_cpu.restype = None
    lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.bpd_batchnorm_cpu_affine_fused.restype = None
    lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_silu_cpu.restype = None
    lib.bpd_residual_add_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]
    lib.bpd_residual_add_cpu.restype = None
    lib.bpd_concat_channel_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p]
    lib.bpd_concat_channel_cpu.restype = None
    lib.bpd_maxpool2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*8
    lib.bpd_maxpool2d_cpu.restype = None
    lib.bpd_upsample_nearest2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
    lib.bpd_upsample_nearest2d_cpu.restype = None
    lib.bpd_sigmoid_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
    lib.bpd_sigmoid_cpu.restype = None
    # Phase 3.1 F3 fused kernel (optional — present in builds with F3 landed)
    if hasattr(lib, 'bpd_conv2d_bn_silu_fused_cpu'):
        lib.bpd_conv2d_bn_silu_fused_cpu.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*11
        lib.bpd_conv2d_bn_silu_fused_cpu.restype = None
    return lib


# ── Preprocessing ──

def preprocess_image(img_path, input_size=640):
    """Letterbox preprocess: resize maintaining aspect ratio, pad to (input_size, input_size).
    
    Matches the yolo_ref_v2.py preprocessing used to generate medayek's reference.
    Returns (preprocessed_array, original_size, scale, pad_xy).
    """
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    w0, h0 = img.size
    scale = min(input_size / w0, input_size / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded = Image.new('RGB', (input_size, input_size), (114, 114, 114))
    padded.paste(img_resized, (pad_w, pad_h))
    arr = np.array(padded, dtype=np.float32) / 255.0  # HWC float32 in [0,1]
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # add batch
    return np.ascontiguousarray(arr), (w0, h0), scale, (pad_w, pad_h)


def scale_boxes_from_letterbox(boxes_xyxy, orig_size, scale, pad_xy, input_size=640):
    """Map boxes from letterbox-640 space back to original image coordinates.
    
    boxes_xyxy: (N, 4) tensor or array
    orig_size: (w0, h0) original image dimensions
    scale: the letterbox scale factor
    pad_xy: (pad_w, pad_h) padding offsets
    """
    boxes = boxes_xyxy.copy() if isinstance(boxes_xyxy, np.ndarray) else boxes_xyxy.clone()
    pad_w, pad_h = pad_xy
    w0, h0 = orig_size
    # Remove padding
    boxes[..., [0, 2]] -= pad_w
    boxes[..., [1, 3]] -= pad_h
    # Reverse scale
    boxes[..., :4] /= scale
    # Clip to image bounds
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, w0) if isinstance(boxes, np.ndarray) else boxes[..., [0, 2]].clamp(0, w0)
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, h0) if isinstance(boxes, np.ndarray) else boxes[..., [1, 3]].clamp(0, h0)
    return boxes


def xywh_to_xyxy(boxes):
    """Convert center-xywh to xyxy format. boxes: (N, 4)."""
    out = boxes.copy() if isinstance(boxes, np.ndarray) else boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


# ── Postprocessing: decode raw → detections ──

def decode_inference_output(inf_output, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                              nc=YOLOV5_NC):
    """Decode YOLOv5 inference output to final detections.
    
    inf_output: (bs, total_anchors, no=85) — already-sigmoided + anchor-scaled raw predictions
    Returns: (boxes_xyxy, scores, class_ids) all numpy
    """
    # bs=1 assumption
    pred = inf_output[0]  # (total_anchors, 85)
    
    # Split: [0:4]=xywh, [4]=obj_conf, [5:85]=class_probs
    xywh = pred[:, 0:4]
    obj_conf = pred[:, 4]
    cls_probs = pred[:, 5:5+nc]
    
    # Per-class score = obj_conf × cls_prob, then max-class per anchor
    # YOLOv5 default: keep the per-class scores and let NMS deduplicate
    scores_all = cls_probs * obj_conf[:, None]  # (total_anchors, nc)
    
    # For each anchor, pick the best class
    cls_ids = scores_all.argmax(axis=1)  # (total_anchors,)
    scores = scores_all[np.arange(len(cls_ids)), cls_ids]  # (total_anchors,)
    
    # Filter by confidence
    mask = scores > conf_threshold
    if not mask.any():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    
    xywh = xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    
    # Convert xywh → xyxy
    boxes_xyxy = xywh_to_xyxy(xywh)
    
    # NMS via torchvision (deterministic reference)
    boxes_t = torch.from_numpy(boxes_xyxy)
    scores_t = torch.from_numpy(scores)
    keep = nms(boxes_t, scores_t, iou_threshold).numpy()
    
    return boxes_xyxy[keep], scores[keep], cls_ids[keep].astype(np.int32)


# ── Substrate inference path ──

def substrate_forward(x_input, weights, lib):
    """Run substrate forward pass through all 24 layers, return [P3, P4, P5]."""
    cache = {}
    x = x_input.copy()
    for layer_idx, kind, cfg in YOLOV5N_LAYERS:
        if kind == 'cbs':
            w = get_layer_weights(weights, layer_idx)
            x = run_cbs(x, w['conv_weight'], w['bn_weight'], w['bn_bias'],
                          w['bn_mean'], w['bn_var'],
                          stride=cfg['s'], pad=cfg['p'], lib=lib)
        elif kind == 'c3':
            cn_w = get_cN_weights(weights, layer_idx, cfg['n'])
            x = run_cN(x, cn_w, n=cfg['n'], shortcut=cfg.get('shortcut', True), lib=lib)
        elif kind == 'sppf':
            sppf_w = _get_sppf_weights(weights, layer_idx)
            x = run_sppf(x, sppf_w, k=cfg['k'], lib=lib)
        elif kind == 'upsample':
            x = run_upsample(x, lib=lib)
        elif kind == 'concat':
            tensors = [x] + [cache[i] for i in cfg['from']]
            x = run_concat(tensors, lib=lib)
        cache[layer_idx] = x.copy()
    return [cache[17], cache[20], cache[23]]


def _get_sppf_weights(all_weights, layer_idx):
    prefix = f"_modules.model._modules.{layer_idx}._modules."
    return {
        'cv1_weight': all_weights[f"{prefix}cv1._modules.conv._parameters.weight"],
        'cv1_bn_gamma': all_weights[f"{prefix}cv1._modules.bn._parameters.weight"],
        'cv1_bn_beta': all_weights[f"{prefix}cv1._modules.bn._parameters.bias"],
        'cv1_bn_mean': all_weights[f"{prefix}cv1._modules.bn._buffers.running_mean"],
        'cv1_bn_var': all_weights[f"{prefix}cv1._modules.bn._buffers.running_var"],
        'cv2_weight': all_weights[f"{prefix}cv2._modules.conv._parameters.weight"],
        'cv2_bn_gamma': all_weights[f"{prefix}cv2._modules.bn._parameters.weight"],
        'cv2_bn_beta': all_weights[f"{prefix}cv2._modules.bn._parameters.bias"],
        'cv2_bn_mean': all_weights[f"{prefix}cv2._modules.bn._buffers.running_mean"],
        'cv2_bn_var': all_weights[f"{prefix}cv2._modules.bn._buffers.running_var"],
    }


def _get_detect_weights(all_weights):
    base = "_modules.model._modules.24._modules.m._modules."
    return {
        'm0_weight': all_weights[f"{base}0._parameters.weight"],
        'm0_bias':   all_weights[f"{base}0._parameters.bias"],
        'm1_weight': all_weights[f"{base}1._parameters.weight"],
        'm1_bias':   all_weights[f"{base}1._parameters.bias"],
        'm2_weight': all_weights[f"{base}2._parameters.weight"],
        'm2_bias':   all_weights[f"{base}2._parameters.bias"],
    }


def substrate_infer(img_path, weights, lib):
    """Substrate end-to-end inference on one image."""
    x, orig_size, scale, pad_xy = preprocess_image(img_path, input_size=640)
    feat = substrate_forward(x, weights, lib)
    detect_weights = _get_detect_weights(weights)
    anchors = np.array(YOLOV5_ANCHORS, dtype=np.float32).reshape(3, 3, 2)
    strides = np.array(YOLOV5_STRIDES, dtype=np.float32)
    inf_output, _raw = run_detect(feat, detect_weights, anchors, strides, YOLOV5_NC, lib=lib)
    boxes, scores, cls_ids = decode_inference_output(inf_output)
    if len(boxes) > 0:
        boxes = scale_boxes_from_letterbox(boxes, orig_size, scale, pad_xy)
    return boxes.astype(np.float32), scores.astype(np.float32), cls_ids.astype(np.int32)


# ── PyTorch reference inference path ──

def _load_cbs_weights(mod, w):
    with torch.no_grad():
        mod.conv.weight.copy_(torch.from_numpy(np.ascontiguousarray(w['conv_weight'], dtype=np.float32)))
        mod.bn.weight.copy_(torch.from_numpy(np.asarray(w['bn_weight'], dtype=np.float32)))
        mod.bn.bias.copy_(torch.from_numpy(np.asarray(w['bn_bias'], dtype=np.float32)))
        mod.bn.running_mean.copy_(torch.from_numpy(np.asarray(w['bn_mean'], dtype=np.float32)))
        mod.bn.running_var.copy_(torch.from_numpy(np.asarray(w['bn_var'], dtype=np.float32)))


def _load_c3_weights(mod, all_weights, layer_idx, n):
    w = get_cN_weights(all_weights, layer_idx, n)
    for cv_name, mod_cv in [('cv1', mod.cv1), ('cv2', mod.cv2), ('cv3', mod.cv3)]:
        _load_cbs_weights(mod_cv, {
            'conv_weight': w[f'{cv_name}_conv'],
            'bn_weight': w[f'{cv_name}_bn_gamma'],
            'bn_bias': w[f'{cv_name}_bn_beta'],
            'bn_mean': w[f'{cv_name}_bn_mean'],
            'bn_var': w[f'{cv_name}_bn_var'],
        })
    for i in range(n):
        bn_w = w[f'm{i}']
        _load_cbs_weights(mod.m[i].cv1, {'conv_weight': bn_w['cv1_conv'],
            'bn_weight': bn_w['cv1_bn_gamma'], 'bn_bias': bn_w['cv1_bn_beta'],
            'bn_mean': bn_w['cv1_bn_mean'], 'bn_var': bn_w['cv1_bn_var']})
        _load_cbs_weights(mod.m[i].cv2, {'conv_weight': bn_w['cv2_conv'],
            'bn_weight': bn_w['cv2_bn_gamma'], 'bn_bias': bn_w['cv2_bn_beta'],
            'bn_mean': bn_w['cv2_bn_mean'], 'bn_var': bn_w['cv2_bn_var']})


def _load_sppf_weights(mod, sppf_w):
    _load_cbs_weights(mod.cv1, {'conv_weight': sppf_w['cv1_weight'],
        'bn_weight': sppf_w['cv1_bn_gamma'], 'bn_bias': sppf_w['cv1_bn_beta'],
        'bn_mean': sppf_w['cv1_bn_mean'], 'bn_var': sppf_w['cv1_bn_var']})
    _load_cbs_weights(mod.cv2, {'conv_weight': sppf_w['cv2_weight'],
        'bn_weight': sppf_w['cv2_bn_gamma'], 'bn_bias': sppf_w['cv2_bn_beta'],
        'bn_mean': sppf_w['cv2_bn_mean'], 'bn_var': sppf_w['cv2_bn_var']})


def pytorch_infer(img_path, weights):
    """PyTorch reference end-to-end inference on one image."""
    x, orig_size, scale, pad_xy = preprocess_image(img_path, input_size=640)
    xt = torch.from_numpy(x.copy())
    
    # Build modules and forward
    cache = {}
    for layer_idx, kind, cfg in YOLOV5N_LAYERS:
        if kind == 'cbs':
            mod = _CBS(cfg['c1'], cfg['c2'], cfg['k'], cfg['s'], cfg['p']).eval()
            w = get_layer_weights(weights, layer_idx)
            _load_cbs_weights(mod, w)
            with torch.no_grad():
                xt = mod(xt)
        elif kind == 'c3':
            mod = _C3(cfg['c1'], cfg['c2'], n=cfg['n'], shortcut=cfg.get('shortcut', True)).eval()
            _load_c3_weights(mod, weights, layer_idx, cfg['n'])
            with torch.no_grad():
                xt = mod(xt)
        elif kind == 'sppf':
            mod = _SPPF(cfg['c1'], cfg['c2'], k=cfg['k']).eval()
            _load_sppf_weights(mod, _get_sppf_weights(weights, layer_idx))
            with torch.no_grad():
                xt = mod(xt)
        elif kind == 'upsample':
            with torch.no_grad():
                xt = F.interpolate(xt, scale_factor=2, mode='nearest')
        elif kind == 'concat':
            with torch.no_grad():
                xt = torch.cat([xt] + [cache[i] for i in cfg['from']], dim=1)
        cache[layer_idx] = xt.clone()
    
    # Detect head
    detect = _Detect(nc=YOLOV5_NC, anchors=YOLOV5_ANCHORS, ch=[64, 128, 256]).eval()
    detect.stride = torch.tensor(YOLOV5_STRIDES)
    dw = _get_detect_weights(weights)
    with torch.no_grad():
        detect.m[0].weight.copy_(torch.from_numpy(np.ascontiguousarray(dw['m0_weight'], dtype=np.float32)))
        detect.m[0].bias.copy_(torch.from_numpy(np.asarray(dw['m0_bias'], dtype=np.float32)))
        detect.m[1].weight.copy_(torch.from_numpy(np.ascontiguousarray(dw['m1_weight'], dtype=np.float32)))
        detect.m[1].bias.copy_(torch.from_numpy(np.asarray(dw['m1_bias'], dtype=np.float32)))
        detect.m[2].weight.copy_(torch.from_numpy(np.ascontiguousarray(dw['m2_weight'], dtype=np.float32)))
        detect.m[2].bias.copy_(torch.from_numpy(np.asarray(dw['m2_bias'], dtype=np.float32)))
    
    feats = [cache[17], cache[20], cache[23]]
    with torch.no_grad():
        inf_output, _raw = detect(feats)
    inf_np = inf_output.numpy()
    
    boxes, scores, cls_ids = decode_inference_output(inf_np)
    if len(boxes) > 0:
        boxes = scale_boxes_from_letterbox(boxes, orig_size, scale, pad_xy)
    return boxes.astype(np.float32), scores.astype(np.float32), cls_ids.astype(np.int32)


# ── Main CLI ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['bpd', 'pytorch', 'both'], default='both')
    parser.add_argument('--weights', default='/tmp/yolo_canonical/yolov5n.pt')
    parser.add_argument('--input-dir', default='/tmp/yolo_canonical/images')
    parser.add_argument('--output-dir', default='/tmp/yolo_canonical')
    parser.add_argument('--images', nargs='*', help='Specific image IDs (default: all in input-dir)')
    args = parser.parse_args()

    print("Loading YOLOv5n weights...")
    weights = load_yolov5n_weights(args.weights)
    print(f"  {len(weights)} tensors")

    lib = None
    if args.mode in ('bpd', 'both'):
        lib = setup_lib()
        print(f"  CPU substrate library loaded")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    bpd_out_dir = output_dir / 'bpd_output'
    bpd_out_dir.mkdir(exist_ok=True)
    ref_out_dir = output_dir / 'reference'
    ref_out_dir.mkdir(exist_ok=True)

    if args.images:
        img_paths = [input_dir / f'{img_id}.jpg' for img_id in args.images]
    else:
        img_paths = sorted(input_dir.glob('*.jpg'))

    print(f"\nProcessing {len(img_paths)} images, mode={args.mode}")
    print()

    summary_bpd = {}
    summary_ref = {}

    for img_path in img_paths:
        img_id = img_path.stem
        print(f"  {img_id}...")

        if args.mode in ('pytorch', 'both'):
            t0 = time.perf_counter()
            ref_boxes, ref_scores, ref_cls = pytorch_infer(str(img_path), weights)
            ref_ms = (time.perf_counter() - t0) * 1000
            np.savez(ref_out_dir / f'{img_id}_ref.npz',
                     boxes=ref_boxes, confidence=ref_scores, class_ids=ref_cls)
            summary_ref[img_id] = {'n_detections': len(ref_boxes), 'inference_ms': round(ref_ms, 1)}
            print(f"    pytorch: {len(ref_boxes)} dets, {ref_ms:.1f}ms")

        if args.mode in ('bpd', 'both'):
            t0 = time.perf_counter()
            bpd_boxes, bpd_scores, bpd_cls = substrate_infer(str(img_path), weights, lib)
            bpd_ms = (time.perf_counter() - t0) * 1000
            np.savez(bpd_out_dir / f'{img_id}_bpd.npz',
                     boxes=bpd_boxes, confidence=bpd_scores, class_ids=bpd_cls)
            summary_bpd[img_id] = {'n_detections': len(bpd_boxes), 'inference_ms': round(bpd_ms, 1)}
            print(f"    bpd:     {len(bpd_boxes)} dets, {bpd_ms:.1f}ms")

        if args.mode == 'both' and len(ref_boxes) == len(bpd_boxes):
            # Quick comparison
            if len(ref_boxes) > 0:
                # Sort both by confidence (descending)
                ro = np.argsort(-ref_scores); bo = np.argsort(-bpd_scores)
                conf_bits_match = np.array_equal(
                    ref_scores[ro].view(np.uint32), bpd_scores[bo].view(np.uint32))
                cls_match = np.array_equal(ref_cls[ro], bpd_cls[bo])
                max_box_diff = float(np.abs(ref_boxes[ro] - bpd_boxes[bo]).max())
                print(f"    compare: classes_match={cls_match} conf_bit_identical={conf_bits_match} "
                      f"max_box_diff={max_box_diff:.4f}px")
            else:
                print(f"    compare: both empty (match)")

    # Save summaries
    if summary_ref:
        with open(output_dir / 'reference_summary.json', 'w') as f:
            json.dump({
                'model': 'yolov5n_pytorch_inline',
                'note': 'PyTorch reference using inline nn.Module mirror of yolo_forward primitives',
                'per_image': summary_ref,
            }, f, indent=2)

    if summary_bpd:
        with open(output_dir / 'bpd_summary.json', 'w') as f:
            json.dump({
                'model': 'yolov5n_bpd_substrate',
                'note': 'BPD substrate via bench/yolo_forward.py + bench/bpd_cpu.c',
                'per_image': summary_bpd,
            }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
