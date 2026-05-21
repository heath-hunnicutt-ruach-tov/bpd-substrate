#!/usr/bin/env python3
"""YOLOv5n forward pass orchestrator using BPD substrate kernels.

Loads YOLOv5n weights from .pt file (no ultralytics dependency),
runs inference through our C/CUDA kernels, produces detections.

Architecture: YOLOv5n has 24 layers, each is one of:
  Conv+BN+SiLU (the repeating unit)
  MaxPool (downsampling)
  Upsample (upsampling in FPN)
  Concat (feature fusion)
  C3 (CSP bottleneck — multiple Conv+BN+SiLU blocks)
  Detect (detection head — conv + reshape + sigmoid)
"""
import ctypes, os, sys, numpy as np

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch numpy")

# ── Weight loader (no ultralytics/cv2 needed) ──

def load_yolov5n_weights(pt_path):
    """Extract weight tensors from .pt file using stub model classes."""
    import types

    class Stub:
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["_state"] = state

    class StubFinder:
        def __getattr__(self, name):
            return Stub

    for mod_name in ["models.yolo", "models.common", "models.experimental", "utils.autoanchor"]:
        sys.modules[mod_name] = StubFinder()
    for mod_name in ["models", "utils"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = ckpt["model"]

    def find_tensors(obj, prefix="", depth=0):
        from collections import OrderedDict
        found = OrderedDict()
        if depth > 15: return found
        if isinstance(obj, torch.Tensor):
            found[prefix] = obj.float().numpy()  # f16 → f32
        elif isinstance(obj, (dict, OrderedDict)):
            for k, v in obj.items():
                found.update(find_tensors(v, f"{prefix}.{k}" if prefix else str(k), depth+1))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                found.update(find_tensors(v, f"{prefix}.{i}", depth+1))
        elif hasattr(obj, "__dict__") and not isinstance(obj, type):
            for k, v in obj.__dict__.items():
                found.update(find_tensors(v, f"{prefix}.{k}" if prefix else k, depth+1))
        return found

    return find_tensors(model)


# ── BN parameter precomputation ──

def precompute_bn(gamma, beta, mean, var, eps=1e-5):
    """Precompute bn_scale and bn_offset for fused inference.
    
    y = gamma * (x - mean) / sqrt(var + eps) + beta
      = (gamma / sqrt(var + eps)) * x + (beta - mean * gamma / sqrt(var + eps))
      = bn_scale * x + bn_offset

    ## Opmath precision discipline

    Promote all inputs to f32 at the function boundary. YOLOv5n.pt stores BN
    parameters as float16; without explicit promotion, numpy performs the
    arithmetic in float16 precision, producing ~5e-5 abs error vs PyTorch's
    BatchNorm2d (which promotes to f32 internally). Detected by
    bench/test_opmath_precision_invariance.py — the opmath_precision
    invariance property.

    ## rsqrt_variant: reciprocal_sqrt (matching the substrate kernel)

    Per the named rsqrt_variant substrate-design parameter (see
    lib/implementation_matches.pl): compute scale as

        inv_std = 1.0 / sqrt(var + eps)   # one DIVSS, one rounding
        scale   = gamma * inv_std          # one MULSS, one rounding

    rather than the algebraically-equivalent

        scale = gamma / sqrt(var + eps)   # one DIVSS, single rounding

    Both forms are IEEE-correct but differ by 1 ULP in scale. PyTorch CPU's
    BatchNorm2d uses the multiply-by-reciprocal form, and so does the
    substrate kernel bpd_batchnorm_cpu_affine_fused. Aligning precompute_bn
    to the same variant prevents 1-ULP drift between this numpy fallback
    and the substrate kernel.
    """
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    # rsqrt_variant: reciprocal_sqrt (matches substrate kernel + PyTorch CPU)
    inv_std = (np.float32(1.0) / np.sqrt(var + eps)).astype(np.float32)
    bn_scale = (gamma * inv_std).astype(np.float32)
    bn_offset = (beta - mean * bn_scale).astype(np.float32)
    return bn_scale.astype(np.float32), bn_offset.astype(np.float32)


# ── YOLOv5n architecture definition ──

def yolov5n_architecture():
    """Return the YOLOv5n layer list.
    
    Each entry: (type, params)
    Types: 'cbs' (Conv+BN+SiLU), 'maxpool', 'upsample', 'concat', 'c3', 'detect'
    """
    # From yolov5n.yaml — nano variant
    # Backbone
    layers = [
        ('cbs', {'c_in': 3, 'c_out': 16, 'k': 6, 's': 2, 'p': 2}),       # 0: Focus/Conv
        ('cbs', {'c_in': 16, 'c_out': 32, 'k': 3, 's': 2, 'p': 1}),       # 1
        ('c3',  {'c_in': 32, 'c_out': 32, 'n': 1}),                         # 2
        ('cbs', {'c_in': 32, 'c_out': 64, 'k': 3, 's': 2, 'p': 1}),       # 3
        ('c3',  {'c_in': 64, 'c_out': 64, 'n': 2}),                         # 4: backbone P3
        ('cbs', {'c_in': 64, 'c_out': 128, 'k': 3, 's': 2, 'p': 1}),      # 5
        ('c3',  {'c_in': 128, 'c_out': 128, 'n': 3}),                       # 6: backbone P4
        ('cbs', {'c_in': 128, 'c_out': 256, 'k': 3, 's': 2, 'p': 1}),     # 7
        ('c3',  {'c_in': 256, 'c_out': 256, 'n': 1}),                       # 8
        ('sppf', {'c_in': 256, 'c_out': 256, 'k': 5}),                      # 9: SPPF
        # Head
        ('cbs', {'c_in': 256, 'c_out': 128, 'k': 1, 's': 1, 'p': 0}),     # 10
        ('upsample', {}),                                                     # 11
        ('concat', {'from': [6]}),                                            # 12: cat with P4
        ('c3',  {'c_in': 256, 'c_out': 128, 'n': 1, 'shortcut': False}),   # 13
        ('cbs', {'c_in': 128, 'c_out': 64, 'k': 1, 's': 1, 'p': 0}),      # 14
        ('upsample', {}),                                                     # 15
        ('concat', {'from': [4]}),                                            # 16: cat with P3
        ('c3',  {'c_in': 128, 'c_out': 64, 'n': 1, 'shortcut': False}),    # 17: P3 head
        ('cbs', {'c_in': 64, 'c_out': 64, 'k': 3, 's': 2, 'p': 1}),       # 18
        ('concat', {'from': [14]}),                                           # 19: cat with P4
        ('c3',  {'c_in': 128, 'c_out': 128, 'n': 1, 'shortcut': False}),   # 20: P4 head
        ('cbs', {'c_in': 128, 'c_out': 128, 'k': 3, 's': 2, 'p': 1}),     # 21
        ('concat', {'from': [10]}),                                           # 22: cat with P5
        ('c3',  {'c_in': 256, 'c_out': 256, 'n': 1, 'shortcut': False}),   # 23: P5 head
        ('detect', {'from': [17, 20, 23], 'nc': 80, 'anchors': [
            [[10,13], [16,30], [33,23]],
            [[30,61], [62,45], [59,119]],
            [[116,90], [156,198], [373,326]]
        ]}),
    ]
    return layers


# ── Layer runners (CPU for now — GPU when fused kernels ready) ──

def run_cbs(x, weight, bn_gamma, bn_beta, bn_mean, bn_var, stride, pad, lib=None):
    """Conv + BatchNorm + SiLU, unfused on CPU.

    Substrate-design substantive discipline 2026-05-20 ~17:55 UTC: promote
    all tensor inputs to contiguous float32 at function boundary. Without
    this, f16 weights from .pt files would be:
      - reinterpreted as f32 bytes by bpd_conv2d_cpu (type confusion)
      - operated on in f16 by numpy (precision loss)
    Per the substrate-design opmath_precision discipline (see
    bench/test_opmath_precision_invariance.py).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    bn_gamma = np.asarray(bn_gamma, dtype=np.float32)
    bn_beta = np.asarray(bn_beta, dtype=np.float32)
    bn_mean = np.asarray(bn_mean, dtype=np.float32)
    bn_var = np.asarray(bn_var, dtype=np.float32)

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1

    # Conv
    # SUBSTANTIVE substrate-design substantive choice 2026-05-21 ~02:30 UTC:
    # use bpd_conv2d_full_cpu (im2col+GEMM via Goto-Sandy) instead of
    # bpd_conv2d_cpu (direct naive). The naive form accumulates reductions
    # in a different order than PyTorch at large C_in (e.g., 512), producing
    # 196k ULP divergence on SPPF cv2. The im2col+GEMM form is verified
    # BIT_IDENTICAL across all Stanford L1 conv variants (Phase A.4).
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    if lib and hasattr(lib, 'bpd_conv2d_full_cpu'):
        # bpd_conv2d_full_cpu signature:
        # (in, weight, bias, out, N, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, dH, dW, groups)
        lib.bpd_conv2d_full_cpu(
            x.ctypes.data, weight.ctypes.data, 0,  # bias=NULL
            out.ctypes.data,
            N, C_in, H, W, C_out, kH, kW,
            stride, stride, pad, pad, 1, 1, 1)
    elif lib and hasattr(lib, 'bpd_conv2d_cpu'):
        lib.bpd_conv2d_cpu(x.ctypes.data, weight.ctypes.data, out.ctypes.data,
                            N, C_in, H, W, C_out, kH, kW, stride, pad)
    else:
        # Python fallback
        for n in range(N):
            for co in range(C_out):
                for oh in range(H_out):
                    for ow in range(W_out):
                        s = 0.0
                        for ci in range(C_in):
                            for kh in range(kH):
                                for kw in range(kW):
                                    hi = oh*stride - pad + kh
                                    wi = ow*stride - pad + kw
                                    if 0 <= hi < H and 0 <= wi < W:
                                        s += float(x[n,ci,hi,wi]) * float(weight[co,ci,kh,kw])
                        out[n,co,oh,ow] = s

    # BatchNorm (inference) — use substrate kernel for bit-identity with
    # PyTorch CPU. The numpy expression
    #   out[:, c, :, :] = out[:, c, :, :] * bn_scale[c] + bn_offset[c]
    # produces different bits than the substrate kernel on real YOLOv5n
    # weights (probe_direct_vs_runcbs.py 2026-05-20 ~18:50 UTC: numpy form
    # 131075 ULP divergent from PyTorch, substrate kernel 0 ULP).
    if lib and hasattr(lib, 'bpd_batchnorm_cpu_affine_fused'):
        bn_out = np.zeros_like(out)
        scale_buf = np.zeros(C_out, dtype=np.float32)
        offset_buf = np.zeros(C_out, dtype=np.float32)
        out_c = np.ascontiguousarray(out, dtype=np.float32)
        lib.bpd_batchnorm_cpu_affine_fused(
            out_c.ctypes.data, bn_gamma.ctypes.data, bn_beta.ctypes.data,
            bn_mean.ctypes.data, bn_var.ctypes.data, bn_out.ctypes.data,
            scale_buf.ctypes.data, offset_buf.ctypes.data,
            N, C_out, H_out * W_out, 1e-5)
        out = bn_out
    else:
        # numpy fallback (only if substrate lib unavailable)
        bn_scale, bn_offset = precompute_bn(bn_gamma, bn_beta, bn_mean, bn_var)
        for c in range(C_out):
            out[:, c, :, :] = out[:, c, :, :] * bn_scale[c] + bn_offset[c]

    # SiLU — use substrate kernel (BIT_IDENTICAL with PyTorch per
    # bench/verify_yolo_per_stage.py); numpy's expression diverges by 1 ULP
    # on ~18% of elements due to non-FMA scalar emit on x86.
    if lib and hasattr(lib, 'bpd_silu_cpu'):
        out_c = np.ascontiguousarray(out, dtype=np.float32)
        silu_out = np.zeros_like(out_c)
        lib.bpd_silu_cpu(out_c.ctypes.data, silu_out.ctypes.data, out_c.size)
        out = silu_out
    else:
        # numpy fallback (only used if substrate lib unavailable)
        with np.errstate(over='ignore'):
            out = out * (1.0 / (1.0 + np.exp(-out)))

    return out


def run_bottleneck(x, weights, shortcut=True, lib=None):
    """One Bottleneck block: CBS(1x1) -> CBS(3x3) [+ residual if shortcut].

    YOLOv5 Bottleneck:
        y = cv1(x)         # Conv1x1: c_in -> c_     (CBS)
        y = cv2(y)         # Conv3x3: c_   -> c_out  (CBS)
        out = x + y if shortcut else y

    For shortcut=True, c_in must equal c_out (otherwise residual shape mismatch).
    YOLOv5n always uses Bottleneck inside C3, so c_in == c_out == c_ (the
    hidden dim of the C3 wrapping it).

    `weights` is a dict with keys 'cv1_*', 'cv2_*' for the two CBS sub-blocks
    (same key naming as get_layer_weights returns).
    """
    # cv1: 1x1 conv + BN + SiLU
    y = run_cbs(x, weights['cv1_conv'], weights['cv1_bn_gamma'],
                weights['cv1_bn_beta'], weights['cv1_bn_mean'],
                weights['cv1_bn_var'], stride=1, pad=0, lib=lib)
    # cv2: 3x3 conv + BN + SiLU
    y = run_cbs(y, weights['cv2_conv'], weights['cv2_bn_gamma'],
                weights['cv2_bn_beta'], weights['cv2_bn_mean'],
                weights['cv2_bn_var'], stride=1, pad=1, lib=lib)
    # Residual add (bit-identical with torch.add per
    # bench/verify_layer2_primitives.py)
    if shortcut:
        if lib and hasattr(lib, 'bpd_residual_add_cpu'):
            x_c = np.ascontiguousarray(x, dtype=np.float32)
            y_c = np.ascontiguousarray(y, dtype=np.float32)
            out = np.zeros_like(y_c)
            lib.bpd_residual_add_cpu(x_c.ctypes.data, y_c.ctypes.data,
                                      out.ctypes.data, y.size)
            return out
        else:
            return x.astype(np.float32) + y.astype(np.float32)
    return y


def run_cN(x, weights, n, shortcut=True, lib=None):
    """C3 module — the CSP bottleneck block. Generalized for any n>=1.

    YOLOv5 C3 (the building block at layers 2, 4, 6, 8, 13, 17, 20, 23):
        y1 = cv1(x)
        for _ in range(n):
            y1 = bottleneck(y1, shortcut=shortcut)
        y2 = cv2(x)
        y3 = concat([y1, y2], dim=1)
        out = cv3(y3)

    Each cv* is a CBS (Conv1x1 + BN + SiLU). The hidden channel count c_ is
    c_out // 2 (ultralytics convention). The concat doubles channels back
    to c_out so cv3 can map c_out -> c_out.

    `weights` is a dict with keys for cv1, cv2, cv3, and m.{i}.{cv1,cv2}
    for i in 0..n-1.

    The composition uses:
      - bpd_conv2d_cpu + bn_affine + bpd_silu_cpu (each BIT_IDENTICAL)
      - bpd_residual_add_cpu (BIT_IDENTICAL, only if shortcut=True)
      - bpd_concat_channel_cpu (BIT_IDENTICAL)
    so the entire C3 chain should compose to 0 ULP vs PyTorch CPU.
    """
    # cv1: c_in -> c_   (1x1, stride=1, pad=0)
    y1 = run_cbs(x, weights['cv1_conv'], weights['cv1_bn_gamma'],
                 weights['cv1_bn_beta'], weights['cv1_bn_mean'],
                 weights['cv1_bn_var'], stride=1, pad=0, lib=lib)

    # n bottlenecks in series on y1
    for i in range(n):
        bn_weights = weights[f'm{i}']
        y1 = run_bottleneck(y1, bn_weights, shortcut=shortcut, lib=lib)

    # cv2: c_in -> c_   (1x1, stride=1, pad=0)
    y2 = run_cbs(x, weights['cv2_conv'], weights['cv2_bn_gamma'],
                 weights['cv2_bn_beta'], weights['cv2_bn_mean'],
                 weights['cv2_bn_var'], stride=1, pad=0, lib=lib)

    # Concat along channel axis (bit-identical with torch.cat dim=1 per
    # bench/verify_layer2_primitives.py)
    y1_c = np.ascontiguousarray(y1, dtype=np.float32)
    y2_c = np.ascontiguousarray(y2, dtype=np.float32)
    N, C1, H, W = y1_c.shape
    _, C2, _, _ = y2_c.shape
    C_concat = C1 + C2
    y3 = np.zeros((N, C_concat, H, W), dtype=np.float32)
    if lib and hasattr(lib, 'bpd_concat_channel_cpu'):
        input_ptrs = (ctypes.c_void_p * 2)(y1_c.ctypes.data, y2_c.ctypes.data)
        c_each = (ctypes.c_int * 2)(C1, C2)
        lib.bpd_concat_channel_cpu(input_ptrs, c_each, 2, N, H, W,
                                    y3.ctypes.data)
    else:
        y3[:, :C1, :, :] = y1_c
        y3[:, C1:, :, :] = y2_c

    # cv3: c_concat -> c_out   (1x1, stride=1, pad=0)
    out = run_cbs(y3, weights['cv3_conv'], weights['cv3_bn_gamma'],
                  weights['cv3_bn_beta'], weights['cv3_bn_mean'],
                  weights['cv3_bn_var'], stride=1, pad=0, lib=lib)
    return out


def get_layer_weights(all_weights, layer_idx, submodule=""):
    """Extract weights for a specific layer from the flat tensor dict."""
    prefix = f"_modules.model._modules.{layer_idx}"
    if submodule:
        prefix += f"._modules.{submodule}"
    
    def get(suffix):
        key = f"{prefix}.{suffix}"
        if key in all_weights:
            return all_weights[key]
        # Try alternative key patterns
        for k, v in all_weights.items():
            if k.endswith(suffix) and f".{layer_idx}." in k:
                if not submodule or submodule in k:
                    return v
        return None

    return {
        'conv_weight': get('_modules.conv._parameters.weight'),
        'bn_weight': get('_modules.bn._parameters.weight'),
        'bn_bias': get('_modules.bn._parameters.bias'),
        'bn_mean': get('_modules.bn._buffers.running_mean'),
        'bn_var': get('_modules.bn._buffers.running_var'),
    }


def _get_cbs_weights(all_weights, prefix):
    """Extract the 5 CBS weights at a given prefix path.

    The prefix is the path up to (but not including) the cv1/cv2/m._modules.N
    suffix — e.g., '_modules.model._modules.2._modules.cv1'.

    Returns a tuple (conv, bn_gamma, bn_beta, bn_mean, bn_var) — or None
    if the conv weight is absent (signals the sub-block doesn't exist).
    """
    conv_key = f"{prefix}._modules.conv._parameters.weight"
    if conv_key not in all_weights:
        return None
    return (
        all_weights[conv_key],
        all_weights[f"{prefix}._modules.bn._parameters.weight"],
        all_weights[f"{prefix}._modules.bn._parameters.bias"],
        all_weights[f"{prefix}._modules.bn._buffers.running_mean"],
        all_weights[f"{prefix}._modules.bn._buffers.running_var"],
    )


def get_cN_weights(all_weights, layer_idx, n):
    """Extract weights for one C3 layer.

    Returns a dict suitable for run_cN(..., weights, n, ...):
      cv1_*, cv2_*, cv3_* — the three outer CBS sub-blocks
      m{i}                — dict of cv1_*, cv2_* for the i-th bottleneck
                              (i in 0..n-1)
    """
    base = f"_modules.model._modules.{layer_idx}._modules"
    weights = {}
    for cv in ('cv1', 'cv2', 'cv3'):
        tup = _get_cbs_weights(all_weights, f"{base}.{cv}")
        if tup is None:
            return None
        c, g, b, m, v = tup
        weights[f'{cv}_conv'] = c
        weights[f'{cv}_bn_gamma'] = g
        weights[f'{cv}_bn_beta'] = b
        weights[f'{cv}_bn_mean'] = m
        weights[f'{cv}_bn_var'] = v
    for i in range(n):
        bn = {}
        for cv in ('cv1', 'cv2'):
            tup = _get_cbs_weights(all_weights,
                f"{base}.m._modules.{i}._modules.{cv}")
            if tup is None:
                return None
            c, g, b, m, v = tup
            bn[f'{cv}_conv'] = c
            bn[f'{cv}_bn_gamma'] = g
            bn[f'{cv}_bn_beta'] = b
            bn[f'{cv}_bn_mean'] = m
            bn[f'{cv}_bn_var'] = v
        weights[f'm{i}'] = bn
    return weights


def run_upsample(x, lib=None):
    """Nearest-neighbor 2x upsample matching F.interpolate(scale_factor=2, mode='nearest')."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, C, H, W = x.shape
    out = np.zeros((N, C, 2*H, 2*W), dtype=np.float32)
    lib.bpd_upsample_nearest2d_cpu(x.ctypes.data, out.ctypes.data, N, C, H, W)
    return out


def run_concat(tensors, lib=None):
    """Channel-wise concat matching torch.cat([...], dim=1).

    The substrate kernel takes pointer-arrays. Build them carefully so the
    numpy arrays stay live during the call (Python GC trap).
    """
    tensors = [np.ascontiguousarray(t, dtype=np.float32) for t in tensors]
    N, _, H, W = tensors[0].shape
    c_each = [t.shape[1] for t in tensors]
    C_total = sum(c_each)
    out = np.zeros((N, C_total, H, W), dtype=np.float32)
    n_inputs = len(tensors)
    inputs_arr = (ctypes.c_void_p * n_inputs)(*[t.ctypes.data for t in tensors])
    c_each_arr = (ctypes.c_int * n_inputs)(*c_each)
    lib.bpd_concat_channel_cpu(inputs_arr, c_each_arr, n_inputs, N, H, W, out.ctypes.data)
    return out


def run_maxpool2d(x, k, s, p, lib=None):
    """MaxPool2D matching nn.MaxPool2d(k, s, p).
    
    Substrate signature: bpd_maxpool2d_cpu(in, out, N, C, H, W, kH, kW, stride, pad)
    (8 ints). Symmetric padding/stride/kernel only.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    N, C, H, W = x.shape
    H_out = (H + 2*p - k) // s + 1
    W_out = (W + 2*p - k) // s + 1
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    lib.bpd_maxpool2d_cpu(x.ctypes.data, out.ctypes.data,
                           N, C, H, W, k, k, s, p)
    return out


def _make_grid_yolov5(nx, ny, na, anchors_i, stride_i):
    """Make grid + anchor_grid for one detection level — matches yolov5 Detect._make_grid.

    Per ultralytics/yolov5 models/yolo.py Detect._make_grid (torch>=1.10):
        shape = (1, na, ny, nx, 2)
        y, x = arange(ny), arange(nx)
        yv, xv = meshgrid(y, x, indexing='ij')
        grid = stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (anchors_i * stride_i).view(1, na, 1, 1, 2).expand(shape)
    """
    y_arange = np.arange(ny, dtype=np.float32)
    x_arange = np.arange(nx, dtype=np.float32)
    yv, xv = np.meshgrid(y_arange, x_arange, indexing='ij')
    grid_stack = np.stack((xv, yv), axis=2).astype(np.float32)
    grid = np.broadcast_to(grid_stack, (1, na, ny, nx, 2)).copy() - np.float32(0.5)
    anchor_scaled = (anchors_i.astype(np.float32) * np.float32(stride_i)).reshape(1, na, 1, 1, 2)
    anchor_grid = np.broadcast_to(anchor_scaled, (1, na, ny, nx, 2)).copy()
    return grid, anchor_grid


def run_detect(feature_maps, weights, anchors, strides, nc, lib=None):
    """YOLOv5 Detect head — the final Essence of the model.

    Per ultralytics/yolov5 models/yolo.py Detect.forward (eval mode):
        for i in range(nl):
            x[i] = m[i](x[i])  # 1x1 conv with bias, ch -> na*no
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            grid[i], anchor_grid[i] = _make_grid(nx, ny, i)
            xy, wh, conf = x[i].sigmoid().split((2, 2, nc+1), 4)
            xy = (xy * 2 + grid[i]) * stride[i]
            wh = (wh * 2) ** 2 * anchor_grid[i]
            y = cat((xy, wh, conf), 4)
            z.append(y.view(bs, na*nx*ny, no))
        return (cat(z, 1), x)

    Args:
        feature_maps: list [P3, P4, P5] of numpy (bs, ch, ny, nx)
        weights: dict 'm{i}_weight' (na*no, ch, 1, 1), 'm{i}_bias' (na*no,)
        anchors: numpy (nl, na, 2) anchor box sizes
        strides: numpy (nl,) strides per detection level (typically 8, 16, 32)
        nc: number of classes (80 for COCO)

    Returns:
        (inference_output, raw_outputs)
        inference_output: (bs, total_anchors_all_levels, no)
        raw_outputs: list of (bs, na, ny_i, nx_i, no) per level
    """
    nl = len(feature_maps)
    na = anchors.shape[1]
    no = nc + 5

    raw_outputs = []
    z_list = []
    for i in range(nl):
        xi = np.ascontiguousarray(feature_maps[i], dtype=np.float32)
        weight = np.ascontiguousarray(weights[f'm{i}_weight'], dtype=np.float32)
        bias = np.ascontiguousarray(weights[f'm{i}_bias'], dtype=np.float32)
        bs, ch_in, ny, nx = xi.shape
        ch_out = na * no  # 255 for COCO (3*85)

        # 1x1 conv with bias via bpd_conv2d_full_cpu (im2col+GEMM, bit-identical)
        conv_out = np.zeros((bs, ch_out, ny, nx), dtype=np.float32)
        if lib and hasattr(lib, 'bpd_conv2d_full_cpu'):
            lib.bpd_conv2d_full_cpu(
                xi.ctypes.data, weight.ctypes.data, bias.ctypes.data,
                conv_out.ctypes.data,
                bs, ch_in, ny, nx, ch_out, 1, 1,
                1, 1, 0, 0, 1, 1, 1)
        else:
            raise RuntimeError("bpd_conv2d_full_cpu required for Detect head")

        # Reshape (bs, na*no, ny, nx) -> (bs, na, no, ny, nx) -> permute -> (bs, na, ny, nx, no)
        reshaped = conv_out.reshape(bs, na, no, ny, nx)
        permuted = np.ascontiguousarray(reshaped.transpose(0, 1, 3, 4, 2))
        raw_outputs.append(permuted.copy())

        # Make grid + anchor_grid
        grid, anchor_grid = _make_grid_yolov5(nx, ny, na, anchors[i], strides[i])

        # Sigmoid via substrate kernel (BIT_IDENTICAL with torch.sigmoid)
        sigmoided = np.zeros_like(permuted)
        if lib and hasattr(lib, 'bpd_sigmoid_cpu'):
            lib.bpd_sigmoid_cpu(permuted.ctypes.data, sigmoided.ctypes.data, permuted.size)
        else:
            sigmoided = 1.0 / (1.0 + np.exp(-permuted))

        # Split last axis: xy[..0:2], wh[..2:4], conf[..4:no]
        xy = sigmoided[..., 0:2]
        wh = sigmoided[..., 2:4]
        conf = sigmoided[..., 4:no]

        # xy = (xy * 2 + grid[i]) * stride[i]
        xy_out = (xy * np.float32(2.0) + grid) * np.float32(strides[i])
        # wh = (wh * 2) ** 2 * anchor_grid[i]
        wh_doubled = wh * np.float32(2.0)
        wh_out = wh_doubled * wh_doubled * anchor_grid

        # Concat back along last axis
        y = np.concatenate((xy_out, wh_out, conf), axis=4)
        z_list.append(y.reshape(bs, na * nx * ny, no))

    inference_output = np.concatenate(z_list, axis=1)
    return inference_output, raw_outputs


def run_sppf(x, weights, k, lib=None):
    """SPPF (Spatial Pyramid Pooling Fast) — YOLOv5 Layer 9.

    Algorithm:
        x = cv1(x)                              # CBS halving channels
        y1 = maxpool(x, k, s=1, p=k//2)
        y2 = maxpool(y1, k, s=1, p=k//2)
        y3 = maxpool(y2, k, s=1, p=k//2)
        return cv2(concat([x, y1, y2, y3], dim=1))  # CBS combining back

    weights dict expects: cv1_weight, cv1_bn_{gamma,beta,mean,var},
                          cv2_weight, cv2_bn_{gamma,beta,mean,var}
    """
    # cv1: 1x1 conv, stride 1, pad 0
    x = run_cbs(x, weights['cv1_weight'],
                weights['cv1_bn_gamma'], weights['cv1_bn_beta'],
                weights['cv1_bn_mean'], weights['cv1_bn_var'],
                stride=1, pad=0, lib=lib)
    # 3 maxpools with k, s=1, p=k//2
    y1 = run_maxpool2d(x, k=k, s=1, p=k//2, lib=lib)
    y2 = run_maxpool2d(y1, k=k, s=1, p=k//2, lib=lib)
    y3 = run_maxpool2d(y2, k=k, s=1, p=k//2, lib=lib)
    # Concat all 4 along channel dim
    combined = run_concat([x, y1, y2, y3], lib=lib)
    # cv2: 1x1 conv, stride 1, pad 0
    out = run_cbs(combined, weights['cv2_weight'],
                   weights['cv2_bn_gamma'], weights['cv2_bn_beta'],
                   weights['cv2_bn_mean'], weights['cv2_bn_var'],
                   stride=1, pad=0, lib=lib)
    return out


def main():
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/yolov5n.pt"
    
    print("Loading YOLOv5n weights...")
    weights = load_yolov5n_weights(pt_path)
    print(f"  {len(weights)} tensors loaded")

    # Load CPU kernel library
    cpu_so = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
    lib = None
    if os.path.exists(cpu_so):
        lib = ctypes.CDLL(cpu_so)
        lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
        lib.bpd_conv2d_cpu.restype = None
        # bpd_silu_cpu(input, output, n) — substrate-design BIT_IDENTICAL with PyTorch
        lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.bpd_silu_cpu.restype = None
        # bpd_batchnorm_cpu_affine_fused — BIT_IDENTICAL with PyTorch BN eval
        lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.bpd_batchnorm_cpu_affine_fused.restype = None
        # bpd_residual_add_cpu — Layer 2+ residual add
        lib.bpd_residual_add_cpu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.bpd_residual_add_cpu.restype = None
        # bpd_concat_channel_cpu — Layer 2+ / SPPF / FPN concat
        lib.bpd_concat_channel_cpu.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.bpd_concat_channel_cpu.restype = None
        print(f"  CPU kernel library: {cpu_so}")
    else:
        print(f"  CPU library not found, using Python fallback")

    # Architecture
    arch = yolov5n_architecture()
    print(f"  {len(arch)} layers")
    print()

    # Run first few CBS layers to verify
    print("Running first 2 CBS layers...")
    
    # Create test input (640×640 standard YOLO input)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, 640, 640)).astype(np.float32)
    print(f"  Input: {x.shape}")

    # Layer 0: Conv(3→16, 6×6, s=2, p=2) + BN + SiLU
    w0 = get_layer_weights(weights, 0)
    if w0['conv_weight'] is not None:
        print(f"  Layer 0 weights: conv={list(w0['conv_weight'].shape)} bn={list(w0['bn_weight'].shape)}")
        out0 = run_cbs(x, w0['conv_weight'], w0['bn_weight'], w0['bn_bias'],
                        w0['bn_mean'], w0['bn_var'], stride=2, pad=2, lib=lib)
        print(f"  Layer 0 output: {out0.shape} range=[{out0.min():.4f}, {out0.max():.4f}]")

        # Layer 1: Conv(16→32, 3×3, s=2, p=1) + BN + SiLU
        w1 = get_layer_weights(weights, 1)
        if w1['conv_weight'] is not None:
            print(f"  Layer 1 weights: conv={list(w1['conv_weight'].shape)} bn={list(w1['bn_weight'].shape)}")
            out1 = run_cbs(out0, w1['conv_weight'], w1['bn_weight'], w1['bn_bias'],
                            w1['bn_mean'], w1['bn_var'], stride=2, pad=1, lib=lib)
            print(f"  Layer 1 output: {out1.shape} range=[{out1.min():.4f}, {out1.max():.4f}]")

            # Compare against PyTorch
            if torch.cuda.is_available() or True:  # CPU works too
                print("\n  Comparing against PyTorch...")
                torch.backends.cudnn.enabled = False
                xt = torch.from_numpy(x)
                
                # Layer 0
                conv0 = torch.nn.Conv2d(3, 16, 6, stride=2, padding=2, bias=False)
                conv0.weight.data = torch.from_numpy(w0['conv_weight'])
                bn0 = torch.nn.BatchNorm2d(16)
                bn0.weight.data = torch.from_numpy(w0['bn_weight'])
                bn0.bias.data = torch.from_numpy(w0['bn_bias'])
                bn0.running_mean.data = torch.from_numpy(w0['bn_mean'])
                bn0.running_var.data = torch.from_numpy(w0['bn_var'])
                bn0.eval()
                
                with torch.no_grad():
                    pt0 = torch.nn.functional.silu(bn0(conv0(xt)))
                
                diff0 = np.abs(out0 - pt0.numpy()).max()
                print(f"  Layer 0 max abs diff from PyTorch: {diff0:.8f}")

                # Layer 1
                conv1 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
                conv1.weight.data = torch.from_numpy(w1['conv_weight'])
                bn1 = torch.nn.BatchNorm2d(32)
                bn1.weight.data = torch.from_numpy(w1['bn_weight'])
                bn1.bias.data = torch.from_numpy(w1['bn_bias'])
                bn1.running_mean.data = torch.from_numpy(w1['bn_mean'])
                bn1.running_var.data = torch.from_numpy(w1['bn_var'])
                bn1.eval()

                with torch.no_grad():
                    pt1 = torch.nn.functional.silu(bn1(conv1(pt0)))

                diff1 = np.abs(out1 - pt1.numpy()).max()
                print(f"  Layer 1 max abs diff from PyTorch: {diff1:.8f}")
    else:
        print("  ERROR: Could not find layer 0 weights")
        print("  Available weight keys (first 10):")
        for k in sorted(weights.keys())[:10]:
            print(f"    {k}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
