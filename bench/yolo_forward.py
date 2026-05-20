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

    Substrate-design substantive fix 2026-05-20 ~17:55 UTC (per Heath's TDD
    direction): promote inputs to f32 at the function boundary. YOLOv5n.pt
    stores BN parameters as float16; without explicit promotion, numpy
    performs the arithmetic in float16 precision, producing ~5e-5 abs error
    vs PyTorch's BatchNorm2d (which promotes to f32 internally).

    Detected by bench/test_opmath_precision_invariance.py — the
    substrate-design opmath_precision invariance property. Same substrate-
    design family as rsqrt_variant, k_tile_strategy, reduction_strategy.
    """
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    bn_scale = gamma / np.sqrt(var + eps)
    bn_offset = beta - mean * bn_scale
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
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    if lib and hasattr(lib, 'bpd_conv2d_cpu'):
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

    # BatchNorm (inference)
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
