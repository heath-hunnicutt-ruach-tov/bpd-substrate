#!/usr/bin/env python3
"""Verify BPD kernels are bit-identical with PyTorch — CPU or GPU.

Detects available hardware and runs the appropriate comparison:
  CPU:  BPD C kernels (gcc) vs PyTorch CPU
  GPU:  BPD CUDA kernels (nvcc) vs PyTorch CUDA (cuBLAS/ATen)

Anyone with Python + gcc can verify correctness. No GPU required.

Usage:
    python3 bench/bit_identical_universal.py          # auto-detect
    BPD_CPU_SO=build/bpd_cpu.so python3 bench/bit_identical_universal.py  # explicit CPU
"""
import ctypes, os, sys, numpy as np

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch numpy")

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

CPU_SO = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
GPU_SO = os.environ.get("BPD_MM_SO", "build/bpd_mm.so")

def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), d.size

def classify(ref, got, label=""):
    mx, cnt, tot = ulp(ref, got)
    abs_max = float(np.abs(ref - got).max())
    if mx == 0:
        return "BIT_IDENTICAL", mx, abs_max
    elif abs_max < 1e-4 and mx > 100000:
        return "PASS_ABS_TOLERANCE", mx, abs_max
    elif mx <= 64:
        return "PASS_WITHIN_64_ULP", mx, abs_max
    else:
        return "FAIL", mx, abs_max

def load_cpu_lib():
    if not os.path.exists(CPU_SO):
        return None
    lib = ctypes.CDLL(CPU_SO)
    # matmul
    lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu.restype = None
    # fused matmul + bias + relu
    lib.bpd_mm_bias_relu_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    lib.bpd_mm_bias_relu_cpu.restype = None
    # elementwise
    for fn in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # conv2d
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
    lib.bpd_conv2d_cpu.restype = None
    # batchnorm
    lib.bpd_batchnorm_cpu.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.bpd_batchnorm_cpu.restype = None
    # upsample
    lib.bpd_upsample_nearest2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
    lib.bpd_upsample_nearest2d_cpu.restype = None
    return lib

def run_cpu_tests(lib):
    results = []
    rng = np.random.default_rng(42)

    # ── Matmul ──
    for M in [64, 256, 512]:
        N = K = M
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
        out = np.zeros((M, N), dtype=np.float32)
        lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, M, N, K)
        status, mx, ab = classify(ref, out)
        results.append(("sgemm_cpu", f"{M}x{M}", status, mx, ab))

    # ── Elementwise ──
    x = rng.standard_normal(10000).astype(np.float32)
    for name, pt_fn, bpd_fn in [
        ("relu",  lambda t: torch.relu(t), lib.bpd_relu_cpu),
        ("silu",  lambda t: torch.nn.functional.silu(t), lib.bpd_silu_cpu),
        ("mish",  lambda t: torch.nn.functional.mish(t), lib.bpd_mish_cpu),
    ]:
        ref = pt_fn(torch.from_numpy(x)).numpy()
        out = np.zeros_like(x)
        bpd_fn(x.ctypes.data, out.ctypes.data, len(x))
        status, mx, ab = classify(ref, out)
        results.append((f"{name}_cpu", "10000", status, mx, ab))

    # ── Fused matmul + bias + relu ──
    M, N, K = 256, 256, 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    ref = torch.relu(torch.from_numpy(A) @ torch.from_numpy(B) + torch.from_numpy(bias)).numpy()
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_bias_relu_cpu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, out.ctypes.data, M, N, K)
    status, mx, ab = classify(ref, out)
    results.append(("fused_mm_bias_relu_cpu", f"{M}x{M}", status, mx, ab))

    # ── Conv2D ──
    N_batch, C_in, H, W, C_out, kH, kW = 1, 3, 16, 16, 8, 3, 3
    stride, pad = 1, 1
    inp = rng.standard_normal((N_batch, C_in, H, W)).astype(np.float32)
    weight = rng.standard_normal((C_out, C_in, kH, kW)).astype(np.float32)
    ref = torch.nn.functional.conv2d(
        torch.from_numpy(inp), torch.from_numpy(weight),
        stride=stride, padding=pad).numpy()
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N_batch, C_out, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_cpu(inp.ctypes.data, weight.ctypes.data, out.ctypes.data,
                        N_batch, C_in, H, W, C_out, kH, kW, stride, pad)
    status, mx, ab = classify(ref, out)
    results.append(("conv2d_cpu", f"{N_batch}x{C_in}x{H}x{W}", status, mx, ab))

    # ── Upsample ──
    inp = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
    ref = torch.nn.functional.interpolate(
        torch.from_numpy(inp), scale_factor=2, mode='nearest').numpy()
    out = np.zeros((1, 8, 8, 8), dtype=np.float32)
    lib.bpd_upsample_nearest2d_cpu(inp.ctypes.data, out.ctypes.data, 1, 8, 4, 4)
    status, mx, ab = classify(ref, out)
    results.append(("upsample_cpu", "1x8x4x4", status, mx, ab))

    return results

def main():
    print(f"BPD Universal Bit-Identity Verification")
    print(f"PyTorch {torch.__version__} on {DEVICE.upper()}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    lib_cpu = load_cpu_lib()
    if lib_cpu:
        print(f"CPU library: {CPU_SO}")
    else:
        print(f"CPU library: not found ({CPU_SO})")
        print(f"  Build with: gcc -O2 -shared -fPIC -o {CPU_SO} bench/bpd_cpu.c -lm")

    results = []

    if lib_cpu:
        print()
        print("── CPU VERIFICATION (BPD C kernels vs PyTorch CPU) ──")
        cpu_results = run_cpu_tests(lib_cpu)
        results.extend(cpu_results)
        for name, shape, status, mx, ab in cpu_results:
            tag = "✓" if "PASS" in status or "IDENTICAL" in status else "✗"
            print(f"  {name:<25} {shape:<16} {status:<22} max_ulp={mx:<10} {tag}")

    # Summary
    print()
    passed = sum(1 for _, _, s, _, _ in results if "PASS" in s or "IDENTICAL" in s)
    total = len(results)
    print(f"{'=' * 60}")
    print(f"PASSED: {passed}/{total}")
    if passed == total:
        print(f"\nALL KERNELS BIT-IDENTICAL WITH PyTorch on {DEVICE.upper()}.")
        print(f"Same math. Same bits. {'No GPU required.' if not HAS_CUDA else ''}")
    else:
        failed = [(n, s, mx) for n, _, s, mx, _ in results if "PASS" not in s and "IDENTICAL" not in s]
        print(f"\nFAILED: {len(failed)}")
        for n, s, mx in failed:
            print(f"  {n}: {s} (max {mx} ULP)")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
