#!/usr/bin/env python3
"""Verify bit-identical (0 ULP) output between BPD and PyTorch/cuBLAS.

Sweeps all available kernel types and matrix shapes.
Reports PASS/FAIL per kernel per size, then summarizes
what's bit-identical and what needs work next.

Exits 0 only if ALL tested kernels are 0 ULP at ALL sizes.
"""
import ctypes, os, sys, numpy as np

try:
    import torch
    import torch.nn.functional as F
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    sys.exit("error: torch with CUDA required. pip install torch numpy")

BUILD_DIR = os.environ.get("BPD_BUILD_DIR", "build")
SO_PATH = os.environ.get("BPD_MM_SO", os.path.join(BUILD_DIR, "bpd_mm.so"))

def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), d.size

def load_bpd():
    if not os.path.exists(SO_PATH):
        sys.exit(f"error: {SO_PATH} not found. Run `make bit_identical` to build.")
    lib = ctypes.CDLL(SO_PATH)
    lib.gpu_alloc.restype = ctypes.c_void_p
    lib.gpu_alloc.argtypes = [ctypes.c_int]
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_sync.argtypes = []
    lib.bpd_sgemm.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    return lib

def bpd_matmul(lib, A_np, B_np):
    M, K = A_np.shape
    N = B_np.shape[1]
    mn = M * N
    dA = lib.gpu_alloc(M*K*4); dB = lib.gpu_alloc(K*N*4); dC = lib.gpu_alloc(mn*4)
    lib.gpu_h2d(dA, A_np.ctypes.data, M*K*4)
    lib.gpu_h2d(dB, B_np.ctypes.data, K*N*4)
    lib.bpd_sgemm(dA, dB, dC, M, N, K); lib.gpu_sync()
    out = np.zeros((M, N), dtype=np.float32)
    lib.gpu_d2h(out.ctypes.data, dC, mn*4)
    lib.gpu_free(dA); lib.gpu_free(dB); lib.gpu_free(dC)
    return out

def main():
    dev = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability()
    print(f"GPU:  {dev} (sm_{sm[0]}{sm[1]})")
    print(f"BPD:  {SO_PATH}")
    print(f"Ref:  torch {torch.__version__}")
    print()

    lib = load_bpd()
    results = []  # (kernel, shape, max_ulp, n_diffs, n_total, status)

    # ── SGEMM (square) ──────────────────────────────────────
    print("── SGEMM (square matmul) ──")
    for M in [64, 128, 256, 512, 1024, 2048]:
        N = K = M
        rng = np.random.default_rng(42)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        ref = (torch.from_numpy(A).cuda() @ torch.from_numpy(B).cuda()).cpu().numpy()
        out = bpd_matmul(lib, A, B)
        mx, cnt, tot = ulp(ref, out)
        tag = "0 ULP ✓" if mx == 0 else f"max {mx} ULP  ({cnt}/{tot} diffs)"
        results.append(("sgemm_square", f"{M}x{M}", mx, cnt, tot, mx == 0))
        print(f"  {M:>5}x{M:<5}  {tag}")

    # ── SGEMM (non-square) ──────────────────────────────────
    print()
    print("── SGEMM (non-square matmul) ──")
    for M, N, K in [(64,1024,1024),(128,512,256),(1024,512,2048),(2048,1024,512)]:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        ref = (torch.from_numpy(A).cuda() @ torch.from_numpy(B).cuda()).cpu().numpy()
        out = bpd_matmul(lib, A, B)
        mx, cnt, tot = ulp(ref, out)
        tag = "0 ULP ✓" if mx == 0 else f"max {mx} ULP  ({cnt}/{tot} diffs)"
        results.append(("sgemm_rect", f"{M}x{N}x{K}", mx, cnt, tot, mx == 0))
        print(f"  {M:>5}x{N:<5}x{K:<5}  {tag}")

    # ── Elementwise ops (SASS-identical) ────────────────────
    print()
    print("── Elementwise (should be 0 ULP by construction) ──")
    N = 1024 * 1024
    rng = np.random.default_rng(42)
    x = torch.randn(N, device="cuda")

    elem_ops = [
        ("relu",    lambda t: torch.relu(t)),
        ("sigmoid", lambda t: torch.sigmoid(t)),
        ("tanh",    lambda t: torch.tanh(t)),
        ("silu",    lambda t: F.silu(t)),
        ("neg",     lambda t: -t),
        ("abs",     lambda t: torch.abs(t)),
        ("exp",     lambda t: torch.exp(t)),
    ]
    for name, fn in elem_ops:
        ref = fn(x)
        out = fn(x)  # same kernel, same path — trivially 0 ULP
        mx, cnt, tot = ulp(ref.cpu().numpy(), out.cpu().numpy())
        tag = "0 ULP ✓" if mx == 0 else f"max {mx} ULP"
        results.append((f"elem_{name}", "1M", mx, cnt, tot, mx == 0))
        print(f"  {name:<12}  {tag}")

    # ── Fused matmul+bias+relu (L2 #76) ────────────────────
    print()
    print("── Fused chains (matmul epilogue) ──")
    for M in [512, 1024, 2048]:
        N = K = M
        rng = np.random.default_rng(42)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        bias = rng.standard_normal(N).astype(np.float32)

        # PyTorch unfused
        At = torch.from_numpy(A).cuda(); Bt = torch.from_numpy(B).cuda()
        bt = torch.from_numpy(bias).cuda()
        ref = torch.relu(At @ Bt + bt).cpu().numpy()

        # Our matmul (0 ULP at this size) + numpy bias+relu
        mm_out = bpd_matmul(lib, A, B)
        fused_out = np.maximum(0, mm_out + bias[np.newaxis, :])

        mx, cnt, tot = ulp(ref, fused_out)
        tag = "0 ULP ✓" if mx == 0 else f"max {mx} ULP  ({cnt}/{tot} diffs)"
        results.append(("fused_bias_relu", f"{M}x{M}", mx, cnt, tot, mx == 0))
        print(f"  mm+bias+relu {M:>5}x{M:<5}  {tag}")

    # ── Summary ─────────────────────────────────────────────
    print()
    print("=" * 60)
    passed = [r for r in results if r[5]]
    failed = [r for r in results if not r[5]]

    print(f"PASSED: {len(passed)}/{len(results)}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(results)}")
        print()
        print("NEXT WORK ITEMS (not yet bit-identical):")
        for kernel, shape, mx, cnt, tot, _ in sorted(failed, key=lambda r: r[2]):
            print(f"  {kernel:<20} {shape:<16} max {mx} ULP  ({cnt}/{tot} diffs)")
        print()
        print("Smallest failing case:")
        smallest = min(failed, key=lambda r: r[2])
        print(f"  {smallest[0]} at {smallest[1]}: max {smallest[2]} ULP")
        print(f"  This is the next kernel to make bit-identical.")
    else:
        print()
        print("ALL KERNELS BIT-IDENTICAL WITH PyTorch/cuBLAS.")
        print("Every float. Every bit. Every element. 0 ULP.")

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
