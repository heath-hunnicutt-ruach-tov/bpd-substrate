#!/usr/bin/env python3
"""Verify bit-identical (0 ULP) output between BPD and PyTorch/cuBLAS.

Three-way comparison on the same GPU, same data, same CUDA context:
  A: PyTorch cuBLAS sgemm (torch.matmul)
  B: BPD matmul (our kernel via ctypes)
  C: CPU f64 reference (numpy)

Exits 0 only if A vs B = 0 ULP at ALL sizes.
"""
import ctypes, os, sys, numpy as np

try:
    import torch
    assert torch.cuda.is_available()
except (ImportError, AssertionError):
    sys.exit("error: torch with CUDA required. pip install torch numpy")

BUILD_DIR = os.environ.get("BPD_BUILD_DIR", "build")
SO_PATH = os.environ.get("BPD_MM_SO", os.path.join(BUILD_DIR, "bpd_mm.so"))

SIZES = [(512,512,512),(1024,1024,1024),(2048,2048,2048)]

def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum())

def load_bpd():
    if not os.path.exists(SO_PATH):
        sys.exit(f"error: {SO_PATH} not found. Run `make build` first.")
    lib = ctypes.CDLL(SO_PATH)
    lib.gpu_alloc.restype = ctypes.c_void_p
    lib.gpu_alloc.argtypes = [ctypes.c_int]
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.gpu_sync.argtypes = []
    lib.bpd_sgemm.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    return lib

def main():
    dev = torch.cuda.get_device_name(0)
    sm = torch.cuda.get_device_capability()
    print(f"GPU:  {dev} (sm_{sm[0]}{sm[1]})")
    print(f"BPD:  {SO_PATH}")
    print(f"Ref:  torch {torch.__version__} (cuBLAS sgemm)")
    print()

    lib = load_bpd()
    all_pass = True

    print(f"{'SIZE':>12}  {'PT vs BPD':>16}  {'PT vs CPU':>16}  {'BPD vs CPU':>16}  {'RESULT':>14}")
    print("=" * 82)

    for M, N, K in SIZES:
        mn = M * N
        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((M, K)).astype(np.float32)
        B_np = rng.standard_normal((K, N)).astype(np.float32)

        # A: PyTorch cuBLAS
        At = torch.from_numpy(A_np).cuda()
        Bt = torch.from_numpy(B_np).cuda()
        out_a = (At @ Bt).cpu().numpy()

        # B: BPD
        dA = lib.gpu_alloc(M*K*4); dB = lib.gpu_alloc(K*N*4); dC = lib.gpu_alloc(mn*4)
        lib.gpu_h2d(dA, A_np.ctypes.data, M*K*4)
        lib.gpu_h2d(dB, B_np.ctypes.data, K*N*4)
        lib.bpd_sgemm(dA, dB, dC, M, N, K); lib.gpu_sync()
        out_b = np.zeros((M, N), dtype=np.float32)
        lib.gpu_d2h(out_b.ctypes.data, dC, mn*4)
        lib.gpu_free(dA); lib.gpu_free(dB); lib.gpu_free(dC)

        # C: CPU f64
        out_c = (A_np.astype(np.float64) @ B_np.astype(np.float64)).astype(np.float32)

        ab_max, ab_cnt = ulp(out_a, out_b)
        ac_max, _ = ulp(out_a, out_c)
        bc_max, _ = ulp(out_b, out_c)

        passed = ab_max == 0
        if not passed: all_pass = False
        tag = "*** 0 ULP ***" if passed else f"FAIL ({ab_cnt} diffs)"

        print(f"{M}x{N}x{K:>4}  max={ab_max:>8} ULP  max={ac_max:>8} ULP  max={bc_max:>8} ULP  {tag}")

    print()
    if all_pass:
        print("PASS: BPD output is BIT-IDENTICAL with PyTorch/cuBLAS at all sizes.")
        print("      Every float, every bit, every element. 0 ULP.")
    else:
        print("FAIL: BPD output differs from PyTorch/cuBLAS at one or more sizes.")

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
