#!/usr/bin/env python3
"""test_f3_v2_tdd.py — TDD harness for the F3-v2 multi-K-block primitives.

Per Heath: 'decompose that path into lots of smaller subtasks, and use TDD.
We want to have a bit_identical output that fuses the SIMD for any shape of
(K,Q), and so we can break that down into whatever primitives compose to
perform that calculation, and we can test them all in independent test
cases, and tdd that into precision existence.'

Each primitive is tested in isolation against a reference implementation.
Bit-identity required at 0 ULP.

Primitives (built up from simplest):
  P1. bpd_gemm_v2_init(C, M, N)
  P2. bpd_gemm_v2_kblock_accumulate(A, B, C, M, N, K_total, k_start, k_end)
  P3. bpd_gemm_v2_kblock_accumulate_mtail(...)
  P4. bpd_gemm_v2_kblock_accumulate_ntail(...)
  P5. bpd_gemm_v2_full(A, B, C, M, N, K)  [composition: P1 + P2 + P3 + P4]
  P6. bpd_bn_silu_epilogue_simd(C, M, N, alpha, beta)
  P7. bpd_conv2d_bn_silu_fused_cpu_v2  [composition: P5 + P6]

Run: $PY bench/test_f3_v2_tdd.py [test_name]
Or:  $PY bench/test_f3_v2_tdd.py           (runs all)
"""
import ctypes
import os
import sys

import numpy as np

SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")


# ──────────────────────────── Test infrastructure ────────────────────────────

def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


class TestStatus:
    PASS = "\u2705 PASS"
    FAIL = "\u274c FAIL"
    SKIP = "\u23ed\ufe0f SKIP"
    MISSING = "\u26a0\ufe0f  MISSING"


def assert_bit_identical(name, ref, got, *, max_print=8):
    """Assert ref and got are bit-identical. Return (status, message)."""
    max_ulp, n_diff, n_total = ulp_distance(ref, got)
    if max_ulp == 0:
        return TestStatus.PASS, f"0 ULP / {n_total}"
    # Show first few divergent positions for diagnosis
    ref_flat = np.asarray(ref, dtype=np.float32).reshape(-1)
    got_flat = np.asarray(got, dtype=np.float32).reshape(-1)
    diff_mask = (ref_flat.view(np.uint32) != got_flat.view(np.uint32))
    diff_idx = np.where(diff_mask)[0][:max_print]
    samples = []
    for i in diff_idx:
        samples.append(f"  [{i}]: ref={ref_flat[i]:.7e} got={got_flat[i]:.7e}")
    msg = f"max_ulp={max_ulp} n_diff={n_diff}/{n_total}\n" + "\n".join(samples)
    return TestStatus.FAIL, msg


def setup_lib():
    lib = ctypes.CDLL(SO)
    # Already-existing primitives
    lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu.restype = None
    if hasattr(lib, 'bpd_mm_cpu_avx1_v2'):
        lib.bpd_mm_cpu_avx1_v2.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
        lib.bpd_mm_cpu_avx1_v2.restype = None
    if hasattr(lib, 'bpd_silu_cpu'):
        lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
        lib.bpd_silu_cpu.restype = None
    # New primitives (TDD targets)
    for fname, sig in [
        ('bpd_gemm_v2_init',                          ([ctypes.c_void_p] + [ctypes.c_int]*2, None)),
        ('bpd_gemm_v2_kblock_accumulate',             ([ctypes.c_void_p]*3 + [ctypes.c_int]*5, None)),
        ('bpd_gemm_v2_kblock_accumulate_mtail',       ([ctypes.c_void_p]*3 + [ctypes.c_int]*5, None)),
        ('bpd_gemm_v2_kblock_accumulate_ntail',       ([ctypes.c_void_p]*3 + [ctypes.c_int]*5, None)),
        ('bpd_gemm_v2_full',                          ([ctypes.c_void_p]*3 + [ctypes.c_int]*3, None)),
        ('bpd_bn_silu_epilogue_simd',                 ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p], None)),
    ]:
        if hasattr(lib, fname):
            argtypes, restype = sig
            setattr(getattr(lib, fname), 'argtypes', argtypes)
            setattr(getattr(lib, fname), 'restype', restype)
    return lib


# ──────────────────────────── Reference impls (Python) ────────────────────────────

def py_scalar_gemm_kblock_accumulate(A, B, C, k_start, k_end):
    """Reference: accumulate C[i,j] += sum_{k=k_start..k_end-1} A[i,k] * B[k,j].
    Linear left-fold per (i, j). Uses float32 throughout.
    Mutates C in place.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    M, K_full = A.shape
    _, N = B.shape
    for i in range(M):
        for j in range(N):
            partial = np.float32(0.0)
            for k in range(k_start, k_end):
                partial = np.float32(partial + np.float32(A[i, k] * B[k, j]))
            C[i, j] = np.float32(C[i, j] + partial)
    return C


# ──────────────────────────── Tests ────────────────────────────

def test_p1_gemm_v2_init(lib):
    """P1: bpd_gemm_v2_init zeros out a buffer."""
    if not hasattr(lib, 'bpd_gemm_v2_init'):
        return TestStatus.MISSING, "bpd_gemm_v2_init not in substrate yet"
    M, N = 17, 23
    # Pre-fill with non-zero garbage
    C = (np.random.default_rng(1).standard_normal((M, N)) * 100).astype(np.float32)
    lib.bpd_gemm_v2_init(C.ctypes.data, M, N)
    expected = np.zeros((M, N), dtype=np.float32)
    return assert_bit_identical("p1", expected, C)


def test_p2_gemm_v2_kblock_accumulate_simple(lib):
    """P2 (basic): single full K-block (k_start=0, k_end=K), M and N divisible by tile."""
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate not in substrate yet"
    rng = np.random.default_rng(2)
    M, N, K = 8, 32, 100  # M divisible by 4 (MR=4), N divisible by 16 (NR=16)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    
    # Reference: scalar K-block accumulate using Python
    C_ref = np.zeros((M, N), dtype=np.float32)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, 0, K)
    
    # Substrate
    C = np.zeros((M, N), dtype=np.float32)
    lib.bpd_gemm_v2_kblock_accumulate(
        A.ctypes.data, B.ctypes.data, C.ctypes.data,
        M, N, K, 0, K)
    
    return assert_bit_identical("p2_simple", C_ref, C)


def test_p2_gemm_v2_kblock_accumulate_partial(lib):
    """P2 (partial): partial K-block [k_start=20, k_end=80) within larger K=100."""
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate not in substrate yet"
    rng = np.random.default_rng(3)
    M, N, K = 8, 32, 100
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    k_start, k_end = 20, 80
    
    C_ref = np.zeros((M, N), dtype=np.float32)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, k_start, k_end)
    
    C = np.zeros((M, N), dtype=np.float32)
    lib.bpd_gemm_v2_kblock_accumulate(
        A.ctypes.data, B.ctypes.data, C.ctypes.data,
        M, N, K, k_start, k_end)
    
    return assert_bit_identical("p2_partial", C_ref, C)


def test_p2_gemm_v2_kblock_accumulate_into_nonzero(lib):
    """P2 (accumulate into nonzero): two successive partial K-blocks compose correctly."""
    if not hasattr(lib, 'bpd_gemm_v2_kblock_accumulate'):
        return TestStatus.MISSING, "bpd_gemm_v2_kblock_accumulate not in substrate yet"
    rng = np.random.default_rng(4)
    M, N, K = 8, 32, 200
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    
    # Reference: two K-blocks [0, 100) then [100, 200)
    C_ref = np.zeros((M, N), dtype=np.float32)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, 0, 100)
    py_scalar_gemm_kblock_accumulate(A, B, C_ref, 100, 200)
    
    # Substrate: same two K-blocks
    C = np.zeros((M, N), dtype=np.float32)
    lib.bpd_gemm_v2_kblock_accumulate(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K, 0, 100)
    lib.bpd_gemm_v2_kblock_accumulate(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K, 100, 200)
    
    return assert_bit_identical("p2_two_blocks", C_ref, C)


def test_p6_bn_silu_epilogue_simd(lib):
    """P6: SIMD epilogue applied to a known-value tensor."""
    if not hasattr(lib, 'bpd_bn_silu_epilogue_simd'):
        return TestStatus.MISSING, "bpd_bn_silu_epilogue_simd not in substrate yet"
    rng = np.random.default_rng(6)
    M, N = 4, 32  # M=4 for register block, N=32 = 2 tiles of 16
    C_initial = (rng.standard_normal((M, N)) * 0.5).astype(np.float32)
    alpha = (rng.standard_normal(M) * 0.3 + 1.0).astype(np.float32)
    beta = (rng.standard_normal(M) * 0.1).astype(np.float32)
    alpha = np.ascontiguousarray(alpha); beta = np.ascontiguousarray(beta)
    
    # Reference: scalar epilogue (matches F3 v1's per-element loop)
    C_ref = C_initial.copy()
    for i in range(M):
        a = alpha[i]; b = beta[i]
        for j in range(N):
            x = np.float32(a * C_ref[i, j] + b)
            C_ref[i, j] = np.float32(x / np.float32(1.0 + np.float32(np.exp(-x))))
    
    # Substrate
    C = C_initial.copy()
    lib.bpd_bn_silu_epilogue_simd(
        C.ctypes.data, M, N, alpha.ctypes.data, beta.ctypes.data)
    
    return assert_bit_identical("p6", C_ref, C)


# ──────────────────────────── Test runner ────────────────────────────

TESTS = [
    ("P1 init",                          test_p1_gemm_v2_init),
    ("P2 simple (full K-block)",         test_p2_gemm_v2_kblock_accumulate_simple),
    ("P2 partial (k-range)",             test_p2_gemm_v2_kblock_accumulate_partial),
    ("P2 two-blocks (compose)",          test_p2_gemm_v2_kblock_accumulate_into_nonzero),
    ("P6 SIMD epilogue",                 test_p6_bn_silu_epilogue_simd),
]


def main():
    lib = setup_lib()
    print(f"Substrate: {SO}")
    print()
    print(f"{'Test':<40} {'Result':<60}")
    print("-" * 102)
    n_pass = 0
    n_fail = 0
    n_missing = 0
    for name, test_fn in TESTS:
        try:
            status, msg = test_fn(lib)
        except Exception as e:
            status, msg = TestStatus.FAIL, f"exception: {e}"
        first_line = msg.splitlines()[0] if msg else ""
        print(f"{name:<40} {status} {first_line}")
        rest = msg.splitlines()[1:]
        for r in rest:
            print(f"{'':<40} {'':<10}{r}")
        if status == TestStatus.PASS:
            n_pass += 1
        elif status == TestStatus.MISSING:
            n_missing += 1
        else:
            n_fail += 1
    print()
    print(f"PASS: {n_pass}, FAIL: {n_fail}, MISSING: {n_missing}")
    sys.exit(0 if (n_fail == 0 and n_missing == 0) else 1)


if __name__ == "__main__":
    main()
