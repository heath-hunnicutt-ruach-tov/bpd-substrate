#!/usr/bin/env python3
"""sweep_gemm_params.py — Phase 3.CAT.FUSE.g substrate-design parameter sweep.

Measures bpd_mm_cpu_avx1_v2 timing across:
  - All YOLOv5n GEMM shapes (the workload distribution)
  - SUBSTRATE_AVX1_PREFETCH in {0, 1}

The output is a per-shape table identifying which configuration is fastest.
This data is the empirical foundation for Phase 3.CAT.h shape specialization
('compile each CBS site with its own optimal parameter set').

For Phase 3.CAT.f/g, we have one tunable axis (prefetch on/off). Future
work (CAT.g packing, CAT.h shape specialization) will add more axes:
  - register_blocking variants (MR x NR)
  - unroll_factor_K variants

Bit-identity is preserved across the entire sweep: each kernel variant is
already validated 0-ULP against scalar bpd_mm_cpu. The sweep is purely
about wall-clock performance.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")


def run_with_env(env_value, runs=5):
    """Spawn a subprocess with SUBSTRATE_AVX1_PREFETCH set to env_value,
    measure all YOLO GEMM shapes, return dict of label -> min_ms.
    
    Need subprocess because the prefetch_choice is cached in a static int
    on first call — we can't change it within a single process.
    """
    script = """
import ctypes, time, os, sys
import numpy as np

SO = os.environ['BPD_CPU_SO']
lib = ctypes.CDLL(SO)
lib.bpd_mm_cpu_avx1_v2.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.bpd_mm_cpu_avx1_v2.restype = None

shapes = [
    ('L0_focus',   16,  108, 102400),
    ('L1_cbs',     32,  144,  25600),
    ('L3_cbs',     64,  288,   6400),
    ('L5_cbs',    128,  576,   1600),
    ('L7_cbs',    256, 1152,    400),
    ('L9_sppf',   256,  512,    400),
    ('L13_c3',    128,  128,   1600),
    ('L17_c3',     64,  128,   6400),
]

rng = np.random.default_rng(2026)
runs = """ + str(runs) + """
results = {}
for label, M, K, N in shapes:
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    A = np.ascontiguousarray(A); B = np.ascontiguousarray(B)
    C = np.zeros((M, N), dtype=np.float32)
    # Warmup
    lib.bpd_mm_cpu_avx1_v2(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        lib.bpd_mm_cpu_avx1_v2(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
        times.append(time.perf_counter() - t0)
    results[label] = min(times) * 1000  # min ms (best of N to filter noise)

for label, ms in results.items():
    print(f"RESULT {label} {ms:.3f}")
"""
    env = os.environ.copy()
    env['SUBSTRATE_AVX1_PREFETCH'] = str(env_value)
    env['BPD_CPU_SO'] = SO
    res = subprocess.run([sys.executable, '-c', script], env=env,
                          capture_output=True, text=True, timeout=600)
    results = {}
    for line in res.stdout.splitlines():
        if line.startswith('RESULT'):
            _, label, ms = line.split()
            results[label] = float(ms)
    return results


def main():
    print("Phase 3.CAT.FUSE.g — Substrate-Design Parameter Sweep")
    print("=" * 80)
    print(f"Substrate: {SO}")
    print()
    print("Sweeping: SUBSTRATE_AVX1_PREFETCH ∈ {0, 1}")
    print("Workload: 8 YOLOv5n CBS GEMM shapes")
    print("Reporting: min ms of 5 runs (best-of filter)")
    print()

    print("Running prefetch=0 ...", flush=True)
    no_pf = run_with_env(0)
    print("Running prefetch=1 ...", flush=True)
    yes_pf = run_with_env(1)

    print()
    print(f"{'Shape':<14} {'K':<6} {'no_pf ms':>10} {'with_pf ms':>11} {'speedup':>8} {'winner':<10}")
    print("-" * 70)
    
    # K values for each shape (lookup)
    shape_k = {
        'L0_focus': 108, 'L1_cbs': 144, 'L3_cbs': 288, 'L5_cbs': 576,
        'L7_cbs': 1152, 'L9_sppf': 512, 'L13_c3': 128, 'L17_c3': 128,
    }
    
    total_no_pf = 0.0
    total_with_pf = 0.0
    pf_winners = []
    no_pf_winners = []
    
    for label in ['L0_focus', 'L1_cbs', 'L3_cbs', 'L5_cbs', 'L7_cbs', 'L9_sppf', 'L13_c3', 'L17_c3']:
        n = no_pf.get(label, 0)
        y = yes_pf.get(label, 0)
        if n == 0 or y == 0:
            continue
        speedup = n / y
        winner = "PF" if y < n else "no-PF"
        if y < n:
            pf_winners.append(label)
        else:
            no_pf_winners.append(label)
        total_no_pf += n
        total_with_pf += y
        print(f"{label:<14} {shape_k[label]:<6} {n:>8.3f}    {y:>9.3f}     {speedup:>5.2f}x   {winner:<10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<14} {'':<6} {total_no_pf:>8.3f}    {total_with_pf:>9.3f}     {total_no_pf/total_with_pf:>5.2f}x")
    print()
    print(f"Prefetch HELPS: {pf_winners}")
    print(f"Prefetch HURTS: {no_pf_winners}")
    print()
    print("Substrate-design observation: shape specialization (Phase 3.CAT.h)")
    print("can pick prefetch on/off per CBS site, which yields better wall-clock")
    print("than the universal default in either direction.")


if __name__ == "__main__":
    main()
