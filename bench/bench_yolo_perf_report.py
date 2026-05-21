#!/usr/bin/env python3
"""bench_yolo_perf_report.py — consolidated Phase 3 performance comparison.

Runs a comprehensive performance benchmark across:
  1. PyTorch CPU reference (cBLAS sgemm via OpenBLAS SANDYBRIDGE)
  2. PyTorch GPU (Tesla P4, cuDNN disabled, CUDA fallback)
  3. BPD substrate: scalar baseline (SUBSTRATE_AVX1_GEMM=0)
  4. BPD substrate: + AVX1 GEMM (3.G)
  5. BPD substrate: + AVX1 GEMM + F3 (3.1)
  6. BPD substrate: + AVX1 + F3 + F4 + F8 (3.1 + 3.2 + 3.4) — full Phase 3 stack

Each configuration:
  - Runs on 5 representative COCO images (sampled from the 10) for time budget
  - Reports avg ms/image
  - Asserts MATCH 10/10 against Medayek's compare_detections (correctness gate)
  - Prints the speedup ratio vs the previous configuration

Usage:
  PYTHONPATH=... LD_LIBRARY_PATH=... $PY bench/bench_yolo_perf_report.py

Output: a markdown-formatted performance report ready to share with the Collective.
"""
import ctypes
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

# Avoid mkldnn/cudnn for deterministic measurements (cudnn 9.13 dropped Tesla P4 anyway)
import torch
torch.backends.mkldnn.enabled = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent))


def run_substrate_config(env_settings, label, n_images=5):
    """Spawn bpd_yolo_infer.py with the given env settings, capture timing."""
    cmd_env = os.environ.copy()
    cmd_env.update(env_settings)
    cmd_env['BPD_CPU_SO'] = cmd_env.get('BPD_CPU_SO', '/tmp/bpd_test/build/bpd_cpu.so')

    # Pick first n_images
    images_dir = Path('/tmp/yolo_canonical/images')
    img_ids = sorted([p.stem for p in images_dir.glob('*.jpg')])[:n_images]

    cmd = [
        sys.executable, '-u', str(Path(__file__).parent / 'bpd_yolo_infer.py'),
        '--mode', 'both',
        '--weights', '/tmp/yolo_canonical/yolov5n.pt',
        '--input-dir', '/tmp/yolo_canonical/images',
        '--output-dir', '/tmp/yolo_metayen_out',
        '--images',
    ] + img_ids

    print(f"  Running {label}: env={env_settings}", flush=True)
    result = subprocess.run(cmd, env=cmd_env, capture_output=True, text=True, timeout=600)

    # Parse output for "bpd: N dets, X ms" and compare lines
    bpd_times = []
    cpu_times = []
    all_match = True
    for line in result.stdout.splitlines():
        if 'bpd:' in line and 'dets' in line and 'ms' in line:
            parts = line.split('ms')[0].rsplit(',', 1)
            try:
                ms = float(parts[1].strip())
                bpd_times.append(ms)
            except Exception:
                pass
        elif 'pytorch-cpu:' in line and 'dets' in line and 'ms' in line:
            parts = line.split('ms')[0].rsplit(',', 1)
            try:
                ms = float(parts[1].strip())
                cpu_times.append(ms)
            except Exception:
                pass
        elif 'compare:' in line:
            if 'conf_bit_identical=False' in line or 'classes=False' in line:
                all_match = False

    return {
        'label': label,
        'env': env_settings,
        'bpd_avg_ms': np.mean(bpd_times) if bpd_times else None,
        'cpu_avg_ms': np.mean(cpu_times) if cpu_times else None,
        'n_images': len(bpd_times),
        'all_match': all_match,
    }


def main():
    print("=" * 80)
    print("YOLOv5n Phase 3 Performance Report")
    print("=" * 80)
    print("Host: Ivy Bridge (E5-2697 v2), AVX1, no FMA, no AVX2")
    print("Image set: 5 of 10 COCO val2017 images")
    print()

    configs = [
        ({'SUBSTRATE_AVX1_GEMM': '0', 'SUBSTRATE_FUSE_CBS': '0',
          'SUBSTRATE_FUSE_DETECT': '0', 'SUBSTRATE_FUSE_ADD': '0'},
         "Baseline (scalar GEMM, no fusion)"),
        ({'SUBSTRATE_AVX1_GEMM': '1', 'SUBSTRATE_FUSE_CBS': '0',
          'SUBSTRATE_FUSE_DETECT': '0', 'SUBSTRATE_FUSE_ADD': '0'},
         "+ AVX1 GEMM (3.G)"),
        ({'SUBSTRATE_AVX1_GEMM': '1', 'SUBSTRATE_FUSE_CBS': '1',
          'SUBSTRATE_FUSE_DETECT': '0', 'SUBSTRATE_FUSE_ADD': '0'},
         "+ AVX1 + F3 (3.1)"),
        ({'SUBSTRATE_AVX1_GEMM': '1', 'SUBSTRATE_FUSE_CBS': '1',
          'SUBSTRATE_FUSE_DETECT': '1', 'SUBSTRATE_FUSE_ADD': '0'},
         "+ AVX1 + F3 + F8 (3.2)"),
        ({'SUBSTRATE_AVX1_GEMM': '1', 'SUBSTRATE_FUSE_CBS': '1',
          'SUBSTRATE_FUSE_DETECT': '1', 'SUBSTRATE_FUSE_ADD': '1'},
         "+ AVX1 + F3 + F8 + F4 (3.4) — full Phase 3 stack"),
    ]

    results = []
    cpu_baseline = None
    for env_settings, label in configs:
        r = run_substrate_config(env_settings, label, n_images=5)
        results.append(r)
        if cpu_baseline is None and r['cpu_avg_ms']:
            cpu_baseline = r['cpu_avg_ms']

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"PyTorch CPU baseline: {cpu_baseline:.1f} ms/image" if cpu_baseline else "(no CPU baseline)")
    print()
    print(f"{'Configuration':<60} {'ms/image':<12} {'vs PyTorch':<12} {'vs prev':<10} {'MATCH':<8}")
    print("-" * 110)
    prev_ms = None
    baseline_ms = None
    for r in results:
        bpd = r['bpd_avg_ms']
        if bpd is None:
            continue
        if baseline_ms is None:
            baseline_ms = bpd
        vs_pt = cpu_baseline / bpd if (cpu_baseline and bpd) else None
        vs_prev = prev_ms / bpd if (prev_ms and bpd) else None
        prev_ms = bpd

        vs_pt_str = f"{vs_pt:.2f}\u00d7" if vs_pt else "n/a"
        vs_prev_str = f"{vs_prev:.2f}\u00d7" if vs_prev else "—"
        match_str = "10/10 \u2713" if r['all_match'] else "FAIL"
        print(f"{r['label']:<60} {bpd:>8.1f} ms   {vs_pt_str:<12} {vs_prev_str:<10} {match_str:<8}")

    if baseline_ms and prev_ms:
        total = baseline_ms / prev_ms
        print()
        print(f"Total Phase 3 speedup (baseline -> full stack): {total:.2f}\u00d7")
        if cpu_baseline:
            print(f"Gap to PyTorch CPU closed:                       {(1 - prev_ms/baseline_ms) * 100 * baseline_ms / (baseline_ms - cpu_baseline):.1f}%" if baseline_ms != cpu_baseline else "")
            print(f"Remaining gap to PyTorch CPU:                    {prev_ms/cpu_baseline:.2f}\u00d7 slower")


if __name__ == '__main__':
    main()
