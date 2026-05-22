#!/usr/bin/env python3
"""Test bpd_rope_neox_cpu against ggml's captured ROPE output."""
import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

from bpd_llamatov_infer import (
    BpdLlamaConfig, BpdLlamaLayerWeights, BpdLlamaWeights,
    build_model, c_float_p, c_uint8_p, c_int32_p,
)
from llama_fixture_loader import load_manifest, find_op


def compare(ours, ref, name):
    ours_flat = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if ours_flat.shape != ref_flat.shape:
        print(f"  ! {name}: shape mismatch ours={ours_flat.shape} ref={ref_flat.shape}")
        return None
    max_abs = float(np.abs(ours_flat - ref_flat).max())
    ai = ours_flat.view(np.int32).astype(np.int64)
    bi = ref_flat.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    max_ulp = int(diffs.max())
    n_diff = int((diffs > 0).sum())
    icon = "PASS" if max_ulp == 0 else "FAIL"
    print(f"  [{icon}] {name:55s}: max_abs={max_abs:.6e}, max_ulp={max_ulp}, n_diff={n_diff}/{diffs.size}")
    return max_ulp, max_abs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--so", default="/tmp/bpd_test/build/bpd_cpu.so")
    p.add_argument("--tokens", required=True)
    p.add_argument("--fixture-dir", default="/tmp/llama_dump_hello_8")
    args = p.parse_args()

    lib = ctypes.CDLL(args.so)
    # bpd_rope_neox_cpu: extern void bpd_rope_neox_cpu(const float* input, float* output,
    #     const int32_t* pos, int n_heads, int head_dim, int n_tokens,
    #     float rope_base, int rope_dim, ...)
    # We need to inspect the actual signature
    
    cfg, weights, loader = build_model(args.gguf)
    tensors = load_manifest(args.fixture_dir)

    # Get the captured Qcur and Kcur (pre-ROPE \u2014 from MUL_MAT) and post-ROPE (from ROPE op)
    # The pre-ROPE Q is at idx 7 (MUL_MAT Qcur-0 shape 2048,6); the post-ROPE Q is at idx 11 (ROPE).
    # But idx 7 is shape (2048,6), idx 11 is shape (64,32,6,1) \u2014 the RESHAPE happens.
    # We feed the reshaped Q (64, 32, 6) to ROPE.

    # First check: which Qcur are we looking at exactly?
    qcur_mulmat = find_op(tensors, name_substring="Qcur-0", op_desc="MUL_MAT").as_numpy()  # (6, 2048)
    qcur_reshaped = find_op(tensors, name_substring="Qcur-0 (reshaped)", op_desc="RESHAPE")
    qcur_rope = find_op(tensors, name_substring="Qcur-0", op_desc="ROPE").as_numpy()  # (6, 32, 64) probably

    print(f"\nQcur-0 (MUL_MAT) shape: {qcur_mulmat.shape}")
    print(f"Qcur-0 (RESHAPE) ne: {qcur_reshaped.ne}")
    print(f"Qcur-0 (ROPE) shape: {qcur_rope.shape}")

    # Reshape Qcur to (n_tokens, n_heads, head_dim) layout
    n_tokens = qcur_mulmat.shape[0]
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim
    q_in = qcur_mulmat.reshape(n_tokens, n_heads, head_dim)

    # The captured ROPE output has shape (n_tokens, n_heads, head_dim) likely
    # (or with a transpose). Let me just verify the layout via the ROPE tensor's ne.
    print(f"\nROPE output shape: {qcur_rope.shape}")
    print(f"Expected: ({n_tokens}, {n_heads}, {head_dim}) = ({n_tokens*n_heads*head_dim})")

    # Find rope_neox_cpu signature
    if hasattr(lib, 'bpd_rope_neox_cpu'):
        print(f"\nbpd_rope_neox_cpu is in the library")
    else:
        print(f"\nERROR: bpd_rope_neox_cpu NOT in library")
        return

    # NEW signature: bpd_rope_neox_freqs_cpu(input, output, pos_ids, freq_factors,
    #     n_tokens, n_heads, head_dim, n_dims, freq_base)
    lib.bpd_rope_neox_freqs_cpu.restype = None
    lib.bpd_rope_neox_freqs_cpu.argtypes = [
        c_float_p, c_float_p, c_int32_p, c_float_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
    ]

    # Get rope_freqs from the fixture
    rope_freqs_op = find_op(tensors, name_substring="rope_freqs.weight")
    rope_freqs = np.ascontiguousarray(rope_freqs_op.as_numpy(), dtype=np.float32) if rope_freqs_op else None
    print(f"\nrope_freqs.weight: shape={rope_freqs.shape if rope_freqs is not None else None}")
    if rope_freqs is not None:
        print(f"  values: {rope_freqs.flatten()[:8]} ... {rope_freqs.flatten()[-8:]}")

    pos_ids = np.arange(n_tokens, dtype=np.int32)
    q_in_c = np.ascontiguousarray(q_in, dtype=np.float32)

    # Test 1: WITHOUT freq_factors (NULL) \u2014 should still diverge
    q_out_nofreq = np.zeros_like(q_in_c)
    lib.bpd_rope_neox_freqs_cpu(
        q_in_c.ctypes.data_as(c_float_p),
        q_out_nofreq.ctypes.data_as(c_float_p),
        pos_ids.ctypes.data_as(c_int32_p),
        ctypes.cast(None, c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(n_heads),
        ctypes.c_int(head_dim), ctypes.c_int(cfg.rope_dim),
        ctypes.c_float(cfg.rope_base))
    print(f"\n[Test 1: ROPE without freq_factors (should still diverge)]")
    compare(q_out_nofreq, qcur_rope, "Qcur-0 after ROPE no-freqs")

    # Test 2: WITH freq_factors
    q_out_freqs = np.zeros_like(q_in_c)
    lib.bpd_rope_neox_freqs_cpu(
        q_in_c.ctypes.data_as(c_float_p),
        q_out_freqs.ctypes.data_as(c_float_p),
        pos_ids.ctypes.data_as(c_int32_p),
        rope_freqs.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(n_heads),
        ctypes.c_int(head_dim), ctypes.c_int(cfg.rope_dim),
        ctypes.c_float(cfg.rope_base))
    print(f"\n[Test 2: ROPE WITH freq_factors (expect 0 ULP)]")
    compare(q_out_freqs, qcur_rope, "Qcur-0 after ROPE with-freqs")

    # Compare against ggml's captured Qcur-0 ROPE output
    # Need to handle potential shape transposes
    print(f"\nOurs shape: {q_out.shape}, sample: {q_out.flatten()[:5]}")
    print(f"Ref  shape: {qcur_rope.shape}, sample: {qcur_rope.flatten()[:5]}")
    compare(q_out, qcur_rope, "Qcur-0 after ROPE (ours vs ggml)")


if __name__ == "__main__":
    main()
