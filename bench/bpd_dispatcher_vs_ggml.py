#!/usr/bin/env python3
"""bpd_dispatcher_vs_ggml.py \u2014 empirically compare the tile dispatcher vs ggml fixture.

Tests both bpd_qmatmul_q8_0_cpu (scalar mirror) AND bpd_qmatmul_q8_0_llamafile_cpu
(tile dispatcher) on the EXACT input that flows through layer 0's Q-projection.
Compares each to the captured ggml fixture (Qcur-0).

Question being answered: does the tile dispatcher actually achieve 0 ULP vs ggml
on the n=6 production shape? Or does it produce different bits than ggml's
actual mnpack-dispatched gemm?
"""
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


def compare(ours, ref, name):
    ours_flat = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if ours_flat.shape != ref_flat.shape:
        print(f"  ! {name}: shape mismatch ours={ours_flat.shape} ref={ref_flat.shape}")
        return None
    max_abs = float(np.abs(ours_flat - ref_flat).max())
    max_ulp, n_diff, n_total = ulp_distance(ours_flat, ref_flat)
    icon = "PASS" if max_ulp == 0 else "FAIL"
    print(f"  [{icon}] {name:50s}: max_abs={max_abs:.6e}, max_ulp={max_ulp}, n_diff={n_diff}/{n_total}")
    return max_ulp, max_abs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--so", default="/tmp/bpd_test/build/bpd_cpu.so")
    p.add_argument("--tokens", required=True)
    p.add_argument("--fixture-dir", default="/tmp/llama_dump_hello_8")
    args = p.parse_args()

    lib = ctypes.CDLL(args.so)
    lib.bpd_embed_lookup_q8_0_cpu.restype = None
    lib.bpd_embed_lookup_q8_0_cpu.argtypes = [c_uint8_p, c_int32_p, c_float_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_rmsnorm_llama_cpu.restype = None
    lib.bpd_rmsnorm_llama_cpu.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.bpd_mul_broadcast_cpu.restype = None
    lib.bpd_mul_broadcast_cpu.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int]

    # SCALAR matmul: (W, X, out, M=n_tokens, N=output_dim, K)
    lib.bpd_qmatmul_q8_0_cpu.restype = None
    lib.bpd_qmatmul_q8_0_cpu.argtypes = [c_uint8_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    # LLAMAFILE matmul: (W, X, out, m_weight=output_dim, m_tokens=n_tokens, K)
    lib.bpd_qmatmul_q8_0_llamafile_cpu.restype = None
    lib.bpd_qmatmul_q8_0_llamafile_cpu.argtypes = [c_uint8_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    cfg, weights, loader = build_model(args.gguf)
    tensors = load_manifest(args.fixture_dir)

    prompt_tokens = [int(t) for t in args.tokens.split(",")]
    n_tokens = len(prompt_tokens)

    print(f"\n[setup] n_tokens={n_tokens}, embed_dim={cfg.embed_dim}")

    # STAGE 1: embed lookup
    token_ids = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
    inp_embd = np.zeros(n_tokens * cfg.embed_dim, dtype=np.float32)
    lib.bpd_embed_lookup_q8_0_cpu(
        weights.token_embd, token_ids.ctypes.data_as(c_int32_p),
        inp_embd.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim))

    # STAGE 2: RMSNorm
    norm_out = np.zeros_like(inp_embd)
    ones = np.ones(cfg.embed_dim, dtype=np.float32)
    lib.bpd_rmsnorm_llama_cpu(
        inp_embd.ctypes.data_as(c_float_p),
        ones.ctypes.data_as(c_float_p),
        norm_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim), ctypes.c_float(cfg.rms_eps))

    # STAGE 3: attn_norm MUL
    attn_norm_w_arr = np.ctypeslib.as_array(weights.layers[0].attn_norm_w, shape=(cfg.embed_dim,))
    attn_norm_out = np.zeros_like(inp_embd)
    lib.bpd_mul_broadcast_cpu(
        norm_out.ctypes.data_as(c_float_p),
        attn_norm_w_arr.ctypes.data_as(c_float_p),
        attn_norm_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim))

    # Sanity check vs fixture
    ref_an = find_op(tensors, name_substring="attn_norm-0", op_desc="MUL")
    compare(attn_norm_out.reshape(n_tokens, cfg.embed_dim), ref_an.as_numpy(), "attn_norm-0 (input to Q-projection)")

    # ------- STAGE 4: Q-projection, BOTH kernels -------
    embed_dim = cfg.embed_dim
    w_q_arr = np.ctypeslib.as_array(weights.layers[0].w_q,
        shape=(embed_dim * embed_dim // 32 * 34,))
    ref_qcur = find_op(tensors, name_substring="Qcur-0", op_desc="MUL_MAT")
    ref_qcur_arr = ref_qcur.as_numpy()  # shape (n_tokens, embed_dim)

    # 4a. SCALAR path: bpd_qmatmul_q8_0_cpu(W, X, out, M=n_tokens, N=embed_dim, K=embed_dim)
    qcur_scalar = np.zeros(n_tokens * embed_dim, dtype=np.float32)
    lib.bpd_qmatmul_q8_0_cpu(
        w_q_arr.ctypes.data_as(c_uint8_p),
        attn_norm_out.ctypes.data_as(c_float_p),
        qcur_scalar.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(embed_dim), ctypes.c_int(embed_dim))
    print(f"\n[Q-projection: SCALAR path \u2014 bpd_qmatmul_q8_0_cpu]")
    compare(qcur_scalar.reshape(n_tokens, embed_dim), ref_qcur_arr, "Qcur-0 (scalar) vs ggml fixture")

    # 4b. LLAMAFILE TILE path: bpd_qmatmul_q8_0_llamafile_cpu(W, X, out, m_weight=embed_dim, m_tokens=n_tokens, K)
    qcur_tile = np.zeros(n_tokens * embed_dim, dtype=np.float32)
    lib.bpd_qmatmul_q8_0_llamafile_cpu(
        w_q_arr.ctypes.data_as(c_uint8_p),
        attn_norm_out.ctypes.data_as(c_float_p),
        qcur_tile.ctypes.data_as(c_float_p),
        ctypes.c_int(embed_dim), ctypes.c_int(n_tokens), ctypes.c_int(embed_dim))
    print(f"\n[Q-projection: TILE dispatcher \u2014 bpd_qmatmul_q8_0_llamafile_cpu]")
    compare(qcur_tile.reshape(n_tokens, embed_dim), ref_qcur_arr, "Qcur-0 (tile) vs ggml fixture")

    # 4c. Compare scalar vs tile directly: are they making the same approximation
    # error, or different ones?
    print(f"\n[Compare: scalar vs tile, internally]")
    compare(qcur_scalar.reshape(n_tokens, embed_dim),
            qcur_tile.reshape(n_tokens, embed_dim),
            "Qcur-0 scalar vs tile (substrate-internal)")

    # Diagnosis
    print(f"\n[diagnosis]")
    s_max_abs = float(np.abs(qcur_scalar - ref_qcur_arr.reshape(-1)).max())
    t_max_abs = float(np.abs(qcur_tile - ref_qcur_arr.reshape(-1)).max())
    print(f"  scalar diverges from ggml by: max_abs={s_max_abs:.6e}")
    print(f"  tile   diverges from ggml by: max_abs={t_max_abs:.6e}")
    if t_max_abs < s_max_abs:
        print(f"  -> tile dispatcher is CLOSER to ggml than scalar. Progress.")
    elif t_max_abs > s_max_abs:
        print(f"  -> tile dispatcher is FURTHER from ggml than scalar. Regression.")
    else:
        print(f"  -> tile dispatcher matches scalar exactly. Dispatcher reduction order = scalar reduction order.")


if __name__ == "__main__":
    main()
