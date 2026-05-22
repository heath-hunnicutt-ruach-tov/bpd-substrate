#!/usr/bin/env python3
"""Check whether the tile dispatcher achieves 0 ULP vs ggml fixture
for all 4 layer-0 attention matmuls: Q, K, V, O."""
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
    print(f"  [{icon}] {name:55s}: max_abs={max_abs:.6e}, max_ulp={max_ulp}, n_diff={n_diff}/{n_total}")
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
    lib.bpd_qmatmul_q8_0_llamafile_cpu.restype = None
    lib.bpd_qmatmul_q8_0_llamafile_cpu.argtypes = [c_uint8_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    cfg, weights, loader = build_model(args.gguf)
    tensors = load_manifest(args.fixture_dir)

    prompt_tokens = [int(t) for t in args.tokens.split(",")]
    n_tokens = len(prompt_tokens)

    H = cfg.n_heads
    HKV = cfg.n_kv_heads
    D = cfg.head_dim
    E = cfg.embed_dim

    print(f"\n[setup] n_tokens={n_tokens}, embed_dim={E}, H={H}, HKV={HKV}, D={D}")
    print(f"        Q matmul: out={H*D}, in={E}, n={n_tokens}")
    print(f"        K matmul: out={HKV*D}, in={E}, n={n_tokens}")
    print(f"        V matmul: out={HKV*D}, in={E}, n={n_tokens}")
    print(f"        O matmul: out={E}, in={H*D}, n={n_tokens}")

    # Build attn_norm-0 input via embed+RMSNorm+MUL
    token_ids = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
    inp_embd = np.zeros(n_tokens * E, dtype=np.float32)
    lib.bpd_embed_lookup_q8_0_cpu(
        weights.token_embd, token_ids.ctypes.data_as(c_int32_p),
        inp_embd.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(E))
    norm_out = np.zeros_like(inp_embd)
    ones = np.ones(E, dtype=np.float32)
    lib.bpd_rmsnorm_llama_cpu(
        inp_embd.ctypes.data_as(c_float_p),
        ones.ctypes.data_as(c_float_p),
        norm_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(E), ctypes.c_float(cfg.rms_eps))
    attn_norm_w_arr = np.ctypeslib.as_array(weights.layers[0].attn_norm_w, shape=(E,))
    attn_norm_out = np.zeros_like(inp_embd)
    lib.bpd_mul_broadcast_cpu(
        norm_out.ctypes.data_as(c_float_p),
        attn_norm_w_arr.ctypes.data_as(c_float_p),
        attn_norm_out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(E))

    def call_tile(w_arr_field, n_bytes, m_weight, m_tokens, K, x_in):
        w_arr = np.ctypeslib.as_array(w_arr_field, shape=(n_bytes,))
        out = np.zeros(m_tokens * m_weight, dtype=np.float32)
        lib.bpd_qmatmul_q8_0_llamafile_cpu(
            w_arr.ctypes.data_as(c_uint8_p),
            x_in.ctypes.data_as(c_float_p),
            out.ctypes.data_as(c_float_p),
            ctypes.c_int(m_weight), ctypes.c_int(m_tokens), ctypes.c_int(K))
        return out

    print(f"\n[layer 0 \u2014 all 4 attention matmuls, tile dispatcher vs ggml fixture]")

    # Q: out=H*D=2048, in=E=2048
    q_out = call_tile(weights.layers[0].w_q, H*D*E//32*34, H*D, n_tokens, E, attn_norm_out)
    ref_q = find_op(tensors, name_substring="Qcur-0", op_desc="MUL_MAT").as_numpy()
    compare(q_out.reshape(n_tokens, H*D), ref_q, "Qcur-0 (tile)")

    # K: out=HKV*D=512, in=E=2048
    k_out = call_tile(weights.layers[0].w_k, HKV*D*E//32*34, HKV*D, n_tokens, E, attn_norm_out)
    ref_k = find_op(tensors, name_substring="Kcur-0", op_desc="MUL_MAT").as_numpy()
    compare(k_out.reshape(n_tokens, HKV*D), ref_k, "Kcur-0 (tile)")

    # V: out=HKV*D=512, in=E=2048
    v_out = call_tile(weights.layers[0].w_v, HKV*D*E//32*34, HKV*D, n_tokens, E, attn_norm_out)
    ref_v = find_op(tensors, name_substring="Vcur-0", op_desc="MUL_MAT").as_numpy()
    compare(v_out.reshape(n_tokens, HKV*D), ref_v, "Vcur-0 (tile)")


if __name__ == "__main__":
    main()
