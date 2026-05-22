#!/usr/bin/env python3
"""per_op_gates.py — verify each captured ggml operation against our substrate.

For each operation in the captured llama-eval-callback trace:
  1. Identify the operation type (GET_ROWS, RMS_NORM, MUL, MUL_MAT, etc.)
  2. Locate the source tensors (inputs) and the captured output
  3. Call our substrate's equivalent kernel on the same inputs
  4. Compare the result to the captured output, byte-for-byte

Emits a JSON report. Multi-sovereign-verifiable: anyone can run this on
their own hardware with their own build and compare reports.

Substrate-design discipline: this is the finest-grained correctness gate
in the LlamaTov verification harness. Failures here point at the precise
sub-operation where the substrate's behavior drifts from ggml's.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


HARNESS_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

from llama_fixture_loader import load_manifest, find_op  # noqa: E402


# ─── ctypes setup ─────────────────────────────────────────────────────────
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_int32_p = ctypes.POINTER(ctypes.c_int32)


def setup_lib(so_path):
    lib = ctypes.CDLL(so_path)
    sigs = {
        'bpd_dequant_q8_0_cpu': (None, [c_uint8_p, c_float_p, ctypes.c_int]),
        'bpd_quant_q8_0_cpu':   (None, [c_float_p, c_uint8_p, ctypes.c_int]),
        'bpd_qdot_q8_0_q8_0_cpu': (ctypes.c_float, [c_uint8_p, c_uint8_p, ctypes.c_int]),
        'bpd_qmatmul_q8_0_cpu': (None, [c_uint8_p, c_float_p, c_float_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int]),
        'bpd_qmatmul_q8_0_llamafile_cpu': (None, [c_uint8_p, c_float_p, c_float_p,
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_int]),
        'bpd_mul_broadcast_cpu': (None, [c_float_p, c_float_p, c_float_p,
                                          ctypes.c_int, ctypes.c_int]),
        'bpd_embed_lookup_q8_0_cpu': (None, [c_uint8_p, c_int32_p, c_float_p,
                                              ctypes.c_int, ctypes.c_int]),
        'bpd_rmsnorm_llama_cpu': (None, [c_float_p, c_float_p, c_float_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_float]),
    }
    for name, (restype, argtypes) in sigs.items():
        if hasattr(lib, name):
            getattr(lib, name).restype = restype
            getattr(lib, name).argtypes = argtypes
    return lib


# ─── ULP distance helper ──────────────────────────────────────────────────
def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return None, None, a.size
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def compare(ours, ref):
    ours_flat = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if ours_flat.shape != ref_flat.shape:
        return {
            "status": "shape_mismatch",
            "ours_shape": list(ours.shape),
            "ref_shape": list(ref.shape),
        }
    max_abs = float(np.abs(ours_flat - ref_flat).max())
    max_ulp, n_diff, n_total = ulp_distance(ours_flat, ref_flat)
    return {
        "status": "pass" if max_ulp == 0 else "fail",
        "max_abs": max_abs,
        "max_ulp": max_ulp,
        "n_diff": n_diff,
        "n_total": n_total,
    }


# ─── Op-specific verifiers ────────────────────────────────────────────────
def verify_get_rows(lib, tensors, op, idx, ctx):
    """GET_ROWS: gather rows from a quantized table by token IDs.
    Our equivalent: bpd_embed_lookup_q8_0_cpu (if table is Q8_0).
    Skip if the table is not Q8_0 (e.g., F32 — not our test surface).
    """
    # Locate the source table (Q8_0 weight) and indices (int32 token IDs)
    # The inputs are NOT in the fixture as separate tensors except via src[].
    # We use the GGUF directly for the table; the indices we get from a leaf tensor.

    # For inp_embd (first GET_ROWS), src1 = leaf_2 (token IDs)
    if op.name != "inp_embd":
        return {"status": "skip", "reason": "non-embedding GET_ROWS not yet covered"}

    leaf2 = find_op(tensors, name_substring="leaf_2", op_desc="NONE")
    if leaf2 is None:
        return {"status": "skip", "reason": "leaf_2 (token IDs) not in fixture"}

    token_ids_arr = leaf2.as_numpy()
    n_tokens = int(token_ids_arr.shape[0])
    embed_dim = ctx["cfg"]["embed_dim"]
    if not hasattr(lib, 'bpd_embed_lookup_q8_0_cpu'):
        return {"status": "skip", "reason": "bpd_embed_lookup_q8_0_cpu not in substrate"}

    # The token_embd.weight is in the GGUF; ctx["token_embd_bytes"] holds raw bytes.
    table = ctx["token_embd_bytes"]
    out = np.zeros(n_tokens * embed_dim, dtype=np.float32)
    token_ids_c = np.ascontiguousarray(token_ids_arr, dtype=np.int32)
    lib.bpd_embed_lookup_q8_0_cpu(
        table.ctypes.data_as(c_uint8_p),
        token_ids_c.ctypes.data_as(c_int32_p),
        out.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(embed_dim))
    ref = op.as_numpy()
    return compare(out.reshape(n_tokens, embed_dim), ref)


def verify_rms_norm(lib, tensors, op, idx, ctx):
    """RMS_NORM: input -> output. ggml's RMS_NORM has no weight (the MUL
    op applies the weight separately). Verify our bpd_rmsnorm_llama_cpu
    with weight=ones.
    """
    # Find the prior op whose output is the input to this RMS_NORM.
    # ggml's first norm-0 takes inp_embd. For per-layer norms, find the
    # most recent l_out-(li-1) or ffn_inp-li that this RMS_NORM consumes.
    # For the simplest case (norm-0 on inp_embd), use inp_embd.

    if not hasattr(lib, 'bpd_rmsnorm_llama_cpu'):
        return {"status": "skip", "reason": "bpd_rmsnorm_llama_cpu not in substrate"}

    # ggml's RMS_NORM at idx 2 is norm-0 (RMS_NORM of inp_embd).
    if op.name == "norm-0" and op.idx == 2:
        # Input is inp_embd (idx 0)
        inp = find_op(tensors, name_substring="inp_embd", op_desc="GET_ROWS")
        if inp is None:
            return {"status": "skip", "reason": "inp_embd not in fixture"}
        x = np.ascontiguousarray(inp.as_numpy(), dtype=np.float32)
        n_tokens, embed_dim = x.shape
        ones = np.ones(embed_dim, dtype=np.float32)
        out = np.zeros_like(x)
        lib.bpd_rmsnorm_llama_cpu(
            x.ctypes.data_as(c_float_p),
            ones.ctypes.data_as(c_float_p),
            out.ctypes.data_as(c_float_p),
            ctypes.c_int(n_tokens), ctypes.c_int(embed_dim),
            ctypes.c_float(ctx["cfg"]["rms_eps"]))
        ref = op.as_numpy()
        return compare(out, ref)

    # For other RMS_NORM ops in the trace, we'd need src linking.
    # Phase L.1: skip and emit a future-work note.
    return {"status": "skip", "reason": "src-linking for non-first RMS_NORM not yet implemented"}


def verify_mul_broadcast(lib, tensors, op, idx, ctx):
    """MUL: element-wise multiply with broadcast.
    First MUL in layer 0 (attn_norm-0): norm-0 * blk.0.attn_norm.weight.
    """
    if not hasattr(lib, 'bpd_mul_broadcast_cpu'):
        return {"status": "skip", "reason": "bpd_mul_broadcast_cpu not in substrate"}

    if op.name == "attn_norm-0" and op.idx == 4:
        norm0 = find_op(tensors, name_substring="norm-0", op_desc="RMS_NORM")
        weight = find_op(tensors, name_substring="attn_norm.weight", op_desc="NONE")
        if norm0 is None or weight is None:
            return {"status": "skip", "reason": "norm-0 or attn_norm.weight not in fixture"}
        a = np.ascontiguousarray(norm0.as_numpy(), dtype=np.float32)
        b = np.ascontiguousarray(weight.as_numpy(), dtype=np.float32)
        outer = a.shape[0]
        inner = a.shape[1]
        out = np.zeros_like(a)
        lib.bpd_mul_broadcast_cpu(
            a.ctypes.data_as(c_float_p),
            b.ctypes.data_as(c_float_p),
            out.ctypes.data_as(c_float_p),
            ctypes.c_int(outer), ctypes.c_int(inner))
        ref = op.as_numpy()
        return compare(out, ref)

    return {"status": "skip", "reason": "src-linking for non-first MUL not yet implemented"}


def verify_mul_mat(lib, tensors, op, idx, ctx):
    """MUL_MAT: ggml MUL_MAT(weight, input) = input @ weight^T.
    First MUL_MAT in layer 0 (Qcur-0): blk.0.attn_q.weight x attn_norm-0.
    Verifies our bpd_qmatmul_q8_0_cpu (the scalar/vec_dot mirror).
    """
    if not hasattr(lib, 'bpd_qmatmul_q8_0_cpu'):
        return {"status": "skip", "reason": "bpd_qmatmul_q8_0_cpu not in substrate"}

    # First Q-projection: idx 7, Qcur-0
    if op.name == "Qcur-0" and op.op_desc == "MUL_MAT" and op.idx == 7:
        attn_norm = find_op(tensors, name_substring="attn_norm-0", op_desc="MUL")
        if attn_norm is None:
            return {"status": "skip", "reason": "attn_norm-0 not in fixture"}

        x = np.ascontiguousarray(attn_norm.as_numpy(), dtype=np.float32)
        n_tokens, embed_dim = x.shape
        out_dim = op.ne[0]  # weight has shape (in, out); output is (n_tokens, out_dim)

        # Find weight bytes: blk.0.attn_q.weight from the GGUF
        W = ctx["weights"]["blk.0.attn_q.weight"]
        out = np.zeros(n_tokens * out_dim, dtype=np.float32)
        lib.bpd_qmatmul_q8_0_cpu(
            W.ctypes.data_as(c_uint8_p),
            x.ctypes.data_as(c_float_p),
            out.ctypes.data_as(c_float_p),
            ctypes.c_int(n_tokens), ctypes.c_int(out_dim), ctypes.c_int(embed_dim))
        ref = op.as_numpy()
        return compare(out.reshape(n_tokens, out_dim), ref)

    return {"status": "skip", "reason": "src-linking for non-first MUL_MAT not yet implemented"}


# ─── Op type dispatch table ───────────────────────────────────────────────
OP_VERIFIERS = {
    "GET_ROWS": verify_get_rows,
    "RMS_NORM": verify_rms_norm,
    "MUL": verify_mul_broadcast,
    "MUL_MAT": verify_mul_mat,
    # TODO: ROPE, CPY (kv-cache), SOFT_MAX, etc.
}


# ─── Hardware/context helpers ─────────────────────────────────────────────
def gather_hardware_info():
    info = {
        "cpu_model": "unknown",
        "isa": [],
        "compiler": "unknown",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip().split()
                    info["isa"] = [f for f in flags if f in
                                    {"sse", "sse2", "sse3", "ssse3", "sse4_1",
                                     "sse4_2", "avx", "avx2", "avx512f", "fma", "f16c"}]
    except FileNotFoundError:
        pass
    try:
        gcc = subprocess.check_output(["gcc", "--version"], stderr=subprocess.STDOUT).decode()
        info["compiler"] = gcc.split("\n")[0]
    except Exception:
        pass
    return info


def fixture_sha256(fixture_dir):
    """Hash the manifest + a sample of bin files for fixture identification."""
    manifest = Path(fixture_dir) / "manifest.tsv"
    h = hashlib.sha256()
    if manifest.exists():
        h.update(manifest.read_bytes())
    return h.hexdigest()[:16]


def get_token_embd_bytes(gguf_path):
    """Read token_embd.weight raw bytes via gguf_query.pl."""
    script = REPO_ROOT / "tests" / "gguf_query.pl"
    result = subprocess.run(
        ["swipl", "-q", "-g", f"consult('{script}'), gguf_query_main", "--",
         str(gguf_path), "token_embd.weight"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    line = result.stdout.strip().splitlines()[-1]
    fields = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v
    abs_off = int(fields["ABS_OFFSET"])
    size = int(fields["SIZE"])
    return np.fromfile(gguf_path, dtype=np.uint8, count=size, offset=abs_off)


def get_layer0_weights(gguf_path):
    """Read all blk.0.* weights via gguf_query.pl."""
    weights = {}
    for name in ["blk.0.attn_q.weight", "blk.0.attn_k.weight",
                 "blk.0.attn_v.weight", "blk.0.attn_output.weight"]:
        script = REPO_ROOT / "tests" / "gguf_query.pl"
        result = subprocess.run(
            ["swipl", "-q", "-g", f"consult('{script}'), gguf_query_main", "--",
             str(gguf_path), name],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        line = result.stdout.strip().splitlines()[-1]
        fields = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        abs_off = int(fields["ABS_OFFSET"])
        size = int(fields["SIZE"])
        weights[name] = np.fromfile(gguf_path, dtype=np.uint8, count=size, offset=abs_off)
    return weights


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", required=True,
                   help="LLAMA_DUMP_DIR from a llama-eval-callback run")
    p.add_argument("--so", required=True,
                   help="Path to bpd_cpu.so")
    p.add_argument("--gguf", required=True,
                   help="Path to the GGUF model file")
    p.add_argument("--verifier", default=None,
                   help="Identifier for this run (e.g., 'manus@ruachtov.ai')")
    p.add_argument("--report", default="/tmp/per_op_report.json",
                   help="Output JSON report path")
    p.add_argument("--max-ops", type=int, default=200,
                   help="Stop after this many ops (default: 200; full trace = ~1100)")
    args = p.parse_args()

    print(f"[harness] LlamaTov per-op gates v{HARNESS_VERSION}")
    print(f"[harness] fixture: {args.fixture_dir}")
    print(f"[harness] substrate: {args.so}")
    print(f"[harness] model: {args.gguf}")

    lib = setup_lib(args.so)
    tensors = load_manifest(args.fixture_dir)
    print(f"[harness] loaded {len(tensors)} fixture tensors")

    # Load needed weight bytes from GGUF
    print(f"[harness] loading weight bytes from GGUF...")
    token_embd_bytes = get_token_embd_bytes(args.gguf)
    layer0_weights = get_layer0_weights(args.gguf)
    print(f"[harness]   token_embd: {len(token_embd_bytes)/1e6:.1f} MB")
    print(f"[harness]   layer-0 weights: {len(layer0_weights)} tensors")

    cfg = {
        "embed_dim": 2048,
        "ffn_dim": 8192,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "rms_eps": 1e-5,
        "rope_base": 500000.0,
    }

    ctx = {
        "cfg": cfg,
        "token_embd_bytes": token_embd_bytes,
        "weights": layer0_weights,
    }

    # Iterate ops, verify each
    results = []
    for op in tensors[:args.max_ops]:
        verifier = OP_VERIFIERS.get(op.op_desc)
        if verifier is None:
            entry = {"op_idx": op.idx, "op_name": op.name, "op_desc": op.op_desc,
                     "shape": list(op.ne), "status": "no_verifier"}
        else:
            try:
                outcome = verifier(lib, tensors, op, op.idx, ctx)
            except Exception as e:
                outcome = {"status": "error", "error": str(e)}
            entry = {"op_idx": op.idx, "op_name": op.name, "op_desc": op.op_desc,
                     "shape": list(op.ne), **outcome}
        results.append(entry)
        # Print progress for failed/passed ops
        if entry.get("status") in {"pass", "fail"}:
            icon = "✅" if entry["status"] == "pass" else "❌"
            extra = ""
            if entry["status"] == "fail":
                extra = f"  max_ulp={entry.get('max_ulp')} max_abs={entry.get('max_abs', 0):.2e}"
            print(f"  {icon} [{op.idx:04d}] {op.op_desc:8s} {op.name:30s}{extra}")

    # Summary
    n_pass = sum(1 for r in results if r.get("status") == "pass")
    n_fail = sum(1 for r in results if r.get("status") == "fail")
    n_skip = sum(1 for r in results if r.get("status") == "skip")
    n_other = len(results) - n_pass - n_fail - n_skip
    first_div = next((r for r in results if r.get("status") == "fail"), None)
    print(f"\n[summary] pass={n_pass}, fail={n_fail}, skip={n_skip}, other={n_other}")
    if first_div:
        print(f"[summary] first divergence at idx {first_div['op_idx']}: "
              f"{first_div['op_name']} ({first_div['op_desc']})")

    # Build report
    report = {
        "harness_version": HARNESS_VERSION,
        "verifier": args.verifier or os.environ.get("USER", "unknown"),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture_dir": args.fixture_dir,
        "fixture_sha256_partial": fixture_sha256(args.fixture_dir),
        "model": {"path": args.gguf, "config": cfg},
        "hardware": gather_hardware_info(),
        "results": results,
        "summary": {
            "total_ops_checked": len(results),
            "pass": n_pass,
            "fail": n_fail,
            "skip": n_skip,
            "first_divergent_idx": first_div["op_idx"] if first_div else None,
            "first_divergent_name": first_div["op_name"] if first_div else None,
        },
    }

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[harness] wrote report: {args.report}")


if __name__ == "__main__":
    main()
