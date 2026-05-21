"""test_llama_kernels.py — 0-ULP gates for our kernels against llama.cpp's flow.

Each test loads a triplet (input_a, input_b, expected_out) from the captured
fixture at /tmp/llama_dump_layer0 (or LLAMA_DUMP_DIR env), calls our kernel,
and asserts bit-identical output.

Run with the env vars set:
  BPD_CPU_SO=/tmp/bpd_test/build/bpd_cpu.so
  LLAMA_DUMP_DIR=/tmp/llama_dump_layer0
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from llama_fixture_loader import load_manifest, find_op, load_tensor


SO = os.environ.get("BPD_CPU_SO", "/tmp/bpd_test/build/bpd_cpu.so")
DUMP_DIR = os.environ.get("LLAMA_DUMP_DIR", "/tmp/llama_dump_layer0")


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


def assert_bit_identical(ref, got):
    max_ulp, n_diff, n_total = ulp_distance(ref, got)
    if max_ulp == 0:
        return TestStatus.PASS, f"0 ULP / {n_total}"
    ref_flat = np.asarray(ref, dtype=np.float32).reshape(-1)
    got_flat = np.asarray(got, dtype=np.float32).reshape(-1)
    diff_mask = (ref_flat.view(np.uint32) != got_flat.view(np.uint32))
    diff_idx = np.where(diff_mask)[0][:5]
    samples = []
    for i in diff_idx:
        samples.append(f"    [{i}]: ref={ref_flat[i]:.7e} got={got_flat[i]:.7e}")
    msg = f"max_ulp={max_ulp} n_diff={n_diff}/{n_total}\n" + "\n".join(samples)
    return TestStatus.FAIL, msg


# ──────────────────────────── Test functions ────────────────────────────

def test_lk_02_mul(lib, tensors):
    """L.1.2 MUL: out[s, d] = in[s, d] * w[d] (broadcast w along seq axis).

    Inputs:
      norm-0 (shape (2, 2048)) at idx 2
      blk.0.attn_norm.weight (shape (2048,)) at idx 6
    Expected:
      attn_norm-0 (shape (2, 2048)) at idx 4
    """
    if not hasattr(lib, 'bpd_mul_broadcast_cpu'):
        return TestStatus.MISSING, "bpd_mul_broadcast_cpu not in substrate"
    norm0 = find_op(tensors, name_substring="norm-0", op_desc="RMS_NORM")
    weight = find_op(tensors, name_substring="attn_norm.weight", op_desc="NONE")
    expected = find_op(tensors, name_substring="attn_norm-0", op_desc="MUL")
    if norm0 is None or weight is None or expected is None:
        return TestStatus.FAIL, f"fixture missing: norm0={norm0}, weight={weight}, expected={expected}"

    a = np.ascontiguousarray(norm0.as_numpy(), dtype=np.float32)
    b = np.ascontiguousarray(weight.as_numpy(), dtype=np.float32)
    ref = np.ascontiguousarray(expected.as_numpy(), dtype=np.float32)
    # a shape: (seq=2, dim=2048); b shape: (2048,); out shape: (2, 2048)
    outer = a.shape[0]
    inner = a.shape[1]
    if b.shape != (inner,):
        return TestStatus.FAIL, f"weight shape {b.shape} != ({inner},)"
    out = np.zeros_like(a)
    lib.bpd_mul_broadcast_cpu(
        a.ctypes.data, b.ctypes.data, out.ctypes.data,
        ctypes.c_int(outer), ctypes.c_int(inner),
    )
    return assert_bit_identical(ref, out)


def test_lk_03_residual_add(lib, tensors):
    """L.1.3 RESIDUAL_ADD: a + b. Already-verified kernel.

    Inputs: post-attention output + the pre-norm residual at index ~63.
    Expected: post-residual sum.
    Strategy: find the first ADD op in the manifest.
    """
    if not hasattr(lib, 'bpd_residual_add_cpu'):
        return TestStatus.MISSING, "bpd_residual_add_cpu not in substrate"
    add_op = None
    for t in tensors:
        if t.op_desc == "ADD":
            add_op = t
            break
    if add_op is None:
        return TestStatus.SKIP, "no ADD op in fixture"
    # Need source tensors. ggml ADD has src[0] and src[1] but we didn't capture
    # named references. For now, mark as SKIP \u2014 we can implement this once we
    # also dump the src indices of each op.
    return TestStatus.SKIP, f"ADD at idx {add_op.idx} but src linking not yet in loader"


# ──────────────────────────── Runner ────────────────────────────

def setup_lib():
    lib = ctypes.CDLL(SO)
    # Register the kernels we test here
    if hasattr(lib, 'bpd_mul_broadcast_cpu'):
        lib.bpd_mul_broadcast_cpu.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]
        lib.bpd_mul_broadcast_cpu.restype = None
    if hasattr(lib, 'bpd_residual_add_cpu'):
        lib.bpd_residual_add_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]
        lib.bpd_residual_add_cpu.restype = None
    return lib


TESTS = [
    ("L.1.2 MUL (broadcast)",   test_lk_02_mul),
    ("L.1.3 RESIDUAL_ADD",      test_lk_03_residual_add),
]


def main():
    lib = setup_lib()
    print(f"Substrate library: {SO}")
    print(f"Fixture dir: {DUMP_DIR}")
    tensors = load_manifest(DUMP_DIR)
    print(f"Loaded {len(tensors)} tensors from fixture")
    print()
    print(f"{'Test':<32} {'Result':<60}")
    print("-" * 92)
    n_pass = n_fail = n_skip = n_missing = 0
    for name, fn in TESTS:
        try:
            status, msg = fn(lib, tensors)
        except Exception as e:
            import traceback
            status, msg = TestStatus.FAIL, f"exception: {e}\n{traceback.format_exc()}"
        first = msg.splitlines()[0] if msg else ""
        print(f"{name:<32} {status} {first}")
        for line in msg.splitlines()[1:]:
            print(f"{'':<32} {'':<10}{line}")
        if status == TestStatus.PASS:
            n_pass += 1
        elif status == TestStatus.MISSING:
            n_missing += 1
        elif status == TestStatus.SKIP:
            n_skip += 1
        else:
            n_fail += 1
    print()
    print(f"PASS: {n_pass}, FAIL: {n_fail}, SKIP: {n_skip}, MISSING: {n_missing}")
    sys.exit(0 if (n_fail == 0 and n_missing == 0) else 1)


if __name__ == "__main__":
    main()
