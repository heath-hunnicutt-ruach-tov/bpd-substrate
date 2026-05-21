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


def test_lk_01_embed_lookup(lib, tensors):
    """L.1.1 EMBED LOOKUP: gather rows from a Q8_0 embedding table and dequantize.

    Flow:
      1. Prolog gguf reader returns offset+size of token_embd.weight.
      2. Python reads just the rows we need (rows 128000 and 13347) from the GGUF.
      3. Our C kernel bpd_embed_lookup_q8_0_cpu gathers + dequantizes.
      4. Compare against the captured fixture 0000_inp_embd.bin (GET_ROWS output).
      5. Assert 0 ULP per element.

    The token IDs come from leaf_2 in the fixture: [128000, 13347] (BOS + 'Hi').
    """
    if not hasattr(lib, 'bpd_embed_lookup_q8_0_cpu'):
        return TestStatus.MISSING, "bpd_embed_lookup_q8_0_cpu not in substrate"

    try:
        from bench.gguf_helper import query_tensor
    except ImportError:
        from gguf_helper import query_tensor

    gguf_path = os.environ.get(
        "LLAMA_GGUF",
        "/mnt/data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    )
    if not os.path.exists(gguf_path):
        return TestStatus.SKIP, f"GGUF not at {gguf_path}"

    # Locate the embedding table in the GGUF
    try:
        info = query_tensor(gguf_path, "token_embd.weight")
    except Exception as e:
        return TestStatus.FAIL, f"gguf_query failed: {e}"
    if info.ggml_type != 8:
        return TestStatus.FAIL, f"expected ggml_type=8 (Q8_0), got {info.ggml_type}"

    # Locate the token IDs and expected output in the captured fixture
    leaf2 = find_op(tensors, name_substring="leaf_2", op_desc="NONE")
    expected = find_op(tensors, name_substring="inp_embd", op_desc="GET_ROWS")
    if leaf2 is None or expected is None:
        return TestStatus.FAIL, f"fixture missing: leaf_2={leaf2}, inp_embd={expected}"

    token_ids = np.ascontiguousarray(leaf2.as_numpy(), dtype=np.int32)  # shape (n_tokens,)
    n_tokens = len(token_ids)
    # The expected shape is (embed_dim, n_tokens) in ggml's storage, but numpy
    # reads it as (n_tokens, embed_dim) after shape_no_trailing_ones reverses.
    ref = np.ascontiguousarray(expected.as_numpy(), dtype=np.float32)
    if ref.shape != (n_tokens, info.dims[0]):
        return TestStatus.FAIL, f"expected shape {(n_tokens, info.dims[0])}, got {ref.shape}"
    embed_dim = info.dims[0]
    vocab_size = info.dims[1]
    bytes_per_row = (embed_dim // 32) * 34

    # Read only the rows we need (avoid loading 279 MB).
    # We build a synthetic packed table containing just the rows for our token_ids,
    # and we pass token_ids [0, 1, ...] so the kernel indexes into our packed table.
    packed_table = np.zeros(n_tokens * bytes_per_row, dtype=np.uint8)
    for t in range(n_tokens):
        tok = int(token_ids[t])
        row_abs_offset = info.abs_offset + tok * bytes_per_row
        row_bytes = np.fromfile(
            gguf_path, dtype=np.uint8, count=bytes_per_row, offset=row_abs_offset,
        )
        packed_table[t * bytes_per_row : (t + 1) * bytes_per_row] = row_bytes
    synthetic_ids = np.ascontiguousarray(np.arange(n_tokens, dtype=np.int32))

    out = np.zeros((n_tokens, embed_dim), dtype=np.float32)
    lib.bpd_embed_lookup_q8_0_cpu(
        packed_table.ctypes.data,
        synthetic_ids.ctypes.data,
        out.ctypes.data,
        ctypes.c_int(n_tokens),
        ctypes.c_int(embed_dim),
    )

    return assert_bit_identical(ref, out)


def test_lk_09_q8_0_dequant(lib, tensors):
    """L.1.9 Q8_0 DEQUANT: per-element dequantization.

    Flow:
      1. Prolog gguf reader finds a Q8_0 tensor in the model (blk.0.attn_k.weight).
      2. Raw bytes flow into Python via np.fromfile(offset=...).
      3. Python scalar reference dequantizes (trusted oracle).
      4. Our C kernel dequantizes the same bytes.
      5. Assert per-element 0 ULP.

    Q8_0 layout per 32-element block (34 bytes):
      bytes [0..1]:  uint16 little-endian F16 scale
      bytes [2..33]: int8[32] quantized values

    Per-element: out[i] = (float)int8[i] * f16_to_f32(scale)
    """
    if not hasattr(lib, 'bpd_dequant_q8_0_cpu'):
        return TestStatus.MISSING, "bpd_dequant_q8_0_cpu not in substrate"

    # Import gguf_helper here so the test runner doesn't need it at import time
    try:
        from bench.gguf_helper import query_tensor, read_tensor_bytes
    except ImportError:
        from gguf_helper import query_tensor, read_tensor_bytes

    gguf_path = os.environ.get(
        "LLAMA_GGUF",
        "/mnt/data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    )
    if not os.path.exists(gguf_path):
        return TestStatus.SKIP, f"GGUF not at {gguf_path}"

    try:
        info = query_tensor(gguf_path, "blk.0.attn_k.weight")
    except Exception as e:
        return TestStatus.FAIL, f"gguf_query failed: {e}"
    if info.ggml_type != 8:
        return TestStatus.FAIL, f"expected ggml_type=8 (Q8_0), got {info.ggml_type}"

    # Read raw bytes from the GGUF
    raw = read_tensor_bytes(gguf_path, info)
    n_blocks = len(raw) // 34
    if n_blocks * 34 != len(raw):
        return TestStatus.FAIL, f"tensor size {len(raw)} not a multiple of 34"

    # Python scalar reference (trusted oracle).
    # F16 -> F32 via numpy's native conversion (IEEE 754 conforming).
    raw_u8 = np.asarray(raw, dtype=np.uint8)
    blocks_u8 = raw_u8.reshape(n_blocks, 34)
    scales_u16 = blocks_u8[:, :2].view(np.uint16).reshape(n_blocks)
    scales_f16 = scales_u16.view(np.float16)
    scales_f32 = scales_f16.astype(np.float32)  # F16 -> F32 conversion
    quants_i8 = blocks_u8[:, 2:].view(np.int8)  # shape (n_blocks, 32)
    # Per-element: cast int8 -> float32, multiply by per-block scale.
    quants_f32 = quants_i8.astype(np.float32)   # shape (n_blocks, 32)
    ref = (quants_f32 * scales_f32[:, np.newaxis]).reshape(-1)

    # Our C kernel
    out = np.zeros(n_blocks * 32, dtype=np.float32)
    raw_contig = np.ascontiguousarray(raw, dtype=np.uint8)
    lib.bpd_dequant_q8_0_cpu(
        raw_contig.ctypes.data, out.ctypes.data, ctypes.c_int(n_blocks),
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
    if hasattr(lib, 'bpd_dequant_q8_0_cpu'):
        lib.bpd_dequant_q8_0_cpu.argtypes = [
            ctypes.c_void_p,  # raw uint8*
            ctypes.c_void_p,  # out float*
            ctypes.c_int,     # n_blocks
        ]
        lib.bpd_dequant_q8_0_cpu.restype = None
    if hasattr(lib, 'bpd_embed_lookup_q8_0_cpu'):
        lib.bpd_embed_lookup_q8_0_cpu.argtypes = [
            ctypes.c_void_p,  # table uint8*
            ctypes.c_void_p,  # token_ids int32*
            ctypes.c_void_p,  # out float*
            ctypes.c_int,     # n_tokens
            ctypes.c_int,     # embed_dim
        ]
        lib.bpd_embed_lookup_q8_0_cpu.restype = None
    return lib


TESTS = [
    ("L.1.1 EMBED LOOKUP",      test_lk_01_embed_lookup),
    ("L.1.2 MUL (broadcast)",   test_lk_02_mul),
    ("L.1.3 RESIDUAL_ADD",      test_lk_03_residual_add),
    ("L.1.9 Q8_0 DEQUANT",      test_lk_09_q8_0_dequant),
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
