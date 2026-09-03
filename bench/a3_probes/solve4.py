import sys, ctypes, numpy as np
sys.path.insert(0, "bench")
from bpd_llamatov_infer import (BpdLlamaConfig, BpdLlamaLayerWeights, build_model,
                                c_float_p, c_uint8_p)
from llama_fixture_loader import load_manifest, find_op

L = ctypes.CDLL("/home/dibbur-patch/step3-det-gemv/bpd/build/bpd_cpu.so")
FP, UP = c_float_p, c_uint8_p
I, Fl = ctypes.c_int, ctypes.c_float
for f, a in [("bpd_rmsnorm_llama_cpu", [FP, FP, FP, I, I, Fl]),
             ("bpd_qmatmul_q8_0_llamafile_cpu", [UP, FP, FP, I, I, I])]:
    g = getattr(L, f); g.restype = None; g.argtypes = a

cfg, w, ldr = build_model("/mnt/data/shared/models/tinyllama-q8_0.gguf")
t = load_manifest("fixtures/llama_dump_tinyllama_hello")
embd = find_op(t, name_substring="embd", op_desc="GET_ROWS").as_numpy()
n, E = embd.shape
x = np.ascontiguousarray(embd, dtype=np.float32).reshape(-1).copy()
s1 = np.zeros(n * max(E, cfg.ffn_dim), dtype=np.float32)
L.bpd_rmsnorm_llama_cpu(x.ctypes.data_as(FP), w.layers[0].attn_norm_w,
                        s1.ctypes.data_as(FP), I(n), I(E), Fl(cfg.rms_eps))
A = s1[:n * E].reshape(n, E).copy()
rows = cfg.n_heads * cfg.head_dim
nblk = E // 32
buf = ctypes.cast(w.layers[0].w_q, ctypes.POINTER(ctypes.c_uint8))
raw = np.ctypeslib.as_array(buf, shape=(rows * nblk * 34,)).reshape(rows, nblk, 34)
ws = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, nblk)
wq = raw[:, :, 2:34].view(np.int8).astype(np.int32)

tk, r0 = 1, 0
one = np.ascontiguousarray(A[tk], dtype=np.float32).copy()
ab = one.reshape(nblk, 32)

# THE KERNEL'S OWN PER-BLOCK VALUES, obtained by calling it with K=32 per block.
# These are ground truth: each matched my model exactly in isolation.
kb = np.zeros(nblk, dtype=np.float32)
for b in range(nblk):
    blk = np.ascontiguousarray(raw[r0, b], dtype=np.uint8).copy()
    act = np.ascontiguousarray(ab[b], dtype=np.float32).copy()
    o1 = np.zeros(1, dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), act.ctypes.data_as(FP),
                                     o1.ctypes.data_as(FP), I(1), I(1), I(32))
    kb[b] = o1[0]

# THE FULL CALL
full = np.zeros(rows, dtype=np.float32)
L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
                                 full.ctypes.data_as(FP), I(rows), I(1), I(E))
target = full[r0]

print("   All 64 per-block values are the kernel's OWN (K=32 calls, each exact).")
print("   target (full K=2048 call) = %.9g" % target)
print()

def seq(v):
    a = np.float32(0.0)
    for c in v:
        a = np.float32(a + c)
    return a

cands = {
    "sequential f32 forward": seq(kb),
    "sequential f32 reverse": seq(kb[::-1]),
    "f64 sum then round": np.float32(kb.astype(np.float64).sum()),
    "numpy f32 sum (pairwise)": np.float32(kb.sum(dtype=np.float32)),
}
# strided orders: 4-way and 8-way interleave, as a vectorised kernel would
for w_ in (2, 4, 8):
    part = [np.float32(0.0)] * w_
    for i, c in enumerate(kb):
        part[i % w_] = np.float32(part[i % w_] + c)
    cands["%d-way interleaved f32" % w_] = seq(np.array(part, dtype=np.float32))

for lbl, v in cands.items():
    tag = "*** MATCH ***" if v == target else "diff %.3e" % abs(target - v)
    print("     %-26s -> %.9g  %s" % (lbl, v, tag))
