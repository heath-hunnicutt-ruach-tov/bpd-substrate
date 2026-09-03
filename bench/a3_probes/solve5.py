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

# ★ THE KEY QUESTION: the per-block K=32 calls each matched my model, and
# sequentially summing THOSE reproduces the full call exactly.  So where do
# MY per-block values differ from the kernel's?  Compare directly.
kb = np.zeros(nblk, dtype=np.float32)
for b in range(nblk):
    blk = np.ascontiguousarray(raw[r0, b], dtype=np.uint8).copy()
    act = np.ascontiguousarray(ab[b], dtype=np.float32).copy()
    o1 = np.zeros(1, dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), act.ctypes.data_as(FP),
                                     o1.ctypes.data_as(FP), I(1), I(1), I(32))
    kb[b] = o1[0]

# my per-block values, computed the way my model does when it sees the WHOLE row
amx_row = np.abs(ab).max(axis=1)
dq = (amx_row / 127.0).astype(np.float32)
dm = dq.astype(np.float16).astype(np.float32)
q = np.round(ab / np.where(dq[:, None] > 0, dq[:, None], 1)).astype(np.int32).clip(-127, 127)
mb = np.array([np.float32(dm[b] * ws[r0, b] * np.float32(int((q[b] * wq[r0, b]).sum())))
               for b in range(nblk)], dtype=np.float32)

diff = np.abs(mb - kb)
print("   MY per-block values vs THE KERNEL'S OWN per-block values:")
print("     blocks differing: %d / %d" % (int((diff > 0).sum()), nblk))
print("     max block diff  : %.6e" % float(diff.max()))
bad = np.argwhere(diff > 0).ravel()
print("     first few differing blocks:", bad[:6].tolist())
for b in bad[:3]:
    print("       block %2d: kernel=%.9g mine=%.9g" % (b, kb[b], mb[b]))
    print("                 amax=%.9g  dq=%.9g  dm=%.9g" % (amx_row[b], dq[b], dm[b]))
