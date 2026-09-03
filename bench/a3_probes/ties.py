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

# Find EVERY block (token, block) whose per-block value my model gets wrong,
# by comparing against the kernel's own isolated K=32 value.  Then look at the
# tie elements in exactly those blocks.
print("   COLLECT: for every token, which blocks does my model get wrong?")
print("   (kernel's own K=32 per-block value is ground truth)")
rec = []
for tk in range(n):
    ab = np.ascontiguousarray(A[tk], dtype=np.float32).reshape(nblk, 32)
    amax = np.abs(ab).max(axis=1)
    dq = (amax / 127.0).astype(np.float32)
    dm = dq.astype(np.float16).astype(np.float32)
    r = ab / np.where(dq[:, None] > 0, dq[:, None], 1)
    qe = np.round(r).astype(np.int32).clip(-127, 127)
    bad = []
    for b in range(nblk):
        blk = np.ascontiguousarray(raw[0, b], dtype=np.uint8).copy()
        act = np.ascontiguousarray(ab[b], dtype=np.float32).copy()
        o1 = np.zeros(1, dtype=np.float32)
        L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), act.ctypes.data_as(FP),
                                         o1.ctypes.data_as(FP), I(1), I(1), I(32))
        mine = np.float32(dm[b] * ws[0, b] * np.float32(int((qe[b] * wq[0, b]).sum())))
        if mine != o1[0]:
            bad.append(b)
    # exact ties in this token
    frac = np.abs(r - np.trunc(r))
    ties = np.argwhere(frac == 0.5)
    rec.append((tk, bad, ties))
    print("     tok %d: wrong blocks=%s   exact-tie elements=%d %s"
          % (tk, bad if bad else "none", len(ties),
             [(int(i), int(j)) for i, j in ties[:4]]))

print()
print("   ★ do the WRONG blocks coincide with the TIE blocks?")
for tk, bad, ties in rec:
    tieblocks = sorted(set(int(i) for i, j in ties))
    print("     tok %d: wrong=%s  tie-blocks=%s  match=%s"
          % (tk, bad, tieblocks, "YES" if sorted(bad) == tieblocks else "NO"))
