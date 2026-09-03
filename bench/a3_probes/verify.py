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

half_even = lambda r: np.round(r)
half_away = lambda r: np.trunc(r + np.copysign(0.5, r))

def model(a, roundfn):
    ab = a.reshape(nblk, 32)
    amax = np.abs(ab).max(axis=1)
    dq = (amax / 127.0).astype(np.float32)
    dm = dq.astype(np.float16).astype(np.float32)
    q = roundfn(ab / np.where(dq[:, None] > 0, dq[:, None], 1)).astype(np.int32).clip(-127, 127)
    dot = np.einsum("bk,rbk->rb", q, wq).astype(np.float32)
    contrib = (dot * (dm[None, :] * ws)).astype(np.float32)
    acc = np.zeros(rows, dtype=np.float32)
    for b in range(nblk):
        acc = (acc + contrib[:, b]).astype(np.float32)
    return acc

print("   VERIFY ACROSS ALL SIX TOKENS — half-away vs half-even, real data.")
print("   (the alternatives-differ count is the third column)")
tot_e = tot_a = tot_d = 0
for tk in range(n):
    one = np.ascontiguousarray(A[tk], dtype=np.float32).copy()
    o = np.zeros(rows, dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
                                     o.ctypes.data_as(FP), I(rows), I(1), I(E))
    me = model(one, half_even)
    ma = model(one, half_away)
    ne = int((me == o).sum()); na = int((ma == o).sum()); nd = int((me != ma).sum())
    tot_e += ne; tot_a += na; tot_d += nd
    print("     tok %d: half-even matches %4d/%d   half-away matches %4d/%d   alts differ %4d"
          % (tk, ne, rows, na, rows, nd))
print()
print("   TOTAL: half-even %d/%d   half-away %d/%d   alternatives differ on %d"
      % (tot_e, n * rows, tot_a, n * rows, tot_d))
