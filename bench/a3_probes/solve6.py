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

tk, r0, b = 1, 0, 60
ab = np.ascontiguousarray(A[tk], dtype=np.float32).reshape(nblk, 32)
act = np.ascontiguousarray(ab[b], dtype=np.float32).copy()

# the kernel's own value for this block alone
blk = np.ascontiguousarray(raw[r0, b], dtype=np.uint8).copy()
o1 = np.zeros(1, dtype=np.float32)
L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), act.ctypes.data_as(FP),
                                 o1.ctypes.data_as(FP), I(1), I(1), I(32))

amax = np.float32(np.abs(act).max())
dq = np.float32(amax / 127.0)
dm = np.float32(np.float16(dq))
r = act / dq
q_even = np.round(r).astype(np.int32)
q_away = np.trunc(r + np.copysign(0.5, r)).astype(np.int32)

print("   BLOCK 60 of token 1, row 0 — the ONE differing block of 64.")
print("     kernel (K=32 isolated call) = %.9g" % o1[0])
print("     amax = %.9g   dq = %.9g   dm = %.9g" % (amax, dq, dm))
print()
print("     ratios nearest a .5 tie:")
frac = np.abs(r - np.trunc(r))
idx = np.argsort(-np.abs(frac - 0.5) * -1)[:4]
for i in idx:
    print("       i=%2d  a=%.9g  r=%.9f  half-even=%d  half-away=%d"
          % (i, act[i], r[i], q_even[i], q_away[i]))
print()
print("     do the two rounding rules give DIFFERENT q vectors? %s"
      % ("YES" if not np.array_equal(q_even, q_away) else "no"))
for lbl, qq in (("half-even", q_even), ("half-away", q_away)):
    v = np.float32(dm * ws[r0, b] * np.float32(int((qq.clip(-127, 127) * wq[r0, b]).sum())))
    print("       %-10s dot=%6d -> %.9g  %s"
          % (lbl, int((qq.clip(-127, 127) * wq[r0, b]).sum()), v,
             "*** MATCH ***" if v == o1[0] else ""))
