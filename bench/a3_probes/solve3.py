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

# ISOLATE ONE BLOCK: call the kernel with K=32 on the FAILING token's first
# block only.  With one block there is no cross-block accumulation, so any
# difference is purely the per-block arithmetic.
tk = 1
one = np.ascontiguousarray(A[tk], dtype=np.float32).copy()
ab = one.reshape(nblk, 32)

print("   PER-BLOCK ISOLATION on failing token 1 (K=32, one block at a time).")
print("   Any mismatch here is per-block arithmetic, with no cross-block sum.")
mism = 0
first_bad = None
for b in range(min(nblk, 12)):
    blk = np.ascontiguousarray(raw[0, b], dtype=np.uint8).copy()   # row 0, block b
    act = np.ascontiguousarray(ab[b], dtype=np.float32).copy()
    o1 = np.zeros(1, dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), act.ctypes.data_as(FP),
                                     o1.ctypes.data_as(FP), I(1), I(1), I(32))
    amx = np.float32(np.abs(act).max())
    d = np.float32(amx / 127.0)
    dm = np.float32(np.float16(d))
    q = np.round(act / d).astype(np.int32).clip(-127, 127)
    dot = int((q * wq[0, b]).sum())
    mine = np.float32(dm * ws[0, b] * np.float32(dot))
    ok = (mine == o1[0])
    if not ok:
        mism += 1
        if first_bad is None:
            first_bad = b
    print("     block %2d: kernel=%.9g mine=%.9g %s" % (b, o1[0], mine, "" if ok else "MISMATCH"))
print()
print("   mismatching blocks in the first 12: %d   first: %s" % (mism, first_bad))
