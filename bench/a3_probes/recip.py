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

# HYPOTHESIS: the kernel computes a RECIPROCAL id = 127/amax (or 1/dq) once per
# block and MULTIPLIES, rather than dividing by dq.  a*(127/amax) and a/(amax/127)
# differ by an ULP near rounding boundaries -- exactly the observed off-by-one.
def model(a, mode):
    ab = a.reshape(nblk, 32)
    amax = np.abs(ab).max(axis=1)
    dq = (amax / 127.0).astype(np.float32)
    dm = dq.astype(np.float16).astype(np.float32)
    if mode == "divide":
        r = ab / np.where(dq[:, None] > 0, dq[:, None], 1)
    elif mode == "recip_dq":
        inv = np.where(dq > 0, (np.float32(1.0) / dq).astype(np.float32), np.float32(0))
        r = (ab * inv[:, None]).astype(np.float32)
    elif mode == "recip_127_amax":
        inv = np.where(amax > 0, (np.float32(127.0) / amax).astype(np.float32), np.float32(0))
        r = (ab * inv[:, None]).astype(np.float32)
    q = np.round(r).astype(np.int32).clip(-127, 127)
    dot = np.einsum("bk,rbk->rb", q, wq).astype(np.float32)
    contrib = (dot * (dm[None, :] * ws)).astype(np.float32)
    acc = np.zeros(rows, dtype=np.float32)
    for b in range(nblk):
        acc = (acc + contrib[:, b]).astype(np.float32)
    return acc

modes = ["divide", "recip_dq", "recip_127_amax"]
print("   RECIPROCAL-MULTIPLY vs DIVIDE for the activation quantisation.")
print("   (a*(127/amax) and a/(amax/127) differ by an ULP near boundaries)")
tot = {m: 0 for m in modes}
for tk in range(n):
    one = np.ascontiguousarray(A[tk], dtype=np.float32).copy()
    o = np.zeros(rows, dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
                                     o.ctypes.data_as(FP), I(rows), I(1), I(E))
    outs = {m: model(one, m) for m in modes}
    line = []
    for m in modes:
        c = int((outs[m] == o).sum()); tot[m] += c
        line.append("%s %4d" % (m, c))
    diff = int((outs["divide"] != outs["recip_127_amax"]).sum())
    print("     tok %d: %s   (div vs recip differ on %d)" % (tk, "  ".join(line), diff))
print()
for m in modes:
    print("   TOTAL %-16s %5d/%d" % (m, tot[m], n * rows))
