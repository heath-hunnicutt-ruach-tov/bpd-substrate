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
nblk = E // 32

def model(a, W_ptr, out_rows, K):
    """The RECIPROCAL model: q = round(a * (127/amax)), scale by f16 dm * ws."""
    nb = K // 32
    buf = ctypes.cast(W_ptr, ctypes.POINTER(ctypes.c_uint8))
    rw = np.ctypeslib.as_array(buf, shape=(out_rows * nb * 34,)).reshape(out_rows, nb, 34)
    wsc = rw[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(out_rows, nb)
    wqi = rw[:, :, 2:34].view(np.int8).astype(np.int32)
    ab = a.reshape(nb, 32)
    amax = np.abs(ab).max(axis=1)
    dq = (amax / 127.0).astype(np.float32)
    dm = dq.astype(np.float16).astype(np.float32)
    inv = np.where(amax > 0, (np.float32(127.0) / amax).astype(np.float32), np.float32(0))
    q = np.round((ab * inv[:, None]).astype(np.float32)).astype(np.int32).clip(-127, 127)
    dot = np.einsum("bk,rbk->rb", q, wqi).astype(np.float32)
    contrib = (dot * (dm[None, :] * wsc)).astype(np.float32)
    acc = np.zeros(out_rows, dtype=np.float32)
    for b in range(nb):
        acc = (acc + contrib[:, b]).astype(np.float32)
    return acc

print("   GENERALISATION: the reciprocal model on Q, K and V projections.")
print("   (different shapes, different weights — same kernel)")
lw = w.layers[0]
targets = [("w_q", lw.w_q, cfg.n_heads * cfg.head_dim),
           ("w_k", lw.w_k, cfg.n_kv_heads * cfg.head_dim),
           ("w_v", lw.w_v, cfg.n_kv_heads * cfg.head_dim)]
grand_ok = grand_tot = 0
for name, ptr, orows in targets:
    ok = tot = 0
    for tk in range(n):
        one = np.ascontiguousarray(A[tk], dtype=np.float32).copy()
        o = np.zeros(orows, dtype=np.float32)
        L.bpd_qmatmul_q8_0_llamafile_cpu(ptr, one.ctypes.data_as(FP),
                                         o.ctypes.data_as(FP), I(orows), I(1), I(E))
        m = model(one, ptr, orows, E)
        ok += int((m == o).sum()); tot += orows
    grand_ok += ok; grand_tot += tot
    print("     %-5s (%4d rows): %5d/%5d exact   %s"
          % (name, orows, ok, tot, "PERFECT" if ok == tot else "MISMATCH"))
print()
print("   GRAND TOTAL: %d/%d" % (grand_ok, grand_tot))

# And the FFN projections, which have a different K (ffn_dim)
print()
print("   ALSO the FFN down-projection (K=%d, a DIFFERENT K):" % cfg.ffn_dim)
