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
buf = ctypes.cast(w.layers[0].w_q, ctypes.POINTER(ctypes.c_uint8))
raw = np.ctypeslib.as_array(buf, shape=((cfg.n_heads * cfg.head_dim) * nblk * 34,)).reshape(-1, nblk, 34)
ws = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1, nblk)
wq = raw[:, :, 2:34].view(np.int8).astype(np.int32)

# The four known-wrong (token, block) pairs.  Examine the quantised vector
# against what the kernel's dot must have been.
cases = [(1, 60), (3, 43), (5, 4)]
print("   THE WRONG BLOCKS: solve for the kernel's integer dot, compare to mine.")
for tk, b in cases:
    ab = np.ascontiguousarray(A[tk], dtype=np.float32).reshape(nblk, 32)
    act = np.ascontiguousarray(ab[b], dtype=np.float32).copy()
    blk = np.ascontiguousarray(raw[0, b], dtype=np.uint8).copy()
    o1 = np.zeros(1, dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), act.ctypes.data_as(FP),
                                     o1.ctypes.data_as(FP), I(1), I(1), I(32))
    amax = np.float32(np.abs(act).max())
    dq = np.float32(amax / 127.0)
    dm = np.float32(np.float16(dq))
    r = act / dq
    qe = np.round(r).astype(np.int32).clip(-127, 127)
    mydot = int((qe * wq[0, b]).sum())
    # solve: kernel = dm * ws * dot  =>  dot = kernel/(dm*ws)
    kdot = float(o1[0]) / float(dm * ws[0, b])
    print("     tok %d blk %2d: my dot=%6d   kernel implied dot=%10.4f   delta=%8.4f"
          % (tk, b, mydot, kdot, kdot - mydot))
    # which single element would account for the delta?
    d = int(round(kdot)) - mydot
    hits = [i for i in range(32) if wq[0, b][i] != 0 and d % int(wq[0, b][i]) == 0]
    print("        integer delta = %d; elements whose weight divides it: %d" % (d, len(hits)))
    # show the largest |r| fractional parts (nearest to a rounding boundary)
    frac = np.abs(r - np.trunc(r))
    near = np.argsort(-np.abs(frac - 0.5))[-3:]
    for i in near:
        print("        i=%2d r=%.9f qe=%4d  w=%4d" % (i, r[i], qe[i], wq[0, b][i]))
