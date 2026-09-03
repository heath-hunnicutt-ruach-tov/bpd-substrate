import sys, ctypes, numpy as np
sys.path.insert(0,"bench")
from bpd_llamatov_infer import (BpdLlamaConfig, BpdLlamaLayerWeights, build_model,
                                c_float_p, c_uint8_p)
from llama_fixture_loader import load_manifest, find_op
L=ctypes.CDLL("/home/dibbur-patch/step3-det-gemv/bpd/build/bpd_cpu.so")
FP,UP=c_float_p,c_uint8_p; I,Fl=ctypes.c_int,ctypes.c_float
for f,a in [("bpd_rmsnorm_llama_cpu",[FP,FP,FP,I,I,Fl]),
            ("bpd_qmatmul_q8_0_llamafile_cpu",[UP,FP,FP,I,I,I])]:
    g=getattr(L,f); g.restype=None; g.argtypes=a
cfg,w,ldr=build_model("/mnt/data/shared/models/tinyllama-q8_0.gguf")
t=load_manifest("fixtures/llama_dump_tinyllama_hello")
embd=find_op(t,name_substring="embd",op_desc="GET_ROWS").as_numpy()
n,E=embd.shape
x=np.ascontiguousarray(embd,dtype=np.float32).reshape(-1).copy()
s1=np.zeros(n*max(E,cfg.ffn_dim),dtype=np.float32)
L.bpd_rmsnorm_llama_cpu(x.ctypes.data_as(FP), w.layers[0].attn_norm_w,
    s1.ctypes.data_as(FP), I(n), I(E), Fl(cfg.rms_eps))
A=s1[:n*E].reshape(n,E).copy(); rows=cfg.n_heads*cfg.head_dim; nblk=E//32
buf=ctypes.cast(w.layers[0].w_q, ctypes.POINTER(ctypes.c_uint8))
raw=np.ctypeslib.as_array(buf,shape=(rows*nblk*34,)).reshape(rows,nblk,34)
ws=raw[:,:,0:2].copy().view(np.float16).astype(np.float32).reshape(rows,nblk)
wq=raw[:,:,2:34].view(np.int8).astype(np.int32)

# VERIFY f32-SEQUENTIAL ON A CLEAN TOKEN (token 0 matches).  does the
# accumulation choice actually MATTER there, or is it also unexercised?
tk=0
one=np.ascontiguousarray(A[tk],dtype=np.float32).copy()
o=np.zeros(rows,dtype=np.float32)
L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
    o.ctypes.data_as(FP), I(rows), I(1), I(E))
ab=one.reshape(nblk,32); amax=np.abs(ab).max(axis=1)
dq=(amax/127.0).astype(np.float32); dm=dq.astype(np.float16).astype(np.float32)
q=np.round(ab/np.where(dq[:,None]>0,dq[:,None],1)).astype(np.int32).clip(-127,127)
dot=np.einsum("bk,rbk->rb",q,wq).astype(np.float32)
contrib=(dot*(dm[None,:]*ws)).astype(np.float32)
seq=np.zeros(rows,dtype=np.float32)
for b in range(nblk): seq=(seq+contrib[:,b]).astype(np.float32)
f64=contrib.astype(np.float64).sum(axis=1).astype(np.float32)
print("   ★ IS THE ACCUMULATION CHOICE EXERCISED ON REAL DATA?")
print("     token 0 (a CLEAN token):")
print("       f32-sequential vs kernel: n_diff=%d/%d" % (int((seq!=o).sum()),rows))
print("       f64-sum        vs kernel: n_diff=%d/%d" % (int((f64!=o).sum()),rows))
print("       do the two DIFFER from each other? n_diff=%d/%d" % (int((seq!=f64).sum()),rows))

print()
print("   ★ DRAW FROM THE FAILING VECTORS: for each token, does the")
print("     f32-vs-f64 accumulation choice get EXERCISED, and does")
print("     f32-sequential match?  (the alternatives-differ check, per token)")
for tk in range(n):
    one=np.ascontiguousarray(A[tk],dtype=np.float32).copy()
    o=np.zeros(rows,dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
        o.ctypes.data_as(FP), I(rows), I(1), I(E))
    ab=one.reshape(nblk,32); amx=np.abs(ab).max(axis=1)
    dqt=(amx/127.0).astype(np.float32); dmt=dqt.astype(np.float16).astype(np.float32)
    qt=np.round(ab/np.where(dqt[:,None]>0,dqt[:,None],1)).astype(np.int32).clip(-127,127)
    dt=np.einsum("bk,rbk->rb",qt,wq).astype(np.float32)
    ct=(dt*(dmt[None,:]*ws)).astype(np.float32)
    sq=np.zeros(rows,dtype=np.float32)
    for b in range(nblk): sq=(sq+ct[:,b]).astype(np.float32)
    f6=ct.astype(np.float64).sum(axis=1).astype(np.float32)
    print("     tok %d: f32seq-match=%4d/%d  alts-differ=%4d  dq==dm blocks=%2d/%d" %
          (tk, int((sq==o).sum()), rows, int((sq!=f6).sum()), int((dqt==dmt).sum()), nblk))

print()
print("   dq==dm is 0/64 for ALL tokens -> not the discriminator.")
print("   compare the QUANTIZED activation vectors, even vs odd:")
for tk in range(n):
    one=np.ascontiguousarray(A[tk],dtype=np.float32).copy()
    ab=one.reshape(nblk,32); amx=np.abs(ab).max(axis=1)
    dqt=(amx/127.0).astype(np.float32)
    r=ab/np.where(dqt[:,None]>0,dqt[:,None],1)
    qt=np.round(r).astype(np.int32)
    # how many ratios land within a hair of a .5 tie?  those are where
    # my np.round and a C rintf/round could disagree.
    frac=np.abs(r-np.trunc(r))
    near=int(((np.abs(frac-0.5)<1e-6)).sum())
    exact=int((frac==0.5).sum())
    print("     tok %d: EXACT .5 ties=%3d  near-ties(<1e-6)=%3d  |q|>127 pre-clip=%d" %
          (tk, exact, near, int((np.abs(qt)>127).sum())))
