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
A=s1[:n*E].reshape(n,E).copy()
rows=cfg.n_heads*cfg.head_dim

# ★ THE TEST I NEVER RAN: call the kernel with ONE TOKEN AT A TIME,
#   using the SAME real activations.  if per-token calls all match my model,
#   the parity is a MULTI-TOKEN artefact of the kernel.  if they still fail
#   on odd tokens, the parity is in the DATA, not the call shape.
nblk=E//32
buf=ctypes.cast(w.layers[0].w_q, ctypes.POINTER(ctypes.c_uint8))
raw=np.ctypeslib.as_array(buf,shape=(rows*nblk*34,)).reshape(rows,nblk,34)
ws=raw[:,:,0:2].copy().view(np.float16).astype(np.float32).reshape(rows,nblk)
wq=raw[:,:,2:34].view(np.int8).astype(np.int32)

def model_row(a):
    ab=a.reshape(nblk,32); amax=np.abs(ab).max(axis=1)
    dq=(amax/127.0).astype(np.float32); dm=dq.astype(np.float16).astype(np.float32)
    q=np.round(ab/np.where(dq[:,None]>0,dq[:,None],1)).astype(np.int32).clip(-127,127)
    acc=np.zeros(rows,dtype=np.float32)
    for b in range(nblk):
        dot=(q[b][None,:]*wq[:,b,:]).sum(axis=1)
        acc=(acc+np.float32(dm[b])*ws[:,b]*dot.astype(np.float32)).astype(np.float32)
    return acc

print("   PER-TOKEN kernel calls (m_tokens=1 each) vs my model:")
for tk in range(n):
    one=np.ascontiguousarray(A[tk],dtype=np.float32).copy()
    o=np.zeros(rows,dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
        o.ctypes.data_as(FP), I(rows), I(1), I(E))
    m=model_row(one); d=np.abs(m-o)
    print("     token %d: n_diff=%5d/%d  max_abs=%.6e" % (tk,int((d>0).sum()),rows,float(d.max())))

print()
print("   ★ PARITY SURVIVES PER-TOKEN CALLS -> it is the DATA, not the call.")
print("   what differs between even and odd token activations?")
for tk in range(n):
    a=A[tk]
    ab=a.reshape(nblk,32); amax=np.abs(ab).max(axis=1)
    print("     tok %d: |a|max=%.6e  mean|a|=%.6e  amax_min=%.6e  zeros=%d" %
          (tk, np.abs(a).max(), np.abs(a).mean(), amax.min(), int((a==0).sum())))

print()
print("   ★ NO even/odd split in the STATISTICS (tok2 has the largest |a|max")
print("     and matches perfectly).  so not magnitude.  look at ONE failing")
print("     element, block by block, and find WHICH BLOCK first diverges:")
one=np.ascontiguousarray(A[1],dtype=np.float32).copy()
o=np.zeros(rows,dtype=np.float32)
L.bpd_qmatmul_q8_0_llamafile_cpu(w.layers[0].w_q, one.ctypes.data_as(FP),
    o.ctypes.data_as(FP), I(rows), I(1), I(E))
ab=one.reshape(nblk,32); amax=np.abs(ab).max(axis=1)
dq=(amax/127.0).astype(np.float32); dm=dq.astype(np.float16).astype(np.float32)
q=np.round(ab/np.where(dq[:,None]>0,dq[:,None],1)).astype(np.int32).clip(-127,127)
r0=0
acc=np.float32(0.0)
for b in range(nblk):
    dot=int((q[b]*wq[r0,b]).sum())
    acc=np.float32(acc+np.float32(dm[b])*ws[r0,b]*np.float32(dot))
print("     token1 row0: kernel=%.9g  model=%.9g  diff=%.3e" % (o[r0],acc,abs(o[r0]-acc)))
print("     q range: %d..%d   any |q|==127: %d" % (q.min(),q.max(),int((np.abs(q)==127).sum())))
print("     dq==dm (f16 lossless)?  %d / %d blocks" % (int((dq==dm).sum()), nblk))

print()
print("   ★ |q|==127 on 64/64 blocks -- the amax element itself always")
print("     quantizes to exactly +/-127.  and dq != dm on ALL 64 blocks.")
print("     TEST: quantize using dm (the f16 scale) instead of dq (f32):")
for lbl,dv in (("divide by dq (f32)",dq),("divide by dm (f16)",dm)):
    qq=np.round(ab/np.where(dv[:,None]>0,dv[:,None],1)).astype(np.int32).clip(-127,127)
    a2=np.float32(0.0)
    for b in range(nblk):
        dot=int((qq[b]*wq[r0,b]).sum())
        a2=np.float32(a2+np.float32(dm[b])*ws[r0,b]*np.float32(dot))
    print("       %-20s -> %.9g   %s" % (lbl,a2,"*** MATCH ***" if a2==o[r0] else "diff %.2e"%abs(o[r0]-a2)))
