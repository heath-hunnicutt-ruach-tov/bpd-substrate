import ctypes, numpy as np
L=ctypes.CDLL("/home/dibbur-patch/step3-det-gemv/bpd/build/bpd_cpu.so")
FP=ctypes.POINTER(ctypes.c_float); UP=ctypes.POINTER(ctypes.c_uint8); I=ctypes.c_int
L.bpd_qmatmul_q8_0_llamafile_cpu.restype=None
L.bpd_qmatmul_q8_0_llamafile_cpu.argtypes=[UP,FP,FP,I,I,I]
print("   THE ONE-LINE TEST: vary m_tokens, the parameter I never varied.")
print("   (all 160/160 synthetic trials used m_tokens=1)")
for ntok in (1,2,3,4,6):
    nblk=4; K=32*nblk; M=8
    r=np.random.default_rng(31)
    qs=r.integers(-127,128,size=(M,nblk,32)).astype(np.int8)
    sc=np.array([[np.float16(0.01+0.001*b) for b in range(nblk)] for _ in range(M)],dtype=np.float16)
    x=r.standard_normal(ntok*K).astype(np.float32)
    blk=np.zeros(M*nblk*34,dtype=np.uint8)
    for m in range(M):
        for b in range(nblk):
            o0=(m*nblk+b)*34
            blk[o0:o0+2]=np.frombuffer(sc[m,b].tobytes(),dtype=np.uint8)
            blk[o0+2:o0+34]=qs[m,b].view(np.uint8)
    out=np.zeros(ntok*M,dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP),x.ctypes.data_as(FP),
                                     out.ctypes.data_as(FP),I(M),I(ntok),I(K))
    out=out.reshape(ntok,M)
    xb=x.reshape(ntok,nblk,32); amax=np.abs(xb).max(axis=2)
    dq=(amax/127.0).astype(np.float32); dm=dq.astype(np.float16).astype(np.float32)
    q=np.round(xb/np.where(dq[:,:,None]>0,dq[:,:,None],1)).astype(np.int32).clip(-127,127)
    bad=[]
    for tk in range(ntok):
        acc=np.zeros(M,dtype=np.float32)
        for b in range(nblk):
            dot=(q[tk,b][None,:]*qs[:,b,:].astype(np.int32)).sum(axis=1)
            acc=(acc+np.float32(dm[tk,b])*sc[:,b].astype(np.float32)*dot.astype(np.float32)).astype(np.float32)
        bad.append(int((acc!=out[tk]).sum()))
    print("     m_tokens=%d  mismatches per token: %s" % (ntok,bad))
