import ctypes, numpy as np
L=ctypes.CDLL("/home/dibbur-patch/step3-det-gemv/bpd/build/bpd_cpu.so")
FP=ctypes.POINTER(ctypes.c_float); UP=ctypes.POINTER(ctypes.c_uint8); I=ctypes.c_int
L.bpd_qmatmul_q8_0_llamafile_cpu.restype=None
L.bpd_qmatmul_q8_0_llamafile_cpu.argtypes=[UP,FP,FP,I,I,I]

def trial(nblk,seed,combine):
    r=np.random.default_rng(seed)
    qs=r.integers(-127,128,size=(nblk,32)).astype(np.int8)
    sc=np.array([np.float16(0.01+0.001*i) for i in range(nblk)],dtype=np.float16)
    x=r.standard_normal(32*nblk).astype(np.float32)
    blk=np.zeros(nblk*34,dtype=np.uint8)
    for b in range(nblk):
        blk[b*34:b*34+2]=np.frombuffer(sc[b].tobytes(),dtype=np.uint8)
        blk[b*34+2:(b+1)*34]=qs[b].view(np.uint8)
    o=np.zeros(1,dtype=np.float32)
    L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP),x.ctypes.data_as(FP),
                                     o.ctypes.data_as(FP),I(1),I(1),I(32*nblk))
    xb=x.reshape(nblk,32); amax=np.abs(xb).max(axis=1)
    dq=(amax/127.0).astype(np.float32)
    dm=dq.astype(np.float16).astype(np.float32)
    q=np.round(xb/np.where(dq[:,None]>0,dq[:,None],1)).astype(np.int32).clip(-127,127)
    dot=(q*qs.astype(np.int32)).sum(axis=1)
    contrib=(dm*sc.astype(np.float32)*dot.astype(np.float32)).astype(np.float32)
    if combine=="f64":   v=np.float32(contrib.astype(np.float64).sum())
    elif combine=="f32seq":
        a=np.float32(0.0)
        for c in contrib: a=np.float32(a+c)
        v=a
    elif combine=="pairwise":
        c=contrib.copy(); n=len(c)
        while n>1:
            h=(n+1)//2
            for i in range(n-h): c[i]=np.float32(c[i]+c[i+h])
            n=h
        v=c[0]
    return o[0]==v

print("   MULTI-BLOCK COMBINE: which accumulation matches?  40 seeds each.")
for combine in ("f64","f32seq","pairwise"):
    row=[]
    for nblk in (2,4,8,64):
        ok=sum(1 for s in range(40) if trial(nblk,700+s,combine))
        row.append("nblk=%d:%2d/40" % (nblk,ok))
    print("     %-9s %s" % (combine, "  ".join(row)))
