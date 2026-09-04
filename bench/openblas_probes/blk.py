import numpy as np, torch
def ulp(p,q):
    p=np.ascontiguousarray(p,dtype=np.float32).ravel(); q=np.ascontiguousarray(q,dtype=np.float32).ravel()
    pi=p.view(np.int32).astype(np.int64); qi=q.view(np.int32).astype(np.int64)
    B=np.int64(0x80000000); pi=np.where(pi<0,B-pi,pi); qi=np.where(qi<0,B-qi,qi)
    d=np.abs(pi-qi); return int(d.max()), int((d>0).sum()), d.size
# ★ OpenBLAS sgemm_kernel_SANDYBRIDGE: 504 vmulps + 504 vaddps, NO vfmadd.
# The driver blocks K.  Standard OpenBLAS SANDYBRIDGE params: GEMM_P/Q/R.
# Test which K-BLOCK SIZE reproduces torch, by blocking the accumulation.
N=512
g=torch.Generator().manual_seed(42)
A=torch.randn(N,N,generator=g,dtype=torch.float32); B=torch.randn(N,N,generator=g,dtype=torch.float32)
ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
print("   K-BLOCKED accumulation vs torch at N=%d (which block size matches?):" % N)
for KB in (128, 192, 256, 320, 384, 512):
    acc=np.zeros((N,N),dtype=np.float32)
    for k0 in range(0,N,KB):
        k1=min(k0+KB,N)
        # inner block accumulated, then added to the running result
        blk=np.zeros((N,N),dtype=np.float32)
        for k in range(k0,k1):
            blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32)
    m,nd,t=ulp(acc,ref)
    print("     K-block=%-4d max_ulp=%-8d diverged=%d/%d (%.1f%%)" % (KB,m,nd,t,100.0*nd/t))
