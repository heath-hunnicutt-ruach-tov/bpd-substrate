import numpy as np, torch
def ulp(p,q):
    p=np.ascontiguousarray(p,dtype=np.float32).ravel(); q=np.ascontiguousarray(q,dtype=np.float32).ravel()
    pi=p.view(np.int32).astype(np.int64); qi=q.view(np.int32).astype(np.int64)
    B=np.int64(0x80000000); pi=np.where(pi<0,B-pi,pi); qi=np.where(qi<0,B-qi,qi)
    d=np.abs(pi-qi); return int(d.max()), int((d>0).sum()), d.size
def blocked(a,b,sizes):
    M,K=a.shape; N=b.shape[1]
    acc=np.zeros((M,N),dtype=np.float32); k0=0
    for KB in sizes:
        blk=np.zeros((M,N),dtype=np.float32)
        for k in range(k0,min(k0+KB,K)): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32); k0+=KB
    return acc
SPLIT=[384]*9+[320,320]
# ★ DOES THE K=4096 SPLIT HOLD AT WIDER M,N?  The published problem is 4096-SQUARE.
# The observation was at M=8,N=8.  If the split depends on M/N this fails.
print("   K=4096 split, tested at growing M,N (published problem is 4096-square):")
for M,N in ((8,8),(16,16),(32,32),(64,64)):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,4096,generator=g,dtype=torch.float32); B=torch.randn(4096,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy()
    m,n,t=ulp(blocked(A.numpy(),B.numpy(),SPLIT),ref)
    print("     M=%-4d N=%-4d  max_ulp=%-6d diverged=%d/%d  %s" % (M,N,m,n,t,"MATCH" if n==0 else ""))
