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
# ★ OBSERVED at K=4096: nine 384s then 320s.  9*384=3456, 4096-3456=640=320+320.
SPLIT=[384]*9+[320,320]
assert sum(SPLIT)==4096, sum(SPLIT)
print("   THE OBSERVED K=4096 SPLIT: %s  (sums to %d)" % (str(SPLIT[:3])+"...x9...", sum(SPLIT)))
M=N=8
g=torch.Generator().manual_seed(42)
A=torch.randn(M,4096,generator=g,dtype=torch.float32); B=torch.randn(4096,N,generator=g,dtype=torch.float32)
ref=(A@B).numpy()
m,n,t=ulp(blocked(A.numpy(),B.numpy(),SPLIT),ref)
print("   observed-split vs torch at K=4096: max_ulp=%d diverged=%d/%d  %s" % (m,n,t,"★ MATCH" if n==0 else ""))
# alternatives-differ: does a wrong split diverge here?
for alt in ([384]*10+[256], [512]*8, [384]*8+[384,320,320][:3]):
    if sum(alt)!=4096: alt=alt+[4096-sum(alt)] if sum(alt)<4096 else alt
    if sum(alt)!=4096: continue
    am,an,_=ulp(blocked(A.numpy(),B.numpy(),alt),ref)
    print("     alternative %-22s diverges on %d  (so the test is exercised)" % (str(alt[:2])+"...",an))
