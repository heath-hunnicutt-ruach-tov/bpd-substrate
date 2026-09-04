import numpy as np, torch
def ulp(p,q):
    p=np.ascontiguousarray(p,dtype=np.float32).ravel(); q=np.ascontiguousarray(q,dtype=np.float32).ravel()
    pi=p.view(np.int32).astype(np.int64); qi=q.view(np.int32).astype(np.int64)
    B=np.int64(0x80000000); pi=np.where(pi<0,B-pi,pi); qi=np.where(qi<0,B-qi,qi)
    d=np.abs(pi-qi); return int(d.max()), int((d>0).sum()), d.size
def blocked(a,b,KB):
    M,K=a.shape; N=b.shape[1]
    acc=np.zeros((M,N),dtype=np.float32)
    for k0 in range(0,K,KB):
        k1=min(k0+KB,K); blk=np.zeros((M,N),dtype=np.float32)
        for k in range(k0,k1): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32)
    return acc
# ★ THE BOUNDARY SAYS KB=384.  GENERALISATION-CHECK ACROSS SHAPES FIRST --
# this is exactly where the KB=256 fit died last night.
print("   K-BLOCK=384 (from the measured invariant boundary), generalisation-checked:")
shapes=[(8,512,8),(16,768,16),(32,1024,32),(64,1536,64),(128,2048,128),(4,4096,4),(256,512,256)]
for M,K,N in shapes:
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    m,n,t=ulp(blocked(a,b,384),ref)
    alt=ulp(blocked(a,b,256),ref)[1]
    print("     (%3d,%4d)@(%4d,%3d)  KB=384 -> %-18s  (KB=256 diverges on %d)"
          % (M,K,K,N, "0 ULP MATCH" if n==0 else "%d ulp/%d"%(m,n), alt))
