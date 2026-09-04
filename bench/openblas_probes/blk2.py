import numpy as np, torch
def ulp(p,q):
    p=np.ascontiguousarray(p,dtype=np.float32).ravel(); q=np.ascontiguousarray(q,dtype=np.float32).ravel()
    pi=p.view(np.int32).astype(np.int64); qi=q.view(np.int32).astype(np.int64)
    B=np.int64(0x80000000); pi=np.where(pi<0,B-pi,pi); qi=np.where(qi<0,B-qi,qi)
    d=np.abs(pi-qi); return int(d.max()), int((d>0).sum()), d.size
def blocked(a,b,KB):
    M,K=a.shape; K2,N=b.shape
    acc=np.zeros((M,N),dtype=np.float32)
    for k0 in range(0,K,KB):
        k1=min(k0+KB,K); blk=np.zeros((M,N),dtype=np.float32)
        for k in range(k0,k1): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32)
    return acc
print("   ★ GENERALISATION: does K-block=256 hold at other shapes?")
for M,K,N in ((512,512,512),(256,768,256),(128,1024,128),(384,384,384),(64,2048,64)):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    m,nd,t=ulp(blocked(a,b,256),ref)
    alt=ulp(blocked(a,b,128),ref)[1]
    print("     (%d,%d)@(%d,%d): KB=256 -> %s   (KB=128 diverges on %d, so exercised)"
          % (M,K,K,N, "0 ULP MATCH" if nd==0 else "%d ulp/%d"%(m,nd), alt))
