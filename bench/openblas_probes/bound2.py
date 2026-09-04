import numpy as np, torch
def nd(a,b):
    a=np.ascontiguousarray(a,dtype=np.float32).ravel(); b=np.ascontiguousarray(b,dtype=np.float32).ravel()
    return int((a!=b).sum())
def seqacc(a,b,M,N,K):
    acc=np.zeros((M,N),dtype=np.float32)
    for k in range(K): acc=(acc+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
    return acc
# ★ EXACT BOUNDARY, then does it MOVE with M and N?  A driver block constant
# should be stable in K; a coincidence should wander.
print("   1. EXACT K where sequential stops matching (M=N=8):")
lo,hi=384,400
for K in range(lo,hi+1):
    g=torch.Generator().manual_seed(7)
    A=torch.randn(8,K,generator=g,dtype=torch.float32); B=torch.randn(K,8,generator=g,dtype=torch.float32)
    ok = nd(seqacc(A.numpy(),B.numpy(),8,8,K),(A@B).numpy())==0
    if not ok: print("     first divergence at K=%d" % K); break
print()
print("   2. DOES THE BOUNDARY MOVE WITH M,N?  (a real block constant should not)")
for M,N in ((4,4),(8,8),(16,16),(32,32),(8,64)):
    first=None
    for K in range(376, 408, 2):
        g=torch.Generator().manual_seed(7)
        A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
        if nd(seqacc(A.numpy(),B.numpy(),M,N,K),(A@B).numpy())!=0: first=K; break
    print("     M=%-3d N=%-3d  first divergence at K=%s" % (M,N,first))
