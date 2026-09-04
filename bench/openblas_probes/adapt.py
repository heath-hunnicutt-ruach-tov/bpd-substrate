import numpy as np, torch
def nd(a,b):
    a=np.ascontiguousarray(a,dtype=np.float32).ravel(); b=np.ascontiguousarray(b,dtype=np.float32).ravel()
    return int((a!=b).sum())
def blocked(a,b,KB):
    M,K=a.shape; N=b.shape[1]
    acc=np.zeros((M,N),dtype=np.float32)
    for k0 in range(0,K,KB):
        k1=min(k0+KB,K); blk=np.zeros((M,N),dtype=np.float32)
        for k in range(k0,k1): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32)
    return acc
# ★ SOLVE FOR THE BLOCK SIZE PER K, rather than assuming one constant.
# For each K, find every KB that reproduces torch exactly.
print("   WHICH K-BLOCK REPRODUCES torch, PER K?  (solve, don't assume)")
M=N=8
for K in (384,385,512,640,768,896,1024,1152,1536,2048):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    hits=[KB for KB in range(64,K+1,32) if nd(blocked(a,b,KB),ref)==0]
    if K<=512: hits=[KB for KB in range(32,K+1,16) if nd(blocked(a,b,KB),ref)==0]
    print("     K=%-5d  exact-match block sizes: %s" % (K, hits[:8] if hits else "NONE"))
