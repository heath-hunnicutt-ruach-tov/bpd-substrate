import numpy as np, torch
def nd(a,b):
    a=np.ascontiguousarray(a,dtype=np.float32).ravel(); b=np.ascontiguousarray(b,dtype=np.float32).ravel()
    return int((a!=b).sum())
def blocked_list(a,b,sizes):
    M,K=a.shape; N=b.shape[1]
    acc=np.zeros((M,N),dtype=np.float32); k0=0
    for KB in sizes:
        blk=np.zeros((M,N),dtype=np.float32)
        for k in range(k0,min(k0+KB,K)): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32); k0+=KB
    return acc
def split_ceil(K, KBmax=384):
    """OpenBLAS-style: nblocks = ceil(K/KBmax), then divide K as evenly as possible."""
    n=(K+KBmax-1)//KBmax
    base=K//n; rem=K%n
    return [base+(1 if i<rem else 0) for i in range(n)]
def split_ceil_front(K, KBmax=384):
    n=(K+KBmax-1)//KBmax
    base=K//n; rem=K%n
    return [base+1]*rem + [base]*(n-rem)
print("   TEST THE EVEN-SPLIT RULE (nblocks=ceil(K/384), K divided evenly):")
M=N=8
for K in (384,385,512,640,768,896,1024,1152,1536,2048,4096):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    s1=split_ceil(K); s2=split_ceil_front(K)
    r1=nd(blocked_list(a,b,s1),ref); r2=nd(blocked_list(a,b,s2),ref)
    print("     K=%-5d split=%-22s -> %-8s   front-heavy -> %s"
          % (K, str(s1[:4])+("..." if len(s1)>4 else ""), "MATCH" if r1==0 else "%d diff"%r1,
             "MATCH" if r2==0 else "%d diff"%r2))
