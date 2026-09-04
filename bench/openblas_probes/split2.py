import numpy as np, torch
def nd(a,b):
    a=np.ascontiguousarray(a,dtype=np.float32).ravel(); b=np.ascontiguousarray(b,dtype=np.float32).ravel()
    return int((a!=b).sum())
def blocked(a,b,sizes):
    M,K=a.shape; N=b.shape[1]
    acc=np.zeros((M,N),dtype=np.float32); k0=0
    for KB in sizes:
        blk=np.zeros((M,N),dtype=np.float32)
        for k in range(k0,min(k0+KB,K)): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
        acc=(acc+blk).astype(np.float32); k0+=KB
    return acc
# ★ K=1024 was OBSERVED as 384,384,320 -- so for 3+ blocks it does NOT
# even-split; it takes FULL 384 blocks and the TAIL is what remains,
# rounded UP to 16.  Test: full-384s then align16(tail).
def rule(K, Q=384, align=16):
    n=(K+Q-1)//Q
    if n<=1: return [K]
    if n==2:
        base=(K+1)//2
        first=((base+align-1)//align)*align
        return [min(first,K), K-min(first,K)]
    out=[]; rem=K
    while rem>Q:
        out.append(Q); rem-=Q
    out.append(((rem+align-1)//align)*align)
    return out
print("   3+ BLOCKS: full 384s then align-16 tail (K=1024 observed as 384,384,320):")
M=N=8
for K in (385,400,500,512,640,768,896,1024,1152,1536,2048,4096):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); s=rule(K)
    r=nd(blocked(A.numpy(),B.numpy(),s),ref)
    print("     K=%-5d %-32s -> %s" % (K,str(s[:4])+("..." if len(s)>4 else ""),"MATCH" if r==0 else "%d diff"%r))
