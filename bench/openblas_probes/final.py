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
# ★ THE C POINTERS ALTERNATE -> 2 N-tiles.  Per tile, K=1024 -> [384,320,320].
# K=896 -> [384,256,256].  So: FIRST block 384, then the REST split evenly,
# each rounded to 16.
def rule(K, Q=384, align=16):
    if K<=Q: return [K]
    out=[Q]; rem=K-Q
    n=(rem+Q-1)//Q
    base=(rem+n-1)//n
    b=((base+align-1)//align)*align
    for i in range(n-1):
        out.append(min(b,rem)); rem-=out[-1]
    out.append(rem)
    return out
print("   RULE from the observation: first block 384, rest split evenly (align 16)")
for K,obs in ((896,[384,256,256]),(1024,[384,320,320]),(400,[208,192]),(500,[256,244])):
    print("     K=%-5d observed=%-20s rule=%s" % (K,str(obs),rule(K)))
print()
M=N=8
for K in (385,400,500,512,640,768,896,1024,1152,1536,2048,4096):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); s=rule(K)
    r=nd(blocked(A.numpy(),B.numpy(),s),ref)
    print("     K=%-5d %-32s -> %s" % (K,str(s[:4])+("..." if len(s)>4 else ""),"MATCH" if r==0 else "%d diff"%r))
