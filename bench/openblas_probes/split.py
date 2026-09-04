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
# ★ OBSERVED: K=400 -> [208,192]   K=500 -> [256,244]   K=640 -> [320,320]
#             K=768 -> [384,384]   K=1024 -> [384,384,...]
# 208 = ceil(400/2/8)*8 = ceil(200/8)*8 = 25*8 = 200?  no, 208 = 26*8
# 256 = ceil(500/2/8)*8 = ceil(250/8)*8 = 32*8 = 256   YES
# 208: ceil(200/8)*8 = 200.  but observed 208.  try ceil to 16: 208=13*16 YES, 256=16*16 YES
def split_rule(K, Q=384, align=16):
    n = (K + Q - 1)//Q                      # number of blocks
    if n <= 1: return [K]
    base = (K + n - 1)//n                   # ceil(K/n)
    first = ((base + align - 1)//align)*align   # round UP to alignment
    out=[]; rem=K
    for i in range(n-1):
        out.append(min(first, rem)); rem -= out[-1]
    out.append(rem)
    return out
print("   THE OBSERVED SPLITS, and what the rule predicts:")
for K,obs in ((400,[208,192]),(500,[256,244]),(640,[320,320]),(768,[384,384]),(1024,None)):
    print("     K=%-5d observed=%-16s rule=%s" % (K, str(obs), split_rule(K)))
print()
print("   VERIFY the rule numerically:")
M=N=8
for K in (385,400,500,512,640,768,896,1024,1152,1536,2048,4096):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); s=split_rule(K)
    r=nd(blocked(A.numpy(),B.numpy(),s),ref)
    print("     K=%-5d %-30s -> %s" % (K,str(s[:4])+("..." if len(s)>4 else ""),"MATCH" if r==0 else "%d diff"%r))
