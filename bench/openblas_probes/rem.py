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
# ★ EVEN SPLITS MATCH; REMAINDERS DO NOT.  So how does OpenBLAS place the
# remainder?  Candidates: full blocks then a short tail; short head then full;
# or the min(K-k0, KB) walk.  Solve on K=1024 and K=896.
cands = {
  "full-then-tail (384,384,256)": lambda K,KB=384: [KB]*(K//KB) + ([K%KB] if K%KB else []),
  "tail-then-full":               lambda K,KB=384: ([K%KB] if K%KB else []) + [KB]*(K//KB),
  "half-split":                   lambda K,KB=384: [ (K+1)//2, K//2 ],
}
print("   HOW IS THE REMAINDER PLACED?  (even splits already match)")
M=N=8
for K in (896,1024,2048,4096,385):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    out=[]
    for nm,f in cands.items():
        s=f(K); r=nd(blocked_list(a,b,s),ref)
        out.append("%s:%s" % (nm.split()[0], "MATCH" if r==0 else str(r)))
    print("     K=%-5d %s" % (K, "   ".join(out)))
