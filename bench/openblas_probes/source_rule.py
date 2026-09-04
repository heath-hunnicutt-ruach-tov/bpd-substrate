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
# ★ THE SOURCE (driver/level3/level3.c v0.3.29, lines 292-301):
#     for (ls = 0; ls < k; ls += min_l) {
#       min_l = k - ls;
#       if (min_l >= GEMM_Q * 2)  min_l = GEMM_Q;
#       else if (min_l > GEMM_Q)
#            min_l = ((min_l/2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
# GEMM_Q = sgemm_q = 384 (read from the live table).  GEMM_UNROLL_M for
# SANDYBRIDGE sgemm is 16.  THAT IS BOTH REGIMES IN ONE RULE.
def split_source(K, Q=384, UNROLL=16):
    out=[]; ls=0
    while ls < K:
        min_l = K - ls
        if min_l >= Q*2:
            min_l = Q
        elif min_l > Q:
            min_l = ((min_l//2 + UNROLL - 1)//UNROLL) * UNROLL
        out.append(min_l); ls += min_l
    return out
print("   THE SOURCE RULE vs EVERY OBSERVED SPLIT:")
obs={400:[208,192],500:[256,244],640:[320,320],896:[384,256,256],1024:[384,320,320],4096:[384]*9+[320,320]}
for K,o in sorted(obs.items()):
    s=split_source(K)
    print("     K=%-5d source=%-24s observed=%-24s %s" % (K,str(s[:3])+("..." if len(s)>3 else ""),
          str(o[:3])+("..." if len(o)>3 else ""), "AGREE" if s==o else "DIFFER"))
print()
print("   AND NUMERICALLY, across K it never saw:")
M=N=8
for K in (385,400,500,512,640,768,896,1024,1152,1536,2048,3000,4096):
    g=torch.Generator().manual_seed(42)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    r=nd(blocked(A.numpy(),B.numpy(),split_source(K)),(A@B).numpy())
    print("     K=%-5d -> %s" % (K,"MATCH" if r==0 else "%d diff"%r))
