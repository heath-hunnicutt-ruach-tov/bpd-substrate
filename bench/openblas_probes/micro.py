import numpy as np, torch
def nd(a,b):
    a=np.ascontiguousarray(a,dtype=np.float32).ravel(); b=np.ascontiguousarray(b,dtype=np.float32).ravel()
    return int((a!=b).sum())
# ★ THE DISASSEMBLY SHOWS: 4 accumulators ymm4-7, each doing
#   acc = acc + (A_vec * broadcast(B_scalar))  -- ONE k per step, sequential.
# So WITHIN a block the accumulation IS sequential per output element.
# If that is true, a plain sequential sum should match for K <= 384.
# TEST THE BOUNDARY CLAIM DIRECTLY at many K, not just the transition.
print("   IF the micro-kernel accumulates sequentially within a block,")
print("   plain sequential should match for ALL K <= 384:")
M=N=8
bad=[]
for K in list(range(8,385,24))+[384]:
    g=torch.Generator().manual_seed(11)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    acc=np.zeros((M,N),dtype=np.float32)
    for k in range(K): acc=(acc+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
    if nd(acc,ref): bad.append((K,nd(acc,ref)))
print("     K values <=384 that FAIL sequential: %s" % (bad if bad else "NONE -- all match"))
print()
print("   AND ABOVE 384, is the FIRST block still sequential?  Compare a")
print("   K=500 run against sequential-over-first-384 + sequential-over-rest:")
for K in (400,500,640,768):
    g=torch.Generator().manual_seed(11)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    for split in (384, K//2, (K+1)//2):
        acc=np.zeros((M,N),dtype=np.float32)
        for lo,hi in ((0,split),(split,K)):
            blk=np.zeros((M,N),dtype=np.float32)
            for k in range(lo,hi): blk=(blk+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
            acc=(acc+blk).astype(np.float32)
        if nd(acc,ref)==0:
            print("     K=%-4d MATCHES with split at %d" % (K,split)); break
    else:
        print("     K=%-4d no simple 2-way split matches" % K)
