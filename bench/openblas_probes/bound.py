import numpy as np, torch
def diff(a,b):
    a=np.ascontiguousarray(a,dtype=np.float32).ravel(); b=np.ascontiguousarray(b,dtype=np.float32).ravel()
    return int((a!=b).sum())
# ★ MEASURE THE BOUNDARIES, DO NOT FIT A PARAMETER.
# Sweep K one step at a time and find where torch's result STOPS matching a
# plain sequential sum.  The K at which it first diverges is a boundary the
# driver chose -- observed, not guessed.
M=N=8
print("   K-SWEEP: where does torch's gemm stop matching sequential accumulation?")
print("   (M=N=8 fixed, K stepped -- the first divergence is a real boundary)")
prev_ok=None; edges=[]
for K in range(8, 1200, 8):
    g=torch.Generator().manual_seed(7)
    A=torch.randn(M,K,generator=g,dtype=torch.float32); B=torch.randn(K,N,generator=g,dtype=torch.float32)
    ref=(A@B).numpy(); a=A.numpy(); b=B.numpy()
    acc=np.zeros((M,N),dtype=np.float32)
    for k in range(K): acc=(acc+np.float32(a[:,k:k+1]*b[k:k+1,:])).astype(np.float32)
    ok = diff(acc,ref)==0
    if prev_ok is not None and ok!=prev_ok: edges.append((K,ok))
    prev_ok=ok
print("     transitions (K, now_matching):", edges[:12])
print("     total transitions:", len(edges))
