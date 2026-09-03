import ctypes, numpy as np
L=ctypes.CDLL("/home/dibbur-patch/step3-det-gemv/bpd/build/bpd_cpu.so")
FP=ctypes.POINTER(ctypes.c_float); UP=ctypes.POINTER(ctypes.c_uint8); I=ctypes.c_int
L.bpd_qmatmul_q8_0_llamafile_cpu.restype=None
L.bpd_qmatmul_q8_0_llamafile_cpu.argtypes=[UP,FP,FP,I,I,I]

# MINIMAL CASE: K=32 (one q8_0 block), 1 output row, 1 token.
K=32; M=1; N=1
rng=np.random.default_rng(7)
qs=rng.integers(-127,128,size=K).astype(np.int8)      # the int8 weights
scale=np.float16(0.0123456)                            # the block scale
x=rng.standard_normal(K).astype(np.float32)            # the f32 activations

# build the q8_0 block by hand: 2-byte fp16 scale, then 32 int8
blk=np.zeros(34,dtype=np.uint8)
blk[0:2]=np.frombuffer(scale.tobytes(),dtype=np.uint8)
blk[2:34]=qs.view(np.uint8)

out=np.zeros(M*N,dtype=np.float32)
L.bpd_qmatmul_q8_0_llamafile_cpu(blk.ctypes.data_as(UP), x.ctypes.data_as(FP),
                                 out.ctypes.data_as(FP), I(M), I(N), I(K))

# THE TWO CANDIDATE ORDERS, computed in float32 exactly as a kernel would
w = qs.astype(np.float32) * np.float32(scale)
seq = np.float32(0.0)
for i in range(K): seq = np.float32(seq + np.float32(w[i]*x[i]))
prods = (w*x).astype(np.float32)
pair = prods.copy()
nn = K
while nn > 1:                     # pairwise / tree summation
    half = nn//2
    pair[:half] = (pair[:half] + pair[half:nn]).astype(np.float32)
    nn = half
f64 = np.float64(np.float64(w.astype(np.float64))*x.astype(np.float64)).sum()

print("   MINIMAL q8_0 CASE: K=32, M=1, N=1")
print("     kernel out    = %.9g" % out[0])
print("     sequential f32= %.9g   (diff %.3e)" % (seq, abs(out[0]-seq)))
print("     pairwise  f32 = %.9g   (diff %.3e)" % (pair[0], abs(out[0]-pair[0])))
print("     float64 exact = %.9g   (diff %.3e)" % (f64, abs(out[0]-f64)))

print()
print("   ★ ALL THREE ORDERS AGREE (-9.5141) and the kernel says -9.5596.")
print("     4.55e-02 is FAR too large for summation order at K=32.")
print("     So the divergence is NOT order -- it is the DEQUANT or the")
print("     ACTIVATION QUANTIZATION.  llamafile q8_0 matmul QUANTIZES X.")
print()
# llamafile-style: X is quantized to q8_0 too, then int8 dot products
def quant_row(v):
    amax=np.abs(v).max()
    d=np.float32(amax/127.0)
    q=np.round(v/d).astype(np.int32).clip(-127,127) if d>0 else np.zeros_like(v,dtype=np.int32)
    return q.astype(np.int32), np.float16(d)
xq,xd = quant_row(x)
dot = int((qs.astype(np.int32)*xq).sum())
both = np.float32(np.float32(scale)*np.float32(xd)*np.float32(dot))
print("     X-also-quantized (int8 dot x scale_w x scale_x):")
print("       = %.9g   (diff from kernel %.3e)" % (both, abs(out[0]-both)))
