/* bpd_gpu_kernels.cu — GPU kernel implementations with host-callable wrappers.
 *
 * Each kernel has:
 *   1. __global__ kernel function
 *   2. extern "C" host wrapper (handles grid/block config + launch)
 *
 * Usage from Python/ctypes:
 *   lib = ctypes.CDLL("build/bpd_gpu.so")
 *   lib.bpd_relu_gpu(input_ptr, output_ptr, n)
 *
 * Build:
 *   nvcc -O2 -shared -Xcompiler -fPIC -o build/bpd_gpu.so bench/bpd_gpu_kernels.cu
 */

#include <cuda_runtime.h>
#include <math.h>

// ── Helper: grid/block config ──

static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }
#define BLOCK 256

// ── GPU memory management ──

extern "C" {
void* gpu_alloc(int bytes) { void* p; cudaMalloc(&p, bytes); return p; }
void  gpu_free(void* p) { cudaFree(p); }
void  gpu_h2d(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyHostToDevice); }
void  gpu_d2h(void* d, const void* s, int bytes) { cudaMemcpy(d, s, bytes, cudaMemcpyDeviceToHost); }
void  gpu_sync() { cudaDeviceSynchronize(); }
}

// ═══════════════════════════════════════════════════════════════
// Elementwise kernels
// ═══════════════════════════════════════════════════════════════

__global__ void k_relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

__global__ void k_silu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x / (1.0f + expf(-x)); }
}

__global__ void k_mish(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x * tanhf(log1pf(expf(x))); }
}

__global__ void k_sigmoid(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

__global__ void k_tanh_k(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(in[i]);
}

__global__ void k_gelu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }
}

__global__ void k_neg(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

__global__ void k_abs(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fabsf(in[i]);
}

__global__ void k_exp(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(in[i]);
}

// ── Fused matmul + bias + relu ──

__global__ void k_mm_bias_relu(const float* A, const float* B,
                                const float* bias, float* C,
                                int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N, col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[idx] = fmaxf(0.0f, sum + bias[col]);
    }
}

// ── BatchNorm (inference, precomputed scale/offset) ──

__global__ void k_batchnorm_affine(const float* input, const float* scale,
                                    const float* offset, float* output,
                                    int total, int C, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int c = (i / HW) % C;
        output[i] = input[i] * scale[c] + offset[c];
    }
}

// ── Residual add ──

__global__ void k_residual_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ── Upsample nearest 2x ──

__global__ void k_upsample_nearest2d(const float* input, float* output,
                                      int N, int C, int H, int W) {
    int H_out = 2 * H, W_out = 2 * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (i < total) {
        int ow = i % W_out, oh = (i / W_out) % H_out;
        int c = (i / (H_out * W_out)) % C, n = i / (C * H_out * W_out);
        output[i] = input[((n * C + c) * H + oh / 2) * W + ow / 2];
    }
}

// -- L1 missing kernel functions (8 ops) --

__global__ void k_leaky_relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 0.0f ? x : 0.01f * x; }
}

__global__ void k_elu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 0.0f ? x : expm1f(x); }
}

__global__ void k_hardsigmoid(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f; }
}

__global__ void k_softplus(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = in[i]; out[i] = x > 20.0f ? x : log1pf(expf(x)); }
}

__global__ void k_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] * b[i]; }
}

__global__ void k_softmax(const float* in, float* out, int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x; if (row >= rows) return;
    const float* ri = in + row * cols; float* ro = out + row * cols;
    float lmax = -1e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) lmax = fmaxf(lmax, ri[j]);
    sdata[threadIdx.x] = lmax; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x+s]); __syncthreads(); }
    float rmax = sdata[0]; __syncthreads();
    float lsum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) { float e = expf(ri[j]-rmax); ro[j] = e; lsum += e; }
    sdata[threadIdx.x] = lsum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x+s]; __syncthreads(); }
    float sum = sdata[0]; __syncthreads();
    for (int j = threadIdx.x; j < cols; j += blockDim.x) ro[j] /= sum;
}

__global__ void k_layernorm(const float* in, const float* gamma, const float* beta, float* out, int rows, int cols, float eps) {
    extern __shared__ float sd[]; float* sm = sd; float* sv = sd + blockDim.x;
    int row = blockIdx.x; if (row >= rows) return;
    const float* ri = in + row*cols; float* ro = out + row*cols;
    float ls = 0.0f; for (int j = threadIdx.x; j < cols; j += blockDim.x) ls += ri[j];
    sm[threadIdx.x] = ls; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s]; __syncthreads(); }
    float mean = sm[0]/cols; __syncthreads();
    float lv = 0.0f; for (int j = threadIdx.x; j < cols; j += blockDim.x) { float d = ri[j]-mean; lv += d*d; }
    sv[threadIdx.x] = lv; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x+s]; __syncthreads(); }
    float istd = 1.0f/sqrtf(sv[0]/cols + eps); __syncthreads();
    for (int j = threadIdx.x; j < cols; j += blockDim.x) ro[j] = gamma[j]*(ri[j]-mean)*istd + beta[j];
}

__global__ void k_maxpool2d(const float* in, float* out, int N, int C, int H, int W, int kH, int kW, int stride, int pad, int Ho, int Wo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*Ho*Wo; if (i >= total) return;
    int ow = i%Wo, oh = (i/Wo)%Ho, c = (i/(Wo*Ho))%C, n = i/(Wo*Ho*C);
    float val = -1e30f;
    for (int kh = 0; kh < kH; kh++) for (int kw = 0; kw < kW; kw++) {
        int hi = oh*stride-pad+kh, wi = ow*stride-pad+kw;
        if (hi >= 0 && hi < H && wi >= 0 && wi < W) val = fmaxf(val, in[((n*C+c)*H+hi)*W+wi]);
    }
    out[i] = val;
}


// ═══════════════════════════════════════════════════════════════
// Host-callable wrappers (extern "C" for ctypes/FFI)
// ═══════════════════════════════════════════════════════════════

extern "C" {

// Elementwise
void bpd_relu_gpu(const float* in, float* out, int n) {
    k_relu<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_silu_gpu(const float* in, float* out, int n) {
    k_silu<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_mish_gpu(const float* in, float* out, int n) {
    k_mish<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_sigmoid_gpu(const float* in, float* out, int n) {
    k_sigmoid<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_tanh_gpu(const float* in, float* out, int n) {
    k_tanh_k<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_gelu_gpu(const float* in, float* out, int n) {
    k_gelu<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_neg_gpu(const float* in, float* out, int n) {
    k_neg<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_abs_gpu(const float* in, float* out, int n) {
    k_abs<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }
void bpd_exp_gpu(const float* in, float* out, int n) {
    k_exp<<<ceildiv(n, BLOCK), BLOCK>>>(in, out, n); }

// Fused
void bpd_mm_bias_relu_gpu(const float* A, const float* B, const float* bias,
                            float* C, int M, int N, int K) {
    k_mm_bias_relu<<<ceildiv(M * N, BLOCK), BLOCK>>>(A, B, bias, C, M, N, K); }

// BatchNorm
void bpd_batchnorm_affine_gpu(const float* input, const float* scale,
                                const float* offset, float* output,
                                int N, int C, int HW) {
    int total = N * C * HW;
    k_batchnorm_affine<<<ceildiv(total, BLOCK), BLOCK>>>(input, scale, offset, output, total, C, HW); }

// Residual add
void bpd_residual_add_gpu(const float* a, const float* b, float* out, int n) {
    k_residual_add<<<ceildiv(n, BLOCK), BLOCK>>>(a, b, out, n); }

// Upsample
void bpd_upsample_nearest2d_gpu(const float* input, float* output,
                                  int N, int C, int H, int W) {
    int total = N * C * 4 * H * W;
    k_upsample_nearest2d<<<ceildiv(total, BLOCK), BLOCK>>>(input, output, N, C, H, W); }

// -- L1 missing wrappers --
void bpd_leaky_relu_gpu(const float* in, float* out, int n) { k_leaky_relu<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_elu_gpu(const float* in, float* out, int n) { k_elu<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_hardsigmoid_gpu(const float* in, float* out, int n) { k_hardsigmoid<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_softplus_gpu(const float* in, float* out, int n) { k_softplus<<<ceildiv(n,BLOCK),BLOCK>>>(in,out,n); }
void bpd_mul_gpu(const float* a, const float* b, float* out, int n) { k_mul<<<ceildiv(n,BLOCK),BLOCK>>>(a,b,out,n); }
void bpd_softmax_gpu(const float* in, float* out, int rows, int cols) { k_softmax<<<rows,BLOCK,BLOCK*sizeof(float)>>>(in,out,rows,cols); }
void bpd_layernorm_gpu(const float* in, const float* gamma, const float* beta, float* out, int rows, int cols, float eps) { k_layernorm<<<rows,BLOCK,2*BLOCK*sizeof(float)>>>(in,gamma,beta,out,rows,cols,eps); }
void bpd_maxpool2d_gpu(const float* in, float* out, int N, int C, int H, int W, int kH, int kW, int stride, int pad) { int Ho=(H+2*pad-kH)/stride+1,Wo=(W+2*pad-kW)/stride+1; k_maxpool2d<<<ceildiv(N*C*Ho*Wo,BLOCK),BLOCK>>>(in,out,N,C,H,W,kH,kW,stride,pad,Ho,Wo); }

} // extern "C"
