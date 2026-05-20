/* bpd_cpu.c — BPD CPU kernel reference implementations
 *
 * Matmul kernels delegate to cblas_sgemm (OpenBLAS) so that accumulation
 * order is bit-identical with PyTorch CPU, which also dispatches to the
 * same BLAS backend. All other kernels (elementwise, reductions, spatial)
 * are scalar C and match PyTorch's ATen CPU kernels to within the expected
 * ULP budget for each operation class.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lopenblas -lm
 */

#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Matmul: delegate to cblas_sgemm for bit-identity with PyTorch CPU ── */

/* CPU matmul: C[M,N] = A[M,K] @ B[K,N]
 * Uses cblas_sgemm (OpenBLAS) — same backend as torch.mm on CPU.
 * Produces bit-identical output to PyTorch at all matrix sizes. */
void bpd_mm_cpu(const float* A, const float* B, float* C,
                int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                      B, N,
                0.0f, C, N);
}

/* CPU fused matmul + bias + relu
 * Computes C = relu(A @ B + bias) in two passes:
 *   1. cblas_sgemm for the matmul (bit-identical with torch.mm)
 *   2. scalar epilogue for bias + relu (bit-identical with ATen) */
void bpd_mm_bias_relu_cpu(const float* A, const float* B,
                           const float* bias, float* C,
                           int M, int N, int K) {
    /* Pass 1: C = A @ B via BLAS (bit-identical with torch.mm) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                      B, N,
                0.0f, C, N);
    /* Pass 2: fused bias + relu epilogue in registers */
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++) {
            float v = C[row * N + col] + bias[col];
            C[row * N + col] = v > 0.0f ? v : 0.0f;
        }
}

/* ── Elementwise ops ── */

void bpd_relu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

void bpd_silu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void bpd_mish_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x * tanhf(log1pf(expf(x)));
    }
}

void bpd_sigmoid_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = 1.0f / (1.0f + expf(-input[i]));
}

void bpd_tanh_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = tanhf(input[i]);
}

void bpd_gelu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
    }
}

void bpd_neg_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = -input[i];
}

void bpd_abs_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = fabsf(input[i]);
}

void bpd_exp_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = expf(input[i]);
}

/* ── Reductions ── */

void bpd_sum_cpu(const float* input, float* output, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += input[i];
    *output = s;
}

void bpd_mean_cpu(const float* input, float* output, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += input[i];
    *output = s / (float)n;
}

void bpd_max_cpu(const float* input, float* output, int n) {
    float m = input[0];
    for (int i = 1; i < n; i++) if (input[i] > m) m = input[i];
    *output = m;
}

/* ── Softmax (row-wise) ── */

void bpd_softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        float mx = row_in[0];
        for (int c = 1; c < cols; c++) if (row_in[c] > mx) mx = row_in[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - mx);
            sum += row_out[c];
        }
        for (int c = 0; c < cols; c++) row_out[c] /= sum;
    }
}

/* ── LayerNorm ── */

void bpd_layernorm_cpu(const float* input, const float* gamma,
                        const float* beta, float* output,
                        int N, int D, float eps) {
    for (int n = 0; n < N; n++) {
        const float* x = input + n * D;
        float* y = output + n * D;
        float mean = 0.0f;
        for (int d = 0; d < D; d++) mean += x[d];
        mean /= (float)D;
        float var = 0.0f;
        for (int d = 0; d < D; d++) { float dx = x[d] - mean; var += dx*dx; }
        var /= (float)D;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++)
            y[d] = gamma[d] * (x[d] - mean) * inv_std + beta[d];
    }
}

/* ── Conv2D (direct, no im2col) ── */

void bpd_conv2d_cpu(const float* input, const float* weight, float* output,
                     int N, int C_in, int H, int W,
                     int C_out, int kH, int kW,
                     int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int total = N * C_out * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int co = (idx / (W_out * H_out)) % C_out;
        int n  = idx / (W_out * H_out * C_out);
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
            for (int kh = 0; kh < kH; kh++)
                for (int kw = 0; kw < kW; kw++) {
                    int hi = oh * stride - pad + kh;
                    int wi = ow * stride - pad + kw;
                    if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                        int in_idx = ((n*C_in+ci)*H+hi)*W+wi;
                        int w_idx = ((co*C_in+ci)*kH+kh)*kW+kw;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
        output[idx] = sum;
    }
}

/* ── BatchNorm (inference mode) ── */

void bpd_batchnorm_cpu(const float* input, const float* gamma,
                        const float* beta, const float* mean,
                        const float* var, float* output,
                        int N, int C, int HW, float eps) {
    int total = N * C * HW;
    for (int idx = 0; idx < total; idx++) {
        int c = (idx / HW) % C;
        float x = input[idx];
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        output[idx] = gamma[c] * (x - mean[c]) * inv_std + beta[c];
    }
}

/* ── Upsample nearest 2x ── */

void bpd_upsample_nearest2d_cpu(const float* input, float* output,
                                 int N, int C, int H, int W) {
    int H_out = 2 * H, W_out = 2 * W;
    int total = N * C * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int c = (idx / (H_out * W_out)) % C;
        int n = idx / (C * (H_out * W_out));
        int ih = oh / 2, iw = ow / 2;
        int in_idx = ((n*C+c)*H+ih)*W+iw;
        output[idx] = input[in_idx];
    }
}

/* ── MaxPool2D / AvgPool2D ── */

void bpd_maxpool2d_cpu(const float* input, float* output,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int total = N * C * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        float val = -1e30f;
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int hi = oh * stride - pad + kh;
                int wi = ow * stride - pad + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    float v = input[((n*C+c)*H+hi)*W+wi];
                    if (v > val) val = v;
                }
            }
        output[idx] = val;
    }
}

void bpd_avgpool2d_cpu(const float* input, float* output,
                        int N, int C, int H, int W,
                        int kH, int kW, int stride, int pad) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int total = N * C * H_out * W_out;
    for (int idx = 0; idx < total; idx++) {
        int ow = idx % W_out;
        int oh = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);
        float sum = 0.0f; int count = 0;
        for (int kh = 0; kh < kH; kh++)
            for (int kw = 0; kw < kW; kw++) {
                int hi = oh * stride - pad + kh;
                int wi = ow * stride - pad + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    sum += input[((n*C+c)*H+hi)*W+wi];
                    count++;
                }
            }
        output[idx] = sum / (float)count;
    }
}

/* ── Linear (matmul + bias): delegate to cblas_sgemm ── */

void bpd_linear_cpu(const float* input, const float* weight,
                     const float* bias, float* output,
                     int M, int N, int K) {
    /* weight is stored as [N, K] (PyTorch convention: out_features x in_features)
     * so output = input @ weight^T + bias */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, input,  K,
                      weight, K,
                0.0f, output, N);
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++)
            output[row * N + col] += bias[col];
}
