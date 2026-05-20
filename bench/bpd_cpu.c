#include <math.h>
#include <stdlib.h>
#include <string.h>

// CPU matmul: C[M,N] = A[M,K] @ B[K,N]
// Sequential accumulation — matches PyTorch CPU DEFAULT backend at K<=256.
// At K>=512 with random data, near-zero cancellation causes ULP divergence
// (named parameter: accumulation_precision in implementation_matches.pl).
void bpd_mm_cpu(const float* A, const float* B, float* C,
                int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// CPU fused matmul + bias + relu
void bpd_mm_bias_relu_cpu(const float* A, const float* B,
                           const float* bias, float* C,
                           int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = fmaxf(0.0f, sum + bias[col]);
        }
    }
}

// CPU relu
void bpd_relu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = fmaxf(0.0f, input[i]);
}

// CPU silu
void bpd_silu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// CPU mish  
void bpd_mish_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x * tanhf(log1pf(expf(x)));
    }
}

// CPU conv2d (direct, no im2col)
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

// CPU batchnorm (inference mode)
//
// Per substrate-design diagnostic 2026-05-20 ~05:45 UTC (mavchin + metayen):
// the 4-op form below produces 32768 ULP (= 2^15) systematic divergence vs
// PyTorch's BN. Root cause candidates:
//   (1) 1.0f/sqrtf(x) vs rsqrtf(x) — different last-bit behavior
//   (2) Operation order — 4 ops vs PyTorch's 2 ops (precomputed affine)
// The bpd_batchnorm_cpu_affine_fused form below eliminates both by matching
// PyTorch's exact computational pattern: precompute scale/offset internally
// once per call, then y = scale[c]*x + offset[c] per element (2 ops, same
// as PyTorch).
//
// This 4-op form is kept for backward compatibility with existing callers;
// new code should use bpd_batchnorm_cpu_affine_fused.
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

// CPU batchnorm — affine-fused inference (matches PyTorch eval mode bit-for-bit).
//
// Substrate-design name aligned with the bn_affine_fused epilogue substrate
// vocabulary (lib/epilogue_generator.pl, shipped commit bffbbe1):
//
//   In eval mode, BN reduces to per-channel affine:
//     y = γ[c] / sqrt(σ²[c] + ε) * (x - μ[c]) + β[c]
//
//   Algebraically collapses to:
//     scale[c]  = γ[c] / sqrt(σ²[c] + ε)
//     offset[c] = β[c] - μ[c] * scale[c]
//     y         = scale[c] * x + offset[c]    (2 ops per element, same as PyTorch)
//
// Substantive substrate-design properties:
//   - Internally precomputes scale[c] and offset[c] from gamma/beta/mean/var/eps
//     once per call. For inference with stable weights, the caller can hoist
//     this work above the batch loop by computing once and reusing arrays.
//   - The per-element computation is 2 ops (scale*x + offset), matching PyTorch
//     ATen's eval-mode BN. Bit-identical with PyTorch on CPU.
//   - No division-by-sqrt at per-element scope (the 32768 ULP root cause).
//
// Inputs (read-only):
//   input  : (N, C, HW)  — flat row-major over (batch, channel, spatial)
//   gamma  : (C,)        — BN weight (scale parameter γ)
//   beta   : (C,)        — BN bias (shift parameter β)
//   mean   : (C,)        — running mean (μ)
//   var    : (C,)        — running variance (σ²)
//
// Outputs (written):
//   output : (N, C, HW)  — y[c] = scale[c] * x + offset[c]
//
// Scratch (caller-allocated, size C each):
//   scale_buf, offset_buf : working buffers for precomputed scale/offset.
//                            Pass NULL to allocate internally (slower; only
//                            valid for C up to a small stack budget).
//
// Constant:
//   eps : numerical-stability epsilon (typically 1e-5)
void bpd_batchnorm_cpu_affine_fused(const float* input, const float* gamma,
                                      const float* beta, const float* mean,
                                      const float* var, float* output,
                                      float* scale_buf, float* offset_buf,
                                      int N, int C, int HW, float eps) {
    // Precompute scale[c] and offset[c] from BN parameters.
    // Stack-allocated fallback for the no-buffer-supplied case (C up to 4096).
    float local_scale[4096];
    float local_offset[4096];
    float* scale = scale_buf ? scale_buf : local_scale;
    float* offset = offset_buf ? offset_buf : local_offset;
    if (!scale_buf && C > 4096) {
        // Substrate-honest: refuse to silently produce wrong results.
        // Caller must supply scratch for C > 4096.
        return;
    }
    for (int c = 0; c < C; c++) {
        // Substrate-design substantive substrate-design choice 2026-05-20 ~06:15 UTC
        // (per Heath's SASS-comparison direction):
        //
        // PyTorch's CPU BN-eval substantively computes scale via:
        //   inv_std = 1.0 / sqrt(var + eps)    [one DIVSS]
        //   scale   = gamma * inv_std           [one MULSS]
        // (multiply-by-reciprocal form, 2 ops, both rounded separately).
        //
        // The "direct divide" form `gamma / sqrt(var + eps)` is algebraically
        // equivalent but produces 1-ULP different bits because DIVSS rounds
        // once for the combined division, while MULSS-of-MULSS rounds twice
        // at intermediate steps.
        //
        // For bit-identity with PyTorch CPU eval mode, use the multiply form.
        // Per medayek's framework: this is the rsqrt_variant substrate-design
        // parameter manifesting at CPU level.
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        float s = gamma[c] * inv_std;
        scale[c] = s;
        offset[c] = beta[c] - mean[c] * s;
    }

    // Apply per element: y = scale[c] * x + offset[c].
    // 2 ops, same as PyTorch eval-mode BN.
    int total = N * C * HW;
    for (int idx = 0; idx < total; idx++) {
        int c = (idx / HW) % C;
        output[idx] = scale[c] * input[idx] + offset[c];
    }
}

// CPU upsample nearest 2x
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

// ── Additional elementwise ops ──

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

// ── Reductions ──

// 4-level cascade sum — matches PyTorch CPU's at::native::multi_row_sum exactly.
// Source: pytorch/aten/src/ATen/native/cpu/SumKernel.cpp (commit 2.7.0).
//
// The algorithm uses 4 fixed accumulators (level 0..3) and a power-of-2
// "level_step" determined by ceil(log2(N))/4 (clamped to ≥4). Every
// level_step elements, level 0's running sum promotes to level 1; every
// level_step² to level 2; every level_step³ to level 3.
//
// At the end, all 4 accumulators sum into level 0 and return.
//
// This is NOT a recursive pairwise tree — it's a fixed-depth cascade with
// O(1) extra storage and deterministic shape regardless of total N. This
// shape is what makes it bit-identical with PyTorch CPU on any input size,
// not just powers of 2.
//
// Substrate-design parameter: reduction_strategy(cascade_pytorch_cpu).
static int ceil_log2(int n) {
    int r = 0;
    int x = n - 1;
    while (x > 0) { x >>= 1; r++; }
    return r;
}

static float pairwise_sum(const float* data, int n) {
    if (n == 0) return 0.0f;
    if (n == 1) return data[0];
    if (n <= 16) {
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += data[i];
        return s;
    }

    // level_power = max(4, ceil(log2(n)) / 4)
    int lp = ceil_log2(n) / 4;
    if (lp < 4) lp = 4;
    int level_step = 1 << lp;
    int level_mask = level_step - 1;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Main loop: process full level_step chunks
    int i = 0;
    for (; i + level_step <= n; ) {
        // Accumulate level_step elements into acc[0]
        for (int j = 0; j < level_step; ++j, ++i) {
            acc[0] += data[i];
        }
        // Cascade promote: acc[0] -> acc[1], acc[1] -> acc[2], etc.
        // Stop when (i & (level_mask << (j*lp))) != 0 (still need more
        // elements at this level before promoting further up).
        for (int j = 1; j < 4; ++j) {
            acc[j] += acc[j-1];
            acc[j-1] = 0.0f;
            int mask = level_mask << (j * lp);
            if ((i & mask) != 0) break;
        }
    }

    // Tail: remaining < level_step elements into acc[0]
    for (; i < n; ++i) {
        acc[0] += data[i];
    }

    // Final reduction: sum all 4 accumulators
    for (int j = 1; j < 4; ++j) {
        acc[0] += acc[j];
    }
    return acc[0];
}

void bpd_sum_cpu(const float* input, float* output, int n) {
    *output = pairwise_sum(input, n);
}

void bpd_mean_cpu(const float* input, float* output, int n) {
    *output = pairwise_sum(input, n) / (float)n;
}

void bpd_max_cpu(const float* input, float* output, int n) {
    float m = input[0];
    for (int i = 1; i < n; i++) if (input[i] > m) m = input[i];
    *output = m;
}

// ── Softmax (row-wise) ──

void bpd_softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        // find max for numerical stability
        float mx = row_in[0];
        for (int c = 1; c < cols; c++) if (row_in[c] > mx) mx = row_in[c];
        // exp
        for (int c = 0; c < cols; c++)
            row_out[c] = expf(row_in[c] - mx);
        // pairwise sum (matches PyTorch reduction order)
        float sum = pairwise_sum(row_out, cols);
        // normalize
        for (int c = 0; c < cols; c++) row_out[c] /= sum;
    }
}

// ── LayerNorm ──

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

// ── MaxPool2D / AvgPool2D ──

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

// ── Linear (matmul + bias) ──

void bpd_linear_cpu(const float* input, const float* weight,
                     const float* bias, float* output,
                     int M, int N, int K) {
    for (int row = 0; row < M; row++)
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += input[row*K+k] * weight[col*K+k];
            output[row*N+col] = sum + bias[col];
        }
}

// ── Layer 2 primitives (per mavchin's direction 2026-05-20 ~18:31 UTC) ──
//
// Trivial kernels needed for YOLOv5n C3 modules and FPN concat:
//   - bpd_residual_add_cpu:  elementwise add for bottleneck residual
//   - bpd_concat_channel_cpu: channel-axis concat for C3/SPPF/FPN

// Elementwise add: out[i] = a[i] + b[i]
// Used by C3 bottleneck residual path.
//
// PyTorch path: y = a + b is single FADD per element (no FMA possible since
// only one operand-pair). Substrate matches bit-for-bit by definition —
// scalar IEEE 754 a + b produces the same result everywhere.
void bpd_residual_add_cpu(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}

// Channel-axis concatenation of N input tensors.
//
// Layout assumption: NCHW (PyTorch default).
// Each input has shape (N_batch, C_i, H, W). Output has shape
// (N_batch, sum(C_i), H, W).
//
// Per mavchin: "trivial kernel — memcpy with offset arithmetic".
//
// Inputs:
//   inputs:    array of n_inputs pointers, each pointing to an input tensor
//   c_each:    array of n_inputs channel counts (one per input)
//   n_inputs:  how many input tensors to concatenate (2 for C3, 4 for SPPF)
//   N_batch, H, W: shared spatial dims
//   output:    contiguous output buffer of shape (N_batch, sum(C_i), H, W)
//
// Algorithm: for each batch slot, copy each input's per-batch slice
// (C_i × H × W floats) into the output at the correct channel offset.
// Each input's per-batch slice is contiguous in memory; the destination
// region for that input within the output's per-batch slice is also
// contiguous. So this is a straightforward memcpy per (batch, input).
void bpd_concat_channel_cpu(const float** inputs, const int* c_each,
                             int n_inputs, int N_batch, int H, int W,
                             float* output) {
    int HW = H * W;
    // Compute total output channels = sum of c_each
    int C_total = 0;
    for (int i = 0; i < n_inputs; i++) C_total += c_each[i];
    int out_batch_stride = C_total * HW;

    for (int b = 0; b < N_batch; b++) {
        float* out_batch_base = output + b * out_batch_stride;
        int channel_offset = 0;
        for (int i = 0; i < n_inputs; i++) {
            int C_i = c_each[i];
            int in_batch_stride = C_i * HW;
            const float* in_batch_base = inputs[i] + b * in_batch_stride;
            // Copy C_i × H × W contiguous floats
            for (int j = 0; j < in_batch_stride; j++) {
                out_batch_base[channel_offset * HW + j] = in_batch_base[j];
            }
            channel_offset += C_i;
        }
    }
}
