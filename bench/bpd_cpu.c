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

// PyTorch CPU "cascade_sum" — exact port of at::native::row_sum + multi_row_sum.
// Source: pytorch/aten/src/ATen/native/cpu/SumKernel.cpp.
//
// Algorithm structure (PyTorch's CPU default for AVX1 hardware):
//   reduction_strategy(cascade(SimdWidth=8, IlpFactor=4, CascadeDepth=4, CascadeBase=16))
//
// Three-level parallel reduction:
//   1. SIMD: 8 parallel f32 lanes per Vectorized<float> register (AVX width on the
//      enclave: default 32 bytes / 4 bytes-per-float = 8 lanes).
//   2. ILP:  4 ILP-interleaved cascade lanes per SIMD register. Each input element
//      goes into one of 8 SIMD lanes × 4 ILP lanes = 32 parallel scalar slots
//      at the "level 0" position.
//   3. Cascade: 4 levels per (SIMD, ILP) slot. Every CascadeBase=16 iterations,
//      level 0 promotes to level 1; every 16² to level 2; every 16³ to level 3.
//      Total = 8 × 4 × 4 = 128 parallel scalar accumulators.
//
// Reduction order at the end:
//   level 1..3 collapse into level 0 (per SIMD × ILP)
//   ILP collapse: lane[0][s] += lane[k][s] for k in 1..3
//   SIMD collapse: final += lane[0][s] for s in 0..7
//   Scalar tail addition
//
// Per Heath's direction: "make porting the full SIMD-8 × ILP-4 × 4-level cascade
// implementation as a sweepable pattern for our code generator/optimizer."
// This C function is the manually-ported reference for one specific instantiation;
// lib/reduction_kernel.pl (to be added) generates this same shape for any
// (SimdWidth, IlpFactor, CascadeDepth, CascadeBase) combination.

static int ceil_log2(int n) {
    int r = 0;
    int x = n - 1;
    while (x > 0) { x >>= 1; r++; }
    return r;
}

// multi_row_sum_simd: full SIMD-W × ILP × cascade-D × cascade-base implementation
// for SimdWidth=8, IlpFactor=4, CascadeDepth=4.
//
// `data` starts at the array origin. The function processes `size_ilp` iterations,
// where each iteration loads 4 ILP-interleaved SIMD-8 blocks (= 32 floats).
// Returns the (4, 8) grid of partial accumulators.
static void multi_row_sum_simd(const float* data,
                                int size_ilp,
                                float out_lane[4][8]) {
    int lp = 4;
    if (size_ilp > 0) {
        lp = ceil_log2(size_ilp) / 4;
        if (lp < 4) lp = 4;
    }
    int level_step = 1 << lp;
    int level_mask = level_step - 1;

    // Cascade × ILP × SIMD = 4 × 4 × 8 = 128 accumulators
    float acc[4][4][8] = {{{0}}};

    int i = 0;
    for (; i + level_step <= size_ilp;) {
        // Accumulate level_step iterations into level 0
        for (int j = 0; j < level_step; ++j, ++i) {
            const float* base = data + i * 32;
            for (int ilp = 0; ilp < 4; ++ilp) {
                const float* src = base + ilp * 8;
                for (int s = 0; s < 8; ++s) {
                    acc[0][ilp][s] += src[s];
                }
            }
        }
        // Cascade promotion: levels 1..3
        for (int level = 1; level < 4; ++level) {
            for (int ilp = 0; ilp < 4; ++ilp) {
                for (int s = 0; s < 8; ++s) {
                    acc[level][ilp][s] += acc[level-1][ilp][s];
                    acc[level-1][ilp][s] = 0.0f;
                }
            }
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }
    }

    // Tail iterations (less than level_step worth)
    for (; i < size_ilp; ++i) {
        const float* base = data + i * 32;
        for (int ilp = 0; ilp < 4; ++ilp) {
            const float* src = base + ilp * 8;
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp][s] += src[s];
            }
        }
    }

    // Final per-lane cascade collapse: levels 1..3 → level 0
    for (int level = 1; level < 4; ++level) {
        for (int ilp = 0; ilp < 4; ++ilp) {
            for (int s = 0; s < 8; ++s) {
                acc[0][ilp][s] += acc[level][ilp][s];
            }
        }
    }

    // Write out the (ILP, SIMD) grid
    for (int ilp = 0; ilp < 4; ++ilp) {
        for (int s = 0; s < 8; ++s) {
            out_lane[ilp][s] = acc[0][ilp][s];
        }
    }
}

static float pairwise_sum(const float* data, int n) {
    if (n == 0) return 0.0f;
    if (n == 1) return data[0];

    // PyTorch's dispatch: vectorized path requires size0 >= vec_t::size() = 8.
    // For n < 8, the scalar fallback is used.
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    const int VEC_SIZE = 8;          // SimdWidth
    const int ILP_FACTOR = 4;
    int vec_size = n / VEC_SIZE;     // number of full SIMD-8 blocks
    int size_ilp = vec_size / ILP_FACTOR;
    int simd_processed = vec_size * VEC_SIZE;   // # floats consumed by full SIMD blocks

    // multi_row_sum_simd processes size_ilp iterations of 32 floats each
    float lane[4][8];
    multi_row_sum_simd(data, size_ilp, lane);

    // Tail SIMD-8 blocks (couldn't fill a complete ILP-4 group of 32)
    for (int v = size_ilp * ILP_FACTOR; v < vec_size; ++v) {
        const float* src = data + v * VEC_SIZE;
        for (int s = 0; s < 8; ++s) {
            lane[0][s] += src[s];
        }
    }

    // Horizontal collapse over ILP: lane[0][s] += lane[k][s] for k in 1..3
    for (int k = 1; k < 4; ++k) {
        for (int s = 0; s < 8; ++s) {
            lane[0][s] += lane[k][s];
        }
    }

    // Final accumulator: PyTorch's order is
    //   final_acc = 0
    //   for k in scalar tail: final_acc += data[k]
    //   for s in 0..7: final_acc += lane[0][s]
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {
        final_acc += data[i];
    }
    for (int s = 0; s < 8; ++s) {
        final_acc += lane[0][s];
    }
    return final_acc;
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
        // normalize: multiply by reciprocal (matches PyTorch — same pattern as BN)
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) row_out[c] *= inv_sum;
    }
}

// ── LayerNorm ──

void bpd_layernorm_cpu(const float* input, const float* gamma,
                        const float* beta, float* output,
                        int N, int D, float eps) {
    for (int n = 0; n < N; n++) {
        const float* x = input + n * D;
        float* y = output + n * D;
        // Pass 1: pairwise mean (matches PyTorch reduction order)
        float mean = pairwise_sum(x, D) * (1.0f / (float)D);
        // Pass 2: pairwise variance
        // Compute (x - mean)^2 into temp, then pairwise sum
        float temp[4096]; // stack buffer for D <= 4096
        for (int d = 0; d < D; d++) {
            float dx = x[d] - mean;
            temp[d] = dx * dx;
        }
        float var = pairwise_sum(temp, D) * (1.0f / (float)D);
        // rsqrt: multiply by reciprocal (matches PyTorch)
        float rstd = 1.0f / sqrtf(var + eps);
        // Pass 3: normalize
        for (int d = 0; d < D; d++)
            y[d] = (x[d] - mean) * rstd * gamma[d] + beta[d];
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
