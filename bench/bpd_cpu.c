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

// ── Tier 1 activations (Stanford L1 problems 20, 27-32) ──
//
// Each implementation mirrors the formula PyTorch uses in
// aten/src/ATen/native/cpu/Activation.cpp. These are pure elementwise
// kernels — no reduction, no SIMD-specific shuffles — so the substrate's
// scalar implementation produces bit-identical output by construction
// (one IEEE 754 operation per element matches one IEEE 754 operation per
// element regardless of whether PyTorch's vectorized path runs).

// LeakyReLU: a > 0 ? a : a * negval (default negval = 0.01)
// Source: aten/src/ATen/native/cpu/Activation.cpp:871 leaky_relu_kernel
void bpd_leaky_relu_cpu(const float* input, float* output, int n) {
    const float negval = 0.01f;
    for (int i = 0; i < n; i++) {
        float a = input[i];
        output[i] = a > 0.0f ? a : a * negval;
    }
}

// ELU: a < 0 ? expm1(a) * (alpha*scale) : a * scale
// Default: alpha=1, scale=1, input_scale=1 → simplifies to a < 0 ? expm1f(a) : a
// Source: aten/src/ATen/native/cpu/Elu.h:23 get_scalar_elu_elementwise_func
void bpd_elu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float a = input[i];
        output[i] = a < 0.0f ? expm1f(a) : a;
    }
}

// SELU: ELU with alpha=1.6732632, scale=1.0507009 (double constants
// truncated to float at the Scalar→float conversion in elu_kernel).
// Source: aten/src/ATen/native/Activation.cpp:245 SELU_ALPHA/SCALE +
//         aten/src/ATen/native/cpu/Elu.h:23 get_scalar_elu_elementwise_func
void bpd_selu_cpu(const float* input, float* output, int n) {
    // PyTorch truncates these to float when passing through Scalar::to<float>()
    const float alpha = (float)1.6732632423543772848170429916717;
    const float scale = (float)1.0507009873554804934193349852946;
    const float negcoef = alpha * scale;  // PyTorch computes this on float at runtime
    const float poscoef = scale;
    const float negiptcoef = 1.0f;  // input_scale default
    for (int i = 0; i < n; i++) {
        float a = input[i];
        output[i] = a < 0.0f ? expm1f(a * negiptcoef) * negcoef : a * poscoef;
    }
}

// HardSigmoid: min(max(x + 3, 0), 6) / 6
// Source: aten/src/ATen/native/cpu/Activation.cpp:523 hardsigmoid_kernel
void bpd_hardsigmoid_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float t = x + 3.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 6.0f) t = 6.0f;
        output[i] = t / 6.0f;
    }
}

// HardTanh / clamp: clamp(x, min, max). Default for nn.Hardtanh: min=-1, max=1.
// Source: aten/src/ATen/native/cpu/Activation.cpp (hardtanh path) — clamp is
//         exposed via the more general clamp operator.
void bpd_clamp_cpu(const float* input, float* output, int n) {
    const float min_val = -1.0f;
    const float max_val = 1.0f;
    for (int i = 0; i < n; i++) {
        float x = input[i];
        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;
        output[i] = x;
    }
}

// Softplus: a * beta > threshold ? a : log1p(exp(a * beta)) / beta
// Default: beta=1, threshold=20 (nn.Softplus default).
// Source: aten/src/ATen/native/cpu/Activation.cpp:950 softplus_kernel
void bpd_softplus_cpu(const float* input, float* output, int n) {
    const float beta = 1.0f;
    const float threshold = 20.0f;
    for (int i = 0; i < n; i++) {
        float a = input[i];
        float ab = a * beta;
        output[i] = ab > threshold ? a : log1pf(expf(ab)) / beta;
    }
}

// Softsign: x / (1 + |x|)
// Source: aten/src/ATen/native/Activation.cpp (no per-element CPU kernel —
// implemented as composite of abs, add scalar, div). Simpler to inline.
void bpd_softsign_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + fabsf(x));
    }
}

// ── Cumulative reductions (Stanford L1 problems 89-93) ──
//
// PyTorch's cumsum/cumprod on float use `at::acc_type<float, false>` = double
// as the accumulator. Each element is added/multiplied into a double accumulator
// then cast back to float on store. This raises precision throughout the chain.
//
// Source: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:79 cumsum_cpu_kernel,
//         aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:98 cumprod_cpu_kernel
//
// Substrate-design parameter: cumulative_acc_type(double).

// Cumsum: y[i] = y[i-1] + x[i], with y[-1] = 0.
// PyTorch uses double as the running accumulator.
void bpd_cumsum_cpu(const float* input, float* output, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        acc += (double)input[i];
        output[i] = (float)acc;
    }
}

// Cumprod: y[i] = y[i-1] * x[i], with y[-1] = 1.
// PyTorch uses double as the running accumulator.
void bpd_cumprod_cpu(const float* input, float* output, int n) {
    double acc = 1.0;
    for (int i = 0; i < n; i++) {
        acc *= (double)input[i];
        output[i] = (float)acc;
    }
}

// Cumsum reverse: y[i] = x[i] + x[i+1] + ... + x[n-1].
// Equivalent to: reverse → cumsum → reverse. PyTorch uses cumsum + flip.
void bpd_cumsum_reverse_cpu(const float* input, float* output, int n) {
    double acc = 0.0;
    for (int i = n - 1; i >= 0; i--) {
        acc += (double)input[i];
        output[i] = (float)acc;
    }
}

// Exclusive cumsum: y[0] = 0, y[i] = x[0] + ... + x[i-1].
// PyTorch implements this as concat([zeros(1), cumsum[:-1]]).
void bpd_cumsum_exclusive_cpu(const float* input, float* output, int n) {
    double acc = 0.0;
    output[0] = 0.0f;
    for (int i = 1; i < n; i++) {
        acc += (double)input[i - 1];
        output[i] = (float)acc;
    }
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

// PyTorch's softmax uses vec::reduce_all (linear scan with one SIMD-Vec
// accumulator), NOT the cascade sum. The cascade is only used by
// sum_kernel_impl in SumKernel.cpp. Source:
// aten/src/ATen/cpu/vec/functional_base.h:184 inline scalar_t reduce_all.
//
// Algorithm:
//   1. Load first SW=8 elements into acc_vec
//   2. For each subsequent SW=8 block: acc_vec[s] = vec_fun(acc_vec[s], data[d+s])
//   3. Tail (last (n % SW) elements) added to acc_vec via vec::set, then
//      horizontally reduced (acc_vec[0] += acc_vec[1] + ... + acc_vec[7])
//
// This is reduction_strategy(linear_scan_simd(SimdWidth=8)) — a simpler
// substrate-design parameter than cascade, but it produces different bits
// than cascade on the same data. Hence softmax doesn't use pairwise_sum.
static float linear_scan_sum_simd8(const float* data, int n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        // Scalar fallback for very small inputs (matches PyTorch's
        // vec_reduce_all path when size < Vec::size()).
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }

    // Load first SW=8 elements
    float acc[8];
    for (int s = 0; s < 8; ++s) acc[s] = data[s];

    // Linear scan: each SIMD block accumulated lane-wise
    int d = 8;
    int full_end = n - (n % 8);
    for (; d < full_end; d += 8) {
        for (int s = 0; s < 8; ++s) acc[s] += data[d + s];
    }

    // Tail (last n%8 elements). PyTorch uses vec::set which preserves the
    // upper lanes of acc_vec while loading partial data and adding only
    // the first (n - d) lanes. Implementation-wise this means we add only
    // data[d..n) to the first (n - d) accumulator lanes.
    int tail = n - d;
    for (int s = 0; s < tail; ++s) acc[s] += data[d + s];

    // Horizontal reduce: on AVX1 (no AVX2 acceleration), PyTorch falls
    // through to the generic vec_reduce_all path which sums lane 0
    // left-to-right: acc[0] + acc[1] + acc[2] + ... + acc[7].
    // See functional_base.h:174 vec_reduce_all slow path. The bizarre
    // SIMD shuffle this emulates ends up being equivalent to a strict
    // left-to-right scan of acc_arr[].
    float horiz = acc[0];
    for (int s = 1; s < 8; ++s) horiz += acc[s];
    return horiz;
}

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
        // PyTorch softmax uses vec::reduce_all (linear-scan SIMD-8), not cascade
        float sum = linear_scan_sum_simd8(row_out, cols);
        // normalize: multiply by reciprocal (matches PyTorch — same pattern as BN)
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) row_out[c] *= inv_sum;
    }
}

// LogSoftmax: y = x - max(x) - log(sum(exp(x - max(x))))
// Source: aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h:31
// serial_vec_log_softmax_lastdim_range
//
// Same linear-scan SIMD-8 reduction as softmax. PyTorch is careful to keep
// the operation order `x - max - log_sum` (not `x - (max + log_sum)`) to
// avoid catastrophic cancellation when max is large and log_sum is small.
void bpd_logsoftmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        // 1. max via linear-scan reduction (same as softmax)
        float mx = row_in[0];
        for (int c = 1; c < cols; c++) if (row_in[c] > mx) mx = row_in[c];
        // 2. exp(x - max) into a temp, then sum via linear-scan SIMD-8.
        //    Use the output buffer as temp (overwritten in step 4 anyway).
        for (int c = 0; c < cols; c++)
            row_out[c] = expf(row_in[c] - mx);
        float sum_exp = linear_scan_sum_simd8(row_out, cols);
        // 3. log(sum)
        float log_sum = logf(sum_exp);
        // 4. output = x - max - log_sum (in that order, per PyTorch source
        //    note about avoiding cancellation between max and log_sum)
        for (int c = 0; c < cols; c++)
            row_out[c] = row_in[c] - mx - log_sum;
    }
}

// ── LayerNorm ──

// Welford-with-cascade rowwise moments matching PyTorch's
// at::native::RowwiseMomentsImpl exactly.
//
// Source: pytorch/aten/src/ATen/native/cpu/moments_utils.h
//
// Algorithm: SIMD-8 Welford inside chunks of kChunkSize=16 SIMD-vectors,
// then pairwise stack-merge with mask-based promotion (same cascade
// pattern as bpd_sum_cpu, but for 3-tuple (m0, m1, m2) updates).
//
// For D=128 (the test shape): n=16 SIMD-vectors, m=1 chunk, depth=0.
// One UpdateMomentsVec on 16 SIMD-8 iterations + AddMoments horizontal
// reduce across 8 SIMD lanes.
//
// Numerical stability comes from:
//   - Welford recurrence (avoids catastrophic cancellation of sum(x²) - mean²)
//   - Cascade merge across chunks (avoids accumulation drift over long arrays)
//
// Substrate-design parameter: rowwise_moments_strategy(welford_simd8_cascade16).

static int ceil_log2_lm(int n) {
    if (n <= 1) return 0;
    int r = 0; int x = n - 1;
    while (x > 0) { x >>= 1; r++; }
    return r;
}

// AddMoments — Welford parallel combination of (m0_a, m1_a, m2_a) and
// (m0_b, m1_b, m2_b) into (*m0, *m1, *m2). Mirrors moments_utils.h:18
// AddMoments<T> exactly.
static void add_moments(int m0_add, float m1_add, float m2_add,
                         int* m0, float* m1, float* m2) {
    int n = *m0 + m0_add;
    float c = (n == 0) ? 0.0f : (float)m0_add / (float)n;
    float delta = m1_add - *m1;
    *m1 += c * delta;
    *m2 += m2_add + delta * delta * c * (float)(*m0);
    *m0 = n;
}

// rowwise_moments — returns (mean, variance) for one row of D floats,
// matching PyTorch's RowwiseMoments<float>(X, N).
//
// For D=128: kVecSize=8, n=16, m=1, depth=0.
// One UpdateMomentsVec on 16 SIMD-8 chunks + horizontal AddMoments across
// 8 SIMD lanes.
static void rowwise_moments(const float* X, int N,
                              float* out_mean, float* out_var) {
    const int kVecSize = 8;
    const int kChunkSize = 16;

    int n = N / kVecSize;
    int m = (n + kChunkSize - 1) / kChunkSize;  // divup
    int depth = ceil_log2_lm(m);

    // Stack: depth levels × 8 SIMD lanes
    // For typical depth ≤ 32 we use a fixed-size stack array.
    enum { kMaxDepth = 32 };
    int   m0_stk[kMaxDepth] = {0};
    float m1_stk[kMaxDepth][8] = {{0}};
    float m2_stk[kMaxDepth][8] = {{0}};

    // c_vecs: per-iteration constants 1/(j+1) for j in [0..kChunkSize)
    float c_consts[16];
    for (int j = 0; j < kChunkSize; ++j) {
        c_consts[j] = 1.0f / (float)(j + 1);
    }

    for (int i = 0; i < m; ++i) {
        const float* X_ptr = X + i * kChunkSize * kVecSize;
        int m0_local = kChunkSize;
        int remain = n - i * kChunkSize;
        if (remain < kChunkSize) m0_local = remain;

        // UpdateMomentsVec: SIMD-8 Welford over m0_local iterations.
        // Each lane s in 0..7 is independent.
        float m1_vec[8] = {0};
        float m2_vec[8] = {0};
        for (int j = 0; j < m0_local; ++j) {
            float c = c_consts[j];
            for (int s = 0; s < 8; ++s) {
                float x = X_ptr[j * 8 + s];
                float delta = x - m1_vec[s];
                // m1 = fmadd(c, delta, m1)  →  m1 += c * delta (no-FMA on AVX1)
                m1_vec[s] = m1_vec[s] + c * delta;
                float delta2 = x - m1_vec[s];
                // m2 = fmadd(delta, delta2, m2)  →  m2 += delta * delta2
                m2_vec[s] = m2_vec[s] + delta * delta2;
            }
        }

        // AddMomentsVec: merge the per-chunk (m0_local, m1_vec[8], m2_vec[8])
        // into stk[0] using vector AddMoments semantics.
        // The AddMomentsVec from PyTorch does the same scalar update applied
        // to each of the 8 SIMD lanes — equivalent to running add_moments per
        // lane with the SAME m0 value (m0_local), and updating m0_stk only once.
        {
            int old_m0 = m0_stk[0];
            int new_m0 = old_m0 + m0_local;
            float c_vec = (new_m0 == 0) ? 0.0f : (float)m0_local / (float)new_m0;
            for (int s = 0; s < 8; ++s) {
                float delta = m1_vec[s] - m1_stk[0][s];
                m1_stk[0][s] += c_vec * delta;
                m2_stk[0][s] += m2_vec[s] + delta * delta * c_vec * (float)old_m0;
            }
            m0_stk[0] = new_m0;
        }

        // Cascade stack-merge: when chunk index (i+1) has trailing zeros at
        // depth j, promote stk[j-1] → stk[j].
        int mask = i + 1;
        for (int j = 1; j < depth && (mask & 1) == 0; ++j) {
            int old_m0_j = m0_stk[j];
            int add_m0 = m0_stk[j - 1];
            int new_m0_j = old_m0_j + add_m0;
            float c_vec = (new_m0_j == 0) ? 0.0f : (float)add_m0 / (float)new_m0_j;
            for (int s = 0; s < 8; ++s) {
                float delta = m1_stk[j-1][s] - m1_stk[j][s];
                m1_stk[j][s] += c_vec * delta;
                m2_stk[j][s] += m2_stk[j-1][s] + delta * delta * c_vec * (float)old_m0_j;
            }
            m0_stk[j] = new_m0_j;
            m0_stk[j-1] = 0;
            for (int s = 0; s < 8; ++s) {
                m1_stk[j-1][s] = 0.0f;
                m2_stk[j-1][s] = 0.0f;
            }
            mask >>= 1;
        }
    }

    // Scalar tail (last N % kVecSize elements) — uses scalar Welford
    int m0 = 0; float m1 = 0; float m2 = 0;
    for (int i = n * kVecSize; i < N; ++i) {
        float x = X[i];
        float delta = x - m1;
        ++m0;
        m1 += delta / (float)m0;
        m2 += delta * (x - m1);
    }

    // Merge stack levels [1..depth) into stk[0]
    for (int j = 1; j < depth; ++j) {
        int old_m0_0 = m0_stk[0];
        int add_m0 = m0_stk[j];
        int new_m0_0 = old_m0_0 + add_m0;
        float c_vec = (new_m0_0 == 0) ? 0.0f : (float)add_m0 / (float)new_m0_0;
        for (int s = 0; s < 8; ++s) {
            float delta = m1_stk[j][s] - m1_stk[0][s];
            m1_stk[0][s] += c_vec * delta;
            m2_stk[0][s] += m2_stk[j][s] + delta * delta * c_vec * (float)old_m0_0;
        }
        m0_stk[0] = new_m0_0;
    }

    // Horizontal AddMoments across the 8 SIMD lanes of stk[0] into the scalar
    // (m0, m1, m2). PyTorch source:
    //   int64_t m0_add = n * kVecSize / kAccVecSize;
    // For same-precision T=float: kVecSize=8, kAccVecSize=8 → m0_add = n.
    // Each lane represents n elements (lane s processes data[s], data[s+8],
    // data[s+16], ..., data[s+(n-1)*8] — n strided values).
    int m0_add_per_lane = n;  // each lane saw n elements
    for (int s = 0; s < 8; ++s) {
        add_moments(m0_add_per_lane, m1_stk[0][s], m2_stk[0][s], &m0, &m1, &m2);
    }

    *out_mean = m1;
    *out_var = m2 / (float)N;  // ddof = 0
}

void bpd_layernorm_cpu(const float* input, const float* gamma,
                        const float* beta, float* output,
                        int N, int D, float eps) {
    for (int n = 0; n < N; n++) {
        const float* x = input + n * D;
        float* y = output + n * D;
        // Welford rowwise moments matching PyTorch CPU exactly
        float mean, var;
        rowwise_moments(x, D, &mean, &var);
        // rstd via reciprocal_sqrt variant (matches PyTorch CPU + bpd_default)
        float rstd = 1.0f / sqrtf(var + eps);
        // Normalize and apply affine (gamma, beta)
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
