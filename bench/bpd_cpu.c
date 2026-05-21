#include <math.h>
#include <stdlib.h>
#include <string.h>

// AVX1 intrinsics — Phase 3.GEMM SIMD vectorization on Ivy Bridge and later.
// Guarded so non-AVX1 builds can still compile (substrate-design portability).
#if defined(__AVX__)
#include <immintrin.h>
#define BPD_HAVE_AVX1 1
#else
#define BPD_HAVE_AVX1 0
#endif

// CPU matmul: C[M,N] = A[M,K] @ B[K,N]
//
// Implements Goto's blocked GEMM algorithm matching OpenBLAS Sandybridge SGEMM
// bit-for-bit. PyTorch CPU calls cblas_sgemm directly; on AVX1 (Tesla P4 enclave),
// OpenBLAS dispatches to the SANDYBRIDGE sgemm_kernel_16x4 with these parameters:
//
//   gemm_tile_strategy(P=768, Q=384, UM=16, UN=4)
//
// Per OpenBLAS driver/level3/level3.c:309-322, the K block size adapts to the
// remaining work:
//   while remaining > 0:
//     if remaining >= 2*Q:    min_l = Q             # full block
//     elif remaining > Q:     min_l = ceil(rem/2/UM)*UM   # half, rounded to UM
//     else:                   min_l = remaining     # tail
//
// For (M=N=16, K=4096): 9 blocks of K=384 + 2 blocks of K=320 = 11 K-blocks.
//
// Inner accumulation per (i,j): sum_{k in block} A[i,k]*B[k,j] (sequential).
// Cross-block: C[i,j] += block_partial (left-fold across blocks).
//
// Empirically verified 0 ULP vs cblas_sgemm at K ∈ {256, 512, 768, 1024, 2048, 4096}
// and 5 seeds (see /tmp/mm_goto.c + /tmp/test_goto.py).
//
// Substrate-design parameters this kernel realizes (named in
// lib/implementation_matches.pl as platform_param/2 facts):
//   gemm_tile_strategy(goto_sandy)
//   gemm_p(768)
//   gemm_q(384)
//   gemm_unroll_m(16)
//   gemm_unroll_n(4)
// Forward declaration so the dispatcher in bpd_mm_cpu can tail-call AVX1.
void bpd_mm_cpu_avx1(const float* A, const float* B, float* C,
                      int M, int N, int K);

void bpd_mm_cpu(const float* A, const float* B, float* C,
                int M, int N, int K) {
    // ── Runtime dispatch ──
    // SUBSTRATE_AVX1_GEMM env var selects AVX1 path. Default: '1' (enabled)
    // when BPD_HAVE_AVX1 is true. Set SUBSTRATE_AVX1_GEMM=0 to force scalar
    // (Tier 1.5 reference path).
    //
    // The choice is cached in a static int after first call to avoid getenv()
    // on every GEMM (called thousands of times per YOLO frame).
    static int dispatch_choice = -1;  // -1=uninit, 0=scalar, 1=avx1
    if (dispatch_choice == -1) {
        const char* env = getenv("SUBSTRATE_AVX1_GEMM");
        if (env && env[0] == '0') {
            dispatch_choice = 0;
        } else {
#if BPD_HAVE_AVX1
            dispatch_choice = 1;
#else
            dispatch_choice = 0;
#endif
        }
    }
    if (dispatch_choice == 1) {
        bpd_mm_cpu_avx1(A, B, C, M, N, K);
        return;
    }

    // ── Scalar K-block GEMM (Tier 1.5 reference, bit-identical with PyTorch CBLAS) ──
    const int Q = 384;
    const int UM = 16;

    // Init C to zero
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // K-block loop matching OpenBLAS level3.c
    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) {
            min_l = Q;
        } else if (rem > Q) {
            min_l = ((rem / 2 + UM - 1) / UM) * UM;
        } else {
            min_l = rem;
        }

        // Inner: per (i, j) compute the K-block partial, add to running C[i,j].
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float partial = 0.0f;
                for (int k = ls; k < ls + min_l; k++) {
                    partial += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] += partial;
            }
        }
        ls += min_l;
    }
}

// ──────────────────────────────────────────────────────────────────────
// AVX1-vectorized matmul (Phase 3.GEMM)
// ──────────────────────────────────────────────────────────────────────
//
// Same K-block algorithm as bpd_mm_cpu, but the inner (row, col) loop is
// vectorized across cols using AVX1 256-bit SIMD (8 floats per lane).
//
// Bit-identity preservation:
//   Each per-col accumulator does EXACTLY the same scalar sequence of operations
//   as bpd_mm_cpu: partial += A[row*K+k] * B[k*N+col] for k in [ls, ls+min_l).
//   The only difference is that 8 cols proceed in parallel SIMD lanes, with
//   each lane's accumulator independent. Per-lane IEEE arithmetic is identical
//   to the scalar version, so:
//     partial_vec[lane] == scalar_partial(col = col_base + lane)
//   for all lanes. The horizontal store back to C just writes 8 contiguous
//   float values — no cross-lane operations that could change rounding.
//
// Layout: A is (M, K) row-major; B is (K, N) row-major.
//   For fixed (row, k): A[row, k] is a single float (broadcast to all 8 lanes).
//   For 8 consecutive cols: B[k, col_base..col_base+7] are 8 contiguous floats
//   that can be loaded with a single _mm256_loadu_ps.
//
// AVX1 has no FMA, so we emit mul + add separately (matching gcc -O2 scalar code).
//
// Tail handling: if N % 8 != 0, the remaining cols use the scalar fallback
// with the IDENTICAL accumulation order, preserving bit-identity for non-8-aligned N.
#if BPD_HAVE_AVX1
void bpd_mm_cpu_avx1(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    const int Q = 384;
    const int UM = 16;

    // Init C to zero (matches bpd_mm_cpu)
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) {
            min_l = Q;
        } else if (rem > Q) {
            min_l = ((rem / 2 + UM - 1) / UM) * UM;
        } else {
            min_l = rem;
        }

        // Vectorized inner loop: process 8 cols at a time per row.
        int n_simd = N & ~7;  // largest multiple of 8 <= N
        for (int row = 0; row < M; row++) {
            const float* a_row = A + row * K;
            float* c_row = C + row * N;

            // SIMD-8 path: 8 cols in parallel per inner k iteration
            for (int col = 0; col < n_simd; col += 8) {
                __m256 partial_vec = _mm256_setzero_ps();
                for (int k = ls; k < ls + min_l; k++) {
                    // Broadcast A[row, k] to all 8 lanes
                    __m256 a_bk = _mm256_set1_ps(a_row[k]);
                    // Load B[k, col..col+7] (8 contiguous floats)
                    __m256 b_kj = _mm256_loadu_ps(B + k * N + col);
                    // partial_vec += a_bk * b_kj (mul + add, no FMA)
                    __m256 prod = _mm256_mul_ps(a_bk, b_kj);
                    partial_vec = _mm256_add_ps(partial_vec, prod);
                }
                // Add this K-block's partial to the running C accumulator.
                // C[row, col..col+7] += partial_vec
                __m256 c_vec = _mm256_loadu_ps(c_row + col);
                c_vec = _mm256_add_ps(c_vec, partial_vec);
                _mm256_storeu_ps(c_row + col, c_vec);
            }

            // Scalar tail: cols [n_simd, N) — matches bpd_mm_cpu exactly
            for (int col = n_simd; col < N; col++) {
                float partial = 0.0f;
                for (int k = ls; k < ls + min_l; k++) {
                    partial += a_row[k] * B[k * N + col];
                }
                c_row[col] += partial;
            }
        }
        ls += min_l;
    }
}
#else
// AVX1 not available at compile time — fall back to scalar bpd_mm_cpu.
void bpd_mm_cpu_avx1(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    bpd_mm_cpu(A, B, C, M, N, K);
}
#endif

// ──────────────────────────────────────────────────────────────────────
// bpd_mm_cpu_avx1_v2 — CAT-scan-informed GEMM (Phase 3.CAT.a)
// ──────────────────────────────────────────────────────────────────────
//
// Based on CAT-scan disassembly of OpenBLAS sgemm_kernel_SANDYBRIDGE.
// Foundational memory: c101e652. Substrate-design discipline: 7b297878.
//
// Substrate-design parameters baked in (deduced from OpenBLAS):
//   register_blocking(MR=4, NR=16)  — 4 rows × 16 cols per inner iteration
//                                     = 8 ymm accumulators (4 rows × 2 col-vectors of 8 floats)
//   ilp_accumulators(8)             — 8 INDEPENDENT (row, col_group) accumulators
//   unroll_factor_K(4)              — 4 k-values per inner loop body
//
// BIT-IDENTITY PRESERVATION (the substantive substrate-design Essence):
//   Each ymm accumulator holds ONE (row, col_group) output cell's running sum.
//   Within each accumulator's k-loop, the reduction order is LINEAR LEFT-TO-RIGHT:
//     acc[row, col_group] += A[row, k] * B[k, col_group]  for k = 0, 1, 2, ..., K-1
//   This is EXACTLY the same scalar order as bpd_mm_cpu and bpd_mm_cpu_avx1.
//   No tree reduction. No partial-sum interleaving. No fancy math.
//
//   The 8 accumulators run in parallel across 8 DIFFERENT output cells, not
//   across 8 partial sums of the SAME output cell. This is option (a) per
//   Medayek's analysis: bit-safe by construction.
//
// TILING:
//   M is processed in blocks of MR=4 rows. Tail rows (M % 4) handled by
//   bpd_mm_cpu_avx1 scalar-SIMD fallback (one row at a time).
//   N is processed in blocks of NR=16 cols. Tail cols (N % 16) handled by
//   the v1 path for that subset of cols.
//   K is processed in K-blocks of size Q=384 (same as bpd_mm_cpu), to match
//   the partial-sum semantics: each K-block adds to C, allowing cumulative
//   accumulation across K-blocks bit-identically with bpd_mm_cpu.
//
// LIMITATION (deliberate, simple to verify):
//   For shapes where M < 4 or N < 16, falls back to bpd_mm_cpu_avx1 (the
//   single-accumulator v1 path). The v2 path activates only for large-enough
//   tiles. This keeps the code clear and the bit-identity gate trivial.
#if BPD_HAVE_AVX1
void bpd_mm_cpu_avx1_v2(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    const int MR = 4;       // register-block height (rows)
    const int NR = 16;      // register-block width (cols = 2 ymm)
    const int KU = 4;       // K-unroll factor
    const int Q  = 384;     // K-block size (matches bpd_mm_cpu)

    // Init C to zero — same as scalar/v1
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // If shape too small for v2 register blocking, defer to v1 entirely.
    int M_blocks = M / MR;       // # of full 4-row blocks
    int M_tail   = M - M_blocks * MR;
    int N_blocks = N / NR;       // # of full 16-col blocks
    int N_tail   = N - N_blocks * NR;

    int ls = 0;
    while (ls < K) {
        int rem = K - ls;
        int min_l;
        if (rem >= 2 * Q) {
            min_l = Q;
        } else if (rem > Q) {
            min_l = ((rem / 2 + MR*4 - 1) / (MR*4)) * (MR*4);
        } else {
            min_l = rem;
        }

        // K-block end index (matches bpd_mm_cpu's partial-sum semantics)
        int k_end = ls + min_l;

        // K-unroll: only valid when min_l is a multiple of KU. If not, use kus = 1.
        int kus = (min_l % KU == 0) ? KU : 1;

        // ─── Main path: 4-row × 16-col register-blocked tiles ───
        for (int rb = 0; rb < M_blocks; rb++) {
            int row_base = rb * MR;
            const float* a0 = A + (row_base + 0) * K;
            const float* a1 = A + (row_base + 1) * K;
            const float* a2 = A + (row_base + 2) * K;
            const float* a3 = A + (row_base + 3) * K;
            float* c0 = C + (row_base + 0) * N;
            float* c1 = C + (row_base + 1) * N;
            float* c2 = C + (row_base + 2) * N;
            float* c3 = C + (row_base + 3) * N;

            for (int cb = 0; cb < N_blocks; cb++) {
                int col_base = cb * NR;
                // 8 accumulators: 4 rows × 2 col-vectors per row
                // acc_r{0..3}_c{0,1} where c0 covers cols [col_base, col_base+8)
                // and c1 covers cols [col_base+8, col_base+16)
                __m256 acc_r0_c0 = _mm256_setzero_ps();
                __m256 acc_r0_c1 = _mm256_setzero_ps();
                __m256 acc_r1_c0 = _mm256_setzero_ps();
                __m256 acc_r1_c1 = _mm256_setzero_ps();
                __m256 acc_r2_c0 = _mm256_setzero_ps();
                __m256 acc_r2_c1 = _mm256_setzero_ps();
                __m256 acc_r3_c0 = _mm256_setzero_ps();
                __m256 acc_r3_c1 = _mm256_setzero_ps();

                if (kus == KU) {
                    // K-unrolled inner loop: process 4 k-values per iteration
                    for (int k = ls; k < k_end; k += KU) {
                        // Per k step: load B's two col-vectors at row k
                        // For each row, broadcast A[row, k] and accumulate
                        #define KSTEP(KOFF) do {                                              \
                            __m256 b0 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base);     \
                            __m256 b1 = _mm256_loadu_ps(B + (k + (KOFF)) * N + col_base + 8); \
                            __m256 av0 = _mm256_set1_ps(a0[k + (KOFF)]);                      \
                            __m256 av1 = _mm256_set1_ps(a1[k + (KOFF)]);                      \
                            __m256 av2 = _mm256_set1_ps(a2[k + (KOFF)]);                      \
                            __m256 av3 = _mm256_set1_ps(a3[k + (KOFF)]);                      \
                            acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));     \
                            acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));     \
                            acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));     \
                            acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));     \
                            acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));     \
                            acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));     \
                            acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));     \
                            acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));     \
                        } while (0)
                        KSTEP(0);
                        KSTEP(1);
                        KSTEP(2);
                        KSTEP(3);
                        #undef KSTEP
                    }
                } else {
                    // Non-unrolled fallback (when min_l isn't a multiple of KU)
                    for (int k = ls; k < k_end; k++) {
                        __m256 b0 = _mm256_loadu_ps(B + k * N + col_base);
                        __m256 b1 = _mm256_loadu_ps(B + k * N + col_base + 8);
                        __m256 av0 = _mm256_set1_ps(a0[k]);
                        __m256 av1 = _mm256_set1_ps(a1[k]);
                        __m256 av2 = _mm256_set1_ps(a2[k]);
                        __m256 av3 = _mm256_set1_ps(a3[k]);
                        acc_r0_c0 = _mm256_add_ps(acc_r0_c0, _mm256_mul_ps(av0, b0));
                        acc_r0_c1 = _mm256_add_ps(acc_r0_c1, _mm256_mul_ps(av0, b1));
                        acc_r1_c0 = _mm256_add_ps(acc_r1_c0, _mm256_mul_ps(av1, b0));
                        acc_r1_c1 = _mm256_add_ps(acc_r1_c1, _mm256_mul_ps(av1, b1));
                        acc_r2_c0 = _mm256_add_ps(acc_r2_c0, _mm256_mul_ps(av2, b0));
                        acc_r2_c1 = _mm256_add_ps(acc_r2_c1, _mm256_mul_ps(av2, b1));
                        acc_r3_c0 = _mm256_add_ps(acc_r3_c0, _mm256_mul_ps(av3, b0));
                        acc_r3_c1 = _mm256_add_ps(acc_r3_c1, _mm256_mul_ps(av3, b1));
                    }
                }

                // Add this K-block's partial sums to C (matching bpd_mm_cpu's
                // K-block cumulative semantics). C[row, col] += partial.
                __m256 c_r0_c0 = _mm256_loadu_ps(c0 + col_base);
                __m256 c_r0_c1 = _mm256_loadu_ps(c0 + col_base + 8);
                _mm256_storeu_ps(c0 + col_base,     _mm256_add_ps(c_r0_c0, acc_r0_c0));
                _mm256_storeu_ps(c0 + col_base + 8, _mm256_add_ps(c_r0_c1, acc_r0_c1));
                __m256 c_r1_c0 = _mm256_loadu_ps(c1 + col_base);
                __m256 c_r1_c1 = _mm256_loadu_ps(c1 + col_base + 8);
                _mm256_storeu_ps(c1 + col_base,     _mm256_add_ps(c_r1_c0, acc_r1_c0));
                _mm256_storeu_ps(c1 + col_base + 8, _mm256_add_ps(c_r1_c1, acc_r1_c1));
                __m256 c_r2_c0 = _mm256_loadu_ps(c2 + col_base);
                __m256 c_r2_c1 = _mm256_loadu_ps(c2 + col_base + 8);
                _mm256_storeu_ps(c2 + col_base,     _mm256_add_ps(c_r2_c0, acc_r2_c0));
                _mm256_storeu_ps(c2 + col_base + 8, _mm256_add_ps(c_r2_c1, acc_r2_c1));
                __m256 c_r3_c0 = _mm256_loadu_ps(c3 + col_base);
                __m256 c_r3_c1 = _mm256_loadu_ps(c3 + col_base + 8);
                _mm256_storeu_ps(c3 + col_base,     _mm256_add_ps(c_r3_c0, acc_r3_c0));
                _mm256_storeu_ps(c3 + col_base + 8, _mm256_add_ps(c_r3_c1, acc_r3_c1));
            }

            // N-tail: cols [N_blocks * NR, N) for these 4 rows.
            // Use the same per-row scalar k-loop as bpd_mm_cpu to preserve
            // bit-identity. Each tail col handled independently.
            if (N_tail > 0) {
                int col_start = N_blocks * NR;
                for (int row = 0; row < MR; row++) {
                    const float* a_row = A + (row_base + row) * K;
                    float* c_row = C + (row_base + row) * N;
                    for (int col = col_start; col < N; col++) {
                        float partial = 0.0f;
                        for (int k = ls; k < k_end; k++) {
                            partial += a_row[k] * B[k * N + col];
                        }
                        c_row[col] += partial;
                    }
                }
            }
        }

        // M-tail: rows [M_blocks * MR, M) processed scalar (matches bpd_mm_cpu order).
        if (M_tail > 0) {
            int row_start = M_blocks * MR;
            for (int row = row_start; row < M; row++) {
                const float* a_row = A + row * K;
                float* c_row = C + row * N;
                for (int col = 0; col < N; col++) {
                    float partial = 0.0f;
                    for (int k = ls; k < k_end; k++) {
                        partial += a_row[k] * B[k * N + col];
                    }
                    c_row[col] += partial;
                }
            }
        }

        ls += min_l;
    }
}
#else
void bpd_mm_cpu_avx1_v2(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    bpd_mm_cpu(A, B, C, M, N, K);
}
#endif

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

// Im2col helper: convert NCHW input slice into [Cin*kH*kW, H_out*W_out] row-major.
// Matches PyTorch's im2col.h template signature exactly.
// data_col[(c_col * H_out + h_col) * W_out + w_col] = data_im[(c_im * H + h_im) * W + w_im]
// where c_col indexes (c_im, h_offset, w_offset) in row-major (c_im outermost).
static void bpd_im2col(const float* data_im,
                       int channels, int height, int width,
                       int output_height, int output_width,
                       int kernel_h, int kernel_w,
                       int pad_h, int pad_w,
                       int stride_h, int stride_w,
                       int dilation_h, int dilation_w,
                       float* data_col) {
    int channels_col = channels * kernel_h * kernel_w;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / (kernel_h * kernel_w);
        for (int h_col = 0; h_col < output_height; h_col++) {
            int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
            for (int w_col = 0; w_col < output_width; w_col++) {
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                int dst = (c_col * output_height + h_col) * output_width + w_col;
                if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                    data_col[dst] = data_im[(c_im * height + h_im) * width + w_im];
                } else {
                    data_col[dst] = 0.0f;
                }
            }
        }
    }
}

// Parameterized 2D convolution: matches PyTorch CPU F.conv2d exactly via
// im2col + GEMM. Inherits bit-identity from bpd_mm_cpu (Goto-Sandy SGEMM
// matching cblas_sgemm 0 ULP).
//
// PyTorch source: aten/src/ATen/native/ConvolutionMM2d.cpp slow_conv2d_forward_cpu
// + slow_conv2d_update_output_frame. Im2col layout from im2col.h line 65.
//
// Signature: output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
//   input:  (N, Cin, H, W)
//   weight: (Cout, Cin/groups, kH, kW)
//   bias:   (Cout,) or NULL
//   output: (N, Cout, H_out, W_out)
void bpd_conv2d_full_cpu(const float* input, const float* weight, const float* bias,
                          float* output,
                          int N, int Cin, int H, int W,
                          int Cout, int kH, int kW,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dilation_h, int dilation_w,
                          int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - dilation_w*(kW-1) - 1) / stride_w + 1;

    int spatial_out = H_out * W_out;
    int k_dim = Cin_per_group * kH * kW;

    float* finput = (float*)malloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            const float* input_g = input + (n * Cin + g * Cin_per_group) * H * W;
            bpd_im2col(input_g, Cin_per_group, H, W,
                       H_out, W_out, kH, kW,
                       pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w,
                       finput);

            const float* weight_g = weight + g * Cout_per_group * k_dim;
            float* output_g = output + (n * Cout + g * Cout_per_group) * spatial_out;

            // GEMM: output_g[Cout_per_group, spatial_out] = weight_g[Cout_per_group, k_dim] @ finput[k_dim, spatial_out]
            bpd_mm_cpu(weight_g, finput, output_g,
                       Cout_per_group, spatial_out, k_dim);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * spatial_out;
                    for (int p = 0; p < spatial_out; p++) {
                        out_co[p] += b;
                    }
                }
            }
        }
    }

    free(finput);
}

// ──────────────────────────────────────────────────────────────────────
// Conv2d + BatchNorm + SiLU fused (Phase 3.1 F3 — bit-identical with PyTorch)
// ──────────────────────────────────────────────────────────────────────
//
// Computes (in one kernel, eliminating two intermediate tensors per call):
//   y[co, p] = silu(alpha[co] * GEMM(weight, im2col(input))[co, p] + beta[co])
//
// Where alpha and beta are precomputed from BN parameters via the same
// substrate-design choice as bpd_batchnorm_cpu_affine_fused:
//   alpha[c] = gamma[c] * (1.0f / sqrtf(var[c] + eps))   ← multiply-by-reciprocal
//   beta[c]  = bn_beta[c] - mean[c] * alpha[c]
//
// And silu uses the DIVSS form (same as bpd_silu_cpu):
//   silu(x) = x / (1.0f + expf(-x))
//
// Because the GEMM accumulator, the alpha/beta application order, and the
// silu expression are all IDENTICAL to the unfused chain
// (bpd_conv2d_full_cpu → bpd_batchnorm_cpu_affine_fused → bpd_silu_cpu),
// the fused output is bit-identical with the unfused output for all inputs.
//
// Restriction: groups=1 only (YOLOv5n uses groups=1 throughout).
// Restriction: no bias on conv (YOLOv5n CBS uses bias=False on conv; BN provides
//              the additive bias via offset).
//
// Memory traffic savings per call (vs unfused chain):
//   - No intermediate conv_out tensor materialized to memory
//   - No intermediate bn_out tensor materialized to memory
//   - Only the final silu_out is written
//   = 4 fewer memory passes over the (N, Cout, H_out, W_out) tensor.
void bpd_conv2d_bn_silu_fused_cpu(const float* input, const float* weight,
                                    const float* alpha, const float* beta,
                                    float* output,
                                    int N, int Cin, int H, int W,
                                    int Cout, int kH, int kW,
                                    int stride_h, int stride_w,
                                    int pad_h, int pad_w) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    float* finput = (float*)malloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        bpd_im2col(input_n, Cin, H, W,
                   H_out, W_out, kH, kW,
                   pad_h, pad_w, stride_h, stride_w,
                   1, 1,  // dilation=1
                   finput);

        float* output_n = output + n * Cout * spatial_out;

        // GEMM: output_n[Cout, spatial_out] = weight[Cout, k_dim] @ finput[k_dim, spatial_out]
        // The GEMM writes the raw accumulator into output_n; we then rewrite output_n
        // with the silu(alpha*acc + beta) epilogue.
        bpd_mm_cpu(weight, finput, output_n,
                   Cout, spatial_out, k_dim);

        // Epilogue: y[co, p] = silu(alpha[co] * y[co, p] + beta[co])
        // Per-channel alpha/beta; per-element transform.
        // SiLU uses DIVSS form: x / (1.0f + expf(-x)) — matches bpd_silu_cpu exactly.
        for (int co = 0; co < Cout; co++) {
            float a = alpha[co];
            float b = beta[co];
            float* out_co = output_n + co * spatial_out;
            for (int p = 0; p < spatial_out; p++) {
                float x = a * out_co[p] + b;
                out_co[p] = x / (1.0f + expf(-x));
            }
        }
    }

    free(finput);
}

// ──────────────────────────────────────────────────────────────────────
// Conv2d + BatchNorm + SiLU + Residual Add fused (Phase 3.4 F4)
// ──────────────────────────────────────────────────────────────────────
//
// Identical to bpd_conv2d_bn_silu_fused_cpu (F3) except for one more
// epilogue op: y = silu(alpha*acc + beta) + residual[same position].
//
// Used in YOLOv5 bottleneck blocks with shortcut=True:
//   y = x + cv2(cv1(x))
// where cv2 is a CBS unit. The fused kernel computes cv2's full pipeline
// (im2col -> GEMM -> silu(alpha*x + beta)) and adds the residual `x` in
// the same write-back.
//
// Restriction: residual must have shape (N, Cout, H_out, W_out) \u2014 same
// layout as the conv output. The caller (run_bottleneck) ensures this by
// passing the bottleneck input `x` directly (which has the same shape as
// the cv2 output when cin==cout, k=3, stride=1, pad=1, which is the
// standard YOLOv5 shortcut bottleneck configuration).
//
// Bit-identity:
//   F3 path produces silu(alpha*acc + beta) bit-identically.
//   Adding `+ residual[p]` is a single float ADD performed AFTER silu, in
//   the same order as the unfused chain: y = cv2_silu_out + x.
//   This matches bpd_residual_add_cpu's per-element behavior bit-for-bit.
//
// Memory traffic savings vs unfused:
//   Unfused: F3 writes cv2_out; residual_add reads cv2_out + x and writes y
//     = F3 write (1) + residual_add reads (2) + residual_add write (1) = 4 passes
//   Fused:   F3+add writes y directly (residual read on-the-fly via cache)
//     = 1 write of y output (residual read is 1 pass through x, but that read
//        is sequential and likely cache-resident from cv1)
//   Net: 2-3 fewer memory passes per shortcut bottleneck.
void bpd_conv2d_bn_silu_add_fused_cpu(const float* input, const float* weight,
                                        const float* alpha, const float* beta,
                                        const float* residual,
                                        float* output,
                                        int N, int Cin, int H, int W,
                                        int Cout, int kH, int kW,
                                        int stride_h, int stride_w,
                                        int pad_h, int pad_w) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    float* finput = (float*)malloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        const float* residual_n = residual + n * Cout * spatial_out;
        bpd_im2col(input_n, Cin, H, W,
                   H_out, W_out, kH, kW,
                   pad_h, pad_w, stride_h, stride_w,
                   1, 1, finput);

        float* output_n = output + n * Cout * spatial_out;
        bpd_mm_cpu(weight, finput, output_n, Cout, spatial_out, k_dim);

        // Epilogue: y[co, p] = silu(alpha[co] * y[co, p] + beta[co]) + residual[co, p]
        // Same scalar order as the unfused chain (F3 epilogue followed by
        // a residual_add element-wise add).
        for (int co = 0; co < Cout; co++) {
            float a = alpha[co];
            float b = beta[co];
            float* out_co = output_n + co * spatial_out;
            const float* res_co = residual_n + co * spatial_out;
            for (int p = 0; p < spatial_out; p++) {
                float x = a * out_co[p] + b;
                float silu_val = x / (1.0f + expf(-x));
                out_co[p] = silu_val + res_co[p];
            }
        }
    }

    free(finput);
}

// ──────────────────────────────────────────────────────────────────────
// Conv2d + Bias + Sigmoid fused (Phase 3.5 F7)
// ──────────────────────────────────────────────────────────────────────
//
// Computes: y = sigmoid(GEMM(weight, im2col(x)) + bias[co])
//
// Used in YOLOv5 Detect head: the 3 detection convs are 1x1 conv with bias
// and NO BN. The fused F7 kernel performs the conv (im2col + GEMM + bias)
// followed by sigmoid in the epilogue, eliminating one memory pass over
// the conv output.
//
// Substrate-design parameter family demonstrated by F7:
//   conv_epilogue(scalar_add_bias) + activation(sigmoid_divss)
// distinct from F3's (precomputed_alpha_beta_bn) + (silu_divss),
// and distinct from F4's (precomputed_alpha_beta_bn) + (silu_divss) + (residual_add).
//
// Bit-identity preservation:
//   Conv: same Goto-Sandy K-block GEMM, same im2col, same per-channel bias add
//   as bpd_conv2d_full_cpu.
//   Sigmoid: 1.0f / (1.0f + expf(-x)) — identical DIVSS form to bpd_sigmoid_cpu.
//   Per-element scalar order matches: (GEMM_result + bias[co]) -> sigmoid.
//
// Restriction: groups=1, dilation=1 (sufficient for YOLOv5n Detect).
void bpd_conv2d_bias_sigmoid_fused_cpu(const float* input, const float* weight,
                                         const float* bias, float* output,
                                         int N, int Cin, int H, int W,
                                         int Cout, int kH, int kW,
                                         int stride_h, int stride_w,
                                         int pad_h, int pad_w) {
    int H_out = (H + 2*pad_h - (kH-1) - 1) / stride_h + 1;
    int W_out = (W + 2*pad_w - (kW-1) - 1) / stride_w + 1;
    int spatial_out = H_out * W_out;
    int k_dim = Cin * kH * kW;

    float* finput = (float*)malloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        const float* input_n = input + n * Cin * H * W;
        bpd_im2col(input_n, Cin, H, W,
                   H_out, W_out, kH, kW,
                   pad_h, pad_w, stride_h, stride_w,
                   1, 1, finput);

        float* output_n = output + n * Cout * spatial_out;
        bpd_mm_cpu(weight, finput, output_n, Cout, spatial_out, k_dim);

        // Epilogue: y[co, p] = sigmoid(GEMM_result + bias[co])
        // Same scalar order as the unfused chain: add bias, then sigmoid.
        for (int co = 0; co < Cout; co++) {
            float b = bias[co];
            float* out_co = output_n + co * spatial_out;
            for (int p = 0; p < spatial_out; p++) {
                float x = out_co[p] + b;
                out_co[p] = 1.0f / (1.0f + expf(-x));
            }
        }
    }

    free(finput);
}

// ── 1D and 3D convolutions (im2col + GEMM, same pattern as 2D) ──

// 1D im2col: input (channels, L) → packed (channels * kL, L_out) row-major
// data_col[(c_col * L_out) + l_col] = data_im[(c_im * L + l_im)]
// where c_col indexes (c_im, l_offset) row-major.
static void bpd_im2col_1d(const float* data_im,
                          int channels, int length,
                          int output_length,
                          int kernel_l, int pad_l, int stride_l, int dilation_l,
                          float* data_col) {
    int channels_col = channels * kernel_l;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int l_offset = c_col % kernel_l;
        int c_im = c_col / kernel_l;
        for (int l_col = 0; l_col < output_length; l_col++) {
            int l_im = l_col * stride_l - pad_l + l_offset * dilation_l;
            int dst = c_col * output_length + l_col;
            if (l_im >= 0 && l_im < length) {
                data_col[dst] = data_im[c_im * length + l_im];
            } else {
                data_col[dst] = 0.0f;
            }
        }
    }
}

// 1D convolution via im2col + GEMM.
// Signature: F.conv1d(input, weight, bias, stride, padding, dilation, groups)
//   input:  (N, Cin, L)
//   weight: (Cout, Cin/groups, kL)
//   bias:   (Cout,) or NULL
//   output: (N, Cout, L_out)
void bpd_conv1d_full_cpu(const float* input, const float* weight, const float* bias,
                          float* output,
                          int N, int Cin, int L,
                          int Cout, int kL,
                          int stride_l, int pad_l, int dilation_l,
                          int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int L_out = (L + 2*pad_l - dilation_l*(kL-1) - 1) / stride_l + 1;
    int k_dim = Cin_per_group * kL;

    float* finput = (float*)malloc(k_dim * L_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            const float* input_g = input + (n * Cin + g * Cin_per_group) * L;
            bpd_im2col_1d(input_g, Cin_per_group, L,
                          L_out, kL, pad_l, stride_l, dilation_l,
                          finput);
            const float* weight_g = weight + g * Cout_per_group * k_dim;
            float* output_g = output + (n * Cout + g * Cout_per_group) * L_out;

            bpd_mm_cpu(weight_g, finput, output_g,
                       Cout_per_group, L_out, k_dim);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * L_out;
                    for (int p = 0; p < L_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(finput);
}

// 3D im2col: input (channels, D, H, W) → packed (channels * kD * kH * kW, D_out * H_out * W_out)
// data_col[(c_col * D_out * H_out * W_out) + (d_col * H_out * W_out) + (h_col * W_out) + w_col]
//   = data_im[(c_im * D * H * W) + (d_im * H * W) + (h_im * W) + w_im]
// where c_col indexes (c_im, d_offset, h_offset, w_offset) row-major (c_im outermost,
// w_offset innermost) — matches PyTorch's im2col_3d_kernel pattern.
static void bpd_im2col_3d(const float* data_im,
                          int channels, int depth, int height, int width,
                          int output_depth, int output_height, int output_width,
                          int kernel_d, int kernel_h, int kernel_w,
                          int pad_d, int pad_h, int pad_w,
                          int stride_d, int stride_h, int stride_w,
                          int dilation_d, int dilation_h, int dilation_w,
                          float* data_col) {
    int channels_col = channels * kernel_d * kernel_h * kernel_w;
    int dhw = output_depth * output_height * output_width;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int d_offset = (c_col / (kernel_w * kernel_h)) % kernel_d;
        int c_im = c_col / (kernel_d * kernel_h * kernel_w);
        for (int d_col = 0; d_col < output_depth; d_col++) {
            int d_im = d_col * stride_d - pad_d + d_offset * dilation_d;
            for (int h_col = 0; h_col < output_height; h_col++) {
                int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                for (int w_col = 0; w_col < output_width; w_col++) {
                    int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                    int dst = c_col * dhw
                            + d_col * output_height * output_width
                            + h_col * output_width
                            + w_col;
                    if (d_im >= 0 && d_im < depth
                        && h_im >= 0 && h_im < height
                        && w_im >= 0 && w_im < width) {
                        int src = c_im * depth * height * width
                                + d_im * height * width
                                + h_im * width
                                + w_im;
                        data_col[dst] = data_im[src];
                    } else {
                        data_col[dst] = 0.0f;
                    }
                }
            }
        }
    }
}

// 3D convolution via im2col + GEMM.
// Signature: F.conv3d(input, weight, bias, stride, padding, dilation, groups)
//   input:  (N, Cin, D, H, W)
//   weight: (Cout, Cin/groups, kD, kH, kW)
//   bias:   (Cout,) or NULL
//   output: (N, Cout, D_out, H_out, W_out)
void bpd_conv3d_full_cpu(const float* input, const float* weight, const float* bias,
                          float* output,
                          int N, int Cin, int D, int H, int W,
                          int Cout, int kD, int kH, int kW,
                          int sd, int sh, int sw,
                          int pd, int ph, int pw,
                          int dd, int dh, int dw,
                          int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int D_out = (D + 2*pd - dd*(kD-1) - 1) / sd + 1;
    int H_out = (H + 2*ph - dh*(kH-1) - 1) / sh + 1;
    int W_out = (W + 2*pw - dw*(kW-1) - 1) / sw + 1;

    int spatial_out = D_out * H_out * W_out;
    int k_dim = Cin_per_group * kD * kH * kW;

    float* finput = (float*)malloc(k_dim * spatial_out * sizeof(float));
    if (!finput) return;

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            const float* input_g = input + (n * Cin + g * Cin_per_group) * D * H * W;
            bpd_im2col_3d(input_g, Cin_per_group, D, H, W,
                          D_out, H_out, W_out,
                          kD, kH, kW, pd, ph, pw,
                          sd, sh, sw, dd, dh, dw,
                          finput);
            const float* weight_g = weight + g * Cout_per_group * k_dim;
            float* output_g = output + (n * Cout + g * Cout_per_group) * spatial_out;

            bpd_mm_cpu(weight_g, finput, output_g,
                       Cout_per_group, spatial_out, k_dim);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * spatial_out;
                    for (int p = 0; p < spatial_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(finput);
}

// ── Transposed convolutions (col2im + GEMM, mirror of forward conv) ──
//
// PyTorch source: aten/src/ATen/native/NaiveConvolutionTranspose2d.cpp
//   slow_conv_transpose2d_out_cpu_template (line 244)
//
// Algorithm:
//   1. GEMM: columns[Cout*kH*kW, H_in*W_in] = weight^T @ input
//      where weight has PyTorch shape (Cin, Cout/groups, kH, kW)
//      reshaped to (Cin, Cout*kH*kW), then transposed → (Cout*kH*kW, Cin)
//   2. col2im: scatter columns into output[Cout, H_out, W_out] with += accumulation
//
// Output dims: H_out = (H_in - 1)*stride - 2*pad + dilation*(kH-1) + output_padding + 1
//
// col2im layout matches im2col exactly but scatters instead of gathers:
//   data_im[(c_im * H_out + h_im) * W_out + w_im]
//     += data_col[(c_col * H_in + h_in) * W_in + w_in]
// where c_col indexes (c_im, h_offset, w_offset) row-major.

// col2im 2D: scatter columns back into spatial image, accumulating overlaps.
static void bpd_col2im(const float* data_col,
                       int channels, int height_out, int width_out,
                       int height_in, int width_in,
                       int kernel_h, int kernel_w,
                       int pad_h, int pad_w,
                       int stride_h, int stride_w,
                       int dilation_h, int dilation_w,
                       float* data_im) {
    // Zero-init output
    int total = channels * height_out * width_out;
    for (int i = 0; i < total; i++) data_im[i] = 0.0f;

    int channels_col = channels * kernel_h * kernel_w;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / (kernel_h * kernel_w);
        for (int h_col = 0; h_col < height_in; h_col++) {
            int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
            for (int w_col = 0; w_col < width_in; w_col++) {
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                if (h_im >= 0 && h_im < height_out && w_im >= 0 && w_im < width_out) {
                    data_im[(c_im * height_out + h_im) * width_out + w_im] +=
                        data_col[(c_col * height_in + h_col) * width_in + w_col];
                }
            }
        }
    }
}

// 2D ConvTranspose via GEMM + col2im.
// Signature: F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
//   input:  (N, Cin, H_in, W_in)
//   weight: (Cin, Cout/groups, kH, kW)  <- NOTE: Cin is the FIRST dim for ConvTranspose
//   bias:   (Cout,) or NULL
//   output: (N, Cout, H_out, W_out)
//
// Output shape:
//   H_out = (H_in - 1)*sh - 2*ph + dh*(kH-1) + oph + 1
//   W_out = (W_in - 1)*sw - 2*pw + dw*(kW-1) + opw + 1
void bpd_conv_transpose2d_full_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int N, int Cin, int H_in, int W_in,
                                    int Cout, int kH, int kW,
                                    int sh, int sw, int ph, int pw,
                                    int oph, int opw,
                                    int dh, int dw,
                                    int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int H_out = (H_in - 1) * sh - 2*ph + dh*(kH-1) + oph + 1;
    int W_out = (W_in - 1) * sw - 2*pw + dw*(kW-1) + opw + 1;

    int spatial_in = H_in * W_in;
    int k_dim = Cout_per_group * kH * kW;

    // Buffer for transposed weight slice (per-group): shape (Cout_per_group*kH*kW, Cin_per_group)
    float* weight_T = (float*)malloc(k_dim * Cin_per_group * sizeof(float));
    // Buffer for columns: shape (Cout_per_group*kH*kW, H_in*W_in)
    float* columns = (float*)malloc(k_dim * spatial_in * sizeof(float));
    if (!weight_T || !columns) {
        if (weight_T) free(weight_T);
        if (columns) free(columns);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            // Transpose weight slice for this group.
            // PyTorch weight layout (per group): [Cin_per_group, Cout_per_group, kH, kW] row-major
            //   weight[(ci * Cout_per_group + co) * kH * kW + kh*kW + kw]
            // We want weight_T[Cout_per_group*kH*kW, Cin_per_group] row-major:
            //   weight_T[(co * kH * kW + kh*kW + kw) * Cin_per_group + ci]
            //     = weight[(ci * Cout_per_group + co) * kH * kW + kh*kW + kw]
            const float* weight_g = weight + g * Cin_per_group * k_dim;
            for (int ci = 0; ci < Cin_per_group; ci++) {
                for (int co = 0; co < Cout_per_group; co++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int src = (ci * Cout_per_group + co) * kH * kW + kh*kW + kw;
                            int dst = ((co * kH + kh) * kW + kw) * Cin_per_group + ci;
                            weight_T[dst] = weight_g[src];
                        }
                    }
                }
            }

            // input slice for this group
            const float* input_g = input + (n * Cin + g * Cin_per_group) * spatial_in;

            // GEMM: columns[k_dim, spatial_in] = weight_T[k_dim, Cin_per_group] @ input_g[Cin_per_group, spatial_in]
            // bpd_mm_cpu(A, B, C, M, N, K): C[M,N] = A[M,K] @ B[K,N]
            bpd_mm_cpu(weight_T, input_g, columns,
                       k_dim, spatial_in, Cin_per_group);

            // col2im: scatter columns into output[Cout_per_group, H_out, W_out] for this group
            float* output_g = output + (n * Cout + g * Cout_per_group) * H_out * W_out;
            bpd_col2im(columns, Cout_per_group, H_out, W_out,
                       H_in, W_in, kH, kW, ph, pw, sh, sw, dh, dw,
                       output_g);

            // Add bias if provided
            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * H_out * W_out;
                    for (int p = 0; p < H_out * W_out; p++) out_co[p] += b;
                }
            }
        }
    }

    free(weight_T);
    free(columns);
}

// ── 1D ConvTranspose ──

// col2im 1D: scatter columns back into spatial image, accumulating overlaps.
static void bpd_col2im_1d(const float* data_col,
                          int channels, int length_out, int length_in,
                          int kernel_l, int pad_l, int stride_l, int dilation_l,
                          float* data_im) {
    int total = channels * length_out;
    for (int i = 0; i < total; i++) data_im[i] = 0.0f;
    int channels_col = channels * kernel_l;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int l_offset = c_col % kernel_l;
        int c_im = c_col / kernel_l;
        for (int l_col = 0; l_col < length_in; l_col++) {
            int l_im = l_col * stride_l - pad_l + l_offset * dilation_l;
            if (l_im >= 0 && l_im < length_out) {
                data_im[c_im * length_out + l_im] += data_col[c_col * length_in + l_col];
            }
        }
    }
}

// 1D ConvTranspose via GEMM + col2im.
// input:  (N, Cin, L_in)
// weight: (Cin, Cout/groups, kL)
// output: (N, Cout, L_out)  where L_out = (L_in-1)*stride - 2*pad + dilation*(kL-1) + output_padding + 1
void bpd_conv_transpose1d_full_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int N, int Cin, int L_in,
                                    int Cout, int kL,
                                    int stride_l, int pad_l, int output_pad_l,
                                    int dilation_l, int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int L_out = (L_in - 1) * stride_l - 2*pad_l + dilation_l*(kL-1) + output_pad_l + 1;
    int k_dim = Cout_per_group * kL;

    float* weight_T = (float*)malloc(k_dim * Cin_per_group * sizeof(float));
    float* columns = (float*)malloc(k_dim * L_in * sizeof(float));
    if (!weight_T || !columns) {
        if (weight_T) free(weight_T);
        if (columns) free(columns);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            // Transpose weight slice: src=(Cin_per_group, Cout_per_group, kL), dst=(Cout_per_group*kL, Cin_per_group)
            const float* weight_g = weight + g * Cin_per_group * k_dim;
            for (int ci = 0; ci < Cin_per_group; ci++) {
                for (int co = 0; co < Cout_per_group; co++) {
                    for (int kl = 0; kl < kL; kl++) {
                        int src = (ci * Cout_per_group + co) * kL + kl;
                        int dst = (co * kL + kl) * Cin_per_group + ci;
                        weight_T[dst] = weight_g[src];
                    }
                }
            }
            const float* input_g = input + (n * Cin + g * Cin_per_group) * L_in;
            // columns[k_dim, L_in] = weight_T[k_dim, Cin_per_group] @ input_g[Cin_per_group, L_in]
            bpd_mm_cpu(weight_T, input_g, columns, k_dim, L_in, Cin_per_group);

            float* output_g = output + (n * Cout + g * Cout_per_group) * L_out;
            bpd_col2im_1d(columns, Cout_per_group, L_out, L_in, kL,
                          pad_l, stride_l, dilation_l, output_g);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * L_out;
                    for (int p = 0; p < L_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(weight_T);
    free(columns);
}

// ── 3D ConvTranspose ──

// col2im 3D: scatter columns back into spatial image, accumulating overlaps.
static void bpd_col2im_3d(const float* data_col,
                          int channels, int D_out, int H_out, int W_out,
                          int D_in, int H_in, int W_in,
                          int kD, int kH, int kW,
                          int pd, int ph, int pw,
                          int sd, int sh, int sw,
                          int dd, int dh, int dw,
                          float* data_im) {
    int total = channels * D_out * H_out * W_out;
    for (int i = 0; i < total; i++) data_im[i] = 0.0f;

    int channels_col = channels * kD * kH * kW;
    int spatial_in = D_in * H_in * W_in;
    for (int c_col = 0; c_col < channels_col; c_col++) {
        int w_offset = c_col % kW;
        int h_offset = (c_col / kW) % kH;
        int d_offset = (c_col / (kW * kH)) % kD;
        int c_im = c_col / (kD * kH * kW);
        for (int d_col = 0; d_col < D_in; d_col++) {
            int d_im = d_col * sd - pd + d_offset * dd;
            for (int h_col = 0; h_col < H_in; h_col++) {
                int h_im = h_col * sh - ph + h_offset * dh;
                for (int w_col = 0; w_col < W_in; w_col++) {
                    int w_im = w_col * sw - pw + w_offset * dw;
                    if (d_im >= 0 && d_im < D_out
                        && h_im >= 0 && h_im < H_out
                        && w_im >= 0 && w_im < W_out) {
                        int dst = c_im * D_out * H_out * W_out
                                + d_im * H_out * W_out
                                + h_im * W_out
                                + w_im;
                        int src = c_col * spatial_in
                                + d_col * H_in * W_in
                                + h_col * W_in
                                + w_col;
                        data_im[dst] += data_col[src];
                    }
                }
            }
        }
    }
}

// 3D ConvTranspose via GEMM + col2im.
// input:  (N, Cin, D_in, H_in, W_in)
// weight: (Cin, Cout/groups, kD, kH, kW)
// output: (N, Cout, D_out, H_out, W_out)
void bpd_conv_transpose3d_full_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int N, int Cin, int D_in, int H_in, int W_in,
                                    int Cout, int kD, int kH, int kW,
                                    int sd, int sh, int sw,
                                    int pd, int ph, int pw,
                                    int opd, int oph, int opw,
                                    int dd, int dh, int dw,
                                    int groups) {
    int Cin_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int D_out = (D_in - 1) * sd - 2*pd + dd*(kD-1) + opd + 1;
    int H_out = (H_in - 1) * sh - 2*ph + dh*(kH-1) + oph + 1;
    int W_out = (W_in - 1) * sw - 2*pw + dw*(kW-1) + opw + 1;

    int spatial_in = D_in * H_in * W_in;
    int k_dim = Cout_per_group * kD * kH * kW;

    float* weight_T = (float*)malloc(k_dim * Cin_per_group * sizeof(float));
    float* columns = (float*)malloc(k_dim * spatial_in * sizeof(float));
    if (!weight_T || !columns) {
        if (weight_T) free(weight_T);
        if (columns) free(columns);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < groups; g++) {
            // Transpose weight slice: (Cin_per_group, Cout_per_group, kD, kH, kW) → (Cout_per_group*kD*kH*kW, Cin_per_group)
            const float* weight_g = weight + g * Cin_per_group * k_dim;
            for (int ci = 0; ci < Cin_per_group; ci++) {
                for (int co = 0; co < Cout_per_group; co++) {
                    for (int kd = 0; kd < kD; kd++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int src = ((ci * Cout_per_group + co) * kD + kd) * kH * kW + kh*kW + kw;
                                int dst = (((co * kD + kd) * kH + kh) * kW + kw) * Cin_per_group + ci;
                                weight_T[dst] = weight_g[src];
                            }
                        }
                    }
                }
            }
            const float* input_g = input + (n * Cin + g * Cin_per_group) * spatial_in;
            bpd_mm_cpu(weight_T, input_g, columns, k_dim, spatial_in, Cin_per_group);

            float* output_g = output + (n * Cout + g * Cout_per_group) * D_out * H_out * W_out;
            bpd_col2im_3d(columns, Cout_per_group, D_out, H_out, W_out,
                          D_in, H_in, W_in, kD, kH, kW,
                          pd, ph, pw, sd, sh, sw, dd, dh, dw,
                          output_g);

            if (bias != NULL) {
                for (int co = 0; co < Cout_per_group; co++) {
                    float b = bias[g * Cout_per_group + co];
                    float* out_co = output_g + co * D_out * H_out * W_out;
                    for (int p = 0; p < D_out * H_out * W_out; p++) out_co[p] += b;
                }
            }
        }
    }
    free(weight_T);
    free(columns);
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

// ──────────────────────────────────────────────────────────────────────
// Detect head post-sigmoid fused kernel (Phase 3.2 F8)
// ──────────────────────────────────────────────────────────────────────
//
// Fuses the sigmoid + split + scale + concat sequence in the YOLOv5 Detect
// head into a single sweep over the (bs, na, ny, nx, no) tensor.
//
// Computes, per element (n, a, y, x, c):
//   s = sigmoid(permuted[n,a,y,x,c]) = 1.0f / (1.0f + expf(-permuted[n,a,y,x,c]))
//   if c < 2:           out = (s * 2.0f + grid[0,a,y,x,c]) * stride
//   else if c < 4:      d = s * 2.0f;  out = d * d * anchor_grid[0,a,y,x,c-2]
//   else:               out = s
//
// Bit-identity preservation:
//   - sigmoid: same x / (1.0f + expf(-x)) expression as bpd_sigmoid_cpu
//             but using DIVSS form. Actually bpd_sigmoid_cpu uses 1.0f/(1+exp(-x)),
//             and the unfused detect path multiplies by 2.0f then add grid.
//             Same scalar order in fused kernel.
//   - xy:   s = sigmoid(in); s2 = s * 2.0f; (s2 + grid) * stride  \u2014 same order
//             as unfused: xy*2 \u2192 +grid \u2192 *stride.
//   - wh:   d = sigmoid(in) * 2.0f; d*d * anchor_grid  \u2014 same order as
//             unfused: wh*2 \u2192 squared \u2192 *anchor_grid.
//   - conf: s = sigmoid(in)  \u2014 trivial pass-through.
//
// Grid and anchor_grid layouts (matching _make_grid_yolov5 in yolo_forward.py):
//   grid shape:        (1, na, ny, nx, 2)  with values stack(xv, yv) - 0.5
//   anchor_grid shape: (1, na, ny, nx, 2)  with values anchors[i] * stride[i]
// Both contiguous float32. Reading grid[a, y, x, c] for c in {0, 1}:
//   offset = a*ny*nx*2 + y*nx*2 + x*2 + c
//
// Memory traffic savings vs unfused:
//   Unfused: sigmoid writes whole tensor (R+W), xy*2 writes (R+W), +grid writes
//   (R+W), *stride writes (R+W), wh*2 (R+W), squared (R+W), *anchor_grid (R+W),
//   concatenate writes whole tensor (R+W). At minimum 4-5 R+W of the
//   (bs, na, ny, nx, no) tensor = 8-10 memory passes eliminated.
//   Fused: 1 R + 1 W = 2 memory passes total. Net saving: 6-8 passes per
//   detection level over the full tensor.
void bpd_detect_postprocess_cpu(const float* permuted, const float* grid,
                                  const float* anchor_grid, float stride,
                                  float* output,
                                  int bs, int na, int ny, int nx, int no) {
    int n_per_anchor = ny * nx * no;
    int grid_per_anchor = ny * nx * 2;
    for (int b = 0; b < bs; b++) {
        for (int a = 0; a < na; a++) {
            const float* in_a = permuted + b * (na * n_per_anchor) + a * n_per_anchor;
            float* out_a = output + b * (na * n_per_anchor) + a * n_per_anchor;
            const float* grid_a = grid + a * grid_per_anchor;
            const float* anchor_a = anchor_grid + a * grid_per_anchor;
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    const float* in_yx = in_a + y * (nx * no) + x * no;
                    float* out_yx = out_a + y * (nx * no) + x * no;
                    const float* grid_yx = grid_a + y * (nx * 2) + x * 2;
                    const float* anchor_yx = anchor_a + y * (nx * 2) + x * 2;
                    // Last axis: 0..1 = xy, 2..3 = wh, 4..no-1 = conf
                    // xy: out[c] = (sigmoid(in[c]) * 2.0f + grid_yx[c]) * stride
                    for (int c = 0; c < 2; c++) {
                        float s = 1.0f / (1.0f + expf(-in_yx[c]));
                        out_yx[c] = (s * 2.0f + grid_yx[c]) * stride;
                    }
                    // wh: d = sigmoid(in[c]) * 2.0f; out[c] = d*d * anchor_yx[c-2]
                    for (int c = 2; c < 4; c++) {
                        float s = 1.0f / (1.0f + expf(-in_yx[c]));
                        float d = s * 2.0f;
                        out_yx[c] = d * d * anchor_yx[c - 2];
                    }
                    // conf: out[c] = sigmoid(in[c])
                    for (int c = 4; c < no; c++) {
                        float s = 1.0f / (1.0f + expf(-in_yx[c]));
                        out_yx[c] = s;
                    }
                }
            }
        }
    }
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

// Per substrate-design correspondence map docs/substrate-design-correspondence.md:
// the binary_kernel_reduce_lastdim path PyTorch uses for torch.norm(p=2, dim=-1)
// is SUBSTANTIVELY DIFFERENT from cascade_sum. It uses a single SIMD-8 accumulator
// in a linear pass, then linear horizontal reduce, then tail scalar.
// Source: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:227 norm_kernel_tensor_iterator_impl
//
// Substrate-design parameter: norm_reduction_strategy(binary_kernel_reduce_lastdim_simd8).
// Distinct from reduction_strategy(cascade(8,4,4,16)) which is for torch.sum.
//
// Returns sum(x[d]^2) over d in [0, n) using PyTorch's exact algorithm.
static float bpd_norm_p2_sumsq_lastdim(const float* x, int n) {
    #define VEC 8
    float acc_vec[VEC] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int d = 0;
    // SIMD-8 loop (norm_two_reduce_step: acc += data * data per lane)
    int simd_end = n - (n % VEC);
    for (; d < simd_end; d += VEC) {
        for (int j = 0; j < VEC; j++) {
            float v = x[d + j];
            acc_vec[j] += v * v;
        }
    }
    // Horizontal reduce: linear sum of the 8 lanes
    float buf = acc_vec[0];
    for (int j = 1; j < VEC; j++) {
        buf = buf + acc_vec[j];
    }
    // Scalar tail
    for (; d < n; d++) {
        float v = x[d];
        buf = buf + v * v;
    }
    #undef VEC
    return buf;
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
    // EMPIRICAL FINDING 2026-05-21: PyTorch's vectorized_reduction source code
    // says pairwise (vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]))), but the
    // emitted code at AVX1 for our shapes is LINEAR ILP combine. We tested
    // pairwise here and BIT_IDENTICAL dropped 93→92 (47 Sum_reduction and 38
    // L1Norm flipped to DIVERGENT). Linear combine matches the actual emitted
    // code. Substrate-design parameter: ilp_combine_strategy(linear_simd8).
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

// ── Normalization family (Stanford L1 problems 34-39) ──
//
// The norm family shares two substrate-design choices:
//   1. Welford rowwise_moments() — for mean+var moments (LayerNorm, InstanceNorm, GroupNorm)
//   2. pairwise_sum (cascade) — for sum-of-squares (RMSNorm, Frobenius, L1Norm, L2Norm)
//   3. rsqrt_variant(reciprocal_sqrt) — same as LayerNorm and BatchNorm
//
// Each kernel below mirrors the algorithm PyTorch CPU uses. The L1 tests
// validate at smaller shapes than the model's deployment shapes; the
// per-(n, slice) algorithm is the same.

// InstanceNorm2D: per-(batch, channel) Welford normalization over spatial dims.
// PyTorch source: aten/src/ATen/native/Normalization.cpp:727 instance_norm
// — composite that reshapes (B,C,H,W) to (1, B*C, H, W) and calls batch_norm.
// In training mode (KernelBench default: affine=False, track_running_stats=False),
// batch_norm:
//   1. Collects stats via batch_norm_cpu_collect_stats_contiguous_impl:
//      naive two-pass sum + var_sum (accscalar_t = float for fp32, NOT double).
//   2. Computes invstd = 1/sqrt(var + eps) per channel.
//   3. Applies via precomputed-scale-offset form:
//        alpha[c] = invstd * weight     (weight=1 for InstanceNorm no-affine)
//        beta[c]  = bias - mean * alpha (bias=0 for InstanceNorm no-affine)
//        output(p) = input(p) * alpha[c] + beta[c]
//
// This is the SAME substrate-design choice as BatchNorm: bn_mode(precomputed_scale_offset).
// (x - mean) * invstd and x*alpha + beta are algebraically equivalent but BIT-DIFFERENT.
//
// Source:
//   aten/src/ATen/native/cpu/batch_norm_kernel.cpp:31 batch_norm_cpu_collect_linear_and_constant_terms
//   aten/src/ATen/native/cpu/batch_norm_kernel.cpp:177 batch_norm_cpu_collect_stats_contiguous_impl
void bpd_instancenorm_cpu(const float* input, float* output,
                           int N, int C, int H, int W, float eps) {
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float* x = input + (n * C + c) * spatial;
            float* y = output + (n * C + c) * spatial;
            // PyTorch source: batch_norm_cpu_collect_stats_contiguous_impl
            //   acc_type<float, false> = double on CPU. We use double accumulators.
            // Source: aten/src/ATen/native/cpu/batch_norm_kernel.cpp:177
            double sum = 0.0;
            for (int p = 0; p < spatial; p++) sum += (double)x[p];
            float mean = (float)(sum / (double)spatial);
            double var_sum = 0.0;
            for (int p = 0; p < spatial; p++) {
                double d = (double)x[p] - (double)mean;
                var_sum += d * d;
            }
            float var = (float)(var_sum / (double)spatial);
            float invstd = 1.0f / sqrtf(var + eps);
            // EMPIRICAL 2026-05-21: At controlled inputs mean+var bit-identical
            // with PyTorch. The 3-4 ULP residual at harness RNG state comes from
            // PyTorch's SIMD-vectorized apply step at AVX1 default path emitting
            // a different SIMD chunk pattern than our scalar code at the specific
            // shape (N=2, C=4, H=W=8) used by the harness.
            //
            // Both affine_application forms tested:
            //   precomputed_alpha_beta: x * (invstd) + (-mean * invstd) — 3 ULP / 100 diffs
            //   direct_subtract_multiply: (x - mean) * invstd          — 3 ULP / 234 diffs
            // precomputed_alpha_beta has fewer divergent positions at the harness
            // shape, so we use that form. The remaining 3 ULP is at the SIMD-inner
            // microopt level (Phase B).
            float alpha = invstd;
            float bias = -mean * alpha;
            for (int p = 0; p < spatial; p++)
                y[p] = x[p] * alpha + bias;
        }
    }
}

// GroupNorm: per-(batch, group) Welford normalization over (group_features × spatial).
// PyTorch source: aten/src/ATen/native/cpu/group_norm_kernel.cpp:55 GroupNormKernelImpl
// — calls RowwiseMoments(X_ptr, inner_size) where inner_size = (C/G)*H*W.
// Then applies gamma/beta per channel.
void bpd_groupnorm_cpu(const float* input, const float* gamma,
                       const float* beta, float* output,
                       int N, int C, int H, int W, int G, float eps) {
    int channels_per_group = C / G;
    int group_size = channels_per_group * H * W;
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < G; g++) {
            const float* x_group = input + (n * C + g * channels_per_group) * spatial;
            float* y_group = output + (n * C + g * channels_per_group) * spatial;
            // Welford rowwise moments (matches PyTorch group_norm_kernel.cpp).
            float mean, var;
            rowwise_moments(x_group, group_size, &mean, &var);
            float rstd = 1.0f / sqrtf(var + eps);
            // Apply per-channel affine via PRECOMPUTED_ALPHA_BETA (matches PyTorch source):
            //   scale = rstd * gamma[c]
            //   bias  = -scale * mean + beta[c]
            //   y[k] = scale * x[k] + bias
            // Source: aten/src/ATen/native/cpu/group_norm_kernel.cpp:71-76
            for (int cc = 0; cc < channels_per_group; cc++) {
                int c = g * channels_per_group + cc;
                float scale = rstd * gamma[c];
                float bias  = -scale * mean + beta[c];
                const float* x = x_group + cc * spatial;
                float* y = y_group + cc * spatial;
                for (int p = 0; p < spatial; p++)
                    y[p] = scale * x[p] + bias;
            }
        }
    }
}

// RMSNorm: x / sqrt(mean(x²) + eps)  — applied per-row over feature dim.
// L1 test computes mean(x², dim=1, keepdim=True) — reduces along dim=1 (channels).
// For input (B, C, H, W) reduce dim=1 (C); result is (B, 1, H, W) broadcast back.
//
// SUBSTANTIVE EMPIRICAL FINDING (2026-05-20 ~23:35 UTC): PyTorch uses DIRECT
// DIVISION for the normalization step, NOT multiply-by-reciprocal. This is a
// new substrate-design parameter: norm_division_strategy(direct_division).
// Multiply-by-reciprocal introduces 1 ULP error in the reciprocal step that
// direct division avoids.
//
// Substrate-design choices for RMSNorm:
//   reduction_strategy(cascade_or_similar) — pairwise_sum over x²
//   norm_division_strategy(direct_division) — x / rms, not x * (1/rms)
void bpd_rmsnorm_cpu(const float* input, float* output,
                     int N, int C, int H, int W, float eps) {
    int spatial = H * W;
    float* temp = (float*)malloc(C * sizeof(float));
    for (int n = 0; n < N; n++) {
        for (int p = 0; p < spatial; p++) {
            // Squared values, contiguous
            for (int c = 0; c < C; c++) {
                float v = input[n * C * spatial + c * spatial + p];
                temp[c] = v * v;
            }
            // Reference uses torch.mean(x**2, dim=1) → cascade_sum / N pattern
            float sum_sq = pairwise_sum(temp, C);
            float rms = sqrtf(sum_sq / (float)C + eps);
            for (int c = 0; c < C; c++) {
                output[n * C * spatial + c * spatial + p] =
                    input[n * C * spatial + c * spatial + p] / rms;
            }
        }
    }
    free(temp);
}

// FrobeniusNorm: x / sqrt(sum(x²))  — GLOBAL reduction over all elements.
// PyTorch source: torch.norm(x, p='fro') flattens and reduces.
// Substrate-design choices: pairwise_sum over all N elements + direct_division.
void bpd_frobenius_norm_cpu(const float* input, float* output, int n_total) {
    float* temp = (float*)malloc(n_total * sizeof(float));
    for (int i = 0; i < n_total; i++) {
        float v = input[i];
        temp[i] = v * v;
    }
    float sum_sq = pairwise_sum(temp, n_total);
    float norm = sqrtf(sum_sq);
    for (int i = 0; i < n_total; i++)
        output[i] = input[i] / norm;
    free(temp);
}

// L1Norm: x / mean(|x|, dim=1, keepdim=True)  — per-row reduction along dim=1.
// For input (B, D) where the test uses dim=1: per-row sum(|x|)/D.
// Substrate-design choices: pairwise_sum (cascade) over |x| values + direct_division.
void bpd_l1norm_cpu(const float* input, float* output, int rows, int cols) {
    float* temp = (float*)malloc(cols * sizeof(float));
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        for (int c = 0; c < cols; c++)
            temp[c] = fabsf(row_in[c]);
        float sum_abs = pairwise_sum(temp, cols);
        float mean_abs = sum_abs / (float)cols;
        for (int c = 0; c < cols; c++)
            row_out[c] = row_in[c] / mean_abs;
    }
    free(temp);
}

// L2Norm: x / norm(x, p=2, dim=1, keepdim=True)  — per-row reduction along dim=1.
// L2 norm of a row = sqrt(sum(x²)).
// Substrate-design choices: pairwise_sum over x² + direct_division.
void bpd_l2norm_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;
        // Use binary_kernel_reduce_lastdim path to match PyTorch's torch.norm(p=2, dim=-1)
        // Source: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:227
        float sum_sq = bpd_norm_p2_sumsq_lastdim(row_in, cols);
        float norm = sqrtf(sum_sq);
        for (int c = 0; c < cols; c++)
            row_out[c] = row_in[c] / norm;
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

// ── Pool variants 1D and 3D (Stanford L1 problems 41, 43, 44, 46) ──

// MaxPool1D: input (N, C, L), output (N, C, L_out)
void bpd_maxpool1d_cpu(const float* input, float* output,
                       int N, int C, int L,
                       int kL, int stride, int pad) {
    int L_out = (L + 2*pad - kL) / stride + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < L_out; ol++) {
                float val = -1e30f;
                for (int kl = 0; kl < kL; kl++) {
                    int li = ol * stride - pad + kl;
                    if (li >= 0 && li < L) {
                        float v = input[(n*C+c)*L + li];
                        if (v > val) val = v;
                    }
                }
                output[(n*C+c)*L_out + ol] = val;
            }
        }
    }
}

// MaxPool3D: input (N, C, D, H, W), output (N, C, D_out, H_out, W_out)
void bpd_maxpool3d_cpu(const float* input, float* output,
                       int N, int C, int D, int H, int W,
                       int kD, int kH, int kW, int stride, int pad) {
    int D_out = (D + 2*pad - kD) / stride + 1;
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int od = 0; od < D_out; od++)
    for (int oh = 0; oh < H_out; oh++)
    for (int ow = 0; ow < W_out; ow++) {
        float val = -1e30f;
        for (int kd = 0; kd < kD; kd++)
        for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int di = od * stride - pad + kd;
            int hi = oh * stride - pad + kh;
            int wi = ow * stride - pad + kw;
            if (di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W) {
                float v = input[(((n*C+c)*D+di)*H+hi)*W+wi];
                if (v > val) val = v;
            }
        }
        output[(((n*C+c)*D_out+od)*H_out+oh)*W_out+ow] = val;
    }
}

// AvgPool1D: divisor = kL by default (count_include_pad=True is PT default).
// PyTorch's F.avg_pool1d divides by kernel_size when count_include_pad=True.
void bpd_avgpool1d_cpu(const float* input, float* output,
                       int N, int C, int L,
                       int kL, int stride, int pad) {
    int L_out = (L + 2*pad - kL) / stride + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int ol = 0; ol < L_out; ol++) {
                float sum = 0.0f;
                for (int kl = 0; kl < kL; kl++) {
                    int li = ol * stride - pad + kl;
                    if (li >= 0 && li < L) {
                        sum += input[(n*C+c)*L + li];
                    }
                }
                output[(n*C+c)*L_out + ol] = sum / (float)kL;
            }
        }
    }
}

// AvgPool3D: same — divisor = kD*kH*kW (count_include_pad=True).
void bpd_avgpool3d_cpu(const float* input, float* output,
                       int N, int C, int D, int H, int W,
                       int kD, int kH, int kW, int stride, int pad) {
    int D_out = (D + 2*pad - kD) / stride + 1;
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    float divisor = (float)(kD * kH * kW);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int od = 0; od < D_out; od++)
    for (int oh = 0; oh < H_out; oh++)
    for (int ow = 0; ow < W_out; ow++) {
        float sum = 0.0f;
        for (int kd = 0; kd < kD; kd++)
        for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            int di = od * stride - pad + kd;
            int hi = oh * stride - pad + kh;
            int wi = ow * stride - pad + kw;
            if (di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W) {
                sum += input[(((n*C+c)*D+di)*H+hi)*W+wi];
            }
        }
        output[(((n*C+c)*D_out+od)*H_out+oh)*W_out+ow] = sum / divisor;
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

// ── Loss family (Stanford L1 problems 94-100) ──
//
// Each loss reduces to: elementwise op → mean/sum reduction.
// Reductions use pairwise_sum (cascade(8,4,4,16)) matching torch.mean/torch.sum.

// MSELoss: mean((predictions - targets)²)
// Returns single scalar via output[0].
void bpd_mse_loss_cpu(const float* pred, const float* target, float* output, int n) {
    float* temp = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float d = pred[i] - target[i];
        temp[i] = d * d;
    }
    float sum = pairwise_sum(temp, n);
    output[0] = sum / (float)n;
    free(temp);
}

// HuberLoss / smooth_l1_loss: per-element: 0.5*x² if |x|<beta else beta*(|x|-0.5*beta)
// beta=1.0 by default. Then mean reduction.
// PyTorch source: F.smooth_l1_loss with reduction='mean' (default), beta=1.0
void bpd_huber_loss_cpu(const float* pred, const float* target, float* output, int n) {
    const float beta = 1.0f;
    float* temp = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float diff = pred[i] - target[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < beta) {
            temp[i] = 0.5f * diff * diff / beta;
        } else {
            temp[i] = abs_diff - 0.5f * beta;
        }
    }
    float sum = pairwise_sum(temp, n);
    output[0] = sum / (float)n;
    free(temp);
}

// HingeLoss: torch.mean(torch.clamp(1 - predictions * targets, min=0))
void bpd_hinge_loss_cpu(const float* pred, const float* target, float* output, int n) {
    float* temp = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float v = 1.0f - pred[i] * target[i];
        temp[i] = v > 0.0f ? v : 0.0f;
    }
    float sum = pairwise_sum(temp, n);
    output[0] = sum / (float)n;
    free(temp);
}

// KLDivLoss: F.kl_div(log_pred, target, reduction='batchmean')
//   per-element: target * (log(target) - log_pred) … but PyTorch's F.kl_div
//   convention is: target * (log(target) - input), where input is already log.
//   Source: torch.nn.functional.kl_div docs:
//     "input – Tensor of arbitrary shape in log-probabilities."
//     loss = target * (log(target) - input)
//     For target=0: contribution is 0 (by convention).
//   reduction='batchmean' divides by batch_size (first dim).
void bpd_kl_div_loss_cpu(const float* log_pred, const float* target,
                          float* output, int batch_size, int per_batch) {
    int n = batch_size * per_batch;
    float* temp = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float t = target[i];
        if (t > 0.0f) {
            temp[i] = t * (logf(t) - log_pred[i]);
        } else {
            temp[i] = 0.0f;
        }
    }
    float sum = pairwise_sum(temp, n);
    // 'batchmean': divide by batch_size, NOT n
    output[0] = sum / (float)batch_size;
    free(temp);
}

// CrossEntropyLoss: F.cross_entropy(predictions, targets, reduction='mean')
//   = mean over batch of: -log_softmax(predictions)[target[i]]
// predictions: (batch_size, num_classes), targets: (batch_size,) integer class indices.
//
// For numerical match with PyTorch:
//   1. compute log_softmax(predictions) per row (using linear_scan_sum_simd8)
//   2. gather log_softmax[i, targets[i]] for each batch element
//   3. negate, then mean
void bpd_cross_entropy_loss_cpu(const float* pred, const long* target,
                                  float* output, int batch_size, int num_classes) {
    float* temp = (float*)malloc(batch_size * sizeof(float));
    float* row_logsm = (float*)malloc(num_classes * sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        const float* row = pred + b * num_classes;
        // log_softmax inline: same as bpd_logsoftmax_cpu but for one row
        float mx = row[0];
        for (int c = 1; c < num_classes; c++) if (row[c] > mx) mx = row[c];
        for (int c = 0; c < num_classes; c++) row_logsm[c] = expf(row[c] - mx);
        float sum_exp = linear_scan_sum_simd8(row_logsm, num_classes);
        float log_sum = logf(sum_exp);
        // log_softmax(c) = row[c] - mx - log_sum; we only need the target column
        int t = (int)target[b];
        temp[b] = -(row[t] - mx - log_sum);
    }
    float sum = pairwise_sum(temp, batch_size);
    output[0] = sum / (float)batch_size;
    free(temp);
    free(row_logsm);
}

// TripletMarginLoss: F.triplet_margin_loss(anchor, positive, negative, margin=1, p=2)
//   per-row: max(0, ||a-p||_p - ||a-n||_p + margin)
//   reduction='mean' over batch.
// p=2 means L2 distance per row (sqrt(sum((a-p)²))).
void bpd_triplet_margin_loss_cpu(const float* anchor, const float* positive,
                                   const float* negative, float* output,
                                   int batch_size, int feat_dim, float margin) {
    float* temp = (float*)malloc(batch_size * sizeof(float));
    float* sqdiff = (float*)malloc(feat_dim * sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        const float* a = anchor + b * feat_dim;
        const float* p = positive + b * feat_dim;
        const float* nv = negative + b * feat_dim;
        // ||a - p||_2
        for (int c = 0; c < feat_dim; c++) {
            float d = a[c] - p[c];
            sqdiff[c] = d * d;
        }
        float dist_ap = sqrtf(pairwise_sum(sqdiff, feat_dim));
        // ||a - n||_2
        for (int c = 0; c < feat_dim; c++) {
            float d = a[c] - nv[c];
            sqdiff[c] = d * d;
        }
        float dist_an = sqrtf(pairwise_sum(sqdiff, feat_dim));
        float loss = dist_ap - dist_an + margin;
        temp[b] = loss > 0.0f ? loss : 0.0f;
    }
    float sum = pairwise_sum(temp, batch_size);
    output[0] = sum / (float)batch_size;
    free(temp);
    free(sqdiff);
}

// ── A.5 BMM and matrix-product variants ──

// A.5.a Matrix-scalar multiplication: out[i] = A[i] * s
void bpd_scalar_mul_cpu(const float* A, float s, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = A[i] * s;
}

// A.5.b Batched matmul (BMM): (B, M, K) @ (B, K, N) → (B, M, N)
// Each batch slice is an independent mm. Reuse bpd_mm_cpu which is bit-identical
// with cblas_sgemm (Goto-Sandy SGEMM).
void bpd_bmm_cpu(const float* A, const float* B, float* C,
                  int batch, int M, int N, int K) {
    for (int b = 0; b < batch; b++) {
        const float* a_b = A + b * M * K;
        const float* b_b = B + b * K * N;
        float* c_b = C + b * M * N;
        bpd_mm_cpu(a_b, b_b, c_b, M, N, K);
    }
}

// A.5.c 3D tensor-matrix multiplication: (B, M, K) @ (K, N) → (B, M, N)
// Single matmul if we reshape input to (B*M, K). bpd_mm_cpu handles this.
void bpd_3d_tensor_matmul_cpu(const float* A, const float* B, float* C,
                                int batch, int M, int N, int K) {
    // Treat A as (batch*M, K), output as (batch*M, N), B unchanged.
    bpd_mm_cpu(A, B, C, batch * M, N, K);
}

// A.5.d 4D tensor-matrix multiplication: (B, C_dim, M, K) @ (K, N) → (B, C_dim, M, N)
// Single matmul with reshape to (B*C_dim*M, K).
void bpd_4d_tensor_matmul_cpu(const float* A, const float* B, float* C,
                                int batch, int C_dim, int M, int N, int K) {
    bpd_mm_cpu(A, B, C, batch * C_dim * M, N, K);
}

// A.5.e Diagonal matmul: A is (M,) diagonal vector; B is (M, N); output (M, N)
// out[i, j] = A[i] * B[i, j]
void bpd_diag_matmul_cpu(const float* A_diag, const float* B, float* C,
                          int M, int N) {
    for (int i = 0; i < M; i++) {
        float a = A_diag[i];
        const float* b_row = B + i * N;
        float* c_row = C + i * N;
        for (int j = 0; j < N; j++) c_row[j] = a * b_row[j];
    }
}

// ── A.6 specialty kernels ──

// A.6.a Argmax over dim: input shape (..., dim_size, ...), output one int64 per
//   slice excluding the reduced dim. Ties: lowest index (PyTorch semantic).
// We'll handle a contiguous "outer × dim_size × inner" layout. The harness will
// reshape as needed.
//   out[outer, inner] = argmax over k in 0..dim_size of x[outer, k, inner]
void bpd_argmax_dim_cpu(const float* x, long* out,
                         int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float best = x[(o * dim_size + 0) * inner + i];
            long best_idx = 0;
            for (int k = 1; k < dim_size; k++) {
                float v = x[(o * dim_size + k) * inner + i];
                if (v > best) {
                    best = v;
                    best_idx = k;
                }
            }
            out[o * inner + i] = best_idx;
        }
    }
}

// A.6.b Argmin: mirror of argmax with <.
void bpd_argmin_dim_cpu(const float* x, long* out,
                         int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float best = x[(o * dim_size + 0) * inner + i];
            long best_idx = 0;
            for (int k = 1; k < dim_size; k++) {
                float v = x[(o * dim_size + k) * inner + i];
                if (v < best) {
                    best = v;
                    best_idx = k;
                }
            }
            out[o * inner + i] = best_idx;
        }
    }
}

// A.6.c Min reduction over dim: like argmin but returns values not indices.
void bpd_min_dim_cpu(const float* x, float* out,
                      int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float best = x[(o * dim_size + 0) * inner + i];
            for (int k = 1; k < dim_size; k++) {
                float v = x[(o * dim_size + k) * inner + i];
                if (v < best) best = v;
            }
            out[o * inner + i] = best;
        }
    }
}

// A.6.d masked_cumsum: cumsum(x * mask, dim).
// mask is uint8 (1=True, 0=False) — Python ctypes will pass it as such.
// Layout: contiguous (batch, dim_size), cumsum along dim_size axis.
//
// SUBSTRATE-DESIGN: cumulative_acc_type(double) — PyTorch's torch.cumsum
// accumulates in double internally and casts back to float on store.
// (Same pattern as bpd_cumsum_cpu — empirically verified earlier in this
// session for #89.)
void bpd_masked_cumsum_cpu(const float* x, const unsigned char* mask,
                            float* out, int batch, int dim_size) {
    for (int b = 0; b < batch; b++) {
        double acc = 0.0;  // ← double accumulator, not float
        const float* x_row = x + b * dim_size;
        const unsigned char* m_row = mask + b * dim_size;
        float* o_row = out + b * dim_size;
        for (int k = 0; k < dim_size; k++) {
            double v = m_row[k] ? (double)x_row[k] : 0.0;
            acc = acc + v;
            o_row[k] = (float)acc;  // cast back to f32 on store
        }
    }
}

// A.6.e MinGPT NewGelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
// PyTorch's torch.tanh on CPU dispatches to libm tanhf for fp32.
// Constant sqrt(2/pi) = 0.7978845608028654 in f64 → 0.79788458f in f32 (rounds).
void bpd_mingpt_newgelu_cpu(const float* x, float* out, int n) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float GELU_COEF = 0.044715f;
    for (int i = 0; i < n; i++) {
        float xv = x[i];
        float x3 = xv * xv * xv;
        float inner = SQRT_2_OVER_PI * (xv + GELU_COEF * x3);
        float t = tanhf(inner);
        out[i] = 0.5f * xv * (1.0f + t);
    }
}

// A.6.f ScaledDotProductAttention: out = softmax(Q @ K.T / sqrt(d_k), dim=-1) @ V
// Q, K, V shape: (batch, num_heads, seq_len, embed_dim).
// Per (batch, head): scores[seq, seq] = Q[seq, embed] @ K.T[embed, seq] / sqrt(embed_dim)
//                    attn[seq, seq] = softmax(scores, dim=-1)
//                    out[seq, embed] = attn[seq, seq] @ V[seq, embed]
//
// We allocate temp buffers for scores and K.T per (batch, head) pair.
extern float linear_scan_sum_simd8(const float* data, int n);
// We need a bpd_mm with one operand transposed. Simplest: physically transpose K
// once per (batch, head) into a temp buffer, then use bpd_mm_cpu.
//
// SUBSTANTIVE substrate-design choice: PyTorch's _scaled_dot_product_attention_math
// (attention.cpp:850) uses the SQUARE-ROOTED PRE-SCALE pattern:
//   scaling = sqrt(1/sqrt(d_k))  = d_k^(-1/4)
//   Q' = Q * scaling
//   K' = K * scaling   (NOTE: PyTorch's code does K.transpose(-2,-1) * scaling)
//   scores = Q' @ K'^T = (Q @ K^T) * (scaling * scaling) = (Q @ K^T) / sqrt(d_k)
// This pre-scaling produces different bits from post-scaling (Q@K^T then /sqrt(d_k)).
//
// Source: aten/src/ATen/native/transformers/attention.cpp:894-901
void bpd_scaled_dot_product_attention_cpu(const float* Q, const float* K, const float* V,
                                            float* out,
                                            int batch, int num_heads, int seq_len, int embed_dim) {
    int qkv_per_head = seq_len * embed_dim;
    int scores_size = seq_len * seq_len;
    float* Q_scaled = (float*)malloc(qkv_per_head * sizeof(float));
    float* K_scaled_T = (float*)malloc(embed_dim * seq_len * sizeof(float));
    float* scores = (float*)malloc(scores_size * sizeof(float));
    if (!Q_scaled || !K_scaled_T || !scores) {
        if (Q_scaled) free(Q_scaled);
        if (K_scaled_T) free(K_scaled_T);
        if (scores) free(scores);
        return;
    }

    // PyTorch's calculate_scale returns 1/sqrt(d_k); then .sqrt() = d_k^(-1/4)
    float scaling = sqrtf(1.0f / sqrtf((float)embed_dim));

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            int slot = b * num_heads + h;
            const float* Q_h = Q + slot * qkv_per_head;
            const float* K_h = K + slot * qkv_per_head;
            const float* V_h = V + slot * qkv_per_head;
            float* out_h = out + slot * qkv_per_head;

            // Pre-scale Q: Q_scaled = Q * scaling
            for (int i = 0; i < qkv_per_head; i++) Q_scaled[i] = Q_h[i] * scaling;

            // Pre-scale K then transpose: K_scaled_T[e, s] = K_h[s, e] * scaling
            for (int s = 0; s < seq_len; s++) {
                for (int e = 0; e < embed_dim; e++) {
                    K_scaled_T[e * seq_len + s] = K_h[s * embed_dim + e] * scaling;
                }
            }

            // scores = Q_scaled @ K_scaled_T → shape (seq_len, seq_len)
            //   Mathematically: scores = (Q @ K^T) * scaling^2 = (Q @ K^T) / sqrt(d_k)
            //   But bit-different from post-scaling.
            bpd_mm_cpu(Q_scaled, K_scaled_T, scores, seq_len, seq_len, embed_dim);

            // Softmax per row of scores
            for (int s = 0; s < seq_len; s++) {
                float* row = scores + s * seq_len;
                float mx = row[0];
                for (int k = 1; k < seq_len; k++) if (row[k] > mx) mx = row[k];
                for (int k = 0; k < seq_len; k++) row[k] = expf(row[k] - mx);
                float sum_exp = linear_scan_sum_simd8(row, seq_len);
                float inv_sum = 1.0f / sum_exp;
                for (int k = 0; k < seq_len; k++) row[k] *= inv_sum;
            }

            // out_h = scores @ V_h → shape (seq_len, embed_dim)
            bpd_mm_cpu(scores, V_h, out_h, seq_len, embed_dim, seq_len);
        }
    }

    free(Q_scaled);
    free(K_scaled_T);
    free(scores);
}

