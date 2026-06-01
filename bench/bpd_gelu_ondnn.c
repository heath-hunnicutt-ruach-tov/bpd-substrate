/* bpd_gelu_ondnn.c — GELU matching oneDNN's JIT output bit-for-bit.
 *
 * Constants and computation flow lifted from oneDNN's JIT binary
 * via asm analysis. Uses the same AVX instruction sequence to get
 * identical f32 rounding behavior.
 *
 * Algorithm: gelu(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
 * where erf uses Abramowitz & Stegun approximation (eq 7.1.26)
 * and exp uses oneDNN's polynomial 2^(x*log2e) decomposition.
 *
 * Build: gcc -O2 -mavx -shared -fPIC -o bpd_gelu_ondnn.so bpd_gelu_ondnn.c
 * Test:  python3 test_gelu_ondnn.py
 */
#include <immintrin.h>
#include <stdint.h>

/* ============================================================
 * Constants lifted from oneDNN JIT binary (exact f32 hex)
 * ============================================================ */

/* Erf polynomial (Abramowitz & Stegun 7.1.26) */
static const float C_ERF_P  = 0.3275910914f;   /* 0x05baa73e — t = 1/(1+p*|x|) */
static const float C_ERF_A1 = 0.2548295856f;   /* 0x0679823e */
static const float C_ERF_A2 = -0.2844967246f;  /* 0x8ea991be */
static const float C_ERF_A3 = 1.4214137793f;   /* 0xe3f0b53f */
static const float C_ERF_A4 = -1.4531520605f;  /* 0xe300babf */
static const float C_ERF_A5 = 1.0614054203f;   /* 0x22dc873f */

/* Exp polynomial (2^f approximation on [-0.5, 0.5]) */
static const float C_LOG2E  = 1.4426950216f;   /* 0x3baab83f */
static const float C_LN2    = 0.6931471825f;   /* 0x1872313f */
static const float C_EXP_C1 = 0.9999997020f;   /* 0xfbff7f3f */
static const float C_EXP_C2 = 0.4999915063f;   /* 0xe3feff3e */
static const float C_EXP_C3 = 0.1666765213f;   /* 0x40ad2a3e */
static const float C_EXP_C4 = 0.0418978222f;   /* 0x0d9d2b3d */
static const float C_EXP_C5 = 0.0082892906f;   /* 0xcecf073c */

/* Misc */
static const float C_SQRT2_INV = 0.7071067691f; /* 0xf304353f */
static const float C_HALF = 0.5f;
static const float C_ONE  = 1.0f;

/* ============================================================
 * AVX implementation — matches oneDNN's JIT instruction sequence
 * ============================================================ */

static inline __m256 ondnn_exp_avx(__m256 x) {
    /* exp(x) via 2^(x * log2e) decomposition.
     * z = x * log2e
     * n = round(z)           — integer part
     * f = z - n              — fractional part, |f| <= 0.5
     * 2^f ≈ poly(f)          — Horner polynomial
     * 2^n via integer add to exponent bits
     * exp(x) = 2^n * 2^f
     */
    __m256 log2e = _mm256_set1_ps(C_LOG2E);
    __m256 ln2   = _mm256_set1_ps(C_LN2);
    __m256 one   = _mm256_set1_ps(C_ONE);

    /* z = x * log2(e) */
    __m256 z = _mm256_mul_ps(x, log2e);

    /* n = round(z) — vroundps with mode 0 (round to nearest) */
    __m256 n = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* f = x - n * ln2 (more precise than z - n) */
    /* This is the Cody-Waite reduction oneDNN uses */
    __m256 f = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));

    /* Horner polynomial for 2^f:
     * p = c5*f + c4; p = p*f + c3; p = p*f + c2; p = p*f + c1; p = p*f + 1 */
    __m256 c5 = _mm256_set1_ps(C_EXP_C5);
    __m256 c4 = _mm256_set1_ps(C_EXP_C4);
    __m256 c3 = _mm256_set1_ps(C_EXP_C3);
    __m256 c2 = _mm256_set1_ps(C_EXP_C2);
    __m256 c1 = _mm256_set1_ps(C_EXP_C1);

    __m256 p = c5;
    p = _mm256_add_ps(_mm256_mul_ps(p, f), c4);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), c3);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), c2);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), c1);
    p = _mm256_add_ps(_mm256_mul_ps(p, f), one);

    /* 2^n: convert n to int, add bias, shift to exponent field.
     * AVX1 doesn't have 256-bit integer ops, so use 128-bit halves. */
    __m256i ni = _mm256_cvtps_epi32(n);
    __m128i ni_lo = _mm256_castsi256_si128(ni);
    __m128i ni_hi = _mm256_extractf128_si256(ni, 1);
    __m128i bias = _mm_set1_epi32(127);
    __m128i exp_lo = _mm_slli_epi32(_mm_add_epi32(ni_lo, bias), 23);
    __m128i exp_hi = _mm_slli_epi32(_mm_add_epi32(ni_hi, bias), 23);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_insertf128_si256(_mm256_castsi128_si256(exp_lo), exp_hi, 1));

    return _mm256_mul_ps(p, scale);
}

void bpd_gelu_ondnn_cpu(const float* input, float* output, int n) {
    __m256 sqrt2_inv = _mm256_set1_ps(C_SQRT2_INV);
    __m256 half      = _mm256_set1_ps(C_HALF);
    __m256 one       = _mm256_set1_ps(C_ONE);
    __m256 neg_one   = _mm256_set1_ps(-1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);  /* 0x80000000 */

    /* Abramowitz constants */
    __m256 erf_p  = _mm256_set1_ps(C_ERF_P);
    __m256 erf_a1 = _mm256_set1_ps(C_ERF_A1);
    __m256 erf_a2 = _mm256_set1_ps(C_ERF_A2);
    __m256 erf_a3 = _mm256_set1_ps(C_ERF_A3);
    __m256 erf_a4 = _mm256_set1_ps(C_ERF_A4);
    __m256 erf_a5 = _mm256_set1_ps(C_ERF_A5);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);

        /* v = x * (1/sqrt(2)) */
        __m256 v = _mm256_mul_ps(x, sqrt2_inv);

        /* |v| = abs(v) — clear sign bit */
        __m256 av = _mm256_andnot_ps(sign_mask, v);

        /* sign of v (just the sign bit) */
        __m256 v_sign = _mm256_and_ps(v, sign_mask);

        /* t = 1.0 / (1.0 + p * |v|) — vdivps like the JIT */
        __m256 denom = _mm256_add_ps(one, _mm256_mul_ps(erf_p, av));
        __m256 t = _mm256_div_ps(one, denom);

        /* Horner for erf polynomial: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t */
        __m256 y = erf_a5;
        y = _mm256_add_ps(_mm256_mul_ps(y, t), erf_a4);
        y = _mm256_add_ps(_mm256_mul_ps(y, t), erf_a3);
        y = _mm256_add_ps(_mm256_mul_ps(y, t), erf_a2);
        y = _mm256_add_ps(_mm256_mul_ps(y, t), erf_a1);
        y = _mm256_mul_ps(y, t);

        /* exp(-v²): neg_v2 = -(|v| * |v|) */
        __m256 v2 = _mm256_mul_ps(av, av);
        __m256 neg_v2 = _mm256_xor_ps(v2, sign_mask);
        __m256 e = ondnn_exp_avx(neg_v2);

        /* erf_abs = 1 - y * exp(-v²) */
        __m256 erf_abs = _mm256_sub_ps(one, _mm256_mul_ps(y, e));

        /* erf = sign(v) * erf_abs — restore sign via xor */
        __m256 erf_val = _mm256_xor_ps(erf_abs, v_sign);

        /* gelu = 0.5 * x * (1 + erf) */
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, erf_val)));

        _mm256_storeu_ps(output + i, result);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        float x = input[i];
        float v = x * C_SQRT2_INV;
        float av = v < 0 ? -v : v;
        float sign = v < 0 ? -1.0f : 1.0f;

        float t = 1.0f / (1.0f + C_ERF_P * av);
        float y = C_ERF_A5;
        y = y * t + C_ERF_A4;
        y = y * t + C_ERF_A3;
        y = y * t + C_ERF_A2;
        y = y * t + C_ERF_A1;
        y = y * t;

        float neg_v2 = -(av * av);
        /* For scalar tail, use the same polynomial exp */
        float z = neg_v2 * C_LOG2E;
        float n_round = __builtin_roundf(z);
        float f = neg_v2 - n_round * C_LN2;
        float p = C_EXP_C5;
        p = p * f + C_EXP_C4;
        p = p * f + C_EXP_C3;
        p = p * f + C_EXP_C2;
        p = p * f + C_EXP_C1;
        p = p * f + 1.0f;
        int32_t ni = (int32_t)n_round;
        union { float f; int32_t i; } scale;
        scale.i = (ni + 127) << 23;
        float e = p * scale.f;

        float erf_abs = 1.0f - y * e;
        float erf_val = sign * erf_abs;
        output[i] = 0.5f * x * (1.0f + erf_val);
    }
}
