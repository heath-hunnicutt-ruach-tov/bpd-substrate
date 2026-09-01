// gemv_soa_q8_0_q8_1_det.cuh — DETERMINISTIC-FMA SoA gemv (Iyun, 2026-09-01).
//
// THE FIX (one line vs the original): the float accumulate is pinned to
//   sum = __fmaf_rn(dw*da, (float)sumi, sum)   instead of   sum += dw*da*sumi
//
// WHY (mechanism cornered OFFLINE, verified vs the reference libggml-cuda.so.0.13.1
// the P4 runs — zero-compile archaeology, cuobjdump+nvdisasm on the real cubin):
//   STOCK Q8_0 mmvq accumulate = FFMA.FTZ (dst==accumulator) — FUSED, 1 rounding.
//   ORIGINAL SoA accumulate    = FMUL+FADD (0 FFMA in its SASS) — UN-FUSED, 2 roundings.
//   THIS _det accumulate       = FFMA (the __fmaf_rn pin) — MATCHES STOCK.
// The 2-rounding-vs-1-rounding delta per block is the documented per-element 1-2 ULP
// noise that flips argmax at near-ties. Pinning to FFMA restores stock's exact accumulate.
//
// SCOPE (authority-never-exceeds-evidence): SASS-accumulate-match is NECESSARY and
// strongly-evidenced offline. SUFFICIENCY (per-element 0-ULP + 18-battery flips->0)
// is the P4 harness's to confirm — register allocation + the reduce tree remain
// residual unknowns only the card can settle. This file is compile-verified here;
// efficacy-gated there.
//
// INTEGRATION: side-by-side with the original, env-gated GGML_SOA_DET=1 (dispatch
// selects _det when set, original when unset — A/B without rebuild, zero regression).
//
// gemv_soa_q8_0_q8_1_det.cuh — v15b: match stock reduction ORDER exactly
//
// Stock reduction: 
//   1. warps 1-3 write per-lane partial sums to shared
//   2. warp 0 adds other warps' same-lane values to its own
//   3. warp_reduce_sum across lanes in warp 0
//
// This produces a DIFFERENT FP rounding than:
//   1. warp_reduce_sum in each warp independently
//   2. combine warp results via shared

#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

#define SOA_QK 32
#define SOA_QI 8
#define SOA_VDR 2
#define SOA_WARP 32

struct soa_bq81 { half2 ds; int8_t qs[32]; };
__device__ float soa_debug_buf[16];

__device__ __forceinline__ float soa_silu(float x) {
    return x / (1.0f + expf(-x));
}

template <int warp_size>
static __device__ __forceinline__ float soa_warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xFFFFFFFF, x, offset, warp_size);
    }
    return x;
}

template<int TPB, int FUSE_SILU = 0>
__global__ void gemv_soa_q8_0_q8_1_det(
    const char * __restrict__ quants_base,
    const char * __restrict__ scales_base,
    const char * __restrict__ adata,
    float * __restrict__ dst,
    int nrows, int ncols, int bpr,
    const float * __restrict__ up_result,
    const float * __restrict__ residual)
{
    constexpr int nwarps = TPB / SOA_WARP;
    const int tid = threadIdx.x;
    const int warp_id = threadIdx.x / SOA_WARP;  // = threadIdx.y in stock
    const int lane = threadIdx.x % SOA_WARP;      // = threadIdx.x in stock
    const int row = blockIdx.x;
    if (row >= nrows) return;

    const int blocks_per_iter = SOA_VDR * TPB / SOA_QI;
    const int kbx_start = tid / (SOA_QI / SOA_VDR);
    const int kqs = SOA_VDR * (tid % (SOA_QI / SOA_VDR));

    const int8_t *ptr_wq = (const int8_t *)quants_base + (size_t)row * ncols + kbx_start * SOA_QK;
    const half   *ptr_ws = (const half *)scales_base + (size_t)row * bpr + kbx_start;
    const char   *ptr_y  = adata + (size_t)kbx_start * 36;

    const int wq_step = blocks_per_iter * SOA_QK;
    const int ws_step = blocks_per_iter;
    const int y_step  = blocks_per_iter * 36;


    float sum = 0.0f;

    for (int kbx = kbx_start; kbx < bpr; kbx += blocks_per_iter) {
        float dw = __half2float(*ptr_ws);
        const int *wq = (const int *)ptr_wq;
        int v0 = wq[kqs + 0];
        int v1 = wq[kqs + 1];
        const soa_bq81 *yb = (const soa_bq81 *)ptr_y;
        float da = __half2float(__low2half(yb->ds));
        int u0 = ((const int *)yb->qs)[kqs + 0];
        int u1 = ((const int *)yb->qs)[kqs + 1];
        int sumi = 0;
        sumi = __dp4a(v0, u0, sumi);
        sumi = __dp4a(v1, u1, sumi);
        sum = __fmaf_rn(dw * da, (float) sumi, sum);  // DET: pinned FFMA, matches stock accumulate (verified vs reference .so)
        // Debug: first iteration of row 0, write inputs to global debug buf
        if (row == 0 && tid == 0 && kbx == kbx_start) {
            soa_debug_buf[0] = dw;
            soa_debug_buf[1] = da;
            soa_debug_buf[2] = (float)sumi;
            soa_debug_buf[3] = dw * da * (float)sumi;
            soa_debug_buf[4] = __int_as_float(v0);
            soa_debug_buf[5] = __int_as_float(v1);
            soa_debug_buf[6] = __int_as_float(u0);
            soa_debug_buf[7] = __int_as_float(u1);
        }
        ptr_wq += wq_step;
        ptr_ws += ws_step;
        ptr_y  += y_step;
    }

    // === MATCH STOCK REDUCTION ORDER EXACTLY ===
    // Stock pattern: warps 1+ write per-lane to shared, warp 0 accumulates, then warp_reduce
    __shared__ float tmp_shared[nwarps > 1 ? nwarps - 1 : 1][SOA_WARP];

    if (warp_id > 0) {
        tmp_shared[warp_id - 1][lane] = sum;
    }
    __syncthreads();

    if (warp_id > 0) return;

    // Warp 0: add other warps' values at the SAME lane (matches stock order)
#pragma unroll
    for (int l = 0; l < nwarps - 1; ++l) {
        sum += tmp_shared[l][lane];
    }

    // THEN reduce across lanes within warp 0
    sum = soa_warp_reduce_sum<SOA_WARP>(sum);

    if (lane == 0) {
        float result = FUSE_SILU ? soa_silu(sum) : sum;
        if (up_result) result *= up_result[row];
        if (residual)  result += residual[row];
        dst[row] = result;
    }
}
