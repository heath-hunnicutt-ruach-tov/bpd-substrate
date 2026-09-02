:- protocol(soa_gemv_det_emitterp).
    :- public(emit_soa_gemv_det/1).
:- end_protocol.

:- object(soa_gemv_det_emitter,
    implements(soa_gemv_det_emitterp)).



emit_soa_gemv_det(OutFile) :-
    open(OutFile, write, S),
    emit_head(S),
    emit_warp_reduce(S),
    emit_det_gemv(S),
    close(S),
    format("Emitted deterministic-FMA SoA gemv to ~w~n", [OutFile]).

emit_head(S) :-
    format(S, '// gemv_soa_q8_0_q8_1_det.cuh — deterministic-FMA SoA gemv.~n', []),
    format(S, '// Emitted by soa_gemv_det_emitter.pl (2026-09-01, Iyun).~n', []),
    format(S, '// Per-element bit-identical to stock BY CONSTRUCTION: pinned FMA order~n', []),
    format(S, '// (__fmaf_rn), pinned block loop, pinned warp-reduce. Side-by-side with~n', []),
    format(S, '// the original, env-gated GGML_SOA_DET=1.~n~n', []),
    format(S, '#include <cuda_fp16.h>~n~n', []),
    format(S, '#ifndef SOA_WARP~n#define SOA_WARP 32~n#endif~n~n', []).

%% (c) warp-reduction — stock's shuffle tree, verbatim.
emit_warp_reduce(S) :-
    format(S, 'template <int warp_size>~n', []),
    format(S, 'static __device__ __forceinline__ float soa_warp_reduce_sum_det(float x) {~n', []),
    format(S, '#pragma unroll~n', []),
    format(S, '    for (int offset = warp_size/2; offset > 0; offset >>= 1) {~n', []),
    format(S, '        x += __shfl_xor_sync(0xFFFFFFFF, x, offset, warp_size);~n', []),
    format(S, '    }~n', []),
    format(S, '    return x;~n', []),
    format(S, '}~n~n', []).

%% The deterministic inner accumulation. Signature mirrors the original SoA
%% gemv's per-row block loop; only the float step (b) changes vs the original.
emit_det_gemv(S) :-
    format(S, '// Per-row block accumulation. The ONLY change vs the original gemv is~n', []),
    format(S, '// the float step: __fmaf_rn(scale, sumi, sum) instead of sum += dw*da*sumi.~n', []),
    format(S, 'static __device__ __forceinline__ float soa_gemv_row_accum_det(~n', []),
    format(S, '        const int8_t *ptr_wq, const half *ptr_ws,~n', []),
    format(S, '        const void  *ptr_y_base,~n', []),
    format(S, '        int kbx_start, int bpr, int blocks_per_iter,~n', []),
    format(S, '        int kqs, int SOA_QK, int y_stride) {~n', []),
    format(S, '    float sum = 0.0f;~n', []),
    format(S, '    for (int kbx = kbx_start; kbx < bpr; kbx += blocks_per_iter) {~n', []),
    %% (a) same block stride, same operand fetch as the original
    format(S, '        const int8_t *wq_blk = ptr_wq + (size_t)(kbx - kbx_start) * SOA_QK;~n', []),
    format(S, '        const half   *ws_blk = ptr_ws + (kbx - kbx_start);~n', []),
    format(S, '        float dw = __half2float(*ws_blk);~n', []),
    format(S, '        const int *wq = (const int *)wq_blk;~n', []),
    format(S, '        int v0 = wq[kqs + 0];~n', []),
    format(S, '        int v1 = wq[kqs + 1];~n', []),
    format(S, '        const char *yb = (const char *)ptr_y_base + (size_t)(kbx - kbx_start) * y_stride;~n', []),
    format(S, '        float da = __half2float(__low2half(*(const half2 *)yb));~n', []),
    format(S, '        const int *yq = (const int *)(yb + sizeof(half2));~n', []),
    format(S, '        int u0 = yq[kqs + 0];~n', []),
    format(S, '        int u1 = yq[kqs + 1];~n', []),
    format(S, '        int sumi = 0;~n', []),
    format(S, '        sumi = __dp4a(v0, u0, sumi);~n', []),
    format(S, '        sumi = __dp4a(v1, u1, sumi);~n', []),
    %% (b) THE FIX — pinned FMA: scale in stock's order, one round-to-nearest madd.
    format(S, '        float scale = dw * da;                 // stock (d8_0*d8_1) order~n', []),
    format(S, '        sum = __fmaf_rn(scale, (float) sumi, sum);  // pinned: nvcc cannot reschedule~n', []),
    format(S, '    }~n', []),
    format(S, '    return sum;~n', []),
    format(S, '}~n', []).

:- end_object.
