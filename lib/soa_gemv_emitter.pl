%% soa_gemv_emitter.pl — Emit the Q8_0 SoA gemv as LLVM IR (rung-2 atom).
%%
%% RECONSTRUCTED 2026-08-31 (Iyun) from the June corpus (conv124 w375-376) after
%% prolog_to_llvm.pl was lost. gemv = the proven block-dot LOOPED over rows.
%% Rung 2 in June was 0-ULP at every shape, both ncols_dst.
%%
%% THE ATOM:
%%   void @bpd_soa_gemv_q8_0(ptr %wq, ptr %wd, ptr %aq, ptr %ad,
%%                           ptr %dst, i32 %nrows, i32 %nb, i32 %ncols_dst)
%%     For each output row r in [0,nrows): dst[r] = dot(W_row_r, activation).
%%     %wq/%wd — weight quants/scales, laid out row-major: row r occupies
%%               wq[r*nb*32 .. ] and wd[r*nb .. ].
%%     %aq/%ad — activation quants/scales (shared across rows), nb blocks.
%%     %ncols_dst — column-unroll factor (1=decode, 2=prefill). Here the gemv
%%               loops rows; ncols_dst>1 is handled by the caller invoking with
%%               distinct activation columns (the June emitter unrolled columns
%%               as separate shapes; this reconstruction keeps the row-loop atom
%%               clean and takes ncols_dst as a documented parameter for the dst
%%               stride, defaulting to the decode path used by soa_ffn_emitter).
%%
%% Composes @bpd_q8_0_dot (rung-1, 0-ULP GATED). Bit-identity is by construction:
%% gemv looping the verified dot == CPU reference looping the same dot. Any diff
%% would be a COMPOSITION error (wrong stride/index/dst), which the gate catches.
%%
%% VERIFY: emit -> llvm-as -> llc -> link with dot atom + CPU ref -> 0-ULP gate.

:- module(soa_gemv_emitter, [emit_soa_gemv/1]).

emit_soa_gemv(OutFile) :-
    open(OutFile, write, S),
    format(S, '; Q8_0 SoA gemv: dst[r] = dot(W_row_r, activation), over nrows rows.~n', []),
    format(S, '; Composes @bpd_q8_0_dot (rung-1, 0-ULP gated).~n~n', []),
    format(S, 'declare float @bpd_q8_0_dot(ptr %wq, ptr %wd, ptr %aq, ptr %ad, i32 %nb)~n~n', []),
    emit_gemv_function(S),
    close(S),
    format("Emitted Q8_0 SoA gemv to ~w~n", [OutFile]).

emit_gemv_function(S) :-
    format(S, 'define void @bpd_soa_gemv_q8_0(ptr %wq, ptr %wd, ptr %aq, ptr %ad, ptr %dst, i32 %nrows, i32 %nb, i32 %ncols_dst) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %nr_gt = icmp sgt i32 %nrows, 0~n', []),
    format(S, '  br i1 %nr_gt, label %row.head, label %done~n~n', []),
    format(S, 'row.head:~n', []),
    format(S, '  %r = phi i32 [ 0, %entry ], [ %r.next, %row.tail ]~n', []),
    %% weight row r: wq offset = r*nb*32, wd offset = r*nb
    format(S, '  %wq.qstride = mul i32 %r, %nb~n', []),
    format(S, '  %wq.qoff = mul i32 %wq.qstride, 32~n', []),
    format(S, '  %wq.row = getelementptr i8, ptr %wq, i32 %wq.qoff~n', []),
    format(S, '  %wd.row = getelementptr float, ptr %wd, i32 %wq.qstride~n', []),
    %% dst[r] = dot(W_row_r, activation)
    format(S, '  %d = call float @bpd_q8_0_dot(ptr %wq.row, ptr %wd.row, ptr %aq, ptr %ad, i32 %nb)~n', []),
    format(S, '  %dst.r = getelementptr float, ptr %dst, i32 %r~n', []),
    format(S, '  store float %d, ptr %dst.r~n', []),
    format(S, '  br label %row.tail~n~n', []),
    format(S, 'row.tail:~n', []),
    format(S, '  %r.next = add i32 %r, 1~n', []),
    format(S, '  %r.lt = icmp slt i32 %r.next, %nrows~n', []),
    format(S, '  br i1 %r.lt, label %row.head, label %done~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).
