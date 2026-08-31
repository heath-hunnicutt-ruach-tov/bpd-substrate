%% soa_q8_0_dot_emitter.pl — Emit the Q8_0 SoA block-dot as LLVM IR.
%%
%% RECONSTRUCTED 2026-08-31 (Iyun) from the June corpus (conv124 w375) after
%% prolog_to_llvm.pl (which held emit_q8_0_dot) was lost to /tmp-evaporation.
%% Confirmed absent from every surviving tree (only the CUDA .cuh forms landed;
%% the CPU-target IR emitters are gone). Reconstruction, not recovery.
%%
%% THE ATOM (rung-1 foundation): the Q8_0 block-dot, SoA-structured.
%%   float @bpd_q8_0_dot(ptr %wq, ptr %wd, ptr %aq, ptr %ad, i32 %nb)
%%     %wq — weight quants  (i8, laid out [nb][32])
%%     %wd — weight scales  (float, one per block — SEPARATE buffer = SoA)
%%     %aq — activation quants (i8, [nb][32])
%%     %ad — activation scales (float, one per block)
%%     %nb — number of 32-lane blocks
%%
%% FP DISCIPLINE (bit-identity — the whole point):
%%   - inner: EXACT int32 accumulate over 32 lanes (sext i8->i32, mul, add) — no rounding
%%   - per-block: scale = fmul(wd, ad); dotf = sitofp(isum); prod = fmul(scale, dotf);
%%                acc = fadd(acc, prod)
%%   - scalar, SSE-no-FMA order (has_fma=false) — the correctness-first choice that
%%     COSTS perf but guarantees bit-identity vs ggml's reference dot.
%%   This matches the June-verified bpd_q8_0_dot exactly (the atom every gemv loops).
%%
%% VERIFY-BEFORE-CLAIMING: emit -> llvm-as parse-check -> gate 0-ULP vs a CPU
%% reference doing the identical int-dot-then-float-accumulate.

:- module(soa_q8_0_dot_emitter, [emit_soa_q8_0_dot/1]).

emit_soa_q8_0_dot(OutFile) :-
    open(OutFile, write, S),
    format(S, '; Q8_0 SoA block-dot: exact int32 inner accumulate, per-block float scale.~n', []),
    format(S, '; float @bpd_q8_0_dot(ptr wq, ptr wd, ptr aq, ptr ad, i32 nb)~n', []),
    format(S, '; SSE-no-FMA scalar order for bit-identity vs ggml reference.~n~n', []),
    emit_dot_function(S),
    close(S),
    format("Emitted Q8_0 SoA dot to ~w~n", [OutFile]).

emit_dot_function(S) :-
    format(S, 'define float @bpd_q8_0_dot(ptr %wq, ptr %wd, ptr %aq, ptr %ad, i32 %nb) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %nb_gt = icmp sgt i32 %nb, 0~n', []),
    format(S, '  br i1 %nb_gt, label %blk.head, label %done~n~n', []),
    %% Outer loop over blocks. acc (float) accumulates per-block products.
    format(S, 'blk.head:~n', []),
    format(S, '  %b = phi i32 [ 0, %entry ], [ %b.next, %blk.tail ]~n', []),
    format(S, '  %acc = phi float [ 0.000000e+00, %entry ], [ %acc.next, %blk.tail ]~n', []),
    %% pointers into this block's 32 quants
    format(S, '  %qoff = mul i32 %b, 32~n', []),
    format(S, '  %wq.blk = getelementptr i8, ptr %wq, i32 %qoff~n', []),
    format(S, '  %aq.blk = getelementptr i8, ptr %aq, i32 %qoff~n', []),
    format(S, '  br label %lane.head~n~n', []),
    %% Inner loop: exact int32 accumulate over 32 lanes.
    format(S, 'lane.head:~n', []),
    format(S, '  %j = phi i32 [ 0, %blk.head ], [ %j.next, %lane.body ]~n', []),
    format(S, '  %isum = phi i32 [ 0, %blk.head ], [ %isum.next, %lane.body ]~n', []),
    format(S, '  %j.lt = icmp slt i32 %j, 32~n', []),
    format(S, '  br i1 %j.lt, label %lane.body, label %lane.done~n~n', []),
    format(S, 'lane.body:~n', []),
    format(S, '  %wq.j = getelementptr i8, ptr %wq.blk, i32 %j~n', []),
    format(S, '  %aq.j = getelementptr i8, ptr %aq.blk, i32 %j~n', []),
    format(S, '  %w.i8 = load i8, ptr %wq.j~n', []),
    format(S, '  %a.i8 = load i8, ptr %aq.j~n', []),
    format(S, '  %w.i32 = sext i8 %w.i8 to i32~n', []),
    format(S, '  %a.i32 = sext i8 %a.i8 to i32~n', []),
    format(S, '  %prod.i = mul i32 %w.i32, %a.i32~n', []),
    format(S, '  %isum.next = add i32 %isum, %prod.i~n', []),
    format(S, '  %j.next = add i32 %j, 1~n', []),
    format(S, '  br label %lane.head~n~n', []),
    %% Per-block float combine: scale = wd*ad; prod = scale * sitofp(isum); acc += prod
    format(S, 'lane.done:~n', []),
    format(S, '  %wd.j = getelementptr float, ptr %wd, i32 %b~n', []),
    format(S, '  %ad.j = getelementptr float, ptr %ad, i32 %b~n', []),
    format(S, '  %wd.v = load float, ptr %wd.j~n', []),
    format(S, '  %ad.v = load float, ptr %ad.j~n', []),
    format(S, '  %scale = fmul float %wd.v, %ad.v~n', []),
    format(S, '  %dotf = sitofp i32 %isum to float~n', []),
    format(S, '  %prod = fmul float %scale, %dotf~n', []),
    format(S, '  %acc.next = fadd float %acc, %prod~n', []),
    format(S, '  br label %blk.tail~n~n', []),
    format(S, 'blk.tail:~n', []),
    format(S, '  %b.next = add i32 %b, 1~n', []),
    format(S, '  %b.lt = icmp slt i32 %b.next, %nb~n', []),
    format(S, '  br i1 %b.lt, label %blk.head, label %done~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  %result = phi float [ 0.000000e+00, %entry ], [ %acc.next, %blk.tail ]~n', []),
    format(S, '  ret float %result~n', []),
    format(S, '}~n', []).
