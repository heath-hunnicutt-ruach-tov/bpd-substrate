%% soa_q8_0_quantize_emitter.pl — Emit the ggml-exact Q8_0 quantization (rung-3 crux).
%%
%% RECONSTRUCTED 2026-08-31 (Iyun) from the June corpus (conv124 w376). This is
%% the FP-sensitive atom that was the crux of rung 3 (matmul chain): the
%% intermediate re-quantization of gemv1's float output into Q8_0 for gemv2.
%% In June it was byte-exact vs ggml. Reconstruction, not recovery.
%%
%% THE ATOM:
%%   void @bpd_q8_0_quantize(ptr %x, ptr %q_out, ptr %d_out, i32 %n)
%%     Quantize n floats (n must be a multiple of 32) into Q8_0 blocks:
%%     for each 32-lane block:
%%       amax = max|x[i]|                       (llvm.fabs + llvm.maxnum)
%%       d    = amax / 127.0                     (fp32)
%%       d    = fp16(d)                          (fptrunc to half, fpext back —
%;                                                the fp16 STORED scale round-trip)
%%       id   = (d != 0) ? 1/d : 0
%%       q[i] = (i8) round(x[i] * id)            (llvm.round = round-half-away,
%%                                                matching roundf)
%%     q_out — i8 quants [n];  d_out — float scales [n/32] (the fp16-round-tripped d).
%%
%% THE FP CRUX (why this is the risky atom):
%%   - the fp16(d) round-trip: ggml STORES d as fp16, so the effective scale is
%%     fpext(fptrunc(d)) — must reproduce exactly or every quant drifts.
%%   - round-half-away (roundf semantics), NOT round-to-even — llvm.round is the
%%     correct intrinsic.
%%
%% VERIFY: emit -> llvm-as -> llc -> link with a ggml-exact CPU reference ->
%% byte-exact gate (quants AND scales bit-identical).

:- module(soa_q8_0_quantize_emitter, [emit_soa_q8_0_quantize/1]).

emit_soa_q8_0_quantize(OutFile) :-
    open(OutFile, write, S),
    format(S, '; ggml-exact Q8_0 quantization: amax/127, fp16(d) round-trip, round-half-away.~n', []),
    format(S, '; void @bpd_q8_0_quantize(ptr x, ptr q_out, ptr d_out, i32 n)~n~n', []),
    format(S, 'declare float @llvm.fabs.f32(float)~n', []),
    format(S, 'declare float @llvm.maxnum.f32(float, float)~n', []),
    format(S, 'declare float @llvm.round.f32(float)~n~n', []),
    emit_quant_function(S),
    close(S),
    format("Emitted Q8_0 quantize to ~w~n", [OutFile]).

emit_quant_function(S) :-
    format(S, 'define void @bpd_q8_0_quantize(ptr %x, ptr %q_out, ptr %d_out, i32 %n) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %nb = sdiv i32 %n, 32~n', []),
    format(S, '  %nb_gt = icmp sgt i32 %nb, 0~n', []),
    format(S, '  br i1 %nb_gt, label %blk.head, label %done~n~n', []),
    format(S, 'blk.head:~n', []),
    format(S, '  %b = phi i32 [ 0, %entry ], [ %b.next, %blk.tail ]~n', []),
    format(S, '  %base = mul i32 %b, 32~n', []),
    %% pass 1: amax over the 32 lanes
    format(S, '  br label %amax.head~n~n', []),
    format(S, 'amax.head:~n', []),
    format(S, '  %j1 = phi i32 [ 0, %blk.head ], [ %j1.next, %amax.body ]~n', []),
    format(S, '  %amax = phi float [ 0.000000e+00, %blk.head ], [ %amax.next, %amax.body ]~n', []),
    format(S, '  %j1.lt = icmp slt i32 %j1, 32~n', []),
    format(S, '  br i1 %j1.lt, label %amax.body, label %amax.done~n~n', []),
    format(S, 'amax.body:~n', []),
    format(S, '  %idx1 = add i32 %base, %j1~n', []),
    format(S, '  %xp1 = getelementptr float, ptr %x, i32 %idx1~n', []),
    format(S, '  %xv1 = load float, ptr %xp1~n', []),
    format(S, '  %absx = call float @llvm.fabs.f32(float %xv1)~n', []),
    format(S, '  %amax.next = call float @llvm.maxnum.f32(float %amax, float %absx)~n', []),
    format(S, '  %j1.next = add i32 %j1, 1~n', []),
    format(S, '  br label %amax.head~n~n', []),
    %% d = amax/127; d = fp16(d) via fptrunc/fpext; id = (d!=0)?1/d:0
    format(S, 'amax.done:~n', []),
    format(S, '  %d0 = fdiv float %amax, 1.270000e+02~n', []),
    format(S, '  %dh = fptrunc float %d0 to half~n', []),
    format(S, '  %d = fpext half %dh to float~n', []),
    format(S, '  %dp = getelementptr float, ptr %d_out, i32 %b~n', []),
    format(S, '  store float %d, ptr %dp~n', []),
    format(S, '  %dz = fcmp oeq float %d, 0.000000e+00~n', []),
    format(S, '  %dinv = fdiv float 1.000000e+00, %d~n', []),
    format(S, '  %id = select i1 %dz, float 0.000000e+00, float %dinv~n', []),
    %% pass 2: q[i] = (i8) round(x[i] * id)
    format(S, '  br label %q.head~n~n', []),
    format(S, 'q.head:~n', []),
    format(S, '  %j2 = phi i32 [ 0, %amax.done ], [ %j2.next, %q.body ]~n', []),
    format(S, '  %j2.lt = icmp slt i32 %j2, 32~n', []),
    format(S, '  br i1 %j2.lt, label %q.body, label %blk.tail~n~n', []),
    format(S, 'q.body:~n', []),
    format(S, '  %idx2 = add i32 %base, %j2~n', []),
    format(S, '  %xp2 = getelementptr float, ptr %x, i32 %idx2~n', []),
    format(S, '  %xv2 = load float, ptr %xp2~n', []),
    format(S, '  %scaled = fmul float %xv2, %id~n', []),
    format(S, '  %rounded = call float @llvm.round.f32(float %scaled)~n', []),
    format(S, '  %qi = fptosi float %rounded to i8~n', []),
    format(S, '  %qp = getelementptr i8, ptr %q_out, i32 %idx2~n', []),
    format(S, '  store i8 %qi, ptr %qp~n', []),
    format(S, '  %j2.next = add i32 %j2, 1~n', []),
    format(S, '  br label %q.head~n~n', []),
    format(S, 'blk.tail:~n', []),
    format(S, '  %b.next = add i32 %b, 1~n', []),
    format(S, '  %b.lt = icmp slt i32 %b.next, %nb~n', []),
    format(S, '  br i1 %b.lt, label %blk.head, label %done~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).
