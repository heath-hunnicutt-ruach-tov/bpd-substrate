:- protocol(soa_ffn_emitterp).
    :- public(emit_soa_ffn/1).
    :- public(emit_soa_ffn/2).
:- end_protocol.

:- object(soa_ffn_emitter,
    implements(soa_ffn_emitterp)).



%% emit_soa_ffn(+OutFile)  —  default llama shapes (n_embd=2048, n_ff=8192).
emit_soa_ffn(OutFile) :-
    emit_soa_ffn(OutFile, ffn_shape(2048, 8192)).

%% emit_soa_ffn(+OutFile, +Shape)  —  Shape = ffn_shape(NEmbd, NFF).
emit_soa_ffn(OutFile, ffn_shape(NEmbd, NFF)) :-
    open(OutFile, write, S),
    emit_header(S, NEmbd, NFF),
    emit_extern_atoms(S),
    emit_ffn_layer(S, NEmbd, NFF),
    close(S),
    format("Emitted SoA FFN layer (n_embd=~w, n_ff=~w) to ~w~n",
           [NEmbd, NFF, OutFile]).

emit_header(S, NEmbd, NFF) :-
    format(S, '; SoA FFN layer: down · quantize(swiglu(W_gate·norm, W_up·norm))~n', []),
    format(S, '; n_embd=~w  n_ff=~w~n', [NEmbd, NFF]),
    format(S, '; Reconstructed by soa_ffn_emitter.pl (2026-08-31) — composed verified atoms.~n', []),
    format(S, '; Bit-identity by construction: SwiGLU is a linked atom, not a droppable hand-step.~n~n', []).

%% The composed atoms, declared as externals.  Signatures recovered from the
%% June corpus (conv 124 w375): the block-dot and gemv are SoA-structured with
%% separate quant/scale buffers; SwiGLU takes gate/up/out/n; quantize takes
%% float in, q8_0 (quants+scales) out.
emit_extern_atoms(S) :-
    format(S, '; --- composed verified atoms (linked at gate/build time) ---~n', []),
    %% block-dot (rung-1 proven): int inner accumulate, per-block fmul scale
    format(S, 'declare float @bpd_q8_0_dot(ptr %wq, ptr %wd, ptr %aq, ptr %ad, i32 %nb)~n', []),
    %% gemv (rung-2 proven, 0-ULP): loops the dot over rows, ncols_dst columns
    format(S, 'declare void @bpd_soa_gemv_q8_0(ptr %wq, ptr %wd, ptr %aq, ptr %ad, ptr %dst, i32 %nrows, i32 %nb, i32 %ncols_dst)~n', []),
    %% SwiGLU fused (SURVIVES): silu(gate)*up, divide form, scalar expf, 0-ULP
    format(S, 'declare void @bpd_swiglu_fused_cpu(ptr %gate, ptr %up, ptr %out, i32 %n)~n', []),
    %% quantize (rung-3 crux, byte-exact): float -> q8_0 (quants+scales)
    format(S, 'declare void @bpd_q8_0_quantize(ptr %x, ptr %q_out, ptr %d_out, i32 %n)~n~n', []).

%% The FFN layer function.  Takes the (already-quantized) normalized input and
%% the three weight matrices (each as SoA quant+scale buffers), plus scratch
%% buffers for the intermediates.  Emits the composition of the four atoms.
%%
%% Signature:
%%   @bpd_soa_ffn(
%%     ptr %norm_q, ptr %norm_d,          normalized+quantized input (n_embd)
%%     ptr %wg_q, ptr %wg_d,              W_gate  (n_ff x n_embd, SoA)
%%     ptr %wu_q, ptr %wu_d,              W_up    (n_ff x n_embd, SoA)
%%     ptr %wd_q, ptr %wd_d,              W_down  (n_embd x n_ff, SoA)
%%     ptr %gate_f, ptr %up_f,           scratch: gate/up float (n_ff)
%%     ptr %fused_f,                     scratch: swiglu output float (n_ff)
%%     ptr %qf_q, ptr %qf_d,             scratch: quantized fused (n_ff)
%%     ptr %out_f)                       output float (n_embd)
%%
%% nb_embd = n_embd/32 blocks per row for the gate/up dots;
%% nb_ff   = n_ff/32   blocks per row for the down dot.
emit_ffn_layer(S, NEmbd, NFF) :-
    NbEmbd is NEmbd // 32,
    NbFF   is NFF // 32,
    format(S, 'define void @bpd_soa_ffn(~n', []),
    format(S, '    ptr %norm_q, ptr %norm_d,~n', []),
    format(S, '    ptr %wg_q, ptr %wg_d, ptr %wu_q, ptr %wu_d, ptr %wd_q, ptr %wd_d,~n', []),
    format(S, '    ptr %gate_f, ptr %up_f, ptr %fused_f, ptr %qf_q, ptr %qf_d, ptr %out_f) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  ; gate = W_gate · norm   (gemv over n_ff rows, nb=~w blocks/row, ncols_dst=1)~n', [NbEmbd]),
    format(S, '  call void @bpd_soa_gemv_q8_0(ptr %wg_q, ptr %wg_d, ptr %norm_q, ptr %norm_d, ptr %gate_f, i32 ~w, i32 ~w, i32 1)~n', [NFF, NbEmbd]),
    format(S, '  ; up = W_up · norm~n', []),
    format(S, '  call void @bpd_soa_gemv_q8_0(ptr %wu_q, ptr %wu_d, ptr %norm_q, ptr %norm_d, ptr %up_f, i32 ~w, i32 ~w, i32 1)~n', [NFF, NbEmbd]),
    format(S, '  ; fused = silu(gate) * up   (SURVIVING SwiGLU atom — the fusion, by construction)~n', []),
    format(S, '  call void @bpd_swiglu_fused_cpu(ptr %gate_f, ptr %up_f, ptr %fused_f, i32 ~w)~n', [NFF]),
    format(S, '  ; qfused = quantize(fused)   (rung-3 crux: byte-exact re-quantization)~n', []),
    format(S, '  call void @bpd_q8_0_quantize(ptr %fused_f, ptr %qf_q, ptr %qf_d, i32 ~w)~n', [NFF]),
    format(S, '  ; down = W_down · qfused   (gemv over n_embd rows, nb=~w blocks/row)~n', [NbFF]),
    format(S, '  call void @bpd_soa_gemv_q8_0(ptr %wd_q, ptr %wd_d, ptr %qf_q, ptr %qf_d, ptr %out_f, i32 ~w, i32 ~w, i32 1)~n', [NEmbd, NbFF]),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).

:- end_object.
