:- protocol(cost_shape_cupti_validationp).
    :- public(predicted_bottleneck/2).
    :- public(profile_kernel/3).
    :- public(fusion_validated/3).
    :- public(stall_of/3).
:- end_protocol.

:- object(cost_shape_cupti_validation,
    implements(cost_shape_cupti_validationp)).



%% PREDICTION (cost-shape, no GPU): a kernel with an eliminable {k}[N]
%% producer->consumer seam is memory-bound -> predicted stall = memory_dependency.
predicted_bottleneck(Kernel, memory_dependency) :-
    has_eliminable_materialization(Kernel).

:- dynamic has_eliminable_materialization/1.

%% MEASUREMENT (CUPTI): global reset->run->flush->report cycle.
%% run_kernel/2 = launch hook (to be wired to cuda_launch PLF).
profile_kernel(Kernel, Config, Stalls) :-
    cupti_init,
    cupti_reset,
    run_kernel(Kernel, Config),
    cupti_flush,
    cupti_stall_report(Stalls).

stall_of(Stalls, Reason, Pct) :-
    member(Reason-Pct, Stalls).

%% VALIDATION: predicted==measured==reduced-by-fusion (not just "faster").
%%
%% The validation is TIGHT:
%%   - cost-shape predicts memory_dependency bottleneck
%%   - CUPTI confirms memory_dependency > 30%
%%   - fusion is proposed from cost-shape analysis
%%   - CUPTI re-measures: memory_dependency DECREASED
%%
%% Returns validated(Before, After, Delta) as evidence.
fusion_validated(Kernel, Config, validated(P0, P1, Delta)) :-
    predicted_bottleneck(Kernel, memory_dependency),
    profile_kernel(Kernel, Config, S0),
    stall_of(S0, memory_dependency, P0),
    P0 > 30,
    propose_fusion(Kernel, Fused),
    profile_kernel(Fused, Config, S1),
    stall_of(S1, memory_dependency, P1),
    P1 < P0,
    Delta is P0 - P1.

%% Integration hooks — to be wired by mavchin (launch) and Iyun (fusion)
:- dynamic run_kernel/2.
:- dynamic propose_fusion/2.

:- end_object.
