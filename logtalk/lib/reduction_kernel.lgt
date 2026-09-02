:- protocol(reduction_kernelp).
    :- public(reduction_pattern/4).
    :- public(enumerate_cascade_patterns/1).
    :- public(cascade_kernel_name/5).
:- end_protocol.

:- object(reduction_kernel,
    implements(reduction_kernelp)).

    :- uses(list, [member/2]).



%% reduction_pattern(+SimdWidth, +IlpFactor, +CascadeDepth, +CascadeBase).
%%
%% Sweepable parameter space:
%%
%%   SimdWidth ∈ {1, 4, 8, 16}     scalar / SSE-NEON / AVX-AVX2 / AVX-512
%%   IlpFactor ∈ {1, 2, 4, 8}      parallel cascade lanes per SIMD register
%%   CascadeDepth ∈ {1, 2, 4, 8}   cascade promotion levels
%%   CascadeBase ∈ {0, 16, 32, 64} level_step value (0 = no cascade)
%%
%% Validity: CascadeBase = 0 iff CascadeDepth = 1 (no cascade means no step).
reduction_pattern(SimdWidth, IlpFactor, CascadeDepth, CascadeBase) :-
    member(SimdWidth, [1, 4, 8, 16]),
    member(IlpFactor, [1, 2, 4, 8]),
    member(CascadeDepth, [1, 2, 4, 8]),
    member(CascadeBase, [0, 16, 32, 64]),
    valid_pattern(CascadeDepth, CascadeBase).

valid_pattern(1, 0).
valid_pattern(D, B) :- D > 1, B > 0.

%% enumerate_cascade_patterns(-Patterns) — returns the full parameter sweep.
%% Used by the sweep harness to know what to test.
enumerate_cascade_patterns(Patterns) :-
    findall(cascade(SW, ILP, CD, CB),
            reduction_pattern(SW, ILP, CD, CB),
            Patterns).

%% cascade_kernel_name(+SW, +ILP, +CD, +CB, -Name).
%% C function name for one pattern. Used by the sweep harness to match
%% generated C functions to their parameter tuples.
cascade_kernel_name(SW, ILP, CD, CB, Name) :-
    format(atom(Name),
           'cascade_sum_simd~w_ilp~w_depth~w_base~w',
           [SW, ILP, CD, CB]).

%% Named platform defaults — which pattern matches each known platform.
%% Read by lib/implementation_matches.pl to add the cascade choice as a
%% platform_param.
%%
%% Tested on Tesla P4 enclave (AVX1, no AVX2): cascade(8, 4, 4, 16) is the
%% PyTorch CPU bit-identical match.
%% Other platforms (untested by us, predicted from PyTorch source):
%%   AVX-512 hosts: cascade(16, 4, 4, 16)
%%   ARM NEON:      cascade(4, 4, 4, 16)
%%   Scalar fallback (verification baseline): cascade(1, 1, 1, 0)
platform_cascade(pytorch_cpu_default_avx1, cascade(8, 4, 4, 16)).
platform_cascade(pytorch_cpu_default_avx512, cascade(16, 4, 4, 16)).
platform_cascade(pytorch_cpu_default_neon, cascade(4, 4, 4, 16)).
platform_cascade(scalar_baseline, cascade(1, 1, 1, 0)).

:- end_object.
