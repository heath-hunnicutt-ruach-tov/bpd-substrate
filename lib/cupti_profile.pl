%% ═══════════════════════════════════════════════════════════════════════
%% cupti_profile.pl — GPU kernel profiling from Prolog
%%
%% The substrate profiles its own kernels and reasons about the results.
%% Uses SWI-Prolog Foreign Language Interface to call CUPTI directly.
%%
%% Licensed under GPLv2
%% ═══════════════════════════════════════════════════════════════════════

/** <module> CUPTI GPU Kernel Profiling

Profile GPU kernels via CUPTI PC sampling and reason about
warp stall data as Prolog facts. The optimizer queries stall
facts to decide what optimizations to apply.

Example:
==
?- cupti_init,
   % ... launch GPU kernel via ctypes or foreign call ...
   cupti_flush,
   cupti_stall_report(Stalls),
   cupti_suggest(Suggestions),
   cupti_shutdown.

Stalls = [none-3.6, exec_dependency-14.4, memory_dependency-61.8, ...],
Suggestions = [suggest(warp_shuffle, memory_dependency, 61.8), ...]
==

@author Ruach Tov Collective
@see lib/bpd_cupti_profile.c for the CUPTI C implementation
@see lib/cupti_bridge.c for the SWI-Prolog PLF binding
*/

:- module(cupti_profile, [
    cupti_init/0,
    cupti_shutdown/0,
    cupti_reset/0,
    cupti_flush/0,
    cupti_stall_report/1,
    cupti_total_samples/1,
    cupti_suggest/1,
    optimization_needed/2,
    stall_threshold/3,
    profile_summary/1
]).

%% Load the foreign library
%% Build with: swipl-ld -shared -o cupti_bridge lib/cupti_bridge.c lib/bpd_cupti_profile.c -lcupti -lcuda
:- use_foreign_library(foreign(cupti_bridge)).

%% ═══════════════════════════════════════════════════════════════════════
%% Stall thresholds — when to suggest each optimization
%% ═══════════════════════════════════════════════════════════════════════

%! stall_threshold(+StallType, +ThresholdPct, -Optimization) is nondet.
%  When StallType exceeds ThresholdPct, suggest Optimization.

stall_threshold(memory_dependency, 30, warp_shuffle).
stall_threshold(memory_dependency, 15, shared_memory).
stall_threshold(exec_dependency, 20, increase_ilp).
stall_threshold(sync, 15, reduce_barriers).
stall_threshold(memory_throttle, 10, coalesce_memory).
stall_threshold(constant_memory, 10, use_registers).

%% ═══════════════════════════════════════════════════════════════════════
%% High-level reasoning predicates
%% ═══════════════════════════════════════════════════════════════════════

%! optimization_needed(+StallList, -Optimization) is nondet.
%  Given a stall report list, determine what optimizations are needed.

optimization_needed(StallList, Optimization) :-
    member(StallType-Pct, StallList),
    stall_threshold(StallType, Threshold, Optimization),
    Pct > Threshold.

%! profile_summary(+StallList) is det.
%  Print a human-readable summary of stall data with suggestions.

profile_summary(StallList) :-
    format("~n=== Stall Analysis ===~n"),
    forall(
        member(Type-Pct, StallList),
        format("  ~w: ~1f%~n", [Type, Pct])
    ),
    format("~n=== Suggestions ===~n"),
    forall(
        optimization_needed(StallList, Opt),
        format("  → ~w~n", [Opt])
    ),
    ( \+ optimization_needed(StallList, _) ->
        format("  → Kernel is well-optimized~n")
    ; true
    ).
