%% implementation_matches.pl — Platform-specific parameter derivation
%%
%% The meta-parameter: implementation_matches(Platform) derives ALL
%% individual substrate parameters needed to produce bit-identical
%% output with that platform.
%%
%% Usage:
%%   ?- implementation_matches(cuBLAS).
%%   % Sets: accumulation_precision(fp32), k_tile_strategy(auto), ...
%%
%%   ?- implementation_matches(pytorch_cpu_default).
%%   % Sets: accumulation_precision(fp32), cpu_fp_mode(strict), ...
%%
%% Adding a new platform: define its parameter combination below.
%% The substrate sweeps these during verification to confirm bit-identity.

:- module(implementation_matches, [
    implementation_matches/1,
    platform_param/2,
    list_platforms/0,
    list_platform_params/1
]).

:- discontiguous implementation_matches/1.
:- discontiguous platform_param/2.

%% ═══════════════════════════════════════════════════════════════
%% Platform definitions
%% ═══════════════════════════════════════════════════════════════

%% NVIDIA cuBLAS (GPU, all architectures)
%% Our primary reference for GPU bit-identity.
implementation_matches(cuBLAS) :-
    platform_param(cuBLAS, accumulation_precision(fp32)),
    platform_param(cuBLAS, k_tile_strategy(auto)),
    platform_param(cuBLAS, reduction_strategy(sequential)),
    platform_param(cuBLAS, bn_mode(multiply_by_reciprocal)),
    platform_param(cuBLAS, rsqrt_variant(hardware)).

platform_param(cuBLAS, accumulation_precision(fp32)).
platform_param(cuBLAS, k_tile_strategy(auto)).
platform_param(cuBLAS, reduction_strategy(sequential)).
platform_param(cuBLAS, bn_mode(multiply_by_reciprocal)).
platform_param(cuBLAS, rsqrt_variant(hardware)).
platform_param(cuBLAS, matmul_backend(ffma)).

%% PyTorch CPU with DEFAULT backend (no MKL, no OpenBLAS)
%% Sequential accumulation, no FMA. Matches our bpd_cpu.so at 0 ULP.
implementation_matches(pytorch_cpu_default) :-
    platform_param(pytorch_cpu_default, accumulation_precision(fp32)),
    platform_param(pytorch_cpu_default, cpu_fp_mode(strict)),
    platform_param(pytorch_cpu_default, bn_mode(precomputed_scale_offset)),
    platform_param(pytorch_cpu_default, reduction_strategy(sequential)).

platform_param(pytorch_cpu_default, accumulation_precision(fp32)).
platform_param(pytorch_cpu_default, cpu_fp_mode(strict)).
platform_param(pytorch_cpu_default, bn_mode(precomputed_scale_offset)).
platform_param(pytorch_cpu_default, reduction_strategy(sequential)).
platform_param(pytorch_cpu_default, rsqrt_variant(reciprocal_sqrt)).

%% PyTorch CPU with MKL/OpenBLAS (AVX2+FMA CPUs)
%% Tiled accumulation, FMA contraction. Different reduction order.
implementation_matches(pytorch_cpu_mkl) :-
    platform_param(pytorch_cpu_mkl, accumulation_precision(fp32)),
    platform_param(pytorch_cpu_mkl, cpu_fp_mode(fma)),
    platform_param(pytorch_cpu_mkl, bn_mode(precomputed_scale_offset)),
    platform_param(pytorch_cpu_mkl, reduction_strategy(tiled)).

platform_param(pytorch_cpu_mkl, accumulation_precision(fp32)).
platform_param(pytorch_cpu_mkl, cpu_fp_mode(fma)).
platform_param(pytorch_cpu_mkl, bn_mode(precomputed_scale_offset)).
platform_param(pytorch_cpu_mkl, reduction_strategy(tiled)).
platform_param(pytorch_cpu_mkl, rsqrt_variant(reciprocal_sqrt)).

%% LAPACK reference (the mathematical gold standard)
%% Double-precision accumulation. Maximum correctness, not bit-compatible.
implementation_matches(lapack_reference) :-
    platform_param(lapack_reference, accumulation_precision(fp64)),
    platform_param(lapack_reference, reduction_strategy(sequential)).

platform_param(lapack_reference, accumulation_precision(fp64)).
platform_param(lapack_reference, reduction_strategy(sequential)).
platform_param(lapack_reference, cpu_fp_mode(strict)).

%% llama.cpp (the Ollama backend)
%% Q4_K_M dequant + fp32 accumulation.
implementation_matches(llama_cpp) :-
    platform_param(llama_cpp, accumulation_precision(fp32)),
    platform_param(llama_cpp, reduction_strategy(sequential)),
    platform_param(llama_cpp, quant_dequant_fused(true)).

platform_param(llama_cpp, accumulation_precision(fp32)).
platform_param(llama_cpp, reduction_strategy(sequential)).
platform_param(llama_cpp, quant_dequant_fused(true)).

%% BPD substrate defaults (our own choices, documented)
implementation_matches(bpd_default) :-
    platform_param(bpd_default, accumulation_precision(fp32)),
    platform_param(bpd_default, k_tile_strategy(k8)),
    platform_param(bpd_default, reduction_strategy(sequential)),
    platform_param(bpd_default, bn_mode(precomputed_scale_offset)),
    platform_param(bpd_default, cpu_fp_mode(strict)),
    platform_param(bpd_default, rsqrt_variant(reciprocal_sqrt)).

platform_param(bpd_default, accumulation_precision(fp32)).
platform_param(bpd_default, k_tile_strategy(k8)).
platform_param(bpd_default, reduction_strategy(sequential)).
platform_param(bpd_default, bn_mode(precomputed_scale_offset)).
platform_param(bpd_default, cpu_fp_mode(strict)).
platform_param(bpd_default, rsqrt_variant(reciprocal_sqrt)).

%% ═══════════════════════════════════════════════════════════════
%% Query helpers
%% ═══════════════════════════════════════════════════════════════

list_platforms :-
    findall(P, (platform_param(P, _), \+ P = bpd_default), Ps),
    sort(Ps, Unique),
    format("Available platforms:~n"),
    forall(member(U, Unique), format("  ~w~n", [U])).

list_platform_params(Platform) :-
    format("Parameters for ~w:~n", [Platform]),
    forall(platform_param(Platform, Param),
           format("  ~w~n", [Param])).

%% ═══════════════════════════════════════════════════════════════
%% Parameter descriptions (for documentation / UI)
%% ═══════════════════════════════════════════════════════════════

:- discontiguous param_description/2.

param_description(accumulation_precision(fp32),
    "IEEE 754 single-precision accumulation. Matches cuBLAS, PyTorch, most GPU libs.").
param_description(accumulation_precision(fp64),
    "Double-precision accumulation. Maximum correctness. 0 ULP vs f64 truth.").
param_description(accumulation_precision(kahan),
    "Kahan compensated summation. Near-f64 accuracy at fp32 speed.").

param_description(k_tile_strategy(auto),
    "Auto-select K_TILE based on shape. Matches cuBLAS dispatch heuristic.").
param_description(k_tile_strategy(k8),
    "K_TILE=8. Matches cuBLAS sgemm_128x128x8 for large/square shapes.").
param_description(k_tile_strategy(k32),
    "K_TILE=32. Matches cuBLAS sgemm_32x32x32 for small/non-square shapes.").

param_description(cpu_fp_mode(strict),
    "No FMA. Sequential accumulation. Matches PyTorch DEFAULT backend.").
param_description(cpu_fp_mode(fma),
    "FMA enabled (-mfma -ffp-contract=on). Matches PyTorch MKL/OpenBLAS on AVX2+.").
param_description(cpu_fp_mode(native),
    "-march=native. Matches whatever your CPU supports.").

param_description(bn_mode(multiply_by_reciprocal),
    "gamma * (1/sqrt(var+eps)). Matches PyTorch GPU BN (DIVSS+MULSS double rounding).").
param_description(bn_mode(precomputed_scale_offset),
    "Precompute scale=gamma/sqrt(var+eps), offset=beta-mean*scale. 2 ops per element.").

param_description(rsqrt_variant(hardware),
    "Use rsqrtf() hardware instruction. Fast, ~1 ULP from 1/sqrt.").
param_description(rsqrt_variant(reciprocal_sqrt),
    "Use 1.0f/sqrtf(). Matches PyTorch CPU eval-mode BN.").

param_description(reduction_strategy(sequential),
    "Left-to-right sequential reduction. Deterministic. Matches most BLAS.").
param_description(reduction_strategy(tiled),
    "Block-tiled reduction. Non-deterministic order. Matches MKL/OpenBLAS.").
param_description(reduction_strategy(pairwise_tree),
    "Pairwise tree reduction. Better numerical stability. Different bits.").
