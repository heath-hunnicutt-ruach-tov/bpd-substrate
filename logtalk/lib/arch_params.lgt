%% arch_params.lgt — B2 leaf migration #1 (from lib/arch_params.pl).
%% The Collective's first migrated Logtalk object.
%%
%% MIGRATION CONTRACT (B1_CENSUS_AND_MAP.md): clause bodies VERBATIM;
%% Logtalk changes the calling structure, not the logic. The facts
%% below are byte-copied from lib/arch_params.pl (modulo the module/2
%% directive → object wrapper). Emission gate must stay green (this
%% file is not in the generator emission path — its gate is
%% derivation-equivalence: same queries, same answers, both hosts).

:- protocol(arch_paramsp).
    %% The dispatch contract for architecture parameter lookup.
    :- public(arch_family/2).   % arch_family(?ArchName, ?Family)
    :- public(arch_param/3).    % arch_param(+ArchName, +ParamName, -Value)
:- end_protocol.

:- object(arch_params,
    implements(arch_paramsp)).

    %% ═══ FAMILY CLASSIFICATION (verbatim from arch_params.pl) ═══
    arch_family(llama, transformer).
    arch_family(qwen2, transformer).
    arch_family(falcon, transformer).
    arch_family(gemma, transformer).
    arch_family(gemma2, transformer).
    arch_family(phi3, transformer).
    arch_family(starcoder2, transformer).
    arch_family(gpt2, transformer).
    arch_family(bloom, transformer).
    arch_family(deepseek, transformer).
    arch_family(mistral, transformer).
    arch_family(granite, transformer).
    arch_family(mamba, ssm).
    arch_family(falcon_h1, hybrid).
    arch_family(jamba, hybrid).
    arch_family(rwkv6, rwkv).
    arch_family(rwkv7, rwkv).

    %% ═══ ARCH PARAMETERS (verbatim from arch_params.pl) ═══
    arch_param(qwen2, norm_type, rms).
    arch_param(qwen2, has_rope_factors, false).
    arch_param(qwen2, kq_scale, fixed).
    arch_param(qwen2, has_kq_norm, false).
    arch_param(qwen2, has_ffn_bias, false).
    arch_param(qwen2, has_moe, false).
    arch_param(qwen2, has_output_bias, true).
    arch_param(qwen2, qkv_style, separate).
    arch_param(qwen2, ffn_activation, silu).
    arch_param(qwen2, ffn_mode, parallel).
    arch_param(qwen2, has_ffn_gate, true).
    arch_param(qwen2, residual_style, single).
    arch_param(qwen2, position_type, rope).
    arch_param(llama, norm_type, rms).
    arch_param(llama, has_rope_factors, true).
    arch_param(llama, kq_scale, configurable).
    arch_param(llama, has_kq_norm, true).
    arch_param(llama, has_ffn_bias, true).
    arch_param(llama, has_moe, true).
    arch_param(llama, has_output_bias, false).
    arch_param(llama, qkv_style, separate).
    arch_param(llama, ffn_activation, silu).
    arch_param(llama, ffn_mode, parallel).
    arch_param(llama, has_ffn_gate, true).
    arch_param(llama, residual_style, single).
    arch_param(llama, position_type, rope).
    arch_param(falcon, norm_type, layer).
    arch_param(falcon, has_rope_factors, false).
    arch_param(falcon, kq_scale, fixed).
    arch_param(falcon, has_kq_norm, false).
    arch_param(falcon, has_ffn_bias, false).
    arch_param(falcon, has_moe, false).
    arch_param(falcon, has_output_bias, false).
    arch_param(falcon, qkv_style, fused).
    arch_param(falcon, ffn_activation, gelu).
    arch_param(falcon, ffn_mode, sequential).
    arch_param(falcon, has_ffn_gate, false).
    arch_param(falcon, residual_style, double).
    arch_param(falcon, position_type, rope).

:- end_object.
