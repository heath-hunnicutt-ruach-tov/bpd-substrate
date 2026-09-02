:- protocol(gpu_roofline_analysisp).
    :- public(roofline_analysis/4).
    :- public(explain_gap/4).
:- end_protocol.

:- object(gpu_roofline_analysis,
    implements(gpu_roofline_analysisp)).

    :- uses(list, [member/2]).



%% measurement_group(Name, Description, Events).
measurement_group(memory_hierarchy, "Where do bytes come from?", [
    fb_subp0_read_sectors, fb_subp1_read_sectors,
    tex0_cache_sector_queries, tex0_cache_sector_misses,
    elapsed_cycles_sm, active_cycles_pm
]).

measurement_group(coalescing, "How efficiently do loads access memory?", [
    gld_inst_8bit, gld_inst_16bit, gld_inst_32bit,
    gld_inst_64bit, gld_inst_128bit,
    fb_subp0_read_sectors, fb_subp1_read_sectors
]).

measurement_group(compute, "How busy are the SMs?", [
    active_warps_pm, active_cycles_pm,
    elapsed_cycles_sm, elapsed_cycles_pm
]).

measurement_group(l2_cache, "L2 cache behavior", [
    l2_subp0_read_sector_misses, l2_subp1_read_sector_misses,
    l2_subp0_read_tex_sector_queries, l2_subp0_read_tex_hit_sectors
]).

%% roofline_analysis(M, K, N, Analysis)
%% Run all measurement groups, produce a structured analysis.
roofline_analysis(M, K, N, Analysis) :-
    findall(Group-Events-Values,
        (measurement_group(Group, _, Events),
         collect_group(M, K, N, Events, Values)),
        Measurements),
    compute_analysis(M, K, N, Measurements, Analysis).

%% The gap decomposition
explain_gap(M, K, N, Explanation) :-
    roofline_analysis(M, K, N, A),
    PeakBW = 192.0,  % GB/s
    member(achieved_bw-AchievedBW, A),
    Gap is PeakBW - AchievedBW,
    member(sm_utilization-SMUtil, A),
    member(cache_hit_rate-CacheHit, A),
    member(load_efficiency-LoadEff, A),
    Explanation = gap_analysis{
        peak_bw: PeakBW,
        achieved_bw: AchievedBW,
        gap_gbps: Gap,
        gap_pct: Gap / PeakBW * 100,
        factors: [
            sm_utilization: SMUtil,
            cache_hit_rate: CacheHit,
            load_efficiency: LoadEff
        ]
    }.

%% Placeholder for the CUPTI collection (wired via the C bridge)
collect_group(M, K, N, Events, Values) :-
    format("Collecting ~w for matmul [~w x ~w] x ~w~n", [Events, M, K, N]),
    Values = [].  % TODO: wire to gpu_profile_matmul

compute_analysis(_M, _K, _N, _Measurements, []) :-
    true.  % TODO: derive analysis from measurements

:- end_object.
