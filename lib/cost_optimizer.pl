%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% cost_optimizer.pl — Cost-based fusion selection
%%
%% Uses the cost model to evaluate different fusion paths and select
%% the one that minimizes total execution time.

:- module(cost_optimizer, [
    optimal_fusion_plan/3
]).

:- use_module(fusible, [fusible_pair/4]).
:- use_module(apply_fusion, [apply_fusion_to_facts/3]).
:- use_module(cost_model, [graph_cost/3]).

%% ────────────────────────────────────────────────────────────────────
%% optimal_fusion_plan(+GraphFacts, +Platform, -BestPlan)
%% ────────────────────────────────────────────────────────────────────
%% Finds the sequence of valid fusions that minimizes total graph cost.
%% Uses a greedy approach for now: at each step, evaluate all valid single
%% fusions and pick the one that yields the lowest graph cost.

optimal_fusion_plan(Facts, Platform, FinalFacts) :-
    %% Find all valid fusions from the current state
    findall(Fusion,
        (member(op(Op1), Facts),
         member(op(Op2), Facts),
         Op1 \= Op2,
         fusible_pair(Facts, Op1, Op2, Fusion)),
        ValidFusions),
    
    %% Remove duplicates
    sort(ValidFusions, UniqueFusions),
    
    ( UniqueFusions = []
    -> %% No more fusions possible, we are at a fixed point
       FinalFacts = Facts
    ;  %% Evaluate each fusion
       evaluate_fusions(UniqueFusions, Facts, Platform, ScoredFusions),
       %% Sort by cost (ascending)
       sort(ScoredFusions, [cost_fusion(_MinCost, _BestFusion, BestNewFacts)|_]),
       %% Check if the best fusion actually reduces cost compared to current
       graph_cost(Facts, Platform, CurrentCost),
       graph_cost(BestNewFacts, Platform, BestCost),
       ( BestCost < CurrentCost
       -> %% It's an improvement, apply it and recurse
          optimal_fusion_plan(BestNewFacts, Platform, FinalFacts)
       ;  %% No fusion improves cost (e.g., due to register pressure in extreme cases), stop
          FinalFacts = Facts
       )
    ).

%% Evaluate a list of fusions, returning cost_fusion(Cost, Fusion, NewFacts)
evaluate_fusions([], _Facts, _Platform, []).
evaluate_fusions([Fusion|Rest], Facts, Platform, [cost_fusion(Cost, Fusion, NewFacts)|ScoredRest]) :-
    apply_fusion_to_facts(Facts, Fusion, NewFacts),
    graph_cost(NewFacts, Platform, Cost),
    evaluate_fusions(Rest, Facts, Platform, ScoredRest).
