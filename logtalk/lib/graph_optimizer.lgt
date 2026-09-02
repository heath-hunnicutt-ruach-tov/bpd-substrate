:- protocol(graph_optimizerp).
    :- public(fuse_graph/3).
:- end_protocol.

:- object(graph_optimizer,
    implements(graph_optimizerp)).

    :- uses(list, [member/2]).



    %% Both dependencies are migrated sibling objects:
    :- uses(fusible, [fusible_pair/4]).
    :- uses(apply_fusion, [apply_fusion_to_facts/3]).

%% ────────────────────────────────────────────────────────────────────
%% fuse_graph(+GraphFacts, +Platform, -FusedPlan)
%% ────────────────────────────────────────────────────────────────────
%% Takes a list of BPD facts representing the compute graph,
%% applies valid fusions, and returns the optimized graph.

%! fuse_graph(+GraphFacts, +Platform, -FusedPlan) is det.
%  Optimize a compute graph for Platform by iteratively applying
%  fusion rules until fixed-point. Returns the optimized FusedPlan.
fuse_graph(GraphFacts, Platform, FusedPlan) :-
    fuse_graph_iter(GraphFacts, Platform, FusedPlan).

%% Iteratively find and apply fusions until no more are possible.
fuse_graph_iter(Facts, Platform, FinalFacts) :-
    %% Find the first valid fusion pair
    (   find_fusion(Facts, Fusion)
    ->  %% Apply it
        apply_fusion_to_facts(Facts, Fusion, NewFacts),
        %% Recurse
        fuse_graph_iter(NewFacts, Platform, FinalFacts)
    ;   %% Fixed point reached
        FinalFacts = Facts
    ).

%% Find a valid fusion pair in the graph.
find_fusion(Facts, Fusion) :-
    %% We need to pick two ops that are fusible.
    %% To avoid infinite loops, we just pick the first valid one.
    member(op(Op1), Facts),
    member(op(Op2), Facts),
    Op1 \= Op2,
    fusible_pair(Facts, Op1, Op2, Fusion),
    !. % Take the first one we find

:- end_object.
