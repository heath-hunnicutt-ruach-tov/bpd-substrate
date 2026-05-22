%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% fusible.pl — Declarative fusion legality predicate
%%
%% Determines whether two operations can be legally fused into a single kernel,
%% based on their operation classification, semantic regions, and platform constraints.

/** <module> Declarative Fusion Legality

Four fusion rules expressed as Prolog clauses. Adding a new fusion
pattern = adding one clause. Each rule specifies when two adjacent
operations in a compute graph can be fused into a single kernel.

Rules:
  1. Epilogue fusion — spatial op followed by elementwise (matmul+relu)
  2. Elementwise chain — elementwise followed by elementwise (silu+mul = SwiGLU)
  3. Layout transparent — reshape elimination between compatible ops
  4. Fused epilogue — epilogue to an already-fused op (matmul+bias+relu)

@author Ruach Tov Collective
@license RTAAL-1.0
@see graph_optimizer.pl for the fixed-point iteration that applies these rules
@see valid_tile.pl for constraint-based tile selection
*/

:- module(fusible, [
    fusible/3,
    fusible_pair/4
]).

:- use_module(auto_fuser, [classify_op/2]).
:- use_module(region_inference, [infer_region_from_facts/5]).

%% ────────────────────────────────────────────────────────────────────
%% fusible(+GraphFacts, +Op1, +Op2)
%% ────────────────────────────────────────────────────────────────────
%% True if Op1 and Op2 can be legally fused given the graph facts.
%% This is a wrapper that delegates to fusible_pair/4, extracting kinds.

%! fusible(+GraphFacts, ?Op1, ?Op2) is nondet.
%  True if Op1 and Op2 can be fused in the compute graph.
%  Enumerates all fusible pairs via backtracking.
fusible(GraphFacts, Op1, Op2) :-
    member(op_kind(Op1, _Kind1), GraphFacts),
    member(op_kind(Op2, _Kind2), GraphFacts),
    fusible_pair(GraphFacts, Op1, Op2, fusion(_RuleName, [Op1, Op2], _EqClass)).

%% ────────────────────────────────────────────────────────────────────
%% fusible_pair(+GraphFacts, +Op1, +Op2, -Fusion)
%% ────────────────────────────────────────────────────────────────────
%% Determines if Op1 and Op2 are fusible and returns the fusion term.

%% Rule 1: Spatial operation followed by elementwise operation (epilogue fusion).
%% Op1's output is consumed by Op2.
fusible_pair(GraphFacts, Op1, Op2, fusion(epilogue_matmul_elementwise, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind1, spatial),
    classify_op(Kind2, elementwise),
    %% Op1 output is consumed by Op2
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    %% Op2 must read Intermediate elementwise or broadcast
    infer_region_from_facts(GraphFacts, Op2, Intermediate, read, region(RegionType, _)),
    (RegionType = elementwise ; RegionType = broadcast).

%% Rule 2: Elementwise chain fusion.
fusible_pair(GraphFacts, Op1, Op2, fusion(elementwise_chain, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, Kind1), GraphFacts),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind1, elementwise),
    classify_op(Kind2, elementwise),
    %% Op1 output is consumed by Op2
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    %% Op2 must read Intermediate elementwise or broadcast
    infer_region_from_facts(GraphFacts, Op2, Intermediate, read, region(RegionType, _)),
    (RegionType = elementwise ; RegionType = broadcast).

%% Rule 4: Epilogue elementwise fusion to already fused matmul
fusible_pair(GraphFacts, Op1, Op2, fusion(epilogue_matmul_elementwise, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, fused(Kind1A, _Kind1B)), GraphFacts),
    %% Check if the original base op was spatial
    classify_op(Kind1A, spatial),
    member(op_kind(Op2, Kind2), GraphFacts),
    classify_op(Kind2, elementwise),
    member(op_output(Op1, Intermediate), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Intermediate, Inputs2),
    infer_region_from_facts(GraphFacts, Op2, Intermediate, read, region(RegionType, _)),
    (RegionType = elementwise ; RegionType = broadcast).

%% Rule 3: Layout transparent fusion (reshape elimination).
%% Op1 is reshape, Op2 is a consumer.
fusible_pair(GraphFacts, Op1, Op2, fusion(layout_transparent, [Op1, Op2], bit_exact)) :-
    member(op_kind(Op1, ggml_reshape_3d), GraphFacts),
    member(op_output(Op1, Reshaped), GraphFacts),
    member(op_inputs(Op2, Inputs2), GraphFacts),
    member(Reshaped, Inputs2).
