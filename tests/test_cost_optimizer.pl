%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

:- use_module(library(plunit)).
:- use_module('../lib/cost_model').
:- use_module('../lib/cost_optimizer').
:- use_module('../lib/graph_complexity').
:- use_module('../lib/matmul_cycle_model').

:- begin_tests(cost_optimizer).

test(op_cost_matmul, [nondet]) :-
    %% Test that op_cost can evaluate a matmul and return a reasonable time
    op_cost(matmul, [1024, 1024, 1024], sm_61, CostUS),
    number(CostUS),
    CostUS > 0, !.

test(graph_cost_simple, [nondet]) :-
    %% Test that graph_cost evaluates a simple graph
    Graph = [
        op(op1), op_kind(op1, matmul), tensor_shape(op1, [1024, 1024, 1024]), op_inputs(op1, [w, x]), op_output(op1, t1),
        op(op2), op_kind(op2, silu), tensor_shape(op2, [1024, 1024]), op_inputs(op2, [t1]), op_output(op2, t2)
    ],
    graph_cost(Graph, sm_61, CostUS),
    number(CostUS),
    CostUS > 0, !.

test(optimal_fusion_plan_epilogue, [nondet]) :-
    %% Test that the optimizer fuses an epilogue because it reduces cost
    %% Need to use classify_op compatible kinds and ensure region_inference works
    Graph = [
        op(op1), op_kind(op1, matmul), tensor_shape(op1, [1024, 1024, 1024]), op_inputs(op1, [w, x]), op_output(op1, t1),
        op(op2), op_kind(op2, ggml_silu), tensor_shape(op2, [1024, 1024]), op_inputs(op2, [t1]), op_output(op2, t2)
    ],
    optimal_fusion_plan(Graph, sm_61, FusedGraph),
    
    %% The graph should be fused
    member(op_kind(_, fused(matmul, ggml_silu)), FusedGraph),
    
    %% The cost of the fused graph should be lower than the unfused graph
    graph_cost(Graph, sm_61, UnfusedCost),
    graph_cost(FusedGraph, sm_61, FusedCost),
    %% Currently, our simple cost model assigns fused ops the same cost as base ops,
    %% so FusedCost should be strictly less than UnfusedCost because the standalone
    %% elementwise op's cost is eliminated.
    FusedCost < UnfusedCost, !.

:- end_tests(cost_optimizer).
