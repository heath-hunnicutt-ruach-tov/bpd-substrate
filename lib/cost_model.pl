%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% cost_model.pl — Hardware-aware cost estimation for operations and graphs
%%
%% Computes execution time (in microseconds) for single operations and
%% entire graphs, taking into account hardware constraints, fusion, and
%% memory traffic.

:- module(cost_model, [
    op_cost/4,            % +OpKind, +Shape, +Platform, -CostUS
    graph_cost/3          % +GraphFacts, +Platform, -TotalCostUS
]).

:- use_module(graph_complexity, [
    op_flops/3,
    op_memory/3
]).

%% Dummy hardware constants since graph_complexity doesn't export them
hw_peak_gflops(5500.0).
hw_peak_bandwidth_gbs(150.0).
hw_launch_overhead_us(9.0).

:- use_module(matmul_cycle_model, [
    predict_matmul/5
]).

:- use_module(matmul_optimizer, [
    valid_config/1,
    config_metrics/2
]).

%% ────────────────────────────────────────────────────────────────────
%% op_cost(+OpKind, +Shape, +Platform, -CostUS)
%% ────────────────────────────────────────────────────────────────────
%% Estimates the execution time of a single operation in microseconds.

%% 1. Matmul: Use the cycle-accurate model and constraint solver to find the best tile.
op_cost(matmul, [M, N, K], _Platform, CostUS) :-
    %% Find all valid configs and their cycle-accurate predictions
    findall(Pred,
        (valid_config(config(Tile, RegH, RegW, _DoubleBuf, _VecW)),
         %% predict_matmul expects config(Tile, RegH, RegW, KTile)
         %% For simplicity, we use KTile=8 which is a common default
         predict_matmul(config(Tile, RegH, RegW, 8), M, N, K, Pred)),
        Predictions),
    %% Sort by time_us (ascending)
    sort(time_us, @=<, Predictions, [BestPred|_]),
    CostUS = BestPred.time_us.

op_cost(gemm, [M, N, K], Platform, CostUS) :-
    op_cost(matmul, [M, N, K], Platform, CostUS).

%% 2. Fused Ops: Compute cost of the base spatial op, but adjust memory traffic
%% for the epilogue. For simplicity, if the base is matmul, we assume the compute
%% hides the epilogue FLOPs, and memory traffic is just the final output.
op_cost(fused(BaseKind, _EpilogueKind), Shape, Platform, CostUS) :-
    %% If the base is matmul, the cost is dominated by the matmul compute.
    %% The epilogue is fused into the final store.
    %% We approximate this by just taking the base cost.
    (BaseKind = matmul ; BaseKind = gemm),
    !,
    op_cost(BaseKind, Shape, Platform, CostUS).

%% 3. Elementwise / Reductions: Roofline model
op_cost(OpKind, Shape, _Platform, CostUS) :-
    OpKind \= matmul, OpKind \= gemm, OpKind \= fused(_, _),
    %% Fallback to a dummy cost if op_flops/op_memory fails (e.g. for ggml_silu)
    ( op_flops(OpKind, Shape, Flops) -> true ; Flops = 1024.0 ),
    ( op_memory(OpKind, Shape, Bytes) -> true ; Bytes = 4096.0 ),
    hw_peak_gflops(PeakGF),
    hw_peak_bandwidth_gbs(PeakBW),
    hw_launch_overhead_us(LaunchUS),
    
    %% Time to compute (us) = FLOPs / (GFLOPS * 1000)
    ComputeUS is Flops / (PeakGF * 1000.0),
    
    %% Time to transfer (us) = Bytes / (GB/s * 1000)
    MemoryUS is Bytes / (PeakBW * 1000.0),
    
    %% Roofline: max of compute and memory time, plus launch overhead
    CostUS is max(ComputeUS, MemoryUS) + LaunchUS.

%% ────────────────────────────────────────────────────────────────────
%% graph_cost(+GraphFacts, +Platform, -TotalCostUS)
%% ────────────────────────────────────────────────────────────────────
%% Estimates the total execution time of a compute graph.

graph_cost(GraphFacts, Platform, TotalCostUS) :-
    findall(Cost,
        (member(op(OpId), GraphFacts),
         member(op_kind(OpId, Kind), GraphFacts),
         member(op_output(OpId, _OutId), GraphFacts),
         %% Extract shape (simplified: assume 2D/3D shapes are available or inferrable)
         %% In a full implementation, we'd look up the shape of OutId or the inputs.
         %% For this model, we'll extract shapes if available, else default to a dummy shape.
         get_shape(GraphFacts, OpId, Shape),
         op_cost(Kind, Shape, Platform, Cost)),
        Costs),
    sum_list(Costs, TotalCostUS).

%% Helper to get shape (dummy implementation for now, should use real tensor shapes)
get_shape(GraphFacts, OpId, Shape) :-
    ( member(tensor_shape(OpId, Shape), GraphFacts)
    -> true
    ;  %% Default dummy shape if not provided
       Shape = [1024, 1024, 1024]
    ).
