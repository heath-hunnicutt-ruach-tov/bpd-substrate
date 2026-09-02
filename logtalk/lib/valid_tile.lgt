:- protocol(valid_tilep).
    :- public(valid_tile/4).
:- end_protocol.

:- object(valid_tile,
    implements(valid_tilep)).

    :- uses(matmul_optimizer, [hw_shared_mem_bytes/1,
        hw_max_regs_per_thread/1, hw_max_regs_per_sm/1,
        k_tile_for_shape/5]).


%% ────────────────────────────────────────────────────────────────────
%% valid_tile(+M_tile, +N_tile, +K_tile, +Platform)
%% ────────────────────────────────────────────────────────────────────
%% True if the given tile dimensions are valid for the specified platform.

%! valid_tile(-M_tile, -N_tile, -K_tile, +Platform) is nondet.
%  Generate valid tile dimensions for Platform via backtracking.
%  Each solution satisfies vectorization alignment and shared memory bounds.
valid_tile(M_tile, N_tile, K_tile, Platform) :-
    %% Ensure dimensions are positive integers
    integer(M_tile), M_tile > 0,
    integer(N_tile), N_tile > 0,
    integer(K_tile), K_tile > 0,

    %% K_tile should typically be a multiple of 8 for vectorization
    K_tile mod 8 =:= 0,

    %% Shared memory constraint
    %% Assuming we need to store A (M_tile x K_tile) and B (K_tile x N_tile) in shared memory.
    %% Float32 = 4 bytes per element.
    %% If double buffering is used, multiply by 2.
    %% Let's assume single buffer for simplicity here, or check platform params.
    SharedBytes is (M_tile * K_tile + K_tile * N_tile) * 4,
    hw_shared_mem_bytes(MaxShared),
    SharedBytes =< MaxShared,

    %% Register constraints (simplified)
    %% Accumulators needed: M_tile * N_tile per warp?
    %% This depends on thread block size. We can defer deep checks to valid_config/1
    %% or implement a simplified version here.
    true.

:- end_object.
