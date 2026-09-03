:- protocol(kernel_templates_stencilp).
    :- public(jacobi1d_kernel/2).
    :- public(jacobi1d_wrapper/2).
    :- public(kernel_available_fixes/2).
    :- public(fix_description/2).
:- end_protocol.

:- object(kernel_templates_stencil,
    implements(kernel_templates_stencilp)).

    :- uses(c_ast, [emit_c/2, emit_program/2, emit_to_file/2, ast_uses_var/2]).





:- dynamic kernel_available_fixes/2.
:- dynamic fix_description/2.
:- discontiguous kernel_available_fixes/2.
:- discontiguous fix_description/2.


%% =============================================================================
%% JACOBI1D — The family's minimum viable representative
%% =============================================================================
%%
%% Algorithm: out[i] = (in[i-1] + in[i] + in[i+1]) / 3
%%
%% This is the box-filter averaging stencil. It's the simplest member of
%% the family and serves as the substrate's verification that the family
%% pattern emits cleanly.
%%
%% Boundary handling: Dirichlet. Boundary cells (i=0, i=N-1) are NOT
%% written; they retain whatever values the input had at those positions.
%% The caller's responsibility to ensure boundary values are preserved
%% across iterations (e.g., by initializing both `in` and `out` buffers
%% with the same boundary state before the first iteration).
%%
%% Used in:
%%   - 1D smoothing iterations (PDE solver pre-conditioning)
%%   - Iterative Poisson solver as the relaxation step
%%   - 1D box filter in signal processing
%%   - 1D average pooling in ML (with kernel size 3, stride 1)
%%
%% Verification (Python reference + harness will verify):
%%   - Symmetric IC (constant value c) → out = c at every interior cell
%%   - Boundary preservation: out[0] and out[N-1] are not modified
%%   - Sinusoidal IC: analytical decay rate (2/3 + cos(2π/N)/3) per iteration

jacobi1d_kernel(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'in'),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), 'N')],
        [%% int i = blockIdx.x * blockDim.x + threadIdx.x;
         c_decl_init(c_type(int), i,
             c_binop('+',
                 c_binop('*',
                     c_member(c_var(blockIdx), x),
                     c_member(c_var(blockDim), x)),
                 c_member(c_var(threadIdx), x))),
         %% Dirichlet skip: boundary cells (i=0 or i>=N-1) are not written.
         %% Combined into one return: if i <= 0 || i >= N - 1 then return.
         %% (i can be < 0 if launched with too-wide grid, defensively
         %% guarded.)
         c_if(c_binop('||',
                  c_binop('<=', c_var(i), c_int(0)),
                  c_binop('>=', c_var(i),
                      c_binop('-', c_var('N'), c_int(1)))),
              [c_return_void]),
         %% Family-pattern neighborhood reads:
         %%   float L = in[i - 1];   left neighbor
         %%   float C = in[i];       center
         %%   float R = in[i + 1];   right neighbor
         c_decl_init(c_type(float), 'L',
             c_index(c_var('in'), c_binop('-', c_var(i), c_int(1)))),
         c_decl_init(c_type(float), 'C',
             c_index(c_var('in'), c_var(i))),
         c_decl_init(c_type(float), 'R',
             c_index(c_var('in'), c_binop('+', c_var(i), c_int(1)))),
         %% Jacobi1D-specific body: out[i] = (L + C + R) / 3.0f
         %%
         %% Note the c_paren around the numerator. The substrate-precedence
         %% audit (commit 890bfc986) found that c_binop('/', N, D) emits
         %% without parens, so multi-term denominators (or numerators that
         %% combine with later context) need explicit c_paren. Here the
         %% denominator is a single literal, but the numerator MUST be
         %% paren'd because (L + C + R) / 3.0f and L + C + R / 3.0f parse
         %% differently in C.
         c_assign(c_index(c_var(out), c_var(i)),
             c_binop('/',
                 c_paren(c_binop('+',
                     c_binop('+', c_var('L'), c_var('C')),
                     c_var('R'))),
                 c_float_f(3.0)))]).


%% jacobi1d_wrapper(+KName, -Wrapper)
%%
%% C-API wrapper for ctypes loading via the harness. Matches the ML and
%% CFD wrapper conventions: gpu_<name> with launch geometry.

jacobi1d_wrapper(KName, Wrapper) :-
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'in'),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), 'N')],
        [c_cuda_launch(KName,
            c_binop('/',
                c_paren(c_binop('+', c_var('N'), c_int(255))),
                c_int(256)),
            c_int(256),
            [c_var('in'), c_var(out), c_var('N')])]).


%% =============================================================================
%% FIX-FLAG METADATA
%% =============================================================================
%%
%% Per the substrate-honesty convention: stencils in this family are
%% PDE-against-analytical (or convolution-against-mathematical-definition,
%% depending on the member). The reference IS the mathematics, not another
%% implementation. No defects to inherit, no fixes to enumerate.
%%
%% The fix-flag mechanism is READY for if/when we later subsume a specific
%% production implementation (e.g., a PolyBench/GPU reference with known
%% numerical defects).

kernel_available_fixes(k_jacobi1d, []).

%% No fix_description/2 facts registered yet for this family.

:- end_object.
