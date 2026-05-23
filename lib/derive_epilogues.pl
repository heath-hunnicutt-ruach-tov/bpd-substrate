:- module(derive_epilogues, [
    derive_epilogue/3,
    derive_chain/3,
    activation_def/2,
    demonstrate_derivation/0,
    emit_c/2
]).

%% ═══════════════════════════════════════════════════════════════
%% Primitive C operations — the atoms of our expression language.
%% Each c_prim maps a symbolic function to a C AST node.
%% ═══════════════════════════════════════════════════════════════

c_prim(max, [A, B], c_call(fmaxf, [A, B])).
c_prim(min, [A, B], c_call(fminf, [A, B])).
c_prim(exp, [A],    c_call(expf, [A])).
c_prim(log, [A],    c_call(logf, [A])).
c_prim(log1p, [A],  c_call(log1pf, [A])).
c_prim(tanh, [A],   c_call(tanhf, [A])).
c_prim(abs, [A],    c_call(fabsf, [A])).
c_prim(erf, [A],    c_call(erff, [A])).
c_prim(expm1, [A],  c_call(expm1f, [A])).
c_prim(neg, [A],    c_unop('-', A)).
c_prim(add, [A, B], c_binop('+', A, B)).
c_prim(sub, [A, B], c_binop('-', A, B)).
c_prim(mul, [A, B], c_binop('*', A, B)).
c_prim(div, [A, B], c_binop('/', A, B)).
c_prim(gt, [A, B],  c_binop('>', A, B)).
c_prim(ternary, [C, T, F], c_ternary(C, T, F)).

%% ═══════════════════════════════════════════════════════════════
%% Activation definitions as symbolic math.
%% Each activation_def(Name, Expr) uses the variable 'x' as input.
%% ═══════════════════════════════════════════════════════════════

activation_def(relu,        max(const(0.0), x)).
activation_def(silu,        div(x, add(const(1.0), exp(neg(x))))).
activation_def(swish, E) :- activation_def(silu, E).
activation_def(sigmoid,     div(const(1.0), add(const(1.0), exp(neg(x))))).
activation_def(tanh,        tanh(x)).
%% GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
%% Matches PyTorch's default GELU which uses erff, not the tanh approximation.
activation_def(gelu,        mul(mul(const(0.5), x),
                                add(const(1.0),
                                    erf(mul(x, const(0.7071067811865476)))))).
activation_def(mish,        mul(x, tanh(log1p(exp(x))))).
activation_def(leaky_relu,  ternary(gt(x, const(0.0)), x, mul(const(0.01), x))).
%% ELU: use expm1 for numerical stability near zero (matches PyTorch)
activation_def(elu,         ternary(gt(x, const(0.0)), x, expm1(x))).
activation_def(hardswish,   mul(x, div(min(max(add(x, const(3.0)), const(0.0)), const(6.0)), const(6.0)))).
activation_def(hardsigmoid, div(min(max(add(x, const(3.0)), const(0.0)), const(6.0)), const(6.0))).
activation_def(hardtanh,    min(max(x, const(-1.0)), const(1.0))).
activation_def(abs,         abs(x)).
activation_def(neg,         neg(x)).
activation_def(softplus,    log1p(exp(x))).
%% SELU: matches PyTorch's float-truncated constants and expm1
activation_def(selu,        ternary(gt(x, const(0.0)),
                                mul(const(1.0507010), x),
                                mul(const(1.7580993), expm1(x)))).
%% Note: 1.7580993 = alpha * scale = 1.6732632 * 1.0507009
%% PyTorch computes negcoef = alpha * scale at float precision.

%% ═══════════════════════════════════════════════════════════════
%% The DERIVATION ENGINE: symbolic math → C AST
%% This is the core program transform.
%% ═══════════════════════════════════════════════════════════════

derive_expr(x, InputVar, InputVar).
derive_expr(const(V), _, c_float_f(V)).
derive_expr(Expr, InputVar, CAST) :-
    Expr \= x,
    Expr \= const(_),
    Expr =.. [Func | Args],
    maplist(derive_one(InputVar), Args, CArgs),
    c_prim(Func, CArgs, CAST).

derive_one(InputVar, Arg, CArg) :-
    derive_expr(Arg, InputVar, CArg).

%% Public API
derive_epilogue(Name, InputVar, CAST) :-
    activation_def(Name, MathExpr),
    derive_expr(MathExpr, InputVar, CAST).

derive_chain([], Var, Var).
derive_chain([Act | Rest], InputVar, CAST) :-
    derive_epilogue(Act, InputVar, Mid),
    derive_chain(Rest, Mid, CAST).

%% ═══════════════════════════════════════════════════════════════
%% Demonstration
%% ═══════════════════════════════════════════════════════════════

demonstrate_derivation :-
    format("=== Derived Epilogue Expressions ===~n~n"),
    forall(
        (   member(Act, [relu, silu, sigmoid, tanh, gelu, mish,
                         leaky_relu, elu, hardswish, hardsigmoid,
                         hardtanh, abs, neg, softplus, selu]),
            derive_epilogue(Act, c_var(x), CAST)),
        format("  ~w(x) = ~w~n", [Act, CAST])
    ),
    nl,
    format("=== Derived Chain Compositions ===~n~n"),
    forall(
        (   member((Label, Ops), [
                ("Conv+ReLU+HardSwish", [relu, hardswish]),
                ("Conv+Tanh+HardSwish", [tanh, hardswish]),
                ("Linear+Hardtanh+GELU", [hardtanh, gelu]),
                ("ReLU+Sigmoid+Neg", [relu, sigmoid, neg])
            ]),
            derive_chain(Ops, c_var(x), CAST)),
        format("  ~w:~n    ~w~n~n", [Label, CAST])
    ).

%% ═══════════════════════════════════════════════════════════════
%% C AST Pretty-Printer
%% ═══════════════════════════════════════════════════════════════

emit_c(S, c_var(Name)) :- format(S, "~w", [Name]).
emit_c(S, c_float_f(V)) :- format(S, "~wf", [V]).
emit_c(S, c_call(Func, Args)) :-
    format(S, "~w(", [Func]),
    emit_c_args(S, Args),
    write(S, ')').
emit_c(S, c_binop(Op, L, R)) :-
    write(S, '('),
    emit_c(S, L),
    format(S, " ~w ", [Op]),
    emit_c(S, R),
    write(S, ')').
emit_c(S, c_unop(Op, A)) :-
    format(S, "(~w", [Op]),
    emit_c(S, A),
    write(S, ')').
emit_c(S, c_ternary(Cond, T, F)) :-
    write(S, '('),
    emit_c(S, Cond),
    write(S, ' ? '),
    emit_c(S, T),
    write(S, ' : '),
    emit_c(S, F),
    write(S, ')').

emit_c_args(_, []).
emit_c_args(S, [A]) :- emit_c(S, A).
emit_c_args(S, [A, B|Rest]) :- emit_c(S, A), write(S, ', '), emit_c_args(S, [B|Rest]).
