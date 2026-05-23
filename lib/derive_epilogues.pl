:- module(derive_epilogues, [
    derive_epilogue/3,
    derive_chain/3,
    activation_def/2,
    demonstrate_derivation/0
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
activation_def(gelu,        mul(mul(const(0.5), x),
                                add(const(1.0),
                                    tanh(mul(const(0.7978845608),
                                        add(x, mul(const(0.044715), mul(x, mul(x, x))))))))).
activation_def(mish,        mul(x, tanh(log1p(exp(x))))).
activation_def(leaky_relu,  ternary(gt(x, const(0.0)), x, mul(const(0.01), x))).
activation_def(elu,         ternary(gt(x, const(0.0)), x, sub(exp(x), const(1.0)))).
activation_def(hardswish,   mul(x, div(min(max(add(x, const(3.0)), const(0.0)), const(6.0)), const(6.0)))).
activation_def(hardsigmoid, div(min(max(add(x, const(3.0)), const(0.0)), const(6.0)), const(6.0))).
activation_def(hardtanh,    min(max(x, const(-1.0)), const(1.0))).
activation_def(abs,         abs(x)).
activation_def(neg,         neg(x)).
activation_def(softplus,    log1p(exp(x))).
activation_def(selu,        mul(const(1.0507),
                                ternary(gt(x, const(0.0)), x,
                                    mul(const(1.6733), sub(exp(x), const(1.0)))))).

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
