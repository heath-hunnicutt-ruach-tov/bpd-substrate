:- protocol(c_ast_legacyp).
    :- public(c_tokenize_enriched/2).
    :- public(c_enrich_tokens/2).
    :- public(c_parse_stmts/2).
    :- public(c_parse_stmt/2).
    :- public(c_parse_type/2).
    :- public(c_parse_tokens/2).
    :- public(c_parse_chain/2).
    :- public(c_parse_full_expr/2).
:- end_protocol.

:- object(c_ast_legacy,
    implements(c_ast_legacyp)).



:- uses(c_ast, [emit_c/2, emit_program/2, emit_to_file/2, ast_uses_var/2]).


%% ─── v1 entry points ─────────────────────────────────────────────
%% Each predicate calls a DCG rule that is still defined in c_ast.pl.
%% The DCG rules remain in c_ast.pl pending separate cleanup.

%% Enriched tokenizer (v1): classifies keywords and unambiguous operators.
%% Superseded by c_tokenize_enriched_v2 (which adds semicolon-as-own-token).
%% Reached only by these (now extracted) entry points; zero external callers.
c_tokenize_enriched(String, Tokens) :-
    c_ast:c_tokenize(String, RawTokens),
    c_enrich_tokens(RawTokens, Tokens).

%% Helper for c_tokenize_enriched (v1 enrichment rules).
%% Reclassifies id() → keyword() and punct() → operator() for unambiguous ops.
c_enrich_tokens([], []).
c_enrich_tokens([id(X)|Rest], [keyword(X)|ERest]) :-
    c_ast:c_keyword(X), !, c_enrich_tokens(Rest, ERest).
c_enrich_tokens([punct(Op)|Rest], [operator(Op)|ERest]) :-
    c_ast:c_unambiguous_op(Op), !, c_enrich_tokens(Rest, ERest).
c_enrich_tokens([T|Rest], [T|ERest]) :-
    c_enrich_tokens(Rest, ERest).

%% v1 single-statement parser entry. Note: c_ast.pl has TWO clauses
%% historically — one at line 756 and one at line 906. Both are equivalent
%% (both phrase parse_stmt). Both are extracted here as one predicate.
c_parse_stmt(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_stmt(AST), Tokens).

%% v1 multi-statement parser entry.
c_parse_stmts(String, ASTs) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_stmt_list(ASTs), Tokens).

%% v1 type parser entry. Zero callers ever.
c_parse_type(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_type(AST), Tokens).

%% v1 raw-token interface. Zero callers ever.
c_parse_tokens(String, Tokens) :-
    c_ast:c_tokenize(String, Tokens).

%% v1 full-expression parser entry. Superseded by parse_expr in v2/v3.
c_parse_full_expr(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_full_expr(AST), Tokens).

%% v1 chained-expression parser entry. Superseded by parse_chain_postfix
%% in v2/v3.
c_parse_chain(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_chain(AST), Tokens).

:- end_object.
