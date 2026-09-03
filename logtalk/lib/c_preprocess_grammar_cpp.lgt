:- set_prolog_flag(double_quotes, codes).
%% ^ PARSE-SEMANTICS (grammar string literals are code lists).

%% Contributor to c_ast::emit//2 (the declared extension point —
%% HUB_SESSION_DESIGN.md). MUST load AFTER c_ast.lgt (separate
%% compilation units required for Logtalk multifile).
:- object(c_preprocess_grammar_cpp).

    %% Contributor to c_ast::emit//2 (declared extension point).
    :- multifile(c_ast::emit//2).

    :- public(c_line_comment//0).
    :- public(c_line_comment_body//0).
    :- public(c_block_comment//0).
    :- public(c_block_comment_body//0).



%% Declare our contribution to c_ast's multifile emit//2.
%% This is NOT a re-export. We do not import c_ast's emit rules.
%% We contribute clauses; the runtime sees them as part of the same
%% predicate due to the multifile declaration in c_ast.pl.





%% ─── Emit grammar for preprocessor AST nodes ─────────────────────

c_ast::emit(c_include(Path), _Indent) -->
    "#include \"", c_ast::emit_atom(Path), "\"\n".

c_ast::emit(c_include_sys(Path), _Indent) -->
    "#include <", c_ast::emit_atom(Path), ">\n".

c_ast::emit(c_header_guard(Guard, Body), _Indent) -->
    "#ifndef ", c_ast::emit_atom(Guard), "\n",
    "#define ", c_ast::emit_atom(Guard), "\n\n",
    c_ast::emit_nodes(Body, 0),
    "\n#endif // ", c_ast::emit_atom(Guard), "\n".


%% ─── Tokenizer grammar for C/C++ comments ────────────────────────

%% Line comment: // followed by any characters up to and including \n.
%% Used by c_ast.pl's c_ws (whitespace) rule to skip comments during
%% tokenization of raw (unpreprocessed) input.
c_line_comment --> [0'/, 0'/], c_line_comment_body, [0'\n].

c_line_comment_body --> [].
c_line_comment_body --> [C], { C \= 0'\n }, c_line_comment_body.


%% Block comment: /* followed by any characters up to and including */.
%% Used by c_ast.pl's c_ws (whitespace) rule to skip comments during
%% tokenization of raw (unpreprocessed) input.
c_block_comment --> [0'/, 0'*], c_block_comment_body.

c_block_comment_body --> [0'*, 0'/].
c_block_comment_body --> [_], c_block_comment_body.

:- end_object.
