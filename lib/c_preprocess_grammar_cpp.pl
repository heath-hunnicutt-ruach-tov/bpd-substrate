%% c_preprocess_grammar_cpp.pl — preprocessor-AST emit clauses.
%%
%% c_ast.pl loads this module via `use_module(c_preprocess_grammar_cpp, [])`
%% and declares `c_ast:emit//2` multifile. Per the comment block at the top
%% of c_ast.pl, this module owns the emit rules for preprocessor nodes
%% (c_include, c_include_sys, c_header_guard) — the clauses live here so
%% c_ast's core stays free of preprocessor-syntax knowledge.
%%
%% The original module source was not in the initial repo snapshot.
%% Reconstructed from the documented contract in c_ast.pl lines 37-61
%% and the AST term definitions in c_ast.pl lines 110-114:
%%   c_include(Path)       — #include "path"
%%   c_include_sys(Path)   — #include <path>
%%   c_header_guard(Name)  — #ifndef NAME / #define NAME / ... / #endif

:- module(c_preprocess_grammar_cpp, []).

:- multifile c_ast:emit//2.

c_ast:emit(c_include(Path), _Indent) -->
    "#include \"", { atom_codes(Path, Codes) }, Codes, "\"\n".

c_ast:emit(c_include_sys(Path), _Indent) -->
    "#include <", { atom_codes(Path, Codes) }, Codes, ">\n".

c_ast:emit(c_header_guard(Name), _Indent) -->
    { atom_codes(Name, Codes) },
    "#ifndef ", Codes, "\n",
    "#define ", Codes, "\n".
