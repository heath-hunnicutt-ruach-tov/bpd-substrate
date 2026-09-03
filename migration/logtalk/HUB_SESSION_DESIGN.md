# c_ast Hub Migration — Session Design (2026-09-02 evening)
**Design-first per the banked discipline: this doc precedes any file movement.**

## The decisive experiment (run before design lock)
Logtalk multifile-across-objects: VERIFIED working (canonical example
/tmp test + minimal repro). Constraints discovered:
1. Primary object declares `:- public(P/A).` + `:- multifile(P/A).`
2. Contributor declares `:- multifile(Primary::P/A).` + clauses `Primary::P(...)`
3. **Primary and contributor MUST be separate compilation units (files).**
   Same-file → "Permission error: modify predicate_declaration" (the
   error that misled the first three probes; canonical example in
   separate files works; my same-file repro isolated the constraint).
4. Contributor clause BODIES compile local to the contributor (per the
   canonical example's comment) — fine for our case (grammar clauses
   call c_ast's own nonterminals via the multifile head's object).

## The design
- **c_ast.lgt**: object. 4 exports public (emit_c/2, emit_program/2,
  emit_to_file/2, ast_uses_var/2). emit//2 public + multifile (the
  extension point, now an EXPLICIT declared API — the thesis applied).
  All 327 DCG rules verbatim. double_quotes=codes at file level
  (parse-semantics, the kernel_templates precedent).
- **c_preprocess_grammar_cpp.lgt**: object. Exports its 4 DCG
  nonterminals (c_line_comment//0 etc — //0 notation needs public//
  handling: Logtalk public/1 accepts N//A). Contributes
  `:- multifile(c_ast::emit//2)` clauses (include/include_sys/header_guard).
- **Load order contract**: c_ast BEFORE grammar contributor (documented
  in both files + the loader).
- **Dependents** (templates_blas/llama/cfd/stencil, epilogue_generator
  rewire, llama_cpp_lifter, qkv_lifter, arch_summary, c_ast_legacy,
  citation_markdown_emitter, gguf_validate): uses(c_ast, [the 4]) —
  the proven object-to-object pattern. Per-file, three gates each.
- **Generators last**: consult → logtalk_load migration, emission gate
  as the primary instrument (byte-identity of the 3 .cu streams).

## Gate plan (per step)
1. emission_gate.sh both hosts (floor — MUST stay 3/3)
2. emit_diff_matrix.py (mavhir's, fires autonomously on commit)
3. Workload: emit_program on a representative AST both hosts (the
   epilogue_generator work-load already exercises this; add a direct
   c_ast workload: the c_include/header_guard/function AST from the
   grammar contributor path — tests the MULTIFILE clauses specifically)

## Order
1. c_ast.lgt + grammar contributor (the pair, gated together)
2. epilogue_generator: {user:emit_program} escape → uses(c_ast) (undo
   the temporary escape — it was the right bridge, now retire it)
3. The 4 kernel_templates variants (big, mechanical)
4. The remaining dependents
5. Generators (the composition roots)
