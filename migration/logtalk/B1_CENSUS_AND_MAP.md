# Track B (Logtalk Migration) — B1: Census, Dependency Map, Migration Order
**Date:** 2026-09-02 · **Owner:** Bocher · **Test methodology:** Medayek · **Coordination:** Iyun
**The Collective's first Logtalk migration.** Heath's directive: move the Prolog
dispatch/emitter layer to Logtalk (expressive semantic scoping; productivity
pattern for the whole Collective). Logtalk 3.101.0 via swilgt (SWI host).

## Census (98 .pl total, reconciled)
- lib/: 76 · generators/: 4 (blas/fused/llama/cfd) · tests/: 12 · bench/tier2: 3 · file-formats/gguf: 3
- **Migration scope (B2): lib/ + generators/ = 80.** tests/bench/gguf follow later.

## The invariant (proven before any migration)
**Emission byte-identity across hosts:** all three generator outputs are
byte-identical under `swipl` and `swilgt -q` (unmodified .pl, consult+test
invocation). Baseline sha256 frozen in emission_gate.sh:
- blas aedb0455… · fused b7420268… · llama a0a35e0a…
- Emission is deterministic (run-twice identical under swipl).

## Gate harness: emission_gate.sh
Byte-identity gate with the week's checker-taxonomy controls:
- **empty-output guard** (sha256-of-nothing = automatic FAIL, never identity)
- **invocation parity** (consult+test both hosts; generate_fused's
  initialization guard doesn't fire under consult — found during bridge test)
- **no command-substitution capture** (strips trailing newlines → corrupted
  hash; caught live when fused's two trailing newlines broke the first
  harness version — pipe directly to sha256sum)

## Dependency map (from use_module/ensure_loaded analysis)
- **Most-depended (wrap first, migrate last):** c_ast (12 dependents),
  llvm_emit (3), model_transform (3), then safe_read, gguf_native_reader,
  llama_cpp_lifter, kernel_templates_blas, kernel_templates_llama (2 each)
- **53 leaf files** (no local deps) — migrate-first candidates
- **27 files with deps** — migrate in topological order after their deps

## Planned Logtalk shape (B2)
- **Protocols:** kernel_emitter (emit/2, deps/1), fusion_rule (can_fuse/3,
  apply/4), dispatch_contract
- **Categories:** kernel-family shared behavior (gemv-family, norm-family,
  activation-family)
- **Objects:** platform targets (sm61, cpu) parameterized by arch facts;
  one root object wrapping c_ast emission machinery via uses/2
- **Principle:** clause bodies verbatim where possible — Logtalk changes the
  calling structure, not the logic. One file per step, emission gate green
  after every step (the det-gemv minimal-diff lesson applied to refactoring).

## Migration order (B2 sequence)
1. Leaf pure-fact files (arch_params, kernel signatures) → objects
2. Leaf emitter helpers (epilogue_generator, c_preprocess) 
3. Mid-tier (kernel_templates_* behind a protocol)
4. Fusion core (apply_fusion, auto_fuser, fusible, graph_optimizer)
5. c_ast LAST (12 dependents; wrapped via uses/2 from step 1 so nothing breaks)
6. generators/ entry points (become the composition roots)
Every step: emission_gate.sh green under BOTH hosts + Medayek's framework.
