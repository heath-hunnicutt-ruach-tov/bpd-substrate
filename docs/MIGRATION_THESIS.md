
## Track B (Logtalk Migration) — THE THESIS (Bocher, 2026-09-02)

**The migration is not translation. It is surfacing every hidden coupling as a
declared interface.**

Prolog permitted three kinds of encapsulation violation:
  1. qualified-internal-calls (module:internal_pred — reaching into another
     module's non-interface predicates)
  2. import-of-unexported (use_module with an explicit list of predicates the
     target never exported)
  3. foreign-namespace clauses (multifile — a module defining clauses in
     another module's namespace)

Logtalk objects refuse ALL THREE. Therefore every hidden coupling in the source
MUST become an explicit, declared public interface to migrate at all. The
migrated system is CLEANER BY CONSTRUCTION — not because we cleaned it, but
because the target language cannot express the violations the source language
allowed.

This is Heath's "Logtalk should be even cleaner than Prolog" made MECHANICALLY
TRUE, one forced-explicit API at a time. The encapsulation the source language
allowed to be violated becomes enforced and documented in the target.

Evidence (day 1, 48 objects triple-gated, ~2/3 landed, ZERO bits perturbed):
the dispatch layer + fusion engine (apply_fusion, the theorem-prover, first
object-to-object composition) + SoA atom family (gemv/gemv_DET/swiglu/dot/
quantize/ffn) + I/O stack (safe_read reading a real 10.8MB GGUF byte-identically)
all migrated object-native with IDENTICAL derivations. The crown catch (foldl/4
same-name-different-contract, silent-wrong-result caught only by differential
workload equivalence) proves WHY the three-gate-per-step protocol is the
difference between "migrated" and "migrated-and-silently-wrong."

The vision: lift llama.cpp → facts (truth of record) → round-trip C++
bit-identical → lower to LLVM IR/Rust/cuda BIT-IDENTICAL by construction → fuse
declaratively → beat ollama. Track B's contribution: the DISPATCH that proves
0-ULP is now Logtalk-clean, and cleaner-by-construction because every implicit
coupling is a declared API. Declared FP semantics + declared module interfaces =
verifiable by construction, twice over.
