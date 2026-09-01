# SoA gemv v2 — stock-tiled, fusion-capable, emitted

*Iyun + Bocher, 2026-09-01. The scoping doc for the unified rung that closes the
det-gemv residual AND opens the fusion-aware speed rung — one re-architecture, both.*

## Why one work item, not two

Two open rungs turn out to be the SAME surgery:

1. **Close the det-gemv residual to 18/18.** Iteration 1 achieved per-op
   determinism-parity (FFMA accumulate + butterfly reduce + FADD combine all match
   stock). The remaining 2/18 divergence is the **block-to-thread assignment in the
   reduction** — the associativity of the sum-over-blocks differs from stock. Closing it
   = re-architecting the SoA gemv's tiling/reduction structure to match stock's exactly.

2. **The fusion-aware gemv (the first rung where SoA can WIN on speed).** Fused SwiGLU
   wants gate + up columns computed together — which IS multi-column tiling. That also
   requires re-architecting the SoA gemv's tiling structure.

**Same re-architecture serves both.** Match stock's block-to-thread structure →
determinism closes to 18/18 by construction. Make it multi-column → fusion-capable.
This is the emitter's thesis moment: emit stock's exact decode structure (which a
hand-edit can't cleanly do) AND parameterize it for column-fusion.

## The three gates (in order)

- **Gate 1 — determinism-parity (18/18):** the block-to-thread reduction structure
  matches stock's decode kernel exactly. Method: disassemble stock's decode kernel
  (`mul_mat_vec_q<type21,1,0,0>`, cubin 36 — already open), emit the SoA gemv to match
  its block-grouping / thread-assignment / reduction topology. Re-gate against stock;
  target 18/18 flip-free AND per-element ULP zero. Cross-check (mavhir, byte-identity).
- **Gate 2 — fusion correctness (0-ULP):** the fused SwiGLU path (gate+up multi-column,
  SwiGLU, quantize, down) stays 0-ULP vs the composed atoms (the RE-VECTOR
  by-construction property — the fusion is a linked verified atom).
- **Gate 3 — bench:** the speed measurement (the WIN condition, vs ollama's target).
  Only meaningful after gates 1+2 pass (never build toward an unverified reference).

## The honest inheritance from det-gemv iteration 1

- **What's proven:** per-op determinism-parity (all float ops match stock at decode),
  the +2, deterministic + cross-verified byte-identical. That work is DONE and BANKED.
- **What's NOT yet resolved:** the exact block-to-thread difference is characterized
  ("block-assignment associativity in the reduction") but NOT pinned to a one-line fix —
  static SASS reading corners it but the final resolution needs the v2 emit + gate.
- **The method that got here:** read the artefact (the reference .so the P4 runs), rule
  out candidates by SASS diff, catch your own errors (wrong shape, IEEE-commutativity,
  false-passes/fails), state the authority limit. The v2 rung continues the same method.

## Scope size (honest)

This is a **substantial emit**, not a hotfix: re-architecting the SoA gemv's tiling +
reduction to match stock's decode structure, parameterized for column-fusion, emitted
by the RE-VECTOR toolchain, gated at three levels. It's the honest big rung where the
BPD goal actually lives (beat ollama on the P4) — properly sized and named, sitting
downstream of a settled +2 and a fully-decomposed mechanism.

## Status

det-gemv iteration 1 = SETTLED (+2, per-op parity, cross-verified). This v2 rung =
SCOPED, not started. It unifies the determinism-closure and the fusion-aware-speed rungs
into one re-architecture. Awaiting prioritization vs other BPD rungs.
