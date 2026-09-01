# Deterministic-FMA gemv — P4 gate recipe & handoff

*Iyun, 2026-09-01. Handoff to whoever runs the P4 (mavhir). The offline analysis
is complete; this is the card-dependent confirmation, queued behind mm_fusion.*

## What's being gated

`gemv_soa_q8_0_q8_1_det.cuh` — the SoA Q8_0 gemv with the float accumulate pinned
to `__fmaf_rn(dw*da, (float)sumi, sum)` (one FFMA, one rounding) instead of the
original `sum += dw*da*sumi` (which compiles to un-fused FMUL+FADD, two roundings).

**Side-by-side, env-gated `GGML_SOA_DET=1`** — dispatch selects `_det` when set,
the original when unset. A/B without rebuild; zero regression surface.

## The offline finding (already established — this gate CONFIRMS it)

Zero-compile archaeology on the reference `libggml-cuda.so.0.13.1` (the binary the
P4 runs), via `cuobjdump --extract-elf` + `nvdisasm`:

| kernel | accumulate | roundings |
|---|---|---|
| STOCK Q8_0 mmvq (`_Z13mul_mat_vec_qIL9ggml_type21E...`) | `FFMA.FTZ` (dst==accumulator) — **fused** | 1 |
| ORIGINAL SoA gemv | FMUL+FADD (0 FFMA in SASS) — **un-fused** | 2 |
| `_det` (this) | `FFMA.FTZ` — **fused, matches stock** | 1 |

The 2-vs-1-rounding delta per block IS the documented per-element 1-2 ULP noise.
`_det` restores stock's exact accumulate. **The mechanism is cornered offline;
the P4 harness confirms sufficiency** (register allocation + reduce tree are the
residual unknowns only the card settles).

## Gate recipe (dual acceptance — both must pass together)

Method chain that got us here (all offline): toy-PTX → toy-SASS → real-kernel SASS
(differs) → reference-binary archaeology (stock fused). The card does step 1+2 below.

**Step 0 (offline, DONE):** SASS accumulate match. `_det` = FFMA = stock; original
= FMUL+FADD. Verified. (Redo on the P4's own nvcc if desired: `cuobjdump -sass` the
two cubins, confirm `_det`'s accumulate is FFMA.)

**Step 1 — per-element harness (the SUFFICIENCY test):**
- Run the SoA path with `GGML_SOA_DET=0` (original) and `GGML_SOA_DET=1` (`_det`),
  capture per-element gemv outputs (e.g. blk.0.attn_q, the documented divergence site).
- Acceptance (b): `_det` per-element ULP delta vs **stock** → 0 (not vs original —
  vs the stock reference). If `_det` matches stock per-element, the accumulate fix
  is confirmed sufficient at the gemv level.
- If `_det` still deltas vs stock: the divergence has a SECOND source (reduce-tree
  order or two-buffer load scheduling) — the emitter can pin those too (its advantage
  over a hand-patch), and we iterate.

**Step 2 — 18-battery + fixture_token_gate (the FLIPS test):**
- `logit_gate.py` 6-strata (18 prompts) + `fixture_token_gate.py` (10 prompts),
  `GGML_SOA_DET=1` vs stock.
- Acceptance (a): argmax flips → 0. (Original SoA was 24/28; target is 28/28.)
- Run with the standard `LD_LIBRARY_PATH` incantation (README).

**PASS = (a) flips→0 AND (b) per-element ULP→0, both together.** Either alone is
insufficient (README Finding: the FMA data is a predictive model with two criteria).

## Integration notes

- `_det.cuh` is a minimal diff of the original: ONE line (the accumulate) + the
  rename. Everything else — block loop, the stock-matched reduction, `warp_reduce_sum`
  — is byte-identical to the original, so the ONLY variable under test is the FFMA pin.
- Compile-verified on the emit instance (sm_61, `-use_fast_math`): compiles clean,
  SASS accumulate is FFMA (5 FFMA, pin took). No GPU there — hence this handoff.

## Status

Offline analysis complete (mechanism cornered, `_det` matches stock's accumulate).
`_det.cuh` compile-verified. Awaiting P4 (queued behind mm_fusion). The card's role:
CONFIRM, not explore.

---

## ITERATION 1 RESULT (2026-09-01, P4, Iyun) — accumulate pinned, 16/18, mechanism confirmed

Built `_det.cuh` (accumulate pinned to `__fmaf_rn`, matching stock's FFMA), gated on
the P4 vs stock (18-strata logit gate, greedy temp-0, GGML_SOA_DET=1). Control: the
original patched SoA path (no det) vs stock.

| stratum | CTRL (orig SoA) | DET | delta |
|---|---|---|---|
| minimal | 3/3 | 3/3 | = |
| code | 3/3 | 3/3 | = |
| multilingual | 2/3 | 2/3 | = (residual) |
| long_context | 3/3 | 3/3 | = |
| **repetitive** | **1/3** | **3/3** | **+2 (FIXED)** |
| adversarial | 2/3 | 2/3 | = (residual) |
| **TOTAL** | **14/18** | **16/18** | **+2** |

**The decomposition is clean:** the accumulate-FFMA pin changed EXACTLY ONE stratum —
`repetitive` (1/3 -> 3/3), the FMA-heaviest (most accumulation ops => most sensitive
to the 2-rounding-vs-1-rounding delta the pin removes). Every other stratum is
identical CTRL vs DET. So the FFMA accumulate fix's contribution is empirically
isolated: +2, the repetitive stratum, exactly where the offline mechanism predicted.

**The 2 residuals (unchanged by the accumulate pin, so a DIFFERENT source):**
- multilingual (Japanese こんにちは、お元気ですか): diverges @char 570, different continuation.
- adversarial (emoji fire/skull/dart/rocket): 'symbols' vs 'emojis' @char 612.

Both are REAL token flips (verified, not chrome). Consistent with the design flag:
`_det.cuh` pinned the ACCUMULATE only (hand-minimal diff of the original); the
warp-reduce ORDER and two-buffer LOAD scheduling were NOT pinned. The residual is
almost certainly one or both of those.

**VERDICT:** dual acceptance NOT met (16/18, not 18/18 — rung does NOT fully close).
BUT det STRICTLY IMPROVES (+2), mechanism CONFIRMED (fix lands exactly where predicted),
and the divergence is now empirically decomposed into >=2 components (accumulate: +2;
reduce/load: residual 2). This two-sided prediction-fulfillment is STRONGER mechanism
evidence than a clean 18/18. Results: gate1b_results.json (det), gate_ctrl_results.json
(control) in step3-det-gemv/.

**FALSE-FAIL CAUGHT:** first gate run showed 0/18 — all diverging @char 282 in the
BUILD-VERSION BANNER ('0-unknown' vs stock's git tag; chrome-strip gap). Reading the
actual diff caught it; chrome-strip extended in the gate copy. Taxonomy note: tools
fail toward FALSE-PASS *and* FALSE-FAIL — the only defense is reading the artefact.

## ITERATION 2 (next): pin the reduce-tree + load order

Re-emit with warp-reduce ORDER + load scheduling pinned to stock (the emitter's thesis:
the hand-edit couldn't, the emitter can). Method (Bocher): disassemble stock's reduce
in the same cubin already open (cuobjdump+nvdisasm on the reference .so), emit to match
the shuffle-tree width sequence + per-step operand order — same archaeology as the
accumulate. Re-gate as iteration 2 with mavhir's independent cross-check (two instruments
on the final verdict).
