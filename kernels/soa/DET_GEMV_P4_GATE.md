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
