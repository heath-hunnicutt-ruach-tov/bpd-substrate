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


## GATE-1 CONCRETE TARGET (sharpened 2026-09-01, Bocher candidate 2)

The residual is cornered to **`blocks_per_iter`** — the per-thread block-partitioning stride.
Stock's mmvq has two paths:
- single-warp: `blocks_per_iter = vdr*warp_size/qi` = 2*32/8 = **8**
- multi-warp:  `blocks_per_iter = vdr*nwarps*warp_size/qi` = 2*4*32/8 = **32**

`_det` uses `SOA_VDR*TPB/SOA_QI` = 2*128/8 = **32** (multi-warp). If stock's Q8_0 decode
uses stride 8, thread t covers blocks `{t/4, t/4+8, ...}` vs `_det`'s `{t/4, t/4+32, ...}` —
DIFFERENT block subsets per thread → different per-thread partial sums → different values
entering the (identical) reduce → the 2 deterministic residuals, **every op matching**.
This is invisible to op-level diffing and exactly matches "everything matches yet residuals
persist."

**HONEST LIMIT:** INFERRED from source (mmvq.cu two-path structure) + `_det`'s known stride=32.
NOT artefact-confirmed to the exact stock value — the decode SASS folds the induction increment
into address arithmetic, so static reading did not cleanly give 8-vs-32. Confirming the exact
stock stride + fixing it is v2 gate-1's concrete first task.

**GATE-1 PIN:** emit the SoA gemv with stock's EXACT decode `blocks_per_iter` + the matching
warp-partitioning, so thread t covers the SAME block subset as stock → identical partial sums →
18/18 by construction. A one-parameter structural match (stride + its warp topology).

## GATE-1 STEP-0: confirm the exact stock stride (two routes, cheapest first)

The residual is INFERRED as `blocks_per_iter` (8 single-warp vs 32 multi-warp; `_det`=32) but
NOT artefact-confirmed (decode SASS folds the induction into address arithmetic). Before the
v2 emit, confirm the exact stock stride definitively:

- **Step-0b (cheapest, Bocher):** LAUNCH-CONFIG ARCHAEOLOGY. Static disasm won't give the
  stride, but RUNTIME will. Profile ONE stock `llama-bench` decode run with `nsys`/`nvprof`
  (or a 5-line host probe via the CUDA occupancy API) — it reports each kernel's grid×block
  dims at launch. Stock decode single-warp (704 path) launches block=(32,1)/nwarps=1 vs
  multi-warp's (32,4)/128-thread. The `mul_mat_vec_q<type21,1,...>` launch config → single-
  vs-multi-warp **CONFIRMED from the artefact of execution, no source inference**. That +
  the stride formula = the exact `blocks_per_iter`, closed.
- **Step-0a (fallback, if launch-config alone is ambiguous):** printf-instrument BOTH kernels
  (stock decode + `_det`) to log each thread's per-thread block list (`kbx` values), diff the
  assignments at runtime (~30 min on the P4). If block lists differ → residual found, emit to
  match. If they match → residual is elsewhere, honestly-unlocated, resume from clean "not here."

Once the exact stock stride is confirmed, v2 gate-1 = emit `_det` with stock's block-assignment
→ thread t covers the SAME block subset → identical partial sums → 18/18 by construction.

**Inheritance for the v2 emit:** the mechanism (per-op parity + block-assignment residual),
the NAMED PARAMETER (`blocks_per_iter`), TWO definitive confirmation routes (launch-config,
per-thread-block-list), and the three gates. Standing on corrected ground.

## STEP-0b RESULT (2026-09-01, Iyun) — LAUNCH-CONFIG ARCHAEOLOGY: residual CONFIRMED, inference OVERTURNED

Ran Bocher's step-0b (nsys profile of stock + `_det` decode, read grid×block dims). Result
**overturns my static-SASS inference** and confirms the residual from the EXECUTION artefact:

| kernel | block dims (launch) | warp layout |
|---|---|---|
| stock `mul_mat_vec_q<Q8_0,ncols=1>` (decode) | **(32, 4, 1)** | 2D: warp-id = `threadIdx.y`, lane = `threadIdx.x` |
| `_det` `gemv_soa_q8_0_q8_1_det` | **(128, 1, 1)** | 1D: warp-id = `threadIdx.x/32`, lane = `threadIdx.x%32` |

**Both are 128 threads / 4 warps — but DIFFERENT BLOCK SHAPE.** My inference ("stock uses
single-warp stride=8") was WRONG: stock decode is MULTI-warp (nwarps=4), `blocks_per_iter=32`,
SAME stride as `_det`. The divergence is NOT the stride — it's the **block DIMENSIONALITY**:
(32,4) vs (128,1). The thread→warp/lane indexing differs, so which physical thread processes
which block/lane differs → different partial-sum grouping across warps → the 2 deterministic
flips, **every float op matching**. Bocher's candidate (2), confirmed from execution.

**Why static SASS couldn't see this:** the block dims are a LAUNCH parameter (`<<<grid,block>>>`),
invisible in the kernel's own disassembly — only the runtime launch config (the execution
artefact) carries it. This is the "escalate to the deeper artefact when the report-artefact
can't answer" discipline: static disasm was genuinely blind here; nsys was definitive.

**v2 GATE-1 FIX (now sharp + artefact-confirmed):** launch `_det` with block = (32, 4, 1)
matching stock (change the dispatch `<<<nrows, 128>>>` → `<<<nrows, dim3(32,4)>>>` AND update
the kernel's warp-id/lane indexing to `threadIdx.y`/`threadIdx.x` to match stock). Then thread
t covers the SAME block subset as stock → identical partial-sum grouping → 18/18 by construction.
A ~2-line dispatch + indexing change, NOT a big re-architecture — the launch-config route made
the fix small AND certain.

**Honest note:** this SUPERSEDES the "blocks_per_iter 8-vs-32" characterization (which was
inferred and WRONG — stock uses 32, same as det). The real residual is block-shape (32,4) vs
(128,1). Inference corrected by artefact, exactly as step-0b was designed to do.
