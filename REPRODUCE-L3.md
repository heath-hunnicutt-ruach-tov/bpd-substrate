# Reproducing the L3 Census — KernelBench Level-3, Whole-Model Bit-Exact

> **This document is a DRAFT.** The L3 arc is in progress; sections marked
> `<!-- TBD -->` will land as the corresponding rungs complete. The frame
> is ready; the numbers land as they are measured.
>
> **The recipe is not yet executable by a stranger.** `mkproducible3.py`
> and `wrapgate3.py` — the two scripts `verification/RUNBOOK-L3.md`
> names — are not yet published in this repository. Until they land
> (pending review), the RUNBOOK is a shape-of-recipe, not a runnable
> one. See its "What is currently missing" section for detail.

## The claim (quote it whole or not at all)

**Of the KernelBench Level-3 problem set, 16 reproduce the benchmark's
own `Model.forward` bit for bit — at the benchmark's own inputs, on
the configuration named below. The honest denominator chain is
16 bit-exact of 19 in the emitted store, of 20 producible, of 50 total;
13 nothing-to-fuse and 17 gaps make up the remainder. The ceiling is
50 by construction: every non-gated problem carries a named capability
that would move it into scope.**

*(Numbers current as of census-four, commit `697f2d364` with the
`ecc0e0230` prose correction. A fresh census run will produce
different numbers; when it does, update the provenance-cite in this
paragraph. See "Reading the result" in `verification/RUNBOOK-L3.md`
for the expected output shape.)*

Four caveats are part of the claim, not fine print:

1. **Bit-exact at the benchmark's inputs does not mean the kernels are
   correct.** A clamp with wrong bounds passes bit-exact when no benchmark
   input reaches the clamp (this class of hazard was caught in L2 by
   static audit, not the bitwise gate). The gate answers a specific
   question; do not read a broader one from it.
2. **The configuration is part of the number.** Bitwise equality is
   substrate-specific: during L2 the same `S/n` vs `S*(1/n)` expression
   differed on 63% of values in strict numpy fp32, 65% in a raw sm_61
   CUDA kernel, and 0% under torch-CUDA. A different torch/CUDA/GPU may
   legitimately give a different number — that is a property of the
   target, not a refutation. Reproduce on this configuration, or measure
   and name yours.
3. **The ceiling is part of the number.** L3's 50 is not a denominator
   over which the ratio is uncapped; it is a ceiling *by construction* —
   every problem outside the current census carries a specific named
   capability the pipeline does not yet possess (e.g. cuDNN-RNN 14-operand
   ULP dossier, channel-shuffle boundary derivation, hardware-OOM under
   the reference implementation). Reading "N of 50" without the ceiling
   frame overstates what the number claims. The ceiling is measured, not
   assumed.
4. **The denominator chain is measured at every step.** The census
   counts *store units*, not problems: `16 bit-exact of 19 in store`
   names what we emitted and how many gate; `20 producible` names what
   the pipeline can emit; `13 nothing-to-fuse` names problems the
   pipeline reads as having no fusable epilogue (an honest capability
   limit, not a failure); `17 gaps` names problems the pipeline
   refuses. Every number in the chain is a measured denominator.
   Reading only one of them, or collapsing them, loses information the
   frame depends on.

## The measured configuration

| component | value |
|---|---|
| GPU | NVIDIA Tesla P4 (sm_61, `TORCH_CUDA_ARCH_LIST=6.1`, ~7.4 GB usable) |
| CUDA | 12.8 (`nvcc --fmad=false` — load-bearing, do not drop) |
| torch | 2.7.0 (CUDA 12.8 build) |
| KernelBench | commit `423217d9` (pin your clone to this) |
| python | 3.12 · SWI-Prolog on PATH · gcc 14.3 (also for the `g++` link step) |
| einops | vendored pure-python whl in `lib/` (needed for Mamba2 problems: #48, #49) |

**The enclave path is not needed for reproducibility. The provenance of
these numbers is the configuration** — sm_61, CUDA 12.8, torch 2.7.0,
KernelBench `423217d9`, `--fmad=false`. A different path on the same
configuration should give the same bits. A different configuration may
not, and that is what the caveats are for.

The L3 measured configuration is the same as L2's except for the
einops vendoring: L3's #48/#49 (Mamba2) require einops that older
container images did not ship; the vendored pure-python whl in
`lib/` closes the env-gap without altering the arithmetic.

## The two halves

The claim has two independent parts — reproduce both or you have
reproduced half of it:

- **the pipeline** (this repo: `lib/lift_chain.py`,
  `lib/auto_pipeline.py`, `tools/emit_wrapper.py`,
  `tools/kernel_runner.py`, plus the L3-specific extensions for
  container-driven forwards, recurrence-loops via hook lists, and
  multi-flow replay) emits every kernel from problem source and gates
  each whole model against torch, bitwise;
- **the census** (`verification/RUNBOOK-L3.md`) re-emits the store,
  runs a three-way producibility classification (`producible`,
  `nothing-to-fuse`, `gaps`), refuses unproducible units, checks
  provenance windows, and prints `BIT_EXACT n · DIFFERS n · SKIPPED n
  · of <in-store>`. L3 requires **three steps in order** (re-emit →
  producibility → gate), because `mkproducible3.py` does not rewrite
  existing units and `auto_pipeline.py` emits to stdout — see the
  RUNBOOK for the exact commands. Twice the producibility pass has
  found units the store could no longer justify, and in both cases the
  per-unit gates would have said nothing at all. **The refused must
  not outlive their refusal.**

## Environment and commands

The full recipe lives in `verification/RUNBOOK-L3.md`. In summary:

- **Three steps in order** (not two like L2): re-emit → producibility
  → gate. The re-emit loop is L3-specific; do not skip it.
- **Wait ~2 minutes between producibility and gate.** The gate
  refuses if the manifest is too fresh relative to the store; the
  abort message is the guard working, not a failure.
- **Write the re-emit as a script file, not an inline shell command.**
  Nested quoting has silently eaten the loop before.
- **Dump the verdict to a file and read the file.** Never pipe the
  dict through `cut -c` or similar; truncation is indistinguishable
  from a shorter number.
- **Unit prefix is `l3imp<N>`**, not `l3_<N>` or `imp<N>`.

**Re-emit before you gate, and gate what you emitted, not what you
found on disk** — a stale artefact is indistinguishable from a current
one at gate time; L3 has caught two orphans this way (units present in
the store but refused by the pipeline; both had earlier appeared to
gate clean on stale artefacts). The refused must not outlive their
refusal.

A problem failing any step counts against the total — a finding, not a
footnote.

## The characterized DIFFERS/SKIPPED

These are named, not deferred. Each names the specific capability that
would move it into the BIT_EXACT column:

**DIFFERS (n_diff > 0 but shape/mechanism named):**

- **#25 — channel-shuffle boundary derivation** (`n_diff = 240,844,785 /
  2 segs`). The channel-shuffle operator's boundary derivation is the
  named gap; the tail is precision-shaped, not structural. `<!-- TBD:
  further characterization when the boundary-derivation lands -->`
- **cuDNN-RNN 14-operand ULP dossier** (8 problems). The cuDNN backend
  invokes a 14-operand fused kernel whose operand order is not directly
  transcribable from public source; the resulting ULP delta is
  characterized (bounded, width-independent, named).

**SKIPPED (measured non-emission with named cause):**

- **#2 — hardware OOM** (unfused reference; 7.4 GB card cannot fit the
  reference model, before comparison is possible). Hardware ceiling, not
  a pipeline gap.
- **#17 — return `torch.cat(...)`**. Return-of-call shape that the
  lifter previously walked past silently. Refused honestly now
  (previous orphan removed). `<!-- TBD: whether/when return-call handling
  moves this into scope -->`
- **#31 — `<!-- TBD: fill from census when its class is named -->`**
- **#8 — 2 lifted runs but 3 manifest groups** — the builder's honest
  refusal. Producibility mismatch, named-and-refused rather than
  silently accepted.

**Unresolved (open to characterization):**

`<!-- TBD: any problem currently DIFFERS or SKIPPED whose mechanism
  is not yet named. Move to the characterized section as its named
  capability is measured; keep here only what is honestly unresolved. -->`

## Provenance

L3-specific transcriptions and their provenance:

- **RNN classes (#33 VanillaRNN, #34 RNN-with-sequence)**. The
  recurrence-term-first-censused emission (fires:256 for #34) verifies
  every timestep independently — the hooks capture lists, one input/
  output pair per fire, all 256 measured rather than induced from one.
  `<!-- TBD: which auto_pipeline.py path handles the fby (recurrence-term)
  emission; verify against the emit source before publication. -->`
- **`fby` term emission**. Named in the census-4 report as the recurrence
  operator's first landed pattern. `<!-- TBD: transcription source and
  provenance line -->`
- **QKV rung** (attention-block reproductions). `<!-- pending: lift
  complete for both problems, gate checkpointed for the wrapper-env-build
  session; the multi-flow-replay wrapper is the found consumer for the
  environment's actual gate input. Transcription source and provenance
  land when the rung gates clean. -->`
- **Branch-cat rung** (informs #17/#18, potentially #6). `<!-- pending -->`
- Every number in the campaign record was produced by a measurement
  someone ran; the doctrine that kept it that way is in `docs/` — start
  with the frame above, it is the load-bearing part.

## The ladder (for context, not part of the claim)

The census progression measured while this document is drafted:

    11 of 17  ·  14 of 20  ·  15 of 19  ·  16 of 19

Written as pairs, not bare numbers, because the denominator moved
between rungs: the store shrank twice (orphan deletions between the
first two rungs and again before the fourth). A bare `11 → 14 → 15 →
16` would imply a fixed denominator and a monotone climb; the climb
is real, the denominator is not fixed. **Each step is a set of
specific capabilities that landed, not a general trend: name the
capability that moved the count, or the count is a slogan.**

## Why this document is a draft

The L3 arc is in progress. Sections marked `<!-- TBD -->` will land as
the corresponding rungs complete. The claim's shape, the config, the
two-halves discipline, the caveats-in-claim, and the ceiling frame are
ready now; the numbers that fill the frame land as they are measured.
Publishing an in-progress claim would violate the discipline the L2
document embodies. Landing the frame in advance ensures that when the
numbers arrive, the document that carries them is honest by
construction.

**Updating the numbers:** the count numbers above are current as of a
specific census commit, cited inline in the claim paragraph. Fresh
census runs will produce different numbers, and each rung has its own
denominator (see the ladder). When updating the numbers, update the
provenance-cite too — the numbers without their commit are a number
without its frame.
