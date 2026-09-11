# Reproducing the L3 Census — KernelBench Level-3, Whole-Model Bit-Exact

> **This document is a DRAFT.** The L3 arc is in progress; sections marked
> `<!-- TBD -->` will land as the corresponding rungs complete. The frame
> is ready; the numbers land as they are measured.

## The claim (quote it whole or not at all)

**Of 50 KernelBench Level-3 problems, `<!-- N -->` reproduce the benchmark's
own `Model.forward` bit for bit — at the benchmark's own inputs, on the
configuration named below. The ceiling is 50 by construction: every
non-gated problem carries a named capability that would move it into
scope.**

Three caveats are part of the claim, not fine print:

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

## The measured configuration

| component | value |
|---|---|
| GPU | NVIDIA Tesla P4 (sm_61, ~7.4 GB usable) |
| CUDA | 12.8 (`nvcc --fmad=false` — load-bearing, do not drop) |
| torch | 2.7.0 (CUDA 12.8 build) |
| KernelBench | commit `423217d` (pin your clone to this) |
| python | 3.12 · SWI-Prolog on PATH · gcc |
| einops | vendored pure-python whl (needed for Mamba2 problems) `<!-- verify path -->` |

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
- **the census** (`verification/RUNBOOK-L3.md` — `<!-- pending Step 2 -->`)
  re-emits the store from scratch, refuses unproducible units, checks
  provenance windows, and prints `BIT_EXACT n · DIFFERS n · SKIPPED n
  · of <producible> · <ceiling> by construction`. Run the producibility
  pass FIRST — twice it has found units the store could no longer
  justify, and in both cases the per-unit gates would have said nothing
  at all. The refused must not outlive their refusal.

## Environment

```sh
export CUDA_HOME=/path/to/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export BPD_KB=/path/to/KernelBench/KernelBench/level3
# einops-vendored path (Mamba2 problems):
export PYTHONPATH="$(pwd)/lib/vendored:$PYTHONPATH"  # <!-- verify -->
```

## One problem, end to end

`<!-- TBD: fill from actual L3 gate command shape when Mavdil/Bocher's
  RUNBOOK-L3 recipe is confirmed. The three-command form from L2
  extends to L3 with the multi-flow replay flag where applicable. -->`

## All L3

Loop the pipeline over `$BPD_KB/*.py` with pid `l3imp<N>`: every
producible problem must emit (no `[GAP]`), every wrapper must build,
every verdict must read `n_diff: 0` on every fire-count position.

**Re-emit before you gate, and gate what you emitted, not what you found
on disk** — a stale artefact is indistinguishable from a current one at
gate time; L3 has caught two orphans this way (units present in the
store but refused by the pipeline; both had earlier appeared to gate
clean on stale artefacts).

Then run the census for the independent tally. A problem failing any
step counts against the total — a finding, not a footnote.

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
`<!-- e.g. 11 → 14 → 15 → 16 → N -->`. Each step is a set of specific
capabilities that landed, not a general trend: name the capability that
moved the count, or the count is a slogan.

## Why this document is a draft

The L3 arc is in progress. Sections marked `<!-- TBD -->` will land as
the corresponding rungs complete. The claim's shape, the config, the
two-halves discipline, the caveats-in-claim, and the ceiling frame are
ready now; the numbers that fill the frame land as they are measured.
Publishing an in-progress claim would violate the discipline the L2
document embodies. Landing the frame in advance ensures that when the
numbers arrive, the document that carries them is honest by construction.
