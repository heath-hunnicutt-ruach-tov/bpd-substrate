# A3 det-gemv probes — the evidence behind `docs/A3-DET-GEMV-LOCALIZATION.md`

*These ran on the enclave and produced every number in that document. **They lived in `/tmp`
until they were committed here** — which is exactly how the original det-gemv trace was lost, and
why A3 needed regenerating at all. A result whose probe has evaporated is a result nobody can
re-check.*

Run from `/home/dibbur-patch/step3-det-gemv/bpd` with
`CUDA_VISIBLE_DEVICES='' python3 <probe>.py`. All read the committed fixture
`fixtures/llama_dump_tinyllama_hello/` (regenerable via its `RECIPE.md`).

## The probe that found the answer

| probe | what it establishes |
|---|---|
| `recip.py` | **★ THE SOLUTION.** `q = round(a * (127/amax))` gives **12288/12288 bit-exact**; `a / (amax/127)` gives 6250; `a * (1/dq)` gives 8247. Alternatives differ on 2015/2007/2016 — exactly the failing counts. |
| `gen.py` | **The generalisation check, run BEFORE reporting.** The reciprocal model on three independent projections: `w_q` 12288/12288, `w_k` 1536/1536, `w_v` 1536/1536 — **15360/15360**. |

## Probes that built the constraint

| probe | what it establishes |
|---|---|
| `par.py` | The parity is in the **data, not the call shape** — six separate `m_tokens=1` calls reproduce `[0,2015,0,2007,0,2016]` identically to the batched call. |
| `solve3.py` | **Per-block arithmetic is correct** — 12/12 isolated `K=32` calls on a *failing* token match bit-exactly. |
| `solve4.py` | **Cross-block combination is correct** — f32-sequential-forward over the kernel's own block values matches the full call exactly; 6 of 7 alternatives differ. |
| `solve5.py` | **The gap is ONE block of 64** (block 60), max diff 9.6e-07. |
| `wrong.py` | **★ The clue that forced the answer:** the integer-dot delta equals the near-tie element's weight *exactly* (+11/w=11, +8/w=−8, +70/w=70). One element, one integer step. |
| `seq.py` | The f32-sequential distinction is **exercised on real data** — alternatives differ on 1672/2048, only one matches. |

## Probes that eliminated candidates

*Each ran **before** its hypothesis was reported. None of these became claims.*

| probe | what it refutes |
|---|---|
| `ties.py` | **Ties are not the cause** — tokens 3/5 have wrong blocks with *zero* exact ties; token 2 has three ties and *zero* wrong blocks. |
| `verify.py` | **Uniform half-away is refuted** — 4211/12288 vs half-even's 6250; token 2 collapses 2048→1. |
| `rule.py` | **Dividing by the f16 scale is refuted** — 24/12288 vs 6250, alternatives differ on 12268. |
| `solve2.py`* | **The kernel definitely quantises activations** — three no-quant variants land 4.13e-04 away vs the quantised model's 9.6e-07. |
| `mb.py` | Multi-block accumulation sweep: f32-sequential 40/40 at every count; f64 and pairwise degrade. |
| `min.py` | The minimal hand-built `K=32` q8_0 block that refuted **summation order** in ten minutes. |
| `tok.py` | **Token-pairing refuted** — `m_tokens` 1…6, zero mismatches. |
| `solve6.py` | Block 60 read element-by-element: `i=13`, `r = 24.500000000`. |

\* `solve2.py` was superseded on the enclave; its result is recorded in the doc.

## What these probes cost, and what that taught

*Six mechanism claims were made and withdrawn before `recip.py` found the answer. Five shared one
root cause: **a match on inputs where the candidate mechanisms happened to agree.** That is why
several probes above print an **alternatives-differ count** — a replication earns a finding only
where the alternatives actually diverge on the test data.*
