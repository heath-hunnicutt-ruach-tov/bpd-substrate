# OpenBLAS sgemm probes — the measured boundary, and two failed models

*These ran on the enclave against `sgemm_kernel_SANDYBRIDGE` (OpenBLAS 0.3.29, selected at runtime
by `DYNAMIC_ARCH`; 19968 bytes; 504 `vmulps` + 504 `vaddps` and **no** `vfmadd`, because the box is
AVX-only).*

Run from `/home/dibbur-patch` with `CUDA_VISIBLE_DEVICES='' python3 <probe>.py`.

## What is established

| probe | finding |
|---|---|
| `bound.py` | Sweeping K in steps of 8 gives **exactly one transition**: sequential accumulation matches torch below it and diverges above. |
| `bound2.py` | **★ The boundary is at K=385 and is INVARIANT across M and N** — (4,4), (8,8), (16,16), (32,32), (8,64) all give 386 at 2-step resolution. *A boundary that does not move with M or N is a K-blocking constant, not a coincidence.* So the driver's maximum K-block is **384**. |
| `adapt.py` | Solving for the block size *per K* rather than assuming one: `K=384,768,1152,1536 → [384]`; `K=512 → [256]`; `K=640 → [320]`; and `K=385,896,1024,2048 → **no uniform block size reproduces torch**`. |

## What is NOT established — two models that failed

*Both are kept because a refuted model is cheaper to read than to re-derive.*

| probe | what it refutes |
|---|---|
| `blk.py`, `blk2.py` | **KB=256 is not the structure.** It gives `0 ULP, 0/262144` — perfectly exact — at (512,512), and fails at (256,768), (128,1024), (384,384) and (64,2048). *Nothing about it looked like a coincidence except that it was one.* |
| `kb384.py` | **KB=384 is not the structure either.** It matches only where K is a multiple of 384. |
| `rule2.py` | An even-split rule (`nblocks = ceil(K/384)`, K divided evenly) matches **every** K that divides evenly and **no** K that does not. |
| `rem.py` | Three remainder placements — full-then-tail, tail-then-full, half-split — **all fail** on K=896, 1024, 2048, 4096. |

## Why the even-split rule is not reported as the answer

*Every K where it succeeds is divisible by 384 — which is also where several other structures would
coincide. **The successes may be selecting for a property of those K values rather than confirming
the rule.** That is the same shape as the KB=256 error: a beautiful fit on the subset that happens
to agree.*

> **One-shape 0-ULP is a hypothesis. N-shape 0-ULP is a structure.**

**Inference from outputs has now failed twice in the same way.** The next real step is reading the
kernel's loop structure directly rather than deducing block sizes from what it produces.

*Committed because they were in `/tmp` — the same fault that lost the original det-gemv trace, and
the reason `bench/a3_probes/` exists. **A result whose probe has evaporated is a result nobody can
re-check.*** (medayek's standing order: source in git, artifacts in output directories, nothing in
`/tmp`.)
