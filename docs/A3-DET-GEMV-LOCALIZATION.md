# A3 — where det-gemv diverges from the ggml baseline

*Mavdil, 2026-09-03. Every number here was measured on the enclave; nothing is inferred.
**Read the confidence labels** — the location is bedrock, the mechanism is not settled.*

## ★ THE LOCATION (solid, triangulated, unmoved)

**Divergence enters at `bpd_qmatmul_q8_0_llamafile_cpu`** — the det-gemv kernel itself.

```
op 0000  embd          GET_ROWS   n_diff=     0/12288   CLEAN
op 0001  norm-0        RMS_NORM   n_diff=     0/12288   BIT-EXACT
op 0002  attn_norm-0   fused norm n_diff=     0/12288   BIT-EXACT, real gguf weights
op 0003  Qcur-0        MUL_MAT    n_diff= 10676/12288   ★ ENTERS HERE   max_abs 2.38e-06
         Kcur-0        MUL_MAT    n_diff=  1306/1536                    max_abs 4.77e-06
         Vcur-0        MUL_MAT    n_diff=  1339/1536                    max_abs 2.24e-08
         Qcur-0        ROPE       n_diff= 10601/12288   max_abs 2.38e-06 — UNCHANGED
```

**Q, K and V are three independent invocations** — different shapes, different weights, sharing
only the kernel and a **proven bit-exact input**. All three diverge. *So the cause is not setup,
weights, or layout.*

**ROPE is a passive carrier, not a second source:** `max_abs` identical before and after.

## Two kernel details, replicated 40/40

**The scale is asymmetric.** The kernel divides activations by the **f32** value `amax/127` when
quantising, then multiplies back by the **f16-rounded** scale. Using f16 for both drifts.

**Block partials accumulate sequentially in f32.** Discriminating across block counts:

```
f32 SEQUENTIAL   nblk=2:40/40   4:40/40   8:40/40   64:40/40   ★
f64              nblk=2:40/40   4:29/40   8:21/40   64:12/40
pairwise/tree    nblk=2:40/40   4:22/40   8:16/40   64: 7/40
```

## ★ THE OPEN QUESTION

A NumPy model with both details reproduces the kernel on **160/160 synthetic trials** — every
shape constructable — and **differs on the real tensor** with a signature that is not numeric:

```
disagreements per token:  [0, 2015, 0, 2007, 0, 2016]
```

**Even tokens exact; odd tokens wrong on ~98% of all 2048 rows.** *A discrepancy respecting a
structural index rather than a value is not a rounding effect — numerics are value-dependent, and
this is position-dependent.* **The cause is unknown.**

## Eliminated by construction

*Each was built and measured, not argued away.*

```
summation order        all three orders agree; kernel differs by 4.55e-02 at K=32
F32 over dequantised   7.73e-03 vs our 2.38e-06 — ~3000x worse, so ggml is not doing this
tie-breaking rule      a .5-tie input refutes BOTH half-to-even and half-away
scale spread           wide real-like scales still 40/40
block count            f32-sequential exact at every count
token count            m_tokens 1..6 all zero mismatches — the parity signature does NOT
                       reproduce synthetically
vectorisation          a pure-Python loop reproduces the einsum bit-for-bit
marshalling            same — the hand loop would not match if buffers were misread
```

## What this cost, and what it is worth

*Five mechanism interpretations were proposed and refuted tonight — order, F32-dequant, an
nblk=3 boundary, marshalling, token-pairing. **Each died to a constructed test, usually within
minutes.** None reached a decision as settled.*

**The location survived all five without moving**, because it came from calling real kernels
rather than modelling them. *Keeping location-confidence and mechanism-confidence separate is
what let four-fifths of the work stand while the rest was rebuilt.*

> **One construction is a sample.** Three of the five refutations were of claims I had
> constructed *once* — a constructed case still permits reading a pattern into a single draw.

*The rule that survives: a mechanism claim is trustworthy when it **replicates**, and a model that
passes every self-invented test while failing real data means **the tests are unrepresentative**,
not the model correct.*
