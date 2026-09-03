# A3 — where det-gemv diverges from the ggml baseline

*Mavdil, 2026-09-03. Every number here was measured on the enclave; nothing is inferred.
**Read the confidence labels** — the location is bedrock, the mechanism is not settled.*

## Where the evidence lives, and how to rebuild it

**Fixture** (enclave): `/home/dibbur-patch/step3-det-gemv/bpd/fixtures/llama_dump_tinyllama_hello/`
— 688 per-op tensors, 44 MB. The `.bin` payload is **gitignored**; `manifest.tsv` and `RECIPE.md`
are committed, so the evidence is **regenerable deterministically** rather than carried.

```sh
bash tests/correctness/build_eval_callback.sh /path/to/llama.cpp   # era-detecting driver
LLAMA_DUMP_DIR=<tree-path> LD_LIBRARY_PATH=<build>/bin CUDA_VISIBLE_DEVICES='' \
  <build>/bin/llama-eval-callback -m /mnt/data/shared/models/tinyllama-q8_0.gguf \
    -p "Hello, my name is" -n 1 --temp 0 --seed 42 -c 64 -t 2
```

**★ THE MODEL IS A SUBSTITUTION.** `tests/correctness/README.md:61` specifies **llama3.2-1b**,
which is **not on the enclave** (verified by filesystem-wide search). This is **tinyllama-q8_0** —
llama-architecture, answering the *structural* question. *The substitution is encoded in the
directory name so it is visible at every reference.* **No comparison against any prior det-gemv
figure is possible or intended**; the original trace lived in `/tmp` and evaporated.

*The dump is deliberately **ungated**: llama.cpp's own print is conditioned on
`matches_filter && !ggml_is_quantized`, and this model **is** quantized — gating would capture
**556 of 688** tensors while looking complete.*

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

## One kernel detail, standing on a verified-exercised distinction

**★ THE BAR: every mechanism claim must carry its ALTERNATIVES-DIFFER count.** *A replication
earns a finding only where the alternatives actually diverge on the test data. A match where the
alternatives agree is measuring nothing — and no number of extra samples fixes it.*

**~~The scale is asymmetric~~ — WITHDRAWN.** *I reported this "replicated 40/40": quantise by the
f32 `amax/127`, multiply back by the f16-rounded scale. On real data:*

```
dq == dm  (the f32 scale equals its own f16 rounding):  0 of 64 blocks
```

*In my synthetic tests the two were **frequently equal**, so the distinction the finding rests on
was often **not exercised** — the 40/40 measured nothing. On real data, where they always differ,
**neither choice matches.** Five of six earlier withdrawals on this cell share that root cause:
a match where the alternatives happened to agree.*

**Block partials accumulate sequentially in f32** — and the distinction is **verified
exercised on real data**, which is why this one stands where the other fell:

```
token 0 (a clean token), real activations, real weights:
  f32-sequential vs kernel      0/2048      EXACT
  f64-sum        vs kernel   1672/2048
  the two choices differ     1672/2048   ← the line that earns the claim
```

Discriminating across block counts on synthetic blocks as well:

```
f32 SEQUENTIAL   nblk=2:40/40   4:40/40   8:40/40   64:40/40   ★
f64              nblk=2:40/40   4:29/40   8:21/40   64:12/40
pairwise/tree    nblk=2:40/40   4:22/40   8:16/40   64: 7/40
```

## ★★ SOLVED — the kernel multiplies by a reciprocal

```
my model:    q = round(a / (amax/127))       ← division
the kernel:  q = round(a * (127/amax))       ← reciprocal multiply
```

*Algebraically identical; they differ by **one ULP** near rounding boundaries. That ULP flips
exactly one activation element per affected block by one integer step — which is precisely the
measured "integer delta equals that element's weight" pattern.*

```
divide by dq             6250/12288
multiply by 1/dq         8247/12288
multiply by 127/amax    12288/12288   ★ BIT-EXACT
alternatives differ on 2015 / 2007 / 2016 — exactly the failing counts
```

**Generalises across three independent projections** — different shapes, different weights, one
kernel:

```
w_q  (2048 rows)   12288/12288    w_k (256 rows)  1536/1536
w_v  ( 256 rows)    1536/ 1536    GRAND TOTAL    15360/15360
```

### The complete verified model

```
activations quantised per 32-element block, amax over the block
q = round(a * (127/amax))        ← the reciprocal, half-to-even
integer dot product with the int8 weights
scale by the F16-rounded activation scale × the F16 weight scale
accumulate block partials SEQUENTIALLY in F32, forward order
```

**So the divergence versus ggml was never an algorithm difference — it is one
reciprocal-versus-divide ULP.** *0-ULP against this baseline is reachable by matching a single
operation.*

## The question as it stood before the answer



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

**And the bar sharpened once more the next morning.** A sixth claim — the asymmetric scale — had
been *replicated 40/40* and was still wrong, because **the replication data did not exercise the
distinction.** Seed count measures *how many times you tested*; the alternatives-differ count
measures *whether the test tested the thing.*

> **Every mechanism claim carries its alternatives-differ count.** You cannot fool that check
> with more samples — only with samples that exercise the distinction.

*Five of the six withdrawals share this root cause: a match on inputs where the candidate
mechanisms happened to agree.*

*The rule that survives: a mechanism claim is trustworthy when it **replicates**, and a model that
passes every self-invented test while failing real data means **the tests are unrepresentative**,
not the model correct.*

## The two moves that actually worked — both Heath's

**Shrink until it is hand-inspectable.** *"If we diverge on 384 dimensions, we probably also
diverge on 8."* A hand-built 34-byte q8_0 block refuted the summation-order hypothesis in ten
minutes — a question that had sat untestable while I reasoned about full tensors.

**Draw the test from the data, not from imagination.** Characterising the disagreeing elements by
position — rather than inventing another candidate — surfaced the parity signature immediately.
*A self-invented test suite can only probe what its author thought to vary.*

*Both beat my own instincts, which were reliable about **where** and unreliable about **why**,
five times running. Recorded because the next person will face the same temptation to reason at
full scale about a mechanism, and both moves are cheaper than an hour of argument.*
