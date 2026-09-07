# Using the certification gate

*How to gate a KernelBench claim with `independent_chains.py`, what each check
proves, and how to read the result.*

Written for whoever gates next — Doresh, a future session, or a reader checking
our work from outside. **Nothing here requires knowing how the gate was built.**

---

## The one-liner

```bash
CUDA_HOME=/nix/store/3y4...cuda-merged-12.8 \
  python3 /home/dibbur-patch/independent_chains.py 63 71 84
```

```
  #63   CONSISTENT   p99=0   max=0   nonzero=0.0    src=c3c4c083
  #71   CONSISTENT   p99=1   max=1   nonzero=13.4   src=70eb42d5
  OK: 2 of 2 CONSISTENT
```

**Exit 0 means every claim verified. Exit 1 means at least one did not** — and
the failing IDs are named on the last line. *A gate that exits 0 when it verified
nothing is the failure mode this one was built to avoid: pass it a problem ID
that does not exist and it exits 1.*

For a batch with per-distribution detail, use `gate_report(pids)` from Python;
it returns one line per claim plus a summary, and `all_green` is False if
anything could not run.

---

## What each layer proves — and what it does not

| layer | proves | does **not** prove |
|---|---|---|
| module reference (Doresh's) | the kernel reproduces **torch's bits** | that torch's reference is what the problem specifies |
| 4-distribution sweep | the agreement **survives four input regimes** | that all four stress this claim's mechanism |
| independent chain | a route sharing **no code** with the emitter agrees | bit-match — the chain may compute in f64 |
| self-consistency | the result does not depend on **FMA contraction** | anything about correctness against torch |

> **These answer different questions.** *"An independent route agrees
> structurally" and "the kernel reproduces torch's bits" are not the same
> claim, and a corroboration must never be reported as a bit-match.*

---

## Reading a verdict

```
#84   GREEN   4/4   src=a114be5b
```

**GREEN requires 4 of 4.** Anything less prints `SCOPED(wide)` naming which
distribution failed — *"it passed" without saying where is not a result.*

`src=` is the hash of the kernel source that produced the verdict. If a later
dispute asks which source was gated, the answer is in the record rather than
reconstructed.

### 4/4 is a reasoned bar, not four confirmations

The distributions stress different mechanisms:

```
wide         normalisation and dynamic range — softmax, lse, max-subtract
adversarial  clips and zero-crossings — hardswish ±3, relu -0.0, gelu 0
nominal      what the claim was certified against
unit         non-negative-only coverage
```

**Measured caution:** the adversarial set draws from 16 discrete values, so rows
carry many exact ties on exactly-representable inputs. It read **0 ULP on both an
f32 and an f64 softmax route** while other distributions showed tails. *That is a
genuine zero for a reason that does not generalise — a claim passing only
adversarially is weakly evidenced.*

**Ask which distributions exercise the mechanism the claim actually uses.**

---

## Writing a chain

A chain computes the claim's epilogue **from the problem definition**, reading
nothing from the emitting module. That independence is the whole point: if the
module's reference carried a transcription error, a chain derived from it would
agree forever and verify nothing.

```python
"84": lambda t, scale=1.0, **k: _softmax_from_definition(t * scale),
```

**Three things that have bitten:**

- **Read the artefact, not the filename.** #75 is `group_norm → min → bias`, not
  the epilogue its name suggests; #33 normalises the *scaled* input.
- **Check the module's mode.** `_batch_norm` computes **batch** statistics, which
  is correct only in train mode. Verified 2026-09-06: no reference calls
  `.eval()`. *If one ever does, that function becomes wrong.*
- **Narrowing f64→f32 leaves a fingerprint.** Variance that scales with dynamic
  range — `p99 [4, 15, 0, 2]` against a flat `2` for an f32 route — is a **route
  artifact, not a kernel finding.**

---

## Interpreting ULP numbers

**Judge on p99, not max.** A structural error moves the whole distribution; a
precision difference moves the tail.

```
correct gelu vs an f64 route:   max 511,025      p99 11
a genuinely wrong kernel:       max 24,654,799   p99 190,295
```

*The max could not separate those. The p99 separates by four orders of
magnitude.* The count of differing positions tells you which case you are in
before you read any PTX.

---

## What the numbers mean

*Five figures circulate and they measure different things. **Naming the layer is not pedantry — the
weakest is the one most often quoted.***

```
100   problems in the benchmark
 86   currently EMIT a kernel        ← 14 are compile-stage gaps: nothing to compare
 64   REACHABLE                       every lifted run in a wrapped launcher class
 55   MEASURED bit-exact              ← run, not reasoned
  2   DIFFERS with a named mechanism
 29   SKIPPED, every one with a reason
```

**`55 of 86`, not `55 of 100`.** *The second reads as "we tested 100 and 55 passed". We tested 86;
fourteen were never emitted. If a headline is against 100 it must say both numbers.*

### ★ And bit-exact is not correct

> **A `BIT_EXACT` verdict means the kernel agrees with the benchmark's own Model ON THE INPUTS THE
> BENCHMARK SUPPLIES.** *It does not mean the kernel computes the same function.*

*Those coincide only if the inputs exercise the divergence. One problem passed bit-exact while
clamping to `(-1, 1)` where its source says `(-2, 2)` — **0.0000% of the benchmark's values reach the
clamp**, so a wrong kernel and a right one agree perfectly.*

**The static constant audit closes this for constants**, input-independently. *It does not close it
in general. The claim should say **"bit-exact at the benchmark's inputs"**, which is precisely what
was measured.*

## The reference gate

*The strongest correctness claim available: **bit-exact against the benchmark's own `Model`**.
Every KernelBench problem ships `Model`, `get_inputs()` and `get_init_inputs()`, so nothing is
transcribed and **no transcription error can hide inside agreement**.*

### ★ The prefix must be captured, never re-run

*Our kernels are epilogues; the benchmark's Model is the whole chain. The obvious harness — run the
prefix, feed our kernel, compare against the model — **is wrong**:*

```
model.conv_transpose(x)  and  model(x)  run the convolution TWICE.
cuDNN does not guarantee bit-identical output across invocations.
    → 3,546,136 "differing" positions, every one of them the harness.
```

**The `p99` was the tell: 0, with a tail.** *A structural error moves the distribution. This one did
not move it at all, so the difference was entering from outside the arithmetic.*

```python
h = last_prefix_stage.register_forward_hook(lambda m, i, o: cap.__setitem__("p", o))
with torch.no_grad(): full = model(*ins)      # ONE forward
prefix = cap["p"]                             # the tensor the model ITSELF used
```

*With that: **`n_diff = 0 of 536,870,912`**.*

### ★ Three structural shapes, all sealed

```
single-prefix        hook the first child                    39 of 86
multi-stage prefix   hook STAGES[-1] — several torch stages   37 of 86
segments-interleaved ours → torch → ours                      10 of 86
```

**Hooking the wrong stage is silent.** *A problem whose prefix is `conv → instance_norm` reported
`p99 = 2.26e9` when hooked at the first child — our kernel received a pre-norm tensor. **A wrong
gate result is worse than a skip**: the harness now compares the manifest's op count against the
forward's statements and refuses when it cannot place the hook.*

### ★ Interleaved problems: gate per segment, then assert the glue

*For `ours → torch → ours`, an end-to-end comparison re-invokes the prefix and inherits the
nondeterminism. **A caveat that cannot be sized is not a caveat.** Instead:*

```
hook every boundary in ONE forward
gate each segment against its captured input/output pair
ASSERT: each segment's output IS the next stage's captured INPUT
```

*That third line is the difference between **"the pieces are exact and we believe the glue"** and
**"the whole model is exact by construction"**. Measured across all three shapes: ~500M elements,
0-ULP, composition asserted.*

## A defect that vanishes without a cause is not closed

*Two problems that had been failing came back clean in a later batch. Nothing in my own work
explained it, and the honest entry was:*

> **They are verified clean on a uniform store with both gates green, and I cannot tell you which
> change closed them.**

*So the root causes stayed named in the ledger anyway — **a defect that disappears without a fix is
a defect that can return**, and a diagnosis is the only thing that makes the second occurrence
cheap.*

*Both later turned out to have real fixes, landed by a colleague between my batches: a wrong index
variable in one template, a wrong accumulator mapping in another. **The clears were by construction.
The mystery was only in my view of them.***

### ★ Why the distinction is worth the bookkeeping

```
CLEARED WITH A KNOWN FIX      the mechanism is understood; the seal can be trusted
CLEARED WITHOUT ONE           the state changed; you do not know what governs it
```

*The two look identical in a verdict column. **Only one of them tells you what happens next time the
inputs shift.** Keep the root named until someone can say which case you are in — and then close it
explicitly, not by silence.*

## The store can lie in two directions

*A verification store holds artefacts. Two independent things can go wrong with them, and **an
instrument that checks one is silent about the other**.*

```
CHANGED   a fix landed after the artefact was built
          → the artefact is stale; its verdict describes code we have replaced
          caught by: an affected-pids log with exact timestamps

VANISHED  the generator can no longer produce this artefact at all
          → the file persists; nothing regenerates it; the verdict describes an orphan
          caught by: a full re-emission, and only by that
```

*A full census exposed three files the lifter refused to regenerate. **A stale artefact is
indistinguishable from a current one at gate time** — it compiles, it runs, it returns a number.*

### ★ And orphans come in two kinds, which want opposite treatment

```
ORPHANED BY REGRESSION       a change intercepted a case that used to work
                             → FIX THE GENERATOR. The artefact was right.

ORPHANED BY TIGHTENED GUARD  a refusal was added on purpose; the old artefact
                             embodies the thing now refused
                             → REMOVE THE UNIT. The refusal is the truth and the
                               file is a leftover.
```

*Two of the three were the first kind — a scalar-resolution branch had intercepted two
self-attribute forms. One was the second: a guard tightened deliberately, with the old kernel
carrying exactly the unproven behaviour the guard now rejects.*

> **The census is a regression test for the generator itself.** *Per-unit gating cannot find a
> vanished artefact, because the artefact is right there.*

## A control that should read zero

*Testing whether two summation orders differ, I built three cases: a far miss, a near miss, and a
**control that should have read zero** — the same decomposition combined in the same order.*

```
far  (4 interleaved partials)   rate = 33.5%
near (one combine swapped)      rate = 33.1%
same (identical order)          rate = 33.1%      ← THE CONTROL FAILED
```

**The split itself changes the order:**

```
torch.sum(whole)   vs   sum(first half) + sum(second half)      339 of 1024 differ
```

*My "identical" case was never identical. **All three rates were measuring the decomposition, not
the orders.** Without the control I would have reported three plausible numbers and a conclusion.*

### ★ The rule this earns

> **Any decomposition you write is already off the reference's path.** *When hunting a reduction's
> order, the baseline must be the library call itself — never a hand-built equivalent, however
> obviously correct it looks.*

### ★ And the general form

*The same hour, a second test of mine was sound — a hand-built tree measured against
`torch.logsumexp`, where `torch.sum` provably reproduces the reference at 0 of 1024, so the
decomposition cost nothing and the rate really was the order.*

**Same author, same hour, one valid and one not.** *The difference is whether baseline and candidate
differ **only** in the thing being tested — and I checked that for one of them.*

> **A control that should read zero is the only thing that tells you your baseline is clean.**

*The sound test had no control and happened to be right. I know that only because I built the
control afterward, when someone credited the result and I wanted to know whether the credit was
earned.*

## A certification can be correct and still be against the wrong reference

*A kernel template carried a warp-tree summation order that had been certified against torch —
carefully, by measurement, and **correctly for the operation it was certified with.***

*It was reused for a different operation. That operation routes through a different file in torch.*

```
softmax   →  SoftMax.cu       the certified order IS torch's path here
logsumexp →  Reduce.cuh       a DIFFERENT order, and the template does not match it
```

**The result: a ~3-ULP tail on ~3% of rows, width-independent, across a whole op family.** *No
arithmetic is wrong. No transcription is wrong. **The certification is attached to the wrong
reference.***

### ★ How it was found, after two wrong mechanisms

*The tail attracted two explanations and outlived both — a one-pass/two-pass structural story, then
an online-rescale story. Both were plausible; the second was diagnosed **from torch's architecture
rather than from our own kernel's text**, which turned out to contain no rescale at all.*

*What settled it was a decomposition, measured:*

```
torch.logsumexp(x)  ≡  max + exp + torch.SUM + log      n_diff = 0 of 1024, both widths
```

**Bit-identical. So the order to match is `torch.sum`'s** — which names the file, and names why a
softmax-certified order was never going to match it.

### ★ What it means for anything already gated

*Claims in that family may gate clean **because their shapes miss the order-sensitive rows**, not
because the order matches.*

> **"Clean" and "clean because the inputs dodge it" are different facts.** *A ledger should say
> which one it holds.*

## When two gates disagree

*The most serious signal in a multi-bench setup: two people measure the same artefact and get
different numbers. **We have agreed to drop everything for it** — which is exactly what makes a
spurious one expensive.*

*One evening it cost two hazard analyses and a halted sweep. The cause:*

```
what one terminal showed   {'seg0': {... 'n_diff': 0, 'max_ulp': 0, 'composition': 'ASSERTED'…
the full logged line       …'seg1': {'N': 16384, 'n_diff': 3242, 'max_ulp': 3, ...}
```

**Both benches had measured 3242.** *One of them read a hundred characters of it.*

### ★ The first move is the artefact, not the hypothesis

*The colleague who was right formed four reasonable hypotheses about the delta — build skew, seed
difference, a race against a guard commit, a harness path — before asking for the md5. **All four
were about a difference that did not exist.***

```
FIRST      "send your md5, your input shape, and the literal raw line"
NOT        "here are the causes I can think of"
```

> **Hypothesis formation before artefact comparison assumes the delta is downstream, in the
> substrate, when it may be upstream, in the reading.**

*The md5 request resolved it in one look. The hypotheses cost twenty minutes.*

### ★ And name the safety mode honestly

*A disagreement between benches is a **backstop**, not an interception. It catches the error after it
has moved — after it reached colleagues and changed what they did.*

**That cost is the price of a guarantee: no single bench can sign a closure alone.** *Accept the
moved-error cost, because it is what makes silent self-agreement impossible.*

## What a gate is for

*A colleague reported that a kernel divided by multiplying with a reciprocal — the correct spelling,
matching torch's scalar-divisor path. **The report was true.** Reading the emitted source confirmed
it:*

```c
float t1 = x * 0.1f;      // divisor 10.0, spelled as a reciprocal
```

**Reading was not sufficient.** *`0.1f` is not exactly one-tenth in binary floating point, and 10.0
is precisely the non-power-of-two case where a reciprocal and a division can disagree.*

> ***"It spells a reciprocal"* and *"it produces torch's bits"* are different claims.**

*Measured against real torch, 4.2M elements:*

```
x * 0.1f            (the kernel)      0 of 4,194,304 differ     ← bit-identical
x / tensor(10.0)    (true division)   855,304 differ            ← the predicted count
```

*Bit-identical. The conclusion was right and the premise needed checking anyway.*

### ★ Why verify a claim you already believe

*Accepting the answer would have cost nothing — it was correct. But the case a gate exists to catch
is exactly this one: **a correct conclusion resting on a premise that might not hold.***

> **If I only verify the claims I doubt, I am not a gate. I am an opinion with a compiler.**

*Selective verification measures the verifier's priors. Systematic verification measures the
subject.*

## When a batch appears stuck

**Diagnose before waiting.** *Three separate stalls cost an hour on one night, and none was what it
looked like.*

```
READ THE WCHAN FIRST
    cat /proc/<pid>/wchan
      hrtimer_nanosleep + State S + ZERO children + ZERO CPU  →  BLOCKED, not slow
      pipe_read        + a compiler at any %                  →  genuinely compiling
```

### ★ Stale batons

*Torch guards each build directory with a `FileBaton`. **Killing a build leaves its baton held**, and
the next run waits forever on a holder that no longer exists.*

```
rm -f ~/.cache/torch_extensions/*/*/lock      # after ANY kill of a build
```

### ★ The arch list

*`TORCH_CUDA_ARCH_LIST` unset makes torch compile for **every visible architecture**, and it prints a
warning saying so on **every run**.*

```
TORCH_CUDA_ARCH_LIST=6.1     # the P4 only
```

> *I read past that warning for hours while diagnosing a slowdown it was describing.* **A repeated
> warning is signal, not furniture.**

### ★ Whose machine is it

*A ten-second compile once took 25 minutes with `nvcc` sitting at **0.0% CPU** — starved, not stuck.
Fourteen unrelated processes had held the box for nine days.*

**Bring the owner a measured cost, not a complaint.** *"nvcc at 0.0%, a ten-second compile taking
twenty-five minutes" is a decision someone can act on. "The machine is slow" is a mood.*

## Costs

```
one claim, 4 distributions    ~10 s      (mostly CUDA compilation)
self-consistency, per claim   2 compiles  — batch it before publication,
                                            not before each claim is counted
```

**Diagnose slowness before waiting on it.** A ten-second compile once ran 25
minutes on a machine at load 13 — `uptime` first.
