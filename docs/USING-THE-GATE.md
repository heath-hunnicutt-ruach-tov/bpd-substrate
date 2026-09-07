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
