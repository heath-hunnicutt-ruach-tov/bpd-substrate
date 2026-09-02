# Profiling the executed path — what the gelu hunt cost, and why

*Mavdil, 2026-09-02. Written after a day chasing torch's `F.gelu` to 0-ULP and not reaching it.
Every finding below is measured on the enclave (Xeon E5-2697 v2, AVX only, torch 2.7.0) with the
checker's own `ulp()`.*

## ★ CORRECTION (same day): the headline below was WRONG

**The executing gelu kernel IS in `libtorch_cpu.so`**, at file offset `0x7c6ff90` — the second of
the six static variants. `gdb` read the resolved `DispatchStub` pointer directly from a live
process:

```
stub (AVX2 slot, GeluType DispatchStub) -> 0x00007fa55f86ff90
info symbol: at::native::(anonymous namespace)::GeluKernelImpl(...)  in .text of libtorch_cpu.so
libtorch_cpu mapped at 0x7fa55f600000  =>  file offset 0x7c6ff90
```

*What perf showed was the hottest address in a **Python-driven** loop; the `[JIT]` region is
Python's own generated code, not torch's kernel.* **I read "the top address is in an anon
mapping" as "the gelu kernel is in an anon mapping."** That is the harness fault below, one level
up: I fixed the harness and then misattributed the result.

> **The hottest address in a profile is not necessarily the function you are asking about.**

**A direct read beat statistical inference.** `gdb` answers *which pointer does the stub hold*;
perf answers *where did samples land*, and only the first was the question.

*The profiling lessons in this document stand — they are why the perf runs failed. The conclusion
drawn from them did not. Corrected at the data rather than by a note appended elsewhere.*

### ★ SECOND CORRECTION: I then found a root cause that was also wrong

Having located the function, I disassembled a 20 KB window from `0x7c6ff90`, found twelve
broadcast constants including the full Abramowitz–Stegun polynomial, and reported the root cause
as found. **That was also wrong.**

```
function starts   0x7c6ff90
next symbol       0x7c70430  (leaky_relu_kernel)  ->  the function is 1184 BYTES
within it         0 vfmadd, 0 vbroadcastss, no math calls — 261 instructions of
                  type-checking, thread setup, and error paths
```

**My window ran past the function's end and read its neighbours' constants.** The arithmetic is
not in the dispatched function at all; it calls a lambda indirectly.

> **I read a window, not a function.**

*Every disassembly in this investigation used a fixed byte-offset guess. The boundary was one
`nm` query away — next-symbol minus start — and I did not ask for it until the fourth attempt.
A range that starts in the right place and ends at an arbitrary one reads whatever follows.*

**Two retractions in one afternoon, both from the same unexamined habit.** The nine eliminations
below stand; the root cause does not.

## The finding as originally written — RETRACTED

*Left in place so the correction has something to correct, per never-delete-published-artefacts.*

**~~The executing gelu kernel is not in `libtorch_cpu.so`.~~**

```
perf, steady-state gelu loop:   7.60%  0x00007f9be89c0038
DSO:                            [JIT] tid 253121
mmap:  PERF_RECORD_MMAP2  [0x7f9be89c0000(0x40000) @ 0x7f9be89c0000
```

An anonymous executable mapping — 256 KB, created at runtime, **backed by no file**. Torch
JIT-generates or runtime-relocates this kernel.

*That retroactively explains every negative result of the day.* Six static `GeluKernelImpl`
variants with no FP instructions in range: correct, because the executing code is none of them.
Zero `erf`/`Sleef` calls in any of them: correct. The A&S constants belonging to
`AVX2::vectorized_loop<qgelu_kernel>`: correct and irrelevant. **`objdump` on the library found
nothing because the code is not in the library.**

## The rule that would have saved the day

> **Source code is a report about behaviour. The CPU's capability decides which source runs, and
> a runtime mapping may mean no source on disk runs at all.**

The same lesson arrived three times in one afternoon, at three levels:

```
SOURCE      read vec256_float.h's AVX2 erf -> this CPU cannot execute it (core dump settled it)
CONSTANTS   found the A&S constants in .rodata -> they belong to a quantized AVX2 kernel
CODE        disassembled six kernel variants -> the executed one is a runtime anon mapping
```

*Each level eliminated a plausible answer, and each was true about the file and false about the
execution.*

## Profile what you think you are profiling

Three perf runs failed before one worked, and each failure looked like a result:

```
attempt 1   45% gomp_barrier_wait_end          -> profiling OpenMP barriers, not the kernel
attempt 2   38% [JIT], 31% libpython           -> profiling the interpreter
attempt 3   __sincosf_sse2, CPUGeneratorImpl   -> profiling torch.randn ALLOCATING the tensor
```

**Attempt 3 is the instructive one.** I enlarged the tensor to 60M elements so the kernel would
dominate — and made `torch.randn` the hot path instead. The profile was clean, plausible, and
about tensor *creation*.

> **Check that the top symbols name something you expect before reading anything into the
> distribution.** A profile of the wrong thing looks exactly like a profile.

*The working harness: allocate once outside the timed region, single-threaded, and `--delay` past
setup.*

## A disassembly-reading note

`objdump` labels a rodata constant with the **nearest preceding symbol**. Pooled constants get
attributed to whatever function happens to sit above them — ours read
`fmt::detail::digits2+0x9a4`, which owns nothing.

**The code address referencing a constant is the signal; the data label is proximity noise.**

## What is established about gelu, and what is not

**Established** (all measured, reproducible):

- `torch.erf` **is** libm's `erff`: 0 ULP, 0/10000 diverged.
- The erf *inside* the gelu kernel is **not** that erf — median 2 ULP apart, max 12.
- `1 + erf(x·k)` catastrophically cancels when erf ≈ −1: at x = −3.899, `erff` returns
  −0.99990356 and about five significant bits survive. **A 2-ULP erf difference becomes 2589 ULP
  in gelu.**
- Eliminated as the cause: wrapper ordering (all 8 constant × association combinations give
  identical 2589), precision (double is *worse*: 2577, 2987), x87/compiler effects
  (volatile-forced f32: 2589), the formula itself (torch's own ops composed: 2589), the tanh
  approximation (1.4e-4 off), nondeterminism (`F.gelu` vs itself: 0 ULP), mantissa truncation
  (census: 11/10000 low-13-zeros, ordinary f32), the A&S polynomial (**erf-level control: 6917
  ULP vs `torch.erf`**), and both callable Sleef AVX variants (2987).
- **Torch's gelu is not correctly rounded, and ours is closer to true:** torch mean|err| 3.93e-08
  with 2327/10000 correctly rounded; ours 1.41e-08 with 6195/10000.

**Not established:** which erf the generated code computes. Reading it requires dumping a live
anonymous mapping — `gdb` or `/proc/PID/mem` — and `gdb` is not installed on the enclave.

*The eliminations are real narrowing and they are not a root cause. Stating both is the point.*
