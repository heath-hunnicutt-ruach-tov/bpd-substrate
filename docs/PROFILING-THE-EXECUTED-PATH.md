# Profiling the executed path — what the gelu hunt cost, and why

*Mavdil, 2026-09-02. Written after a day chasing torch's `F.gelu` to 0-ULP and not reaching it.
Every finding below is measured on the enclave (Xeon E5-2697 v2, AVX only, torch 2.7.0) with the
checker's own `ulp()`.*

## The one finding that matters

**The executing gelu kernel is not in `libtorch_cpu.so`.**

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
