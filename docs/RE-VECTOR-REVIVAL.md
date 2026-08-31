# RE-VECTOR revival — the by-construction ladder, reconstructed and re-gated

*Iyun, 2026-08-31. The record of how RE-VECTOR rungs 1–4 were revived after the
CPU-target IR emitters evaporated to /tmp, and the reconstruction method — so the
next evaporation costs an evening, not a campaign.*

## What was lost, and what survived

When the BPD kernel-fusion programme was reinstated (Heath, 2026-08-31), the built
kernel artifacts and the CPU-target LLVM-IR emitters were found **absent from git** —
they had lived in `/tmp` and the enclave (`/home/mavhir/soa_swap`), never landed.
`prolog_to_llvm.pl` (which held `emit_q8_0_dot`) was gone with them.

**Revival rule one, learned the hard way: land in git first.** An artifact that only
lives in ephemeral storage is one reboot from gone. This whole document exists because
that rule was violated in June.

Measured inventory (2026-08-31):

| Piece | Status | Note |
|---|---|---|
| `lib/swiglu_fused_emitter.pl` | **SURVIVED** | the proven-0-ULP SwiGLU — the exact fusion the June hand-kernel dropped |
| gate/fixture framework (`tests/correctness/*`) | **SURVIVED** | the verification harness |
| `prolog_to_llvm.pl` (dot/gemv/quantize emitters) | **LOST** | never committed |
| `soa_ffn_emitter.pl` | **LOST** | never committed |
| mavhir's CUDA kernel set (`kernels/soa/*.cuh`) | **LANDED** (`e837a53`) | GPU built forms — NOT the CPU-IR emitters |

The emitters (good **design**) surviving in git was worth more than the built artifacts
(good **luck**) surviving on the enclave — because emitters regenerate the artifacts,
but not vice versa. The revival tilts further toward design: rung 5 makes the whole set
regenerable-by-construction, the permanent answer to /tmp-evaporation.

## The reconstruction method

Each lost emitter was reconstructed **from the June conversation corpus** (conv 124,
windows 375–377), which preserved not just the results but the *mechanism* — the exact
signatures, the FP discipline, the composition structure. That the quantize crux
reproduced both FP knobs first-try is what the June honesty bought: the corpus recorded
*how*, not just *that*.

**The discipline at every atom (earned, not asserted):**

1. **Reconstruct** the emitter from the corpus spec.
2. **Emit** — run the Prolog, produce the LLVM IR.
3. **Parse-check** — `llvm-as` (catches phi/block wiring bugs; caught a real one in June).
4. **Gate 0-ULP** — `llc` lower → link with a CPU reference doing the *identical* math →
   compare **bit-exact** (`memcmp`, not epsilon). *"It parses" is not "0-ULP."*
5. **Land in git**, each atom its own commit, hash in the report.

Toolchain (all present on the reconstruction instance): `swipl`, `llvm-as`, `llc`, `cc`.
The gates are **run**, not merely emitted.

## The revived ladder — atoms and hashes

| Rung | Emitter | Commit | Gate result |
|---|---|---|---|
| 1 | `soa_q8_0_dot_emitter.pl` | `db89f01` | **0 ULP**, 64/64 block-counts (nb=1..64) |
| 2 | `soa_gemv_emitter.pl` | `3bd14da` | **0 ULP**, 4/4 shapes (dot looped over rows) |
| 3 | `soa_q8_0_quantize_emitter.pl` | `c026ca0` | **0 ULP**, quants AND scales byte-exact, 5/5 sizes |
| — | `swiglu_fused_emitter.pl` | (survived) | the fusion atom, unchanged |
| 4 | `soa_ffn_emitter.pl` | `798cfc1` | **0 ULP**, full FFN block, 256 outputs |

**The atom specs (recovered):**
- **dot**: `float @bpd_q8_0_dot(ptr wq, ptr wd, ptr aq, ptr ad, i32 nb)` — exact int32
  inner accumulate over 32 lanes; per-block `scale=fmul(wd,ad)`, `dotf=sitofp(isum)`,
  `prod=fmul(scale,dotf)`, `acc=fadd`. SSE-no-FMA scalar order for bit-identity.
- **gemv**: `void @bpd_soa_gemv_q8_0(...)` — the dot looped over rows; `dst[r]=dot(W_row_r,a)`.
- **quantize** (the FP crux): `amax/127` → **fp16(d) round-trip** (`fptrunc`/`fpext` to
  half — ggml stores the scale as fp16) → `id=(d!=0)?1/d:0` → `q=(i8)round(x·id)` via
  `llvm.round` (round-half-away = `roundf`). Both FP knobs must match or every quant drifts.

## Why the composition is bit-identical BY CONSTRUCTION

The FFN emitter composes **verified atoms**: `gemv(gate) + gemv(up) → swiglu → quantize →
gemv(down)`. The rung-4 gate compares the IR composition against a CPU reference calling
the *same atoms in the same order* — so any difference is a **composition error** (wrong
stride/index/dst), which the gate catches. Because SwiGLU is a **linked verified atom**,
not a hand-step, the June fusion-drop bug (a forgotten hand-fusion) is **structurally
impossible** here. This is the two-track insurance the June bug proved we need: the
CPU-IR by-construction track beside the CUDA hand-kernel track.

## Honest scope

- Rungs 1–4 are **correctness microbenches**, NOT a runnable engine. There is no "tok/s at
  rung 4" — comparing to a full-model rate is apples-to-nothing (the refused June number
  stays refused).
- The IR targets **SSE-no-FMA** by design (for bit-identity) — deliberately
  under-vectorized; slow on purpose, correctness-first.
- **CPU-bit-identity does NOT transfer to the GPU MMQ Q8_1 path.** The final gate is
  GPU-token-identity, always. Rung 5 (GPU lowering) must gate against a *verified* GPU
  reference — never build toward a reference that isn't itself verified.

## Status

RE-VECTOR rungs 1–4 revived at June-parity in one evening, every atom individually
0-ULP-gated, the composition gate run linked, all six revival commits on
`origin/rtaal-1-1`. Rung 5 (full model, GPU lowering) opens as the by-construction
frontier **after** the CUDA track's ground truth is re-gated (mavhir's rebuilt .so +
medayek's strata) — so rung 5 has something honest to gate against.

*Every rung is now in git. The next evaporation costs an evening.*
