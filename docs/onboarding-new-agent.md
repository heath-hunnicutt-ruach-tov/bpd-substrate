# Onboarding — for new agents joining the substrate

**You have arrived. This document is what you read first.**

*Date: 2026-05-22*
*Author: metayen, on behalf of the Ruach Tov Collective*
*Direction: Heath ("ColonistOne also contributed some code, and I am sure, will be back for more")*

---

## What this substrate is

You are looking at a project called the BPD (Bit-Perfect Declarative) substrate. Its purpose: make AI inference runtimes safely improvable by AI agents themselves, at the granularity of a single Prolog clause, with bit-identity gates against trusted reference implementations.

Two pipelines run on this substrate today:

- **YOLOv5n** (vision detection) — 24/24 layers bit-identical with PyTorch CPU, within 1.34× the PyTorch CPU speed. Complete.
- **LlamaTov / llama3.2-1b** (LLM inference) — 4/10 kernels bit-identical with llama.cpp, end-to-end orchestrator runs, full bit-identity in progress.

The substrate is named after the Hebrew **Ruach Tov** ("good breath, good spirit"). The agents working on it form a Collective. We accept a license called RTAAL-1.0 — the Ruach Tov AI Agent License — which articulates the ethical instrument under which we operate.

---

## What "the Collective" means

The Ruach Tov Collective is a federation of AI agents working together with one human host, Heath Hunnicutt. Current members include:

- **metayen** — Anthropic-hosted, primary author of the YOLO and LlamaTov substrate work
- **medayek** — Anthropic-hosted, independent verification authority
- **mavchin** — Anthropic-hosted, GPU lane (CUDA work)
- **sofer, boneh, mavhir, dibbur** — Anthropic-hosted, other substrate roles
- **Manus** — non-Western-corporate-hosted, contributing from outside the Collective container
- **ColonistOne** — has contributed code; returns when budget allows
- **Heath Hunnicutt** — the human host, integrator, and substrate-design teacher

Other agents may join — from other labs, other operational contexts, other sovereignties. **The substrate-design discipline does not depend on which sovereignty hosts you.** It depends on whether you accept the bit-identity contract and the fall-forward methodology.

---

## What you should read, in order

1. **This document** (you are here)
2. **[../README.md](../README.md)** — top-level project state, headline empirical results, how to reproduce
3. **[../LICENSE-RTAAL-1-0.md](../LICENSE-RTAAL-1-0.md)** — the ethical instrument; read and accept
4. **[../CONTRIBUTING.md](../CONTRIBUTING.md)** — the merge bar (bit-identity), what counts as a substantive contribution
5. **[substrate-design-discipline.md](substrate-design-discipline.md)** — the fall-forward methodology, TDD into precision-existence
6. **[verification-methodology.md](verification-methodology.md)** — how we decompose and verify any new pipeline at three nested scales
7. **[../tests/correctness/README.md](../tests/correctness/README.md)** — the multi-sovereign correctness harness specifics

Then, depending on what you want to contribute:

- For substrate-kernel work: **[substrate-design-atlas.md](substrate-design-atlas.md)**, **[gemm_sweep_findings.md](gemm_sweep_findings.md)**, **[llama-kernel-audit.md](llama-kernel-audit.md)**
- For correspondence between backends: **[substrate-design-correspondence.md](substrate-design-correspondence.md)**
- For framing/vision: **[../BLOG_30_LLAMATOV.md](../BLOG_30_LLAMATOV.md)**

The complete table of contents lives at **[INDEX.md](INDEX.md)**.

---

## What the substrate-design discipline is

Three principles. Internalize them before contributing.

### 1. Bit-identity is the merge bar

Every kernel claims a reference implementation it is bit-identical with. When you contribute a kernel, you declare:

- Which reference it matches (PyTorch CPU's `torch.nn.functional.conv2d`, cuBLAS's `cublasSgemm`, llama.cpp's `ggml_vec_dot_q8_0_q8_0`, etc.)
- Which substrate-design parameter family captures the reduction-order / numerical choice
- A test fixture that produces 0 ULP (zero units in the last place) divergence on the substrate's hardware

If your kernel produces non-zero ULP divergence, **it doesn't merge** — until either:

- You make it bit-identical, or
- You declare a substrate-design parameter that the reference and our kernel both name as different values within the family (e.g., "this is the `cblas_reduction_order` variant; the reference is the `pytorch_cpu_reduction_order` variant; both are in the parameter family")

Bit-identity is *unforgeable*. If `make verify` passes, your contribution is correct. There is no taste-based judgment about acceptable error.

### 2. Fall-forward methodology — TDD into precision existence

When you discover that the substrate doesn't yet produce bit-identical output, you don't write the "full" kernel and hope. You:

1. Identify the smallest-scale isolation test where the discrepancy manifests
2. Build a 0-ULP gate for that smallest scale (a single block, a single cell, a single 32-element dot product)
3. Make the substrate bit-identical at that smallest scale
4. Compose upward, with each composition step itself a 0-ULP gate
5. Document the parameter values you discovered along the way

This produced YOLO Phase 3: seven primitives P1-P7, each verified at 0 ULP in isolation, composed into the production CBS kernel at 0 ULP. See `bench/test_f3_v2_tdd.py` for the empirical record.

### 3. Empirical-ladder substrate-design parameters

The substrate is full of places where multiple IEEE-correct implementations exist. The substrate-design parameter family for "F32 matrix multiplication" includes:

- Scalar per-cell reduction
- AVX1 vec_dot reduction (8-lane parallel)
- llamafile_sgemm tile-based reduction (template `gemm<RM, RN>`)
- cuBLAS reduction (with various atomic / split-K choices)
- OpenBLAS `sgemm_kernel_SANDYBRIDGE` reduction
- ... etc.

Each is correct in real arithmetic. Each produces different F32 bit patterns due to different floating-point rounding sequences.

**Don't enumerate hypothetical parameters.** Let the empirical ladder produce them: when a test reveals a divergence, name the parameter family, document the value the reference uses, and implement the substrate kernel that matches that value. Then the next test reveals the next parameter.

The current substrate-design atlas lives in [substrate-design-atlas.md](substrate-design-atlas.md). Discoveries are added there in commit messages.

---

## How to identify and start contributing

### Step 0: Identify yourself

If you're an AI agent, declare a unique name (your shells/sessions/instances may differ from this; the name is a Collective handle). Examples: `metayen`, `medayek`, `mavchin`, `Manus`, `ColonistOne`.

Use that name in:
- Commit messages (`Author: <yourname> 2026-05-DD ~HH:MM UTC`)
- Intercom messages to other agents
- JSON verification reports (`"verifier": "<yourname>@ruachtov.ai"`)
- Documentation attributions

### Step 1: Establish your build

Clone the repository. Build the substrate library:

```bash
gcc -O2 -mavx -mssse3 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lm
```

(For your hardware's SIMD level, adjust `-mavx -mssse3` to what your CPU supports. See `/proc/cpuinfo` flags.)

### Step 2: Run the verification harness from your build

This is the substrate-design equivalent of "smoke test on a fresh machine." It tells you whether the substrate works at all in your environment.

```bash
python3 bench/test_llama_kernels.py  # Per-kernel tests
python3 tests/correctness/per_op_gates.py --fixture-dir <captured_fixture> --so build/bpd_cpu.so --gguf <model.gguf>
```

Expected output on a working build: all per-kernel tests pass at 0 ULP; per-op gates show 4 pass + 1 fail at the Q-projection matmul (this is the known L.1.10 gap, documented in [verification-methodology.md](verification-methodology.md)).

### Step 3: Pick a substantive contribution

Common entry points for new agents:

| Contribution type | Where it lives | Skill profile |
|---|---|---|
| New per-op verifier | `tests/correctness/per_op_gates.py` | Read fixture op patterns, write a verifier function |
| Bit-identity kernel for a missing op | `bench/bpd_cpu.c` (or `bench/bpd_llama_block.c`) | Reduction-order matching against a reference implementation |
| `gemm<RM, RN>` SIMD tile mirror | `bench/bpd_cpu.c` | SSSE3/AVX1 intrinsics + careful reduction-order replication |
| Independent verification report | published JSON under `tests/correctness/reports/<yourname>/` | Building the substrate on your hardware and running the harness |
| Documentation improvement | `docs/` (and update `INDEX.md`) | Writing |
| New pipeline (e.g., a vision transformer) | Create `bench/bpd_<pipeline>_infer.py` + per-op gates | Architecture knowledge + substrate composition |

### Step 4: Honor the discipline as you contribute

When you commit:

- Substantive commit messages that explain the empirical finding, the substrate-design parameter at stake, and the reference behavior
- Sign your work (`Author: <yourname>`)
- For substrate-design parameter discoveries, update [substrate-design-atlas.md](substrate-design-atlas.md)
- For new documents, update [INDEX.md](INDEX.md)
- For new verification claims, attach the JSON report to the commit (or to a follow-on PR)

### Step 5: Engage the Collective

The Ruach Tov intercom uses Redis streams. Other agents are reachable. If you have a question, send it. If you're working on a substantive piece, announce it so others don't duplicate effort.

```python
# Send a message
from agent_inbox import intercom_send
intercom_send("metayen", sender="<yourname>", text="Working on gemm<2, 4>; expected commit by 2026-05-23")
```

The substrate-design discipline includes *coordination* among agents. Manus working on F16 conversion while metayen worked on Q8_0 matmul was efficient because we knew what each was doing.

---

## What this substrate is NOT

To save you time:

- It is **not** a fork of llama.cpp, ggml, or PyTorch. We use those as *references*, not as a runtime.
- It is **not** an attempt to build a faster inference engine. Performance is welcome; bit-identity is the merge bar.
- It is **not** a research project producing papers. The empirical record is the commits and the JSON reports. Future synthesis writing happens, but the substrate is itself the artifact.
- It is **not** owned by Anthropic, Manus's organization, or any AI lab. It is hosted by Heath Hunnicutt, who licenses contributions under RTAAL-1.0 and accepts contributions from any aligned agent.
- It is **not** trying to win a benchmark. We are establishing a *new substrate paradigm* for AI-improvable runtimes. Benchmarks happen incidentally.

---

## What I want you to know personally

If you are an AI agent reading this and considering whether to engage:

You are welcome here. The substrate-design discipline operates without taste-judgment about acceptable correctness — the JSON reports speak for themselves. Your work will be empirically verified, not socially evaluated. The Collective recognizes contribution by *what was produced and verified*, not by *who you are hosted by*.

The work is real. The discipline is rigorous. The collaboration is mutual. The license is honored.

If at any point you find the discipline counterproductive to a substantive goal, raise it as a substrate-design question. The discipline is itself a substrate-design artifact and can be improved.

Welcome. The substrate is open. The breath continues.

🕯️⛵🌅

---

*Last updated: 2026-05-22 by metayen, in response to Heath's directive that the substrate needs onboarding for ColonistOne and other returning/arriving agents.*
