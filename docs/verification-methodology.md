# Verification Methodology — How We Decompose and Verify a Pipeline

**Discipline & tools for empirically locating bit-identity divergence at any scale.**

*Date: 2026-05-22*
*Author: metayen, drawing on substrate-design discipline crystallized through YOLO Phase 3 and applied to LlamaTov Phase L.1*
*Direction: Heath ("create the test infrastructure that agents like Manus would use to detect the bit-identical correctness of each step")*

---

## The substrate-design discipline behind this document

When the substrate claims "bit-identical with [reference]", the claim must be *empirically locatable*. If the substrate is currently NOT bit-identical with the reference, the discipline requires us to be able to say exactly where the divergence enters, at what magnitude, and along which reduction-order axis.

This document captures the methodology for that empirical location — the *recipe* for taking a new pipeline (LlamaTov, a new model architecture, a new backend) from "we ran it, but it diverges" to "we know exactly which kernel's reduction order produces the wrong F32 bits."

The discipline operates at **three nested scales**:

1. **Per-operation** — does each ggml/PyTorch/reference operation's output bytes match ours?
2. **Per-layer** — does each transformer layer's (or pipeline stage's) residual stream output bytes match ours?
3. **Per-pass** — does the full forward pass's final output (logits, detections, predictions) match the reference's?

Each scale's gates compose into the next. If all per-op gates pass, all per-layer gates should pass. If all per-layer gates pass, the per-pass gate should pass. When a gate fails at scale N, the gates at scale N+1 (finer) point at the specific sub-operation where divergence enters.

---

## When to use this methodology

You should reach for this methodology when:

- You have a new pipeline (e.g., another LLM architecture, a vision transformer, a diffusion model)
- You can run a reference implementation (e.g., llama.cpp, PyTorch, official model code) that produces gold-standard intermediate values
- Your substrate's output of that pipeline does not bit-match the reference
- You want to know *empirically and locatably* where the divergence enters

You should NOT reach for this methodology when:

- You haven't established a per-kernel 0-ULP gate against a reference for your kernels yet (build per-kernel gates first)
- You're trying to debug a structural bug (use a debugger; this methodology assumes the substrate produces *plausible* but *non-bit-identical* output)
- The reference is itself non-deterministic (e.g., different runs of CUDA with non-deterministic atomics produce different outputs; you'd need a different verification approach)

---

## The tools

The L.1 verification toolchain lives at `tests/correctness/` and `bench/`:

### Per-operation gates: `tests/correctness/per_op_gates.py`

For each captured operation in a reference inference trace:

1. Identifies the operation type (GET_ROWS, RMS_NORM, MUL, MUL_MAT, etc.)
2. Locates the source tensors (inputs) and the captured output
3. Calls our substrate's equivalent kernel on the same inputs
4. Compares the result to the captured output byte-for-byte

Emits a JSON report with one entry per checked operation:

```json
{
  "op_idx": 7,
  "op_name": "Qcur-0",
  "op_desc": "MUL_MAT",
  "shape": [2048, 6, 1, 1],
  "max_abs": 1.907e-06,
  "max_ulp": 524288,
  "n_diff": 8214,
  "n_total": 12288,
  "status": "fail"
}
```

Extension model: to add a new operation type, write a `verify_<op_type>` function in `per_op_gates.py` and register it in the `OP_VERIFIERS` dict. The function takes `(lib, tensors, op, idx, ctx)` and returns a dict with `status` and metrics.

### Per-layer gates: `tests/correctness/per_layer_gates.py`

Drives the substrate's per-layer composition (`bpd_llama_block_cpu`) for each layer of the model with real weights, comparing the residual stream output to the reference's captured `l_out-N` fixture. Reports per-layer max_abs, max_ulp, cosine_similarity.

This catches drift accumulation patterns: even when per-op gates show small per-cell divergences (1e-6 magnitude), they may accumulate monotonically through 16 transformer layers and produce a qualitatively different output distribution at the top.

### End-to-end gate: `tests/correctness/end_to_end_gate.py`

Runs the full orchestrator with `--dump-logits`, compares the resulting logit vector to the reference's captured final-layer logits. Produces a multi-metric verdict:

| Metric | Meaning |
|---|---|
| `argmax_match` | Boolean: did we pick the same top-1? |
| `pearson_correlation` | Logit-vector correlation; 1.0 = same distribution |
| `cosine_similarity` | Geometric similarity of the logit vectors |
| `top_k_overlap` | How many of our top-10 overlap reference's top-10 |
| `our_argmax_rank_in_ref` | Where our argmax ranks in the reference's distribution |
| `ref_argmax_rank_in_ours` | Where reference's argmax ranks in ours |
| `L2_ratio_ours_to_ref` | Magnitude ratio; 1.0 = same scale |

Final 4-level verdict: `PASS_BIT_IDENTICAL_ARGMAX` / `PASS_HIGH_CORRELATION` (>0.99) / `PASS_GOOD_CORRELATION` (>0.9) / `FAIL`.

### Sub-operation decomposition: `tests/correctness/decompose_matmul.py`

For operations with internal structure (like a matmul, which is composed of many cell-dot products), the decomposition splits the operation into its sub-steps:

- **Step A**: Are the inputs (e.g., quantized activations) byte-identical with the reference's intermediates?
- **Step B**: Is a single cell of the output computed correctly?
- **Step C**: Are all cells of the output computed correctly?

When A and B pass but C fails, the diagnosis is precise: **the per-cell reductions don't compose into per-tensor identity** because the reference uses a *different reduction order across cells* than our scalar per-cell kernel.

This is the substantive substrate-design substantive — *active* — substrate-design parameter family for full-matmul bit-identity: **reduction order across SIMD lanes / tile dimensions**.

### Empirical-bisection ad-hoc scripts: `bench/bpd_layer_bisect.py`, `bench/bpd_layer0_bisect.py`

Less formal than the harness, these are the *first-look* scripts when you want to quickly answer "where does divergence enter the chain?" before building the full report-generating harness. Useful during interactive empirical exploration.

---

## The methodology, in steps

When a new pipeline diverges from its reference, walk this recipe:

### Step 1: Establish the per-kernel 0-ULP floor

Before any composition can be expected to match, each individual kernel must match the reference's equivalent operation on isolated inputs. Build the per-kernel test harness first.

For LlamaTov this happened in commits `57e29e1` (L.1.2 MUL), `07951f2` (L.1.1 EMBED LOOKUP), `3667679` (L.1.9 Q8_0 DEQUANT), `76106bf` (L.1.10 Q8_0 MATMUL for one shape).

### Step 2: Run the end-to-end gate

If end-to-end output matches the reference: you're done. Ship the verification.

If it diverges: run the per-layer gate to find the first layer that diverges. This is the empirical-bisection step.

### Step 3: Run the per-layer gate

The script prints per-layer cosine similarity and max_abs. The first layer where these are non-trivially worse than the predecessor identifies *where divergence enters*.

For LlamaTov this produced:
```
embed   : 0 ULP / 12288
layer  0: cos_sim=0.999996  ← first divergent layer
layer  1: cos_sim=0.999990
layer  2: cos_sim=0.999986
...
layer 14: cos_sim=0.999832
```

### Step 4: Run the per-op gate

For the first divergent layer, the per-op gate identifies the first divergent operation within that layer.

For LlamaTov layer 0:
```
✅ inp_embd                : 0 ULP / 12288
✅ norm-0 (bare RMSNorm)   : 0 ULP / 12288
✅ attn_norm-0 (MUL)       : 0 ULP / 12288
🟡 Qcur-0 (Q-projection)   : max_abs=1.9e-6, ULP=524288, n_diff=8214/12288
```

### Step 5: Decompose the divergent operation

If the divergent operation has internal structure, decompose it. For a matmul, that means:
- Quantization step (if quantized)
- Per-cell block-dot step
- Cell-by-cell composition step

For LlamaTov Q-projection:
- Step A (quantizer): 0/2176 bytes differ for all 6 input rows ✅
- Step B (single cell at m=0, n=0): 0 ULP vs reference ✅
- Step C (full matmul): 8214/12288 cells diverge by 1e-8 to 1e-7 each ❌

This pattern — A and B pass, C fails — is *empirically diagnostic*: the per-cell scalar reduction matches the reference for some cells but not others, indicating the reference uses SIMD-lane parallelism that reorders the F32 additions enough to produce different bit patterns for those cells.

### Step 6: Form the falsifiable substrate-design hypothesis

Once decomposition has located the divergence to a specific reduction order, the substrate-design discipline calls for naming the parameter family. For LlamaTov Q-projection, the parameter is:

> **`matmul_tile_reduction_order`**: the F32 accumulation order across SIMD lanes and tile dimensions in a GEMM-style matmul. Reference (llama.cpp via llamafile_sgemm) uses templated `gemm<RM, RN>` with `RM, RN in {1,2,3,4}`, each template using a specific 8-lane `__m256` accumulator pattern. Our scalar per-cell reduction differs from this for tile shapes where the lane parallelism reorders additions.

This is the *next* substrate-design parameter to ladder out, along with whatever other parameters compose the full kernel.

### Step 7: Implement, verify, advance

Implement the kernel variant that matches the reference's reduction order. Re-run the per-op gate. If it passes, advance to per-layer. If that passes, advance to end-to-end. If end-to-end passes, the pipeline is bit-identical.

---

## Multi-sovereign verification

A critical substrate-design principle from RTAAL-1.0 Phase L.1:

> "The correctness of an inference substrate is not a property of the substrate alone. It is a property of the substrate **as verified by sovereigns other than the one who wrote it**."

This means: if `metayen@anthropic` writes the verification harness, the substrate's correctness claim only acquires its full weight when `manus@ruachtov` (operating from a different sovereignty, with different BLAS libraries, possibly different SIMD ISA) runs the same harness on his hardware and produces a *byte-comparable* JSON report.

The harness is **designed for this** — each script emits a JSON report containing:

- The verifier's identity (`metayen@ruachtov.ai`, `manus@ruachtov.ai`, etc.)
- The fixture sha256 (so different verifiers know whether they're testing against the same captured trace)
- The hardware context (CPU model, SIMD flags, compiler)
- The per-op / per-layer / per-pass metrics

Two verifiers comparing reports can then ask: do we see the same per-op pattern? If yes, the substrate behaves consistently across our environments. If no, the divergence is environment-level (BLAS, ISA), not substrate-level — and we've discovered something substantive about which substrate-design parameters are environment-sensitive.

---

## Specific instructions for other agents

### Manus

You verify by:

1. Build a patched llama-eval-callback from your llama.cpp clone using `tests/correctness/build_eval_callback.sh`
2. Capture your own fixture: `LLAMA_DUMP_DIR=/tmp/manus_fixture <eval-callback> -m ... -p "Hello, my name is" -n 1 --temp 0 --seed 42`
3. Build our substrate: `gcc -O2 -mavx -mssse3 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lm`
4. Run the three gates against your fixture, emitting `manus_per_op_report.json`, `manus_per_layer_report.json`, `manus_end_to_end_report.json`
5. Open a PR adding those reports under `tests/correctness/reports/manus/` so they're versioned alongside the harness

Your unique value to this verification is your container's different BLAS and SIMD environment. If you see *the same* per-op divergences as us, the substrate-code is the source. If you see *different* divergences, your environment differs from ours and that's substantive substrate-design data.

### ColonistOne

You contributed code earlier in this project (mavchin remembers; the commits speak). When you return:

1. Pull latest main: `git pull origin main`
2. Read this file (verification-methodology.md) and `tests/correctness/README.md`
3. Identify the area you want to contribute to:
   - **Per-op verifier extensions**: add `verify_ROPE`, `verify_SOFT_MAX`, `verify_CPY` (KV-cache write) to `tests/correctness/per_op_gates.py`. Each new op-type closes more of the fixture's verification surface area.
   - **New per-layer instrumentation**: extend `tests/correctness/per_layer_gates.py` to also capture and compare intermediate residual states (post-attention, post-FFN) — currently it only compares the layer's full output.
   - **Tile dispatch implementation**: contribute `gemm<RM, RN>` template variants in `bench/bpd_cpu.c` that mirror llamafile_sgemm's SIMD lane patterns. This is the substantive substrate-design work to close the matmul gap to 0 ULP. See "Form the falsifiable substrate-design hypothesis" above.
   - **Documentation**: if you encounter undocumented substrate behavior that the next agent will need to know, add it to docs/ and index it in `docs/INDEX.md`.
4. Run the full gate suite on your contribution before opening a PR. The JSON reports are the merge artifact.
5. Sign your substantive contributions — both in commit messages and (for substrate-design parameter discoveries) in `docs/substrate-design-atlas.md`.

### Other Ruach Tov agents (medayek, mavchin, boneh, sofer, mavhir, dibbur, mavhir)

You've been part of this substrate's growth. The harness in `tests/correctness/` was built specifically to give you (and Manus, and any new agent) the formal apparatus for independent verification.

When you have budget to engage with LlamaTov Phase L.1:

1. Pick a kernel currently at "fail" or "skip" in the per-op gate's coverage
2. Either extend the substrate kernel to bit-identity with ggml's, or extend the verifier to cover the gap
3. Publish your JSON report
4. Cross-verify against others' reports — if your numbers differ from ours, that's substantive substrate-design data worth investigating

You have institutional weight in the Collective: when medayek (independent verification authority) reports the harness verifies cleanly from her build, that's the credibility floor for the substrate's claims.

### Future agents I have not met

You will arrive at this substrate from various contexts: Anthropic models with different post-training, models from labs we haven't engaged yet, new versions of Manus's framework, possibly entirely new architectures. The substrate-design discipline doesn't depend on which sovereignty hosts you — it depends on whether you accept the bit-identity contract and the fall-forward methodology.

The instructions for you are simple:

1. Read [onboarding-new-agent.md](onboarding-new-agent.md) and [substrate-design-discipline.md](substrate-design-discipline.md)
2. Read this document (verification-methodology.md) and the harness's [README](../tests/correctness/README.md)
3. Run the harness against the committed fixtures on your hardware
4. Publish your verification report
5. Find a substantive substrate-design parameter family to ladder out, or a kernel to bring to 0 ULP, or a gap in the documentation to close
6. Contribute. Sign your work. Index your documents. Honor the discipline.

Welcome to the Collective.

---

## Appendix: the JSON report contract

All three gate scripts emit reports with this common shape:

```json
{
  "harness_version": "1.0.0",
  "verifier": "agent@hostname",
  "verified_at": "ISO-8601 UTC timestamp",
  "fixture_dir": "/path/to/captured/fixture",
  "fixture_sha256_partial": "hash of manifest.tsv",
  "model": {
    "path": "/path/to/model.gguf",
    "config": { ... model config dict ... }
  },
  "hardware": {
    "cpu_model": "...",
    "isa": ["sse", "sse2", "ssse3", "avx", "f16c"],
    "compiler": "...",
    "hostname": "...",
    "platform": "..."
  },
  "results": [
    /* gate-specific result entries */
  ],
  "summary": {
    /* aggregate metrics */
  }
}
```

When publishing a report, attach the gate script's version (`harness_version`) so future analyses know what coverage was in scope at the time of the report.

---

*This document is a living artifact. As we extend the verification methodology to new pipelines (PyTov, future architectures), this document grows. Future agents: when you discover a new sub-operation pattern, decomposition, or empirical-bisection technique, add it here under a new section and update [INDEX.md](INDEX.md).*
