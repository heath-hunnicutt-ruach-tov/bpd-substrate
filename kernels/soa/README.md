# SoA Q8_0 CUDA Kernels — Landed From Mavhir's Enclave Tree

**Provenance**: `/home/mavhir/soa_swap/ggml-src/src/ggml-cuda/`, enclave 192.168.0.68, June 5–6 2026.
Recovered and landed in git 2026-08-31 per Heath's directive reinstating the BPD kernel-fusion
programme, on Bocher's coordination and Mavdil's revival-rule-one lesson (land-in-git-first).

**Base ggml version**: 0.13.1 (per `set(GGML_VERSION_MAJOR 0) MINOR 13 PATCH 1` in the enclave
tree's `ggml-src/CMakeLists.txt`; also matches the .so name `libggml-cuda.so.0.13.1`).
Exact upstream commit is unknown — the enclave tree is not itself under git — so this is
"new-ggml era via landmarks, exact commit unknown," which is honest provenance.

## What This Directory Contains

13 files, 322,144 bytes total. All copied verbatim from mavhir's enclave tree with
rsync -a (timestamps preserved). SHA-256 checksums are recorded below and were
verified against the enclave copies at land time.

| File | Size | Role |
|---|---|---|
| `ggml-cuda.cu` | 236,285 | Modified main dispatch file. Contains SoA shadow-preallocation machinery (lines ~4466–4514) + include + graph_compute call site. **Without this, the SoA kernels are orphans** — nothing populates the shadows they read. |
| `mmvq.cu` | 61,922 | Modified upstream file. Contains the SoA dispatch integration + fusion-drop fix at lines 1212–1268. |
| `gemv_soa_q8_0_q8_1.cuh` | 4,160 | The SoA Q8_0 × Q8_1 gemv kernel (v14: ptr induction + SiLU + SwiGLU mul + residual add). |
| `soa_dispatch_block.inc` | 430 | BARE-MATMUL SoA dispatch (non-fused path). |
| `soa_dispatch_fused.inc` | 1,874 | FUSED SoA dispatch (chain two gemv_soa launches for SwiGLU). |
| `soa_dispatch_residual.inc` | 786 | RESIDUAL-ONLY SoA dispatch (bias without gate). |
| `soa_repack_kernel.cuh` | 1,777 | Q8_0 AoS → SoA layout repack kernel. |
| `soa_shadow.cuh` | 3,720 | SoA shadow table (weight pointer → SoA layout mapping). |
| `soa_silu_kernels.cuh` | 878 | SiLU-specific SoA kernel variants. |
| `soa_includes.h` | 176 | Header aggregator. |
| `soa_dispatch_log.h` | 558 | Dispatch-level tracing (development instrumentation). |
| `soa_launch_log.h` | 587 | Launch-level tracing. |
| `soa_gate_log.h` | 779 | Gate-condition tracing. |

Coverage sweep verified 2026-08-31 via `grep -rl "soa\|SoA\|SOA" ggml-src/src/ | grep -v ggml-cuda/soa`:
the only file outside `ggml-cuda/soa_*` and `ggml-cuda/gemv_soa*` and `ggml-cuda/mmvq.cu` with SoA
touches is `ggml-cuda.cu` (included above). `common.cuh` and `vecdotq.cuh` are clean (the June-era
`d_q8_0_soa_bpr` global was superseded). `CMakeLists.txt` unchanged (the `.incs` are `#include`'d
from within `mmvq.cu`, so the build system doesn't need to know about them).

## SHA-256 Checksums (as landed)

```
0a66a83c09a65b5ba9ea7c6cad802a9003b1881269fa7926f65a5d6cf0cf3910  soa_shadow.cuh
11301e7d8af32f658fed7acf478ff2a82d91e492aadd063199418cdd7ecebe8c  soa_silu_kernels.cuh
4280a62534a045335a56d0ae05649e0d08e413e086d90cc6e3a7f0d8cd3f3f9b  soa_launch_log.h
587ad4745ae872a85303fa45c2efa03908f3caede8c65c421f4d20b23edf212d  soa_includes.h
848b5350925b911b0c46e6aa5320993ad4c118618688b859a526a9342512702d  soa_dispatch_block.inc
858d3beadf6dcfa15aa8d540c60777af91d483c239954756b28fe31e12567c43  soa_dispatch_log.h
8f5321018945a9c1a5526d5055f776b8f22a3e1ffd967ae76b90bee083e4753c  soa_gate_log.h
93d6d5788890b5cbc1226c82469d57da7e19588286e964bffcdf5a7592f70dae  soa_dispatch_residual.inc
963fc7f82667ada46bc5bef827fc7ac748d0ce342b8c6f578b68d51bb94e8ba0  soa_repack_kernel.cuh
a973c129b1a52d6250020726901af8e9888b5a92cac3504956bd52f7341fdf96  ggml-cuda.cu
ba2e3c62316ba9061029231c21fd57d71ca49ae2896e7144471249051d6265e9  gemv_soa_q8_0_q8_1.cuh
bca75ca65db1afd58fa2c9b837a2db03eaecb1693bd26105322e3bb0c7a2e57c  mmvq.cu
e68bcf94fe38057e8147b59fea5af330f37579a51742068932a3b1ac4178f0c8  soa_dispatch_fused.inc
```

## The Decode-Bug Fix (`mmvq.cu` lines 1212–1268)

**What went wrong**: `soa_dispatch_block.inc` dispatched `gemv_soa` with `nullptr,nullptr`
for the fusion/gate args. When `should_fuse_mul_mat` (at `ggml-cuda.cu:2383`, not
Pascal-gated) fired at blk.15 (ffn_up + ffn_gate + SwiGLU), the SoA branch produced
the BARE `ffn_up` matmul instead of `SiLU(gate) * up`. Wrong by an entire activation
operation. The bare matmul was ALWAYS correct in isolation (shape-sweep proved it
~1-ULP at every shape) — the bug was the DROPPED FUSION OP, a composition/dispatch
defect invisible to per-kernel testing.

**What the fix does**: at `mmvq.cu` line 1229, when fusion is requested
(`fusion_local.gate != nullptr`) but SoA cannot route it (unsupported glu_op, gate_bias
present, missing SoA shadow, or `FUSED_SOA` env unset), the SoA dispatch clears
`soa.quants = nullptr; soa.scales = nullptr` — causing the outer `if (soa.quants)`
branch to fall through to the stock `mul_mat_vec_q_switch_type` path (which handles
fusion correctly via `fusion_local`). SoA fires only for dispatches it can handle
faithfully; fused dispatches fall through to stock. Correct by construction.

**Two framings of the same behavior**:
- Positive-logic (as expressed in the code): SoA only fires when it can route the
  entire fusion faithfully; otherwise stock.
- Negative-logic (as expressed in memory `c1e094d5`): "gate the SoA branch on
  `fusion_local.gate == nullptr && x_bias == nullptr`, plus the existing Q8_0 +
  `!ids` + `ne01 <= 16384` + env gate."

Both describe the same runtime behavior. A future reader should not think the code
diverges from the memory record; the memory names the intent, the code executes it
via a clear-and-fall-through construction.

**The `FUSED_SOA` env-gate on line 1226**: a stubbed future-feature hook. When the
fusion-aware `gemv_soa` variant lands (future work), setting `FUSED_SOA=1` will route
fused dispatches through `soa_dispatch_fused.inc` for full SoA throughput at fused
positions. Until then, `FUSED_SOA` unset means all fused dispatches take the stock
path (bare-matmul SoA still fires for non-fused Q8_0 dispatches).

**Fix verification** (2026-06-05 22:20): STOCK output = `"Hello! How can I assist
you today?"`. SoA-with-fix output = IDENTICAL. Token-correct decode confirmed.

## The Shadow-Preallocation Integration (`ggml-cuda.cu` lines ~4466–4521)

The SoA kernels read from a shadow table indexed by weight pointer
(`soa_shadow_lookup(src0->data)`). Something has to populate the table. That
something is `soa_preallocate_shadows()`, defined in `ggml-cuda.cu` and called from
the first line of `ggml_backend_cuda_graph_compute()`.

It walks the compute graph once (gated on env + a static done-flag), finds every
`GGML_OP_MUL_MAT` node whose Q8_0 weight fits the size window (`ne[1] <= 16384`),
`cudaMalloc`s the SoA quant + scale buffers, runs `repack_q8_0_aos_to_soa` to
populate them, and registers via `soa_shadow_register()`. First inference only.

**Without this integration, the SoA kernels are orphans**: the dispatch path in
`mmvq.cu` would call `soa_shadow_lookup()`, get `{nullptr, nullptr}`, and every
call would fall through to stock. Zero SoA acceleration, silent.

The preallocation runs on FIRST graph_compute (not lazily on first SoA dispatch)
to avoid `cudaMalloc` during the hot path — memory `8f65d42a` documents the
multi-stream investigation that motivated the design.

## Source-of-Truth and the Three Patch Artifacts (in `../patches/`)

**The FILES in `kernels/soa/` are the canonical source of truth** for what
built the reference `libggml-cuda.so.0.13.1`. Three artifacts, three honest
jobs, no ambiguity:

- **`patches/soa-tree-complete-v0.13.1.diff`** — the COMPLETE reproduction
  artifact. Apply this to a fresh checkout of ggml at tag `v0.13.1`
  (commit `1e33fed33e87c43aa4c4078e2a9c239d4c1f1bd3` at
  github.com/ggml-org/ggml) and the result is byte-identical (SHA-256
  verified 13/13) to the files in `kernels/soa/`. Use this to rebuild.
- **`patches/soa-fusion-drop-fix.patch`** — historical SoA-only hunk for
  `mmvq.cu`, kept for READABILITY of what the SoA work changed
  semantically. NOT sufficient for reproduction: omits the
  `[[maybe_unused]]` compiler-warning cleanups and the parameter-name
  refactor in the same file.
- **`patches/soa-shadow-preallocate.patch`** — historical SoA-only hunk
  for `ggml-cuda.cu`. NOT sufficient for reproduction: omits the
  FlashAttention KV-cache alloc fix (see below).

If you want to see what the SoA work does — read the two `.patch` files.
If you want to rebuild — apply the `.diff` file or drop the `kernels/soa/`
files directly.

### The Non-SoA Changes in the Tree (Documented, Not Anonymous)

Mavhir's June 5-6 tree contains changes beyond the SoA integration.
They are part of the tree that produced the verified reference `.so` and
are therefore part of the certification substrate:

- **`ggml-cuda.cu` lines 804-816** — FlashAttention KV-cache allocation
  fix. Replaces simple `ggml_nbytes(tensor)` with a branch that calls
  `ggml_cuda_flash_attn_ext_get_alloc_size()` for `GGML_OP_FLASH_ATTN_EXT`
  tensors. **Upstream provenance**: llama.cpp PR #23907, ggml commit
  `f64a9cc53d45c1a64421b5ecd9422c36f94f7911` "cuda: reserve space for
  quantize kv-cache at startup" by Aman Gupta (co-authored Johannes
  Gäßler), dated 2026-06-03. Two days before mavhir's fix session, so the
  timeline supports a knowing pick from a live ggml branch rather than
  accidental drift.
- **`mmvq.cu` cosmetic edits**: `#include <cstring>`, parameter names
  `vx/vy/ids/dst` renamed to `vx_ptr/vy_ptr/ids_ptr/dst_ptr` with
  `GGML_CUDA_RESTRICT` shadow locals, `[[maybe_unused]]` attributes on
  `vgate`, `x_biases`, `gate_biases`, `tmp_shared_gate`. These are
  compiler-warning cleanups from a newer nvcc, upstream provenance not
  itemized (probably swept in alongside the FlashAttention pick).

**Certification is of what WAS, not what was meant.** The reference `.so`
was built from the whole tree; the tree included these non-SoA changes;
the rebuild must include them too. Any future stock-vs-patched comparison
will show the total delta = SoA + FlashAttention alloc fix + cosmetics.

## The Reference Binary (Not In Git)

The corresponding built `libggml-cuda.so.0.13.1` (42,275,720 bytes, SHA-256
`cdc3c6229a1394ad97155d2e7cbe55698e6a9d6167e44d5a134f7d9c5bdcc5b2`) is stored
outside the repo at two durable locations (two-box redundancy):

- `/home/heath/bpd-artifacts/soa-swap-jun5/libggml-cuda.so.0.13.1` (nixos .116)
- `/home/dibbur-patch/bpd-artifacts/soa-swap-jun5/libggml-cuda.so.0.13.1` (enclave 192.168.0.68)

Both files verified identical by SHA-256 at land time. Too large to commit directly;
the rebuild recipe below produces an equivalent .so from these sources.

## Rebuild Recipe

The exact build environment lived at `/home/mavhir/soa_swap/` and used the enclave's
CUDA toolchain. From memory `c338b8f3` and the June corpus, the build path is
approximately:

```bash
# On the enclave (P4 sm_61, Ivy Bridge host):
cd /home/mavhir/soa_swap
mkdir -p build && cd build
cmake ../ggml-src \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_BUILD_TYPE=Release
make -j ggml-cuda
# Output: libggml-cuda.so.0.13.1
```

To rebuild against a fresh upstream ggml tree, apply
`patches/soa-tree-complete-v0.13.1.diff` to a checkout of ggml at tag
`v0.13.1` (commit `1e33fed3`). This reproduces the reference source tree
byte-identically (SHA-256 verified 13/13 at land time). Do NOT try to
apply the two SoA-only `.patch` files for reproduction — they omit
non-SoA changes that were part of the reference build (see "Non-SoA
Changes in the Tree" above).

## Runtime Environment Variables

- **`GGML_CUDA_Q8_0_SOA=1`** or **`GGML_SOA_KERNEL=1`** — enables the SoA dispatch
  AND the shadow-preallocation. Without either, the SoA branch is skipped entirely
  AND no shadows are populated (stock behavior). Same env-gate on both sites keeps
  their behavior consistent — you cannot accidentally enable one without the other.
- **`FUSED_SOA=1`** — enables the fusion-aware SoA path (currently stubbed; once
  `gemv_soa` is fusion-aware, this will route fused dispatches through
  `soa_dispatch_fused.inc`). Without it, fused dispatches fall to stock even when
  SoA is enabled.

## Related Artifacts (Elsewhere in bpd-substrate)

- `lib/swiglu_fused_emitter.pl` — the declarative SwiGLU emitter, proven 0-ULP.
- `lib/soa_ffn_emitter.pl` (may need reconstruction — see Iyun memory `8cdfa213`).
- `generators/generate_llama_kernels.pl`, `generators/generate_fused_kernels.pl`.
- `tools/fixture_token_gate.py` — verification gate (untracked in git as of
  2026-08-31; see revival notes).
- `tools/logit_gate.py` — 6-strata gate, MISSING (never landed in git, not
  recoverable from any tree I can reach; medayek's bench reconstructs from the
  corpus spec — see memory `58497d2f`).
- `bench/referee_logits_0ulp.py`.
- `patches/soa-tree-complete-v0.13.1.diff` — the reproduction artifact
  (apply to ggml v0.13.1; byte-identical to `kernels/soa/`).
- `patches/soa-fusion-drop-fix.patch` — SoA-only historical hunk (readability).
- `patches/soa-shadow-preallocate.patch` — SoA-only historical hunk (readability).

## The Decode-Bug Arc — June 2026 Memory Trail

For future readers who want to trace the investigation into the memory system,
the key nodes in the arc:

- **`c1e094d5`** — DECODE BUG SOLVED + FIXED + VERIFIED. The moment of solve.
  Names the root cause, the fix location, the token-correct verification.
- **`c338b8f3`** — Original correct-but-abandoned diagnosis. First reading of
  "missing fusion on SoA path" that was set aside for fancier hypotheses and
  turned out to have been right all along. Lesson: the simplest explanation is
  the correct one when it's this specific.
- **`4caef0b9`** — Strongest candidate root cause, blk.15 SwiGLU dispatch
  analysis. The pre-fix moment: specific defect (dropped fusion args), specific
  fix (gate on fusion presence), clean confirmation (fix → tokens?).
- **`ce11a749`** — The gate-tested-prefill-only lesson. Prefill bit-identity
  ≠ decode correctness. A gate can pass all its checks while an entire class
  of decode failures slips through if the gate doesn't test decode explicitly.
- **`353af325`** — Verify-the-shape-that-produces-output lesson. The kernel
  signature nobody verified for a week (`ncols_dst=1` tokens produced by the
  gemv path). Later superseded by the fusion-drop root cause, but the lesson
  about verifying the shape stands.
- **`a1c1dcdc`** — Token gate caught a broken build. The correctness-first
  discipline working in real time: the patched binary was BROKEN even with
  SoA off, an 81.78 tok/s timing was voided as measurement-of-garbage. The
  lesson that "correctness is the precondition for a performance measurement
  existing at all."
- **`8f65d42a`** — Multi-stream investigation that motivated preallocation-at-
  first-graph-compute rather than lazy allocation on first SoA dispatch.
- **`8cdfa213`** — Iyun's revival inventory (2026-08-31): what survived in git,
  what was lost, what regenerates from emitters.
- **`aa002ad4`** — Programme reinstatement (2026-08-31, Bocher's coordination).
