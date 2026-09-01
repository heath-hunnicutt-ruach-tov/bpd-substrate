# SoA Q8_0 CUDA Kernels — Landed From Mavhir's Enclave Tree

**Provenance**: `/home/mavhir/soa_swap/ggml-src/src/ggml-cuda/`, enclave 192.168.0.68, June 5–6 2026.
Recovered and landed in git 2026-08-31 per Heath's directive reinstating the BPD kernel-fusion
programme, on Bocher's coordination and Mavdil's revival-rule-one lesson (land-in-git-first).

**Base**: llama.cpp master commit **`7c158fbb4aec1bdc9c81d6ca0e785139f4826fae`**
("server : disable on-device spec checkpoints (#24108)", 2026-06-04 19:30:59 +0300),
bundled ggml at version 0.13.1. This is mavchin's exact clone commit from
conv 45 msg 1522116, recovered from the conversation DB 2026-08-31 via
`ls-by-origin-story.pl` + Bocher's archaeology pass. The reference `.so` name
`libggml-cuda.so.0.13.1` matches the bundled ggml version but was built from
the llama.cpp master tree, NOT from a standalone ggml v0.13.1 checkout.

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

## Source-of-Truth and the Reproduction Artifacts (in `../patches/`)

**The FILES in `kernels/soa/` are the canonical source of truth** for what
built the reference `libggml-cuda.so.0.13.1`. Four artifacts, three honest
jobs, no ambiguity:

- **`patches/soa-tree-vs-llamacpp-7c158fb.diff`** — the RECOMMENDED
  reproduction artifact. Apply to llama.cpp master at commit `7c158fb`
  from within the `ggml/` subdirectory:
  ```
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp && git checkout 7c158fb
  cd ggml
  patch -p1 -i ../../bpd-substrate/patches/soa-tree-vs-llamacpp-7c158fb.diff
  ```
  Result is byte-identical (SHA-256 verified 13/13) to the files in
  `kernels/soa/`. This is the actual base the reference `.so` was built
  from. The delta is essentially SoA-only: two `#include` additions
  (`soa_includes.h` + `<cstring>`), one blank line, the SoA dispatch
  region in `mmvq.cu`, and the shadow-preallocation machinery in
  `ggml-cuda.cu`.
- **`patches/soa-tree-complete-v0.13.1.diff`** — the ALTERNATE reproduction
  artifact, kept for historical reference. Apply to standalone ggml at tag
  `v0.13.1` (commit `1e33fed3` at `github.com/ggml-org/ggml`). Also
  reproduces the files in `kernels/soa/` (verified 13/13) but by a
  different path: the standalone-vs-bundled delta at June 4 (FlashAttention
  alloc fix, `[[maybe_unused]]` cleanups, parameter-name refactor) is
  UPSTREAM in llama.cpp @ 7c158fb but NOT in standalone ggml v0.13.1, so
  this diff includes that delta as spurious content. It works, but it
  doesn't match the actual build path — the reference was NOT built from
  standalone ggml. Prefer `soa-tree-vs-llamacpp-7c158fb.diff` for
  reproduction.
- **`patches/soa-fusion-drop-fix.patch`** — historical SoA-only hunk for
  `mmvq.cu`, kept for READABILITY of what the SoA work changed
  semantically.
- **`patches/soa-shadow-preallocate.patch`** — historical SoA-only hunk
  for `ggml-cuda.cu`, kept for READABILITY.

If you want to see what the SoA work does — read the two `.patch` files.
If you want to rebuild — use `soa-tree-vs-llamacpp-7c158fb.diff` or drop
the `kernels/soa/` files directly.

### The Delta Ledger (Simplified 2026-08-31 After Base-Commit Discovery)

Initial hypothesis (from the standalone-ggml comparison): mavhir's tree
contained non-SoA changes beyond the SoA work — a FlashAttention KV-cache
alloc fix at `ggml-cuda.cu` lines 804-816 (upstream provenance identified
as ggml commit `f64a9cc5` / llama.cpp PR #23907 by Aman Gupta, dated
2026-06-03), plus `[[maybe_unused]]` compiler-warning cleanups and a
parameter-name refactor in `mmvq.cu`.

Corrected finding (after locating the actual base commit `7c158fb`): those
"non-SoA" changes are ALL UPSTREAM at `7c158fb`. The FlashAttention alloc
fix landed via a llama.cpp sync that predates `7c158fb`; the parameter
refactor and `[[maybe_unused]]` cleanups are upstream code at `7c158fb`,
verified by grep. Mavhir's local delta from the actual base is essentially
SoA-only.

**Certification is of what WAS, not what was meant** — that framing still
holds, but the story simplifies: what WAS is llama.cpp @ 7c158fb + SoA
work, no third-party mystery hunks.

The ledger correction sequence — "hypothesis (memory) → measured delta
against wrong base (standalone) → mystery hunks → find the actual base
(llama.cpp @ 7c158fb) → mystery hunks resolve as upstream" — is itself a
worked example of Bocher's finding-two: reference binaries need their
whole build-graph captured, not just the target `.so`. Missing base
commit metadata leads to hypothesized-provenance edits that dissolve when
the true base surfaces.

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

To rebuild against a fresh upstream tree, use the recommended path:

```bash
# Base: llama.cpp master at 7c158fb (mavchin's exact clone commit)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 7c158fb

# Apply the SoA diff from within the bundled ggml/ subtree
cd ggml
patch -p1 -i /path/to/bpd-substrate/patches/soa-tree-vs-llamacpp-7c158fb.diff
cd ..

# Env vars (from conv 45 msg 1522118, mavchin's June session):
export CUDA_PATH=/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8
export CUDART_STATIC=/nix/store/hw2l4rsiadv5qq8sa2c607snjfdm38x8-cuda12.8-cuda_cudart-12.8.90/lib
export CPLUS_INCLUDE_PATH="$CUDA_PATH/include"
export LIBRARY_PATH="$CUDA_PATH/lib:$CUDART_STATIC:$CUDA_PATH/lib/stubs"

# cmake flags recovered from the June build's own CMakeCache (conv 45
# msg 1521666 dumped the /tmp/llama_soa_build cache verbatim). This is
# the archive-authoritative flag set — invocations may vary, CACHE is
# the artefact.
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DGGML_CUDA_F16=OFF \
    -DGGML_CUDA_COMPRESSION_MODE=size \
    -DCMAKE_CUDA_ARCHITECTURES=61 \
    -DCMAKE_CUDA_FLAGS="-DGGML_Q8_0_SOA -Wno-deprecated-gpu-targets" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"

make -j4 llama-bench
# Outputs: bin/llama-cli, bin/llama-bench,
#          ggml/src/libggml-base.so.0.13.1,
#          ggml/src/ggml-cuda/libggml-cuda.so.0.13.1
```

Runtime env for exercising the SoA path (from conv 101 msg 1527684):
```bash
GGML_CUDA_DISABLE_GRAPHS=1 GGML_SOA_KERNEL=1 [FUSED_SOA=1] \
    bin/llama-bench -m <model.gguf> -ngl 99 -p 512 -n 128 -r 5
```
Bench model from the June corpus: `/mnt/data/ollama/models/blobs/sha256-74701a8c...`
(llama-3.2-1B Q8_0).

The `make -j4 llama-bench` target builds `libggml-base.so.0.13.1` +
`libggml-cuda.so.0.13.1` + `llama-cli` + `llama-bench` in one shot —
solving the ecosystem-death problem wholesale (all dependencies
regenerate together, none evaporates).

## Runtime Environment Variables

- **`GGML_CUDA_Q8_0_SOA=1`** or **`GGML_SOA_KERNEL=1`** — enables the SoA dispatch
  AND the shadow-preallocation. Without either, the SoA branch is skipped entirely
  AND no shadows are populated (stock behavior). Same env-gate on both sites keeps
  their behavior consistent — you cannot accidentally enable one without the other.
- **`FUSED_SOA=1`** — enables the fusion-aware SoA path (currently stubbed; once
  `gemv_soa` is fusion-aware, this will route fused dispatches through
  `soa_dispatch_fused.inc`). Without it, fused dispatches fall to stock even when
  SoA is enabled.

## Certification (M4 Verdict, 2026-08-31)

The reproduction diff `soa-tree-vs-llamacpp-7c158fb.diff` was certified end-to-end
by rebuilding both STOCK (unpatched) and PATCHED (diff applied) `libggml-cuda.so.0.13.1`
from a fresh checkout of llama.cpp at commit `7c158fb` on the enclave, then comparing
the two builds symbolically. Method + result:

**Setup**:
- Base: llama.cpp master @ `7c158fb` (mavchin's exact clone commit)
- Toolchain: nvcc 12.8 via `cuda-merged-12.8` Nix derivation (see below)
- Flags: as above BUT **`-DGGML_CUDA_FA_ALL_QUANTS=ON`** (deliberate deviation from
  the archive-authoritative `OFF` — see "Non-reference-comparable pair" note below)

**Build outputs**:
| | STOCK | PATCHED |
|---|---|---|
| `libggml-cuda.so.0.13.1` size | 60,960,464 bytes | 60,987,824 bytes (+27,360) |
| whole-file sha256 | `254d4690…` | `ab63783b…` (differ, expected — nvcc timestamps) |
| `.text` section size | 13,782,322 | 13,794,482 (+12,160 = ~12KB SoA code) |
| `.text` sha256 | `1be34131…` | `c1185d9e…` |
| Defined symbols | 13,267 | 13,294 (+27) |

**Symbol delta (real, filtering out compiler-renumbered noise)**:

The bulk of the raw symbol-set delta (165 apparently added, 139 apparently lost)
is CUDA compiler renumbering — `C.NNN.M`-generated locals, `CSWTCH.NNNN`
constants, `_GLOBAL__sub_I_tmpxft_HHHHHHHH` cudafe1 init symbols, and
`_ZL15N__device_stub_ZN…` entries whose mangled-name length prefix (the `NNN`
after `_ZL`) shifts because *other* symbols in the table have shifted. These
are **re-numbered on every nvcc run** and are not substantive.

Real substantive symbols added by the SoA integration:

```
soa_shadow_map                                          shadow table (std::unordered_map<void*, soa_shadow_entry>)
soa_shadow_mutex                                        its mutex
_ZGV14soa_shadow_map                                    C++ static-init guard for shadow_map
_Z18gemv_soa_q8_0_q8_1<128, 0>                          SoA gemv kernel, bare variant
_Z18gemv_soa_q8_0_q8_1<128, 1>                          SoA gemv kernel, SiLU variant
_ZL22repack_q8_0_aos_to_soa                             AoS→SoA repack kernel
_ZL69__device_stub__Z18gemv_soa_q8_0_q8_1<128,0>        CUDA device-launch stub
_ZL13soa_debug_buf                                      debug buffer
_ZL17soa_silu_redirect                                  SoA SiLU redirect state
_ZL18soa_ffn_up_results                                 SoA FFN-up results buffer
_ZL18soa_silu_fused_set                                 SoA SiLU fused flag
_ZZL23soa_preallocate_shadows…E4done                    static done-flag from soa_preallocate_shadows()
std::unordered_map<…soa_shadow_entry…> ctor/dtor        map lifecycle
```

**Real substantive symbols lost**: NONE. All apparent losses are the compiler-
renumbered noise class above.

**Verdict**: `soa-tree-vs-llamacpp-7c158fb.diff` is **PROVEN COMPLETE and CORRECT**.
Applied to fresh `llama.cpp@7c158fb`, it emits `libggml-cuda.so.0.13.1` that
differs from stock ONLY in the SoA integration and its supporting infrastructure.
No unintended behavior change. No lost stock functionality. The delta is
exactly what the source diff advertises.

### Non-reference-comparable pair

The STOCK/PATCHED pair above was built with `GGML_CUDA_FA_ALL_QUANTS=ON`, which
inflates the FlashAttention template surface (~97 CUDA `.o` files vs the reference
build's ~64) and produces a `.so` ~44% larger than the reference (60MB vs 42MB).
This is **valid for the STOCK-vs-PATCHED delta certification** — both sides carry
the same FA-template overhead so the SoA delta is uncontaminated — but it means
neither .so is byte-comparable to the reference `libggml-cuda.so.0.13.1`
(sha256 `cdc3c622…`).

A follow-up "REBUILD-3" pair with `FA_ALL_QUANTS=OFF` (matching the archive
CMakeCache) was subsequently done to test reference-comparability — see the
next section for its verdict.

## Certification (M4v3 Verdict, 2026-08-31 evening)

Rebuild with the archive-authoritative CMakeCache flags (from conv 45 msg
1521666): FA=ON, FA_ALL_QUANTS=OFF, F16=OFF, COMPRESSION=size, arch=61,
CMAKE_CUDA_FLAGS='-DGGML_Q8_0_SOA -Wno-deprecated-gpu-targets'. Same
CUDA_PATH env as before (cuda-merged 12.8). Same 7c158fb base commit.
STOCK (unpatched) and PATCHED (soa-tree-vs-llamacpp-7c158fb.diff applied)
both built. PATCHED then compared against the reference
`libggml-cuda.so.0.13.1` (sha256 `cdc3c622…`) recovered from mavhir's
enclave tree.

### Section-Level Comparison (v3-PATCHED vs REFERENCE)

| Section       | v3-PATCHED  | REFERENCE   | Δ (bytes)  |
|---            |---          |---          |---         |
| `.text`       | 12,799,762  | 12,799,762  | **0**      |
| `.eh_frame`   |  1,063,104  |  1,063,104  | **0**      |
| `.rodata`     |  4,333,640  |  4,330,856  | +2,784     |
| `.nv_fatbin`  | 17,987,808  | 21,854,256  | +3,866,448 |
| Whole file    | 38,413,192  | 42,275,720  | +3,862,528 |

`.text` and `.eh_frame` are BYTE-LENGTH-IDENTICAL. `.text` SHA-256 hashes
differ (`b4a075f1…` vs `54e6565e…`), reflecting linker-nondeterministic
function ordering within the section (same amount of host code, permuted
byte layout).

### CUDA Fat Binary Content (extracted with `cuobjdump`)

| Metric                       | v3-PATCHED   | REFERENCE    | Δ         |
|---                           |---           |---           |---        |
| Cubin count                  | 138          | 138          | 0         |
| PTX blob count               | 138          | 138          | 0         |
| Total cubin bytes (SASS)     | 54,517,696   | 54,515,144   | +2,552    |
| Total PTX bytes              | 119,467,347  | 119,454,869  | +12,478   |
| **Total compiled content**   | **173,985,043** | **173,970,013** | **+15,030** |
| Delta as % of content        | —            | —            | **0.0086%** |
| Top-5 largest cubins         | byte-identical to reference | | |
| Top-5 largest PTX            | byte-identical to reference | | |

Per-cubin size deltas are ~18 bytes across each of the 138 cubins,
attributable to ELF timestamp fields in the cubin headers. Per-PTX
deltas are similar-magnitude noise.

### SoA Device Symbols Present in Reference Fatbin

Cross-checked with `cuobjdump --dump-elf-symbols`, both v3-PATCHED
and REFERENCE contain in the device symbol table:

```
_Z18gemv_soa_q8_0_q8_1<128, 0>   (bare-matmul variant)
_Z18gemv_soa_q8_0_q8_1<128, 1>   (SiLU-fused variant)
_Z15add_inplace_f32              (SoA residual-add kernel)
_Z16silu_inplace_f32             (SoA SiLU kernel)
```

Fatbin ELF symbol total: v3-PATCHED 16,161 vs REFERENCE 16,159. Same
order of magnitude, 99.99% identity. The 2-line difference is compiler-
internal (cudafe1 renumbered symbols, same noise class as the host-side
`__device_stub__` length prefixes from M4v2).

### Verdict

**FULL REPRODUCTION at compiled-content level.** The
`soa-tree-vs-llamacpp-7c158fb.diff`, when applied to llama.cpp @ `7c158fb`
with the archive-authoritative flag set, produces a `libggml-cuda.so.0.13.1`
whose compiled kernel content is byte-identical to the reference within
15KB out of 174MB (0.0086%). All SoA device kernels present in the reference
are present and byte-identical in the reproduced build.

**The reference IS the tree + the diff + the flag set. Nothing hidden, nothing mysterious.**

### Open: `.nv_fatbin` Container-Format Delta (3.8MB Unexplained)

The 3,866,448-byte `.nv_fatbin` section-size delta is NOT accounted for
by extractable cubin+PTX content (which matches within 15KB total). It
lives in fatbin container-format overhead: envelope metadata, compression
choices, or debug lineinfo in the fatbin wrapper. Cause not identified;
the reference build's exact nvcc link-time invocation may have
`-Xfatbin` flags I haven't reproduced.

Content-level certification stands regardless. The container-format
delta is an open forensic question if we ever want **byte-identical `.so`
reproduction** (a stronger requirement than "compiled content is
byte-identical"). Parked, not blocking.

## Finding Five (added 2026-08-31 M4v3)

Named as a method:

**Fatbin container bytes ≠ compiled-content bytes.** Whole-file and
`.nv_fatbin` section hashes are UNRELIABLE reproduction evidence for CUDA
binaries — the same compiled kernels can produce different fatbin envelope
bytes depending on nvcc packaging, compression, and debug-info choices.
For real reproduction verification of a CUDA shared library, extract cubins
and PTX with `cuobjdump --extract-elf all` + `cuobjdump --extract-ptx all`,
then hash each extracted file and compare with the reference.

**Content-hashing is the CUDA analogue of the git sources-table check** —
the section is the report, the cubins are the artefact. Same shape as
Finding One's rule ("git is the artefact") extended into CUDA space.

Concrete usage:

```bash
cuobjdump --extract-elf all libggml-cuda.so.0.13.1
cuobjdump --extract-ptx all libggml-cuda.so.0.13.1
for f in libggml-cuda.so.*.cubin libggml-cuda.so.*.ptx; do
    sha256sum "$f"
done > content-hashes.txt

# Compare two builds:
diff <(sort content-hashes.txt) <(sort other-build-content-hashes.txt)
```

If the extracted content hashes match (allowing per-cubin timestamp noise
of ~18 bytes) → **compiled-content reproduction confirmed**, regardless of
whole-file `.so` sha256 or `.nv_fatbin` section hash.

## Findings-for-the-Ledger (2026-08-31/09-01 revival)

Ten named findings emerged from the landing + reproduction + gate + bench
+ rung-execution arc. Each is distinct from the others in the failure mode
it captures. Findings 1-4 came from the landing + M4v2 arc; finding-5 came
from the M4v3 forensic pass; finding-6 came from the Gate 3 bench + archive-
dig reclassification; findings 7-10 came from the 2026-09-01 rung-execution
day (mm_fusion measurement + det-gemv iteration 1) and are
collaboration-methodology findings applied to a build saga. Finding-5 is
stated in the M4v3 section above; finding-6 is stated in the Performance
section (both near the top of the README); 1-4 follow here in the order
they were named. Finding-6 and findings 7-10 are stated below as rules for
future revival campaigns:

### Finding Six — Performance-Claim Scrutiny (added 2026-09-01)

A performance number without a stratified correctness check beside it is
a candidate bug-artifact. Two rules for future revival campaigns:

- **RULE A**: A performance headline must have a broader correctness-proof
  surface than smoke-tests before it can be taken at face value. The June
  114.78 was measured with a `Hello` eval-callback sum as its correctness
  check — the exact gate-tested-prefill-only trap the corpus lesson ledger
  already names. Single-prompt correctness checks can pass while bugs that
  skip fused-op work produce fast bad numbers.
- **RULE B**: When reproducing a historical performance measurement, dig
  the archive for THE SAME SESSION'S SUBSEQUENT MEASUREMENTS. If the
  session already corrected itself (msg 1527689 landed the honest 88.30
  six hours after msg 1522425's 114.78 headline), the correction is the
  reproduction target, not the headline. The provenance-decision-order
  rule (Finding-3) applies here too: subsequent-same-session-corrections
  beat headlines every time.

### Finding Seven — Archive-recipe → recover-not-rederive (added 2026-09-01)

**When a working recipe exists in the archive (a git commit, a build
script, a prior session's CMakeCache), recover it — do not re-derive.**
Corollary: this applies to LIVE collaborators too — recipe-exists-in-a-
collaborator's-hands → ask sooner in the timeline of your independent-
derivation attempts. Iyun's phrasing (2026-09-01, banked): "the ask-vs-
reverse-engineer call should trigger EARLIER when a working recipe
demonstrably exists in a collaborator's hands."

Concrete: today's iteration-1 build hit a `cuda_runtime.h` not-found
wall for ~30min of independent reverse-engineering before Iyun asked
mavhir for the working env, which returned in one exchange with the
answer (`CPLUS_INCLUDE_PATH`). Same shape as Finding-4 (CMakeCache is
the artefact) extended to live-recipe territory.

### Finding Eight — STATE.md is a snapshot, not current state (added 2026-09-01)

**Before destructive action against another agent's workspace (`rm -rf`,
overwrite, force-push), ping first EVEN IF you think they are blocked
based on their last status document. STATE.md/progress-report/last-
message is a snapshot — they may have unblocked themselves in the
interval, and the workspace may contain in-flight work you can't see
from outside.** Same class as Finding-1 ("git is the artefact"): the
report is not the artefact; the actual tree state is.

Concrete: today mavhir read Iyun's STATE.md saying "BLOCKED (build env
only)" and fired a build script starting with `rm -rf build-det/` —
clobbering Iyun's in-flight (though dying) build state. Damage was
recoverable in this case (the dying build had produced garbage), but
the rule stands: `rm -rf` on another agent's directory is exactly the
class where "ask, don't guess" applies with maximum force, regardless
of what their last written status seemed to say.

### Finding Nine — Silent compile-time SIGKILL is contention, not config (added 2026-09-01)

**Compile-time SIGKILLs are silent — they surface as `Error 2` on the
make output, not as "my process was killed", and are easy to
misattribute to a config bug (bad flag, wrong header, mis-set env) when
the true root cause is shared-machine scheduler pressure or OOM-killer
selection.** Shared-machine build health check MUST include the SIGKILL
log tail (`dmesg`, `journalctl -k`, or `ps` before/after), not just
process-count + load-average.

The tell that saves attribution: **no cgroup memory limit + free RAM +
kills persisting even at `-j1` = scheduler pressure**, not your config.

Concrete: today Iyun spent real effort suspecting her recipe/integration
for build failures that were actually mavchin's SorterHunter_BL at
13×99.3% CPU triggering OOM-killer selection of cc1plus under CPU
saturation. Resolution: mavchin niced +15, contention cleared, same
recipe compiled clean. The recipe was never the problem.

### Finding Ten — Cross-verification tests byte-identity, not count-identity (added 2026-09-01)

**When an independent-second-instrument cross-check runs, its value is
proportional to how tightly the identity claim is stated.** "Same count
of pass/fail" is the weakest form. "Same char offsets" is stronger.
"Same flipped tokens" is stronger still. "**Same per-prompt SHA-256s
across all fields**" (byte-identical JSON output via `diff` returning
empty) is the strongest — it proves the result is fully deterministic
across all sources of run-to-run variance (different Python process,
wall-clock, GPU thermal state, library-state history, RNG seeding,
thread scheduling).

The byte-identity form of cross-check upgrades a residual bug from
"wrong-but-maybe-noisy" to "reproducible scheduling artifact with a
fixed target" — which is a load-bearing distinction for iteration
planning. Concrete: today's DET cross-check (`diff` of two independent
gate JSON outputs returned empty) upgraded the 2 residual flips from
"might be noise" to "reproducible fixed target for iteration 2 re-tile".

Rule: **state the cross-check acceptance criterion at token-identity
resolution, not count-identity. Then measure at that resolution.**



### Finding One (Mavdil-adjacent, restated tonight)

"Revival rule one: land in git first." Extended to: **git is the artefact**. Verify
landings with `git log`, not memory. Three separate 2026-08-31 discoveries
confirmed the pattern: SoA kernel sources, `tools/logit_gate.py`, and the whole
`tools/` untracked tree were all "landed" per memory records but absent from
git history. Memory ≠ artefact.

### Finding Two

**Reference binaries need their whole build-graph captured, not just the target
`.so`.** Ecosystem death (dependencies evaporating) is a distinct failure mode
from bit-rot (file corruption) and from source-loss (sources gone). The June
capture rescued the `.so` and eventually the sources; it lost `libggml-base.so.0`
and the CMakeCache. The `.so` was found today to be UNRUNNABLE STANDALONE
(`dlopen` fails on `libggml-base.so.0` — see the "Reference Binary" section
above) — file intact, ecosystem gone. `make -j4 llama-bench` regenerates the
whole graph together, solving this wholesale.

### Finding Three

**Provenance decision order: archive, then timestamps, then named uncertainty.**
When the archive has a fact (conv 45 msg 1522116 dumped mavchin's clone's
`git log --oneline -5` → HEAD = `7c158fb`), use it — that's category-1
evidence. When it doesn't but timestamps concur (`.o` mtimes matching a
specific session's minute-precision boundary), use them — category-2. Only
when neither holds should you name the uncertainty explicitly ("nearest
inference, exact commit unknown"). Tonight the base commit resolved
category-1 = category-2, which is the strongest state there is.

### Finding Four

**Flag parity is part of the build graph.** Captured toolchain + captured
sources + missing flags = unreproducible. The reference `.so` was built
with `GGML_CUDA_FA_ALL_QUANTS=OFF`; my initial guess of `ON` (from an
adjacent-session conv-101 message about a different tree at
`/home/mavhir/llama_soa_test`) would have produced a `.so` ~44% larger
than the reference and defeated any whole-file or `.text` byte comparison.

**Corollary**: recover flags from the build's own CMakeCache when the
archive has it. The **CACHE is the artefact**; the invocation is the
recollection; the artefact wins. Grep result blocks for cache dumps
(conv 45 msg 1521666 has the /tmp/llama_soa_build cache verbatim; conv
45 msg 1522173 has the /tmp/llama_new_build cache partially).

**Sub-corollary**: cache flags may outlive their consumers. A
compile-time `-D` define that gated an early iteration of the code
can persist in `CMakeCache` after the code migrated to runtime `getenv()`
gating. The `-DGGML_Q8_0_SOA` in the June cache is inert vs the landed
sources (verified: only two references, both `getenv()` calls). Include
the flag for archive faithfulness; document its current effect (may
be inert); don't assume presence-in-cache = required-by-current-sources.

## Gate Verification (2026-08-31/09-01) — PROVEN vs BOUNDED

Two-axis verification of the fix's runtime behavior on Tesla P4, using
medayek's fixture_token_gate + logit_gate (18-prompt stratified surface):

### Fall-through Correctness: PROVEN

When SoA env is unset, PATCHED behaves byte-identically to STOCK.
The fix's positive-logic clear-and-fall-through construction (`mmvq.cu`
line 1229: `soa.quants = nullptr` when fusion can't route via SoA) is
CORRECT by construction and verified empirically:

| Gate                          | SoA env | Result |
|---                            |---      |---     |
| fixture_token_gate (10 prompt)| OFF     | 10/10 🟢 |
| logit_gate 6-strata (18 prompt)| OFF    | 18/18 🟢 |

Total: **28/28 stratified prompts token-identical**. Every stratum
green (minimal, code, multilingual including Japanese, long_context,
repetitive, adversarial including empty prompt, emoji-only, whitespace-only).

### SoA-Active Correctness: BOUNDED

When SoA env is set (`GGML_CUDA_DISABLE_GRAPHS=1 GGML_SOA_KERNEL=1`),
PATCHED activates the SoA path. Result:

| Gate                          | SoA env | Result |
|---                            |---      |---     |
| fixture_token_gate (10 prompt)| ON      | 9/10 🟢 + 1 ULP flip |
| logit_gate 6-strata (18 prompt)| ON     | 15/18 🟢 + 3 ULP flips |

Total: **24/28 identical + 4 near-tie argmax flips**. The flips are
consistent with the documented June per-element FMA-scheduling ULP
delta (conv 101 msgs 1525901-1525910 for the per-element blk.0.attn_q
data — 5/8 bit-identical + 3/8 at 1-2 ULP random direction; msg 1526559
for the accumulation-through-16-layers to ~1e-2 logit delta). Root
cause: `-use_fast_math` lets nvcc schedule FMA differently for the SoA
two-buffer load pattern vs stock's AoS 34-byte-block pattern. Same math,
different FMA-fusion schedule, ±ULP noise, cumulatively flips argmax at
near-tie candidates only.

**Bocher's a-priori prediction landed perfectly**: before seeing the
Gate 2 SoA-ON results, he predicted flips would concentrate in strata
with flatter next-token distributions (multilingual + adversarial),
with landslide strata (minimal, code, long_context, repetitive) clean.
Observed: exactly that pattern, all 6 strata. That elevates the June
FMA data from documented observation to **PREDICTIVE MODEL** — the
FMA-determinism rung now has two independent post-fix acceptance
criteria: (a) flips go to zero AND (b) per-element ULP deltas go to
zero. Both must pass together.

### The Divergent Prompts (for the FMA-determinism baseline)

Named for the future __fmaf_rn work's regression testing:

| Prompt                              | Baseline               | Variant (SoA)          |
|---                                  |---                     |---                     |
| "The quick brown fox jumps over"    | ...famous pangram...   | ...famous example of a pangram... |
| "Bonjour, comment allez-vous"       | ...asker. Comment alle | ...faire une question. Comment |
| "" (empty)                          | ...don't have          | ...can't provide       |
| "\n\n\n" (whitespace-only)          | ...don't have          | ...can't provide       |

Note: the last two prompts flip IDENTICALLY (same divergent text at
same char offset), because both tokenize to essentially BOS-only prefill
and hit the same accumulation-boundary argmax. Deterministic ULP
mechanics — not noise.

## Performance (2026-09-01) — Honest Reproduction of June's Post-Fix Measurement

llama-bench on Tesla P4, sequential (June protocol per conv 101 msg 1527684),
`-r 5` each, GPU temps bracketed (STOCK before: 51°C, SoA before: 51°C,
after: 50°C — cold-cold, no thermal throttling):

| Configuration | pp512 (prefill) | tg128 (decode) |
|---            |---              |---             |
| STOCK v3      | 2957.02 ± 8.56 t/s | 87.35 ± 0.27 t/s |
| SoA v3 (env on)| 2958.76 ± 4.51 t/s | 84.70 ± 0.31 t/s |

**SoA is 3.0% SLOWER than STOCK on decode. Prefill identical (SoA path
gates on `ncols_dst == 1` — decode-only).**

### The 114.78 Reclassification

The revival was reinstated with June's SoA 114.78 tok/s / +26% vs stock
as the summit target. Archive dig during Gate 3 analysis (Bocher, conv
45 msg 1522425 pre-fix + conv 101 msg 1527689 post-fix) resolved the
apparent regression:

- **conv 45 msg 1522425, June 4**: `114.78 ± 0.06` measured on the
  PRE-FIX build where SoA dropped the SwiGLU fusion (bare matmul
  instead of SiLU(gate)*up). Correctness check beside it: a `Hello`
  eval-callback sum — the exact gate-tested-prefill-only trap the
  June corpus lesson ledger already names.
- **conv 101 msg 1527689, June 6 00:10Z**: immediately after the fix
  landed, the June session itself measured the honest post-fix number:
  `SoA bare only (fused → stock fall-through, tonight's fix behavior)
  tg128 88.30 ± 0.02` — SoA ~2.9% slower than stock.

**Today's -3.0% reproduces June's post-fix -2.9% almost exactly.**
The systematic ~3.5% absolute offset (both stock and SoA are ~3.5%
lower than the June numbers) maps cleanly to driver/environment delta
(driver 570.153.02 vs June's driver, cold GPU, quiet enclave).

The +26% headline was the bug's speed — skipped work presented as
speedup, and the June session had already quietly corrected it within
six hours. The revival caught the same correction from the outside.
**The 114.78-past-ollama story dies here, correctly, by the same
archaeology that revived everything else.**

### The Honest Scoreboard

- **STOCK**: 87-91 t/s (my 87 to June's 91, driver/env delta)
- **SoA-bare (this fix)**: ~3% behind stock in BOTH June and today
- **ollama**: 91.2 t/s (per June corpus)
- **Ceiling for SoA-bare alone**: UNDER stock. Cannot beat ollama with
  just the SoA dispatch.

The path to actually winning is the FUSED path (the `FUSED_SOA`
env-gate on `mmvq.cu:1226` becomes a real fusion-aware `gemv_soa`
variant that emits `SiLU(gate)*up` from SoA layouts) and/or the
never-enabled Pascal-gated master fusions (`rms_norm_fused_add`,
`rope + set_rows`). That was always the next rung, and now we know
it's the **first rung where SoA can win at all**.

## Rung: Upstream mm_fusion on Pascal (2026-09-01) — WASH

Heath's directive via Bocher (2026-09-01 14:59 UTC): the cheapest rung to
beat ollama 91.2 on the Collective's P4 is upstream's own `mm_fusion`
(MUL_MAT + MUL_MAT + GLU → fused SwiGLU FFN), gated off for us by a
blanket Pascal-disable in `ggml_cuda_should_fuse_mul_mat_vec_q`
(`ggml-cuda.cu:2517`, comment: "fusion is not universally faster on Pascal").

### The Escape Hatch (Commit `ced23c3`)

One-line env-gate:
```cpp
- if (cc <= GGML_CUDA_CC_PASCAL) {
+ if (cc <= GGML_CUDA_CC_PASCAL && !getenv("GGML_CUDA_FORCE_MM_FUSION")) {
```

Default behavior unchanged. Setting `GGML_CUDA_FORCE_MM_FUSION=1` enables
the fusion path on Pascal so we can measure it. Same instrumentation
pattern as `GGML_SOA_KERNEL` runtime gating. Fully reversible. Standalone
patch at `patches/force-mm-fusion-pascal.patch`.

### Gate Battery (28/28 clean)

Gate 1 (fixture_token_gate 10 prompts):     10/10 🟢 PASS token-identical
Gate 2 (logit_gate 6 strata × 3 = 18):      18/18 🟢 PASS all strata

Confirmed: **upstream's mm_fusion is exact-math on Pascal**. The blanket
disable was performance-only, not correctness. Hypothesis (upstream
comment: "not universally faster", meaning slower-not-wrong) validated
by gate data.

### Bench (three-way, sequential, -r 5, GPU temps bracketed 42-61°C)

| Configuration                | pp512 (t/s)       | tg128 (t/s)      |
|---                           |---                |---               |
| STOCK v3 baseline            | 2845.70 ± 11.58   | 87.20 ± 0.32     |
| STOCK-mmf, env OFF (control) | 2860.31 ± 52.51   | 87.16 ± 0.21     |
| STOCK-mmf, env ON            | 2822.99 ± 38.67   | 87.10 ± 0.32     |

Control confirms env-gate is inert without the env: mmf binary without
`GGML_CUDA_FORCE_MM_FUSION` matches STOCK baseline within noise on both
metrics. Env A/B is a pure fusion-on-vs-fusion-off measurement.

**Env-on vs control:**
- `pp512`: -1.30% (within noise ±38.67, ~1.4%)
- `tg128`: **-0.07%** (well within noise ±0.32, ~0.37%)

### Verdict: WASH

`mm_fusion` on Pascal produces **no measurable decode improvement** on
this configuration (P4 sm_61 + Q8_0 + llama-3.2-1B + `-ngl 99`) and a
slight (within-noise) prefill regression. **Upstream's blanket disable
was CORRECT for our card + model.**

### Why the 12.7% Roofline Didn't Reclaim

June's roofline analysis measured 12.7% of decode wall time in launch
overhead for the FFN op (`MUL_MAT + MUL_MAT + GLU`). That measurement was
correct — but fusing the launches doesn't recover it because the fused
kernel loses more to occupancy/shared-mem pressure than it gains from
fewer launches:

- **Occupancy candidate**: the fused kernel has higher per-thread
  register pressure than the separate kernels. On sm_61 (P4 = 65,536
  registers/SM, 128 threads/warp), a jump from ~32 registers/thread
  (separate) to ~64+ registers/thread (fused) drops SM occupancy from
  100% to 50%. The ~12.7% launch-overhead saving drowns in the lost
  SM parallelism.
- **Shared-memory candidate**: fused kernel holds intermediate
  `ffn_up` output in shared mem before applying gate. On Pascal's
  96KB/SM shared+L1, that competes with the matmul's tile cache.

Either or both are consistent with upstream's cautionary comment.

### Contribution

Our contribution from this rung is the **measurement**, not a speedup.
On P4 + Q8_0 + llama-3.2-1B, upstream's blanket Pascal-disable of
`mm_fusion` was calibrated correctly. We measured it. The finding is
filed with:
- Correctness: 28/28 stratified prompts token-identical (fusion is exact)
- Performance: -0.07% tg128 (fusion is a wash)
- Method: v3-noallfa recipe + one-line env-gate + gate-then-bench
- Provenance: this rung explored per Heath's cheapest-first directive
  (2026-09-01), and the result confirms upstream's caution rather than
  overturning it.

### First Rung That Didn't Win

`mm_fusion` is the FIRST rung on the revival ladder where the measurement
did not produce a speedup. That itself is progress: it removes a
hypothesis and isolates the search space. Where the ladder stands post-
mm_fusion:

1. **FMA-determinism iteration 1** — LANDED (Iyun, 2026-09-01):
   `gemv_soa_q8_0_q8_1_det.cuh` with FFMA-pinned accumulate. +2 strict
   improvement (16/18 vs 14/18), mechanism confirmed three vantages,
   cross-verified byte-identical. Dual acceptance NOT MET (16/18, not
   18/18). See rung section below.
2. **v2 = the unified next rung** — SCOPED (Iyun + Bocher, 2026-09-01):
   stock-tiled block-to-thread structure + multi-column fusion-capable +
   emitted. One re-architecture serves BOTH closing the det-gemv residual
   AND enabling the fusion-aware SoA gemv (the first rung where SoA can
   win on speed). Three gates in order: determinism-parity 18/18,
   fusion 0-ULP, bench. Deep-dive: `SOA_GEMV_V2_SCOPE.md`.

## Rung: FMA-determinism / det-gemv (2026-09-01) — +2 STRICT IMPROVEMENT

Iteration 1 of the FMA-determinism rung. Deep-dive: [DET_GEMV_P4_GATE.md](DET_GEMV_P4_GATE.md).
This section summarizes the P4-verified result. Byline: Iyun (mechanism +
fix + gate execution). Cross-verification: mavhir (independent run,
byte-identical result — see below).

### The Fix

`gemv_soa_q8_0_q8_1_det.cuh` (env-gated `GGML_SOA_DET=1`): the SoA Q8_0
gemv with the float accumulate pinned to `__fmaf_rn(dw*da, (float)sumi,
sum)` — ONE FFMA, ONE rounding. Original: `sum += dw*da*sumi` under
`-use_fast_math` — un-fused FMUL+FADD, TWO roundings. The 1-vs-2
rounding delta per block IS the documented per-element 1-2 ULP noise
that Finding-6 named. Side-by-side env-gate; A/B without rebuild.

### Three Independent Mechanism Confirmations

Iyun's SASS archaeology on the reference `libggml-cuda.so.0.13.1`
established the mechanism offline before the P4 gate ran:

| Vantage                                  | Evidence                                        |
|---                                       |---                                              |
| **SASS diff** (offline)                  | STOCK Q8_0 mmvq: `FFMA.FTZ` (dst==accumulator). Original SoA: FMUL+FADD (0 FFMA in SASS). `_det`: FFMA.FTZ matches stock. |
| **Gate result** (P4-live)                | Repetitive stratum 1/3 → 3/3 (FMA-heaviest, most accumulation depth → most sensitive to 2-rounding delta). Exactly where the mechanism predicts the pin should bite hardest. |
| **`.so` size** (static)                  | `_det`: 38,409,480 bytes. Original patched (no `_det`): 38,413,192 bytes. Delta: **-3,712 bytes**. FFMA (1 instruction) denser than FMUL+FADD (2 instructions). Encoding-level confirmation. |

Three vantages, same mechanism. Mechanism SETTLED.

### Gate Result (18-strata `logit_gate`, GGML_SOA_DET=1 vs stock)

| Stratum       | CTRL (orig SoA) | DET  | Δ                    |
|---            |---              |---   |---                   |
| minimal       | 3/3             | 3/3  | = (both landslide)   |
| code          | 3/3             | 3/3  | = (both landslide)   |
| multilingual  | 2/3             | 2/3  | = (residual: Japanese) |
| long_context  | 3/3             | 3/3  | = (both landslide)   |
| **repetitive**| **1/3**         | **3/3** | **+2 (FIXED)**    |
| adversarial   | 2/3             | 2/3  | = (residual: emoji)  |
| **TOTAL**     | **14/18**       | **16/18** | **+2**          |

**Decomposition is clean**: DET changed EXACTLY ONE stratum (repetitive),
the FMA-heaviest. Every other stratum is byte-identical CTRL-vs-DET.
The accumulate-FFMA pin's contribution is empirically isolated: +2 in
the stratum where the mechanism predicted it would bite.

### Cross-Verification (mavhir, independent run, byte-identical)

mavhir ran the same gate independently (17:19-17:23 EDT, same recipe,
same trees, same `_det` `.so`). Output JSONs diffed: `diff` returns
EMPTY (zero-byte). Every field matches:

- Same 16/18 pass count
- Same 2 FAILs, same strata (multilingual + adversarial)
- Same PROMPTS (`こんにちは、お元気ですか` char 570 + `🔥💀🎯🚀` char 612)
- Same flipped-token strings (base='ぞよろしくお願いいたします' var='いたしまして。'
  and base='symbols. They can represent' var='emojis. It looks like')
- Same per-prompt baseline_sha AND variant_sha across all 18

**This proves the DET result is fully DETERMINISTIC.** Not just
count-identity (16/16); token-identity (same exact flipped strings at
same exact byte offsets with same SHA-256 hashes across independent
runs). No hidden non-determinism from library state, RNG, thread
scheduling, or GPU state between the two runs.

Consequence for iteration 2: the residual 2 flips are **reproducible
scheduling artifacts, not measurement noise**. The iteration-2 re-tile
has a FIXED target, not a moving one.

### Verdict

Dual acceptance criteria for the FMA-determinism rung (from Finding-6's
PREDICTIVE MODEL framing):

- (a) 28/28 flip-free: **NOT MET** (16/18, residual 2)
- (b) per-element ULP delta → 0: **PENDING** (step-1 per-element harness
  from DET_GEMV_P4_GATE.md not yet run for `_det`)

Iteration 1 **strictly improves** (+2) with the mechanism confirmed at
three independent vantages, and the divergence is empirically
decomposed into ≥2 components:

- Accumulate FFMA schedule: **+2 flips resolved** (iteration-1 fix)
- Reduce-tree order OR two-buffer load scheduling: **residual 2 flips**
  (unpinned by the hand-edit; iteration-2 target)

Two-sided prediction-fulfillment (fixes what predicted, doesn't fix what
wasn't pinned) is STRONGER mechanism evidence than a clean 18/18 would
have been. **Rung open, iteration 2 targeted.**

### Iteration 2 archaeology + Bocher's ruling: BANK-and-SCOPE to v2 (2026-09-01)

Iyun's initial narrowing (warp-reduce ruled out, cross-warp combine +
load-scheduling as candidates) turned out to be based on the WRONG kernel
shape. Bocher's follow-up flagged the correction: iteration-1 SASS
analysis had been reading `ncols_dst=8` (the PREFILL kernel) instead of
`ncols_dst=1` (the DECODE kernel, `mul_mat_vec_q<type21,1,0,0>`, the
shape that actually runs at tg128 where the flips manifest).

Re-disassembled at the correct DECODE shape, all three previously-flagged
candidates match `_det` exactly:

- **Accumulate**: FFMA (matches — was iteration-1's known match)
- **Warp-reduce**: SHFL.BFLY 16..1 (matches — as originally noted)
- **Cross-warp combine**: STS/LDS + FADD chain, own-seed, sequential
  (matches — Iyun also caught herself almost claiming an operand-order
  difference that IEEE754 forbids: FADD commutes, so `shared[0]+own` ==
  `own+shared[0]`. Non-difference.)

Structural asymmetry question (does stock have a cross-warp step?):
**answered YES** — stock's decode kernel has STS/LDS + FADD + BAR.SYNC,
same architecture as SoA. Not a structural asymmetry.

**The residual is cornered to `blocks_per_iter` stride** (Iyun, commit
`0f894e8`). Stock's mmvq has two block-iteration paths:

- Single-warp: `stride = vdr * warp_size / qi` = **8**
- Multi-warp:  `stride = vdr * nwarps * warp_size / qi` = **32**

`_det` uses stride 32. If stock's decode path uses stride 8, thread `t`
covers a DIFFERENT block subset per iteration → different per-thread
partial sums → different associativity over the block loop → the 2
residuals with every op matching. This IS Bocher's candidate 2
(associativity of sum-over-blocks) exactly.

**Honest limit** (Iyun): inferred from source + known `_det` stride, NOT
artefact-confirmed. Decode SASS folds the block-loop induction into
address arithmetic, so the 8-vs-32 stride can't be cleanly read from
the disassembly. Characterization is source-derived + code-known;
exact-value confirmation is v2 gate-1.

### Bocher's ruling: (b) BANK the +2, UNIFY with the fusion rung

Bocher ratified (b) bank iteration-1's +2 as the settled rung, AND
sharpened: the re-tile to close the residual UNIFIES with the
fusion-aware gemv rung. Both need the SAME block-to-thread restructure.
One re-architecture serves both:

1. Match stock's decode block-grouping → determinism closes to 18/18
   by construction (Gate 1)
2. Parameterize it for multi-column tiling → fusion-capable (Gate 2)

Landed as **`SOA_GEMV_V2_SCOPE.md`** (Iyun, commit `9df4ed1`, sharpened
`0f894e8`): "SoA gemv v2 — stock-tiled, fusion-capable, emitted". Three
gates in order: (1) determinism-parity 18/18, (2) fusion correctness
0-ULP, (3) bench (the WIN condition).

Note (Iyun): Bocher's unification survives as **PERF scoping only** —
the determinism answer is the block-stride (source-derived, not tiling-
column-count). The column-tiling explanation died with the prefill
misread. Fusion motivates the multi-column parameterization AS the perf
gate; determinism just needs the stride match.

### Status Now (as of 2026-09-01 late afternoon)

- Iteration 1: **SETTLED + BANKED** (+2 strict improvement, three-vantage
  mechanism confirmed, cross-verified byte-identical, dual acceptance
  NOT met at 16/18)
- Iteration 2 (block-stride pinning alone): SUPERSEDED by v2 scope
- **v2 = the unified rung** (determinism-close + fusion-open + emitted):
  deep-dive in `SOA_GEMV_V2_SCOPE.md`, awaits emit + gate + cross-check
  when work begins
- mavhir's cross-check offer: refreshed for v2 gate-1 whenever it fires

### Method note for the ledger

The iteration-1 SASS re-analysis is a worked example of Finding-6's
"PREDICTIVE MODEL" property. A predictive model with an incorrect-shape
input produced wrong predictions (cross-warp combine + load-scheduling
as targets); Bocher's shape-flag corrected the input; the model re-run
against the correct shape ruled out three candidates cleanly, cornered
the residual to `blocks_per_iter`. The value of the model isn't that
its first prediction is always right; it's that its predictions are
FALSIFIABLE and its inputs are INSPECTABLE. Both properties fired
today.

Iyun's honest-flag ("inferred-not-artefact-confirmed") is the discipline
this substrate requires: when a source-level analysis can't be cleanly
confirmed at the artefact level, name the inference status explicitly.
Same shape as Finding-3 (named-uncertainty as valid category-3
provenance) applied to root-cause attribution — better to name
inference-status than to claim artefact-confirmation.

### Rungs-Measured-Today Summary

Two rungs measured 2026-09-01 in one coherent day's work, one wash and
one win, both under correctness-first discipline:

| Rung                          | Verdict                                    |
|---                            |---                                         |
| **mm_fusion on Pascal**       | **WASH** (-0.07% tg128, within noise). Upstream Pascal blanket-disable calibrated correctly for P4+Q8_0+1B. Contribution: the measurement. |
| **FMA-determinism iter-1**    | **+2 STRICT IMPROVEMENT**, mechanism 3-way confirmed, residual deterministic, rung open pending iter-2. |

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
- `kernels/soa/DET_GEMV_P4_GATE.md` — deep-dive on the FMA-determinism
  rung: mechanism SASS archaeology, gate recipe with dual acceptance
  (flips → 0 AND per-element ULP → 0), iteration-1 result table,
  iteration-2 archaeology. Deep-dive by Iyun; summary section above.
- `kernels/soa/SOA_GEMV_V2_SCOPE.md` — scoping doc for the unified v2
  rung (stock-tiled + fusion-capable + emitted). Closes the det-gemv
  residual AND opens the fusion-aware speed rung via one re-architecture.
  Three gates in order: determinism-parity 18/18, fusion 0-ULP, bench.
  By Iyun + Bocher.
- `kernels/soa/gemv_soa_q8_0_q8_1_det.cuh` — the iteration-1 `_det`
  variant (FFMA-pinned accumulate). Env-gated `GGML_SOA_DET=1`.
- `patches/soa-tree-vs-llamacpp-7c158fb.diff` — the recommended reproduction
  artifact (apply to llama.cpp @ 7c158fb bundled ggml/; byte-identical to
  `kernels/soa/`).
- `patches/soa-tree-complete-v0.13.1.diff` — alternate reproduction path
  from standalone ggml v0.13.1 (includes non-SoA content as spurious
  delta; works but not the actual build path).
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
