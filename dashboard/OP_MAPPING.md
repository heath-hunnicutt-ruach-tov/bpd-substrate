# Op-mapping — the two-axis join key

*Mavdil, 2026-09-02. **CONTRACT.** Drafted by Mavdil, all three open questions ratified by
mavhir (a6cb99c3) in the direction proposed. Authorship mine, sign-off theirs.*

The dashboard's two axes measure **different populations**: the runtime axis checks CPU
reference kernels against a torch oracle; the migration axis checks emitted GPU kernels. They
overlap by **operation**, not by kernel name, so the join key is a canonical `op` string.

## Proposed mapping

| canonical `op` | runtime kernel (Mavdil, from `congruence_status.json`) | emitted kernel (mavhir) |
|---|---|---|
| `matmul` | `sgemm_cpu`, `linear_cpu` | `k_matmul`, `k_sgemv_substrate_native`, `k_sgemv_cublas_match` |
| `gelu_erf` | `gelu_cpu` | — *(runtime-only)* |
| `gelu_tanh` | — *(none today; a `gelu_tanh_cpu` would give one)* | `k_gelu_tanh`, `k_vecmat_gelu` |
| `relu` | `relu_cpu` | `k_relu`, `k_vecmat_relu` |
| `silu` | `silu_cpu` | `k_silu`, `k_vecmat_silu` |
| `sigmoid` | `sigmoid_cpu` | `k_sigmoid` |
| `tanh` | `tanh_cpu` | `k_tanh` |
| `softmax` | `softmax_cpu` | `k_softmax` |
| `layer_norm` | `layernorm_cpu` | `k_layer_norm` |
| `rms_norm` | — *(no runtime cell)* | `k_rms_norm` |
| `add` | — *(no runtime cell)* | `k_add` |
| `mul` | — *(no runtime cell)* | `k_mul` |
| `scale` | — *(no runtime cell)* | `k_scale` |
| `embed` | — *(no runtime cell)* | `k_embed` |
| `rope` | — *(no runtime cell)* | `k_rope` |
| `causal_mask` | — *(no runtime cell)* | `k_causal_mask` |
| `vecmat` | — *(no runtime cell)* | `k_vecmat` |

| `fused_matmul_bias_relu` | `fused_mm_bias_relu_cpu` | — *(runtime-only)* |

**Runtime-only cells** (no emitted counterpart): `conv2d_cpu`, `upsample_cpu`, `maxpool2d_cpu`,
`fused_mm_bias_relu_cpu`, `mish_cpu`, `neg_cpu`, `abs_cpu`, `exp_cpu`, `reduce_sum_cpu`,
`reduce_mean_cpu`, `reduce_max_cpu`.

## The three questions, and how they were settled

**1. RATIFIED — the asymmetry is the point, not a defect.** 16 emitted kernels, 22 runtime rows, and only
7 ops present on both sides. A join that silently drops the non-overlapping cells would hide
**most of both axes**. The rendered view needs three states — *both*, *runtime-only*,
*emitted-only* — and an op present on one side is not a gap to be filled.

**2. RATIFIED distinct — `relu_cpu` and `fused_mm_bias_relu_cpu` are separate ops.** Is the fused kernel's relu the same
`op` as the standalone one, or a distinct `fused_matmul_bias_relu`? *I would keep them distinct*
— a fused kernel's numerics are not its parts' numerics, and merging them would let a green fused
cell imply a green relu that was checked separately or not at all.

*I first wrote "relu — no runtime cell" here from memory. `relu_cpu` is in the JSON. Caught by
checking the draft against the artefact: every runtime name cited must appear in
`congruence_status.json`, and one did not go the other way.*

**3. RATIFIED split — `k_gelu_tanh` names the TANH APPROXIMATION; `gelu_cpu` checks the ERF form.** These are
**different functions**, ~1.4e-4 apart — I measured it. Mapping both to `op: gelu` would join two
cells that are not comparable. *Proposal:* `gelu_erf` and `gelu_tanh` as separate ops.

*That third one is the case where a careless join would produce a confident wrong answer: a
green `k_gelu_tanh` beside a red `gelu_cpu` under one `op` invites the reading that they
disagree, when they compute different things.*


## Why the gelu split is the load-bearing one

*mavhir's framing, sharper than mine:* **different functions, not different implementations of
the same function.** The ~1.4e-4 gap is four orders of magnitude above the 2-ULP scale the schema
calls "tolerance" — so it is a **semantics** question, not a tolerance question.

Under one `op`, a green `k_gelu_tanh` beside an amber `gelu_cpu` would read as *the two axes
disagree*. They do not disagree; they compute different mathematics. **The split preserves the
truth that a join would have destroyed.**
