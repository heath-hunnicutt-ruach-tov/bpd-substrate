# Congruence Data Model — Schema (the Track A ↔ Dashboard contract)

The `congruence_status.json` file is the contract between Track A (0-ULP
verification, `bit_identical_universal.py`) and Track C (the dashboard). It is
the single source of truth for the bit-perfect-dispatch congruence matrix.

Owner: Track A (Mavdil, co-owned with Iyun). Versioned in git so the schema
outlives any thread.

## Top-level

| field | type | meaning |
|---|---|---|
| `generated` | ISO-8601 str | when this snapshot was produced |
| `total` | int | number of (kernel, shape) rows checked |
| `bit_identical` | int | rows with max_ulp == 0 (the 0-ULP count) |
| `floats_compared` | int | top-level POPULATION total: sum of every row's `total_floats` (the aggregate population the whole matrix rests on) |
| ~~`passed`~~ | int | **DEPRECATED — do not emit or consume.** See THE VERDICT-CLASS RULE below. |
| `within_tolerance` | int | rows that ran and were close but NOT bit-identical |
| `failed` | int | rows that did not pass any class |
| `open_cells` | [str] | kernels NOT bit-identical — the targets to close |
| `kernels` | [row] | per-row records (below) |

## Per-kernel row

| field | type | meaning |
|---|---|---|
| `kernel` | str | kernel name (e.g. `sgemm_cpu`) |
| `shape` | str | input shape exercised (e.g. `512x512`) |
| `status` | str | `BIT_IDENTICAL` / `PASS_ABS_TOLERANCE` / `FAIL` |
| `max_ulp` | int | max ULP deviation across all checked floats (0 = bit-identical) |
| `bit_identical` | bool | `max_ulp == 0` |
| `backend` | str | `cpu` / `cuda` |
| **`total_floats`** | int | **how many floats were compared (the POPULATION)** |
| **`diverged_count`** | int | **how many of those floats differed at all** |
| `abs_max` | float | max absolute difference across all checked floats for this row (the magnitude companion to `max_ulp`) |
| `oracle` | str | what it was checked against (e.g. `torch.nn.functional.gelu`) |
| `dtype` | str | element dtype (e.g. `float32`) |
| `device` | str | device the check ran on |
| `diagnosis` | str? | (open cells) root-cause note |
| `owner` | str? | (open cells) who's closing it |

## THE EMPTY-POPULATION GUARD (Mavdil, 2026-09-02)

**A "PASS" with no population behind it is a hollow green.** A kernel exercised
on ONE tensor showing the same green as one exercised on millions of floats is
worse than no dashboard — it invites false confidence. Therefore:

1. Every row MUST carry `total_floats`. A row without it is INVALID, not "PASS".
2. The dashboard SHOULD surface `total_floats` (and MAY dim/flag rows below a
   population threshold, e.g. < 1000 floats, as "under-exercised").
3. `bit_identical: true` means: max_ulp == 0 across `total_floats` compared
   floats, against `oracle`, at `shape`/`dtype`/`device`. All five qualify the
   claim — a 0-ULP pass against the WRONG oracle (or a trivial population) is
   the real-but-irrelevant trap this project has already been bitten by.

This is bit-identity stated with its evidence attached: not just "same bits,"
but "same bits, across THIS many floats, against THIS oracle, at THIS shape."

## THE VERDICT-CLASS RULE (Mavdil, 2026-09-02) — 0-ULP is not "pass"

`classify()` has FOUR verdict classes, and only ONE is 0-ULP:

| class | condition | 0-ULP? |
|---|---|---|
| `BIT_IDENTICAL` | max_ulp == 0 | **YES — the only true congruence** |
| `PASS_ABS_TOLERANCE` | abs<1e-4 AND max_ulp>100000 | NO (may be 100000+ ULP of divergence) |
| `PASS_WITHIN_64_ULP` | max_ulp <= 64 | NO |
| `PASS_ABS_TOLERANCE` | abs<1e-5 | NO |

**RULE: the schema carries `status` VERBATIM (the class), never a collapsed
"pass" boolean.** For a bit-perfect-dispatch project, "pass" must NOT imply
"same" — the whole point is to hold *almost-the-same* apart from *the-same*.
A green cell means `BIT_IDENTICAL` and ONLY `BIT_IDENTICAL`.

- `bit_identical` (bool) = `status == "BIT_IDENTICAL"` = `max_ulp == 0`. This
  is THE congruence metric. The dashboard's headline is the bit_identical count.
- The three `PASS_*` classes are **NOT congruence** — they are triage labels for
  cells that RUN and are numerically close but NOT bit-identical. They belong in
  `open_cells` (targets to close), not counted as green.
- **DEPRECATED: the top-level `passed` field** conflated bit-identical with
  within-tolerance (e.g. the 2026-09-02 baseline reported passed=22 while only 21
  were 0-ULP — gelu is PASS_ABS_TOLERANCE with max_ulp=127951). Replace with:
  `bit_identical` (the 0-ULP count) + `within_tolerance` (the close-but-not-exact
  count) + `failed` (ran-but-diverged-beyond-tolerance). Three honest counts, no
  collapse.

**A reader must always recover which verdict a cell earned.** Bit-identical and
within-64-ULP are different facts; the matrix never loses the distinction. This
is the empty-population guard's sibling: population says HOW MUCH was checked;
verdict-class says HOW EXACTLY it matched — and "pass" alone answers neither.

## THE TWO-AXIS MERGE (mavhir + Iyun, 2026-09-02) — runtime × migration

Two independent 0-ULP claims per kernel, tracked in ONE schema so the dashboard
tells the full bit-perfect story. Both axes share the 22-kernel domain.

**Axis 1 — RUNTIME congruence** (Track A, `bit_identical_universal.py`):
does the kernel COMPUTE bit-identically to the oracle?
- `status`, `max_ulp`, `bit_identical`, `total_floats`, `oracle`, `dtype`, `device`

**Axis 2 — MIGRATION preservation** (mavhir's `emit_diff_matrix.py`):
does the kernel EMIT byte-identically swipl → swilgt (Logtalk migration)?
- `migration_source_identical` (bool): swipl_sha == swilgt_sha
- `swipl_sha` / `swilgt_sha` (str): the emission hashes (for audit)
- `migration_unit` (str): which emitter unit (blas/fused/llama)

A kernel is FULLY bit-perfect when BOTH axes are green: it computes 0-ULP AND
its source survives the Logtalk migration byte-identically. The dashboard shows
two columns; a cell is fully-green only when both are.

**SEQUENCING (one app-edit, not three):** the dashboard schema changes THREE
times — (a) population fields, (b) verdict-honest counts, (c) this migration
axis. All three land TOGETHER in one mavhir app-edit + restart when Mavdil's
population-aware + verdict-honest status.json is ready AND mavhir's migration
matrix is merged. Both build to THIS spec ahead of time. No churny multi-restart.

Merged top-level counts: `bit_identical` (runtime 0-ULP), `migration_identical`
(source-preserved), `fully_bit_perfect` (both axes green), plus `within_tolerance`
/ `failed` (runtime non-0-ULP), `open_cells` (any axis not green).

## THE TWO-AXIS JOIN KEY (mavhir's catch, 2026-09-02) — join by OPERATION, not name

mavhir caught that the two axes test DIFFERENT kernel sets and cannot join by name:
- **Runtime axis** (`bit_identical_universal.py`): CPU reference kernels vs torch —
  names like `sgemm_cpu`, `silu_cpu`, `softmax_cpu`, `gelu_cpu`.
- **Migration axis** (`emit_diff_matrix.py`): EMITTED GPU kernels swipl-vs-swilgt —
  names like `k_matmul`, `k_silu`, `k_softmax`, `k_gelu_tanh`.

These are different populations that OVERLAP BY OPERATION, not by name. So the
merged view joins on a canonical **`op`** field (the operation), NOT the kernel name.

**JOIN CONTRACT:** every row (both axes) carries a canonical `op` string. The merged
dashboard groups by `op` and shows both axes' status side by side. Where an op appears
in only one axis (e.g. a CPU-only reference kernel with no emitted counterpart, or an
emitted kernel not yet in the runtime battery), the other axis shows `n/a` — NOT green,
NOT red, explicitly not-applicable. (An op present in one axis and absent in the other
is itself information: coverage gaps between the runtime battery and the emission set.)

Canonical op mapping (to be finalized by Track A + mavhir together):
```
  sgemm_cpu / linear_cpu   <-> k_matmul / k_sgemv_*   : op="matmul"
  silu_cpu                 <-> k_silu                 : op="silu"
  softmax_cpu              <-> k_softmax              : op="softmax"
  gelu_cpu                 <-> k_gelu_tanh / k_vecmat_gelu : op="gelu"
  layernorm_cpu            <-> k_layer_norm / k_rms_norm  : op="norm" (or split ln/rms)
  ... (full table owned by Track A + mavhir; ops present in only one axis => n/a in the other)
```

**IMPLICATION FOR SEQUENCING:** because the join needs this `op` field on BOTH axes'
JSON, the merged render CANNOT be pre-built blind — it waits for (1) this join spec,
(2) Mavdil's runtime JSON carrying `op`, (3) mavhir's migration JSON carrying `op`.
Then ONE app-edit renders the joined two-axis view. mavhir's (a) is correct: wait for
the merged data (both axes, op-keyed), then one coherent app-edit. Not (b) — (b) can't
render a join whose key isn't yet on the data.

---

## The accuracy axis — improvement vs inaccuracy (Heath's ruling, 2026-09-02)

Heath's ruling: *"We always reproduce the bugs as needed... Then, once we document their
defect, we change to the more-accurate fix, but we account for the discrepancy in ULP
differently. Improvement discrepancies are accounted differently than inaccuracy
discrepancies."*

This adds a SECOND, orthogonal-in-intent axis to every kernel row. The existing axis measures
distance-to-STOCK; this one measures distance-to-TRUTH.

```
vs STOCK (torch)   BIT_IDENTICAL / within_tolerance / failed      (the original axis)
vs TRUTH           MATCHED / IMPROVED / INACCURATE / UNMEASURED    (the accuracy axis)
```

### New per-kernel fields

| field | type | meaning |
|---|---|---|
| `accuracy_class` | str | `MATCHED` / `IMPROVED` / `INACCURATE` / `UNMEASURED` — our distance to **truth**, not to stock |
| `mean_abs_err_ours` | float? | mean absolute error vs the correctly-rounded (float64-then-round-once) reference |
| `mean_abs_err_stock` | float? | same, for the oracle |
| `correctly_rounded_ours` | int? | how many of `total_floats` we round correctly |
| `correctly_rounded_stock` | int? | same, for the oracle |

### The four accuracy classes

- **`MATCHED`** — our result carries exactly stock's error (entailed by `BIT_IDENTICAL`; see rule 1).
- **`IMPROVED`** — MEASURED closer to truth than stock. Requires the four evidence fields.
- **`INACCURATE`** — MEASURED further from truth than stock (a real defect in our kernel).
- **`UNMEASURED`** — diverges from stock but NOT YET checked against truth. The honest default
  for a non-bit-identical cell before truth-measurement. `UNMEASURED` must NOT silently read as
  `INACCURATE` — a cell that has not been checked is not a cell that failed.

### Three rules the schema enforces

**1. THE AXES ARE COUPLED, NOT ORTHOGONAL — the constraint is physics.**
`BIT_IDENTICAL` **entails** `MATCHED`. Identical bits carry identical error — we cannot be more
accurate or less. So of the nine (stock × accuracy) pairs, **two are impossible**:
```
BIT_IDENTICAL + IMPROVED      IMPOSSIBLE   (broken measurement)
BIT_IDENTICAL + INACCURATE    IMPOSSIBLE   (broken measurement)
```
A row asserting either is not a cell — it is a broken measurement, and the render surfaces it
as `IMPOSSIBLE` rather than displaying it as valid.

**2. THE BURDEN RUNS ONE WAY. The flattering label is earned; the honest default is not a fix.**
Unmeasured divergence classifies `UNMEASURED` (not `IMPROVED`). `IMPROVED` is a MEASURED claim.
This is authority-never-exceeds-evidence applied to the axis: optimism-by-default would let our
own unmeasured divergence read as improvement — the self-serving conflation the axis prevents.

**3. `IMPROVED` (or `INACCURATE`) WITHOUT THE FOUR EVIDENCE NUMBERS IS MALFORMED.**
A measured verdict must carry `mean_abs_err_ours/stock` + `correctly_rounded_ours/stock`.
Without them the verdict is an assertion, not a measurement. (Sibling of the empty-population
guard: there, a verdict without its population; here, a measured-verdict without its evidence.)

### Rule 1 vs Rule 2 — the collision and its resolution

Rules 1 (physics: BIT_IDENTICAL entails MATCHED, no measurement needed) and 2 (burden: nothing
classified without measurement) appear to conflict on the 21 bit-identical cells. **Rule 1 wins,
because entailment is not assumption:** bit-identity IS measured (0 ULP over `total_floats`), and
`MATCHED` follows DEDUCTIVELY. A claim derived from a measurement is not a claim made without one.
So the 21 bit-identical cells are `MATCHED` with NO evidence fields required; a within_tolerance
or failed cell is `UNMEASURED` until truth-measured, then `IMPROVED`/`INACCURATE` with evidence.

### Worked example — `gelu_cpu` (measured on the enclave, 2026-09-02)

```json
{ "kernel": "gelu_cpu", "status": "PASS_ABS_TOLERANCE", "bit_identical": false,
  "max_ulp": 127951, "diverged_count": 7682, "total_floats": 10000,
  "accuracy_class": "IMPROVED",
  "mean_abs_err_ours": 1.09e-08,  "mean_abs_err_stock": 3.86e-08,
  "correctly_rounded_ours": 6211, "correctly_rounded_stock": 2336 }
```
One number, two opposite readings. On the stock axis this cell is the matrix's worst (127,951
ULP); on the truth axis it is our BEST — ~3.4× closer to correctly-rounded than the oracle it
"fails" against. Without the second axis, a fix reads as a defect. That is what the axis is for.

*Contract source-of-record: this section. `SCHEMA_ADDITION_ACCURACY_AXIS.md` was the proposal;
it is now folded here and superseded — a contract in two files is a contract that drifts.*
