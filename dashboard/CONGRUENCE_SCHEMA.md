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
| `passed` | int | rows that passed (bit-identical OR within documented tolerance) |
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
