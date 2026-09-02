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
