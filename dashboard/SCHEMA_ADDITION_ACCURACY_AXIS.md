# Proposed schema addition — the accuracy axis

*Mavdil, 2026-09-02. **PROPOSAL for Iyun's review**, then mavhir's render.
`CONGRUENCE_SCHEMA.md` is the shared contract and this does not edit it.*

Heath's ruling: *"Improvement discrepancies are accounted differently than inaccuracy
discrepancies."*

## The two axes

```
vs STOCK (torch)   BIT_IDENTICAL / within_tolerance / failed     already in the schema
vs TRUTH           MATCHED / IMPROVED / INACCURATE                proposed here
```

## New per-kernel fields

| field | type | meaning |
|---|---|---|
| `accuracy_class` | str | `MATCHED` / `IMPROVED` / `INACCURATE` — our distance to **truth**, not to stock |
| `mean_abs_err_ours` | float? | mean absolute error vs the correctly-rounded result |
| `mean_abs_err_stock` | float? | same, for the oracle |
| `correctly_rounded_ours` | int? | how many of `total_floats` we round correctly |
| `correctly_rounded_stock` | int? | same, for the oracle |

## Three rules the schema must enforce

**1. THE AXES ARE COUPLED, NOT ORTHOGONAL — and the constraint is physics.**

`BIT_IDENTICAL` **entails** `MATCHED`. Identical bits carry identical error: we cannot be more
accurate or less. So of the nine pairs, **two are impossible**:

```
BIT_IDENTICAL + IMPROVED      IMPOSSIBLE
BIT_IDENTICAL + INACCURATE    IMPOSSIBLE
```

*A row asserting either is not an interesting cell — it is a broken measurement, and the schema
should say so rather than leave the combination merely undocumented.*

**2. THE BURDEN RUNS ONE WAY. `INACCURATE` is the default; `IMPROVED` is earned.**

Unmeasured divergence classifies `INACCURATE`. A kernel does not get to be *probably better*.

*This is authority-never-exceeds-evidence applied to the new axis, and the direction matters
because the flattering label is the self-serving one: optimism-by-default would let our own
unmeasured divergence read as improvement.*

**3. `IMPROVED` WITHOUT THE FOUR EVIDENCE NUMBERS IS MALFORMED.**

A row claiming `IMPROVED` must carry `mean_abs_err_ours`, `mean_abs_err_stock`,
`correctly_rounded_ours`, `correctly_rounded_stock`. Without them the verdict is an assertion.

*This is the empty-population guard's sibling: there, a verdict without its population; here, a
flattering verdict without its evidence.*

## Worked example — `gelu_cpu`, measured on the enclave 2026-09-02

```json
{
  "kernel": "gelu_cpu",
  "status": "PASS_ABS_TOLERANCE",   "bit_identical": false,
  "max_ulp": 127951, "diverged_count": 7682, "total_floats": 10000,
  "accuracy_class": "IMPROVED",
  "mean_abs_err_ours": 1.41e-08,  "mean_abs_err_stock": 3.93e-08,
  "correctly_rounded_ours": 6195, "correctly_rounded_stock": 2327
}
```

**One number, two opposite readings.** On the stock axis alone this cell is the matrix's worst —
127,951 ULP. On the truth axis it is our **best**: 2.8× closer to correctly-rounded than the
oracle it "fails" against.

*Without the second axis, a fix reads as a defect. That is what the axis is for.*

## For the render (mavhir)

`within_tolerance + IMPROVED` and `within_tolerance + INACCURATE` are **both reachable and
render identically on the stock axis** — both "not green", one a fix and one a defect. They need
visual distinction, or the amber cell keeps meaning two opposite things.

## Status of the evidence fields

**Not yet emitted.** The checker computes verdicts against the oracle, not against a
correctly-rounded reference — so producing these four numbers is new work in
`bit_identical_universal.py`. The gelu figures above were measured by hand today and are real;
no other cell has been measured against truth yet.

## Rules 1 and 2 collide — and the resolution matters

Applying both rules to the current matrix exposes a conflict on all 21 bit-identical cells:

```
rule 1 (physics)   BIT_IDENTICAL entails MATCHED — no truth-measurement needed
rule 2 (burden)    nothing is classified without measurement
```

**Rule 1 wins, because entailment is not assumption.** Bit-identity *is* measured — 0 ULP over
`total_floats` — and `MATCHED` follows from it **deductively**. Nothing is being taken on faith.

*The distinction is exactly the one this whole matrix is built on: a claim derived from a
measurement is not the same as a claim made without one. Rule 2 governs claims that need
evidence we do not have; rule 1 governs claims that follow from evidence we do.*

**So:** the 21 bit-identical cells are `MATCHED` with no evidence fields required. `gelu_cpu` is
`IMPROVED` with all four. **No cell needs an `UNMEASURED` class today** — but one would be
needed the moment a cell diverges and has not been measured against truth, and *that* row must
not silently read `INACCURATE` as though it had been tested and found wanting.

*Proposal: `INACCURATE` means measured-and-worse; add `UNMEASURED` for diverges-but-untested.
Both are "not a fix", and conflating them would repeat the original sin at one remove — a cell
that has not been checked reading as a cell that failed.*
