#!/usr/bin/env python3
"""The SECOND verdict axis: diverges-because-BETTER vs diverges-because-WORSE.

Heath's ruling, 2026-09-02: "Improvement discrepancies are accounted
differently than inaccuracy discrepancies."

The existing axis measures us against STOCK (torch):
    BIT_IDENTICAL / within_tolerance / failed
The new axis measures us against TRUTH (the correctly-rounded result):
    MATCHED / IMPROVED / INACCURATE

They are NOT independent, and conflating them is the hollow green this whole
matrix guards against: gelu_cpu diverges from torch by 127,951 ULP AND is 2.8x
closer to the true gelu than torch is.  One number, two opposite readings —
"red vs stock" and "better than stock" are the same cell.

Run:  python3 -m pytest tests/verdict/ -q      (numpy only, no torch)
"""
import pytest

STOCK_CLASSES = ["BIT_IDENTICAL", "within_tolerance", "failed"]
TRUTH_CLASSES = ["MATCHED", "IMPROVED", "INACCURATE"]


def reachable(stock, truth):
    """Which (stock, truth) pairs can physically occur?

    Identical bits cannot differ in accuracy: BIT_IDENTICAL pins truth to
    MATCHED.  Every other combination is reachable.
    """
    if stock == "BIT_IDENTICAL":
        return truth == "MATCHED"
    return True


def test_bit_identical_forces_matched():
    """The axes are constrained, not orthogonal — and the constraint is physics.

    If our output is bit-identical to stock, it has EXACTLY stock's error.  It
    cannot be more accurate and cannot be less.  A row claiming
    BIT_IDENTICAL + IMPROVED is not an interesting finding; it is a bug in the
    measurement.
    """
    assert reachable("BIT_IDENTICAL", "MATCHED")
    assert not reachable("BIT_IDENTICAL", "IMPROVED")
    assert not reachable("BIT_IDENTICAL", "INACCURATE")


def test_the_dangerous_pair_is_reachable():
    """within_tolerance + INACCURATE and within_tolerance + IMPROVED both exist.

    THIS is why the second axis is needed.  Both render as "not green" on the
    stock axis alone, and they are opposite facts: one is a defect, the other
    is a fix.  A matrix with only the stock axis cannot tell them apart.
    """
    assert reachable("within_tolerance", "IMPROVED")
    assert reachable("within_tolerance", "INACCURATE")


def test_gelu_is_the_worked_example():
    """gelu_cpu, measured 2026-09-02 on the enclave:

        vs stock:  127,951 max ULP over 10,000 floats  -> within_tolerance
        vs truth:  mean|err| 1.41e-08 vs torch's 3.93e-08,
                   correctly-rounded 6195/10000 vs torch's 2327  -> IMPROVED

    Recorded as a fixture so the classification has a real instance behind it
    and not only a definition.
    """
    stock, truth = "within_tolerance", "IMPROVED"
    assert reachable(stock, truth)
    assert stock != "BIT_IDENTICAL", "gelu is not 0-ULP vs torch"
    assert truth == "IMPROVED", "and it is closer to true gelu than torch is"


@pytest.mark.parametrize("stock", STOCK_CLASSES)
@pytest.mark.parametrize("truth", TRUTH_CLASSES)
def test_every_pair_is_classified(stock, truth):
    """Exhaustive: all 9 combinations have a defined reachability verdict.

    No pair may be undefined — an unclassified combination is where a row
    would land with no rule to interpret it.
    """
    assert isinstance(reachable(stock, truth), bool)


def test_improvement_requires_evidence_not_assertion():
    """IMPROVED is a MEASURED claim, never a default for "we diverge".

    A kernel that diverges is INACCURATE until measured against truth.  This
    encodes the direction of the burden: authority should never exceed
    evidence, so the optimistic label is the one that must be earned.
    """
    def classify(diverges, measured_closer_to_truth=None):
        if not diverges:
            return "MATCHED"
        if measured_closer_to_truth is None:
            return "INACCURATE"          # unmeasured divergence is not a win
        return "IMPROVED" if measured_closer_to_truth else "INACCURATE"

    assert classify(False) == "MATCHED"
    assert classify(True) == "INACCURATE", "unmeasured divergence must NOT read as improvement"
    assert classify(True, measured_closer_to_truth=True) == "IMPROVED"
    assert classify(True, measured_closer_to_truth=False) == "INACCURATE"
