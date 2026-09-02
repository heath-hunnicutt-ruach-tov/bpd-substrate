#!/usr/bin/env python3
"""Tests for the truth-reference module — the accuracy axis's evidence source."""
import os, sys, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))
import truth_reference as tr


def test_unavailable_truth_gives_UNMEASURED_not_INACCURATE():
    """THE CENTRAL GUARD: a kernel we cannot check must not read as one that failed.

    sgemm/conv2d/softmax have no cheap float64 truth — reimplementing them
    would be a second implementation with its own accumulation-order choices,
    not a truth.  Those cells are UNMEASURED.
    """
    assert tr.truth_of("sgemm_cpu", np.zeros(4, dtype=np.float32)) is None
    cls, ev = tr.accuracy_class(np.zeros(4, np.float32), np.zeros(4, np.float32), None)
    assert cls == "UNMEASURED"
    assert ev == {}, "an UNMEASURED verdict carries no evidence, and claims none"


def test_identical_inputs_classify_MATCHED():
    x = np.array([0.5, -1.5, 2.0], dtype=np.float32)
    t = tr.truth_of("tanh_cpu", x)
    cls, ev = tr.accuracy_class(t, t, t)
    assert cls == "MATCHED"


def test_IMPROVED_requires_being_better_on_BOTH_measures():
    """Closer mean error AND at least as many correctly-rounded.

    One metric alone can be gamed by a kernel that is closer on average while
    rounding correctly less often.  Both, or it is not an improvement.
    """
    x = np.linspace(-3, 3, 512).astype(np.float32)
    t = tr.truth_of("sigmoid_cpu", x)
    worse = (t.astype(np.float64) + 1e-6).astype(np.float32)
    cls, ev = tr.accuracy_class(t, worse, t)
    assert cls == "IMPROVED"
    assert ev["mean_abs_err_ours"] < ev["mean_abs_err_stock"]
    assert ev["correctly_rounded_ours"] >= ev["correctly_rounded_stock"]


def test_being_worse_classifies_INACCURATE():
    x = np.linspace(-3, 3, 512).astype(np.float32)
    t = tr.truth_of("sigmoid_cpu", x)
    worse = (t.astype(np.float64) + 1e-6).astype(np.float32)
    cls, _ = tr.accuracy_class(worse, t, t)
    assert cls == "INACCURATE"


def test_every_verdict_but_UNMEASURED_carries_its_four_numbers():
    """Schema rule 3: IMPROVED/INACCURATE without evidence is malformed."""
    x = np.linspace(-2, 2, 256).astype(np.float32)
    t = tr.truth_of("exp_cpu", x)
    for ours in (t, (t.astype(np.float64) * 1.0000001).astype(np.float32)):
        cls, ev = tr.accuracy_class(ours, t, t)
        if cls != "UNMEASURED":
            for k in ("mean_abs_err_ours", "mean_abs_err_stock",
                      "correctly_rounded_ours", "correctly_rounded_stock"):
                assert k in ev, f"{cls} row missing {k}"
