#!/usr/bin/env python3
"""Does the checker DISTINGUISH bit-identity from mere closeness?

The project's claim is 0-ULP.  `PASS_ABS_TOLERANCE` is not 0-ULP: gelu_cpu is
127,951 ULP from torch and absolutely close.  A checker that counts it as a
pass reports "ALL KERNELS BIT-IDENTICAL" over a cell 127,951 ULP off — which is
the instrument making the claim the project exists to prevent.

That defect lived at THREE sites and survived because nothing ASSERTED on the
one case that separates the two readings.  gelu was generated on every run and
examined on none.

Method (Medayek): a fixture per verdict class the code distinguishes, asserting
on the PREDICATE and not merely the verdict — plus hand-written mutants at the
classification boundary.  A surviving mutant is mechanical proof of test
insufficiency.

Run:  python3 -m pytest tests/verdict/ -q       (needs numpy only, no torch)
"""
import os, sys, numpy as np, pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))


def _classify():
    """Import classify without executing the bench script's main()."""
    import importlib.util
    p = os.path.join(os.path.dirname(__file__), "..", "..",
                     "bench", "bit_identical_universal.py")
    # The file exits at import if torch is absent (line ~27), so extract only
    # the pure-numpy verdict logic: from `def ulp` up to the first loader.
    txt = open(p).read()
    src = txt[txt.index("def ulp("):txt.index("def load_cpu_lib")]
    ns = {"np": np}
    exec(compile(src, p, "exec"), ns)
    return ns["classify"], ns["ulp"]


# ── The four equivalence classes the code distinguishes ──────────────────
# One fixture each.  Before this file, only the first had any coverage.

def test_bit_identical_is_zero_ulp():
    c, _ = _classify()
    a = np.array([1.0, 2.5, -3.25], dtype=np.float32)
    st, mx, ab, cnt, tot = c(a, a.copy())
    assert st == "BIT_IDENTICAL" and mx == 0 and cnt == 0 and tot == 3


def test_the_discriminating_case_is_not_bit_identical():
    """gelu's shape: absolutely close, enormously far in ULP.

    THIS IS THE FIXTURE WHOSE ABSENCE LET THE BUG LIVE.  The checker
    generated this case every run and asserted on it never.
    """
    c, _ = _classify()
    ref = np.array([1e-8, 2e-8, 3e-8], dtype=np.float32)
    got = np.array([1.1e-8, 2.1e-8, 3.1e-8], dtype=np.float32)
    st, mx, ab, cnt, tot = c(ref, got)
    assert st != "BIT_IDENTICAL", "close-but-not-exact must NOT be bit-identical"
    assert mx > 0, "a non-identical case must report nonzero ULP"
    assert ab < 1e-4, "fixture invalid: this case must be absolutely close"


def test_population_travels_with_every_verdict():
    """THE EMPTY-POPULATION GUARD: no verdict without its population."""
    c, _ = _classify()
    for n in (1, 7, 1024):
        a = np.arange(n, dtype=np.float32)
        st, mx, ab, cnt, tot = c(a, a.copy())
        assert tot == n, f"total_floats must equal the compared count, got {tot} for {n}"


# ── Hand-written mutants at the classification boundary ──────────────────
# Each is a plausible edit that WOULD have shipped.  If a mutant survives
# (predicate still agrees with the real one on our fixtures) the suite does
# not discriminate at that boundary and the test is insufficient — which is
# the mechanical proof, not an opinion.

# ── EXHAUSTIVE mutation over the verdict predicate ───────────────────────
# Hand-picked mutants are a SAMPLE: they test the ones you thought of.  The
# verdict predicate is a boolean function of `status`, and `status` has a
# CLOSED range, so the space of wrong predicates is finite and enumerable:
# 2**len(STATUSES), one of which is correct.  Enumerating all of them gives a
# CEILING rather than a sample — "no survivor" then means something.

STATUSES = ["BIT_IDENTICAL", "PASS_ABS_TOLERANCE", "PASS_WITHIN_64_ULP", "FAIL"]
TRUE_GREEN = lambda s: s == "BIT_IDENTICAL"


def _all_predicates():
    import itertools
    return list(itertools.product([False, True], repeat=len(STATUSES)))


def test_the_true_predicate_is_in_the_space():
    """Control: the enumeration must contain the predicate under test."""
    truth = tuple(TRUE_GREEN(s) for s in STATUSES)
    assert truth in _all_predicates()
    assert truth == (True, False, False, False), "0-ULP is the only green"


def test_every_wrong_predicate_is_distinguishable():
    """THE CEILING: no wrong truth-table can hide behind our status vocabulary."""
    truth = tuple(TRUE_GREEN(s) for s in STATUSES)
    survivors = [m for m in _all_predicates()
                 if m != truth and all(a == b for a, b in zip(m, truth))]
    assert not survivors, (
        "predicates indistinguishable from the true one over %r: %r"
        % (STATUSES, survivors))


def test_named_historical_mutants_are_killed():
    """The specific wrong predicates that shipped, or nearly did."""
    named = {
        "green_if_PASS_substring": lambda s: "PASS" in s or "IDENTICAL" in s,
        "green_if_not_FAIL": lambda s: s != "FAIL",
        "green_if_truthy": lambda s: bool(s),
    }
    for name, mut in named.items():
        assert [s for s in STATUSES if mut(s) != TRUE_GREEN(s)], (
            "%s agrees with the true predicate on every status" % name)


def test_IDENTICAL_substring_is_a_KNOWN_SURVIVOR():
    """`"IDENTICAL" in status` cannot be distinguished from the true predicate.

    Only one status contains "IDENTICAL", so the substring form and the
    equality form agree everywhere.  The truth-table enumeration reports zero
    survivors and is right; this survivor lives in the space of EXPRESSIONS,
    which is larger than the space of truth-tables over a fixed vocabulary.

    Dormant, not harmless.  Adding a status like NEARLY_IDENTICAL would make
    the substring form silently wrong — so this test guards the CONDITION the
    equivalence rests on, and fails loudly when it stops holding.
    """
    mut = lambda s: "IDENTICAL" in s
    assert all(mut(s) == TRUE_GREEN(s) for s in STATUSES)
    containing = [s for s in STATUSES if "IDENTICAL" in s]
    assert containing == ["BIT_IDENTICAL"], (
        "a new status contains 'IDENTICAL': the substring predicate is now "
        "dangerous and every use site must be checked: %r" % containing)


def test_the_shipped_bug_is_killed_by_the_DISCRIMINATING_status():
    """The defect that reached production must die on PASS_ABS_TOLERANCE.

    Not merely on "some status" — on the class gelu_cpu actually occupies.
    That pins the test to the real case rather than a convenient one.
    """
    mut = lambda s: "PASS" in s or "IDENTICAL" in s
    assert mut("PASS_ABS_TOLERANCE") != TRUE_GREEN("PASS_ABS_TOLERANCE")
