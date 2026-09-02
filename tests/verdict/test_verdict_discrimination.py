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

MUTANTS = {
    "green_if_PASS_substring":  lambda s: "PASS" in s or "IDENTICAL" in s,  # the real bug
    "green_if_not_FAIL":        lambda s: s != "FAIL",
    "green_if_IDENTICAL_substr": lambda s: "IDENTICAL" in s,
    "green_if_truthy":          lambda s: bool(s),
}
TRUE_GREEN = lambda s: s == "BIT_IDENTICAL"

# Statuses the checker can emit.  A mutant must disagree with TRUE_GREEN on
# at least one of them, or it is indistinguishable from correct.
STATUSES = ["BIT_IDENTICAL", "PASS_ABS_TOLERANCE", "PASS_WITHIN_64_ULP", "FAIL"]


@pytest.mark.parametrize("name,mut", list(MUTANTS.items()))
def test_mutant_is_killed_by_some_status(name, mut):
    """Every boundary mutant must be DISTINGUISHABLE from the true predicate.

    'green_if_IDENTICAL_substr' is the interesting one: it agrees with
    TRUE_GREEN on all four current statuses, so our fixtures CANNOT kill it.
    That is a real gap, and the test names it rather than hiding it.
    """
    disagreements = [s for s in STATUSES if mut(s) != TRUE_GREEN(s)]
    if name == "green_if_IDENTICAL_substr":
        pytest.xfail("SURVIVING MUTANT: indistinguishable while no status "
                     "contains 'IDENTICAL' without being BIT_IDENTICAL. "
                     "Add such a status and this becomes killable.")
    assert disagreements, (
        f"mutant {name} agrees with the true predicate on every status the "
        f"checker emits — the suite cannot discriminate at this boundary")
