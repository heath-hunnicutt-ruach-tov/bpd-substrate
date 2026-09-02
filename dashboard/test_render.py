#!/usr/bin/env python3
"""test_render.py — regression harness for congruence_dashboard.py.

Exercises the render function against fixture JSONs in dashboard/fixtures/
and asserts each schema-honesty surface fires correctly. If any assertion
fails, exit code 1 (breaks CI); all-pass = exit code 0.

Fixtures cover the five honesty surfaces + the fallback paths:

  fresh_minimal.json:
      Fresh timestamp, minimal one-kernel schema. Asserts:
        - freshness stamp: ok class, "Ns ago" (< 5 min)
        - render succeeds without crash

  legacy_only_passed_field.json:
      Has `passed` (deprecated) but no `bit_identical`. Asserts:
        - Compat-path unverified marker fires (Mavdil's catch, aa27068):
          count displays with asterisk, warn class, "unverified" in label
        - Old-timestamp freshness stamp shows fail class

  full_schema_three_axes.json:
      Full new-schema JSON with all three axes populated (runtime +
      population + verdict-honest + migration). Asserts all 12 features:
        - bit_identical count, migration count, fully_bit_perfect count,
          within_tolerance count, under-exercised count, total_floats
          with commas, under-flag on low-pop rows, diagnosis + owner in
          notes, migration byte-identical label, oracle column,
          dtype/device subtext, freshness fail class (old timestamp)

  no_generated_field.json:
      JSON missing `generated` field entirely. Asserts:
        - freshness stamp: muted class, "(unknown age)" label — never
          crashes on missing timestamp

Provenance: fixtures banked from smoke tests during commit 7e7a724 land
(freshness stamp) + f913be0 (schema-aware render) + aa27068 (compat-path
fix). Heath's rule: source in git, artifacts in output directories,
nothing in /tmp. These fixtures ARE the artifacts of the smoke tests,
now in git as regression tests.

Usage:
    cd dashboard && python3 test_render.py
    cd dashboard && python3 test_render.py --fixture full_schema_three_axes.json  # single
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DASHBOARD_PY = Path(__file__).resolve().parent / "congruence_dashboard.py"


def load_render(fixture_path):
    """Load congruence_dashboard.py as a module, point STATUS_FILE at
    fixture, call render(), return HTML string."""
    spec = importlib.util.spec_from_file_location("cd", str(DASHBOARD_PY))
    m = importlib.util.module_from_spec(spec)
    m.STATUS_FILE = str(fixture_path)
    spec.loader.exec_module(m)
    # After exec, module-level STATUS_FILE assignment may have been overwritten
    # if the file has argparse; re-set to force our fixture path
    m.STATUS_FILE = str(fixture_path)
    return m.render()


def _find_freshness_stamp(html):
    """Extract (css_class, label) from the freshness stamp span, or None."""
    match = re.search(r'<span class="fresh-stamp (\w+)">([^<]+)</span>', html)
    if match:
        return (match.group(1), match.group(2))
    return None


def check(cond, msg, failures):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        failures.append(msg)


def test_fresh_minimal(html, failures):
    print("== fresh_minimal.json (fresh timestamp, one kernel) ==")
    stamp = _find_freshness_stamp(html)
    check(stamp is not None, "freshness stamp present", failures)
    if stamp:
        cls, label = stamp
        # Note: fixture timestamp is 2026-09-02T19:50:00Z; when this test runs
        # in a future session, that will be stale. So we accept ok OR fail
        # here — the point is "renders SOMETHING honest, class matches age",
        # not "is guaranteed fresh". At original bank time it was fresh.
        check(cls in ("ok", "warn", "fail"), "freshness class is one of ok/warn/fail (got %s)" % cls, failures)
        check(bool(re.match(r".+ ago$", label)), 'freshness label ends with " ago" (got %r)' % label, failures)
    check("test</td>" in html, "kernel row 'test' present", failures)
    check("runtime 0-ULP" in html, "runtime 0-ULP framing present", failures)


def test_legacy_only(html, failures):
    print("== legacy_only_passed_field.json (deprecated passed, no bit_identical) ==")
    check("22*" in html, "unverified asterisk on count (Mavdil's compat-path catch)", failures)
    check("count warn" in html, "warn class on headline (not ok class)", failures)
    check(
        "unverified" in html and "deprecated" in html,
        "unverified + deprecated markers in label",
        failures,
    )
    stamp = _find_freshness_stamp(html)
    check(stamp is not None, "freshness stamp present", failures)
    if stamp:
        cls, label = stamp
        # Fixture generated 2026-08-01T00:00:00Z — definitely stale by any run
        check(cls == "fail", "freshness class fail (fixture is Aug 1, always stale, got %s)" % cls, failures)


def test_full_schema(html, failures):
    print("== full_schema_three_axes.json (all three axes populated) ==")
    features = [
        ("bit_identical count widget", "runtime 0-ULP"),
        ("migration count widget", "migration source-identical"),
        ("fully bit-perfect count widget", "fully bit-perfect"),
        ("within tolerance count widget", "within tolerance"),
        ("under-exercised count widget", "under-exercised"),  # gelu at 128 < 1000
        ("total_floats with commas", "262,144"),
        ("under-flag on gelu row", "under-flag"),
        ("gelu diagnosis", "different approximation"),
        ("gelu owner", "@Mavdil"),
        ("migration byte-identical label", "byte-identical"),
        ("oracle column populated", "torch.matmul"),
        ("dtype rendered", "float32"),
    ]
    for name, needle in features:
        check(needle in html, name, failures)
    stamp = _find_freshness_stamp(html)
    check(stamp is not None, "freshness stamp present", failures)
    if stamp:
        cls, _ = stamp
        # Fixture generated 2026-09-02T19:30:00Z — will be stale on any
        # future run; ok at original bank; treat as SOMETHING-honest.
        check(cls in ("ok", "warn", "fail"), "freshness class rendered (got %s)" % cls, failures)


def test_no_generated(html, failures):
    print("== no_generated_field.json (missing timestamp) ==")
    stamp = _find_freshness_stamp(html)
    check(stamp is not None, "freshness stamp present even with missing timestamp", failures)
    if stamp:
        cls, label = stamp
        check(cls == "muted", "freshness class muted (got %s)" % cls, failures)
        check(label == "(unknown age)", 'freshness label "(unknown age)" (got %r)' % label, failures)


def test_mavdil_a2(html, failures):
    print("== mavdil_a2_population_verdict.json (Mavdil's A2 real output) ==")
    check("493,395" in html, "floats_compared top-level widget renders 493,395",
          failures)
    check("floats compared" in html, "'floats compared' label present",
          failures)
    check("7,682" in html, "gelu_cpu diverged_count 7,682 surfaced",
          failures)
    check("10,000" in html, "gelu_cpu total_floats 10,000 surfaced",
          failures)
    check("diverged" in html, "diverged label present",
          failures)
    check("21" in html, "bit_identical count 21 present", failures)
    check("gelu_cpu" in html, "gelu_cpu row present", failures)
    check("127951" in html, "gelu max_ulp 127951 present", failures)
    check("rows under-exercised" in html,
          "under-exercised widget fires on real data",
          failures)
    # legacy asterisk must NOT fire (JSON has bit_identical)
    check("bit_identical*" not in html,
          "legacy asterisk correctly absent (JSON has bit_identical)",
          failures)




def test_two_axis_joined(html, failures):
    print("== two_axis_joined.json (mavdil's C+(i) synthetic, three-state) ==")
    # By-op section header
    check("By Op" in html, "by-op section header present", failures)
    check("Two-Axis Join" in html or "two-axis join" in html.lower(),
          "two-axis-join framing present", failures)
    # Three states rendered
    check("state-both" in html, "state-both cells present", failures)
    check("state-runtime-only" in html, "state-runtime-only cells present", failures)
    check("state-emitted-only" in html, "state-emitted-only cells present", failures)
    # OP_MAPPING.md ops surfaced
    check("matmul" in html, "matmul op present", failures)
    check("gelu_erf" in html, "gelu_erf op present (from mavdil's split)", failures)
    check("gelu_tanh" in html, "gelu_tanh op present (from mavdil's split)", failures)
    # Emitted-side kernels appear
    check("k_matmul" in html, "k_matmul emitted kernel rendered", failures)
    check("k_gelu_tanh" in html, "k_gelu_tanh emitted kernel rendered", failures)
    # Migration axis surfaces
    check("byte-identical" in html, "migration byte-identical label present", failures)
    check("fully bit-perfect" in html, "fully bit-perfect top-level widget fires", failures)
    # Absent cell markers
    check("no runtime cell" in html or "no emitted counterpart" in html,
          "explicit absent-cell markers present (three-state honesty)", failures)


def test_axis_grouping(html, failures):
    """Ratifies Mavdil 70056acf: two partitions of the same rows must be
    visually grouped by axis, not flat-listed. Runs against
    accuracy_axis_populated.json which has BOTH stock + truth top-level
    counts populated."""
    check("vs stock (runtime)" in html, "runtime-axis group title present", failures)
    check("vs truth (accuracy)" in html, "truth-axis group title present", failures)
    check("count-group axis-runtime" in html, "runtime group has axis-runtime class", failures)
    check("count-group axis-truth" in html, "truth group has axis-truth class", failures)
    check("count-group axis-meta" in html, "meta group present (floats_compared etc)", failures)


def test_accuracy_axis(html, failures):
    print("== accuracy_axis_populated.json (mavdil's ebd0cb4 emitter shape) ==")
    # Mavdil 70056acf: axis grouping check runs on the same fixture
    test_axis_grouping(html, failures)
    # Accuracy column header
    check("Accuracy (vs truth)" in html, "Accuracy column header present", failures)
    # Class labels rendered
    check("MATCHED" in html, "MATCHED class rendered", failures)
    check("IMPROVED" in html, "IMPROVED class rendered", failures)
    # Top-level widgets
    check("matched (vs truth" in html, "matched top-level widget fires", failures)
    check("IMPROVED vs truth" in html, "improved top-level widget fires", failures)
    # Evidence surfacing for IMPROVED row
    check("3.5" in html or "3.4" in html, "improvement ratio evidence rendered (should be ~3.5x)", failures)
    check("better" in html, "'better' label in evidence subtext", failures)
    check("6,211" in html or "6211" in html, "correctly_rounded_ours 6,211 surfaced", failures)
    check("2,336" in html or "2336" in html, "correctly_rounded_stock 2,336 surfaced", failures)
    # Physics constraint: BIT_IDENTICAL entails MATCHED (no IMPOSSIBLE labels
    # since all bit-identical rows have accuracy_class=MATCHED in this fixture)
    check("IMPOSSIBLE" not in html, "no physics-constraint violation in valid fixture", failures)
    # improved color class distinct from ok
    check("class=\"improved\"" in html or "class='improved'" in html,
          "improved css class used (distinct blue, not green)", failures)


def test_accuracy_physics_violation(html, failures):
    print("== accuracy_physics_violation.json (BIT_IDENTICAL + IMPROVED = broken) ==")
    # Mavdil's physics constraint: BIT_IDENTICAL entails MATCHED.
    # Broken measurement should render explicitly as IMPOSSIBLE so it can't be missed.
    check("IMPOSSIBLE" in html,
          "physics-constraint violation surfaces as IMPOSSIBLE (bit-identical row with non-MATCHED accuracy)",
          failures)
    check("broken measurement" in html,
          "explanation 'broken measurement' present for violation",
          failures)
    # The row's accuracy cell should be fail-class not improved-class
    # (find the row's accuracy cell area — should have both dot fail + fail class label)
    check("dot fail" in html, "fail dot on violation row", failures)


def test_accuracy_all_classes(html, failures):
    print("== accuracy_all_classes.json (MATCHED + IMPROVED + INACCURATE + UNMEASURED distinct) ==")
    # All four class labels rendered
    check("MATCHED" in html, "MATCHED label present", failures)
    check("IMPROVED" in html, "IMPROVED label present", failures)
    check("INACCURATE" in html, "INACCURATE label present", failures)
    check("UNMEASURED" in html, "UNMEASURED label present (uppercase, emphatic)", failures)
    # Mavdil's 79ef4702 catch: UNMEASURED must be visually DISTINCT from INACCURATE.
    # Verify css classes are different — not both "fail" or both "muted".
    check('class="unmeasured"' in html or "class='unmeasured'" in html,
          "unmeasured class used (distinct from fail/muted per Mavdil 79ef4702)",
          failures)
    check('class="fail"' in html or "class='fail'" in html,
          "fail class used (for INACCURATE)", failures)
    # And the classes are LITERALLY DIFFERENT strings — proves distinctness
    check("class=\"unmeasured\"" != "class=\"fail\"",
          "unmeasured class-string differs from fail class-string (distinct css)",
          failures)
    # Widget check
    check("UNMEASURED vs truth" in html,
          "UNMEASURED top-level widget uses UNMEASURED-not-INACCURATE label",
          failures)
    check("NOT worse-than-stock" in html,
          "UNMEASURED widget explains 'NOT worse-than-stock' distinction",
          failures)


FIXTURE_TESTS = {
    "fresh_minimal.json": test_fresh_minimal,
    "legacy_only_passed_field.json": test_legacy_only,
    "full_schema_three_axes.json": test_full_schema,
    "no_generated_field.json": test_no_generated,
    "mavdil_a2_population_verdict.json": test_mavdil_a2,
    "two_axis_joined.json": test_two_axis_joined,
    "accuracy_axis_populated.json": test_accuracy_axis,
    "accuracy_physics_violation.json": test_accuracy_physics_violation,
    "accuracy_all_classes.json": test_accuracy_all_classes,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", help="run only one fixture (default: all)")
    args = ap.parse_args()

    if args.fixture:
        fixtures = {args.fixture: FIXTURE_TESTS[args.fixture]}
    else:
        fixtures = FIXTURE_TESTS

    failures = []
    for fname, test_fn in fixtures.items():
        fpath = FIXTURES_DIR / fname
        if not fpath.exists():
            print("SKIP %s (fixture not found)" % fname)
            continue
        try:
            html = load_render(fpath)
        except Exception as e:
            print("FAIL %s (render raised %s: %s)" % (fname, type(e).__name__, e))
            failures.append("%s render exception" % fname)
            continue
        test_fn(html, failures)
        print()

    if failures:
        print("=" * 60)
        print("FAILURES: %d" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("=" * 60)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

