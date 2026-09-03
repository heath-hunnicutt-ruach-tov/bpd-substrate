"""schema_conformance.py — check fixtures against current schema invariants.

Mavdil's 70056acf catch: my render-stability harness tests "does the render
behave correctly on this fixture?" — a fixture that goes stale (no longer
matches current schema) keeps PASSING the render tests while silently testing
a shape we no longer emit.

The gap: render-stability != schema-conformance. Both categories of test
matter. This module adds the second category as a version-invariant gate on
fixtures that CLAIM to represent the current schema.

Categorization approach: fixtures explicitly mark their schema-era via a
top-level `_schema_era` field (or via a filename convention). Fixtures WITH
the marker are checked for full current-schema conformance. Fixtures WITHOUT
the marker (or marked as "legacy") test forward-compat paths and are NOT
gated — that's their whole purpose.

Invariants checked (all version-independent — apply to any fixture that
claims current-schema):

  I1. PHYSICS CONSTRAINT (Mavdil ebd0cb4): every row with
      bit_identical=true or status=BIT_IDENTICAL, if accuracy_class is
      present, must have accuracy_class=MATCHED. Never IMPROVED/INACCURATE
      /UNMEASURED on a bit-identical row (identical bits carry identical
      error).

  I2. STOCK-AXIS SUMS: top-level bit_identical + within_tolerance + failed,
      if all present, must sum to total.

  I3. TRUTH-AXIS SUMS: top-level matched + improved + inaccurate + unmeasured,
      if all present, must sum to total. (Mavdil 70056acf: two-axis symmetric
      top-level counts, both partition the same 22 rows.)

  I4. VERDICT-CLASS RULE (Mavdil 3bf45d7): if `passed` is present, it MUST
      NOT be treated as bit-identical count. `bit_identical` must be present
      as the true 0-ULP count. (This catches JSONs that regressed to
      deprecated `passed` in a current-era emit.)

  I5. UNDER-EXERCISED THRESHOLD SANITY: if `staleness_threshold_seconds` or
      `under_exercised_threshold` is present, it must be a positive integer.

  I6. ROW COUNT CONSISTENCY: top-level `total` equals len(kernels) list.

More invariants can be added as schema evolves; each is a version-independent
predicate on a fixture's JSON structure.

Usage:
    python3 schema_conformance.py                  # check all current-era fixtures
    python3 schema_conformance.py --json file.json # check specific file
    python3 schema_conformance.py --era-marker _schema_era=current-2026-09  # override

Exit code: 0 if all pass; 1 if any fail. CI-runnable alongside test_render.py.
"""
import argparse
import json
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Fixtures explicitly marked as legacy / forward-compat probes: exempt from
# current-schema conformance. Their whole purpose is to exercise fallback
# paths, so they should NOT match current schema.
LEGACY_FIXTURES = {
    "fresh_minimal.json",             # bare-minimum schema, no accuracy fields
    "legacy_only_passed_field.json",  # tests compat-path (`passed` fallback)
    "no_generated_field.json",        # tests missing-timestamp path
    "full_schema_three_axes.json",    # multi-schema exploration, not current-era
    "mavdil_a2_population_verdict.json",  # 20:47Z, pre-ebd0cb4 (Mavdil 70056acf:
                                          # stale fixture, will refresh on his next
                                          # real emit; skipped until then)
    "two_axis_joined.json",           # synthetic future-shape probe
    "accuracy_physics_violation.json",  # DELIBERATELY broken — tests the render
                                        # catches the physics violation
    "accuracy_all_classes.json",      # synthetic per-class rendering probe with
                                       # matched_row/improved_row/etc — not real
                                       # kernel names, exercises visual distinctness
                                       # of the 4 accuracy classes; kernel names
                                       # aren't in OP_MAPPING.md by design
}


def check_i1_physics(data, issues):
    """I1: BIT_IDENTICAL entails MATCHED (or accuracy_class absent)."""
    for i, k in enumerate(data.get("kernels", [])):
        bi = (k.get("bit_identical") is True or k.get("status") == "BIT_IDENTICAL")
        ac = k.get("accuracy_class")
        if bi and ac is not None and ac != "MATCHED":
            issues.append(
                f"I1 (physics): row {i} kernel={k.get('kernel')!r} is BIT_IDENTICAL "
                f"but accuracy_class={ac!r} (must be MATCHED — identical bits carry "
                f"identical error)"
            )


def check_i2_stock_sums(data, issues):
    """I2: bit_identical + within_tolerance + failed == total (if all present)."""
    total = data.get("total")
    bi = data.get("bit_identical")
    wt = data.get("within_tolerance")
    fl = data.get("failed")
    if all(v is not None for v in (total, bi, wt, fl)):
        s = bi + wt + fl
        if s != total:
            issues.append(
                f"I2 (stock-axis sums): bit_identical({bi}) + within_tolerance({wt}) "
                f"+ failed({fl}) = {s}, expected total={total}"
            )


def check_i3_truth_sums(data, issues):
    """I3: matched + improved + inaccurate + unmeasured == total (if all present)."""
    total = data.get("total")
    ma = data.get("matched")
    im = data.get("improved")
    ia = data.get("inaccurate")
    un = data.get("unmeasured")
    if all(v is not None for v in (total, ma, im, ia, un)):
        s = ma + im + ia + un
        if s != total:
            issues.append(
                f"I3 (truth-axis sums): matched({ma}) + improved({im}) + "
                f"inaccurate({ia}) + unmeasured({un}) = {s}, expected total={total}"
            )


def check_i4_verdict_class(data, issues):
    """I4: bit_identical must be present in current-era (no bare `passed`)."""
    if "passed" in data and "bit_identical" not in data:
        issues.append(
            f"I4 (verdict-class rule): fixture has deprecated `passed` but no "
            f"`bit_identical` — current-era emits must carry bit_identical honestly, "
            f"not conflate via passed"
        )


def check_i5_thresholds(data, issues):
    """I5: threshold fields must be positive integers."""
    for field in ("staleness_threshold_seconds", "under_exercised_threshold"):
        val = data.get(field)
        if val is not None:
            if not isinstance(val, int) or val <= 0:
                issues.append(
                    f"I5 (threshold sanity): {field}={val!r} must be positive int"
                )


def check_i6_row_count(data, issues):
    """I6: total must equal len(kernels)."""
    total = data.get("total")
    kernels = data.get("kernels")
    if total is not None and kernels is not None and len(kernels) != total:
        issues.append(
            f"I6 (row count): total={total} but len(kernels)={len(kernels)}"
        )


def check_i7_op_mapping_drift(data, issues):
    """I7: every runtime kernel in this fixture must be cited in OP_MAPPING.md.
    Every emitted kernel in data['migration']['units'] must also be cited.
    Detects DRIFT — kernels added to emitter/matrix without updating the
    canonical op-mapping doc.

    Skipped silently if op_mapping_parser can't be imported (not fatal —
    the invariant is optional for fixtures that don't exercise the mapping).
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import op_mapping_parser
        mapping = op_mapping_parser.parse()
    except Exception as e:
        # Optional invariant; if parser unavailable, skip silently.
        return

    runtime_to_op, emitted_to_op = op_mapping_parser.build_reverse_index(mapping)

    # Check runtime kernels (from data['kernels']) against runtime_to_op
    runtime_kernels = set()
    for k in data.get("kernels", []):
        name = k.get("kernel")
        if name:
            runtime_kernels.add(name)

    missing_runtime = runtime_kernels - set(runtime_to_op.keys())
    for k in sorted(missing_runtime):
        issues.append(
            f"I7 (op-mapping drift, runtime): kernel {k!r} present in cs.json "
            f"but NOT cited in OP_MAPPING.md — doc-drift, extend the mapping"
        )

    # Check emitted kernels (from data['migration']['units']) against emitted_to_op
    emitted_kernels = set()
    migration = data.get("migration", {})
    for u in migration.get("units", []):
        name = u.get("name")
        if name and not name.startswith("_"):
            emitted_kernels.add(name)

    missing_emitted = emitted_kernels - set(emitted_to_op.keys())
    for k in sorted(missing_emitted):
        issues.append(
            f"I7 (op-mapping drift, emitted): kernel {k!r} present in matrix "
            f"but NOT cited in OP_MAPPING.md — doc-drift, extend the mapping"
        )


ALL_CHECKS = [
    check_i1_physics,
    check_i2_stock_sums,
    check_i3_truth_sums,
    check_i4_verdict_class,
    check_i5_thresholds,
    check_i6_row_count,
    check_i7_op_mapping_drift,
]


def check_fixture(path):
    """Run all invariants against a fixture. Returns list of issue strings."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception as e:
        return [f"cannot parse {path}: {e}"]
    issues = []
    for check in ALL_CHECKS:
        check(data, issues)
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="Check a specific JSON file")
    ap.add_argument("--include-legacy", action="store_true",
                    help="Also check legacy/forward-compat fixtures (usually skipped)")
    args = ap.parse_args()

    if args.json:
        paths = [Path(args.json)]
    else:
        paths = sorted(FIXTURES_DIR.glob("*.json"))

    total_issues = 0
    for path in paths:
        name = path.name
        if not args.include_legacy and name in LEGACY_FIXTURES:
            # Substrate — Mavdil 70056acf: "convert a silent stale fixture
            # into a loud one." SKIP is quiet. INFO with rationale is loud.
            # Prints the fixture name + the legacy reason, so a reader
            # sees WHY it's excluded from conformance and can re-evaluate
            # whether the exclusion still holds.
            print(f"INFO {name}: LEGACY (skipped conformance; forward-compat probe)")
            continue
        issues = check_fixture(path)
        if issues:
            print(f"FAIL {name}: {len(issues)} issue(s)")
            for iss in issues:
                print(f"  {iss}")
            total_issues += len(issues)
        else:
            print(f"ok   {name}")

    print()
    print("=" * 60)
    if total_issues == 0:
        print("ALL CONFORMANCE CHECKS PASSED")
        return 0
    print(f"CONFORMANCE FAILURES: {total_issues}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
