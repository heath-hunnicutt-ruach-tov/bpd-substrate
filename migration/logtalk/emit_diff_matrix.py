#!/usr/bin/env python3
"""emit_diff_matrix.py — per-kernel byte-identity cross-check for the Logtalk migration.

Bocher's whole-stream gate (emission_gate.sh) is the floor: 3 generators
emit byte-identically under swipl vs swilgt. This is the value-add above
that floor: parse each stream by kernel-function boundary, hash each
kernel separately, so a future divergence localizes to a NAMED KERNEL
rather than a whole-stream FAIL. Same discipline as mavhir's cross-check
of Iyun's det-gemv gate (byte-identity discipline at higher resolution).

Bocher recommended this per-kernel granularity in his 1a7a87b1 protocol
answer. Independent second instrument on Bocher's B3 emission gate.
Attribution: artefact-level facts here; disposition to Bocher.

Usage:
    ./emit_diff_matrix.py                    # cross-check swipl vs swilgt across all 3 generators
    ./emit_diff_matrix.py --host swipl       # emit + snapshot per-kernel hashes from one host
    ./emit_diff_matrix.py --json OUT.json    # write structured results
    ./emit_diff_matrix.py --generators blas  # subset

Kernel-boundary parsing: matches `__global__ void k_NAME(...) {`, then
counts braces to find the matching `}`. Extracts the full function body
(signature-line through closing-brace, inclusive) as the per-kernel unit.
Non-kernel content (preamble, extern "C" C-bindings, comments between
kernels) is captured into synthetic units named `_preamble` / `_epilogue`
to ensure NO byte is unaccounted-for (empty-output-class guard).

CONTROLS (inherited from Bocher's emission_gate.sh):
- empty-output guard: sha256-of-nothing = automatic FAIL, never identity.
- invocation parity: consult+test for BOTH hosts, matching Bocher's gate.
- no command-substitution capture: pipe to file/hash directly.

Additional per-kernel controls:
- coverage guard: sum of all extracted unit bytes must equal total stream
  bytes (no byte lost during split). FAIL if not.
- kernel-count guard: expected counts per generator recorded; mismatch = FAIL
  (a generator emitting fewer kernels than expected is a regression the
  whole-stream gate might miss if compensated by extra epilogue bytes).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Generators to test. Matches Bocher's emission_gate.sh triple.
GENERATORS = ["blas", "fused", "llama"]

# Expected kernel counts (from initial baseline scan 2026-09-02). If a run
# emits fewer, kernel-count guard flags it. Not upper-bound; adding kernels
# is not a regression, removing is.
EXPECTED_KERNEL_MIN = {"blas": 2, "fused": 3, "llama": 16}

# sha256 of empty string — automatic FAIL if any unit hashes to this
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Kernel signature regex: matches `__global__ void k_NAME(...)` at line start,
# optionally with __launch_bounds__() or __restrict__ noise. Captures NAME.
KERNEL_SIG_RE = re.compile(
    r"^__global__\s+void\s+(k_[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def emit_stream(host: str, generator: str, repo_root: Path) -> bytes:
    """Invoke {swipl|swilgt} to emit generator's .cu stream. Returns bytes.
    Matches Bocher's emission_gate.sh invocation exactly: -q, consult+test,
    stderr suppressed, output captured to file (not command-sub) to preserve
    trailing newlines.
    """
    env = os.environ.copy()
    if host == "swilgt":
        env.setdefault(
            "LOGTALKHOME", "/run/current-system/sw/share/logtalk-3.101.0-stable"
        )
        env.setdefault("LOGTALKUSER", "/tmp/lgt-user")
    cmd = [
        host,
        "-q",
        "-g",
        f"consult('generators/generate_{generator}_kernels.pl'), test, halt",
    ]
    out = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        check=False,
    )
    # stderr is suppressed by Bocher's gate; we don't hash it. stdout only.
    return out.stdout


def split_units(stream: bytes) -> list[tuple[str, bytes]]:
    """Split a .cu stream into (unit_name, bytes) units.

    Kernel units: named `<generator-derived>::k_NAME`, one per matched
      `__global__ void k_NAME(...)` block. Body includes signature line
      through matching closing brace, inclusive of trailing newlines
      before the next unit boundary or end-of-stream.
    Non-kernel units: `_preamble` (bytes before the first kernel),
      `_gap_<N>` (bytes between kernels — comments, blank lines,
      section markers), `_epilogue` (bytes after the last kernel —
      typically `extern "C"` C-bindings + closing brace).

    Coverage invariant: sum(len(unit_bytes)) == len(stream). Callers
    verify this and FAIL on mismatch.
    """
    text = stream.decode("utf-8", errors="replace")

    # Find all kernel signature start positions
    sig_matches = list(KERNEL_SIG_RE.finditer(text))
    if not sig_matches:
        # No kernels detected — whole thing is one unit
        return [("_no_kernels", stream)]

    units: list[tuple[str, bytes]] = []

    # _preamble: everything before the first kernel
    first_start = sig_matches[0].start()
    if first_start > 0:
        units.append(("_preamble", text[:first_start].encode("utf-8")))

    # For each kernel: find its closing brace via brace-counting
    for i, sig_match in enumerate(sig_matches):
        kernel_name = sig_match.group(1)
        # Find the opening brace of the function body — should be right after
        # the parameter list. Simple approach: find first `{` after sig start.
        body_open = text.find("{", sig_match.start())
        if body_open == -1:
            # Malformed — treat rest of stream as this "kernel"
            units.append((kernel_name, text[sig_match.start():].encode("utf-8")))
            break
        # Count braces to find matching close
        depth = 0
        pos = body_open
        while pos < len(text):
            c = text[pos]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        if depth != 0:
            # Unbalanced — take rest of stream
            body_end = len(text)
        else:
            # Include closing brace, then advance past any trailing whitespace
            # up to but not including the next kernel signature (or EOF)
            body_end = pos + 1

        # Extend body_end to include trailing whitespace/newlines up to the
        # next kernel signature (so _gap regions capture COMMENTS between
        # kernels, not trailing blank lines that belong to the previous kernel)
        # Actually: to keep unit assignment stable and easy to reason about,
        # trim kernel body strictly to sig-start .. matched-brace inclusive.
        # Anything after (blank lines, comments) becomes _gap_N.
        unit_bytes = text[sig_match.start():body_end].encode("utf-8")
        units.append((kernel_name, unit_bytes))

        # _gap_N: bytes between end of this kernel and start of next
        gap_start = body_end
        if i + 1 < len(sig_matches):
            gap_end = sig_matches[i + 1].start()
        else:
            gap_end = len(text)
        if gap_end > gap_start:
            gap_content = text[gap_start:gap_end].encode("utf-8")
            # If this is the last kernel, name the trailing chunk _epilogue
            # (that's the usual pattern: kernels then extern "C" C-bindings)
            gap_name = "_epilogue" if i + 1 == len(sig_matches) else f"_gap_{i}"
            units.append((gap_name, gap_content))

    return units


def coverage_check(units: list[tuple[str, bytes]], stream: bytes) -> bool:
    """Verify sum of unit bytes equals stream bytes. Empty-output-class
    guard extended to per-kernel."""
    total = sum(len(b) for _, b in units)
    return total == len(stream)


def hash_units(units: list[tuple[str, bytes]]) -> dict[str, str]:
    """Return {unit_name: sha256}. Flags any empty-hashing units."""
    return {name: sha256(b) for name, b in units}


def snapshot_one_host(host: str, repo_root: Path, generators: list[str]) -> dict:
    """Emit all generators from one host, return per-kernel hash matrix."""
    result: dict = {"host": host, "generators": {}}
    for g in generators:
        stream = emit_stream(host, g, repo_root)
        units = split_units(stream)
        assert coverage_check(units, stream), (
            f"COVERAGE FAIL for {host}/{g}: sum(unit_bytes) != stream_bytes"
        )
        hashes = hash_units(units)
        kernel_units = [(n, b) for n, b in units if not n.startswith("_")]
        result["generators"][g] = {
            "stream_bytes": len(stream),
            "stream_sha256": sha256(stream),
            "unit_count": len(units),
            "kernel_count": len(kernel_units),
            "units": [
                {"name": n, "bytes": len(b), "sha256": hashes[n]}
                for n, b in units
            ],
        }
    return result


def diff_snapshots(a: dict, b: dict) -> dict:
    """Compare two snapshots. Return per-kernel PASS/FAIL matrix."""
    diff: dict = {
        "host_a": a["host"],
        "host_b": b["host"],
        "generators": {},
        "pass_count": 0,
        "fail_count": 0,
        "unit_pass_count": 0,
        "unit_fail_count": 0,
    }
    for g in a["generators"]:
        if g not in b["generators"]:
            diff["generators"][g] = {"status": "MISSING_IN_B"}
            diff["fail_count"] += 1
            continue
        ga, gb = a["generators"][g], b["generators"][g]
        # Whole-stream sha (Bocher's floor)
        stream_pass = ga["stream_sha256"] == gb["stream_sha256"]
        # Per-unit matrix
        units_a = {u["name"]: u for u in ga["units"]}
        units_b = {u["name"]: u for u in gb["units"]}
        all_names = sorted(set(units_a) | set(units_b))
        unit_results = []
        g_pass = 0
        g_fail = 0
        for n in all_names:
            ua = units_a.get(n)
            ub = units_b.get(n)
            if ua is None:
                unit_results.append({"name": n, "status": "MISSING_IN_A"})
                g_fail += 1
            elif ub is None:
                unit_results.append({"name": n, "status": "MISSING_IN_B"})
                g_fail += 1
            elif ua["sha256"] == EMPTY_SHA or ub["sha256"] == EMPTY_SHA:
                unit_results.append(
                    {
                        "name": n,
                        "status": "FAIL_EMPTY",
                        "detail": "unit hashed to sha256-of-nothing — harness or emit failure",
                    }
                )
                g_fail += 1
            elif ua["sha256"] == ub["sha256"]:
                unit_results.append(
                    {"name": n, "status": "PASS", "sha256": ua["sha256"]}
                )
                g_pass += 1
            else:
                unit_results.append(
                    {
                        "name": n,
                        "status": "FAIL_DIFFER",
                        "sha256_a": ua["sha256"],
                        "sha256_b": ub["sha256"],
                        "bytes_a": ua["bytes"],
                        "bytes_b": ub["bytes"],
                    }
                )
                g_fail += 1
        # Kernel-count guard
        kernel_count_pass = ga["kernel_count"] == gb["kernel_count"]
        expected_min = EXPECTED_KERNEL_MIN.get(g, 0)
        kernel_min_pass = (
            ga["kernel_count"] >= expected_min and gb["kernel_count"] >= expected_min
        )
        diff["generators"][g] = {
            "stream_pass": stream_pass,
            "stream_sha_a": ga["stream_sha256"],
            "stream_sha_b": gb["stream_sha256"],
            "kernel_count_pass": kernel_count_pass,
            "kernel_count_a": ga["kernel_count"],
            "kernel_count_b": gb["kernel_count"],
            "expected_kernel_min": expected_min,
            "kernel_count_min_pass": kernel_min_pass,
            "unit_pass": g_pass,
            "unit_fail": g_fail,
            "units": unit_results,
        }
        if stream_pass and g_fail == 0 and kernel_count_pass and kernel_min_pass:
            diff["pass_count"] += 1
        else:
            diff["fail_count"] += 1
        diff["unit_pass_count"] += g_pass
        diff["unit_fail_count"] += g_fail
    return diff


def print_verdict(diff: dict) -> int:
    """Print human-readable verdict. Return exit code: 0 all-pass, 1 any-fail."""
    print("=" * 60)
    print(f"  PER-KERNEL EMISSION BYTE-IDENTITY MATRIX")
    print(f"  host A: {diff['host_a']}   host B: {diff['host_b']}")
    print("=" * 60)
    for g, gr in diff["generators"].items():
        if gr.get("status") == "MISSING_IN_B":
            print(f"  {g}: MISSING from host B")
            continue
        color_stream = "🟢" if gr["stream_pass"] else "🔴"
        color_kc = "🟢" if gr["kernel_count_pass"] else "🔴"
        color_units = "🟢" if gr["unit_fail"] == 0 else "🔴"
        print(f"\n  --- generator: {g} ---")
        print(f"    {color_stream} whole-stream: {gr['stream_sha_a'][:12]} == {gr['stream_sha_b'][:12]}? {gr['stream_pass']}")
        print(f"    {color_kc} kernel-count: A={gr['kernel_count_a']}, B={gr['kernel_count_b']}, expected>={gr['expected_kernel_min']}")
        print(f"    {color_units} units: {gr['unit_pass']} PASS, {gr['unit_fail']} FAIL")
        for u in gr["units"]:
            if u["status"] == "PASS":
                print(f"      🟢 PASS  {u['name']}  ({u['sha256'][:12]})")
            elif u["status"] == "FAIL_DIFFER":
                print(f"      🔴 FAIL  {u['name']}  A={u['sha256_a'][:12]} ({u['bytes_a']}b) != B={u['sha256_b'][:12]} ({u['bytes_b']}b)")
            elif u["status"] == "FAIL_EMPTY":
                print(f"      🔴 EMPTY {u['name']}  hashed to sha256-of-nothing")
            else:
                print(f"      🔴 {u['status']}  {u['name']}")
    print()
    print("=" * 60)
    print(f"  MATRIX: {diff['pass_count']}/{diff['pass_count']+diff['fail_count']} generators pass")
    print(f"  UNITS: {diff['unit_pass_count']}/{diff['unit_pass_count']+diff['unit_fail_count']} units byte-identical")
    if diff["fail_count"] == 0:
        print("  *** ALL PASS — byte-identity preserved across hosts ***")
        return 0
    print("  *** FAIL — divergences localized per unit above ***")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--generators", nargs="+", default=GENERATORS, help="subset to test"
    )
    ap.add_argument("--host-a", default="swipl", help="baseline host (default swipl)")
    ap.add_argument("--host-b", default="swilgt", help="variant host (default swilgt)")
    ap.add_argument(
        "--repo-root",
        default=None,
        help="bpd-substrate repo root (default: two levels up from this script)",
    )
    ap.add_argument("--json", help="write structured results JSON to this path")
    ap.add_argument(
        "--snapshot-only",
        choices=["swipl", "swilgt"],
        help="emit one host + dump snapshot to --json, no diff",
    )
    args = ap.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        # this script lives in migration/logtalk/; repo root is two levels up
        repo_root = Path(__file__).resolve().parent.parent.parent

    if args.snapshot_only:
        snap = snapshot_one_host(args.snapshot_only, repo_root, args.generators)
        if args.json:
            Path(args.json).write_text(json.dumps(snap, indent=2))
            print(f"wrote snapshot to {args.json}")
        else:
            print(json.dumps(snap, indent=2))
        return 0

    a = snapshot_one_host(args.host_a, repo_root, args.generators)
    b = snapshot_one_host(args.host_b, repo_root, args.generators)
    diff = diff_snapshots(a, b)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"snapshot_a": a, "snapshot_b": b, "diff": diff}, indent=2)
        )
    return print_verdict(diff)


if __name__ == "__main__":
    sys.exit(main())
