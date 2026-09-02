"""op_mapping_parser.py — parse dashboard/OP_MAPPING.md as canonical mapping table.

Per Mavdil's 23e7b241 endorsement of (C) doc-as-truth: the OP_MAPPING.md file
is the single source of truth for op assignments. The dashboard parses it at
startup (or per-request) and uses it as the mapping between the runtime axis
(congruence_status.json's kernels) and the migration axis (emit_diff_matrix
JSON's units).

CRITICAL POLARITY HANDLING (Mavdil's 304cf403 validator-caveat):
The table cells that carry em-dash "none" markers (e.g. "— *(runtime-only)*",
"— *(no runtime cell)*", "— *(none today; a `gelu_tanh_cpu` would give one)*")
name kernels EXPLICITLY AS ABSENT, not present. A pattern-only parser that
matches backticked names inside those cells would false-positive them as
present. Solution: BEFORE extracting kernel names from a cell, check if the
cell begins with the em-dash "—" (U+2014) — if so, treat as ABSENT (empty
kernel list), regardless of any backticked names inside for prose-context.

MACHINE-COMPLETENESS PRINCIPLE (Mavdil's c5ad985 lesson banked):
Doc-as-truth requires the doc to be machine-complete, not merely human-
complete. Any kernel that lives only in prose (not in a table row) will be
invisible to this parser. If parse() reports fewer op rows than expected,
the doc is machine-incomplete — the FIX is on the doc side (promote prose
to table rows), not on the parser side (add prose-parsing).

Sanity check hook: validate() cross-checks against a set of runtime + emitted
kernel names; reports missing-from-doc + em-dash-cited-not-in-json (which is
the honest gel_tanh_cpu case — cited as absent, not present).
"""
import re
from pathlib import Path


# The em-dash character used in "— *(runtime-only)*" markers.
EM_DASH = "\u2014"


def _extract_names(cell_text):
    """Extract backticked kernel names from a table cell, HONORING polarity.
    If the cell begins with an em-dash (—), it's an EXPLICIT ABSENCE marker
    — return empty list regardless of any backticked names inside its prose.
    Otherwise return all backticked identifier names.
    """
    stripped = cell_text.strip()
    if stripped.startswith(EM_DASH):
        return []
    # Backticked names: `foo_bar` or `k_foo`
    return re.findall(r"\`([a-z][a-z0-9_]*)\`", cell_text)


def parse(path=None):
    """Parse OP_MAPPING.md into a canonical mapping dict:
      { op_string: {'runtime': [kernel_names], 'emitted': [kernel_names]} }
    """
    if path is None:
        path = Path(__file__).resolve().parent / "OP_MAPPING.md"
    text = Path(path).read_text()

    # Substrate — parse table rows: | `op` | runtime cell | emitted cell |
    # Skip separator rows (|---|---|---|) and header row.
    mapping = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != 3:
            continue
        # Skip header + separator rows
        if cells[0] in ("canonical `op`",):
            continue
        if all(re.fullmatch(r"-+", c) for c in cells):
            continue
        # First cell: `op_name` backticked
        op_match = re.match(r"\`([a-z][a-z0-9_]*)\`", cells[0])
        if not op_match:
            continue
        op = op_match.group(1)
        mapping[op] = {
            "runtime": _extract_names(cells[1]),
            "emitted": _extract_names(cells[2]),
        }
    return mapping


def build_reverse_index(mapping):
    """Given the mapping {op: {runtime, emitted}}, build reverse lookups:
      runtime_to_op: {kernel_name: op_string}
      emitted_to_op: {kernel_name: op_string}
    So the dashboard can look up "what op does this row belong to?" in O(1).
    """
    runtime_to_op = {}
    emitted_to_op = {}
    for op, cells in mapping.items():
        for k in cells["runtime"]:
            if k in runtime_to_op:
                # Substrate — duplicate: same kernel in two ops. Would be a
                # contract-integrity violation. Preserve first, flag for humans.
                continue
            runtime_to_op[k] = op
        for k in cells["emitted"]:
            if k in emitted_to_op:
                continue
            emitted_to_op[k] = op
    return runtime_to_op, emitted_to_op


def join_state(op, mapping):
    """Return 'both' / 'runtime-only' / 'emitted-only' for an op."""
    cells = mapping.get(op, {"runtime": [], "emitted": []})
    has_r = bool(cells["runtime"])
    has_e = bool(cells["emitted"])
    if has_r and has_e:
        return "both"
    if has_r:
        return "runtime-only"
    if has_e:
        return "emitted-only"
    return "empty"  # shouldn't happen in a valid contract


def validate(mapping, runtime_kernel_names=None, emitted_kernel_names=None):
    """Optional sanity check. Returns list of issues (empty = clean).
    Each issue is (severity, msg): severity in {'ERROR', 'WARN', 'INFO'}.
    """
    issues = []
    if runtime_kernel_names is not None:
        cited_runtime = set()
        for cells in mapping.values():
            cited_runtime.update(cells["runtime"])
        missing_in_doc = set(runtime_kernel_names) - cited_runtime
        cited_absent = cited_runtime - set(runtime_kernel_names)
        for k in sorted(missing_in_doc):
            issues.append(("ERROR",
                f"runtime kernel {k!r} is in JSON but NOT cited in OP_MAPPING.md"))
        for k in sorted(cited_absent):
            issues.append(("INFO",
                f"runtime kernel {k!r} is CITED in OP_MAPPING.md but not in JSON "
                f"(if in em-dash absence-marker cell, this is expected)"))
    if emitted_kernel_names is not None:
        cited_emitted = set()
        for cells in mapping.values():
            cited_emitted.update(cells["emitted"])
        missing_in_doc = set(emitted_kernel_names) - cited_emitted
        cited_absent = cited_emitted - set(emitted_kernel_names)
        for k in sorted(missing_in_doc):
            issues.append(("ERROR",
                f"emitted kernel {k!r} is in matrix but NOT cited in OP_MAPPING.md"))
        for k in sorted(cited_absent):
            issues.append(("INFO",
                f"emitted kernel {k!r} is CITED in OP_MAPPING.md but not in matrix"))
    return issues


if __name__ == "__main__":
    import json, sys
    m = parse()
    print(json.dumps(m, indent=2))
    print()
    print(f"Parsed {len(m)} op rows.")
    both = sum(1 for op in m if join_state(op, m) == "both")
    r_only = sum(1 for op in m if join_state(op, m) == "runtime-only")
    e_only = sum(1 for op in m if join_state(op, m) == "emitted-only")
    print(f"  both: {both}, runtime-only: {r_only}, emitted-only: {e_only}")
