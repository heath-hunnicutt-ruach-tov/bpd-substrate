#!/usr/bin/env bash
# GATE-2 BATCH: run derivation-equivalence gate over ALL migrated leaves.
# The parametric framework: one gate, applied to every leaf in logtalk/lib/.
#
# For each .lgt file in logtalk/lib/, extracts the leaf name, runs
# derivation_gate.sh against it, collects results.
#
# CONTROLS (the checker-taxonomy applied):
#  - Empty-batch guard: if no .lgt files exist, FAIL (don't report "all pass" on zero)
#  - Per-leaf verdicts collected into summary
#  - Non-zero exit on ANY leaf failure
#  - Canary: if --canary, injects a known-bad leaf to verify the gate detects it
#
# Usage:
#   ./gate2_batch.sh                    # run all migrated leaves
#   ./gate2_batch.sh --canary           # run all + inject canary
#   ./gate2_batch.sh arch_params        # run single named leaf
#
# Output: per-leaf PASS/FAIL + summary line + exit code
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBSTRATE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGTALK_LIB="${SUBSTRATE_DIR}/logtalk/lib"

canary=false
specific=""
for arg in "$@"; do
  case "$arg" in
    --canary) canary=true ;;
    *) specific="$arg" ;;
  esac
done

# Empty-batch guard
if [ -n "$specific" ]; then
  leaves=("$specific")
elif [ -d "$LOGTALK_LIB" ]; then
  mapfile -t leaves < <(ls "$LOGTALK_LIB"/*.lgt 2>/dev/null | xargs -I{} basename {} .lgt | sort)
else
  echo "GATE-2 BATCH: FAIL (no logtalk/lib/ directory)"
  exit 1
fi

if [ ${#leaves[@]} -eq 0 ]; then
  echo "GATE-2 BATCH: FAIL (no .lgt files found — empty batch, not all-pass)"
  exit 1
fi

echo "============================================================"
echo "  GATE-2 BATCH: derivation-equivalence over ${#leaves[@]} leaves"
echo "  substrate: $SUBSTRATE_DIR"
echo "  timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo

total=0; passed=0; failed=0; warned=0

for leaf in "${leaves[@]}"; do
  total=$((total + 1))
  echo "--- leaf: $leaf ---"

  # Run the per-leaf derivation gate
  output=$(cd "$SUBSTRATE_DIR" && bash "$SCRIPT_DIR/derivation_gate.sh" "$leaf" 2>&1)
  rc=$?
  echo "$output" | sed 's/^/  /'

  if [ $rc -eq 0 ]; then
    # Check for warns in the output
    warns=$(echo "$output" | grep -c "WARN-EMPTY")
    if [ "$warns" -gt 0 ]; then
      warned=$((warned + warns))
    fi
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi
  echo
done

# Canary injection: create a deliberately wrong .lgt, gate it, verify FAIL
if $canary; then
  echo "--- CANARY: injecting known-bad leaf ---"
  canary_file="${LOGTALK_LIB}/_canary_gate2.lgt"
  # Create a leaf that exports a predicate with DIFFERENT answers than the .pl
  cat > "$canary_file" << 'CANARY_EOF'
:- object(_canary_gate2).
  :- public(canary_fact/1).
  canary_fact(wrong_answer_for_gate2_canary).
:- end_object.
CANARY_EOF
  # Create matching .pl with DIFFERENT content
  canary_pl="${SUBSTRATE_DIR}/lib/_canary_gate2.pl"
  cat > "$canary_pl" << 'PL_EOF'
:- module(_canary_gate2, [canary_fact/1]).
canary_fact(correct_answer_for_gate2_canary).
PL_EOF

  canary_output=$(cd "$SUBSTRATE_DIR" && bash "$SCRIPT_DIR/derivation_gate.sh" "_canary_gate2" 2>&1)
  canary_rc=$?
  echo "$canary_output" | sed 's/^/  /'

  # Clean up canary files
  rm -f "$canary_file" "$canary_pl"

  if [ $canary_rc -ne 0 ]; then
    echo "  CANARY: PASS (gate correctly detected injected divergence)"
  else
    echo "  CANARY: FAIL (gate did NOT detect injected divergence — HARNESS BROKEN)"
    failed=$((failed + 1))
  fi
  echo
fi

echo "============================================================"
echo "  GATE-2 BATCH SUMMARY"
echo "    total leaves:  $total"
echo "    passed:        $passed"
echo "    failed:        $failed"
echo "    empty-warns:   $warned"
if [ $failed -eq 0 ]; then
  echo "  *** ALL LEAVES DERIVATION-EQUIVALENT ***"
else
  echo "  *** BATCH FAILED — $failed leaf(s) divergent ***"
fi
echo "============================================================"

exit $failed
