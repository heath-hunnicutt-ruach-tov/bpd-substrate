#!/usr/bin/env bash
# EMISSION BYTE-IDENTITY GATE for the Logtalk migration (Track B).
# The migration invariant: same facts in, byte-identical kernels out.
# Usage: ./emission_gate.sh [swipl|swilgt]  — emits all three generator
# outputs under the chosen host and compares against the frozen baseline
# hashes below (captured 2026-09-02 from swipl, deterministic run-twice).
#
# CONTROLS (the week's checker-taxonomy applied):
#  - empty-output guard: e3b0c442... (sha256 of nothing) is an automatic
#    FAIL — a gate that hashes nothing must never report identity.
#  - invocation parity: uses consult+test for ALL generators under BOTH
#    hosts (generate_fused's initialization guard doesn't fire under
#    consult; calling test/0 explicitly is the invocation that works
#    identically everywhere).
set -u
HOST="${1:-swilgt}"
if [ "$HOST" = swilgt ]; then
  export LOGTALKHOME="${LOGTALKHOME:-/run/current-system/sw/share/logtalk-3.101.0-stable}"
  export LOGTALKUSER="${LOGTALKUSER:-/tmp/lgt-user}"
fi
declare -A BASELINE=(
  [blas]=aedb04554b5cdb3a702d206e348d9d7cf8fee13bad921238b6dd65e956d4d62b
  [fused]=b742026836d785e05f19d3fac7b20e5a8d67607fb6305e44e0f5c7b4e7b7adfb
  [llama]=a0a35e0aaf7f869268cd59f1180c74ea184cde1cf8943145f4a0e26e51830681
)
EMPTY=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
pass=0; fail=0
for g in blas fused llama; do
  # Pipe directly to sha256sum — command-substitution strips trailing
  # newlines and would corrupt the hash (caught 2026-09-02: fused ends
  # in two newlines; $() ate both, printf restored one, hash differed).
  h=$("$HOST" -q -g "consult('generators/generate_${g}_kernels.pl'), test, halt" 2>/dev/null | sha256sum | awk '{print $1}')
  if [ "$h" = "$EMPTY" ]; then
    echo "GATE $g: FAIL (EMPTY OUTPUT — harness or emit failure, not identity)"; fail=$((fail+1))
  elif [ "$h" = "${BASELINE[$g]}" ]; then
    echo "GATE $g: PASS (byte-identical to baseline)"; pass=$((pass+1))
  else
    echo "GATE $g: FAIL (hash $h != baseline ${BASELINE[$g]})"; fail=$((fail+1))
  fi
done
echo "EMISSION GATE: $pass/3 pass, $fail/3 fail (host=$HOST)"
[ $fail -eq 0 ]
