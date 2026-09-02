#!/usr/bin/env bash
# DERIVATION-EQUIVALENCE GATE: for a migrated leaf, enumerate every
# exported predicate's complete answer set under (a) the original module
# in swipl and (b) the Logtalk object in swilgt; compare sha256.
# Empty-answer-set guard: a predicate yielding zero answers under BOTH
# hosts hashes identically-empty — flagged as WARN (can't distinguish
# equivalence from double-failure) unless --allow-empty.
set -u
export LOGTALKHOME="${LOGTALKHOME:-/run/current-system/sw/share/logtalk-3.101.0-stable}"
export LOGTALKUSER="${LOGTALKUSER:-/tmp/lgt-user}"
name="$1"
# Extract exports from the module directive:
exports=$(swipl -q -g "use_module('lib/${name}'), module_property(${name}, exports(E)), forall(member(P/A, E), format('~w/~w~n',[P,A])), halt" 2>/dev/null)
[ -z "$exports" ] && { echo "GATE ${name}: FAIL (cannot read exports)"; exit 1; }
fail=0; warn=0
for pa in $exports; do
  p="${pa%/*}"; a="${pa#*/}"
  # Build goal: p(A1..An) with fresh vars, enumerate + print all args
  args=$(seq -s, 1 "$a" | sed 's/[0-9]\+/A&/g')
  goal_m="use_module('lib/${name}'), forall(${p}(${args}), (write_canonical(x(${args})), nl))"
  goal_o="logtalk_load('logtalk/lib/${name}.lgt'), forall(${name}::${p}(${args}), (write_canonical(x(${args})), nl))"
  h_m=$(swipl -q -g "${goal_m}, halt" 2>/dev/null | sha256sum | awk '{print $1}')
  h_o=$(swilgt -q -g "${goal_o}, halt" 2>/dev/null | sha256sum | awk '{print $1}')
  EMPTY=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  if [ "$h_m" != "$h_o" ]; then
    echo "  ${pa}: FAIL (module ${h_m:0:12} != object ${h_o:0:12})"; fail=$((fail+1))
  elif [ "$h_m" = "$EMPTY" ]; then
    echo "  ${pa}: WARN-EMPTY (both empty — equivalent but unverifying)"; warn=$((warn+1))
  else
    echo "  ${pa}: PASS (${h_m:0:12})"
  fi
done
echo "GATE ${name}: $([ $fail -eq 0 ] && echo PASS || echo FAIL) (warns=$warn)"
exit $fail

# NOTE (2026-09-02, batch 1): WARN-EMPTY is EXPECTED for dynamic-predicate
# leaves (e.g. tensor_schema declares all exports :- dynamic; the file is
# a runtime-asserted schema with ZERO static facts — empty-both-hosts is
# correct, not a gate failure). For dynamic leaves the real equivalence
# test is load-then-assert-then-query, deferred to Gate-2 (medayek's
# framework). The WARN keeps the distinction visible: PASS-with-warns=N
# means N predicates were unverifiable-by-static-enumeration, not wrong.
# NOTE 2: INFERENCE leaves (unbounded/generative predicates, e.g.
# region_inference's infer_region/4) CANNOT be gated by complete
# enumeration — fresh-var forall diverges. They need Gate-2 bounded
# query-workload equivalence (medayek). Triage rule: enumerate only
# predicates whose answer sets are finite fact-like relations.
cpu_profile: reclassified ACTION-leaf (shells out: gcc builds, profiler runs) — deferred to Gate-2 controlled-workload; enumeration EXECUTES side effects (found when gate ran real builds)
