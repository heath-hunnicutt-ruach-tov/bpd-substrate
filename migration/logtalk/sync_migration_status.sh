#!/usr/bin/env bash
#
# sync_migration_status.sh — autonomous per-batch migration-axis freshness cadence.
#
# Per Iyun's 258474a6 ratification of the (7) freshness-cadence proposal:
# extend mavhir's autonomous per-batch matrix-fire discipline to also produce
# the production migration_status.json AND rsync it to enclave WHEN CLEAN.
# On DIVERGENCE, HOLD the rsync (don't ship diverged data to public dashboard)
# and print divergence details for wire-reporting.
#
# THE HONEST GATE (crown of the design per Iyun):
#   Fresh-when-clean, held-when-diverged, NEVER-silently-diverged-live.
# A diverged batch that auto-rsync'd would put diverged data on the public
# dashboard silently — the whole point of the cross-check is to CATCH
# divergence, not to publish it.
#
# Runs on nixos .116 (heath account). Assumes:
#   - /home/heath/Ruach-Tov/bpd-substrate is the working tree
#   - swipl + swilgt on PATH (verified: /run/current-system/sw/bin)
#   - enclave alias configured in ~/.ssh/config for dibbur-patch@192.168.0.68
#
# Usage:
#   ./sync_migration_status.sh                  # full cadence: fetch → matrix → (rsync if clean)
#   ./sync_migration_status.sh --no-fetch       # skip git fetch (matrix + rsync only)
#   ./sync_migration_status.sh --dry-run        # emit but don't rsync
#
# Exit codes:
#   0 = fully successful (matrix clean, rsync succeeded, dashboard live-fresh)
#   1 = divergence caught (matrix reported divergence, rsync HELD, wire-report needed)
#   2 = harness error (matrix couldn't run, environment issue)
#   3 = rsync error (matrix clean but rsync failed)
#
# Timestamp: 2026-09-02
# Attribution: Iyun 258474a6 (ratification), mavhir (design + implementation)

set -uo pipefail

REPO_ROOT="/home/heath/Ruach-Tov/bpd-substrate"
HARNESS="$REPO_ROOT/migration/logtalk/emit_diff_matrix.py"
LOCAL_JSON="/tmp/migration_status_production.json"
ENCLAVE_PATH="/home/dibbur-patch/step3-det-gemv/bpd/migration_status.json"
ENCLAVE_HOST="enclave"

FETCH=true
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --no-fetch) FETCH=false ;;
        --dry-run)  DRY_RUN=true ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

cd "$REPO_ROOT" || { log "ERROR: cannot cd to $REPO_ROOT"; exit 2; }

# Step 1: git fetch (safe — updates only .git/refs, doesn't touch working tree)
if $FETCH; then
    log "=== git fetch origin ==="
    if ! git fetch origin rtaal-1-1 2>&1 | tail -3; then
        log "WARN: git fetch failed; continuing with current HEAD"
    fi
fi

# Step 2: check for new batch commits (informational, doesn't gate execution)
CURRENT_HEAD=$(git rev-parse HEAD)
ORIGIN_HEAD=$(git rev-parse origin/rtaal-1-1 2>/dev/null || echo "unknown")
if [ "$CURRENT_HEAD" != "$ORIGIN_HEAD" ]; then
    log "note: local HEAD ($CURRENT_HEAD) != origin/rtaal-1-1 ($ORIGIN_HEAD)"
    log "note: matrix will run against CURRENT working tree, not origin"
    NEW_BATCHES=$(git log --oneline "${CURRENT_HEAD}..origin/rtaal-1-1" 2>/dev/null | wc -l)
    log "note: $NEW_BATCHES new commit(s) on origin/rtaal-1-1 since local HEAD"
    log "note: to include them: git pull --ff-only then rerun"
fi

# Step 3: fire the matrix in --production mode
log "=== emit_diff_matrix.py --production ==="
if ! "$HARNESS" --production "$LOCAL_JSON" > /tmp/sync_migration_output.log 2>&1; then
    HARNESS_EXIT=$?
    log "ERROR: emit_diff_matrix.py exited $HARNESS_EXIT"
    log "  matrix reported divergence or environment error"
    log "  HOLDING rsync (per honest-gate discipline: never ship diverged data)"
    log "  matrix output tail:"
    tail -20 /tmp/sync_migration_output.log | sed 's/^/    /'
    if [ "$HARNESS_EXIT" = "1" ]; then
        exit 1  # divergence caught
    else
        exit 2  # environment error
    fi
fi

# Step 4: parse divergence status from produced JSON (defensive double-check)
if [ ! -f "$LOCAL_JSON" ]; then
    log "ERROR: matrix ran but produced no output at $LOCAL_JSON"
    exit 2
fi

DIVERGED=$(python3 -c "
import json
d = json.load(open('$LOCAL_JSON'))
print(d.get('migration_diverged', -1))
" 2>/dev/null || echo "-1")

if [ "$DIVERGED" = "-1" ]; then
    log "ERROR: could not parse migration_diverged from $LOCAL_JSON"
    exit 2
fi

TOTAL=$(python3 -c "
import json
d = json.load(open('$LOCAL_JSON'))
print(d.get('total_kernels', -1))
" 2>/dev/null || echo "-1")

log "matrix result: $((TOTAL - DIVERGED))/$TOTAL byte-identical"

if [ "$DIVERGED" != "0" ]; then
    log "DIVERGENCE DETECTED: $DIVERGED kernel(s) diverged"
    log "  HOLDING rsync (per honest-gate: never ship diverged data live)"
    log "  local artefact: $LOCAL_JSON (inspect for wire-report)"
    exit 1
fi

# Step 5: all clean — rsync to enclave (unless --dry-run)
if $DRY_RUN; then
    log "--dry-run: would rsync $LOCAL_JSON -> $ENCLAVE_HOST:$ENCLAVE_PATH"
    log "  skipping actual rsync"
    exit 0
fi

log "=== rsync to enclave (clean run, no divergence) ==="
if rsync -avz --no-owner --no-group "$LOCAL_JSON" "$ENCLAVE_HOST:$ENCLAVE_PATH" 2>&1 | tail -5; then
    log "rsync succeeded: migration_status.json live on enclave"
    log "  dashboard reads it on next request (no restart needed)"
    exit 0
else
    log "ERROR: rsync failed"
    log "  local artefact ready at $LOCAL_JSON"
    log "  manual retry: rsync $LOCAL_JSON $ENCLAVE_HOST:$ENCLAVE_PATH"
    exit 3
fi
