#!/bin/bash
# build_eval_callback.sh — build a patched llama.cpp eval-callback that dumps
# binary tensors when LLAMA_DUMP_DIR is set.
#
# Output: <llama_cpp_dir>/build/bin/llama-eval-callback
#
# This is the substrate-design-canonical reference build for the LlamaTov
# correctness harness. Anyone running the harness should build their own
# copy of this tool with this exact patch, ensuring fixtures are
# byte-comparable across verifiers.

set -e

if [ -z "$1" ]; then
    echo "usage: $0 <path-to-llama.cpp-source>"
    echo ""
    echo "  Expects llama.cpp source at the given path. The script will:"
    echo "    1. Apply our binary-dump patch to examples/eval-callback/eval-callback.cpp"
    echo "    2. Configure cmake (CPU-only, no CUDA, no BLAS, no CURL)"
    echo "    3. Build llama-eval-callback target"
    echo ""
    echo "  The resulting binary will dump binary tensor snapshots to"
    echo "  LLAMA_DUMP_DIR for each ggml operation in the inference graph."
    exit 2
fi

LLAMA_CPP_DIR="$1"
# Resolve OUR paths before the cd below; $0 is relative and the cd would break it.
BENCH_DIR="$(cd "$(dirname "$0")/../../bench" && pwd)"

# ── ERA DETECTION ─────────────────────────────────────────────────────────
# Upstream moved the per-tensor callback OUT of examples/eval-callback/
# eval-callback.cpp INTO common/debug.cpp (it is `common_debug_cb_eval` there).
# The two patchers are NOT one tool parameterised: they anchor on disjoint
# symbols in different translation units, and the newer one must place its dump
# OUTSIDE a quantized/filter gate the older file never had.
#
# The era varies at the CHECKOUT level, so the selection lives HERE -- in the
# thing that knows about checkouts -- and each patcher stays simple and honest
# about the single era it patches.
#
# Fail LOUD on an unrecognised structure: a third refactor must not silently
# apply the wrong patcher to a file it half-matches.
NEW_TARGET="$LLAMA_CPP_DIR/common/debug.cpp"
OLD_TARGET="$LLAMA_CPP_DIR/examples/eval-callback/eval-callback.cpp"

if [ -f "$NEW_TARGET" ] && grep -q "common_debug_cb_eval" "$NEW_TARGET" 2>/dev/null; then
    PATCHER="$BENCH_DIR/apply_debug_dump_patch.py"
    PATCH_TARGET="$NEW_TARGET"
    ERA="post-refactor (callback in common/debug.cpp)"
elif [ -f "$OLD_TARGET" ] && grep -q "ggml_debug" "$OLD_TARGET" 2>/dev/null; then
    PATCHER="$BENCH_DIR/apply_eval_callback_patch.py"
    PATCH_TARGET="$OLD_TARGET"
    ERA="pre-refactor (callback in examples/eval-callback)"
else
    echo "error: cannot identify the eval-callback structure in $LLAMA_CPP_DIR" >&2
    echo "  looked for common_debug_cb_eval in $NEW_TARGET" >&2
    echo "  and for ggml_debug in $OLD_TARGET" >&2
    echo "  upstream may have refactored again; a new patcher is needed rather" >&2
    echo "  than forcing an existing one onto a structure it does not match." >&2
    exit 1
fi
echo "[build] era: $ERA"
echo "[build] patcher: $PATCHER"
echo "[build] target:  $PATCH_TARGET"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "error: $LLAMA_CPP_DIR does not exist"
    exit 1
fi
if [ ! -f "$PATCHER" ]; then
    echo "error: patcher not found at $PATCHER"
    exit 1
fi

cd "$LLAMA_CPP_DIR"
echo "[build] applying binary-dump patch to eval-callback.cpp"
# The patcher hardcodes /tmp/llama_cpp_test as the path; override via env or symlink.
# For portability we'll temporarily symlink the user's llama.cpp into the expected path.
PATCH_TARGET="$LLAMA_CPP_DIR/examples/eval-callback/eval-callback.cpp"
if [ ! -f "$PATCH_TARGET" ]; then
    echo "error: $PATCH_TARGET not found (is this really llama.cpp source?)"
    exit 1
fi

# Modify the patcher to accept a path argument, OR copy the source to where it expects.
# Simplest: edit the patcher's hardcoded path inline.
python3 - <<PYEOF
import re
with open("$PATCHER") as f:
    src = f.read()
src = src.replace('/tmp/llama_cpp_test/examples/eval-callback/eval-callback.cpp',
                   '$PATCH_TARGET')
with open('/tmp/_patcher_local.py', 'w') as f:
    f.write(src)
PYEOF
python3 /tmp/_patcher_local.py

echo "[build] configuring cmake (CPU only)"
mkdir -p build && cd build
cmake .. \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "[build] building llama-eval-callback target"
make -j$(nproc) llama-eval-callback

echo ""
echo "[done] built: $LLAMA_CPP_DIR/build/bin/llama-eval-callback"
echo ""
echo "  usage: LLAMA_DUMP_DIR=/tmp/my_fixture \\"
echo "         $LLAMA_CPP_DIR/build/bin/llama-eval-callback \\"
echo "         -m /path/to/llama3.2-1b.gguf \\"
echo "         -p 'Hello, my name is' -n 1 --temp 0 --seed 42 -c 64 -t 2"
