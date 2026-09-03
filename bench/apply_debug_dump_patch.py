#!/usr/bin/env python3
"""Patch llama.cpp's common/debug.cpp to dump binary tensors when LLAMA_DUMP_DIR is set.

★ WHY THIS IS A SECOND PATCHER, not a retarget of apply_eval_callback_patch.py:

Upstream refactored the per-tensor callback OUT of examples/eval-callback/
eval-callback.cpp (now an 88-line driver) INTO common/debug.cpp, where it lives
as `common_debug_cb_eval`.  The old patcher's anchors did not merely move — the
code they anchored on is in a different translation unit with a different
surrounding structure.  Retargeting one tool at two genuinely different files
would give it two sets of anchors and one name, which is how a tool becomes
untrustworthy.  The old patcher stays valid for pre-refactor checkouts.

★ THE HOOK POINT, read from b9518's source:

`common_debug_cb_eval` already materialises tensor data to a host pointer —
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) { resize; ggml_backend_tensor_get(...); }
— for exactly the reason we need it.  The dump goes after that.

★ AND IT MUST SIT OUTSIDE THE EXISTING PRINT'S GATE.  That print is conditioned
on `matches_filter && !ggml_is_quantized(t->type)`.  A quantized model
(tinyllama-q8_0) hits quantized tensors constantly, and a filtered model hits
the filter — so a dump inside that gate would capture A SUBSET WHILE LOOKING
LIKE A COMPLETE TRACE.  That is a confident-wrong-answer generator: the bisect
would localise divergence to the first layer it happened to record.

Layout per .bin (unchanged from the original patcher, so fixtures stay readable
by the same consumers):
    uint32 dtype_code · uint32 n_dims(4) · int64 ne[4] · size_t nb[4]
    uint64 n_bytes · raw bytes
plus manifest.tsv listing them in execution order.
"""
import os
import re
import sys

if len(sys.argv) != 2:
    sys.exit("usage: apply_debug_dump_patch.py <path-to-common/debug.cpp>")
SRC = sys.argv[1]
if not os.path.isfile(SRC):
    sys.exit("ERROR: not a file: %s" % SRC)
content = open(SRC).read()

if "LLAMA_DUMP_DIR" in content:
    sys.exit("ERROR: already patched (LLAMA_DUMP_DIR present); refusing to double-apply")

# ── 1. includes: anchor on the LAST #include, a structural invariant ──────
NEEDED = ["<cstdlib>", "<cstring>", "<cstdio>", "<sys/stat.h>", "<atomic>"]
incs = list(re.finditer(r'^#include [<"][^>"]+[>"]\s*$', content, re.M))
if not incs:
    sys.exit("ERROR: no #include lines found; this is not the expected source")
end = incs[-1].end()
add = "".join("\n#include %s" % h for h in NEEDED if ("#include %s" % h) not in content)
content = content[:end] + add + content[end:]

# ── 2. the dump helper, inserted before the callback that will call it ────
ANCHOR_FN = "bool common_debug_cb_eval("
if ANCHOR_FN not in content:
    sys.exit("ERROR: common_debug_cb_eval not found; upstream structure has changed again")

HELPER = r'''
// ── LLAMA_DUMP_DIR: binary per-tensor snapshots for the BPD correctness harness ──
// Writes every visited tensor, in execution order, plus a manifest.  Deliberately
// NOT gated on the print filter or on !is_quantized: a partial trace that looks
// complete would localise divergence to the first layer it happened to record.
static void bpd_dump_tensor(const uint8_t * data, const struct ggml_tensor * t) {
    const char * dir = std::getenv("LLAMA_DUMP_DIR");
    if (!dir || !data) return;
    static std::atomic<int> seq{0};
    const int idx = seq++;
    ::mkdir(dir, 0755);
    char path[1024];
    char safe[256];
    std::snprintf(safe, sizeof(safe), "%s", t->name);
    for (char * p = safe; *p; ++p) if (*p == '/' || *p == ' ') *p = '_';
    // %04d, not %06d: bench/llama_fixture_loader.py globs f"{idx:04d}_*.bin".
    // The loader is the established consumer; the producer conforms to it.
    std::snprintf(path, sizeof(path), "%s/%04d_%s.bin", dir, idx, safe);
    FILE * fp = std::fopen(path, "wb");
    if (!fp) return;
    const uint32_t dtype_code = (uint32_t) t->type;
    const uint32_t n_dims = 4;
    const uint64_t n_bytes = (uint64_t) ggml_nbytes(t);
    std::fwrite(&dtype_code, sizeof(dtype_code), 1, fp);
    std::fwrite(&n_dims,     sizeof(n_dims),     1, fp);
    std::fwrite(t->ne,       sizeof(int64_t),    4, fp);
    std::fwrite(t->nb,       sizeof(size_t),     4, fp);
    std::fwrite(&n_bytes,    sizeof(n_bytes),    1, fp);
    std::fwrite(data,        1, (size_t) n_bytes, fp);
    std::fclose(fp);
    char mpath[1100];
    std::snprintf(mpath, sizeof(mpath), "%s/manifest.tsv", dir);
    FILE * mfp = std::fopen(mpath, "a");
    if (!mfp) return;
    // Manifest schema is the loader's, not ours:
    //     idx \t name \t op_desc \t dtype_NAME \t dims  [\t src_indices]
    // It parses parts[:5] and requires >=5 columns, and it wants a dtype NAME
    // (f32/f16/q8_0) rather than the numeric ggml code.
    const char * dname = ggml_type_name(t->type);
    std::fprintf(mfp, "%04d\t%s\t%s\t%s\t%lld,%lld,%lld,%lld\n",
                 idx, t->name, ggml_op_desc(t) ? ggml_op_desc(t) : "op",
                 dname ? dname : "unknown",
                 (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2], (long long) t->ne[3]);
    std::fclose(mfp);
}

'''
content = content.replace(ANCHOR_FN, HELPER + ANCHOR_FN, 1)

# ── 3. the call site: after materialisation, OUTSIDE the print's gate ─────
ANCHOR_CALL = "    if (!ggml_is_quantized(t->type) && matches_filter) {"
if ANCHOR_CALL not in content:
    sys.exit("ERROR: materialisation block not found; upstream structure has changed again")
CALL = ('    {\n'
        '        const uint8_t * dump_data = is_host ? (const uint8_t *) t->data\n'
        '                                            : pimpl->data.data();\n'
        '        bpd_dump_tensor(dump_data, t);\n'
        '    }\n\n')
content = content.replace(ANCHOR_CALL, CALL + ANCHOR_CALL, 1)

open(SRC, "w").write(content)
print("patched: %s" % SRC)
