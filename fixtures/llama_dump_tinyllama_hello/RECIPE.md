# Fixture: per-layer trace, tinyllama-q8_0

*The `.bin` payload is **gitignored** (44 MB, 688 files). The `manifest.tsv` beside this file is
committed, and so is this recipe. **The evidence is regenerable deterministically; the payload is
not worth carrying in git.***

## ★ THE MODEL IS A SUBSTITUTION, AND IT IS IN THE DIRECTORY NAME

`tests/correctness/README.md:61` specifies **`llama3.2-1b.gguf`**. That model is **not on the
enclave** — verified by a filesystem-wide search. This fixture uses **`tinyllama-q8_0`**, which is
llama-architecture and answers the *structural* KV/prefill question.

**No comparison against any prior det-gemv figure is possible or intended.** The original trace
lived in `/tmp` and evaporated, which is why this exists at all. *The substitution is encoded in
the path — `llama_dump_TINYLLAMA_hello` — so it is visible at every reference, not only in a note
someone may not read.*

## Regeneration

```sh
# 1. patch + build (the driver picks the patcher by detecting the llama.cpp era)
bash tests/correctness/build_eval_callback.sh /path/to/llama.cpp

# 2. capture — parameters from the documented recipe, model substituted
LLAMA_DUMP_DIR=fixtures/llama_dump_tinyllama_hello \
LD_LIBRARY_PATH=<build>/bin \
CUDA_VISIBLE_DEVICES='' \
  <build>/bin/llama-eval-callback \
    -m /mnt/data/shared/models/tinyllama-q8_0.gguf \
    -p "Hello, my name is" -n 1 --temp 0 --seed 42 -c 64 -t 2
```

Deterministic: `--temp 0 --seed 42`.

## What it contains

**688 tensors, every ggml op in execution order**, 44 MB.

*The dump is deliberately **ungated**. llama.cpp's own per-tensor print is conditioned on
`matches_filter && !ggml_is_quantized(t->type)`, and `tinyllama-q8_0` is quantized — dumping
inside that gate would have captured **556 of 688** tensors while looking like a complete trace,
and a layer-bisect would then localise divergence to the first layer it happened to record.*

Block 0's 31 operations, as captured:

```
0000 embd GET_ROWS · 0001 norm-0 RMS_NORM · 0002 attn_norm-0 MUL
0003 Qcur-0 MUL_MAT · 0005 Qcur-0 ROPE · 0006 Vcur-0 MUL_MAT · 0008 Kcur-0 MUL_MAT
0010 Kcur-0 ROPE · 0012 cache_k_l0 SET_ROWS · 0014 cache_v_l0 SET_ROWS
0021 __fattn__-0 FLASH_ATTN_EXT · 0023 attn_out-0 MUL_MAT · 0024 ffn_inp-0 ADD
0025 norm-0 RMS_NORM · 0027 ffn_gate-0 MUL_MAT · 0028 ffn_up-0 MUL_MAT
0029 ffn_swiglu-0 SWIGLU · 0030 ffn_out-0 MUL_MAT · 0031 l_out-0 ADD
```

## Findings from it so far

```
embedding        max_ulp 0, n_diff 0/12288          CLEAN
end of block 0   12097/12288 differ, max_abs 4.1e-04
by layer 20      max_abs 6.4e-02
```

*`max_ulp` reads ≈2.4e9 throughout. **That is a sign-crossing artefact of the metric, not a kernel
property** — a differing pair straddling zero pins the ordered-int distance at ~2³¹ regardless of
how small the actual difference is (`-0.0002` vs `0.0002` → 2371654098, while same-sign pairs at
the same absolute difference give ~53687). Read `n_diff` and `max_abs`; not `max_ulp`.*
