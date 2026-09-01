#!/usr/bin/env python3
"""fixture_token_gate.py — e2e TOKEN-IDENTITY gate across the prompt fixtures.

THE GATE for the ultimate goal ("run llama faster than ollama, SAME output"):
runs llama-cli (greedy, temp 0, -ngl 99) on each fixture prompt for BOTH a
baseline build (stock ggml-CUDA = ollama's engine) and a variant build (e.g.
SoA-vec128-patched), and confirms IDENTICAL token sequences across ALL prompts.

This is the RIGHT gate for a GPU e2e claim: GPU-vs-GPU token identity (not the
CPU-fixture comparison — ggml-GPU uses MMQ Q8_1 activation, a different numeric
path than ggml-CPU, so CPU-bit-identity does NOT transfer to the GPU path).

  e2e_bench.py          -> is it faster? (tok/sec)
  fixture_token_gate.py -> is it the SAME model? (token identity)   <- this file
  Both must pass to claim "faster AND identical".

Usage:
  fixture_token_gate.py --baseline <stock-llama-cli> --variant <patched-llama-cli> \\
      --gguf M --prompts prompts.txt --n-predict 32
"""
import argparse, subprocess, os, sys, json, hashlib

DEFAULT_GGUF = "/mnt/data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
DRIVER = "/run/opengl-driver/lib"
# the prose fixture prompts (the 32-prompt test set). Short, diverse.
DEFAULT_PROMPTS = [
    "We the People of the United States",
    "In the beginning God created",
    "To be or not to be, that is",
    "The quick brown fox jumps over",
    "Four score and seven years ago",
    "It is a truth universally acknowledged",
    "Call me Ishmael. Some years ago",
    "When in the Course of human events",
    "I think, therefore I",
    "The mitochondria is the powerhouse",
]

import re as _re

# Cursor-animation bytes emitted by llama-cli's streaming display: sequences
# of {|,/,\,-} followed by \x08 (backspace). Deterministic model output is
# hidden inside this stream and comes out at run-timing-dependent byte offsets.
_CHROME_CURSOR = _re.compile(r'[|/\\\-]\x08')
# Per-run tok/s figures embedded in output ("[ Prompt: 994.5 t/s | Generation: 86.4 t/s ]").
_CHROME_TIMING = _re.compile(r'\[\s*Prompt:\s*[\d.]+\s*t/s\s*\|\s*Generation:\s*[\d.]+\s*t/s\s*\]')
# llama-cli's trailer, prompt-echo prefix, and interactive banner
_CHROME_TRAILERS = _re.compile(r'^(Exiting\.\.\.|>|\s|/\S+\s+.*)$', _re.MULTILINE)

def _strip_llama_chrome(text):
    """Strip non-deterministic chrome from llama-cli stdout: cursor animation
    bytes, tok/s timing figures, and interactive banners. Returns just the
    deterministic generated-token content, hashable across runs."""
    text = _CHROME_CURSOR.sub('', text)
    text = _CHROME_TIMING.sub('', text)
    text = _CHROME_TRAILERS.sub('', text)
    # Collapse whitespace runs left over from removals
    text = _re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def gen_tokens(binary, gguf, prompt, n_predict, ngl):
    """Greedy generation (temp 0). Returns the generated continuation text (deterministic).

    Chrome-filtered: llama-cli's cursor animation + tok/s timing figures are
    stripped before return so hashing this text is deterministic across runs.
    See _strip_llama_chrome() for the filter's scope.
    """
    ld = ":".join([DRIVER, os.path.dirname(binary)])
    # -st (single-turn) exits after generation, avoiding interactive mode.
    # llama.cpp at commit 7c158fb autodetects chat-template and defaults to
    # conversation mode; -no-cnv alone is insufficient. -st is authoritative:
    # runs one turn with the predefined --prompt and exits.
    cmd = [binary, "-m", gguf, "-ngl", str(ngl), "-n", str(n_predict),
           "-p", prompt, "--temp", "0", "--top-k", "1", "--seed", "0",
           "-c", "512", "--no-warmup", "-st"]
    env = dict(os.environ, LD_LIBRARY_PATH=ld)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    # Chrome-strip: remove cursor animation + tok/s figures so hash is deterministic.
    return _strip_llama_chrome(out.stdout)

def main():
    ap = argparse.ArgumentParser(description="e2e token-identity gate across fixture prompts")
    ap.add_argument("--baseline", required=True, help="stock llama-cli (ollama's engine)")
    ap.add_argument("--variant", required=True, help="patched llama-cli (the variant)")
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--prompts-file", help="file with one prompt per line (else built-in set)")
    ap.add_argument("--json", help="write per-prompt results")
    a = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if a.prompts_file:
        prompts = [l.rstrip("\n") for l in open(a.prompts_file) if l.strip()]

    print(f"=== e2e TOKEN-IDENTITY GATE: variant vs baseline, {len(prompts)} prompts, greedy ===")
    print(f"    baseline: {os.path.basename(a.baseline)}  variant: {os.path.basename(a.variant)}")
    results = []; npass = 0
    for i, p in enumerate(prompts):
        b = gen_tokens(a.baseline, a.gguf, p, a.n_predict, a.ngl)
        v = gen_tokens(a.variant, a.gguf, p, a.n_predict, a.ngl)
        bh = hashlib.sha256(b.encode()).hexdigest()[:12]
        vh = hashlib.sha256(v.encode()).hexdigest()[:12]
        ok = (b == v)
        npass += ok
        results.append({"prompt": p[:40], "match": ok, "baseline_sha": bh, "variant_sha": vh})
        print(f"  [{i+1:2d}] {'PASS' if ok else 'FAIL'}  \"{p[:40]}\"  ({bh} vs {vh})")
        if not ok:
            # show first divergence point for debugging
            for j,(cb,cv) in enumerate(zip(b,v)):
                if cb != cv:
                    print(f"        first diff @char {j}: base={b[j:j+30]!r} var={v[j:j+30]!r}")
                    break
    print(f"\n  {'='*50}")
    print(f"  GATE: {npass}/{len(prompts)} prompts token-identical")
    print(f"  {'*** PASS — variant produces IDENTICAL output to stock ***' if npass==len(prompts) else '*** FAIL — variant DIVERGES (not the same model) ***'}")
    if a.json: json.dump(results, open(a.json,"w"), indent=2)
    return 0 if npass==len(prompts) else 1

if __name__ == "__main__":
    sys.exit(main())
