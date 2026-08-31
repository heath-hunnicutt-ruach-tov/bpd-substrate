#!/usr/bin/env python3
"""logit_gate.py — stratified 6-strata decode correctness gate.

THE GATE for stratified decode coverage: runs llama-cli (greedy, temp 0)
on prompts drawn from 6 strata, comparing STOCK output vs VARIANT (e.g. SoA).
Catches the class of bug where "Hello" alone gives false confidence but 5/6
strata are RED (the June 2026 lesson).

Strata:
  1. minimal     — trivial short prompts ("Hello", "Hi")
  2. code        — programming/structured ("def fibonacci(n):")
  3. multilingual — non-English scripts ("Bonjour, comment")
  4. long_context — longer prefill (multi-sentence)
  5. repetitive   — pattern-heavy ("1 2 3 4 5 6 7 8")
  6. adversarial  — edge cases, unusual tokens

Design principles (June corpus):
  - Token-only comparison (phase 1): compare generated TEXT, not logits
  - ERROR-on-crash hardening: subprocess errors → RED, not silent skip
  - Stock-vs-variant comparison via binary path (not env flag alone)
  - Every stratum reported independently — partial pass is visible

Usage:
  logit_gate.py --baseline <stock-llama-cli> --variant <patched-llama-cli> \
      --gguf <model.gguf> --n-predict 16

Reconstructed 2026-08-31 by medayek from June 2026 spec.
Original design: medayek (Iyun built, mavhir hardened).
"""
import argparse, subprocess, os, sys, json, hashlib, time

DEFAULT_GGUF = "/mnt/data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
DRIVER = "/run/opengl-driver/lib"

# === THE 6 STRATA ===
STRATA = {
    "minimal": [
        "Hello",
        "Hi there",
        "The capital of France is",
    ],
    "code": [
        "def fibonacci(n):",
        "SELECT * FROM users WHERE",
        "int main(int argc, char** argv) {",
    ],
    "multilingual": [
        "Bonjour, comment allez-vous",
        "Guten Tag, wie geht es Ihnen",
        "こんにちは、お元気ですか",
    ],
    "long_context": [
        "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife",
        "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty",
    ],
    "repetitive": [
        "1 2 3 4 5 6 7 8 9 10",
        "the the the the the",
        "A B C D E F G H I J K L M N O P",
    ],
    "adversarial": [
        "",  # empty prompt
        "🔥💀🎯🚀",  # emoji-only
        "\n\n\n",  # whitespace-only
    ],
}


def gen_tokens(binary, gguf, prompt, n_predict, ngl, timeout=180):
    """Greedy generation. Returns (text, error_msg).
    ERROR-on-crash: never silently skip — return error string on failure."""
    ld = ":".join([DRIVER, os.path.dirname(binary)])
    cmd = [binary, "-m", gguf, "-ngl", str(ngl), "-n", str(n_predict),
           "-p", prompt if prompt else " ", "--temp", "0", "--top-k", "1",
           "--seed", "0", "-c", "512", "--no-warmup", "-no-cnv"]
    env = dict(os.environ, LD_LIBRARY_PATH=ld)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             env=env, timeout=timeout)
        if out.returncode != 0:
            return None, f"EXIT {out.returncode}: {out.stderr[:200]}"
        return out.stdout, None
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT ({timeout}s)"
    except Exception as e:
        return None, f"ERROR: {e}"


def main():
    ap = argparse.ArgumentParser(
        description="Stratified 6-strata decode correctness gate")
    ap.add_argument("--baseline", required=True,
                    help="stock llama-cli (reference)")
    ap.add_argument("--variant", required=True,
                    help="patched llama-cli (the variant under test)")
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-prompt timeout in seconds")
    ap.add_argument("--json", help="write per-prompt results to file")
    a = ap.parse_args()

    print(f"{'='*60}")
    print(f"  LOGIT GATE: 6-strata decode correctness")
    print(f"  baseline: {os.path.basename(a.baseline)}")
    print(f"  variant:  {os.path.basename(a.variant)}")
    print(f"  model:    ...{os.path.basename(a.gguf)[-30:]}")
    print(f"  n_predict={a.n_predict}  ngl={a.ngl}  timeout={a.timeout}s")
    print(f"{'='*60}\n")

    results = []
    strata_summary = {}

    for stratum_name, prompts in STRATA.items():
        stratum_pass = 0
        stratum_total = len(prompts)
        print(f"  --- stratum: {stratum_name} ({stratum_total} prompts) ---")

        for prompt in prompts:
            display = repr(prompt[:40])

            b_text, b_err = gen_tokens(
                a.baseline, a.gguf, prompt, a.n_predict, a.ngl, a.timeout)
            v_text, v_err = gen_tokens(
                a.variant, a.gguf, prompt, a.n_predict, a.ngl, a.timeout)

            if b_err:
                status = "ERROR-BASE"
                detail = b_err
            elif v_err:
                status = "ERROR-VAR"
                detail = v_err
            elif b_text == v_text:
                status = "PASS"
                detail = ""
                stratum_pass += 1
            else:
                status = "FAIL"
                # find first divergence
                detail = ""
                for j, (cb, cv) in enumerate(zip(b_text, v_text)):
                    if cb != cv:
                        detail = (f"diff @char {j}: "
                                  f"base={b_text[j:j+30]!r} "
                                  f"var={v_text[j:j+30]!r}")
                        break
                if not detail and len(b_text) != len(v_text):
                    detail = f"len diff: base={len(b_text)} var={len(v_text)}"

            color = "🟢" if status == "PASS" else "🔴"
            print(f"    {color} {status:10s} {display}")
            if detail:
                print(f"       {detail}")

            results.append({
                "stratum": stratum_name,
                "prompt": prompt[:60],
                "status": status,
                "detail": detail,
                "baseline_sha": hashlib.sha256(
                    (b_text or "").encode()).hexdigest()[:12],
                "variant_sha": hashlib.sha256(
                    (v_text or "").encode()).hexdigest()[:12],
            })

        color = "🟢" if stratum_pass == stratum_total else "🔴"
        strata_summary[stratum_name] = (stratum_pass, stratum_total)
        print(f"    {color} {stratum_name}: "
              f"{stratum_pass}/{stratum_total}\n")

    # === SUMMARY ===
    total_pass = sum(v[0] for v in strata_summary.values())
    total_prompts = sum(v[1] for v in strata_summary.values())
    all_green = all(v[0] == v[1] for v in strata_summary.values())

    print(f"{'='*60}")
    print(f"  GATE SUMMARY: {total_pass}/{total_prompts} prompts passed")
    for name, (p, t) in strata_summary.items():
        color = "🟢" if p == t else "🔴"
        print(f"    {color} {name:15s}: {p}/{t}")
    print()
    if all_green:
        print("  *** ALL STRATA GREEN — variant token-identical to stock ***")
    else:
        print("  *** GATE FAILED — variant DIVERGES from stock ***")
        red = [n for n, (p, t) in strata_summary.items() if p < t]
        print(f"  RED strata: {', '.join(red)}")
    print(f"{'='*60}")

    if a.json:
        json.dump(results, open(a.json, "w"), indent=2)
        print(f"  Results written to {a.json}")

    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
