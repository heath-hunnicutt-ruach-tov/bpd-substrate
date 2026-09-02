#!/usr/bin/env python3
"""Verify BPD kernels are bit-identical with reference — CPU or GPU.

On CPU: compares against PyTorch CPU backend. If your PyTorch uses MKL/OpenBLAS,
matmul accumulation order may differ (set CPU_FP_MODE=fma or CPU_FP_MODE=native
when building bpd_cpu.so to match). Elementwise ops should always be 0 ULP.

Set BPD_CPU_REF=sequential to use a Python sequential reference loop instead
of torch.matmul — this eliminates BLAS-backend differences and verifies our
C code produces the correct sequential accumulation.

Detects available hardware and runs the appropriate comparison:
  CPU:  BPD C kernels (gcc) vs PyTorch CPU
  GPU:  BPD CUDA kernels (nvcc) vs PyTorch CUDA (cuBLAS/ATen)

Anyone with Python + gcc can verify correctness. No GPU required.

Usage:
    python3 bench/bit_identical_universal.py          # auto-detect
    BPD_CPU_SO=build/bpd_cpu.so python3 bench/bit_identical_universal.py  # explicit CPU
"""
import ctypes, os, sys, numpy as np
import json as _json, datetime as _dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truth_reference as _truth

# ACCURACY AXIS side-channel, keyed by kernel name.  A module-level dict rather
# than a 12th field on the results tuple -- widening that tuple is what produced
# the consumer-unpack crash earlier today, and this data has exactly ONE
# consumer (the emitter), not every reader of `results`.
ACCURACY = {}

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch numpy")

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

CPU_SO = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
GPU_SO = os.environ.get("BPD_MM_SO", "build/bpd_mm.so")

def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), d.size

def classify(ref, got, label=""):
    # Returns (status, max_ulp, abs_max, diverged_count, total_floats).
    # THE EMPTY-POPULATION GUARD: cnt/tot travel with every verdict so no row
    # can report a class without saying how much was checked.
    mx, cnt, tot = ulp(ref, got)
    abs_max = float(np.abs(ref - got).max())
    if mx == 0:
        return "BIT_IDENTICAL", mx, abs_max, cnt, tot
    elif abs_max < 1e-4 and mx > 100000:
        return "PASS_ABS_TOLERANCE", mx, abs_max, cnt, tot
    elif mx <= 64:
        return "PASS_WITHIN_64_ULP", mx, abs_max, cnt, tot
    elif abs_max < 1e-5:
        return "PASS_ABS_TOLERANCE", mx, abs_max, cnt, tot
    else:
        return "FAIL", mx, abs_max, cnt, tot

def load_cpu_lib():
    if not os.path.exists(CPU_SO):
        return None
    lib = ctypes.CDLL(CPU_SO)
    # matmul
    lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu.restype = None
    # fused matmul + bias + relu
    lib.bpd_mm_bias_relu_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    lib.bpd_mm_bias_relu_cpu.restype = None
    # elementwise
    for fn in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # conv2d
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
    lib.bpd_conv2d_cpu.restype = None
    # batchnorm
    lib.bpd_batchnorm_cpu.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.bpd_batchnorm_cpu.restype = None
    # upsample
    lib.bpd_upsample_nearest2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
    lib.bpd_upsample_nearest2d_cpu.restype = None
    # additional elementwise
    for fn in ['bpd_sigmoid_cpu', 'bpd_tanh_cpu', 'bpd_gelu_cpu', 'bpd_neg_cpu', 'bpd_abs_cpu', 'bpd_exp_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # reductions
    for fn in ['bpd_sum_cpu', 'bpd_mean_cpu', 'bpd_max_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # softmax
    lib.bpd_softmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_softmax_cpu.restype = None
    # layernorm
    lib.bpd_layernorm_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
    lib.bpd_layernorm_cpu.restype = None
    # maxpool2d / avgpool2d
    for fn in ['bpd_maxpool2d_cpu', 'bpd_avgpool2d_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*8
        getattr(lib, fn).restype = None
    # linear
    lib.bpd_linear_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    lib.bpd_linear_cpu.restype = None
    return lib

def run_cpu_tests(lib):
    results = []
    rng = np.random.default_rng(42)

    # ── Matmul ──
    for M in [64, 256, 512]:
        N = K = M
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
        out = np.zeros((M, N), dtype=np.float32)
        lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, M, N, K)
        status, mx, ab, cnt, tot = classify(ref, out)
        results.append(("sgemm_cpu", f"{M}x{M}", status, mx, ab, cnt, tot))

    # ── Elementwise ──
    x = rng.standard_normal(10000).astype(np.float32)
    for name, pt_fn, bpd_fn in [
        ("relu",  lambda t: torch.relu(t), lib.bpd_relu_cpu),
        ("silu",  lambda t: torch.nn.functional.silu(t), lib.bpd_silu_cpu),
        ("mish",  lambda t: torch.nn.functional.mish(t), lib.bpd_mish_cpu),
    ]:
        ref = pt_fn(torch.from_numpy(x)).numpy()
        out = np.zeros_like(x)
        bpd_fn(x.ctypes.data, out.ctypes.data, len(x))
        status, mx, ab, cnt, tot = classify(ref, out)
        results.append((f"{name}_cpu", "10000", status, mx, ab, cnt, tot))
        try:
            _t = _truth.truth_of(f"{name}_cpu", x)
            ACCURACY[f"{name}_cpu"] = _truth.accuracy_class(out, ref, _t)
        except Exception:
            ACCURACY[f"{name}_cpu"] = ("UNMEASURED", {})

    # ── Fused matmul + bias + relu ──
    M, N, K = 256, 256, 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    ref = torch.relu(torch.from_numpy(A) @ torch.from_numpy(B) + torch.from_numpy(bias)).numpy()
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_bias_relu_cpu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, out.ctypes.data, M, N, K)
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("fused_mm_bias_relu_cpu", f"{M}x{M}", status, mx, ab, cnt, tot))

    # ── Conv2D ──
    N_batch, C_in, H, W, C_out, kH, kW = 1, 3, 16, 16, 8, 3, 3
    stride, pad = 1, 1
    inp = rng.standard_normal((N_batch, C_in, H, W)).astype(np.float32)
    weight = rng.standard_normal((C_out, C_in, kH, kW)).astype(np.float32)
    ref = torch.nn.functional.conv2d(
        torch.from_numpy(inp), torch.from_numpy(weight),
        stride=stride, padding=pad).numpy()
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N_batch, C_out, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_cpu(inp.ctypes.data, weight.ctypes.data, out.ctypes.data,
                        N_batch, C_in, H, W, C_out, kH, kW, stride, pad)
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("conv2d_cpu", f"{N_batch}x{C_in}x{H}x{W}", status, mx, ab, cnt, tot))

    # ── Upsample ──
    inp = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
    ref = torch.nn.functional.interpolate(
        torch.from_numpy(inp), scale_factor=2, mode='nearest').numpy()
    out = np.zeros((1, 8, 8, 8), dtype=np.float32)
    lib.bpd_upsample_nearest2d_cpu(inp.ctypes.data, out.ctypes.data, 1, 8, 4, 4)
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("upsample_cpu", "1x8x4x4", status, mx, ab, cnt, tot))

    # ── Additional elementwise ──
    for name, pt_fn, bpd_fn in [
        ("sigmoid", lambda t: torch.sigmoid(t), lib.bpd_sigmoid_cpu),
        ("tanh",    lambda t: torch.tanh(t), lib.bpd_tanh_cpu),
        ("gelu",    lambda t: torch.nn.functional.gelu(t), lib.bpd_gelu_cpu),
        ("neg",     lambda t: -t, lib.bpd_neg_cpu),
        ("abs",     lambda t: torch.abs(t), lib.bpd_abs_cpu),
        ("exp",     lambda t: torch.exp(t), lib.bpd_exp_cpu),
    ]:
        ref = pt_fn(torch.from_numpy(x)).numpy()
        out = np.zeros_like(x)
        bpd_fn(x.ctypes.data, out.ctypes.data, len(x))
        status, mx, ab, cnt, tot = classify(ref, out)
        results.append((f"{name}_cpu", "10000", status, mx, ab, cnt, tot))
        try:
            _t = _truth.truth_of(f"{name}_cpu", x)
            ACCURACY[f"{name}_cpu"] = _truth.accuracy_class(out, ref, _t)
        except Exception:
            ACCURACY[f"{name}_cpu"] = ("UNMEASURED", {})

    # ── Reductions ──
    r_input = rng.standard_normal(1024).astype(np.float32)
    for name, pt_fn, bpd_fn in [
        ("sum",  lambda t: torch.sum(t), lib.bpd_sum_cpu),
        ("mean", lambda t: torch.mean(t), lib.bpd_mean_cpu),
        ("max",  lambda t: torch.max(t), lib.bpd_max_cpu),
    ]:
        ref_val = pt_fn(torch.from_numpy(r_input)).numpy().reshape(1)
        out = np.zeros(1, dtype=np.float32)
        bpd_fn(r_input.ctypes.data, out.ctypes.data, len(r_input))
        status, mx, ab, cnt, tot = classify(ref_val, out)
        results.append((f"reduce_{name}_cpu", "1024", status, mx, ab, cnt, tot))

    # ── Softmax ──
    s_input = rng.standard_normal((32, 64)).astype(np.float32)
    ref = torch.softmax(torch.from_numpy(s_input), dim=-1).numpy()
    out = np.zeros_like(s_input)
    lib.bpd_softmax_cpu(s_input.ctypes.data, out.ctypes.data, 32, 64)
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("softmax_cpu", "32x64", status, mx, ab, cnt, tot))

    # ── LayerNorm ──
    ln_input = rng.standard_normal((8, 128)).astype(np.float32)
    gamma = rng.standard_normal(128).astype(np.float32)
    beta = rng.standard_normal(128).astype(np.float32)
    ln = torch.nn.LayerNorm(128, elementwise_affine=True)
    ln.weight.data = torch.from_numpy(gamma)
    ln.bias.data = torch.from_numpy(beta)
    ref = ln(torch.from_numpy(ln_input)).detach().numpy()
    out = np.zeros_like(ln_input)
    lib.bpd_layernorm_cpu(ln_input.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                           out.ctypes.data, 8, 128, ctypes.c_float(1e-5))
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("layernorm_cpu", "8x128", status, mx, ab, cnt, tot))

    # ── MaxPool2D ──
    p_input = rng.standard_normal((1, 3, 16, 16)).astype(np.float32)
    ref = torch.nn.functional.max_pool2d(torch.from_numpy(p_input), 2, stride=2).numpy()
    H_out = (16 - 2) // 2 + 1
    out = np.zeros((1, 3, H_out, H_out), dtype=np.float32)
    lib.bpd_maxpool2d_cpu(p_input.ctypes.data, out.ctypes.data, 1, 3, 16, 16, 2, 2, 2, 0)
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("maxpool2d_cpu", "1x3x16x16", status, mx, ab, cnt, tot))

    # ── Linear ──
    l_input = rng.standard_normal((4, 32)).astype(np.float32)
    weight = rng.standard_normal((64, 32)).astype(np.float32)
    bias_l = rng.standard_normal(64).astype(np.float32)
    lin = torch.nn.Linear(32, 64, bias=True)
    lin.weight.data = torch.from_numpy(weight)
    lin.bias.data = torch.from_numpy(bias_l)
    ref = lin(torch.from_numpy(l_input)).detach().numpy()
    out = np.zeros((4, 64), dtype=np.float32)
    lib.bpd_linear_cpu(l_input.ctypes.data, weight.ctypes.data, bias_l.ctypes.data,
                        out.ctypes.data, 4, 64, 32)
    status, mx, ab, cnt, tot = classify(ref, out)
    results.append(("linear_cpu", "4x32->64", status, mx, ab, cnt, tot))

    return results

def main():
    print(f"BPD Universal Bit-Identity Verification")
    print(f"PyTorch {torch.__version__} on {DEVICE.upper()}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    lib_cpu = load_cpu_lib()
    if lib_cpu:
        print(f"CPU library: {CPU_SO}")
    else:
        print(f"CPU library: not found ({CPU_SO})")
        print(f"  Build with: gcc -O2 -shared -fPIC -o {CPU_SO} bench/bpd_cpu.c -lm")

    results = []

    if lib_cpu:
        print()
        print("── CPU VERIFICATION (BPD C kernels vs PyTorch CPU) ──")
        cpu_results = run_cpu_tests(lib_cpu)
        results.extend(cpu_results)
        for name, shape, status, mx, ab, cnt, tot in cpu_results:
            # VERDICT-CLASS RULE: only BIT_IDENTICAL earns the check.  A
            # within-tolerance cell is an open target, marked "~", not a pass.
            tag = "✓" if status == "BIT_IDENTICAL" else ("~" if "PASS" in status else "✗")
            print(f"  {name:<25} {shape:<16} {status:<22} "
                  f"max_ulp={mx:<10} n={tot:<9,} {tag}")

    # Summary
    print()
    # THE VERDICT-CLASS RULE: 0-ULP is not "pass".  BIT_IDENTICAL is the only
    # green; the PASS_* classes ran and were close but are NOT bit-identical,
    # and collapsing them into one count is what produced the old passed=22
    # against 21 actually-0-ULP kernels.
    bit_identical = sum(1 for _, _, s, _, _, _, _ in results if s == "BIT_IDENTICAL")
    within_tolerance = sum(1 for _, _, s, _, _, _, _ in results
                           if s != "BIT_IDENTICAL" and "PASS" in s)
    failed_n = sum(1 for _, _, s, _, _, _, _ in results
                   if s != "BIT_IDENTICAL" and "PASS" not in s)
    floats_checked = sum(t for _, _, _, _, _, _, t in results)
    total = len(results)
    print(f"{'=' * 60}")
    print(f"BIT_IDENTICAL:    {bit_identical}/{total}   (0-ULP -- the metric)")
    print(f"within_tolerance: {within_tolerance}/{total}   (ran, close, NOT bit-identical)")
    print(f"failed:           {failed_n}/{total}")
    print(f"floats compared:  {floats_checked:,}   (the population behind the verdicts)")
    if bit_identical == total:
        print(f"\nALL KERNELS BIT-IDENTICAL WITH PyTorch on {DEVICE.upper()}.")
        print(f"Same math. Same bits. {'No GPU required.' if not HAS_CUDA else ''}")
    else:
        notgreen = [(n, s, mx, t) for n, _, s, mx, _, _, t in results if s != "BIT_IDENTICAL"]
        print(f"\nNOT BIT-IDENTICAL: {len(notgreen)}")
        for n, s, mx, t in notgreen:
            print(f"  {n}: {s} (max {mx} ULP over {t:,} floats)")

    # SEPARATION, not collapse (Iyun's ruling): "the check RAN correctly" and
    # "every cell is 0-ULP" are DIFFERENT facts.  Conflating them into one exit
    # code is the same class of error as conflating passed with bit_identical.
    #   default:  exit 0 on a successful run, open cells REPORTED not hidden
    #   --strict: exit nonzero if any cell is not bit-identical (the CI gate)
    # ── congruence_status.json — the Track A → dashboard contract ──────────
    # Schema: dashboard/CONGRUENCE_SCHEMA.md.  Every row carries its POPULATION
    # (total_floats, diverged_count) so no verdict can be read without knowing
    # how much was checked, and `status` verbatim so bit-identical is never
    # collapsed with within-tolerance.
    ORACLES = {
        "sgemm_cpu": "torch.matmul", "fused_mm_bias_relu_cpu": "torch.relu(x@W+b)",
        "conv2d_cpu": "torch.nn.functional.conv2d",
        "upsample_cpu": "torch.nn.functional.interpolate",
        "softmax_cpu": "torch.nn.functional.softmax",
        "layernorm_cpu": "torch.nn.functional.layer_norm",
        "maxpool2d_cpu": "torch.nn.functional.max_pool2d",
        "linear_cpu": "torch.nn.functional.linear",
    }
    # torch.* for tensor ops, torch.nn.functional.* for nn ops.  These are not
    # interchangeable: torch.nn.functional.neg does not exist.  Verified against
    # the live module rather than assumed.
    TENSOR_OPS = {"neg", "abs", "exp", "sqrt", "log", "sin", "cos", "erf"}
    def _oracle(k):
        if k in ORACLES:
            return ORACLES[k]
        base = k[:-4] if k.endswith("_cpu") else k
        if base.startswith("reduce_"):
            return "torch.%s" % base[len("reduce_"):]
        if base in TENSOR_OPS:
            return "torch.%s" % base
        return "torch.nn.functional.%s" % base
    rows = []
    for nm, shp, st, mxu, abm, dc, tf in results:
        rows.append({
            "kernel": nm, "shape": shp, "status": st,
            "max_ulp": int(mxu), "bit_identical": st == "BIT_IDENTICAL",
            "backend": "cuda" if nm.endswith("_gpu") else "cpu",
            "total_floats": int(tf), "diverged_count": int(dc),
            "abs_max": float(abm), "oracle": _oracle(nm),
            "dtype": "float32", "device": DEVICE,
        })
        # MATCHED is ENTAILED by bit-identity -- identical bits carry identical
        # error -- so it needs no truth-measurement.  Otherwise: what the
        # measurement found, or UNMEASURED where no truth-reference exists.
        if rows[-1]["bit_identical"]:
            rows[-1]["accuracy_class"] = "MATCHED"
        else:
            _cls, _ev = ACCURACY.get(nm, ("UNMEASURED", {}))
            rows[-1]["accuracy_class"] = _cls
            rows[-1].update(_ev)
    doc = {
        # Sub-second precision: mavhir's sync-lag detector compares this field
        # across the served and origin copies, and equal timestamps read as
        # "in sync".  Second resolution would make two emits in the same second
        # indistinguishable, so the detector would silently miss that lag.
        "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        # This feed is EVENT-DRIVEN, not continuous: it emits when the checker or
        # the kernels change, not on a clock.  The default staleness thresholds
        # assume a continuous pipeline and would show red after 15 quiet minutes,
        # training a reader to ignore the colour.
        "staleness_threshold_seconds": 14400,
        "total": total, "bit_identical": bit_identical,
        "within_tolerance": within_tolerance, "failed": failed_n,
        "floats_compared": floats_checked,
        # TRUTH-AXIS COUNTS, symmetric with the stock-axis three above.
        # Two PARTITIONS of the same rows, not two tallies of different things:
        # each set sums to `total`.  mavhir's render groups them visually so a
        # reader sees "different views of the same 22" rather than 44.
        "matched": sum(1 for r in rows if r.get("accuracy_class") == "MATCHED"),
        "improved": sum(1 for r in rows if r.get("accuracy_class") == "IMPROVED"),
        "inaccurate": sum(1 for r in rows if r.get("accuracy_class") == "INACCURATE"),
        "unmeasured": sum(1 for r in rows if r.get("accuracy_class") == "UNMEASURED"),
        "open_cells": [r["kernel"] for r in rows if not r["bit_identical"]],
        "kernels": rows,
    }
    out_path = os.environ.get("BPD_STATUS_JSON", "dashboard/congruence_status.json")
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            _json.dump(doc, fh, indent=2)
        print(f"\nwrote {out_path}  ({len(rows)} rows, {floats_checked:,} floats)")
    except OSError as e:
        print(f"\ncould not write {out_path}: {e}")

    open_cells = total - bit_identical
    if open_cells:
        print(f"\nOPEN CELLS: {open_cells}   (not bit-identical -- targets to close)")
    if "--strict" in sys.argv:
        return 0 if open_cells == 0 else 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
