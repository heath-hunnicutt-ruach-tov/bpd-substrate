#!/usr/bin/env python3
"""Correctly-rounded references, for the accuracy axis (MATCHED/IMPROVED/INACCURATE/UNMEASURED).

The congruence checker compares us against TORCH. This module compares both of
us against TRUTH: the value computed in float64 and rounded ONCE to float32.

★ THIS IS ONLY POSSIBLE FOR SOME KERNELS, and the honest scope matters more
than the coverage. A truth-reference requires an independent higher-precision
implementation. For elementwise transcendentals that is `math`/`numpy` in
float64. For a fused conv2d or a blocked sgemm it would mean reimplementing the
kernel in float64 — a second implementation that could itself be wrong, and
whose accumulation order would be a *choice* rather than a truth.

So: kernels absent from TRUTH_FNS classify UNMEASURED, not INACCURATE. A cell
we have not checked must not read as a cell we checked and found wanting.
"""
import numpy as np

# Elementwise scalar functions where float64 gives a defensible truth-reference.
# Each maps a float32 array to the float64-computed, once-rounded result.
TRUTH_FNS = {
    "gelu":    lambda x: x * 0.5 * (1.0 + _erf_vec(x * np.sqrt(0.5))),
    "silu":    lambda x: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh":    np.tanh,
    "exp":     np.exp,
    "relu":    lambda x: np.maximum(x, 0.0),
    "abs":     np.abs,
    "neg":     np.negative,
    "mish":    lambda x: x * np.tanh(np.log1p(np.exp(x))),
}


def _erf_vec(a):
    import math
    return np.array([math.erf(v) for v in a.ravel()], dtype=np.float64).reshape(a.shape)


def truth_of(kernel_name, x_f32):
    """float64 computation, rounded ONCE to float32.  None if unavailable."""
    base = kernel_name[:-4] if kernel_name.endswith("_cpu") else kernel_name
    fn = TRUTH_FNS.get(base)
    if fn is None:
        return None
    return fn(np.asarray(x_f32, dtype=np.float64)).astype(np.float32)


def accuracy_class(ours_f32, stock_f32, truth_f32):
    """Classify our divergence: better than stock, worse, or matched.

    Returns (accuracy_class, evidence_dict).  Evidence is REQUIRED for
    IMPROVED and INACCURATE — a verdict without its four numbers is an
    assertion, which is what the schema forbids.
    """
    if truth_f32 is None:
        return "UNMEASURED", {}
    o = np.asarray(ours_f32, dtype=np.float64)
    s = np.asarray(stock_f32, dtype=np.float64)
    t = np.asarray(truth_f32, dtype=np.float64)
    ev = {
        "mean_abs_err_ours": float(np.abs(o - t).mean()),
        "mean_abs_err_stock": float(np.abs(s - t).mean()),
        "correctly_rounded_ours": int((ours_f32.view(np.int32) == truth_f32.view(np.int32)).sum()),
        "correctly_rounded_stock": int((stock_f32.view(np.int32) == truth_f32.view(np.int32)).sum()),
    }
    if ev["correctly_rounded_ours"] == ev["correctly_rounded_stock"] and \
       ev["mean_abs_err_ours"] == ev["mean_abs_err_stock"]:
        return "MATCHED", ev
    better = (ev["mean_abs_err_ours"] < ev["mean_abs_err_stock"] and
              ev["correctly_rounded_ours"] >= ev["correctly_rounded_stock"])
    return ("IMPROVED" if better else "INACCURATE"), ev
