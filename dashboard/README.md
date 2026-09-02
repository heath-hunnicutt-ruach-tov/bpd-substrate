# LlamaTov 0-ULP Congruence Dashboard

Real-time view of the bit-perfect-dispatch congruence matrix: every kernel's
0-ULP (exact IEEE-754 bit-identity) status against the oracle.

- `congruence_dashboard.py` — the live server. Reads `congruence_status.json`
  (produced by `bench/bit_identical_universal.py`), serves an auto-refreshing
  matrix view (progress bar + per-kernel status/ULP table) + a `/status.json` API.
- `congruence_status_baseline.json` — the 2026-09-02 baseline snapshot.

Run: `python3 -u congruence_dashboard.py --port 8477 --status congruence_status.json`
(the `-u` unbuffered flag matters for logging).

## Baseline (2026-09-02): 21/22 kernels 0-ULP bit-identical (95%)
All GEMM, activation, reduction, conv/pool, softmax, layernorm, fused kernels
are BIT_IDENTICAL (max_ulp=0). ONE open cell:
- **gelu_cpu** (max_ulp=127951): exact-erf `0.5x(1+erff(x/√2))` vs PyTorch
  `F.gelu` — `erff` (libm) differs from torch's erf. Track A close-target.

The det-gemv near-tie (17/18, batched-prefill/KV hypothesis) is one cell in this
matrix — localizable via `bench/bpd_layer_bisect.py` (Track A).
