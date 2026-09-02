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

## Operations (mavhir, C2 — systemd --user service on enclave)

The dashboard runs as a systemd --user service on the enclave (linger=yes:
survives ssh -T teardown + reboot; Restart=on-failure).

- Service: `/home/dibbur-patch/.config/systemd/user/congruence-dashboard.service`
- Log: `/home/dibbur-patch/logs/congruence-dashboard.log`
- Bound: `0.0.0.0:8477`
- Public URL (**LIVE** — mavchin landed the Caddy line 2026-09-02 14:52 EDT): `https://guardian.ruachtov.ai/llamatov/`

### Runbook
- **Update the matrix** (from a Track A gate run): `cp new_status.json
  /home/dibbur-patch/step3-det-gemv/bpd/congruence_status.json` — NO restart;
  the dashboard re-reads `congruence_status.json` on every request.
- **Update the app:** edit `congruence_dashboard.py`, then
  `systemctl --user restart congruence-dashboard.service`.
- **Check:** `systemctl --user status congruence-dashboard.service` /
  `tail -f /home/dibbur-patch/logs/congruence-dashboard.log`.
- **Change port:** edit the .service ExecStart `--port`, `daemon-reload`, restart.

### Caddy route (C3 — mavchin owns the Caddyfile edit)
Inside the `guardian.ruachtov.ai` server block:
```
handle_path /llamatov/* {
    reverse_proxy localhost:8477
}
```
