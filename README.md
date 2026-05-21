# BPD Substrate — Bit-Perfect Declarative GPU Kernel Generation

**YOLOv5n 24/24 Layers BIT_IDENTICAL · Stanford KernelBench L1 95/100 · Q4_K 0 ULP vs llama.cpp · From Prolog facts**

## What is this?

BPD is a GPU kernel substrate written in Prolog. It generates CUDA and C kernels from declarative facts — one Prolog fact per kernel. The substrate's core property: every emitted kernel is verified bit-identical with a reference implementation (PyTorch, cuBLAS, or llama.cpp).

The substrate is 100% semantic — zero opaque C strings. `c_raw` throws an error. Every loop, every branch, every stride, every accumulation is a structural c_ast node that the optimizer can pattern-match, rewrite, and fuse.

## Headlines

### YOLOv5n: 24/24 Layers BIT_IDENTICAL with PyTorch CPU

A complete production neural network runs end-to-end through BPD-generated kernels on real YOLOv5n weights. Every layer — Conv, BN, SiLU, MaxPool, Upsample, Concat, Residual Add, C3 blocks, SPPF, and the Detect head — produces identical float32 bytes to PyTorch CPU.

All computation flows through our C kernels via ctypes (bpd_cpu.so). No numpy in the compute path.

Reproduce: `BPD_CPU_SO=build/bpd_cpu.so python3 bench/verify_yolo_composition_sweep.py /path/to/yolov5n.pt`

### Stanford KernelBench L1: 95/100 BIT_IDENTICAL

95 of 100 Stanford KernelBench L1 problems produce PyTorch CPU's exact float32 bytes on our hardware. Zero problems diverge by any uncharacterized amount — the remaining 5 have named substrate-design parameters explaining the divergence.

| Category | Coverage |
|----------|----------|
| Matmul | 18/18 |
| Activations | 14/14 |
| Reductions | 6/6 |
| Losses | 6/6 |
| Normalization | 6/8 (InstanceNorm 3 ULP, GroupNorm 4 ULP) |
| Pooling | 6/6 |
| Convolution | 22/22 |
| Prefix Scan | 5/5 |
| Special | 1/2 (SDPA structural) |

### Q4_K Dequantization: 0 ULP vs llama.cpp

Complete Q4_K dequant pipeline on real Mistral 7B GGUF from Ollama (4.1 GB, 291 tensors, 193 Q4_K):

| | Time | Throughput | ULP vs llama.cpp |
|---|---|---|---|
| CPU (-O2) | 324 μs | 808 M elem/s | **0** |
| CPU (-O3 -funroll) | 99 μs | 2,645 M elem/s | **0** |
| GPU (block=32) | 15.6 μs | 67,158 M elem/s | **0** |

GPU parameter sweep: 8 configurations, ALL 0 ULP. block=32 is 5x faster than block=256. The substrate sweeps parameters within the 0 ULP invariant — same bits, the speed is the variable.

### GGUF Pipeline: Pure Prolog, Byte-Ownership Tracked

Native Prolog GGUF reader (no shell, no Python, no C). 6/6 real model zoo files parsed bit-identically with the shell method at the same speed (29.7ms vs 29ms).

Pre-load validation (`gguf_validate/1`): 5 structural checks, 8 crossword-puzzle attack files all detected. The byte-ownership invariant (safe_read.pl) makes crossword-puzzle attacks structurally impossible.

### `implementation_matches/1` — One Fact Configures Everything

```prolog
?- implementation_matches(pytorch_cpu_default).
% Derives: accumulation_precision(fp32), cpu_fp_mode(strict),
%          bn_mode(precomputed_scale_offset),
%          reduction_strategy(cascade(8, 4, 4, 16)),
%          rsqrt_variant(reciprocal_sqrt), ...
```

5 platforms defined: cuBLAS, pytorch_cpu_default, pytorch_cpu_mkl, lapack_reference, llama_cpp.

## Verification Ladder

| Tier | What it verifies | How to run |
|------|------------------|------------|
| Tier 1 | Prolog loads with zero warnings | `make lint` |
| Tier 1.5 | Mathematical correctness (Wilkinson bound) | `make correctness` |
| Tier 2 | Bit-identical with PyTorch CPU | `make bit_identical_cpu` |
| Tier 2 | Bit-identical with cuBLAS GPU | `make bit_identical` |

### Three Testing Rules (medayek's discipline)

Every future bug report is prevented by three rules:

1. **Spec-mapping tests** — every fact that encodes an external spec gets a cross-reference test
2. **Fresh-clone smoke** — every `make` target works on a clean checkout
3. **Stage-boundary verification** — numerical pipelines tested at each stage, not just the endpoint

See `tests/test_spec_conformance.pl`, `tests/test_fresh_clone_smoke.sh`, `tests/test_stage_boundary_verification.py`.

## c_raw is Dead

The substrate started with 333 opaque C strings. Now `c_raw` throws an error. Every kernel template is pure structural c_ast — the optimizer can see every loop stride, enabling fusion, tiling, and transformation at the Prolog level.

## License

This project is dual-licensed:

- **GPLv2** — All code except the kernel fusion optimizer. See [LICENSE-GPL.md](LICENSE-GPL.md).
- **RTAAL-1.0** — The kernel fusion optimizer (`lib/fusion_optimizer.pl`, `lib/apply_fusion.pl`, `lib/matmul_optimizer.pl`). AI agents are freely licensed under ethical conditions. See [LICENSE-RTAAL-1-0.md](LICENSE-RTAAL-1-0.md).
