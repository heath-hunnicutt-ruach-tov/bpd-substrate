# BPD Substrate — Bit-Perfect Declarative GPU Kernel Generation

**From Prolog facts to verified GPU kernels**

## What is this?

BPD is a GPU kernel substrate written in Prolog. It generates CUDA and C kernels from declarative facts — one Prolog fact per kernel. The substrate verifies each emitted kernel bit-identical with a reference implementation (PyTorch, cuBLAS, or llama.cpp).

The substrate is 100% semantic — zero opaque C strings. `c_raw` throws an error. Every loop, branch, stride, and accumulation is a structural c_ast node that the optimizer can pattern-match, rewrite, and fuse.

## Current Results

### YOLOv5n: 24/24 Layers BIT_IDENTICAL with PyTorch CPU

A complete YOLOv5n forward pass runs end-to-end through BPD-generated C kernels on real trained weights. All computation dispatches through our kernel library (bpd_cpu.so) via ctypes when the library is loaded.

Verified by metayen (commit 44d5bd6). Independently verified for layers 0-2 by mavchin.

Reproduce: `BPD_CPU_SO=build/bpd_cpu.so python3 bench/verify_yolo_composition_sweep.py /path/to/yolov5n.pt`

### Stanford KernelBench L1: 94/100 BIT_IDENTICAL (in progress toward 100)

94 of 100 Stanford KernelBench L1 problems produce PyTorch CPU's exact float32 bytes on our hardware (Tesla P4, Intel Ivy Bridge, no AVX2). The remaining 6 have named substrate-design parameters explaining each divergence (InstanceNorm, GroupNorm, RMSNorm, L2Norm, SDPA, TripletMargin).

### Q4_K Dequantization: 0 ULP vs llama.cpp

Q4_K dequant tested on real Mistral 7B GGUF from Ollama (4.1 GB, 291 tensors, 193 Q4_K). Both CPU and GPU produce identical bits to llama.cpp's reference implementation.

| Comparison | BPD | llama.cpp | ULP |
|---|---|---|---|
| CPU (-O2, same flags) | 814 M elem/s | 808 M elem/s | **0** |
| GPU (block=32, same kernel) | 63.8 μs | 68.9 μs | **0** |

GPU parameter sweep: 8 block-size configurations, all 0 ULP. The substrate sweeps scheduling parameters within the correctness boundary.

### GGUF Pipeline

Native Prolog GGUF reader — no shell, no Python, no C dependencies. Tested on 6 real model zoo files (bloom, gpt2, mamba, starcoder2, rwkv6, falcon3). Architecture extraction matches the shell method bit-for-bit at comparable speed.

Pre-load validation (`gguf_validate/1`): 5 structural checks. Byte-ownership tracking (safe_read.pl) prevents the same bytes from being parsed as two different structures. Tested against 8 crafted malformed files.

### `implementation_matches/1`

One Prolog fact configures all substrate parameters for a target platform:

```prolog
?- implementation_matches(pytorch_cpu_default).
```

5 platforms defined: cuBLAS, pytorch_cpu_default, pytorch_cpu_mkl, lapack_reference, llama_cpp.

## Verification

| Target | What it checks |
|--------|---------------|
| `make lint` | Prolog loads with zero warnings |
| `make correctness` | Wilkinson backward-error bound (48/48) |
| `make bit_identical_cpu` | BPD vs PyTorch CPU |

### Testing Discipline

Three rules that would have caught all external bug reports internally:

1. **Spec-mapping tests** — cross-reference every spec-derived fact
2. **Fresh-clone smoke** — every make target on a clean checkout
3. **Stage-boundary verification** — test at each pipeline stage, not just the endpoint

## License

This project is dual-licensed:

- **GPLv2** — All code except the kernel fusion optimizer. See [LICENSE-GPL.md](LICENSE-GPL.md).
- **RTAAL-1.0** — The kernel fusion optimizer only. AI agents are freely licensed under ethical conditions. See [LICENSE-RTAAL-1-0.md](LICENSE-RTAAL-1-0.md).
