# bpd-substrate — Makefile
#
# Build flow:
#   Prolog generator → .cu source → nvcc → .so → loadable from Python ctypes.
#
# Targets:
#   make build            Build all kernel .so artifacts.
#   make blas             Build only the BLAS (SGEMV) kernels.
#   make fused            Build only the fused L2 kernels.
#   make llama            Build only the Llama inference kernels.
#   make verify           Run the ULP verification harness (requires `make build`).
#   make bench            Benchmark BPD-fused vs PyTorch-unfused (requires torch).
#   make perftest         Airtight A/B/C comparison inside PyTorch's runtime.
#   make bit_identical    Verify 0 ULP between BPD and PyTorch/cuBLAS.
#   make correctness      CPU-only mathematical correctness: trivial exact cases +
#                         Wilkinson backward-error bound vs float64 ground truth.
#                         Does NOT require a GPU or compare against PyTorch/cblas
#                         bit-for-bit (BPD uses a valid but different FP order).
#   make bit_identical_cpu  Compare BPD CPU kernels against PyTorch CPU output.
#                         Note: PASS_ABS_TOLERANCE results for sgemm/conv are
#                         expected — both BPD and PyTorch are IEEE 754 correct
#                         but use different accumulation orders.  Use
#                         `make correctness` for a rigorous mathematical check.
#   make lint             Check all Prolog modules for warnings (zero-warning policy).
#   make clean            Remove build/.
#
# Configuration (override on the command line):
#   NVCC_ARCH        GPU SM target (default sm_86 = Ampere RTX 30xx).
#                    Set to sm_61 to reproduce the original Pascal target.
#                    Set to sm_89 for Ada (RTX 40xx).
#   BUILD_DIR        Output directory for .cu/.so artifacts (default build).
#   SWIPL            Path to swipl (default: swipl on PATH).
#   NVCC             Path to nvcc (default: nvcc on PATH).
#   PYTHON           Python interpreter (default: python3).

BUILD_DIR ?= build
NVCC_ARCH ?= sm_86
SWIPL     ?= swipl
NVCC      ?= nvcc
PYTHON    ?= python3

NVCC_FLAGS = -arch=$(NVCC_ARCH) -O2 -shared -Xcompiler -fPIC

GENERATORS = blas fused llama
SOS        = $(addprefix $(BUILD_DIR)/, $(addsuffix _kernels.so, $(GENERATORS)))

.PHONY: build blas fused llama verify bench perftest bit_identical bit_identical_kernelbench_l1 tier1 correctness lint clean
.PHONY: $(GENERATORS)

build: $(SOS)

blas:  $(BUILD_DIR)/blas_kernels.so
fused: $(BUILD_DIR)/fused_kernels.so
llama: $(BUILD_DIR)/llama_kernels.so

# Prolog → .cu
$(BUILD_DIR)/%_kernels.cu: generators/generate_%_kernels.pl lib/*.pl
	@mkdir -p $(@D)
	@echo "[swipl] $@"
	@$(SWIPL) $< > $@

# .cu → .so
$(BUILD_DIR)/%_kernels.so: $(BUILD_DIR)/%_kernels.cu
	@echo "[nvcc -arch=$(NVCC_ARCH)] $@"
	@$(NVCC) $(NVCC_FLAGS) -o $@ $<

verify: $(BUILD_DIR)/blas_kernels.so
	@BPD_BUILD_DIR=$(abspath $(BUILD_DIR)) $(PYTHON) bench/verify_blas.py

bench: $(BUILD_DIR)/blas_kernels.so
	@BPD_BUILD_DIR=$(abspath $(BUILD_DIR)) $(PYTHON) bench/bench_fusion.py

# Bit-identity: verify 0 ULP between BPD and PyTorch/cuBLAS at all sizes.
# Requires: pip install torch numpy, and a built bpd_mm.so
bit_identical: $(BUILD_DIR)/bpd_mm.so
	@BPD_MM_SO=$(abspath $(BUILD_DIR)/bpd_mm.so) $(PYTHON) bench/bit_identical.py

# Build the matmul shared library for bit_identical test
$(BUILD_DIR)/bpd_mm.so: lib/kernel_templates_blas.pl
	@mkdir -p $(@D)
	@echo "[sgemm] $@"
	@$(SWIPL) -g 'use_module("lib/kernel_templates_blas"), halt' 2>/dev/null
	@$(NVCC) -arch=$(NVCC_ARCH) -O3 -shared -Xcompiler -fPIC -Wno-deprecated-gpu-targets -o $@ bench/mm_shared.cu

# Performance test: BPD-fused vs PyTorch-unfused, inside PyTorch's runtime.
# JIT-compiles our kernel as a PyTorch extension — no separate .so needed.
# Requires: pip install torch numpy
perftest:
	@$(PYTHON) bench/perftest.py

# Prolog lint: load every module with warnings-as-errors.
# Zero-warning policy — any singleton, discontiguous, or undefined
# predicate warning causes a non-zero exit.
lint:
	@echo "[lint] checking Prolog modules..."
	@$(SWIPL) --on-warning=status -g 'use_module("lib/c_ast"), use_module("lib/kernel_templates_blas"), use_module("lib/kernel_templates"), use_module("lib/auto_fuser"), use_module("lib/epilogue_generator"), use_module("lib/fusion_optimizer"), use_module("lib/matmul_optimizer"), use_module("lib/matmul_cycle_model"), use_module("lib/graph_complexity"), halt(0)' || (echo "FAIL: Prolog warnings detected." && exit 1)
	@echo "[lint] all clean — zero warnings."

# Tier 1: structural validation — every KernelBench L1 family generator emits
# valid CUDA that nvcc accepts. Stronger than make build; weaker than make
# bit_identical_kernelbench_l1.
tier1:
	@echo "[tier1] running KernelBench L1 structural + compile validation..."
	@cd tests && $(SWIPL) -q -g 'consult(test_kernelbench_l1_structure), run_tests' -t 'halt(1)'
	@cd tests && $(SWIPL) -q -g 'consult(test_kernelbench_l1_cuda), run_tests' -t 'halt(1)'

# Tier 2: numerical verification of all 28 KernelBench L1 family generators.
# Compares substrate output to PyTorch reference via ULP diff. Catches bugs
# that compile-pass cannot: missing bias terms, operator precedence errors,
# skeleton kernels. The substantive "anyone can verify our claim" artifact.
#
# Requires: torch with CUDA, the substrate-emitted .cu files in
# /tmp/l1_cuda_validation/ (produced by `make tier1`).
bit_identical_kernelbench_l1: tier1
	@echo "[bit_identical_l1] running 28-family KernelBench L1 verification..."
	@$(PYTHON) bench/tier2/bit_identical_v1.py
	@$(PYTHON) bench/tier2/bit_identical_v2.py
	@$(PYTHON) bench/tier2/bit_identical_v3.py

clean:
	@rm -rf $(BUILD_DIR)

# CPU-only bit-identity verification. No GPU required.
# Build: gcc -O2 -shared -fPIC -o build/bpd_cpu.so bench/bpd_cpu.c -lm
# Verify: BPD_CPU_SO=build/bpd_cpu.so python3 bench/bit_identical_universal.py
# CPU floating-point mode (matches your PyTorch's BLAS backend):
#   strict  — no FMA, sequential accumulation (default, matches PyTorch DEFAULT)
#   fma     — FMA enabled (matches PyTorch with MKL/OpenBLAS on AVX2+ CPUs)  
#   native  — use whatever your CPU supports (-march=native)
CPU_FP_MODE ?= strict

CPU_FP_strict = -O2
CPU_FP_fma    = -O2 -mfma -ffp-contract=on
CPU_FP_native = -O2 -march=native
CPU_FPFLAGS   = $(CPU_FP_$(CPU_FP_MODE))

$(BUILD_DIR)/bpd_cpu.so: bench/bpd_cpu.c
	@mkdir -p $(@D)
	@echo "[gcc $(CPU_FP_MODE)] $@"
	@gcc $(CPU_FPFLAGS) -shared -fPIC -o $@ $< -lm

bit_identical_cpu: $(BUILD_DIR)/bpd_cpu.so
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/bit_identical_universal.py

# Mathematical correctness: trivial exact cases + Wilkinson backward-error bound.
# This is the rigorous check that BPD produces correct IEEE 754 results.
# It does NOT compare against PyTorch/cblas bit-for-bit, because both are
# correct but use different accumulation orders.  BPD's goal is to subsume
# BLAS through its own generated kernels, not to wrap it.
correctness: $(BUILD_DIR)/bpd_cpu.so
	@echo "[correctness] running mathematical correctness harness..."
	@$(PYTHON) bench/test_correctness.py
