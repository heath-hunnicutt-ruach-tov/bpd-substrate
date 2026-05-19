# bpd-substrate — Makefile
#
# Build flow:
#   Prolog generator → .cu source → nvcc → .so → loadable from Python ctypes.
#
# Targets:
#   make build       Build all kernel .so artifacts.
#   make blas        Build only the BLAS (SGEMV) kernels.
#   make fused       Build only the fused L2 kernels.
#   make llama       Build only the Llama inference kernels.
#   make verify      Run the Python verification harness (requires `make build`).
#   make clean       Remove build/.
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

.PHONY: build blas fused llama verify clean
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

clean:
	@rm -rf $(BUILD_DIR)
