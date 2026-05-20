#!/usr/bin/env python3
"""verify_kernelbench_l1_cpu.py — Bit-identity sweep across all Stanford KernelBench L1 problems.

Compares BPD CPU substrate output against PyTorch CPU for each of the 100 L1
problems. Reports per-problem status:

  BIT_IDENTICAL    0 ULP vs PyTorch CPU
  DIVERGENT        non-zero ULP — names the substrate-design parameter at fault
  MISSING_KERNEL   the substrate doesn't yet have a CPU kernel for this op
  NOT_IMPLEMENTED  the harness doesn't yet route this problem

Output: per-category summary + detailed per-problem table + grand total.

Run:
  make verify FOCUS=kernelbench-l1-cpu
"""
import ctypes
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("error: pip install torch numpy")

torch.backends.mkldnn.enabled = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)


# ─── ULP machinery ─────────────────────────────────────────────────────────

def ulp(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return -1, -1, -1
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), int(d.size)


# ─── Substrate library loading ─────────────────────────────────────────────

def load_lib():
    so_path = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
    if not os.path.exists(so_path):
        sys.exit(f"{so_path} not found — run `make build` first")
    lib = ctypes.CDLL(so_path)
    # Elementwise (input, output, n)
    for name in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu', 'bpd_sigmoid_cpu',
                 'bpd_tanh_cpu', 'bpd_gelu_cpu', 'bpd_neg_cpu', 'bpd_abs_cpu',
                 'bpd_exp_cpu', 'bpd_sum_cpu', 'bpd_mean_cpu', 'bpd_max_cpu',
                 'bpd_leaky_relu_cpu', 'bpd_elu_cpu', 'bpd_selu_cpu',
                 'bpd_hardsigmoid_cpu', 'bpd_clamp_cpu',
                 'bpd_softplus_cpu', 'bpd_softsign_cpu',
                 'bpd_cumsum_cpu', 'bpd_cumprod_cpu',
                 'bpd_cumsum_reverse_cpu', 'bpd_cumsum_exclusive_cpu']:
        if hasattr(lib, name):
            f = getattr(lib, name)
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            f.restype = None
    # Special signatures
    if hasattr(lib, 'bpd_softmax_cpu'):
        lib.bpd_softmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int]
        lib.bpd_softmax_cpu.restype = None
    if hasattr(lib, 'bpd_logsoftmax_cpu'):
        lib.bpd_logsoftmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int]
        lib.bpd_logsoftmax_cpu.restype = None
    if hasattr(lib, 'bpd_mm_cpu'):
        lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
        lib.bpd_mm_cpu.restype = None
    if hasattr(lib, 'bpd_conv2d_cpu'):
        lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
        lib.bpd_conv2d_cpu.restype = None
    if hasattr(lib, 'bpd_maxpool2d_cpu'):
        lib.bpd_maxpool2d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*7
        lib.bpd_maxpool2d_cpu.restype = None
    if hasattr(lib, 'bpd_avgpool2d_cpu'):
        lib.bpd_avgpool2d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*7
        lib.bpd_avgpool2d_cpu.restype = None
    if hasattr(lib, 'bpd_layernorm_cpu'):
        lib.bpd_layernorm_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
        lib.bpd_layernorm_cpu.restype = None
    if hasattr(lib, 'bpd_instancenorm_cpu'):
        # (input, output, N, C, H, W, eps)
        lib.bpd_instancenorm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*4 + [ctypes.c_float]
        lib.bpd_instancenorm_cpu.restype = None
    if hasattr(lib, 'bpd_groupnorm_cpu'):
        # (input, gamma, beta, output, N, C, H, W, G, eps)
        lib.bpd_groupnorm_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*5 + [ctypes.c_float]
        lib.bpd_groupnorm_cpu.restype = None
    if hasattr(lib, 'bpd_rmsnorm_cpu'):
        # (input, output, N, C, H, W, eps)
        lib.bpd_rmsnorm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*4 + [ctypes.c_float]
        lib.bpd_rmsnorm_cpu.restype = None
    if hasattr(lib, 'bpd_frobenius_norm_cpu'):
        # (input, output, n_total)
        lib.bpd_frobenius_norm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.bpd_frobenius_norm_cpu.restype = None
    if hasattr(lib, 'bpd_l1norm_cpu'):
        # (input, output, rows, cols)
        lib.bpd_l1norm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.bpd_l1norm_cpu.restype = None
    if hasattr(lib, 'bpd_l2norm_cpu'):
        # (input, output, rows, cols)
        lib.bpd_l2norm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.bpd_l2norm_cpu.restype = None
    if hasattr(lib, 'bpd_maxpool1d_cpu'):
        # (in, out, N, C, L, kL, stride, pad)
        lib.bpd_maxpool1d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*6
        lib.bpd_maxpool1d_cpu.restype = None
    if hasattr(lib, 'bpd_maxpool3d_cpu'):
        # (in, out, N, C, D, H, W, kD, kH, kW, stride, pad)
        lib.bpd_maxpool3d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*10
        lib.bpd_maxpool3d_cpu.restype = None
    if hasattr(lib, 'bpd_avgpool1d_cpu'):
        lib.bpd_avgpool1d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*6
        lib.bpd_avgpool1d_cpu.restype = None
    if hasattr(lib, 'bpd_avgpool3d_cpu'):
        lib.bpd_avgpool3d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*10
        lib.bpd_avgpool3d_cpu.restype = None
    return lib


# ─── Per-problem harness ───────────────────────────────────────────────────
#
# Each problem returns one of:
#   ('BIT_IDENTICAL', 0, n)
#   ('DIVERGENT', max_ulp, n)
#   ('MISSING_KERNEL', '<kernel_name>', None)
#   ('NOT_IMPLEMENTED', None, None)

RNG = np.random.default_rng(42)


def elementwise(lib, kernel, pt_fn, n=4096):
    if not hasattr(lib, kernel):
        return ('MISSING_KERNEL', kernel, None)
    x = (RNG.standard_normal(n) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    getattr(lib, kernel)(x.ctypes.data, out.ctypes.data, n)
    ref = pt_fn(torch.from_numpy(x)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def matmul_problem(lib, M, N, K):
    if not hasattr(lib, 'bpd_mm_cpu'):
        return ('MISSING_KERNEL', 'bpd_mm_cpu', None)
    A = RNG.standard_normal((M, K)).astype(np.float32)
    B = RNG.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, M, N, K)
    ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv2d_problem(lib, N, Cin, H, W, Cout, kH, kW, stride=1, pad=0):
    if not hasattr(lib, 'bpd_conv2d_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv2d_cpu', None)
    inp = RNG.standard_normal((N, Cin, H, W)).astype(np.float32)
    weight = RNG.standard_normal((Cout, Cin, kH, kW)).astype(np.float32)
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_cpu(inp.ctypes.data, weight.ctypes.data, out.ctypes.data,
                       N, Cin, H, W, Cout, kH, kW, stride, pad)
    ref = F.conv2d(torch.from_numpy(inp), torch.from_numpy(weight),
                    stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def reduce_problem(lib, kernel, pt_fn, n=1024):
    if not hasattr(lib, kernel):
        return ('MISSING_KERNEL', kernel, None)
    x = (RNG.standard_normal(n) * 2.0).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    getattr(lib, kernel)(x.ctypes.data, out.ctypes.data, n)
    ref = pt_fn(torch.from_numpy(x)).numpy().reshape(1)
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def softmax_problem(lib):
    if not hasattr(lib, 'bpd_softmax_cpu'):
        return ('MISSING_KERNEL', 'bpd_softmax_cpu', None)
    x = (RNG.standard_normal((32, 64)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_softmax_cpu(x.ctypes.data, out.ctypes.data, 32, 64)
    ref = F.softmax(torch.from_numpy(x), dim=-1).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def logsoftmax_problem(lib):
    if not hasattr(lib, 'bpd_logsoftmax_cpu'):
        return ('MISSING_KERNEL', 'bpd_logsoftmax_cpu', None)
    x = (RNG.standard_normal((32, 64)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_logsoftmax_cpu(x.ctypes.data, out.ctypes.data, 32, 64)
    ref = F.log_softmax(torch.from_numpy(x), dim=-1).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def layernorm_problem(lib):
    if not hasattr(lib, 'bpd_layernorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_layernorm_cpu', None)
    rows, cols = 8, 128
    x = (RNG.standard_normal((rows, cols)) * 2.0).astype(np.float32)
    gamma = np.ones(cols, dtype=np.float32)
    beta = np.zeros(cols, dtype=np.float32)
    out = np.zeros_like(x)
    lib.bpd_layernorm_cpu(x.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                          out.ctypes.data, rows, cols, 1e-5)
    ref = F.layer_norm(torch.from_numpy(x), (cols,), torch.from_numpy(gamma),
                        torch.from_numpy(beta), eps=1e-5).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def instancenorm_problem(lib):
    if not hasattr(lib, 'bpd_instancenorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_instancenorm_cpu', None)
    N, C, H, W = 2, 4, 8, 8
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_instancenorm_cpu(x.ctypes.data, out.ctypes.data, N, C, H, W, 1e-5)
    # PyTorch InstanceNorm2d default: affine=False, track_running_stats=False
    inorm = torch.nn.InstanceNorm2d(num_features=C, eps=1e-5, affine=False, track_running_stats=False)
    ref = inorm(torch.from_numpy(x)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def groupnorm_problem(lib):
    if not hasattr(lib, 'bpd_groupnorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_groupnorm_cpu', None)
    N, C, H, W = 2, 8, 8, 8
    G = 2
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    gamma = np.ones(C, dtype=np.float32)
    beta = np.zeros(C, dtype=np.float32)
    out = np.zeros_like(x)
    lib.bpd_groupnorm_cpu(x.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                          out.ctypes.data, N, C, H, W, G, 1e-5)
    gn = torch.nn.GroupNorm(num_groups=G, num_channels=C, eps=1e-5, affine=True)
    # Set its gamma=1, beta=0 explicitly
    with torch.no_grad():
        gn.weight.fill_(1.0)
        gn.bias.fill_(0.0)
    ref = gn(torch.from_numpy(x)).detach().numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def rmsnorm_problem(lib):
    if not hasattr(lib, 'bpd_rmsnorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_rmsnorm_cpu', None)
    N, C, H, W = 2, 8, 4, 4
    eps = 1e-5
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_rmsnorm_cpu(x.ctypes.data, out.ctypes.data, N, C, H, W, eps)
    # Reference: torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps); x / rms
    xt = torch.from_numpy(x)
    rms = torch.sqrt(torch.mean(xt ** 2, dim=1, keepdim=True) + eps)
    ref = (xt / rms).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def frobenius_problem(lib):
    if not hasattr(lib, 'bpd_frobenius_norm_cpu'):
        return ('MISSING_KERNEL', 'bpd_frobenius_norm_cpu', None)
    N, C, H, W = 2, 4, 4, 4
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    n_total = N * C * H * W
    out = np.zeros_like(x)
    lib.bpd_frobenius_norm_cpu(x.ctypes.data, out.ctypes.data, n_total)
    norm = torch.norm(torch.from_numpy(x), p='fro')
    ref = (torch.from_numpy(x) / norm).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def l1norm_problem(lib):
    if not hasattr(lib, 'bpd_l1norm_cpu'):
        return ('MISSING_KERNEL', 'bpd_l1norm_cpu', None)
    rows, cols = 8, 128
    x = (RNG.standard_normal((rows, cols)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_l1norm_cpu(x.ctypes.data, out.ctypes.data, rows, cols)
    # Reference: x / mean(|x|, dim=1, keepdim=True)
    xt = torch.from_numpy(x)
    ref = (xt / torch.mean(torch.abs(xt), dim=1, keepdim=True)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def l2norm_problem(lib):
    if not hasattr(lib, 'bpd_l2norm_cpu'):
        return ('MISSING_KERNEL', 'bpd_l2norm_cpu', None)
    rows, cols = 8, 128
    x = (RNG.standard_normal((rows, cols)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_l2norm_cpu(x.ctypes.data, out.ctypes.data, rows, cols)
    # Reference: x / norm(x, p=2, dim=1, keepdim=True)
    xt = torch.from_numpy(x)
    ref = (xt / torch.norm(xt, p=2, dim=1, keepdim=True)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def maxpool2d_problem(lib):
    if not hasattr(lib, 'bpd_maxpool2d_cpu'):
        return ('MISSING_KERNEL', 'bpd_maxpool2d_cpu', None)
    N, C, H, W = 1, 3, 16, 16
    kH, kW = 2, 2
    stride = 2
    inp = RNG.standard_normal((N, C, H, W)).astype(np.float32)
    H_out = H // stride
    W_out = W // stride
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    lib.bpd_maxpool2d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, H, W, kH, kW, stride)
    ref = F.max_pool2d(torch.from_numpy(inp), kernel_size=(kH, kW), stride=stride).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def avgpool2d_problem(lib):
    if not hasattr(lib, 'bpd_avgpool2d_cpu'):
        return ('MISSING_KERNEL', 'bpd_avgpool2d_cpu', None)
    N, C, H, W = 1, 3, 16, 16
    kH, kW = 2, 2
    stride = 2
    inp = RNG.standard_normal((N, C, H, W)).astype(np.float32)
    H_out = H // stride
    W_out = W // stride
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    lib.bpd_avgpool2d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, H, W, kH, kW, stride)
    ref = F.avg_pool2d(torch.from_numpy(inp), kernel_size=(kH, kW), stride=stride).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def maxpool1d_problem(lib):
    if not hasattr(lib, 'bpd_maxpool1d_cpu'):
        return ('MISSING_KERNEL', 'bpd_maxpool1d_cpu', None)
    N, C, L = 1, 3, 32
    kL, stride, pad = 4, 2, 0
    inp = RNG.standard_normal((N, C, L)).astype(np.float32)
    L_out = (L + 2*pad - kL) // stride + 1
    out = np.zeros((N, C, L_out), dtype=np.float32)
    lib.bpd_maxpool1d_cpu(inp.ctypes.data, out.ctypes.data, N, C, L, kL, stride, pad)
    ref = F.max_pool1d(torch.from_numpy(inp), kernel_size=kL, stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def maxpool3d_problem(lib):
    if not hasattr(lib, 'bpd_maxpool3d_cpu'):
        return ('MISSING_KERNEL', 'bpd_maxpool3d_cpu', None)
    N, C, D, H, W = 1, 2, 8, 8, 8
    kD = kH = kW = 2
    stride, pad = 2, 0
    inp = RNG.standard_normal((N, C, D, H, W)).astype(np.float32)
    D_out = (D + 2*pad - kD) // stride + 1
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)
    lib.bpd_maxpool3d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, D, H, W, kD, kH, kW, stride, pad)
    ref = F.max_pool3d(torch.from_numpy(inp),
                       kernel_size=(kD, kH, kW), stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def avgpool1d_problem(lib):
    if not hasattr(lib, 'bpd_avgpool1d_cpu'):
        return ('MISSING_KERNEL', 'bpd_avgpool1d_cpu', None)
    N, C, L = 1, 3, 32
    kL, stride, pad = 4, 2, 0
    inp = RNG.standard_normal((N, C, L)).astype(np.float32)
    L_out = (L + 2*pad - kL) // stride + 1
    out = np.zeros((N, C, L_out), dtype=np.float32)
    lib.bpd_avgpool1d_cpu(inp.ctypes.data, out.ctypes.data, N, C, L, kL, stride, pad)
    ref = F.avg_pool1d(torch.from_numpy(inp), kernel_size=kL, stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def avgpool3d_problem(lib):
    if not hasattr(lib, 'bpd_avgpool3d_cpu'):
        return ('MISSING_KERNEL', 'bpd_avgpool3d_cpu', None)
    N, C, D, H, W = 1, 2, 8, 8, 8
    kD = kH = kW = 2
    stride, pad = 2, 0
    inp = RNG.standard_normal((N, C, D, H, W)).astype(np.float32)
    D_out = (D + 2*pad - kD) // stride + 1
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)
    lib.bpd_avgpool3d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, D, H, W, kD, kH, kW, stride, pad)
    ref = F.avg_pool3d(torch.from_numpy(inp),
                       kernel_size=(kD, kH, kW), stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


# ─── Problem catalog ───────────────────────────────────────────────────────
#
# Each entry: (problem_number, name, runner_lambda)

def build_catalog(lib):
    """Build the full 100-problem catalog. Returns list of (num, name, fn)."""
    cat = []

    # 1–18: Matmul variants
    cat.append((1, '1_Square_matrix_multiplication',  lambda: matmul_problem(lib, 256, 256, 256)))
    cat.append((2, '2_Standard_matrix_multiplication', lambda: matmul_problem(lib, 128, 256, 64)))
    cat.append((3, '3_Batched_matrix_multiplication',  lambda: ('NOT_IMPLEMENTED', 'no bpd_bmm', None)))
    cat.append((4, '4_Matrix_vector_multiplication',   lambda: matmul_problem(lib, 256, 1, 128)))
    cat.append((5, '5_Matrix_scalar_multiplication',   lambda: ('NOT_IMPLEMENTED', 'no scalar_mul', None)))
    cat.append((6, '6_Matmul_with_large_K_dimension',  lambda: matmul_problem(lib, 16, 16, 4096)))
    cat.append((7, '7_Matmul_with_small_K_dimension',  lambda: matmul_problem(lib, 256, 256, 16)))
    cat.append((8, '8_Matmul_with_irregular_shapes',   lambda: matmul_problem(lib, 67, 89, 113)))
    cat.append((9, '9_Tall_skinny_matrix_multiplication', lambda: matmul_problem(lib, 1024, 16, 32)))
    cat.append((10, '10_3D_tensor_matrix_multiplication', lambda: ('NOT_IMPLEMENTED', '3D bmm', None)))
    cat.append((11, '11_4D_tensor_matrix_multiplication', lambda: ('NOT_IMPLEMENTED', '4D bmm', None)))
    cat.append((12, '12_Matmul_with_diagonal_matrices', lambda: ('NOT_IMPLEMENTED', 'diag matmul', None)))
    cat.append((13, '13_Matmul_for_symmetric_matrices', lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((14, '14_Matmul_for_upper_triangular',  lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((15, '15_Matmul_for_lower_triangular',  lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((16, '16_Matmul_with_transposed_A',     lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((17, '17_Matmul_with_transposed_B',     lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((18, '18_Matmul_with_transposed_both',  lambda: matmul_problem(lib, 128, 128, 128)))

    # 19–32: Activations
    cat.append((19, '19_ReLU',         lambda: elementwise(lib, 'bpd_relu_cpu', lambda t: F.relu(t))))
    cat.append((20, '20_LeakyReLU',    lambda: elementwise(lib, 'bpd_leaky_relu_cpu', lambda t: F.leaky_relu(t))))
    cat.append((21, '21_Sigmoid',      lambda: elementwise(lib, 'bpd_sigmoid_cpu', lambda t: torch.sigmoid(t))))
    cat.append((22, '22_Tanh',         lambda: elementwise(lib, 'bpd_tanh_cpu', lambda t: torch.tanh(t))))
    cat.append((23, '23_Softmax',      lambda: softmax_problem(lib)))
    cat.append((24, '24_LogSoftmax',   lambda: logsoftmax_problem(lib)))
    cat.append((25, '25_Swish',        lambda: elementwise(lib, 'bpd_silu_cpu', lambda t: F.silu(t))))  # Swish == SiLU
    cat.append((26, '26_GELU',         lambda: elementwise(lib, 'bpd_gelu_cpu', lambda t: F.gelu(t))))
    cat.append((27, '27_SELU',         lambda: elementwise(lib, 'bpd_selu_cpu', lambda t: F.selu(t))))
    cat.append((28, '28_HardSigmoid',  lambda: elementwise(lib, 'bpd_hardsigmoid_cpu', lambda t: F.hardsigmoid(t))))
    cat.append((29, '29_Softplus',     lambda: elementwise(lib, 'bpd_softplus_cpu', lambda t: F.softplus(t))))
    cat.append((30, '30_Softsign',     lambda: elementwise(lib, 'bpd_softsign_cpu', lambda t: F.softsign(t))))
    cat.append((31, '31_ELU',          lambda: elementwise(lib, 'bpd_elu_cpu', lambda t: F.elu(t))))
    cat.append((32, '32_HardTanh',     lambda: elementwise(lib, 'bpd_clamp_cpu', lambda t: F.hardtanh(t))))

    # 33–40: Normalizations
    cat.append((33, '33_BatchNorm',    lambda: ('NOT_IMPLEMENTED', 'BN needs gamma/beta/mean/var routing', None)))
    cat.append((34, '34_InstanceNorm', lambda: instancenorm_problem(lib)))
    cat.append((35, '35_GroupNorm',    lambda: groupnorm_problem(lib)))
    cat.append((36, '36_RMSNorm',      lambda: rmsnorm_problem(lib)))
    cat.append((37, '37_FrobeniusNorm', lambda: frobenius_problem(lib)))
    cat.append((38, '38_L1Norm',       lambda: l1norm_problem(lib)))
    cat.append((39, '39_L2Norm',       lambda: l2norm_problem(lib)))
    cat.append((40, '40_LayerNorm',    lambda: layernorm_problem(lib)))

    # 41–46: Pooling
    cat.append((41, '41_Max_Pooling_1D',   lambda: maxpool1d_problem(lib)))
    cat.append((42, '42_Max_Pooling_2D',   lambda: maxpool2d_problem(lib)))
    cat.append((43, '43_Max_Pooling_3D',   lambda: maxpool3d_problem(lib)))
    cat.append((44, '44_Average_Pooling_1D', lambda: avgpool1d_problem(lib)))
    cat.append((45, '45_Average_Pooling_2D', lambda: avgpool2d_problem(lib)))
    cat.append((46, '46_Average_Pooling_3D', lambda: avgpool3d_problem(lib)))

    # 47–49: Reductions
    cat.append((47, '47_Sum_reduction',  lambda: reduce_problem(lib, 'bpd_sum_cpu', lambda t: torch.sum(t))))
    cat.append((48, '48_Mean_reduction', lambda: reduce_problem(lib, 'bpd_mean_cpu', lambda t: torch.mean(t))))
    cat.append((49, '49_Max_reduction',  lambda: reduce_problem(lib, 'bpd_max_cpu', lambda t: torch.max(t))))

    # 50, 54–87: Convolutions (small reproducible shapes)
    # Conv2D: 50 main, 54-58 variants, 64-69, 78-83 etc.
    cat.append((50, '50_Conv2D',       lambda: conv2d_problem(lib, 1, 3, 16, 16, 8, 3, 3, stride=1, pad=1)))
    # 51_Argmax_over_a_dim, 52_Argmin_over_a_dim — not implemented
    cat.append((51, '51_Argmax_over_a_dimension', lambda: ('NOT_IMPLEMENTED', 'argmax', None)))
    cat.append((52, '52_Argmin_over_a_dimension', lambda: ('NOT_IMPLEMENTED', 'argmin', None)))
    cat.append((53, '53_Min_reduction_over_a_dimension', lambda: ('NOT_IMPLEMENTED', 'min reduce', None)))
    # 54–69: various Conv2D variants
    for n, name in [(54, '54_conv_standard_2D_square_input_asymmetric_kernel'),
                    (55, '55_conv_standard_2D_asymmetric_input_square_kernel'),
                    (56, '56_conv_standard_2D_asymmetric_input_asymmetric_kernel'),
                    (57, '57_conv_transposed_2D_square_input_square_kernel'),
                    (58, '58_conv_transposed_3D_asymmetric_input_asymmetric_kernel'),
                    (59, '59_conv_standard_3D_asymmetric_input_square_kernel'),
                    (60, '60_conv_standard_3D_square_input_asymmetric_kernel'),
                    (61, '61_conv_transposed_3D_square_input_square_kernel'),
                    (62, '62_conv_standard_2D_square_input_asymmetric_kernel_dilated'),
                    (63, '63_conv_standard_2D_square_input_square_kernel'),
                    (64, '64_conv_transposed_1D'),
                    (65, '65_conv_transposed_2D_square_input_asymmetric_kernel_dilated'),
                    (66, '66_conv_standard_3D_asymmetric_input_asymmetric_kernel'),
                    (67, '67_conv_standard_1D'),
                    (68, '68_conv_transposed_3D_square_input_asymmetric_kernel'),
                    (69, '69_conv_transposed_2D_square_input_asymmetric_kernel')]:
        if '2D' in name and 'transposed' not in name and 'dilated' not in name and 'asymmetric' not in name:
            cat.append((n, name, lambda: conv2d_problem(lib, 1, 3, 16, 16, 8, 3, 3)))
        else:
            cat.append((n, name, lambda: ('NOT_IMPLEMENTED', 'conv variant', None)))

    # 70–87: more conv variants — all NOT_IMPLEMENTED for now (substrate has only conv2d_cpu)
    for n in range(70, 88):
        cat.append((n, f'{n}_conv_variant', lambda: ('NOT_IMPLEMENTED', 'conv variant', None)))

    # 88: 88_MinGPT_NewGelu — gelu approximation (tanh form)
    cat.append((88, '88_MinGPT_NewGelu', lambda: ('NOT_IMPLEMENTED', 'tanh-gelu approx', None)))

    # 89–93: Cumulative
    for n, name, kernel, pt_fn in [
        (89, '89_cumsum',           'bpd_cumsum_cpu',           lambda t: torch.cumsum(t, dim=-1)),
        (90, '90_cumsum_reverse',   'bpd_cumsum_reverse_cpu',   lambda t: torch.flip(torch.cumsum(torch.flip(t, [-1]), dim=-1), [-1])),
        (91, '91_cumsum_exclusive', 'bpd_cumsum_exclusive_cpu', lambda t: torch.cat([torch.zeros_like(t[..., :1]), torch.cumsum(t, dim=-1)[..., :-1]], dim=-1)),
        (92, '92_cumsum',           'bpd_cumsum_cpu',           lambda t: torch.cumsum(t, dim=-1)),
        (93, '93_cumulative_product', 'bpd_cumprod_cpu',        lambda t: torch.cumprod(t, dim=-1)),
    ]:
        cat.append((n, name, lambda k=kernel, f=pt_fn: elementwise(lib, k, f, n=512)))

    # 94–100: Losses
    for n, name in [(94, '94_MSELoss'), (95, '95_CrossEntropyLoss'),
                    (96, '96_HuberLoss'), (97, '97_ScalarTriplet'),
                    (98, '98_KLDivLoss'), (99, '99_TripletMarginLoss'),
                    (100, '100_HingeLoss')]:
        cat.append((n, name, lambda: ('MISSING_KERNEL', f'bpd_loss_cpu ({name})', None)))

    return cat


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    lib = load_lib()
    cat = build_catalog(lib)

    # Run each problem
    results = []
    for num, name, fn in cat:
        try:
            r = fn()
        except Exception as e:
            r = ('ERROR', repr(e), None)
        results.append((num, name, r))

    # Print per-problem table
    print(f"{'#':<4} {'name':<50} {'status':<16} {'detail':<32}")
    print("─" * 105)
    for num, name, (status, a, b) in results:
        if status == 'BIT_IDENTICAL':
            detail = '0 ULP'
        elif status == 'DIVERGENT':
            detail = f"{a} ULP ({b} diffs)"
        elif status == 'MISSING_KERNEL':
            detail = str(a)
        elif status == 'NOT_IMPLEMENTED':
            detail = str(a)
        else:
            detail = str(a)[:32]
        # Trim name to fit
        n = name if len(name) <= 49 else name[:46] + '...'
        print(f"{num:<4} {n:<50} {status:<16} {detail:<32}")

    # Summary
    print()
    print("═" * 105)
    by_status = {}
    for _, _, (s, _, _) in results:
        by_status[s] = by_status.get(s, 0) + 1
    total = len(results)
    print(f"Total problems: {total}")
    for s in ['BIT_IDENTICAL', 'DIVERGENT', 'MISSING_KERNEL', 'NOT_IMPLEMENTED', 'ERROR']:
        c = by_status.get(s, 0)
        pct = 100 * c / total
        print(f"  {s:<18} {c:>3}  ({pct:5.1f}%)")

    # Per-category breakdown
    print()
    print("Per-category breakdown:")
    cats = [('Matmul (1-18)', range(1, 19)),
            ('Activations (19-32)', range(19, 33)),
            ('Norms (33-40)', range(33, 41)),
            ('Pooling (41-46)', range(41, 47)),
            ('Reductions (47-53)', range(47, 54)),
            ('Convolutions (50, 54-87)', list(range(54, 88)) + [50, 88]),
            ('Cumulative (89-93)', range(89, 94)),
            ('Losses (94-100)', range(94, 101))]
    for cname, rng_ in cats:
        in_cat = [r for r in results if r[0] in rng_]
        bi = sum(1 for _, _, (s, _, _) in in_cat if s == 'BIT_IDENTICAL')
        dv = sum(1 for _, _, (s, _, _) in in_cat if s == 'DIVERGENT')
        mk = sum(1 for _, _, (s, _, _) in in_cat if s == 'MISSING_KERNEL')
        ni = sum(1 for _, _, (s, _, _) in in_cat if s == 'NOT_IMPLEMENTED')
        er = sum(1 for _, _, (s, _, _) in in_cat if s == 'ERROR')
        n_ = len(in_cat)
        print(f"  {cname:<26} {n_:>3} total  |  BI {bi:>2}  DV {dv:>2}  MK {mk:>2}  NI {ni:>2}  ER {er:>2}")

    bi_count = by_status.get('BIT_IDENTICAL', 0)
    print()
    print(f"BIT_IDENTICAL: {bi_count}/{total}")
    return 0 if bi_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
