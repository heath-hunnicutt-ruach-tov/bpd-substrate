import numpy as np
import ctypes
import sys
sys.path.insert(0, "bench")
try:
    from llama_fixture_loader import get_sources
except ImportError:
    def get_sources(tensors, t): return []

c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_int32_p = ctypes.POINTER(ctypes.c_int32)


def compare(our, ref):
    our_flat = np.ascontiguousarray(our, dtype=np.float32).flatten()
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).flatten()
    if our_flat.shape != ref_flat.shape:
        return {"status": "fail", "reason": f"shape mismatch: {our_flat.shape} vs {ref_flat.shape}"}
    our_bits = our_flat.view(np.int32)
    ref_bits = ref_flat.view(np.int32)
    diffs = np.abs(our_bits.astype(np.int64) - ref_bits.astype(np.int64))
    max_ulp = int(diffs.max()) if diffs.size > 0 else 0
    n_diffs = int((diffs > 0).sum())
    max_abs = float(np.abs(our_flat - ref_flat).max())
    if max_ulp == 0:
        return {"status": "pass", "max_ulp": 0, "n_diffs": 0, "n_total": len(our_flat)}
    return {"status": "fail", "max_ulp": max_ulp, "n_diffs": n_diffs,
            "n_total": len(our_flat), "max_abs": max_abs}


def verify_add(lib, tensors, op, idx, ctx):
    """ADD: element-wise residual addition.
    
    In the transformer:
      ffn_inp-N  = attn_out-N + inp_embd (or prev layer out)  
      result-N   = ffn_out-N + ffn_inp-N
    
    Inputs found by name convention:
      ffn_inp-N: sources are attn_out-N (MUL_MAT) and the residual
      result-N:  sources are down-proj output and ffn_inp-N
    
    Since ADD is trivial (no rounding choices), we can verify by
    computing a + b and comparing to ref. But without src links,
    we verify that ref == a + b by finding the two most recent
    tensors before this ADD that have matching shapes.
    """
    if not hasattr(lib, 'bpd_add_f32_cpu'):
        return {"status": "skip", "reason": "bpd_add_f32_cpu not in substrate"}
    
    if not hasattr(lib.bpd_add_f32_cpu, '_argtypes_set'):
        lib.bpd_add_f32_cpu.restype = None
        lib.bpd_add_f32_cpu.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int]
        lib.bpd_add_f32_cpu._argtypes_set = True

    ref = op.as_numpy()
    ref_shape = tuple(ref.shape)
    n = ref.size

    # Find the two source tensors: walk backwards from idx to find
    # the two most recent tensors with matching total element count
    # Use explicit source links if available (issue #56)
    sources = get_sources(tensors, op)
    if len(sources) >= 2:
        candidates = sources[:2]
    else:
        # Fallback: backward walk
        candidates = []
        for t in reversed(tensors[:idx]):
            if t.as_numpy().size == n and t.op_desc not in ("RESHAPE", "VIEW", "PERMUTE", "CONT", "TRANSPOSE"):
                candidates.append(t)
                if len(candidates) >= 2:
                    break
    
    if len(candidates) < 2:
        return {"status": "skip", "reason": f"could not find 2 source tensors for ADD at idx {idx}"}
    
    a = np.ascontiguousarray(candidates[0].as_numpy(), dtype=np.float32).flatten()
    b = np.ascontiguousarray(candidates[1].as_numpy(), dtype=np.float32).flatten()
    
    out = np.zeros(n, dtype=np.float32)
    lib.bpd_add_f32_cpu(
        a.ctypes.data_as(c_float_p),
        b.ctypes.data_as(c_float_p),
        out.ctypes.data_as(c_float_p),
        ctypes.c_int(n))
    
    return compare(out.reshape(ref.shape), ref)


def verify_silu(lib, tensors, op, idx, ctx):
    """SILU: x * sigmoid(x).
    
    In SwiGLU FFN: gate = silu(x @ W_gate)
    Source: the MUL_MAT output immediately before this SILU.
    """
    if not hasattr(lib, 'bpd_silu_f32_cpu'):
        return {"status": "skip", "reason": "bpd_silu_f32_cpu not in substrate"}
    
    if not hasattr(lib.bpd_silu_f32_cpu, '_argtypes_set'):
        lib.bpd_silu_f32_cpu.restype = None
        lib.bpd_silu_f32_cpu.argtypes = [c_float_p, c_float_p, ctypes.c_int]
        lib.bpd_silu_f32_cpu._argtypes_set = True

    ref = op.as_numpy()
    n = ref.size

    # Find source: walk backwards to find the matching-size tensor
    # SILU's input is the gate projection output (a MUL_MAT)
    # Use explicit source links if available (issue #56)
    sources = get_sources(tensors, op)
    if sources:
        src = sources[0]
    else:
        # Fallback: backward walk
        src = None
        for t in reversed(tensors[:idx]):
            if t.as_numpy().size == n and t.op_desc not in ("RESHAPE", "VIEW", "PERMUTE", "CONT", "TRANSPOSE"):
                src = t
                break
    
    if src is None:
        return {"status": "skip", "reason": f"could not find source for SILU at idx {idx}"}
    
    inp = np.ascontiguousarray(src.as_numpy(), dtype=np.float32).flatten()
    out = np.zeros(n, dtype=np.float32)
    lib.bpd_silu_f32_cpu(
        inp.ctypes.data_as(c_float_p),
        out.ctypes.data_as(c_float_p),
        ctypes.c_int(n))
    
    return compare(out.reshape(ref.shape), ref)


def verify_soft_max(lib, tensors, op, idx, ctx):
    """SOFT_MAX: row-wise softmax (causal-masked in attention).
    
    Source: QK^T scores (the MUL_MAT output before this SOFT_MAX,
    after reshape/permute/scale).
    """
    if not hasattr(lib, 'bpd_softmax_causal_cpu'):
        return {"status": "skip", "reason": "bpd_softmax_causal_cpu not in substrate"}
    
    if not hasattr(lib.bpd_softmax_causal_cpu, '_argtypes_set'):
        lib.bpd_softmax_causal_cpu.restype = None
        lib.bpd_softmax_causal_cpu.argtypes = [
            c_float_p, c_float_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.bpd_softmax_causal_cpu._argtypes_set = True

    ref = op.as_numpy()

    # Find source tensor (QK^T scores)
    src = None
    for t in reversed(tensors[:idx]):
        if t.as_numpy().shape == ref.shape:
            src = t
            break
    
    if src is None:
        return {"status": "skip", "reason": f"could not find source for SOFT_MAX at idx {idx}"}
    
    inp = np.ascontiguousarray(src.as_numpy(), dtype=np.float32)
    
    # Flatten to 2D: (total_rows, cols)
    if inp.ndim == 1:
        total_rows, cols = 1, inp.size
    elif inp.ndim == 2:
        total_rows, cols = inp.shape
    elif inp.ndim >= 3:
        cols = inp.shape[-1]
        total_rows = inp.size // cols
    else:
        return {"status": "skip", "reason": f"unexpected SOFT_MAX ndim={inp.ndim}"}
    
    inp_flat = inp.reshape(total_rows, cols).copy()
    out = np.zeros_like(inp_flat)
    
    lib.bpd_softmax_causal_cpu(
        inp_flat.ctypes.data_as(c_float_p),
        out.ctypes.data_as(c_float_p),
        ctypes.c_int(total_rows), ctypes.c_int(cols),
        ctypes.c_int(1))  # is_causal=1
    
    return compare(out.reshape(ref.shape), ref)


def verify_cpy(lib, tensors, op, idx, ctx):
    """CPY: tensor copy (KV cache writes, type conversions).
    
    For F32->F32, CPY is just memcpy (possibly with reshape).
    Source: the tensor immediately before this CPY with matching size.
    """
    ref = op.as_numpy()
    n = ref.size
    
    # Use explicit source links if available (issue #56)
    sources = get_sources(tensors, op)
    if sources:
        src = sources[0]
    else:
        # Fallback: backward walk
        src = None
        for t in reversed(tensors[:idx]):
            if t.as_numpy().size == n and t.op_desc not in ("RESHAPE", "VIEW", "PERMUTE", "CONT", "TRANSPOSE"):
                src = t
                break
    
    if src is None:
        return {"status": "skip", "reason": f"could not find source for CPY at idx {idx}"}
    
    inp = np.ascontiguousarray(src.as_numpy(), dtype=np.float32).flatten()
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).flatten()
    
    # For F32->F32 CPY, output should be bitwise identical to input
    return compare(inp, ref_flat)
