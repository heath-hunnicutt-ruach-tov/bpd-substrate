#!/usr/bin/env python3
"""lift_chain.py — auto-LIFT a KernelBench problem's forward method
into chain facts for the bridge (bridge_emit.pl).

The lifter parses the forward() body's op sequence (the x = ... lines)
into chain(Problem, [op1(args), op2(args), ...]) Prolog facts.
Covers the elementwise vocabulary; reduction/matrix ops are recognized
and marked (they select the buffered-row/whole-matrix patterns).
ZERO per-problem hand-work: the same parser lifts any problem file.
"""
import ast
import re
import sys


def lift(path):
    """Parse a KernelBench problem file → (name, [ops]) or raise."""
    src = open(path).read()
    tree = ast.parse(src)
    # find Model.forward:
    fwd = None
    init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Model':
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == 'forward':
                        fwd = item
                    elif item.name == '__init__':
                        init = item
    if fwd is None:
        raise ValueError('no Model.forward')
    # module-level constants (divisor = 2.0 etc):
    consts = {}

    def resolve_const(e):
        """literal_eval extended: resolves Names via consts (in order),
        chained assigns (a = b = 1), and tuples of both."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.Name) and e.id in consts:
            return consts[e.id]
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            v = resolve_const(e.operand)
            return -v if v is not None else None
        if isinstance(e, (ast.Tuple, ast.List)):
            elems = []
            for el in e.elts:
                r = resolve_const(el)
                if r is None:
                    return None
                elems.append(r)
            return tuple(elems)
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            val = resolve_const(node.value)
            if val is None:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    consts[tgt.id] = val
                elif isinstance(tgt, ast.Tuple) and \
                     isinstance(val, tuple) and \
                     len(tgt.elts) == len(val):
                    # height, width = 384, 384 (#65's form):
                    for el, vv in zip(tgt.elts, val):
                        if isinstance(el, ast.Name):
                            consts[el.id] = vv
    # resolve __init__ parameter values: signature defaults overridden by
    # get_init_inputs() positional mapping (the KernelBench harness
    # convention — proven from source, not assumed):
    init_args = {}
    if init:
        params = [a.arg for a in init.args.args[1:]]   # skip self
        defaults = init.args.defaults
        # defaults align to the TAIL of params:
        for p, dflt in zip(params[len(params)-len(defaults):], defaults):
            try:
                init_args[p] = ast.literal_eval(dflt)
            except Exception:
                pass
        # get_init_inputs(): positional values (Names resolved via consts):
        gii = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and \
               node.name == 'get_init_inputs':
                gii = node
        if gii:
            for st in gii.body:
                if isinstance(st, ast.Return) and \
                   isinstance(st.value, ast.List):
                    for p, el in zip(params, st.value.elts):
                        if isinstance(el, ast.Constant):
                            init_args[p] = el.value
                        elif isinstance(el, ast.Name) and el.id in consts:
                            init_args[p] = consts[el.id]
    # make init-args visible to scalar_of via consts (params shadow module
    # consts inside __init__ scope):
    for k, v2 in init_args.items():
        if k not in consts:
            consts[k] = v2

    # __init__ self.X = value bindings (scalars only) + nn-module ops:
    attrs = {}
    scalar_params = set()   # self.X params that are provably scalar-shaped
    channel_params = set()  # (C,1,1)-shaped params: channel-broadcast
    stage_classes = {}      # self.X -> nn module class name (Linear/Conv2d/..)
    stage_widths = {}       # self.X -> output width (Linear out_features) if provable
    stage_chans = {}        # self.X -> out_channels (Conv*) if provable
    stage_spatial = {}      # self.X -> (kind, k, s, p) spatial transform
    module_red_dims = {}    # self.X -> dim arg for nn.Softmax(dim=..)
    STAGE_RANKS = {'Linear': 2, 'Gemm': 2, 'Bilinear': 2, 'Identity': None,
                   'Conv1d': 3, 'ConvTranspose1d': 3, 'BMM': 3,
                   'Conv2d': 4, 'ConvTranspose2d': 4,
                   'Conv3d': 5, 'ConvTranspose3d': 5,
                   # rank-PRESERVING stages (the #55/#98 root — absent
                   # classes nulled the rank at the stage; mavhir's
                   # trace + Mavdil's probes):
                   'MaxPool1d': 3, 'AvgPool1d': 3,
                   'MaxPool2d': 4, 'AvgPool2d': 4,
                   'MaxPool3d': 5, 'AvgPool3d': 5}
    module_ops = {}   # self.X -> op tuple, for nn activation modules
    param_shapes = {}  # self.X -> resolved dim list (the #14 width)
    NN_OP_MODULES = {
        'ReLU': ('relu',), 'Tanh': ('tanh',), 'Sigmoid': ('sigmoid',),
        'GELU': ('gelu',), 'Mish': ('mish',), 'Hardswish': ('hardswish',),
        'Hardtanh': ('hardtanh',), 'Softmax': ('reduction', 'softmax'),
        'LogSoftmax': ('reduction', 'log_softmax'),
    }
    if init:
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Attribute) and \
                   isinstance(t.value, ast.Name) and t.value.id == 'self':
                    v = node.value
                    if isinstance(v, ast.Call):
                        fn = v.func
                        cls = fn.attr if isinstance(fn, ast.Attribute) else \
                              (fn.id if isinstance(fn, ast.Name) else None)
                        # nn.Parameter(torch.randn(1,1,..)) / (torch.tensor(x))
                        # = SCALAR param (all dims literal 1, or 0-d tensor):
                        if cls == 'Parameter' and v.args:
                            inner = v.args[0]
                            # track the param's SHAPE (dims resolved
                            # via module-level consts — the #14
                            # matmul width):
                            if isinstance(inner, ast.Call):
                                _ifn = inner.func
                                _in2 = (_ifn.attr if isinstance(
                                    _ifn, ast.Attribute) else None)
                                if _in2 in ('randn', 'zeros', 'ones'):
                                    _dims2 = []
                                    for a in inner.args:
                                        if isinstance(a, ast.Constant):
                                            _dims2.append(a.value)
                                        elif isinstance(a, ast.Name) \
                                                and a.id in consts:
                                            _dims2.append(
                                                consts[a.id])
                                        else:
                                            _dims2 = None
                                            break
                                    if _dims2:
                                        param_shapes[t.attr] = _dims2
                            if isinstance(inner, ast.Call):
                                ifn = inner.func
                                iname = ifn.attr if isinstance(ifn, ast.Attribute) else None
                                if iname == 'tensor':
                                    scalar_params.add(t.attr)
                                elif iname in ('randn', 'zeros', 'ones'):
                                    dims = inner.args
                                    if dims and all(
                                        isinstance(a, ast.Constant) and a.value == 1
                                        for a in dims):
                                        scalar_params.add(t.attr)
                                    elif len(dims) == 1:
                                        shp = None
                                        if isinstance(dims[0], ast.Name) and \
                                           dims[0].id in consts:
                                            shp = consts[dims[0].id]
                                        elif isinstance(dims[0], ast.Tuple):
                                            elems = []
                                            for el in dims[0].elts:
                                                if isinstance(el, ast.Constant):
                                                    elems.append(el.value)
                                                elif isinstance(el, ast.Name) and \
                                                     el.id in consts:
                                                    elems.append(consts[el.id])
                                                else:
                                                    elems = None
                                                    break
                                            shp = tuple(elems) if elems else None
                                        if isinstance(shp, (tuple, list)):
                                            if all(d == 1 for d in shp):
                                                scalar_params.add(t.attr)
                                            elif shp[0] > 1 and \
                                                 all(d == 1 for d in shp[1:]):
                                                # (C,1,1)-style: CHANNEL param
                                                channel_params.add(t.attr)
                        if cls == 'LeakyReLU':
                            # Resolve the slope from a Constant OR a Name (an
                            # __init__ param reference against module consts).
                            # Doresh caught (#12): the old handler only read
                            # ast.Constant, so nn.LeakyReLU(negative_slope) with
                            # a Name arg silently fell through to the DEFAULT
                            # 0.01 instead of the configured 0.1 -> wrong
                            # hyperparameter, confidently wrong. Resolve Names
                            # via consts; if UNresolved, FLAG (never silently
                            # default).
                            def _slope_of(node):
                                if isinstance(node, ast.Constant):
                                    return float(node.value)
                                if isinstance(node, ast.Name) and node.id in consts:
                                    return float(consts[node.id])
                                return None  # unresolved -> flag, don't default
                            slope = None
                            src = None
                            if v.args:
                                slope = _slope_of(v.args[0]); src = v.args[0]
                            for kw in v.keywords:
                                if kw.arg == 'negative_slope':
                                    slope = _slope_of(kw.value); src = kw.value
                            if slope is None and src is not None:
                                # a slope was GIVEN but couldn't be resolved:
                                # flag it, don't silently use the default.
                                module_ops[t.attr] = (
                                    'unknown_call', 'leaky_relu_unresolved_slope')
                                continue
                            if slope is None:
                                slope = 0.01  # no arg given: PyTorch default IS 0.01
                            module_ops[t.attr] = ('leaky_relu', slope)
                            continue
                        if cls == 'Hardtanh' and (v.keywords or v.args):
                            # CUSTOM bounds (the #30 catch: nn.Hardtanh
                            # (min_val=m, max_val=M) ≠ the default
                            # (-1,1) clamp — resolve like leaky_relu's
                            # slope; flag-don't-default on unresolved):
                            lo, hi = -1.0, 1.0
                            ok = True
                            vals = {}
                            for kw in v.keywords:
                                if kw.arg in ('min_val', 'max_val'):
                                    try:
                                        vals[kw.arg] = ast.literal_eval(
                                            kw.value)
                                    except Exception:
                                        if isinstance(kw.value, ast.Name) \
                                           and kw.value.id in init_args:
                                            vals[kw.arg] = \
                                                init_args[kw.value.id]
                                        else:
                                            ok = False
                            for j, a in enumerate(v.args[:2]):
                                key = ('min_val', 'max_val')[j]
                                try:
                                    vals[key] = ast.literal_eval(a)
                                except Exception:
                                    if isinstance(a, ast.Name) and \
                                       a.id in init_args:
                                        vals[key] = init_args[a.id]
                                    else:
                                        ok = False
                            if not ok:
                                module_ops[t.attr] = (
                                    'unknown_call',
                                    'hardtanh_unresolved_bounds')
                                continue
                            lo = float(vals.get('min_val', lo))
                            hi = float(vals.get('max_val', hi))
                            module_ops[t.attr] = ('clamp', lo, hi)
                            continue
                        # AvgPool3d(kernel_size=2) → a LIFTABLE op
                        # (#72; the #49/#84 REGRESSION LESSON: this
                        # elif once severed the Softmax dim-tracking
                        # from its NN_OP_MODULES branch — order
                        # matters in this cascade):
                        if cls == 'AvgPool3d':
                            _k = None
                            if v.args and isinstance(v.args[0],
                                                     ast.Constant):
                                _k = v.args[0].value
                            for kw in v.keywords:
                                if kw.arg == 'kernel_size' and \
                                        isinstance(kw.value,
                                                   ast.Constant):
                                    _k = kw.value.value
                            if _k == 2 and not any(
                                    kw.arg in ('stride', 'padding')
                                    for kw in v.keywords):
                                module_ops[t.attr] = ('avgpool3d_k2',)
                        if cls in NN_OP_MODULES:
                            module_ops[t.attr] = NN_OP_MODULES[cls]
                            if cls in ('Softmax', 'LogSoftmax'):
                                d = None
                                for kw in v.keywords:
                                    if kw.arg == 'dim' and \
                                       isinstance(kw.value, ast.Constant):
                                        d = kw.value.value
                                if d is None and v.args and \
                                   isinstance(v.args[0], ast.Constant):
                                    d = v.args[0].value
                                module_red_dims[t.attr] = d
                            continue
                        if cls is not None:
                            stage_classes[t.attr] = cls
                            if cls in ('AdaptiveAvgPool1d',
                                       'AdaptiveAvgPool2d',
                                       'AdaptiveAvgPool3d',
                                       'AdaptiveMaxPool1d',
                                       'AdaptiveMaxPool2d',
                                       'AdaptiveMaxPool3d') and v.args:
                                try:
                                    osz = ast.literal_eval(v.args[0])
                                    if isinstance(osz, int):
                                        osz = (osz,)
                                    if isinstance(osz, (tuple, list)) and \
                                       all(isinstance(z, int) for z in osz):
                                        stage_spatial[t.attr] = \
                                            ('adaptive', tuple(osz), 0, 0,
                                             0)
                                except Exception:
                                    pass
                            if cls in ('Conv1d', 'Conv2d', 'Conv3d',
                                       'ConvTranspose1d', 'ConvTranspose2d',
                                       'ConvTranspose3d',
                                       'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
                                       'AvgPool1d', 'AvgPool2d', 'AvgPool3d'):
                                # spatial transform params (k, stride, pad):
                                def _cv(node, dflt=None):
                                    if node is None:
                                        return dflt
                                    try:
                                        val = ast.literal_eval(node)
                                    except Exception:
                                        if isinstance(node, ast.Name) and \
                                           node.id in consts:
                                            val = consts[node.id]
                                        else:
                                            return None
                                    return val
                                kw = {k.arg: k.value for k in v.keywords}
                                if cls.startswith(('Conv', 'ConvT')):
                                    kk = _cv(v.args[2] if len(v.args) > 2
                                             else kw.get('kernel_size'))
                                    ss = _cv(v.args[3] if len(v.args) > 3
                                             else kw.get('stride'), 1)
                                    pp = _cv(v.args[4] if len(v.args) > 4
                                             else kw.get('padding'), 0)
                                    # output_padding (the #44 catch:
                                    # convT out = (d-1)s-2p+k+OP —
                                    # missing OP made 255² of a 256²):
                                    op = _cv(v.args[5] if len(v.args) > 5
                                             else kw.get('output_padding'),
                                             0)
                                    kind = ('convT' if 'Transpose' in cls
                                            else 'conv')
                                else:
                                    kk = _cv(v.args[0] if len(v.args) > 0
                                             else kw.get('kernel_size'))
                                    ss = _cv(v.args[1] if len(v.args) > 1
                                             else kw.get('stride'), None)
                                    if ss is None:
                                        ss = kk  # pool default stride = k
                                    pp = _cv(kw.get('padding'), 0)
                                    op = 0
                                    kind = 'pool'
                                def _ok(z):
                                    return isinstance(z, int) or (
                                        isinstance(z, (tuple, list)) and
                                        all(isinstance(w, int) for w in z))
                                if kind != 'convT':
                                    op = 0
                                if all(_ok(z) for z in (kk, ss, pp)) and \
                                        _ok(op):
                                    stage_spatial[t.attr] = (kind, kk, ss,
                                                             pp, op)
                            if cls in ('Conv1d', 'Conv2d', 'Conv3d',
                                       'ConvTranspose1d', 'ConvTranspose2d',
                                       'ConvTranspose3d') and len(v.args) >= 2:
                                w = None
                                a1 = v.args[1]
                                if isinstance(a1, ast.Constant):
                                    w = a1.value
                                elif isinstance(a1, ast.Name) and a1.id in consts:
                                    w = consts[a1.id]
                                if isinstance(w, int):
                                    stage_chans[t.attr] = w
                            if cls in ('Linear', 'Gemm') and len(v.args) >= 2:
                                w = None
                                a1 = v.args[1]
                                if isinstance(a1, ast.Constant):
                                    w = a1.value
                                elif isinstance(a1, ast.Name) and a1.id in consts:
                                    w = consts[a1.id]
                                if isinstance(w, int):
                                    stage_widths[t.attr] = w
                    if isinstance(v, ast.Name) and v.id in consts:
                        attrs[t.attr] = consts[v.id]
                    else:
                        try:
                            attrs[t.attr] = ast.literal_eval(v)
                        except Exception:
                            pass
    def scalar_of(e):
        """Resolve an expression to a scalar if possible."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            v = scalar_of(e.operand)
            return -v if v is not None else None
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) \
           and e.value.id == 'self' and e.attr in attrs:
            return attrs[e.attr]
        if isinstance(e, ast.Name) and e.id in consts:
            return consts[e.id]
        return None
    def _axis_is_last(d, rank):
        """True iff reduction axis d is provably the LAST contiguous dim.
        dim=-1 always last; dim=k needs known rank with k == rank-1.
        Unknown dim or unknown rank -> NOT provable -> refuse."""
        if d == -1:
            return True
        if d is None or rank is None:
            return False
        return d == rank - 1

    current_rank = [None]   # rank of x after the latest stage (boxed)
    current_width = [None]  # last-dim width after the latest stage (if provable)
    current_chan = [None]   # channel count (dim 1) after the latest stage
    # input SPATIAL dims from get_inputs (torch.rand/randn args past
    # batch+channel, resolved via consts) — for spatial-size proofs:
    input_spatial = None
    input_batch = None
    input_chan_in = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'get_inputs':
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    fn = f.attr if isinstance(f, ast.Attribute) else None
                    if fn in ('rand', 'randn'):
                        dims = []
                        for a in sub.args:
                            try:
                                dims.append(ast.literal_eval(a))
                            except Exception:
                                if isinstance(a, ast.Name) and a.id in consts:
                                    dims.append(consts[a.id])
                                else:
                                    dims.append(None)
                        if len(dims) >= 3 and all(
                                isinstance(d, int) for d in dims):
                            input_batch = dims[0]
                            input_chan_in = dims[1]
                            input_spatial = dims[2:]
                        break
    current_spatial = [list(input_spatial) if input_spatial else None]
    # second forward argument (two-input models: forward(self, x, y)):
    fwd_args = [a.arg for a in fwd.args.args[1:]]   # skip self
    arg2 = fwd_args[1] if len(fwd_args) > 1 else None
    ops = []
    _pending = {'t': None}
    _last_target = ['x']  # the chain-thread tracker (the #92 fix:
    # SSA-style forwards name each step — x_conv, x_norm... — the
    # binop handlers key on 'x'; the tracker knows which NAME holds
    # the current chain value)
    for stmt in fwd.body:
        if not isinstance(stmt, ast.Assign):
            continue
        v = stmt.value
        # the chain-thread tracker update (runs for EVERY assign,
        # before the handlers — the PREVIOUS stmt's target became
        # the thread if the value consumed the thread; heuristic:
        # any assignment whose VALUE references the current thread
        # moves the thread to its target):
        # (apply the PREVIOUS stmt's thread move first)
        if _pending.get('t'):
            _last_target[0] = _pending['t']
        try:
            _refs = {n.id for n in ast.walk(v)
                     if isinstance(n, ast.Name)}
            # a bare-Name alias (original_x = x — the SAVE form)
            # does NOT move the thread: the save COPIES the value,
            # the chain continues in the source var (the #70/#40/
            # #20 regression: the save stole the thread and every
            # later chain-op mismatched):
            _core = v
            while isinstance(_core, ast.Call) and \
                    isinstance(_core.func, ast.Attribute) and \
                    _core.func.attr in ('clone', 'detach'):
                _core = _core.func.value
            if isinstance(_core, ast.Name):
                # a bare Name or x.clone().detach() chain = a SAVE
                # (aliases/copies; does NOT consume the thread):
                _pending['t'] = None
            elif _last_target[0] in _refs and \
                    isinstance(stmt.targets[0], ast.Name):
                _pending['t'] = stmt.targets[0].id
            else:
                _pending['t'] = None
        except Exception:
            _pending['t'] = None
        # THE PEEL (the #55/#98 double/missed-walk fix): a statement
        # like self.pool(x.unsqueeze(1)).squeeze(1) must process
        # innermost-out — inner shape-op, then the module, then the
        # trailing shape-op — each EXACTLY ONCE. Unwrap trailing
        # .shapeop(...) wrappers; process nested arg shape-ops first:
        _SHAPE_OPS = ('squeeze', 'unsqueeze', 'view', 'reshape',
                      'flatten', 'permute', 'transpose')
        _post_shape = []
        while (isinstance(v, ast.Call)
               and isinstance(v.func, ast.Attribute)
               and isinstance(v.func.value, ast.Call)
               and v.func.attr in _SHAPE_OPS):
            try:
                _pargs = [ast.literal_eval(a) for a in v.args]
            except Exception:
                _pargs = None
            _post_shape.append((v.func.attr, _pargs))
            v = v.func.value
        if (isinstance(v, ast.Call) and len(v.args) == 1
                and isinstance(v.args[0], ast.Call)
                and isinstance(v.args[0].func, ast.Attribute)
                and isinstance(v.args[0].func.value, ast.Name)
                and v.args[0].func.value.id == 'x'
                and v.args[0].func.attr in _SHAPE_OPS):
            _in = v.args[0].func.attr
            _in_args = []
            try:
                _in_args = [ast.literal_eval(a)
                            for a in v.args[0].args]
            except Exception:
                _in_args = None
            ops.append(('stage', f'm_{_in}', _in_args))
            if _in == 'unsqueeze' and current_rank[0] is not None:
                current_rank[0] += 1
            elif _in == 'squeeze' and current_rank[0] is not None:
                current_rank[0] -= 1
            else:
                current_rank[0] = None
            v = ast.copy_location(
                ast.Call(func=v.func, args=[v.args[0].func.value],
                         keywords=v.keywords), v)

        def _apply_post_shape():
            for _op, _pargs in reversed(_post_shape):
                ops.append(('stage', f'm_{_op}', _pargs))
                if _op == 'unsqueeze' and current_rank[0] is not None:
                    current_rank[0] += 1
                elif _op == 'squeeze' and current_rank[0] is not None:
                    current_rank[0] -= 1
                else:
                    current_rank[0] = None
        # x = self.MODULE(x): activation-module -> op; else stage marker:
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
           and isinstance(v.func.value, ast.Name) and v.func.value.id == 'self':
            if v.func.attr in module_ops:
                op = module_ops[v.func.attr]
                if op[0] == 'reduction':
                    d = module_red_dims.get(v.func.attr)
                    if not _axis_is_last(d, current_rank[0]):
                        if op[1] == 'softmax' and d == 1 and \
                           current_rank[0] is not None and \
                           current_rank[0] >= 3:
                            ops.append(('reduction_chan', 'softmax', current_chan[0],
                                (__import__('math').prod(current_spatial[0])
                                 if current_spatial[0] else None)))
                            continue
                        ops.append(('unknown_call',
                                    'non_lastdim_reduction'))
                        continue
                    # lastdim module-reduction: carry the PROVEN width
                    # (#84's nn.Softmax(dim=1) on 2D — the width comes
                    # from the Linear stage; without it the emitter
                    # can't select a regime and refuses):
                    ops.append(('reduction', op[1], current_width[0]))
                    continue
                if op[0] == 'avgpool3d_k2':
                    # carry the module attr so the guard can revert
                    # to the true stage form in mixed chains (#50):
                    ops.append(('avgpool3d_k2', v.func.attr))
                else:
                    ops.append(op)
            else:
                ops.append(('stage', v.func.attr))
                cls = stage_classes.get(v.func.attr)
                if v.func.attr in stage_spatial and \
                   current_spatial[0] is not None:
                    kind, kk, ssz, pp, opad = stage_spatial[v.func.attr]
                    try:
                        n_sp = (len(current_spatial[0])
                                if kind != 'adaptive' else 0)
                        def _bc(z):
                            return (list(z) if isinstance(z, (tuple, list))
                                    else [z] * n_sp)
                        if kind == 'adaptive':
                            current_spatial[0] = list(kk)
                        elif kind == 'conv':
                            kkl, ssl, ppl = _bc(kk), _bc(ssz), _bc(pp)
                            current_spatial[0] = [
                                (d + 2 * ppl[j] - kkl[j]) // ssl[j] + 1
                                for j, d in enumerate(current_spatial[0])]
                        elif kind == 'convT':
                            kkl, ssl, ppl = _bc(kk), _bc(ssz), _bc(pp)
                            opl = _bc(opad)
                            current_spatial[0] = [
                                (d - 1) * ssl[j] - 2 * ppl[j] + kkl[j]
                                + opl[j]
                                for j, d in enumerate(current_spatial[0])]
                        elif kind == 'pool':
                            kkl, ssl, ppl = _bc(kk), _bc(ssz), _bc(pp)
                            current_spatial[0] = [
                                (d + 2 * ppl[j] - kkl[j]) // ssl[j] + 1
                                for j, d in enumerate(current_spatial[0])]
                    except Exception:
                        current_spatial[0] = None
                elif v.func.attr not in stage_spatial and \
                     stage_classes.get(v.func.attr) not in (
                         'GroupNorm', 'BatchNorm1d', 'BatchNorm2d',
                         'BatchNorm3d', 'InstanceNorm2d', 'InstanceNorm3d',
                         'LayerNorm', 'Dropout', 'Linear', 'Gemm'):
                    pass  # unknown spatial effect for non-tracked stages:
                if v.func.attr in stage_classes and \
                   v.func.attr not in stage_spatial and \
                   stage_classes.get(v.func.attr) in (
                       'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
                       'ConvTranspose2d', 'ConvTranspose3d', 'MaxPool1d',
                       'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d',
                       'AvgPool3d'):
                    current_spatial[0] = None  # spatial stage, params unproven
                if v.func.attr in stage_chans:
                    current_chan[0] = stage_chans[v.func.attr]
                elif cls in ('MaxPool1d', 'MaxPool2d', 'MaxPool3d',
                             'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
                             'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
                             'AdaptiveAvgPool3d', 'AdaptiveMaxPool1d',
                             'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
                             'Dropout', 'GroupNorm', 'BatchNorm1d',
                             'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm2d',
                             'InstanceNorm3d', 'LayerNorm'):
                    pass  # channel-preserving stages keep current_chan
                else:
                    current_chan[0] = None
                if v.func.attr in stage_widths:
                    current_width[0] = stage_widths[v.func.attr]
                elif cls not in ('GroupNorm', 'BatchNorm1d', 'LayerNorm',
                                 'Dropout', 'Identity'):
                    current_width[0] = None
                RANK_PRESERVING = {'GroupNorm', 'BatchNorm1d', 'BatchNorm2d',
                                   'BatchNorm3d', 'LayerNorm', 'InstanceNorm1d',
                                   'InstanceNorm2d', 'InstanceNorm3d',
                                   'Dropout', 'AvgPool1d', 'AvgPool2d',
                                   'AvgPool3d', 'MaxPool1d', 'MaxPool2d',
                                   'MaxPool3d'}
                if cls in RANK_PRESERVING:
                    pass   # keep current_rank
                else:
                    current_rank[0] = STAGE_RANKS.get(cls) if cls else None
            _apply_post_shape()
            continue
        # ORIG = x — plain-name residual save (#70's form):
        if isinstance(stmt.targets[0], ast.Name) and \
           stmt.targets[0].id != 'x' and isinstance(v, ast.Name) and \
           v.id == 'x':
            ops.append(('save', stmt.targets[0].id))
            continue
        # x = ORIG.clone()... — residual save (target name captured):
        if isinstance(stmt.targets[0], ast.Name) and \
           stmt.targets[0].id != 'x' and isinstance(v, ast.Call):
            fn = v.func
            # x.clone() or x.clone().detach() chains:
            base = fn
            while isinstance(base, ast.Attribute) and \
                  base.attr in ('clone', 'detach'):
                base = base.value
                if isinstance(base, ast.Call):
                    base = base.func
            if isinstance(base, ast.Name) and base.id == 'x':
                ops.append(('save', stmt.targets[0].id))
                continue
        # x - x.mean(dim=K, keepdim=True) when width==1: mean of a
        # single element IS the element → x − x ≡ 0 EXACTLY (#80's
        # degenerate identity — DERIVED, not assumed; the census entry
        # carries PASS-DEGENERATE framing):
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub) and \
           isinstance(v.left, ast.Name) and v.left.id == 'x' and \
           isinstance(v.right, ast.Call) and \
           isinstance(v.right.func, ast.Attribute) and \
           v.right.func.attr == 'mean' and \
           isinstance(v.right.func.value, ast.Name) and \
           v.right.func.value.id == 'x' and current_width[0] == 1:
            kd = any(kw.arg == 'keepdim' and
                     getattr(kw.value, 'value', False) is True
                     for kw in v.right.keywords)
            if kd:
                ops.append(('zero_fold',))
                continue
        # x = x * sigmoid(x + 3) / 6 — the sigmoid-hardswish variant
        # (#58): Div(Mult(x, sigmoid(x+3)), 6) — lift as ONE fused op
        # (caught by the silent-drop audit: the old scalar branch
        # matched /6 and DROPPED the x*sigmoid(x+3) — bug-class 9):
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Div) and \
           scalar_of(v.right) == 6.0 and \
           isinstance(v.left, ast.BinOp) and \
           isinstance(v.left.op, ast.Mult) and \
           isinstance(v.left.left, ast.Name) and v.left.left.id == 'x' and \
           isinstance(v.left.right, ast.Call):
            sf = v.left.right.func
            sname = sf.attr if isinstance(sf, ast.Attribute) else \
                    (sf.id if isinstance(sf, ast.Name) else None)
            if sname == 'sigmoid' and v.left.right.args and \
               isinstance(v.left.right.args[0], ast.BinOp) and \
               isinstance(v.left.right.args[0].op, ast.Add) and \
               scalar_of(v.left.right.args[0].right) == 3.0:
                ops.append(('xsig3_div6',))
                continue
        # x = x * S / x = x / S / x = x + S / x = x - S
        # (v.left MUST be plain x — a nested left expression through
        # the scalar branch would SILENT-DROP it; bug-class 9):
        if isinstance(v, ast.BinOp):
            s = scalar_of(v.right)
            if s is not None and isinstance(v.left, ast.Name) and \
               v.left.id == 'x':
                kind = {ast.Mult: 'multiply', ast.Div: 'divide',
                        ast.Add: 'add_scalar', ast.Sub: 'subtract'}.get(type(v.op))
                if kind:
                    ops.append((kind, float(s)))
                    continue
            elif s is not None:
                ops.append(('unknown_call', 'binop_nested_left'))
                continue
            # x OP self.ATTR.view(...) — channel-broadcast via explicit
            # view (#89's subtract.view(1,-1,1,1,1)) — unwrap to the
            # param Attribute (the view IS the chan-broadcast):
            vr = v.right
            if isinstance(vr, ast.Call) and \
               isinstance(vr.func, ast.Attribute) and \
               vr.func.attr == 'view' and \
               isinstance(vr.func.value, ast.Attribute) and \
               isinstance(vr.func.value.value, ast.Name) and \
               vr.func.value.value.id == 'self':
                view_args = []
                for a in vr.args:
                    try:
                        view_args.append(ast.literal_eval(a))
                    except Exception:
                        view_args.append(None)
                # (1, -1, 1...) = channel-broadcast:
                if view_args and view_args[0] == 1 and view_args[1] == -1 \
                   and all(z == 1 for z in view_args[2:]):
                    kind = {ast.Mult: 'mul_param_chan',
                            ast.Add: 'add_param_chan',
                            ast.Sub: 'sub_param_chan'}.get(type(v.op))
                    if kind:
                        ops.append((kind, vr.func.value.attr))
                        continue
            # x = x + self.ATTR / x = x * self.ATTR — PARAM-tensor ops:
            # COMMUTATIVE FORMS: #84 writes `self.scale * x` -- the param on the
            # LEFT. mul and add commute, so the same lift is valid. sub and div
            # DO NOT: `self.b - x` is not `x - self.b`, so those stay refused
            # in reversed form rather than lifted wrongly.
            if (not (isinstance(v.right, ast.Attribute)
                     and isinstance(v.right.value, ast.Name)
                     and v.right.value.id == 'self')) \
               and isinstance(v.left, ast.Attribute) \
               and isinstance(v.left.value, ast.Name) \
               and v.left.value.id == 'self' \
               and isinstance(v.op, (ast.Mult, ast.Add)):
                v = ast.BinOp(left=v.right, op=v.op, right=v.left)
            if isinstance(v.right, ast.Attribute) and \
               isinstance(v.right.value, ast.Name) and \
               v.right.value.id == 'self' and \
               v.right.attr not in attrs:
                kind = {ast.Mult: 'mul_param', ast.Add: 'add_param',
                        ast.Sub: 'sub_param'}.get(type(v.op))
                if kind:
                    if v.right.attr in scalar_params:
                        ops.append((kind + '_scalar', v.right.attr))
                    elif v.right.attr in channel_params:
                        ops.append((kind + '_chan', v.right.attr))
                    else:
                        ops.append((kind, v.right.attr))
                    continue
            # x = x OP ARG2 — second-input elementwise (two-input model):
            if arg2 is not None and isinstance(v.right, ast.Name) \
               and v.right.id == arg2 and isinstance(v.left, ast.Name) \
               and v.left.id == 'x':
                ikind = {ast.Add: 'add_input2', ast.Mult: 'mul_input2',
                         ast.Sub: 'sub_input2'}.get(type(v.op))
                if ikind:
                    ops.append((ikind,))
                    continue
            # SAVED OP chain — the REVERSED residual (#92's
            # x_res = x_conv + x_hard_swish: left = an EARLIER
            # intermediate, right = the current chain thread):
            if isinstance(v.left, ast.Name) and \
               isinstance(v.right, ast.Name) and \
               v.right.id == _last_target[0] and \
               v.left.id != _last_target[0] and \
               isinstance(v.op, ast.Add):
                ops.append(('add_saved', v.left.id))
                if isinstance(stmt.targets[0], ast.Name):
                    _last_target[0] = stmt.targets[0].id
                continue
            # x = x OP SAVED_NAME — residual ops (NOT x+x — add_self):
            if isinstance(v.right, ast.Name) \
               and isinstance(v.left, ast.Name) and v.left.id == 'x' \
               and v.right.id != 'x':
                skind = {ast.Add: 'add_saved', ast.Mult: 'mul_saved',
                         ast.Sub: 'sub_saved'}.get(type(v.op))
                if skind:
                    ops.append((skind, v.right.id))
                    continue
            # x = torch.sigmoid(x) * x — REVERSED swish (sigmoid on left);
            # commutative product, same value BUT preserve source order:
            if isinstance(v.op, ast.Mult) and isinstance(v.left, ast.Call) \
               and isinstance(v.right, ast.Name) and v.right.id == 'x':
                fn = v.left.func
                if isinstance(fn, ast.Attribute) and fn.attr == 'sigmoid':
                    ops.append(('swish_rev',))
                    continue
            # x = x * ACT(x) — self-gated: swish (sigmoid) / x·mish(x):
            if isinstance(v.op, ast.Mult) and isinstance(v.right, ast.Call):
                fn = v.right.func
                if isinstance(fn, ast.Attribute) and fn.attr == 'sigmoid':
                    ops.append(('swish',))
                    continue
                if isinstance(fn, ast.Attribute) and fn.attr == 'mish':
                    ops.append(('xmish',))
                    continue
                if isinstance(fn, ast.Attribute) and fn.attr == 'hardswish':
                    ops.append(('xhardswish',))
                    continue
                if isinstance(fn, ast.Attribute) and fn.attr == 'clamp':
                    # x * clamp((x+3)/6, 0, 1) = written-out hardswish-CLIP
                    # form; keep the SOURCE arithmetic (clip form) — this
                    # is what the problem computes, NOT F.hardswish:
                    ca = v.right.args
                    if len(ca) >= 3:
                        ops.append(('xclip_hswish',))
                        continue
            # x = x + x:
            if isinstance(v.op, ast.Add) and isinstance(v.left, ast.Name) \
               and isinstance(v.right, ast.Name) and v.left.id == v.right.id:
                ops.append(('add_self',))
                continue
            # x = x - torch.mean(x, dim=(...), keepdim=True) — the
            # SELF-REFERENTIAL spatial-mean subtract (the #15 class):
            if isinstance(v.op, ast.Sub) and isinstance(v.left, ast.Name) \
               and isinstance(v.right, ast.Call) \
               and isinstance(v.right.func, ast.Attribute) \
               and v.right.func.attr == 'mean' \
               and v.right.args \
               and isinstance(v.right.args[0], ast.Name) \
               and v.right.args[0].id == v.left.id:
                _dims = None
                _keep = False
                for kw in v.right.keywords:
                    if kw.arg == 'dim':
                        try:
                            _dims = tuple(ast.literal_eval(kw.value)) \
                                if not isinstance(kw.value, ast.Constant) \
                                else (kw.value.value,)
                        except Exception:
                            _dims = None
                    if kw.arg == 'keepdim':
                        try:
                            _keep = bool(ast.literal_eval(kw.value))
                        except Exception:
                            _keep = False
                if _dims is not None and _keep:
                    ops.append(('sub_own_mean', _dims))
                    continue
                ops.append(('unknown_call', 'sub_own_mean_unresolved'))
                continue
            # BinOp that matched NO case above (e.g. x = x + self.bias where
            # self.bias is a non-scalar nn.Parameter): scalar_of(rhs) was None,
            # so it fell through. FLAG it — never silently drop (a dropped
            # bias-add produces a confidently-wrong kernel: caught by Doresh
            # on #91, max abs diff 0.69). Honest-failure: name the gap.
            opname = {ast.Mult: 'multiply', ast.Div: 'divide',
                      ast.Add: 'add', ast.Sub: 'subtract'}.get(type(v.op), 'op')
            ops.append(('unknown_call', f'binop_{opname}_nonscalar_rhs'))
            continue
        # x = x.METHOD(...) — method-calls on the tensor:
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
           and isinstance(v.func.value, ast.Name) \
           and v.func.value.id == 'x':
            mname = v.func.attr
            if mname in ('squeeze', 'unsqueeze', 'view', 'reshape',
                         'flatten', 'permute', 'transpose', 'contiguous'):
                # THE FREE-VIEW UNLOCK (#38): x.view(b, c, -1) on a
                # contiguous tensor is ZERO-COPY — no stage break;
                # the rank drops to 3 (making a dim=2 softmax the
                # LAST dim — the row-softmax machinery). The
                # view-back (5 Name/attr args) restores rank 5,
                # also free:
                if mname == 'view' and len(v.args) == 3 and \
                        isinstance(v.args[2], ast.UnaryOp) and \
                        isinstance(v.args[2].op, ast.USub):
                    current_rank[0] = 3
                    current_width[0] = (
                        __import__('math').prod(current_spatial[0])
                        if current_spatial[0] else None)
                    continue
                if mname == 'view' and len(v.args) == 5:
                    current_rank[0] = 5
                    continue
                try:
                    _margs = [ast.literal_eval(a) for a in v.args]
                except Exception:
                    _margs = None
                ops.append(('stage', f'm_{mname}', _margs))
                if mname == 'unsqueeze' and current_rank[0] is not None:
                    current_rank[0] += 1
                elif mname == 'squeeze' and current_rank[0] is not None:
                    current_rank[0] -= 1
                elif mname == 'contiguous':
                    pass  # rank-preserving
                else:
                    current_rank[0] = None
                _apply_post_shape()
                continue
            if mname in ('clamp',):
                lo = hi = None
                for kw in v.keywords:
                    if kw.arg == 'min':
                        lo = scalar_of(kw.value)
                    if kw.arg == 'max':
                        hi = scalar_of(kw.value)
                if lo is not None and hi is not None:
                    ops.append(('clamp', float(lo), float(hi)))
                    continue
        # x = torch.max(x, dim=D, keepdim=..).values — the .values
        # Attribute form ≡ [0]-subscript (#80):
        if isinstance(v, ast.Attribute) and v.attr == 'values' and \
           isinstance(v.value, ast.Call):
            inner = v.value
            fn = inner.func
            name = (fn.attr if isinstance(fn, ast.Attribute) else
                    (fn.id if isinstance(fn, ast.Name) else None))
            if name in ('min', 'max'):
                d = None
                for kw in inner.keywords:
                    if kw.arg == 'dim':
                        if isinstance(kw.value, ast.Constant):
                            d = kw.value.value
                        elif isinstance(kw.value, ast.Attribute) and \
                             isinstance(kw.value.value, ast.Name) and \
                             kw.value.value.id == 'self' and \
                             kw.value.attr in init_args:
                            d = init_args[kw.value.attr]
                kd = any(kw.arg == 'keepdim' and
                         getattr(kw.value, 'value', False) is True
                         for kw in inner.keywords)
                if _axis_is_last(d, current_rank[0]):
                    ops.append(('reduction', name, current_width[0]))
                    if kd:
                        current_width[0] = 1
                    continue
                ops.append(('unknown_call', 'values_nonlast_reduce'))
                continue
        # x = torch.min(x, dim=D, keepdim=..)[0] — subscripted reduce:
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Call):
            inner = v.value
            fn = inner.func
            name = fn.attr if isinstance(fn, ast.Attribute) else \
                   (fn.id if isinstance(fn, ast.Name) else None)
            if name in ('min', 'max'):
                d = None
                for kw in inner.keywords:
                    if kw.arg == 'dim' and isinstance(kw.value, ast.Constant):
                        d = kw.value.value
                    # dim=self.ATTR where ATTR is an init-arg (#24):
                    elif kw.arg == 'dim' and \
                            isinstance(kw.value, ast.Attribute) and \
                            isinstance(kw.value.value, ast.Name) and \
                            kw.value.value.id == 'self' and \
                            kw.value.attr in init_args:
                        d = init_args[kw.value.attr]
                if d is None and len(inner.args) >= 2 and \
                   isinstance(inner.args[1], ast.Constant):
                    d = inner.args[1].value
                if not _axis_is_last(d, current_rank[0]):
                    if d == 1 and current_rank[0] is not None and \
                       current_rank[0] >= 3:
                        ops.append(('reduction_chan', name, current_chan[0],
                                    (__import__('math').prod(current_spatial[0])
                                     if current_spatial[0] else None)))
                        continue
                    # THE ARBITRARY-AXIS FOLD (the #24 site — the
                    # subscripted torch.min(x, dim=k)[0] form; no
                    # keepdim → the rank SHRINKS, the folded dim
                    # DROPS from spatial):
                    if name in ('min', 'max') and d is not None and \
                       d >= 2 and current_rank[0] is not None and \
                       d < current_rank[0] - 1 and \
                       current_spatial[0] and \
                       len(current_spatial[0]) > (d - 2):
                        _sp = current_spatial[0]
                        _chan_f = _sp[d - 2]
                        _after = _sp[d - 1:]
                        _spatial_f = (__import__('math').prod(_after)
                                      if _after else 1)
                        ops.append(('reduction_chan_fold', name,
                                    _chan_f, _spatial_f))
                        _kd = any(kw.arg == 'keepdim' and
                                  getattr(kw.value, 'value', False)
                                  is True for kw in inner.keywords)
                        _sp2 = list(_sp)
                        if _kd:
                            _sp2[d - 2] = 1
                        else:
                            _sp2.pop(d - 2)
                            current_rank[0] -= 1
                        current_spatial[0] = _sp2
                        continue
                    ops.append(('unknown_call', 'non_lastdim_reduction'))
                    continue
                if current_width[0] == 1:
                    # size-1 reduce = IDENTITY (bit-exact: max/min of one
                    # element is itself) — drop the op (#18 class):
                    kd1 = any(kw.arg == 'keepdim' and
                              getattr(kw.value, 'value', False) is True
                              for kw in inner.keywords)
                    if kd1:
                        continue
                    current_width[0] = None  # rank shrinks, width unknown
                    continue
                kd1 = any(kw.arg == 'keepdim' and
                          getattr(kw.value, 'value', False) is True
                          for kw in inner.keywords)
                ops.append(('reduction', name, current_width[0]))
                if kd1:
                    current_width[0] = 1
                continue
        # torch.multiply(tanh(softplus(x)), x) = MISH (#52's spelled-out
        # form — the composition IS the activation):
        if isinstance(v, ast.Call):
            fnm = v.func
            nm = fnm.attr if isinstance(fnm, ast.Attribute) else \
                 (fnm.id if isinstance(fnm, ast.Name) else None)
            if nm == 'multiply' and len(v.args) == 2 and \
               isinstance(v.args[1], ast.Name) and v.args[1].id == 'x' and \
               isinstance(v.args[0], ast.Call):
                inner1 = v.args[0]
                n1 = inner1.func.attr if isinstance(inner1.func, ast.Attribute) else None
                if n1 == 'tanh' and inner1.args and \
                   isinstance(inner1.args[0], ast.Call):
                    inner2 = inner1.args[0]
                    n2 = inner2.func.attr if isinstance(inner2.func, ast.Attribute) else None
                    if n2 == 'softplus' and inner2.args and \
                       isinstance(inner2.args[0], ast.Name) and \
                       inner2.args[0].id == 'x':
                        ops.append(('mish',))
                        continue
        # x = torch.F(x, ...) / F.f(x):
        if isinstance(v, ast.Call):
            fn = v.func
            name = fn.attr if isinstance(fn, ast.Attribute) else \
                   (fn.id if isinstance(fn, ast.Name) else None)
            if name in ('relu', 'tanh', 'sigmoid', 'gelu', 'mish',
                        'hardswish', 'hardtanh'):
                ops.append((name,))
                continue
            if name == 'leaky_relu':
                slope = 0.01
                if len(v.args) >= 2:
                    s = scalar_of(v.args[1])
                    if s is not None:
                        slope = float(s)
                for kw in v.keywords:
                    if kw.arg == 'negative_slope':
                        s = scalar_of(kw.value)
                        if s is not None:
                            slope = float(s)
                ops.append(('leaky_relu', slope))
                continue
            if name in ('min', 'max') and len(v.args) == 2 and \
                    not v.keywords:
                # torch.min(x, torch.tensor(V)) — ELEMENTWISE min/max
                # vs a scalar tensor (the #83 unlock, mavhir's sweep):
                # min = clamp-from-above, max = clamp-from-below.
                # Resolve V from a torch.tensor(literal-or-init-arg)
                # wrapper or a bare literal/name:
                a1 = v.args[1]
                vv = None
                if isinstance(a1, ast.Call) and \
                        isinstance(a1.func, ast.Attribute) and \
                        a1.func.attr == 'tensor' and a1.args:
                    a1 = a1.args[0]
                try:
                    vv = float(ast.literal_eval(a1))
                except Exception:
                    if isinstance(a1, ast.Name) and a1.id in init_args:
                        vv = float(init_args[a1.id])
                    # self.ATTR where ATTR is an init-arg (the #31/#68
                    # regression from the #83 branch intercepting —
                    # the older resolution restored):
                    elif isinstance(a1, ast.Attribute) and \
                            isinstance(a1.value, ast.Name) and \
                            a1.value.id == 'self' and \
                            a1.attr in init_args:
                        vv = float(init_args[a1.attr])
                if vv is not None:
                    ops.append(('clamp_max' if name == 'min'
                                else 'clamp_min', vv))
                    continue
                ops.append(('unknown_call',
                            f'{name}_with_unresolved_tensor'))
                continue
            if name == 'clamp':
                lo = hi = None
                for kw in v.keywords:
                    if kw.arg == 'min':
                        lo = scalar_of(kw.value)
                    if kw.arg == 'max':
                        hi = scalar_of(kw.value)
                if lo is None and len(v.args) >= 3:
                    lo = scalar_of(v.args[1])
                    hi = scalar_of(v.args[2])
                if lo is not None and hi is not None:
                    ops.append(('clamp', float(lo), float(hi)))
                    continue
                if lo is not None:
                    ops.append(('clamp_min', float(lo)))
                    continue
                if hi is not None:
                    ops.append(('clamp_max', float(hi)))
                    continue
                ops.append(('unknown_call', 'clamp_unresolved'))
                continue
            if name in ('group_norm', 'batch_norm', 'layer_norm',
                        'instance_norm', 'normalize'):
                ops.append(('stage', f'F_{name}'))
                continue
            # pooling/dropout: rank-preserving stage-ops (torch runs them):
            if name in ('avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                        'max_pool1d', 'max_pool2d', 'max_pool3d',
                        'adaptive_avg_pool1d', 'adaptive_avg_pool2d',
                        'adaptive_avg_pool3d', 'dropout'):
                try:
                    _fargs = [ast.literal_eval(a) for a in v.args[1:]]
                except Exception:
                    _fargs = None
                ops.append(('stage', f'F_{name}', _fargs))
                if name != 'dropout':
                    current_width[0] = None   # pools change spatial dims
                continue
            # shape-ops: stage-markers with UNKNOWN rank after (conservative —
            # post-op reductions will refuse via the axis-check):
            if name in ('squeeze', 'unsqueeze', 'view', 'reshape',
                        'flatten', 'permute', 'transpose'):
                ops.append(('stage', f'F_{name}'))
                if name == 'unsqueeze' and current_rank[0] is not None:
                    current_rank[0] += 1
                elif name == 'squeeze' and current_rank[0] is not None:
                    current_rank[0] -= 1
                else:
                    current_rank[0] = None
                _apply_post_shape()
                continue
            if name in ('group_norm', 'batch_norm', 'layer_norm',
                        'instance_norm', 'normalize'):
                ops.append(('stage', f'F_{name}'))
                continue
            # torch.min(x, torch.tensor(CONST)): elementwise vs a literal —
            # min(x, c) = clamp_max(c); max(x, c) = clamp_min(c):
            if name in ('min', 'max') and len(v.args) >= 2 and \
               isinstance(v.args[1], ast.Call) and \
               not any(kw.arg == 'dim' for kw in v.keywords):
                ifn = v.args[1].func
                iname = ifn.attr if isinstance(ifn, ast.Attribute) else None
                if iname == 'tensor' and v.args[1].args and \
                   isinstance(v.args[1].args[0], ast.Constant):
                    cval = float(v.args[1].args[0].value)
                    ops.append(('clamp_max' if name == 'min'
                                else 'clamp_min', cval))
                    continue
                # torch.min(x, torch.tensor(self.ATTR)) — param-valued
                # elementwise min/max (#31's form): scalar param p[0]:
                if iname == 'tensor' and v.args[1].args and \
                   isinstance(v.args[1].args[0], ast.Attribute) and \
                   isinstance(v.args[1].args[0].value, ast.Name) and \
                   v.args[1].args[0].value.id == 'self':
                    ops.append(('min_param_scalar' if name == 'min'
                                else 'max_param_scalar',
                                v.args[1].args[0].attr))
                    continue
            # torch.min(x, self.ATTR)/torch.max(x, self.ATTR): ELEMENTWISE
            # binary min/max with a param (NOT a reduction — no dim arg):
            if name in ('min', 'max') and len(v.args) >= 2 and \
               isinstance(v.args[1], ast.Attribute) and \
               isinstance(v.args[1].value, ast.Name) and \
               v.args[1].value.id == 'self' and \
               not any(kw.arg == 'dim' for kw in v.keywords):
                attr = v.args[1].attr
                if attr in scalar_params:
                    ops.append((f'{name}_param_scalar', attr))
                    continue
                ops.append(('unknown_call', f'{name}_with_tensor_param'))
                continue
            if name in ('softmax', 'logsumexp', 'mean', 'max', 'min', 'sum'):
                d = None
                for kw in v.keywords:
                    if kw.arg == 'dim':
                        try:
                            d = ast.literal_eval(kw.value)
                        except Exception:
                            pass
                if d is None and len(v.args) >= 2:
                    try:
                        d = ast.literal_eval(v.args[1])
                    except Exception:
                        pass
                # dim=self.ATTR (resolved via init args — #8's sum_dim):
                if d is None:
                    for kw in v.keywords:
                        if kw.arg == 'dim' and \
                           isinstance(kw.value, ast.Attribute) and \
                           isinstance(kw.value.value, ast.Name) and \
                           kw.value.value.id == 'self' and \
                           kw.value.attr in init_args:
                            d = init_args[kw.value.attr]
                    if d is None and len(v.args) >= 2 and \
                       isinstance(v.args[1], ast.Attribute) and \
                       isinstance(v.args[1].value, ast.Name) and \
                       v.args[1].value.id == 'self' and \
                       v.args[1].attr in init_args:
                        d = init_args[v.args[1].attr]
                # dim-TUPLE: trailing-contiguous tuples lift to the
                # Case-1 tuple template (mean/sum only):
                if isinstance(d, (tuple, list)) and \
                   name in ('mean', 'sum') and \
                   current_rank[0] is not None:
                    rk = current_rank[0]
                    dd = sorted(x if x >= 0 else rk + x for x in d)
                    if dd == list(range(rk - len(dd), rk)):
                        kdt = any(kw.arg == 'keepdim' and
                                  getattr(kw.value, 'value', False) is True
                                  for kw in v.keywords)
                        # SIZE-1 tuple reduce = IDENTITY (the #42/#18
                        # class: after a keepdim tuple-mean the spatial
                        # dims are all 1; sum/mean over them is the
                        # element itself — bit-exact drop):
                        if current_spatial[0] is not None and \
                           len(dd) == len(current_spatial[0]) and \
                           all(z == 1 for z in current_spatial[0]):
                            if kdt:
                                continue
                            current_rank[0] = (current_rank[0] - len(dd)
                                               if current_rank[0] else None)
                            current_spatial[0] = None
                            continue
                        nred = None
                        rows = None
                        if current_spatial[0] is not None and \
                           len(dd) <= len(current_spatial[0]) + 1:
                            import math
                            sp = current_spatial[0]
                            if len(dd) == len(sp):        # spatial only
                                nred = math.prod(sp)
                                if input_batch and current_chan[0]:
                                    rows = input_batch * current_chan[0]
                            elif len(dd) == len(sp) + 1:  # chan+spatial
                                if current_chan[0]:
                                    nred = math.prod(sp) * current_chan[0]
                                    rows = input_batch
                        ops.append(('reduction_tuple', name, nred, rows))
                        if kdt and current_spatial[0] is not None and \
                           len(dd) == len(current_spatial[0]):
                            # keepdim: spatial dims collapse to 1s —
                            # subsequent dim-1 reduces see inner=1
                            # (the shape-collapse that turns chan-lse
                            # into ROW-lse, #42):
                            current_spatial[0] = [1] * len(dd)
                        continue
                    ops.append(('unknown_call', 'nontrailing_tuple_reduction'))
                    continue
                if not _axis_is_last(d, current_rank[0]):
                    if d == 1 and current_spatial[0] is not None and \
                       all(z == 1 for z in current_spatial[0]) and \
                       name in ('min', 'max', 'softmax', 'mean', 'sum',
                                'logsumexp'):
                        # spatial all-1s: dim-1 IS lastdim-equivalent
                        # (the trailing dims are size-1) → ROW-reduce
                        # with width = chan_count (#42's lse-over-C
                        # after global-avg-pool):
                        kdr = any(kw.arg == 'keepdim' and
                                  getattr(kw.value, 'value', False) is True
                                  for kw in v.keywords)
                        ops.append(('reduction', name, current_chan[0]))
                        if kdr:
                            current_chan[0] = 1
                            current_width[0] = 1
                        continue
                    if name in ('min', 'max', 'softmax', 'mean', 'sum', 'logsumexp') and d == 1 and \
                       current_rank[0] is not None and current_rank[0] >= 3:
                        ops.append(('reduction_chan', name, current_chan[0],
                                    (__import__('math').prod(current_spatial[0])
                                     if current_spatial[0] else None)))
                        continue
                    # THE ARBITRARY-AXIS FOLD (#13/#24/#36 — measured
                    # bit-exact for mean AND min at the real shapes):
                    # a reduce over spatial axis d (2 <= d < rank-1)
                    # folds to a chan-reduce: outer = prod(dims before
                    # d), chan = spatial[d-2], spatial = prod(after).
                    # The chan template's semantics don't care what
                    # the outer is.
                    if name in ('min', 'max', 'mean', 'sum') and \
                       d is not None and d >= 2 and \
                       current_rank[0] is not None and \
                       d < current_rank[0] - 1 and \
                       current_spatial[0] and \
                       len(current_spatial[0]) > (d - 2):
                        _sp = current_spatial[0]
                        _chan_f = _sp[d - 2]
                        _after = _sp[d - 1:]
                        _spatial_f = (__import__('math').prod(_after)
                                      if _after else 1)
                        ops.append(('reduction_chan_fold', name,
                                    _chan_f, _spatial_f))
                        # shape after (keepdim assumed — the folded
                        # axis becomes 1):
                        _sp2 = list(_sp)
                        _sp2[d - 2] = 1
                        current_spatial[0] = _sp2
                        continue
                    ops.append(('unknown_call', 'non_lastdim_reduction'))
                    continue
                kd2 = any(kw.arg == 'keepdim' and
                          getattr(kw.value, 'value', False) is True
                          for kw in v.keywords)
                if current_width[0] == 1:
                    # size-1 reduce = IDENTITY, bit-exact for sum/mean/
                    # max/min (x itself) AND lse (torch: log(exp(x-x))+x
                    # = x exactly) — drop (#18's serial-identity chain):
                    if kd2:
                        continue
                    current_width[0] = None
                    continue
                ops.append(('reduction', name, current_width[0]))
                if kd2:
                    current_width[0] = 1
                continue
            # torch.matmul(x, self.W.T) — a GEMM STAGE (torch runs
            # it — the #14 unlock): rank stays 2; the width becomes
            # the weight's OUT dim (W stored (out,in), used .T):
            if name == 'matmul' and len(v.args) == 2 and \
                    isinstance(v.args[0], ast.Name):
                _w = v.args[1]
                # unwrap .T:
                if isinstance(_w, ast.Attribute) and _w.attr == 'T':
                    _w = _w.value
                if isinstance(_w, ast.Attribute) and \
                        isinstance(_w.value, ast.Name) and \
                        _w.value.id == 'self':
                    ops.append(('stage', 'm_matmul', [_w.attr]))
                    current_rank[0] = 2
                    _wshape = param_shapes.get(_w.attr)
                    current_width[0] = (_wshape[0] if _wshape
                                        else None)
                    current_chan[0] = None
                    current_spatial[0] = None
                    continue
            ops.append(('unknown_call', name))
            continue
        # CATCH-ALL: any assignment statement that matched NO handler above
        # (not a stage, not a recognized BinOp, not a recognized Call) is
        # FLAGGED, never silently dropped. A silently-dropped statement
        # produces a confidently-wrong kernel — the exact hole Doresh found
        # on #91. The honest-failure property ('name the gap, never fake')
        # requires that NOTHING falls through unflagged.
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            ops.append(('unknown_call', 'unrecognized_assign'))
    return ops


def _avgpool_guard(ops):
    """avgpool3d_k2 emits ONLY in pure-avgpool chains (the #72
    route); in MIXED chains it stays a STAGE (the #50 regression:
    the op fired mid-chain where the elementwise harness has no
    spelling — vocabulary gap at emit)."""
    _lifted = [o for o in ops if o[0] != 'stage']
    if any(o[0] == 'avgpool3d_k2' for o in _lifted) and \
            any(o[0] != 'avgpool3d_k2' for o in _lifted):
        return [(('stage', o[1]) if (o[0] == 'avgpool3d_k2'
                                     and len(o) > 1) else o)
                for o in ops]
    return ops


def to_prolog(pid, ops):
    ops = _avgpool_guard(ops)
    """Chain fact for the bridge; epilogue ops only (post-stage)."""
    terms_meta = []
    # find the last 'stage' marker; the epilogue is what follows:
    idx = max((i for i, o in enumerate(ops) if o[0] == 'stage'), default=-1)
    epi = ops[idx + 1:]
    # COMPLETENESS CHECK (Doresh caught on #94): 'everything after the LAST
    # stage' silently DISCARDS real fusable ops SANDWICHED between non-terminal
    # stage markers (e.g. gemm->+bias->hardtanh->mish->groupnorm drops the
    # bias/hardtanh/mish -> emits out[i]=x, a pure no-op, reported PASS).
    # If there are any NON-stage ops BEFORE the last stage marker, this is a
    # multi-stage fusion the epilogue-extraction can't express — FLAG it,
    # never silently drop. (idx-1 because ops[idx] is the last stage itself.)
    pre = ops[:idx] if idx >= 0 else []
    dropped = [o for o in pre if o[0] not in ('stage', 'save')]
    # ('save', name) = the residual's bookkeeping (x.clone().detach()),
    # not a dropped computation (the #51 unlock — mavhir's trace,
    # Mavdil's tested confirmation)
    if dropped:
        # real ops lost between stages: refuse honestly, name the gap.
        return f"chain({pid}, [UNSUPPORTED_multi_stage_gap])."
    terms = []
    for o in epi:
        if o[0] == 'multiply':
            terms.append(f'multiply({o[1]})')
        elif o[0] == 'divide':
            terms.append(f'divide({o[1]})')
        elif o[0] == 'add_scalar':
            terms.append(f'add_scalar({o[1]})')
        elif o[0] == 'subtract':
            terms.append(f'subtract({o[1]})')
        elif o[0] == 'clamp':
            terms.append(f'clamp({o[1]}, {o[2]})')
        elif o[0] == 'clamp_min':
            terms.append(f'clamp_min({o[1]})')
        elif o[0] == 'clamp_max':
            terms.append(f'clamp_max({o[1]})')
        elif o[0] == 'swish':
            terms.append('swish')
        elif o[0] == 'add_self':
            terms.append('add_self')
        elif o[0] == 'xmish':
            terms.append('xmish')
        elif o[0] == 'mish':
            terms.append('mish')
        elif o[0] == 'xsig3_div6':
            terms.append('xsig3_div6')
        elif o[0] == 'zero_fold':
            terms.append('zero_fold')
        elif o[0] == 'swish_rev':
            terms.append('swish_rev')
        elif o[0] == 'xclip_hswish':
            terms.append('xclip_hswish')
        elif o[0] == 'xhardswish':
            terms.append('xhardswish')
        elif o[0] in ('add_input2', 'mul_input2', 'sub_input2'):
            terms.append(o[0])
        elif o[0] == 'mul_saved':
            terms.append(f"mul_saved('{o[1]}')")
        elif o[0] == 'sub_saved':
            terms.append(f"sub_saved('{o[1]}')")
        elif o[0] == 'leaky_relu':
            terms.append(f'leaky_relu({o[1]})')
        elif o[0] == 'reduction':
            terms.append(f'reduction({o[1]})')
            if len(o) > 2 and o[2] is not None:
                terms_meta.append(f'%% reduce_width {o[2]}')
        elif o[0] == 'reduction_chan':
            terms.append(f'reduction_chan({o[1]})')
            if len(o) > 2 and o[2] is not None:
                terms_meta.append(f'%% chan_count {o[2]}')
            if len(o) > 3 and o[3] is not None:
                terms_meta.append(f'%% spatial_inner {o[3]}')
        elif o[0] == 'reduction_tuple':
            terms.append(f'reduction_tuple({o[1]})')
            if len(o) > 2 and o[2] is not None:
                terms_meta.append(f'%% tuple_n {o[2]}')
            if len(o) > 3 and o[3] is not None:
                terms_meta.append(f'%% tuple_rows {o[3]}')
        elif o[0] in ('add_param', 'mul_param', 'sub_param',
                      'add_param_scalar', 'mul_param_scalar',
                      'sub_param_scalar', 'add_param_chan',
                      'mul_param_chan', 'sub_param_chan',
                      'min_param_scalar', 'max_param_scalar'):
            terms.append(f"{o[0]}('{o[1]}')")
        elif o[0] == 'save':
            terms.append(f"save('{o[1]}')")
        elif o[0] == 'add_saved':
            terms.append(f"add_saved('{o[1]}')")
        elif o[0] == 'reduction_chan_fold':
            terms.append(f'reduction_chan_fold({o[1]})')
            terms_meta.append(f'%% fold_chan {o[2]}')
            terms_meta.append(f'%% fold_spatial {o[3]}')
        elif o[0] == 'avgpool3d_k2':
            terms.append('avgpool3d_k2')
        elif o[0] == 'sub_own_mean':
            # the self-referential spatial-mean subtract (#15):
            # only the all-spatial form (dims = every dim after
            # batch+chan) emits — the row-form (rows=B*C,
            # width=prod(spatial)):
            terms.append(f"sub_own_mean({','.join(str(d) for d in o[1])})")
        elif o[0] in ('relu', 'tanh', 'sigmoid', 'gelu', 'mish',
                      'hardswish', 'hardtanh'):
            terms.append(o[0])
        else:
            # ★ the REASON is already carried in o[1] and was being discarded.
            # An UNSUPPORTED term that does not say WHY sends the reader back to
            # the source to re-derive what the lifter already knew.
            _why = o[1] if len(o) > 1 and isinstance(o[1], str) else None
            terms.append('UNSUPPORTED_%s%s' % (o[0], (':' + _why) if _why else ''))
    fact = f"chain({pid}, [{', '.join(terms)}])."
    if terms_meta:
        fact += '\n' + '\n'.join(terms_meta)
    return fact


def to_prolog_segments(pid, ops):
    ops = _avgpool_guard(ops)
    """Multi-stage form: one chain fact PER inter-stage segment that has
    ops. Returns (facts, n_segments, ok). Each segment must be fully
    expressible (no UNSUPPORTED terms) or ok=False (honest GAP —
    never emit a partial segment)."""
    # split ops by stage markers into segments:
    segments = []   # (after_stage_idx, [ops], stage_name)
    cur = []
    stage_no = 0
    empties = []    # (seg_idx, between-stages note) — the LEGIBLE ZEROS
    prev_stage = 'input'
    for o in ops:
        if o[0] == 'stage':
            if cur:
                segments.append((stage_no, cur))
            elif stage_no > 0:
                # consecutive stages with NOTHING between: emit a
                # legible zero, not a silent absence (Mavdil's #84
                # seg1 question — 'absence is ambiguous by
                # construction'; the reader must see the segmenter
                # CONSIDERED it):
                empties.append((stage_no, f'{prev_stage}→{o[1]}'))
            cur = []
            stage_no += 1
            prev_stage = o[1]
        else:
            cur.append(o)
    if cur:
        segments.append((stage_no, cur))
    facts = []
    ok = True
    for seg_idx, seg_ops in segments:
        fake = [('stage', 'seg')] + seg_ops   # reuse to_prolog's epilogue path
        fact = to_prolog(f'{pid}_seg{seg_idx}', fake)
        if 'UNSUPPORTED' in fact:
            ok = False
        facts.append(fact)
    for seg_idx, note in empties:
        facts.append(f'%% seg{seg_idx}: no fusable content ({note}, '
                     f'consecutive torch stages) — considered, not dropped')
    return facts, len(segments), ok


if __name__ == '__main__':
    path = sys.argv[1]
    pid = sys.argv[2] if len(sys.argv) > 2 else 'problem'
    ops = lift(path)
    if len(sys.argv) > 3 and sys.argv[3] == '--wrapper-info':
        # THE WRAPPER-EMITTER's export (the whole-model gate, Heath's
        # unlock): the full ops walk as JSON — stage markers (torch's
        # prefix, in order) + the lifted tail ops. The split point IS
        # where the leading stage-run ends.
        import json
        # the SAME guard the emit path applies (the #50 skip: the
        # wrapper counted the RAW walk — 1 lifted run — while the
        # emit's guard had reverted mid-chain avgpools to stages —
        # 2 segments; the walks MUST match):
        print(json.dumps({'pid': pid,
                          'ops': [list(o)
                                  for o in _avgpool_guard(ops)]}))
        sys.exit(0)
    if len(sys.argv) > 3 and sys.argv[3] == '--segments':
        facts, n, ok = to_prolog_segments(pid, ops)
        for f in facts:
            print(f)
        if not ok:
            sys.exit(4)
    else:
        print(to_prolog(pid, ops))
