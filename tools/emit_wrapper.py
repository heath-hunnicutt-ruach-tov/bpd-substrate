#!/usr/bin/env python3
"""THE WRAPPER-EMITTER (the whole-model gate — Heath's unlock).

Generates a Model-equivalent Python wrapper per problem:
  torch runs the stage-prefix (MODULE-REUSE: the benchmark Model's own
  modules, same code, same mode — the mode-check lesson), our compiled
  CUDA kernel runs each lifted segment, torch stages replay BETWEEN
  segments where the walk interleaves them (the #8 shape).

The gate then becomes: get_inputs() -> Model AND wrapper -> ulp-diff.

Usage: emit_wrapper.py <problem.py> <pid> <kernel.cu> [out.py]
The kernel.cu must carry CALL-MANIFEST headers (launch geometry,
params) — the manifests ARE the wrapper's inputs.
"""
import json
import re
import subprocess
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wrapper_source(problem_path, pid, ops, manifests):
    """Build the wrapper .py source. ops = the lift walk (stages +
    lifted tail ops). manifests = parsed CALL-MANIFEST dicts (one per
    emitted kernel, in launch order)."""
    # split the walk into runs: torch-stage runs vs lifted runs:
    runs = []
    cur_kind = None
    for o in ops:
        kind = 'stage' if o[0] == 'stage' else 'lifted'
        if kind != cur_kind:
            runs.append((kind, []))
            cur_kind = kind
        runs[-1][1].append(o)
    # pure-save runs emit NO kernel (the save marks a capture point;
    # the saved tensor reaches the kernel as x2 — #51):
    runs = [(k, ops_) for k, ops_ in runs
            if not (k == 'lifted'
                    and all(o[0] == 'save' for o in ops_))]
    n_lifted = sum(1 for k, _ in runs if k == 'lifted')
    # SPLIT-UNIT GROUPING (the #42 unlock): a split route emits
    # multiple kernels for ONE lifted run, with suffixed pids
    # (imp42_t, imp42_r) — group consecutive suffixed manifests into
    # one chain; the wrapper launches them sequentially (kernel i's
    # out feeds kernel i+1):
    mgroups = []
    for mn in manifests:
        mpid = mn.get('pid', '')
        _sfx = mpid[len(pid):].lstrip('_') if mpid.startswith(pid) \
            else ''
        _is_chain_part = (_sfx and not _sfx.startswith('seg'))
        if (_is_chain_part and mgroups):
            # non-seg suffix (_t/_r — the split route): chain into
            # the previous group (ONE lifted run, kernels sequential).
            # _segN manifests are SEPARATE lifted runs — never chained.
            mgroups[-1].append(mn)
        else:
            mgroups.append([mn])
    if n_lifted != len(mgroups):
        raise SystemExit(
            f'{pid}: {n_lifted} lifted runs but {len(mgroups)} '
            f'manifest groups ({len(manifests)} kernels) — the wrapper '
            f'cannot map them; refuse.')

    _has_input2 = any('input2' in str(mn.get('fact', ''))
                      for mn in manifests)
    lines = [
        f'"""Auto-emitted whole-model wrapper for {pid}.',
        'torch runs the stage-prefix (module-reuse); the emitted CUDA',
        'kernel(s) run the lifted tail. The gate: get_inputs() -> both',
        '-> ulp-diff."""',
        'import importlib.util as _ilu',
        'import torch.nn.functional as _F',
        'import torch',
        '',
        '# ABSOLUTE path resolved at emit time (Mavdil: a wrapper that',
        '# only runs from one directory is a wrapper the next person',
        '# reports as broken):',
        f'_spec = _ilu.spec_from_file_location('
        f'"_prob", {os.path.abspath(problem_path)!r})',
        '_prob = _ilu.module_from_spec(_spec)',
        '_spec.loader.exec_module(_prob)',
        '',
        '',
        'class WrappedModel(torch.nn.Module):',
        '    """Drop-in Model replacement: same init args, same forward',
        '    contract; the lifted tail runs on the emitted kernel."""',
        '',
        '    def __init__(self, *a, **kw):',
        '        super().__init__()',
        '        self.inner = _prob.Model(*a, **kw)',
        '        from tools.kernel_runner import KernelRunner',
        f'        self.runner = KernelRunner({pid!r})',
        '',
        ('    def forward(self, x, *extra):' if _has_input2
         else '    def forward(self, x):'),
    ]
    mi = 0
    for kind, run in runs:
        if kind == 'stage':
            for o in run:
                name = o[1]
                if name == 'm_matmul':
                    # the #14 GEMM stage: torch.matmul(x, W.T)
                    # with W = the recorded model attr:
                    _wa = (o[2] or ['weight'])[0]
                    lines.append(
                        f'        x = torch.matmul('
                        f'x, model.{_wa}.T)')
                    continue
                if name.startswith('m_'):
                    # tensor-method shape-op replay (the #55/#98
                    # unlock): args recorded literal at lift:
                    margs = o[2] if len(o) > 2 else None
                    if margs is None:
                        raise SystemExit(
                            f'{pid}: shape-op {name} with unresolved '
                            f'args; refuse.')
                    astr = ', '.join(repr(a) for a in margs)
                    lines.append(
                        f'        x = x.{name[2:]}({astr})')
                    continue
                if name.startswith('F_'):
                    fargs = o[2] if len(o) > 2 else None
                    if fargs is None:
                        raise SystemExit(
                            f'{pid}: functional stage {name} with '
                            f'unresolved args; refuse.')
                    astr = ', '.join(repr(a) for a in fargs)
                    sep = ', ' if astr else ''
                    lines.append(
                        f'        x = _F.{name[2:]}(x{sep}{astr})')
                    continue
                lines.append(f'        x = self.inner.{name}(x)')
        else:
            # launch every kernel in this run's manifest group
            # sequentially (split units chain: out_i -> in_{i+1}):
            _flat = sum(len(g) for g in mgroups[:mi])
            for _j in range(len(mgroups[mi])):
                _m2 = mgroups[mi][_j]
                if 'input2' in str(_m2.get('fact', '')):
                    lines.append(
                        f'        x = self.runner.launch('
                        f'{_flat + _j}, x, x2=extra[0])')
                else:
                    lines.append(
                        f'        x = self.runner.launch('
                        f'{_flat + _j}, x)')
            mi += 1
    lines += [
        '        return x',
        '',
        '',
        '# what the KERNEL computes (the gate asserts this against the',
        '# manifest chain — a wrapper must declare what it did NOT do;',
        "# Mavdil's contract item 3):",
        f'COVERS = {[o[0] for o in ops if o[0] != "stage"]!r}',
        f'STAGES = {[o[1] for o in ops if o[0] == "stage" and not o[1].startswith(("m_", "F_"))]!r}',
        '# the full stage sequence incl shape-ops (for gate replay):',
        f'ALL_STAGES = {[list(o[1:]) for o in ops if o[0] == "stage"]!r}',
        '',
        '',
        f'RUNS = {[(k, [((o[1], (o[2] if len(o) > 2 else None)) if k == "stage" else o[0]) for o in r]) for k, r in runs]!r}',
        '',
        '',
        'def gate(seed=0):',
        '    """The whole-model gate, HOOK-CAPTURE form (Mavdil contract',
        '    item 1: cuDNN does NOT guarantee bit-identical output across',
        '    two invocations — the prefix must be captured from the',
        '    Model\'s OWN forward, never re-run; his #16 measurement:',
        '    3.5M false diffs re-run vs 0 hooked). ONE Model forward with',
        '    a hook on the last stage; the captured tensor feeds our',
        '    kernel; the Model\'s own output is the reference. No .eval()',
        '    (contract item 2: mode stated, stateful modules invoked',
        '    exactly once)."""',
        '    import torch as _t',
        '    import numpy as np',
        '    _t.manual_seed(seed)',
        '    init = _prob.get_init_inputs()',
        '    m = _prob.Model(*init).cuda()   # NO .eval() — the benchmark',
        '                                    # mode is the constructed mode',
        '    ins = [t.cuda() if hasattr(t, "cuda") else t',
        '           for t in _prob.get_inputs()]',
        '    # HOOK EVERY STAGE (Mavdil design (A) + the composition',
        '    # assertion): ONE Model forward captures every boundary;',
        '    # each lifted segment gates against its captured pair; the',
        '    # composition link is ASSERTED, not assumed.',
        '    _flat_stages = ALL_STAGES',
        '    cap_out, cap_in = {}, {}',
        '    hooks = []',
        '    for s in STAGES:',
        '        mod = getattr(m, s)',
        '        # CPU capture (Mavdil: the hook holding a GPU',
        '        # intermediate through the forward IS the OOM peak;',
        '        # off-device at capture, back at use):',
        '        hooks.append(mod.register_forward_hook(',
        '            lambda mo, i, o, _s=s:',
        '            cap_out.__setitem__(_s, o.detach().cpu())))',
        '        hooks.append(mod.register_forward_pre_hook(',
        '            lambda mo, i, _s=s:',
        '            cap_in.__setitem__(_s, i[0].detach().cpu())))',
        '    with _t.no_grad():',
        '        ref = m(*ins)               # the ONE forward',
        '    ref = ref.detach().cpu()',
        '    torch.cuda.empty_cache()  # the launch phase needs the',
        '    # headroom (ref+src+out together OOM the big units —',
        '    # the peak moves, it does not vanish; m stays for the',
        '    # runner param resolution)',
        '    for h in hooks:',
        '        h.remove()',
        '    from tools.kernel_runner import KernelRunner',
        f'    runner = KernelRunner({pid!r}, model=m)',
        '    results = {}',
        f'    GROUP_SIZES = {[len(g) for g in mgroups]!r}',
        '    ki = 0',
        '    stage_idx = -1',
        '    for ri, (kind, names) in enumerate(RUNS):',
        '        if kind == "stage":',
        '            stage_idx += len(names)',
        '            continue',
        '        # this lifted run: input = the preceding stage\'s',
        '        # captured OUTPUT; expected = the next stage\'s captured',
        '        # INPUT (or the model output if this run is last):',
        '        # the preceding boundary: the last MODULE stage\'s',
        '        # captured output, with any TRAILING shape-ops',
        '        # replayed (the #55 shape — pool(x.unsqueeze).squeeze):',
        '        _mods_before = [s for s in _flat_stages[:stage_idx + 1]',
        '                        if not s[0].startswith(("m_", "F_"))]',
        '        if _mods_before:',
        '            src = cap_out[_mods_before[-1][0]].cuda()',
        '            _last_mod_pos = max(_j for _j, s in',
        '                                enumerate(',
        '                                    _flat_stages[:stage_idx + 1])',
        '                                if not s[0].startswith(',
        '                                    ("m_", "F_")))',
        '        else:',
        '            # NO module stage before this run (the #14 form:',
        '            # a functional/tensor-op prefix only) — replay',
        '            # from the raw input:',
        '            src = ins[0]',
        '            _last_mod_pos = -1',
        '        for _s in _flat_stages[_last_mod_pos + 1:stage_idx + 1]:',
        '            _nm = _s[0]',
        '            if _nm == "m_matmul":',
        '                src = torch.matmul(',
        '                    src, getattr(m, (_s[1] or ["weight"])[0]).T)',
        '            elif _nm.startswith("m_"):',
        '                src = getattr(src, _nm[2:])(*(_s[1] or []))',
        '        nxt = None',
        '        _suffix = []',
        '        for kind2, names2 in RUNS[ri + 1:]:',
        '            if kind2 == "stage":',
        '                for _nm2 in names2:',
        '                    _s2 = _nm2 if isinstance(_nm2, str) else _nm2',
        '                if all((n if isinstance(n, str) else n[0])',
        '                       .startswith(("m_", "F_"))',
        '                       for n in names2):',
        '                    _suffix.extend(names2)',
        '                    continue',
        '                nxt = next(',
        '                    (n if isinstance(n, str) else n[0])',
        '                    for n in names2',
        '                    if not (n if isinstance(n, str) else n[0])',
        '                    .startswith(("m_", "F_")))',
        '                break',
        '        # expected STAYS CPU — the diff site reads via',
        '        # .cpu(); uploading it cost 2GB against the launch',
        '        # headroom (the #19 OOM):',
        '        expected = (cap_in[nxt] if nxt is not None',
        '                    else ref)',
        '        _flat_ki = sum(GROUP_SIZES[:ki])',
        '        got = src',
        '        for _gj in range(GROUP_SIZES[ki]):',
        '            _mf = runner.manifests[_flat_ki + _gj]',
        '            if "input2" in str(_mf.get("fact", "")):',
        '                got = runner.launch(_flat_ki + _gj, got,',
        '                                    x2=ins[1])',
        '            elif "add_saved" in str(_mf.get("fact", "")):',
        '                # the saved tensor: an INTERMEDIATE capture',
        '                # (#92: add_saved(x_conv) → cap_out[conv])',
        '                # or the model INPUT (#51: original_x =',
        '                # a save-seg0 of ins[0]):',
        '                import re as _re2',
        '                _sm = _re2.search(',
        r'                    "add_saved\\(.(\\w+).\\)",',
        '                    str(_mf.get("fact", "")))',
        '                _sv = _sm.group(1) if _sm else ""',
        '                _x2 = None',
        '                if _sv.startswith("x_") and \\',
        '                        _sv[2:] in cap_out:',
        '                    _x2 = cap_out[_sv[2:]].cuda()',
        '                if _x2 is None:',
        '                    _x2 = ins[0]',
        '                got = runner.launch(_flat_ki + _gj, got,',
        '                                    x2=_x2)',
        '            else:',
        '                got = runner.launch(_flat_ki + _gj, got)',
        '        if nxt is None and _suffix:',
        '            # non-module suffix (F_/m_ stages): replay on OUR',
        '            # output; the comparison stays terminal vs ref:',
        '            for _sfx in _suffix:',
        '                _snm = _sfx if isinstance(_sfx, str) else _sfx[0]',
        '                _sar = ([] if isinstance(_sfx, str)',
        '                        else (_sfx[1] or []))',
        '                if _snm.startswith("m_"):',
        '                    got = getattr(got, _snm[2:])(*_sar)',
        '                else:',
        '                    import torch.nn.functional as _Fx',
        '                    got = getattr(_Fx, _snm[2:])(got, *_sar)',
        '        r = expected.cpu().numpy().ravel()',
        '        g = got.cpu().numpy().ravel().reshape(r.shape)',
        '        ulp = np.abs(r.view(np.int32).astype(np.int64) -',
        '                     g.view(np.int32).astype(np.int64))',
        '        results[f"seg{ki}"] = {',
        '            "N": int(r.size), "n_diff": int((r != g).sum()),',
        '            "max_ulp": int(ulp.max()),',
        '            "composition": "ASSERTED" if nxt is not None',
        '                           else "terminal"}',
        '        ki += 1',
        '    return results',
        '',
        '',
        'get_inputs = _prob.get_inputs',
        'get_init_inputs = _prob.get_init_inputs',
        '',
        '',
        'if __name__ == "__main__":',
        '    # self-demonstrating (Mavdil: a library with no __main__',
        '    # looks exactly like a hang):',
        '    print(gate())',
        '',
    ]
    src = '\n'.join(lines)
    # the runner import must work from ANY cwd (his PYTHONPATH catch):
    src = src.replace(
        'import importlib.util as _ilu',
        'import sys as _sys\n'
        f'_sys.path.insert(0, {REPO!r})  # tools.kernel_runner '
        '(any-cwd bootstrap)\n'
        'import importlib.util as _ilu')
    return src


def main():
    problem_path, pid, kernel_cu = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else f'wrapper_{pid}.py'
    r = subprocess.run(
        ['python3', os.path.join(REPO, 'lib', 'lift_chain.py'),
         problem_path, pid, '--wrapper-info'],
        capture_output=True, text=True)
    info = json.loads(r.stdout)
    cu = open(kernel_cu).read()
    manifests = [json.loads(m) for m in
                 re.findall(r'/\* CALL-MANIFEST (.*?) \*/', cu)]
    # split-units carry the unit manifest + sub-manifests; keep the
    # per-kernel ones (sub-pids) if present, else the unit one:
    subs = [m for m in manifests if m['pid'] != pid]
    manifests = subs if subs else manifests
    src = wrapper_source(problem_path, pid, info['ops'], manifests)
    with open(out, 'w') as f:
        f.write(src)
    print(f'[wrapper] {out} ({len(src.splitlines())} lines)')


if __name__ == '__main__':
    main()
