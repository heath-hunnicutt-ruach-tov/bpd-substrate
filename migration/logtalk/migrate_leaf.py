#!/usr/bin/env python3
"""Mechanized leaf migration: lib/<name>.pl -> logtalk/lib/<name>.lgt
Pattern from B2 leaf #1 (arch_params): protocol from module exports,
object with the .pl's clauses VERBATIM (everything after the module
directive), original untouched.  The migration is textual + minimal:
Logtalk changes the calling structure, not the logic."""
import re, sys, os

def migrate(name):
    src_path = f'lib/{name}.pl'
    txt = open(src_path).read()
    # Parse module directive: :- module(name, [exports]).
    m = re.search(r':-\s*module\(\s*' + name + r'\s*,\s*\[(.*?)\]\s*\)\s*\.', txt, re.S)
    if not m:
        print(f"SKIP {name}: no module directive (plain file — different pattern)")
        return False
    exports_raw = m.group(1)
    # exports like: foo/2, bar/3  (strip comments)
    exports = re.findall(r'([a-z_0-9]+)\s*/\s*(\d+)', exports_raw)
    # Body = everything AFTER the module directive line
    body = txt[m.end():]
    # Strip library use_module directives — the injected uses/2 replaces
    # them; keeping both causes 'Permission error: modify uses_object_
    # predicate' (found: llvm_match_status batch 5).
    body = re.sub(r':-\s*use_module\(library\([a-z_]+\)(?:\s*,\s*\[[^\]]*\])?\)\s*\.\n?', '', body)
    # foldl/4: Logtalk meta::foldl has DIFFERENT arg order than SWI
    # (meta: Closure,Acc0,List,Acc vs SWI: Closure,List,Acc0,Acc).
    # Verbatim clauses calling SWI-style foldl must escape to user
    # context (SWI library(apply) autoload) — found via safe_read's
    # claimed-bytes silent failure.
    body = re.sub(r'(?<![:\w])foldl\(', '{user:foldl}(', body)
    body = body.replace('{user:foldl}(', 'user_foldl(')
    if 'user_foldl(' in body:
        body += "\n    %% SWI-ordered foldl, escaped to user context.\n    user_foldl(G, L, A0, A) :- {foldl(G, L, A0, A)}.\n"
    # Comment-preserve the header (everything before module directive)
    header = txt[:m.start()]
    header_c = "\n".join('%% ' + l if l.strip() and not l.startswith('%') else l
                          for l in header.splitlines())
    pubs = "\n".join(f"    :- public({e}/{a})." for e, a in exports)
    # Logtalk objects don't inherit SWI's autoloaded library predicates
    # (append/3, select/3, etc. — module code gets them free, object code
    # does not: found via fusion_optimizer batch-2 compile error).
    # Inject a uses/2 for the common list predicates when the body
    # references them.
    list_preds = [(p, a) for p, a in
                  [('append',3),('select',3),('member',2),('length',2),
                   ('nth0',3),('nth1',3),('reverse',2),('msort',2),
                   ('sum_list',2),('max_list',2),('min_list',2),('last',2),
                   ('delete',3),('subtract',3),('permutation',2)]
                  if re.search(r'\b' + p + r'\(', body)]
    meta_preds = [(p, a) for p, a in
                  [('exclude',3),('include',3),('partition',4),
                   ('maplist',2),('maplist',3),('maplist',4)]
                  if re.search(r'\b' + p + r'\(', body)]
    uses_line = ("\n    :- uses(list, [" +
                 ", ".join(f"{p}/{a}" for p, a in list_preds) +
                 "]).\n" if list_preds else "")
    uses_line += ("    :- uses(meta, [" +
                  ", ".join(f"{p}/{a}" for p, a in meta_preds) +
                  "]).\n" if meta_preds else "")
    lgt = f""":- protocol({name}p).
{pubs}
:- end_protocol.

:- object({name},
    implements({name}p)).
{uses_line}
{body.rstrip()}

:- end_object.
"""
    out = f'logtalk/lib/{name}.lgt'
    open(out, 'w').write(lgt)
    print(f"MIGRATED {name}: {len(exports)} exports, {len(body.splitlines())} body lines -> {out}")
    return True

if __name__ == '__main__':
    for name in sys.argv[1:]:
        migrate(name)
