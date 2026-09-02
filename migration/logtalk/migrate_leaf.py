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
    # Comment-preserve the header (everything before module directive)
    header = txt[:m.start()]
    header_c = "\n".join('%% ' + l if l.strip() and not l.startswith('%') else l
                          for l in header.splitlines())
    pubs = "\n".join(f"    :- public({e}/{a})." for e, a in exports)
    lgt = f""":- protocol({name}p).
{pubs}
:- end_protocol.

:- object({name},
    implements({name}p)).

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
