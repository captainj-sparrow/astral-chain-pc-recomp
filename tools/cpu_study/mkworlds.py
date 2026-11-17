"""Merge two lifts of the same game into one generated directory whose code can switch between them while running.

usage: mkworlds.py <gen A> <gen B> <out> [<line for the top of every world B chunk, e.g. "#define X 1">]

World A keeps its functions as f_<addr>, world B's become g_<addr>. Every direct call in both worlds turns into
`(rt_world ? g_<addr> : f_<addr>)(c)`, so a thread moves to the other world at its next call, and the dispatch table
of the study runtime (build/cpu-study/rt-ab) holds both worlds for the indirect calls. Both worlds pay the same
select per call, so the difference between the A and B blocks of a AC_AB run is the difference between the two lifts.
The output builds with tools/build_game.py --gen <out> --runtime build/cpu-study/rt-ab.
"""
import json
import re
import shutil
import sys
from pathlib import Path

CALL = re.compile(r'(?<!void )\bf_([0-9a-f]+)\(c\)')
PROTO = re.compile(r'^void f_([0-9a-f]+)\(Cpu \*c\);$', re.M)
DEFN = re.compile(r'^void f_([0-9a-f]+)\(Cpu \*c\) \{$', re.M)
INCLUDE = '#include "cpu.h"\n'


TOP_B = ''


def world(text, b):
    if b and TOP_B:
        first = '#define RT_VLOCALS' if '#define RT_VLOCALS' in text else '#include "cpu.h"'
        text = text.replace(first, TOP_B + '\n' + first, 1)
        if 'attribute push' in TOP_B:
            text += '\n#pragma clang attribute pop\n'
    text = PROTO.sub(r'void f_\1(Cpu *c); void g_\1(Cpu *c);', text)
    text = CALL.sub(r'(rt_world ? g_\1 : f_\1)(c)', text)
    if b:
        text = DEFN.sub(r'void g_\1(Cpu *c) {', text)
    assert text.count(INCLUDE) == 1
    return text.replace(INCLUDE, INCLUDE + 'extern int rt_world;\n')


def table_b(text, name):
    for ident in ('fn_addrs', 'fn_ptrs', 'fn_count'):
        text = text.replace(f'{name}_{ident}', f'{name}_{ident}_b')
    return re.sub(r'\bf_([0-9a-f]+)\b', r'g_\1', text)


def main():
    global TOP_B
    a, b, out = (Path(p) for p in sys.argv[1:4])
    TOP_B = sys.argv[4] if len(sys.argv) > 4 else ''
    out.mkdir(parents=True, exist_ok=True)
    for f in ('layout.h', 'layout.json', 'patches.h'):
        shutil.copy2(a / f, out / f)
    for mod in json.loads((a / 'layout.json').read_text()):
        name = mod['name']
        ma = json.loads((a / name / 'manifest.json').read_text())
        mb = json.loads((b / name / 'manifest.json').read_text())
        assert ma['function_count'] == mb['function_count'], name
        d = out / name
        d.mkdir(exist_ok=True)
        chunks = []
        for src, m, isb in ((a, ma, False), (b, mb, True)):
            for c in m['chunks']:
                dst = c.replace(f'{name}_', f'{name}_b') if isb else c
                text = world((src / name / c).read_text(encoding='utf-8'), isb)
                p = d / dst
                if not p.is_file() or p.read_text(encoding='utf-8') != text:
                    p.write_text(text, encoding='utf-8', newline='\n')
                chunks.append(dst)
        shutil.copy2(a / name / ma['table'], d / ma['table'])
        tb = f'{name}_tableb.c'
        (d / tb).write_text(table_b((b / name / mb['table']).read_text(encoding='utf-8'), name), encoding='utf-8', newline='\n')
        chunks.append(tb)
        man = dict(ma)
        man['chunks'] = chunks
        man.pop('chunk_info', None)
        (d / 'manifest.json').write_text(json.dumps(man))
        print(f'{name}: {len(chunks)} files')


if __name__ == '__main__':
    main()
