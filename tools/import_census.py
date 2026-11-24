"""Import census for the Astral Chain port: which imports of main need host overrides.

Loads the extracted ExeFS modules of 1.0.1 (main and sdk, what runtime/loader.c loads) with recomp/nso.py, parses the
HOST_OVERRIDE / HOST_OBSERVER names out of runtime/*.c (string literals, or macros defined to one), and reports with
the loader's rules:

- the imports (undefined symbols) of both that neither module defines and the host does not override (a strong one
  stops the game when it is called),
- imports of main grouped by namespace, "host" in front of those with an override, to plan the service
  implementations,
- overrides that no relocation of either module names, and observers that no JUMP_SLOT names (candidates for
  pruning). An override that only the sdk's own relocations use is bound, so it is not one.

The overrides registered at run time (hostfn_add_override: ov_libc.c's six libc functions) are not listed; the sdk
defines all six. No game files are copied anywhere; only names are printed.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'recomp'))

from nso import Nso, R_AARCH64_JUMP_SLOT, R_AARCH64_RELATIVE   # noqa: E402

EXEFS = ROOT / 'build' / 'extracted' / 'v101' / 'exefs'
MODULES = ('main', 'sdk')
STB_WEAK = 2
STRING = r'"(?:[^"\\\n]|\\.)*"'


def host_names(kind):
    runtime = ROOT / 'runtime'
    define = re.compile(r'^#define[ \t]+(\w+)[ \t\\\n]+(%s)' % STRING, re.M)
    shared = {}
    for path in runtime.glob('*.h'):
        shared.update(define.findall(path.read_text(encoding='utf-8', errors='replace')))
    entry = re.compile(r'^HOST_%s\(\s*((?:%s\s*)+|\w+)\s*,' % (kind, STRING), re.M)
    names = set()
    for path in sorted(runtime.glob('*.c')):
        text = path.read_text(encoding='utf-8', errors='replace')
        macros = {**shared, **dict(define.findall(text))}
        for arg in entry.findall(text):
            literal = arg if arg.startswith('"') else macros.get(arg)
            if literal is None:
                sys.exit(f'import_census: {path.name}: HOST_{kind}({arg}) is neither a string nor a macro of one')
            names.add(''.join(part[1:-1] for part in re.findall(STRING, literal)))
    return names


def namespace(name):
    """nn::<area> for the SDK's C++ names, the outer name for another nested one."""
    rest = re.match(r'_ZNK?(.*)', name)
    if not rest:
        return 'C++' if name.startswith('_Z') else 'C'
    parts, text = [], rest.group(1)
    while len(parts) < 2:
        size = re.match(r'\d+', text)
        if not size:
            break
        start, end = size.end(), size.end() + int(size.group())
        parts.append(text[start:end])
        text = text[end:]
        if parts[0] != 'nn':
            break
    return '::'.join(parts) or 'C++'


def main():
    missing = [m for m in MODULES if not (EXEFS / m).is_file()]
    if missing:
        sys.exit(f'import_census: no {", ".join(missing)} in {EXEFS}: run tools/extract.py first')
    modules = {m: Nso(EXEFS / m) for m in MODULES}
    overrides = host_names('OVERRIDE')
    observers = host_names('OBSERVER')

    defined = {s[0] for nso in modules.values() for s in nso.exports()}
    named, jumps = set(), set()   # what the relocations name
    unresolved = []
    for module, nso in modules.items():
        for _, kind, sym, _ in nso.relocs:
            if kind != R_AARCH64_RELATIVE:
                named.add(nso.syms[sym][0])
                if kind == R_AARCH64_JUMP_SLOT:
                    jumps.add(nso.syms[sym][0])
        for name, info, *_ in nso.imports():
            if name not in defined and name not in overrides and info >> 4 != STB_WEAK:
                unresolved.append((module, name))

    imports = sorted({s[0] for s in modules['main'].imports()})
    print(f'main imports: {len(imports)}, sdk imports: {len(modules["sdk"].imports())}, '
          f'sdk exports: {len(modules["sdk"].exports())}, host overrides: {len(overrides)}, observers: {len(observers)}')
    print(f'unresolved strong imports (the game stops when it calls one): {len(unresolved)}')
    for module, name in sorted(unresolved):
        print('  UNRESOLVED', module, name)

    groups = {}
    for name in imports:
        groups.setdefault(namespace(name), []).append(name)
    print('\nimports of main by namespace (with a host override / all):')
    for ns in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        names = groups[ns]
        print(f'  {ns}: {sum(n in overrides for n in names)} / {len(names)}')
        for name in names:
            print(f'    {"host" if name in overrides else "    "}  {name}')

    dead_ovr = sorted(overrides - named)
    dead_obs = sorted(observers - jumps)
    print(f'\noverrides no relocation uses: {len(dead_ovr)}')
    for name in dead_ovr:
        print('  DEAD', name)
    print(f'observers no JUMP_SLOT uses: {len(dead_obs)}')
    for name in dead_obs:
        print('  DEAD-OBS', name)


if __name__ == '__main__':
    main()
