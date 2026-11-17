"""Count the LDXP/STXP sites of generated C and the ones that are the atomic 128-bit read idiom: a pair loaded and at
once stored back unchanged to the same address. With a profile and a map, weight them by the samples of their function.

usage: ldxp_scan.py <generated dir> [<profile> <map>]
"""
import bisect
import collections
import re
import sys
from pathlib import Path

LD = re.compile(r'^\s*\{ __uint128_t p = ldxp128\(c, (.+?)\); (\S+) = \(uint64_t\)\(\(uint64_t\)p\); (\S+) = \(uint64_t\)\(\(uint64_t\)\(p >> 64\)\); \}$')
ST = re.compile(r'^\s*\{ uint32_t st = stxp128\(c, (.+?), (\S+), (\S+)\); (\S+) = \(uint32_t\)\(st\); \}$')
FN = re.compile(r'^void (f_[0-9a-f]+)\(Cpu \*c\) \{$')

gen = Path(sys.argv[1])
per_fn = collections.Counter()
per_fn_all = collections.Counter()
ld = st = idiom = 0
for f in sorted(gen.glob('*/*_[0-9]*.c')):
    lines = f.read_text(encoding='utf-8').split('\n')
    cur = None
    for i, l in enumerate(lines):
        m = FN.match(l)
        if m:
            cur = m.group(1)
            continue
        a = LD.match(l)
        if a:
            ld += 1
            per_fn_all[cur] += 1
            b = ST.match(lines[i + 1]) if i + 1 < len(lines) else None
            if b and b.group(1) == a.group(1) and b.group(2) == a.group(2) and b.group(3) == a.group(3):
                idiom += 1
                per_fn[cur] += 1
        elif 'stxp128(' in l:
            st += 1
print(f'{gen}: ldxp128 {ld}, stxp128 {st}, read idiom (stored back unchanged at once) {idiom}')
if len(sys.argv) > 3:
    SYM = re.compile(r'^([0-9a-f]{8}) +[0-9a-f]{8} +\d+ {2,}(\S+)$')
    syms = {}
    for line in open(sys.argv[3], encoding='utf-8', errors='replace'):
        m = SYM.match(line.rstrip('\n'))
        if m and '(' not in m.group(2):
            syms.setdefault(int(m.group(1), 16), m.group(2))
    addrs = sorted(syms)
    base, cnt, run = 0, collections.Counter(), 0
    for line in open(sys.argv[2]):
        p = line.split()
        if p and p[0] == 'base':
            base = int(p[1], 16)
        elif p and p[0] == 'rip':
            i = bisect.bisect_right(addrs, int(p[1], 16) - base) - 1
            if i >= 0 and base <= int(p[1], 16) < base + addrs[-1] + (1 << 20):
                cnt[syms[addrs[i]]] += int(p[2])
    guest = sum(n for f, n in cnt.items() if f.startswith('f_'))
    with_ld = sum(n for f, n in cnt.items() if per_fn_all.get(f))
    with_idiom = sum(n for f, n in cnt.items() if per_fn.get(f))
    print(f'samples in lifted code {guest}; in functions with an LDXP {with_ld} ({100 * with_ld / guest:.1f}%); with the read idiom '
          f'{with_idiom} ({100 * with_idiom / guest:.1f}%)')
    for f, n in sorted(((f, n) for f, n in cnt.items() if per_fn.get(f)), key=lambda x: -x[1])[:25]:
        print(f'  {n:6d} {100 * n / guest:5.2f}%  {f}: {per_fn[f]} idioms of {per_fn_all[f]} LDXP')
