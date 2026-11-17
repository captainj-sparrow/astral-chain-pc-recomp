"""x86 instructions per ARM instruction for the hottest lifted functions of a profile, in two builds' objects.

usage: expand.py <profile> <map> <objdir A> <objdir B> [N]
"""
import bisect
import collections
import re
import subprocess
import sys
from pathlib import Path

prof, mapf, da, db = sys.argv[1:5]
top = int(sys.argv[5]) if len(sys.argv) > 5 else 40
SYM = re.compile(r'^([0-9a-f]{8}) +[0-9a-f]{8} +\d+ {2,}(\S+)$')
syms = {}
for line in open(mapf, encoding='utf-8', errors='replace'):
    m = SYM.match(line.rstrip('\n'))
    if m and '(' not in m.group(2):
        syms.setdefault(int(m.group(1), 16), m.group(2))
addrs = sorted(syms)
base = 0
cnt = collections.Counter()
for line in open(prof):
    p = line.split()
    if p and p[0] == 'base':
        base = int(p[1], 16)
    elif p and p[0] == 'rip':
        i = bisect.bisect_right(addrs, int(p[1], 16) - base) - 1
        if i >= 0:
            cnt[syms[addrs[i]]] += int(p[2])
guest = [(f, n) for f, n in cnt.most_common() if f.startswith('f_7000')][:top]
starts = sorted(int(s[2:], 16) for s in syms.values() if re.fullmatch(r'f_[0-9a-f]+', s))
OBJDUMP = 'C:/msys64/ucrt64/bin/objdump.exe'


def index(d):
    """symbol -> object, from one nm-like pass: the chunk that defines f_X is found through the generated C names."""
    out = {}
    for c in Path('generated').glob('*/main/main_*.c'):
        pass
    return out


def count(objdir, gen, f):
    # the chunk: the C file that defines the function
    for c in sorted(Path(gen, 'main').glob('main_*.c')):
        if f'\nvoid {f}(Cpu *c) {{' in c.read_text(encoding='utf-8', errors='replace'):
            o = Path(objdir, 'main', c.stem + '.o')
            s = subprocess.run([OBJDUMP, '-d', '-M', 'intel', '--no-show-raw-insn', f'--disassemble={f}', str(o)],
                               capture_output=True, text=True).stdout
            ins = [l for l in s.splitlines() if re.match(r'^\s+[0-9a-f]+:\t', l)]
            mem = sum(1 for l in ins if 'PTR [' in l)
            stores = sum(1 for l in ins if re.search(r'\t(mov|vmov\w*|movzx)\s+\w+ PTR \[', l) or re.search(r'PTR \[[^\]]*\],', l))
            return len(ins), mem, stores
    return 0, 0, 0


gens = {'A': ('generated/v101', da), 'B': ('generated/cs-abi', db)}
tot = collections.Counter()
print(f'{"function":16s} {"samples":>7s} {"arm":>5s}  {"x86 A":>6s} {"mem":>5s} {"st":>5s}  {"x86 B":>6s} {"mem":>5s} {"st":>5s}  ratio A  B')
for f, n in guest:
    a = int(f[2:], 16)
    i = bisect.bisect_right(starts, a)
    arm = (starts[i] - a) // 4 if i < len(starts) else 0
    ia, ma, sa = count(da, 'generated/v101', f)
    ib, mb, sb = count(db, 'generated/cs-abi', f)
    if not arm or not ia or not ib:
        continue
    print(f'{f:16s} {n:7d} {arm:5d}  {ia:6d} {ma:5d} {sa:5d}  {ib:6d} {mb:5d} {sb:5d}  {ia / arm:5.2f} {ib / arm:5.2f}')
    for k, v in (('arm', arm), ('ia', ia), ('ma', ma), ('sa', sa), ('ib', ib), ('mb', mb), ('sb', sb)):
        tot[k] += v * n
    tot['n'] += n
n = tot['n']
print(f'weighted by samples: x86/arm A {tot["ia"] / tot["arm"]:.2f} B {tot["ib"] / tot["arm"]:.2f}; memory operands per arm insn A '
      f'{tot["ma"] / tot["arm"]:.2f} B {tot["mb"] / tot["arm"]:.2f}; stores A {tot["sa"] / tot["arm"]:.2f} B {tot["sb"] / tot["arm"]:.2f}')
