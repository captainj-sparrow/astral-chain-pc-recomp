"""Call-chain queries on an AC_PROFILE profile taken with chains (AC_PROFILE_RENDER=1 or AC_PROFILE_CHAINS=1).

usage: chains.py <profile> <totk.map> self [N]            self time per function, running samples only
       chains.py <profile> <totk.map> incl [N]            inclusive time per function
       chains.py <profile> <totk.map> callers <fn> [N]    immediate callers of fn (by samples at or below fn)
       chains.py <profile> <totk.map> callees <fn> [N]    what fn's inclusive time is made of (immediate callees + self)
       chains.py <profile> <totk.map> paths <fn> [N] [D]  the most common caller paths of depth D down to fn
"""
import bisect
import collections
import re
import sys

SYM = re.compile(r'^([0-9a-f]{8}) +[0-9a-f]{8} +\d+ {2,}(\S+)$')


def load_map(path):
    syms = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = SYM.match(line.rstrip('\n'))
            if m and '(' not in m.group(2):
                syms.setdefault(int(m.group(1), 16), m.group(2))
    addrs = sorted(syms)
    return addrs, [syms[a] for a in addrs]


def main():
    prof, mapf, cmd = sys.argv[1:4]
    args = sys.argv[4:]
    addrs, names = load_map(mapf)
    base, mods, stacks = 0, [], []
    for line in open(prof):
        p = line.split()
        if not p:
            continue
        if p[0] == 'base':
            base = int(p[1], 16)
        elif p[0] == 'module':
            q = line.split(None, 3)
            mods.append((int(q[1], 16), int(q[2], 16), q[3].strip().replace('\\', '/').rsplit('/', 1)[-1]))
        elif p[0] == 'stack':
            stacks.append((int(p[1]), [int(x, 16) for x in p[2:]]))
    exe_end = base + (addrs[-1] if addrs else 0) + (1 << 20)
    cache = {}

    def name(rip):
        if rip in cache:
            return cache[rip]
        r = None
        if base <= rip < exe_end:
            i = bisect.bisect_right(addrs, rip - base) - 1
            r = names[i] if i >= 0 else 'exe?'
        else:
            for b, s, n in mods:
                if b <= rip < b + s:
                    r = n
                    break
        cache[rip] = r or 'unknown'
        return cache[rip]

    BLOCKED = ('ntdll.dll', 'win32u.dll', 'KERNELBASE.dll')
    chains = []
    for n, rips in stacks:
        c = [name(r) for r in rips]
        chains.append((n, c))
    total = sum(n for n, _ in chains)
    running = sum(n for n, c in chains if c[0] not in BLOCKED)
    print(f'{total} samples, {running} running')
    if cmd == 'self':
        cnt = collections.Counter()
        for n, c in chains:
            if c[0] not in BLOCKED:
                cnt[c[0]] += n
        for f, n in cnt.most_common(int(args[0]) if args else 40):
            print(f'{100 * n / running:6.2f}% {n:7d}  {f}')
    elif cmd == 'incl':
        cnt = collections.Counter()
        for n, c in chains:
            if c[0] in BLOCKED:
                continue
            for f in set(c):
                cnt[f] += n
        for f, n in cnt.most_common(int(args[0]) if args else 40):
            print(f'{100 * n / running:6.2f}% {n:7d}  {f}')
    elif cmd in ('callers', 'callees', 'paths'):
        fn = args[0]
        top = int(args[1]) if len(args) > 1 else 25
        depth = int(args[2]) if len(args) > 2 else 4
        cnt = collections.Counter()
        tot = 0
        for n, c in chains:
            if c[0] in BLOCKED or fn not in c:
                continue
            i = c.index(fn)   # innermost occurrence
            tot += n
            if cmd == 'callers':
                cnt[c[i + 1] if i + 1 < len(c) else '(cut)'] += n
            elif cmd == 'callees':
                cnt[c[i - 1] if i else '(self)'] += n
            else:
                cnt[' < '.join(c[i:i + 1 + depth])] += n
        print(f'{fn}: {tot} samples inclusive, {100 * tot / running:.2f}% of running')
        for f, n in cnt.most_common(top):
            print(f'{100 * n / running:6.2f}% {n:7d}  {f}')


if __name__ == '__main__':
    main()
