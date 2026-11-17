"""Which lifted functions break what gen.py --abi-trust assumes (AAPCS64 at calls and returns)?

For every function of main and sdk: a backward liveness over the lifted instructions, with a call killing X0-X17 and
using X0-X8, SP, X29. Reported:
  entry:  functions that read X9-X17 before writing them (a caller's temporary read as an implicit argument, or a
          fragment of another function reached by a branch)
  after:  call sites after which the caller reads X2-X17 before writing them (an implicit result, or a caller that
          knows its callee keeps the register)
Then how each `entry` function is reached: by a call (bl target, or address taken) or only by branches from other
functions (a tail exit stores every changed register, so those are safe).

usage: abi_scan.py [--jobs N] [--out report.json]
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'recomp'))

import numpy as np  # noqa: E402

import gen  # noqa: E402
import patches  # noqa: E402
import regs  # noqa: E402
from layout import layout  # noqa: E402
from module import Module  # noqa: E402

EXEFS = ROOT / 'build' / 'extracted' / 'v101' / 'exefs'
PATCHES = ROOT / 'recomp' / 'patches-v101.txt'
TEMPS = frozenset(range(9, 18))
CALLER_SAVED = frozenset(range(0, 18))
CALL_USE = frozenset({29, regs.SP})   # explicit reads only: arguments a call passes on stay in the Cpu either way
RET_USE = frozenset({0, 1, 30}) | regs.CALLEE_SAVED
AFTER_BAD = frozenset(range(2, 18))
WRITE_ALL = re.compile(r'c->(?:x\[(\d+)\]|(sp))\s*=(?!=)')
SVC = re.compile(r'\brt_svc\(c\b')

_W = {}


def rw(text):
    """(reads, writes) of one instruction's C: a register read is an occurrence that is not an assignment target."""
    occ = Counter(regs.refs(text))
    wr = Counter(regs.SP if m.group(2) else int(m.group(1)) for m in WRITE_ALL.finditer(text))
    reads = {r for r, n in occ.items() if n > wr.get(r, 0)}
    return reads, set(wr)


def init(mods, starts, table):
    _W['mods'], _W['starts'], _W['table'], _W['em'] = mods, starts, table, {}


def emitter(name):
    e = _W['em'].get(name)
    if e is None:
        path, base = _W['mods'][name]
        mod = Module(name, path, base, 'c')
        e = _W['em'][name] = gen.Emitter(mod, set(_W['starts'][name].tolist()), _W['table'].for_module(name, base),
                                         spins=_W['table'].spins_for(name, base))
    return e


def scan(task):
    name, fns = task
    e = emitter(name)
    mod = e.mod
    out = {'entry': [], 'after': [], 'bl': set(), 'tail': set(), 'plain': 0, 'fns': 0, 'calls': 0, 'blr': 0,
           'after_regs': Counter()}
    for entry, end in fns:
        out['fns'] += 1
        insns = [mod.lift(a) for a in range(entry, end, 4)]
        labels, tables = e.scan(insns, entry, end)
        if 'all' in tables.values():
            out['plain'] += 1
            continue
        succ = e.flow(insns, entry, end, tables)
        n = len(insns)
        use, kill, iscall, exit_use = [set() for _ in range(n + 1)], [set() for _ in range(n + 1)], [False] * (n + 1), [None] * (n + 1)
        for i, ins in enumerate(insns):
            a = entry + 4 * i
            k = ins.kind
            texts = []
            if a in e.patched:
                for words in e.replacements(a):
                    for w in words:
                        texts.append((w.code or '') + ' ' + (w.cond or '') + ' ' + (w.target if isinstance(w.target, str) else ''))
                        if w.kind == 'bl':
                            out['bl'].add(w.target)
                r = set()
                for t in texts + [(ins.code or '') + ' ' + (ins.cond or '') + ' ' + (ins.target if isinstance(ins.target, str) else '')]:
                    r |= rw(t)[0]
                use[i] = r   # no kill: the alternatives differ
                if k == 'bl':
                    out['bl'].add(ins.target)
                continue
            if k == 'op':
                r, w = rw(ins.code or '')
                if SVC.search(ins.code or ''):
                    r |= set(range(0, 8))
                    w |= set(range(0, 8))
                use[i], kill[i] = r, w
            elif k == 'bcond':
                use[i] = rw(ins.cond or '')[0]
                if not entry <= ins.target < end:
                    out['tail'].add(ins.target)
                    exit_use[i] = CALL_USE | regs.CALLEE_SAVED | {30}
            elif k == 'b':
                if not entry <= ins.target < end:
                    out['tail'].add(ins.target)
                    exit_use[i] = CALL_USE | regs.CALLEE_SAVED | {30}
            elif k in ('bl', 'blr'):
                out['calls'] += 1
                iscall[i] = True
                if k == 'bl':
                    out['bl'].add(ins.target)
                    use[i] = set(CALL_USE)
                else:
                    out['blr'] += 1
                    use[i] = set(CALL_USE) | rw(ins.target)[0]
                kill[i] = set(CALLER_SAVED) | {30}
            elif k == 'br':
                use[i] = rw(ins.target)[0]
                jt = tables.get(a)
                if not jt:
                    exit_use[i] = CALL_USE | regs.CALLEE_SAVED | {30}
            elif k == 'ret':
                exit_use[i] = RET_USE
        if insns and insns[-1].kind not in gen.TERMINATORS:
            exit_use[n] = CALL_USE | regs.CALLEE_SAVED | {30}   # falls through into the next function
            if end < mod.hi:
                out['tail'].add(end)
        # backward liveness to a fixed point
        live_in = [set() for _ in range(n + 1)]
        live_out = [set() for _ in range(n + 1)]
        pred = [[] for _ in range(n + 1)]
        for i in range(n + 1):
            for s in succ[i]:
                pred[s].append(i)
        work = list(range(n + 1))
        while work:
            i = work.pop()
            o = set(exit_use[i] or ())
            for s in succ[i]:
                o |= live_in[s]
            live_out[i] = o
            li = use[i] | (o - kill[i])
            if li != live_in[i]:
                live_in[i] = li
                work.extend(pred[i])
        bad = live_in[0] & TEMPS if n else set()
        if bad:
            out['entry'].append((entry, sorted(bad)))
        for i in range(n):
            if iscall[i]:
                b = live_out[i] & AFTER_BAD
                if b:
                    ins = insns[i]
                    out['after'].append((entry + 4 * i, ins.target if ins.kind == 'bl' else None, sorted(b)))
                    out['after_regs'].update(b)
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'cpu-study' / 'abi_scan.json')
    args = ap.parse_args()
    mods = layout(EXEFS, 'v101')
    table = patches.load(PATCHES, mods)
    paths, starts_abs, tasks, taken = {}, {}, [], {}
    for name, base, _ in mods:
        if name not in ('main', 'sdk'):
            continue
        mod = Module(name, EXEFS / name, base, 'c')
        starts, dstats = gen.discover(mod)
        abs_starts = mod.lo + starts * 4
        ends = np.append(abs_starts[1:], mod.hi)
        fns = list(zip(abs_starts.tolist(), ends.tolist()))
        for i in range(0, len(fns), 400):
            tasks.append((name, fns[i:i + 400]))
        paths[name] = (str(EXEFS / name), base)
        starts_abs[name] = abs_starts
        print(f'{name}: {len(fns)} functions; discovery stats {dstats}', flush=True)
    res = {'main': [], 'sdk': []}
    with mp.get_context('spawn').Pool(args.jobs, initializer=init, initargs=(paths, starts_abs, table)) as pool:
        for name, out in pool.imap_unordered(scan, tasks, chunksize=1):
            res[name].append(out)
    report = {}
    bl, tail = set(), set()
    for name, outs in res.items():
        for o in outs:
            bl |= o['bl']
            tail |= o['tail']
    for name, outs in res.items():
        entry = sorted(x for o in outs for x in o['entry'])
        after = sorted((a, t, r) for o in outs for a, t, r in o['after'])
        regs_after = Counter()
        for o in outs:
            regs_after.update(o['after_regs'])
        fns = sum(o['fns'] for o in outs)
        called = [e for e in entry if e[0] in bl]
        only_tail = [e for e in entry if e[0] not in bl and e[0] in tail]
        other = [e for e in entry if e[0] not in bl and e[0] not in tail]
        print(f'\n{name}: {fns} functions ({sum(o["plain"] for o in outs)} all-label), {sum(o["calls"] for o in outs)} calls '
              f'({sum(o["blr"] for o in outs)} indirect)')
        print(f'  read X9-X17 at entry: {len(entry)} functions; {len(called)} are bl targets, {len(only_tail)} only branch '
              f'targets, {len(other)} neither (address taken or seeds)')
        print(f'  read X2-X17 after a call: {len(after)} sites ({sum(1 for a in after if a[1] is None)} indirect); by register: '
              + ', '.join(f'x{r}: {c}' for r, c in sorted(regs_after.items())))
        report[name] = {'entry_called': called, 'entry_tail': only_tail, 'entry_other': other, 'after': after}
    args.out.write_text(json.dumps(report))
    print(f'\nreport: {args.out}')


if __name__ == '__main__':
    main()
