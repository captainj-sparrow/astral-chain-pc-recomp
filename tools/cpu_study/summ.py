"""Per-frame CPU summary of a run log (AC_TRACE=perf,sched) over the world-clock window [lo, hi) seconds.

usage: summ.py <log> [lo] [hi]
Per guest thread: the time it ran guest code holding a core (run) and its OS CPU time (cpu), in ms per frame.
"""
import re
import sys
from collections import defaultdict

PERF = re.compile(r'perf: frame (\d+): interval avg ([\d.]+) ms.*render busy avg ([\d.]+) ms; replay avg ([\d.]+) ms;.*world (-?[\d.]+) s')
SCHED = re.compile(r'sched: frame (\d+): ([\d.]+) s, cores busy ([\d.]+) of 3')
THREAD = re.compile(r'sched: (render )?(0x[0-9a-f]+) (p\d+ m\w+) run (\d+)% core (\d+)% host (\d+)% sleep (\d+)% sync (\d+)% cv (\d+)% lock (\d+)% addr (\d+)% cpu (\d+)%')
SPIN = re.compile(r'sched:   (0x[0-9a-f]+) spin wait passes (\d+), core handoffs (\d+)')


def main():
    log = sys.argv[1]
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    hi = float(sys.argv[3]) if len(sys.argv) > 3 else 78
    world = {}   # frame -> world s
    perf = []
    lines = open(log, encoding='utf-8', errors='replace').read().splitlines()
    for line in lines:
        m = PERF.search(line)
        if m:
            f, iv, busy, rep, w = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            world[f] = w
            if lo <= w < hi:
                perf.append((iv, busy, rep))
    frames = sorted(world)

    def world_at(f):
        best = None
        for x in frames:
            if x <= f:
                best = x
        return world[best] if best is not None else -1

    run, cpu, spin, secs, nwin, busy_cores = defaultdict(float), defaultdict(float), defaultdict(float), 0.0, 0, 0.0
    cur = None
    for line in lines:
        m = SCHED.search(line)
        if m:
            w = world_at(int(m.group(1)))
            cur = float(m.group(2)) if lo <= w < hi else None
            if cur:
                secs += cur
                nwin += 1
                busy_cores += float(m.group(3)) * cur
            continue
        if cur is None:
            continue
        m = THREAD.search(line)
        if m:
            name = ('render ' if m.group(1) else '') + m.group(2) + ' ' + m.group(3)
            run[name] += int(m.group(4)) / 100 * cur
            cpu[name] += int(m.group(12)) / 100 * cur
            continue
        m = SPIN.search(line)
        if m:
            spin[m.group(1)] += int(m.group(2))
    if not perf or not secs:
        print('no windows in range')
        return
    iv = sum(p[0] for p in perf) / len(perf)
    print(f'{log}: {len(perf)} perf windows, interval {iv:.2f} ms, render busy {sum(p[1] for p in perf) / len(perf):.2f} ms, '
          f'replay {sum(p[2] for p in perf) / len(perf):.2f} ms; {nwin} sched windows ({secs:.0f} s), cores busy {busy_cores / secs:.2f} of 3')
    tot_run = tot_cpu = 0
    for name in sorted(run, key=lambda n: -cpu[n]):
        r, c = run[name] / secs * iv, cpu[name] / secs * iv
        tot_run += r
        tot_cpu += c
        if c >= 0.05 or r >= 0.05:
            sp = spin.get(name.replace('render ', '').split()[0], 0) / secs
            print(f'  {name:22s} run {r:5.2f} ms/frame  cpu {c:5.2f} ms/frame' + (f'  spin passes {sp:9.0f}/s' if sp else ''))
    print(f'  guest threads total: run {tot_run:.2f} ms/frame, cpu {tot_cpu:.2f} ms/frame')


if __name__ == '__main__':
    main()
