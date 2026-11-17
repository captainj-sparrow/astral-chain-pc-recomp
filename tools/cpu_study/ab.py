"""A/B summary of a AC_AB run: the game thread's busy time per frame (frame minus pacing sleep) in the A and B blocks.

usage: ab.py <log>... [--lo S] [--hi S]
Only perf windows (60 presents) wholly inside one block count. With AC_TRACE=sched and blocks at least twice a sched
window long, also each guest thread's run time per frame.
"""
import argparse
import math
import re
import statistics

PERF = re.compile(r'perf: frame (\d+): interval avg ([\d.]+) ms.*render busy avg ([\d.]+) ms; replay avg ([\d.]+) ms; ab (\d+)/(\d+); world (-?[\d.]+) s')
SCHED = re.compile(r'sched: frame (\d+): ([\d.]+) s, cores busy ([\d.]+) of 3')
THREAD = re.compile(r'sched: (render )?(0x[0-9a-f]+) (p\d+ m\w+) run (\d+)% .*? cpu (\d+)%')


def stats(v):
    m = statistics.fmean(v)
    se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--lo', type=float, default=42)
    ap.add_argument('--hi', type=float, default=1e9)
    a = ap.parse_args()
    for log in a.logs:
        side = {0: [], 1: []}
        iv = {0: [], 1: []}
        pure = {}   # frame at the window's end -> side
        lines = open(log, encoding='utf-8', errors='replace').read().splitlines()
        for line in lines:
            m = PERF.search(line)
            if not m:
                continue
            f, w, b, n = int(m.group(1)), float(m.group(7)), int(m.group(5)), int(m.group(6))
            if not a.lo <= w < a.hi or (b and b != n):
                continue
            s = 1 if b else 0
            side[s].append(float(m.group(3)))
            iv[s].append(float(m.group(2)))
            pure[f] = s
        if not side[0] or not side[1]:
            print(f'{log}: no whole A and B windows')
            continue
        (ma, sa), (mb, sb) = stats(side[0]), stats(side[1])
        d = 100 * (mb - ma) / ma
        e = 100 * math.hypot(sa, sb) / ma
        print(f'{log}: game thread busy  A {ma:.3f} ms ({len(side[0])} windows)  B {mb:.3f} ms ({len(side[1])})  B-A {d:+.2f}% +-{e:.2f}  '
              f'(interval A {statistics.fmean(iv[0]):.2f} B {statistics.fmean(iv[1]):.2f} ms)')
        # sched windows: pure when the perf windows ending at this frame and 60 frames before are on one side
        cur = None
        run = {0: {}, 1: {}}
        secs = {0: 0.0, 1: 0.0}
        for line in lines:
            m = SCHED.search(line)
            if m:
                f = int(m.group(1))
                s0, s1 = pure.get(f), pure.get(f - 60)
                cur = (s0, float(m.group(2))) if s0 is not None and s0 == s1 else None
                if cur:
                    secs[cur[0]] += cur[1]
                continue
            m = THREAD.search(line)
            if m and cur:
                name = ('render ' if m.group(1) else '') + m.group(2)
                r = run[cur[0]].setdefault(name, [0.0, 0.0])
                r[0] += int(m.group(4)) / 100 * cur[1]
                r[1] += int(m.group(5)) / 100 * cur[1]
        if secs[0] > 3 and secs[1] > 3:
            fa, fb = statistics.fmean(iv[0]), statistics.fmean(iv[1])
            names = sorted(run[0], key=lambda n: -run[0][n][1])
            tot = [0.0, 0.0, 0.0, 0.0]
            for n in names:
                ra, ca = (x / secs[0] * fa for x in run[0][n])
                rb, cb = (x / secs[1] * fb for x in run[1].get(n, [0, 0]))
                tot = [tot[0] + ra, tot[1] + rb, tot[2] + ca, tot[3] + cb]
                if ca >= 0.3:
                    print(f'    {n:16s} run A {ra:5.2f} B {rb:5.2f} ms/frame ({100 * (rb - ra) / ra if ra else 0:+5.1f}%)   cpu A {ca:5.2f} B {cb:5.2f} ({100 * (cb - ca) / ca:+5.1f}%)')
            print(f'    guest threads    run A {tot[0]:5.2f} B {tot[1]:5.2f} ({100 * (tot[1] - tot[0]) / tot[0]:+5.1f}%)   cpu A {tot[2]:5.2f} B {tot[3]:5.2f} ({100 * (tot[3] - tot[2]) / tot[2]:+5.1f}%)  [{secs[0]:.0f} s A, {secs[1]:.0f} s B]')


if __name__ == '__main__':
    main()
