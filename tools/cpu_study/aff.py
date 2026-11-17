"""Run a command and, <delay> seconds after its start, restrict it to the logical processors of <mask> (hex).

usage: aff.py <mask hex> <delay s> <command...>
"""
import ctypes
import subprocess
import sys
import threading

mask, delay, cmd = int(sys.argv[1], 16), float(sys.argv[2]), sys.argv[3:]
p = subprocess.Popen(cmd)


def pin():
    if p.poll() is None:
        ok = ctypes.windll.kernel32.SetProcessAffinityMask(int(p._handle), ctypes.c_size_t(mask))
        print(f'aff.py: affinity {mask:#x} after {delay} s: {"set" if ok else "FAILED"}', file=sys.stderr, flush=True)


t = threading.Timer(delay, pin)
t.daemon = True
t.start()
sys.exit(p.wait())
