"""Logical processors with their efficiency class (GetSystemCpuSetInformation)."""
import ctypes
from ctypes import wintypes

k32 = ctypes.windll.kernel32
n = wintypes.ULONG(0)
k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
buf = (ctypes.c_ubyte * n.value)()
k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0)
off = 0
raw = bytes(buf)
while off < n.value:
    size = int.from_bytes(raw[off:off + 4], 'little')
    # SYSTEM_CPU_SET_INFORMATION: Size, Type, Id, Group(2), LogicalProcessorIndex(1), CoreIndex(1), LastLevelCacheIndex(1), NumaNodeIndex(1), EfficiencyClass(1)
    lp, core, llc, numa, eff = raw[off + 14], raw[off + 15], raw[off + 16], raw[off + 17], raw[off + 18]
    print(f'lp {lp:2d} core {core:2d} llc {llc} eff {eff}')
    off += size
