"""Multithreaded test of the exclusive monitor helpers in cpu.h (ldxN/stxN, ldxp128/stxp128).

- Change: thread A loads X exclusively, thread B changes X->Y with its own exclusive pair, then A's store exclusive
  must fail. Checked for the 32-bit, 64-bit and 128-bit pair forms, plus a control where B touches nothing.
- Restored value (ABA): B changes X->Y->X. Hardware fails A's store; the compare-and-swap of cpu.h lets it succeed.
  Either result is right for the game and the line only says which one cpu.h gives: no exclusive window of the game
  reads memory or calls, so a store that succeeds there gives what hardware gives when A runs after B
  (tests/excl_windows.py checks this on the game's modules and explains it).
- Contention: N threads increment one counter with load/store-exclusive retry loops; the total must be exact.

Needs no game files. Usage: excl_mt.py [--runtime DIR] [--threads N] [--iters K]
"""
import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'diff'))
from common import MSYS_BIN, ROOT, compiler  # noqa: E402

SRC = r'''
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "cpu.h"

static volatile int step;
static uint64_t cell[4] __attribute__((aligned(64)));
static int nthreads, iters;
static int fails;

static void wait_step(int s) { while (__atomic_load_n(&step, __ATOMIC_ACQUIRE) != s) {} }
static void set_step(int s) { __atomic_store_n(&step, s, __ATOMIC_RELEASE); }

// Thread B: aba_steps exclusive pairs of the given width, each an XOR with one constant: 1 leaves another value
// (X -> Y), 2 puts X back (X -> Y -> X), 0 does nothing (the control).
static int aba_width, aba_steps;
static void *aba_b(void *arg) {
    (void)arg;
    Cpu c = {0};
    uint64_t a = (uint64_t)(uintptr_t)cell;
    wait_step(1);
    for (int k = 0; k < aba_steps; k++) {
        for (;;) {
            if (aba_width == 32) { uint32_t v = (uint32_t)ldx32(&c, a); if (!stx32(&c, a, v ^ 0x5a5a5a5au)) break; }
            else if (aba_width == 64) { uint64_t v = ldx64(&c, a); if (!stx64(&c, a, v ^ 0x5a5a5a5a5a5a5a5aull)) break; }
            else { __uint128_t v = ldxp128(&c, a); if (!stxp128(&c, a, (uint64_t)v ^ 1, (uint64_t)(v >> 64) ^ 2)) break; }
        }
    }
    set_step(2);
    return 0;
}

static void aba_case(int width, int steps) {
    pthread_t t;
    Cpu c = {0};
    uint64_t a = (uint64_t)(uintptr_t)cell;
    cell[0] = 0x1111222233334444ull;
    cell[1] = 0x5555666677778888ull;
    aba_width = width;
    aba_steps = steps;
    set_step(0);
    pthread_create(&t, 0, aba_b, 0);
    uint32_t st;
    if (width == 128) {
        __uint128_t v = ldxp128(&c, a);
        set_step(1);
        wait_step(2);
        st = stxp128(&c, a, (uint64_t)v + 7, (uint64_t)(v >> 64));
    } else if (width == 32) {
        uint32_t v = (uint32_t)ldx32(&c, a);
        set_step(1);
        wait_step(2);
        st = stx32(&c, a, v + 7);
    } else {
        uint64_t v = ldx64(&c, a);
        set_step(1);
        wait_step(2);
        st = stx64(&c, a, v + 7);
    }
    pthread_join(t, 0);
    if (steps == 2) {   // a restored value: both results are right (see the docstring)
        printf("restored value width %3d: stx status %u (%s)\n", width, st,
               st ? "seen, as on hardware" : "not seen: compare-and-swap");
        return;
    }
    uint32_t want = steps != 0;
    printf("%s width %3d: stx status %u, want %u %s\n", steps ? "changed value" : "control", width, st, want,
           st == want ? "ok" : "FAIL");
    if (st != want) fails++;
}

static int inc_width;
static void *inc(void *arg) {
    (void)arg;
    Cpu c = {0};
    uint64_t a = (uint64_t)(uintptr_t)cell;
    for (int i = 0; i < iters; i++) {
        for (;;) {
            if (inc_width == 32) { uint32_t v = (uint32_t)ldx32(&c, a); if (!stx32(&c, a, v + 1)) break; }
            else if (inc_width == 64) { uint64_t v = ldx64(&c, a); if (!stx64(&c, a, v + 1)) break; }
            else { __uint128_t v = ldxp128(&c, a); if (!stxp128(&c, a, (uint64_t)v + 1, (uint64_t)(v >> 64) + 3)) break; }
        }
    }
    return 0;
}

static void inc_case(int width) {
    pthread_t t[64];
    cell[0] = cell[1] = 0;
    inc_width = width;
    for (int i = 0; i < nthreads; i++) pthread_create(&t[i], 0, inc, 0);
    for (int i = 0; i < nthreads; i++) pthread_join(t[i], 0);
    uint64_t want = (uint64_t)nthreads * iters;
    uint64_t got = width == 32 ? (uint32_t)cell[0] : cell[0];
    int ok = got == (width == 32 ? (uint32_t)want : want) && (width != 128 || cell[1] == 3 * want);
    printf("contention width %3d: %d threads x %d, counter %llu (hi %llu), want %llu %s\n", width, nthreads, iters,
           (unsigned long long)got, (unsigned long long)cell[1], (unsigned long long)want, ok ? "ok" : "FAIL");
    if (!ok) fails++;
}

int main(int argc, char **argv) {
    nthreads = atoi(argv[1]);
    iters = atoi(argv[2]);
    aba_case(64, 0);
    aba_case(32, 1);
    aba_case(64, 1);
    aba_case(128, 1);
    aba_case(32, 2);
    aba_case(64, 2);
    aba_case(128, 2);
    inc_case(32);
    inc_case(64);
    inc_case(128);
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails != 0;
}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', type=Path, default=ROOT / 'runtime', help='directory holding cpu.h')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--iters', type=int, default=200000)
    args = ap.parse_args()
    env = dict(os.environ)
    if os.name == 'nt' and MSYS_BIN.exists():
        env['PATH'] = str(MSYS_BIN) + os.pathsep + env.get('PATH', '')
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'excl_mt.c'
        exe = Path(tmp) / ('excl_mt.exe' if os.name == 'nt' else 'excl_mt')
        src.write_text(SRC)
        arch = ['-march=x86-64-v3'] if platform.machine().lower() in ('amd64', 'x86_64') else []
        subprocess.check_call([compiler(), '-O2', *arch, '-fno-strict-aliasing', '-I', str(args.runtime),
                               str(src), '-o', str(exe), '-lpthread'], env=env)
        return subprocess.run([str(exe), str(args.threads), str(args.iters)], env=env).returncode


if __name__ == '__main__':
    sys.exit(main())
