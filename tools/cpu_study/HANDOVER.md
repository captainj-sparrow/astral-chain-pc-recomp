# Handover: CPU levers for low-end machines (study of 2026-11-17)

Goal: at least 5% less CPU time per frame on the weakest machines the port targets (Steam Deck: Zen 2, 4 cores / 8
threads, 4 MiB L3, always CPU-bound), PGO excluded. A previous session established that most of the CPU time is the
game's lifted code, the port's own overhead being limited, so this study works on the generated code and on how guest
threads wait.

Status: three levers prototyped behind switches that are **off by default**; nothing is enabled. The default lift is
byte-identical to main's (checked over all 125 generated files). `--abi-trust` is **not safe yet** (section 4).

## 1. Bench

Everything runs on the dev machine (Core Ultra 9 285H, Arc 140T), never on Zen 2, so it uses proxies for a weak CPU:

- **E-cores**: the game pinned to the 8 Skymont cores (logical processors 2-9, mask `3FC`).
- **Small cache**: the game on 4 E-cores (mask `3C`) while `thrash.exe 2 96 3 <seconds>` walks 2 x 96 MiB on logical
  processors 0-1, evicting the shared L3. This alone lengthens the game thread's frame by 56% (7.7 to 12.0 ms): the
  game is memory-bound, which matches what the user sees on the Deck.
- GPU settings low (1080p output, no upscaling, no AO, shadow x1), a 1920x1080 window, the Harmony Square recording
  `build/barriers/rec.txt` plus slow right-stick turns from 38 s (`rec-cam.txt`), 60 fps cap. The metric is the game
  thread's busy time per frame (frame minus pacing sleep), the `render busy` of the `perf:` lines.

Run-to-run noise is about +-5% (core frequency, background load), larger than the effects. So variants are compared
**inside one run**: `mkworlds.py <gen A> <gen B> <out>` merges two lifts into one generated directory (world A keeps
`f_<addr>`, world B becomes `g_<addr>`, every direct call selects with `rt_world`), and the study runtime (this folder's
`rt-ab.patch` on a copy of `runtime/`) switches the world, and the other study knobs, with the A/B blocks of `AC_AB`.
Both worlds pay the same select per call. `ab.py <log>` gives A, B, B - A with its standard error, and per guest thread
the run and CPU time.

```sh
cp -r runtime build/cpu-study/rt-ab && patch -d build/cpu-study/rt-ab -p1 < tools/cpu_study/rt-ab.patch
.venv/Scripts/python recomp/gen.py --exefs build/extracted/v101/exefs --out generated/cs-abiv \
    --patches recomp/patches-v101.txt --version v101 --abi-trust --v-abi-trust
.venv/Scripts/python tools/cpu_study/mkworlds.py generated/v101 generated/cs-abiv generated/cs-w-abiv
.venv/Scripts/python tools/build_game.py --version v101 --gen generated/cs-w-abiv --out build/cpu-study/b-w-abiv \
    --runtime build/cpu-study/rt-ab
# copy build/v101's folder (shaders, caches, ini) to build/cpu-study/v101, the exe in as ac-wabiv.exe, then:
AFF=3FC AFFDELAY=2 sh tools/cpu_study/run.sh ac-wabiv.exe wabiv-e1 125 200 AC_AB=2 AC_STUDY_WORLD=ab
.venv/Scripts/python tools/cpu_study/ab.py build/cpu-study/logs/wabiv-e1.log
```

Study runtime knobs (`rt-ab.patch`): `AC_STUDY_WORLD=ab` (lifted code world), `AC_STUDY_SPIN=ab` (applies
`AC_SPIN_SLEEP_US` in B only), `AC_STUDY_LIBC=ab` (host memcmp/memset in B only), `AC_STUDY_GUEST_AFF=<hex>` (guest
threads' affinity). `-DSTUDY_FP_NOCHECK` in a world's chunks (4th argument of `mkworlds.py`) drops the FP result checks
of `runtime/simd_fp.h` there, as a ceiling only.

Other tools: `aff.py` (pins a process after a delay; `run.sh` uses it through `AFF` / `AFFDELAY`), `cpusets.py` (cores
and efficiency classes), `summ.py` (per-thread time over a world-clock window), `chains.py` (callers / callees from an
`AC_PROFILE_RENDER=1` profile), `expand.py` and `mix.py` (x86 instructions per ARM instruction and instruction mix of
hot functions), `abi_scan.py` (section 4), `ldxp_scan.py` (section 5). `run.sh` expects `build/barriers/prof-start`
(a copy of the user's profile, outside the repository).

## 2. What a frame is made of

Main-thread profile in play (Harmony Square): 94-96% of its running samples are lifted code, 4-5% runtime. No single
hot spot: the top function is the sdk's lifted `memcmp` (4-7%, called on each uniform block the game fills, see
main+0x9d6510), then a long tail of 1-2% functions. Hot FP functions expand to 7-14 x86 instructions per ARM
instruction statically (cold slow paths included), integer ones to 2-3; about 1.4 memory operands per ARM instruction
in the hot set, many of them `c->x[n]` traffic the C compiler cannot remove because guest stores may alias the `Cpu`.
A priority 59 guest thread spins on a lock word (main+0x3e3e74) a million passes a second, about 2 ms a frame.

## 3. Results (game thread busy per frame, B against main's lift, in-run A/B)

| Lever | E-cores | Small cache | P-cores |
|---|---|---|---|
| `gen.py --abi-trust` | -4.5% +-1.3 | -4.6% +-1.8 | -2.6% +-1.2 |
| `--abi-trust --v-abi-trust` | -5.5% +-2.4 | -4.1% +-1.8 | |
| `AC_SPIN_SLEEP_US=200` | -1.7% +-1.8; spinning thread's CPU -40%, guest CPU total -8% | | |
| all three plus host memcmp/memset | -3.2% +-2.9; guest run time -6.3% | **-6.5% +-3.4**; guest run time -11.4% | |

`--abi-trust` also makes `.text` 8% smaller (48.5 to 44.7 MiB) and takes memory operands per ARM instruction of the hot
set from 1.39 to 1.24. On the Deck the CPU time the spin sleep frees matters beyond the frame: CPU and GPU share 15 W.

Dead ends, measured:

- PRFM lifted as host prefetches: 0 (69 sites in main).
- Host `memcmp` / `memset` in place of the sdk's: about -0.6%. The lifted `memcmp`'s time is cache misses on the
  compared blocks, not its byte loop.
- FP result checks removed (ceiling, inexact): -3.3%. Not worth the correctness.
- `--abi-trust` without storing X19-X28 at returns (unsafe): 0 more. Passing arguments in host registers is therefore
  not expected to pay either.
- Guest threads kept off the runtime threads' cache: 0. 2 LP-E cores: unplayable in both worlds.
- `minsize` through a pragma: code larger (inline helpers stop being inlined); not run.

## 4. The levers and what they still need

### `gen.py --abi-trust` (recomp/regs.py `plan_trust`, recomp/gen.py)

Every general register a function uses (X18 excepted) is a C local, and calls and returns sync only what AAPCS64 passes:
before a call the changed X0-X8, X29, X30 and SP are stored; after it only X0 and X1 are loaded again; before a return
the changed X0, X1, X19-X29 and SP. Every other sync point (SVC, trap, `@fn` patch helper, tail jump) stores everything
changed, and SVCs and helpers reload everything. Today's default keeps 6 registers in locals and reloads X0-X18 and X30
after every call.

**Not safe as is.** `abi_scan.py` (a liveness pass over every lifted function) finds code that does not follow
AAPCS64:

- 27 functions read X9-X17 at entry: 2 are `bl` targets in main (0x700075b380 reads x15, 0x70009abc40 x9/x10), 10
  are address-taken or seeds, the rest are reached only by branches (safe: a tail jump stores everything).
- 63 call sites read X2-X17 after a call before writing them: x8 (33), x2 (22, sdk 0x726779c... helpers), x9-x11.

Under `--abi-trust` those read a stale value. Fix in the generator, from global information it already has at
discovery: (1) per function, the registers it reads at entry; store them too before a call to it; (2) per call site,
the caller-saved registers live after the call; reload them there. Indirect calls (`blr`) need the union over possible
targets, or the default sync. Then extend `tests/diff/func.py` (and `tests/vlocals_check.py`) to the modes and run the
whole differential battery of `docs/building.md` "Checks". About 40 scripted runs with the mode on showed no crash and
the same pictures; that proves nothing.

### `gen.py --v-abi-trust` (recomp/vlocals.py `rewrite_one` with `abi`)

With `--v-locals`: every V register a function names is a local; a guest call stores the written V0-V7 and reloads
V0-V3; a return stores the written V0-V3 and V8-V15. Host helpers and other sync points keep the full sync. Same
audit needed (V16-V31 or V4-V7 read after a call, V8-V31 read at entry), plus the upper halves of V8-V15, which
AAPCS64 does not preserve.

### `AC_SPIN_SLEEP_US` (runtime/kernel.c `rt_spin_gate`)

After n us of a spin wait, each pass leaves the virtual core and sleeps an eighth of the wait so far (at most 250 us).
Before a default: the A/V subset (`tools/av_sync_test.py`, short), since it changes how long waits last; a measurement
under Linux, where `plat_wait_ns` is `nanosleep` (finer than Windows' timer, so possibly more wakeups).

## 5. Not studied yet

- Zen 2 itself: run the A/B exe on the Ally at 14 W (`tools/deck_power.sh`) or a Deck. The study runtime builds on
  Windows only as written (`thrash.c`, `aff.py`); on Linux use `taskset` and `stress-ng --cache`.
- Transparent huge pages for the guest heap on Linux (the game is memory-bound, TLB misses not measured; the
  write-watched pools must stay on 4 KiB pages).
- The LDXP/STXP read idiom: 984 of the 1284 pairs in main store the value they loaded back unchanged at once (an
  atomic 128-bit read), each one a `lock cmpxchg16b`; the functions holding them take 2.6-3.6% of the guest samples.
  A plain 16-byte atomic load (`vmovdqa`, atomic on AVX CPUs) would do, if the exclusive monitor's state allows it.
- Uniform block compare (main+0x9d6510): the game compares each filled block with the previous one; the cost is the
  cache misses on the old block, which a host-side cache of the last block's hash could avoid.
