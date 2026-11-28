# macOS port work

Status: boots. Native Apple Silicon, the existing Vulkan renderer through MoltenVK,
game version 1.0.1. On 2026-11-28 a 90-second run on an M6 Mac mini reached the
game's start screen ("ZL + ZR"), rendered correctly: 5,377 frames at a mean of
58-60 fps, 12 million NVN calls, none unimplemented, no shader failures and no
validation errors. Gameplay past the start screen is not checked yet.

Build and run (Apple clang from the Xcode tools; Homebrew: ninja, sdl3, vulkan-loader,
molten-vk, opus, libvpx, zstd; the shader translator as below):

```sh
.venv/bin/python tools/build_game.py            # build/v101/astralchain
cd build/v101 && ./astralchain --seconds 90 --shot ../shot.png
```

`tools/build_game.py` fetches VMA and AMD's FidelityFX files from GitHub on first
use. Where IPv6 to GitHub stalls, Python waits a long time on each address; fetch
over IPv4 (for example `curl -4`) or resolve IPv4 only for that run.

## Checkpoints

1. Query the actual GPU and test ARM64 instruction generation on synthetic inputs.
2. Add a Darwin platform backend, ARM64 floating-point/SIMD support, and native build
   and shader-translator targets. Verify memory, synchronization and fiber semantics.
3. Render synthetic scenes in an SDL window through MoltenVK. Validate shader
   translation, memory sharing and synchronization before booting the game.
4. Extract verified user inputs, lift the game, and reach character creation with
   graphics, audio and input.
5. Validate the motorcycle chase and early on-foot combat in a 30-minute session,
   including save/reload. Initial performance target: approximately 30 fps at
   720p/900p. This is an acceptance target, not a measured result.
6. Provide a repeatable build/launch workflow and document known limitations.

## GPU capability probe

`tools/macos_vk_probe.c` queries the required extensions and features currently
checked by `runtime/vk_device.c`. It needs no game files. Keep its requirement list
in sync with that runtime file. It checks capabilities only: it does not create a
logical device, open a window, exercise host-memory imports or validate rendering.

With Homebrew Vulkan headers/loader and MoltenVK installed:

```sh
mkdir -p build/macos-probes
clang -std=c11 -Wall -Wextra -Werror tools/macos_vk_probe.c \
  $(pkg-config --cflags --libs vulkan) -o build/macos-probes/vulkan
VK_DRIVER_FILES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json" \
  build/macos-probes/vulkan
```

Exit 0 means at least one device reports all current requirements, 1 means no
device meets them, and 2 means the probe failed. GPU access is required; a sandbox
that blocks Metal can fail at instance creation before any capabilities are read.

Observed on 2026-11-28: M6 Mac mini, 24 GB RAM, macOS 27.2.1, MoltenVK 1.4.2:

- Device reports Vulkan 1.3.357.
- All four required extensions are present, including external host memory.
- All queried required features are present except `geometryShader`.
- Probe returns 1, as intended for that missing feature.

Do not simply delete the geometry-shader requirement. Audit the translated shaders
and pipeline uses, then implement a supported path for any stages the game needs.
The presence of external host memory also does not establish alignment, coherence
or dirty-tracking correctness on this Mac.

## CPU and operating-system gaps

The shader translator now builds for `osx-arm64`. With Homebrew's .NET 10 SDK:

```sh
export PATH="$(brew --prefix dotnet@10)/bin:$PATH"
export DOTNET_ROOT="$(brew --prefix dotnet@10)/libexec"
# Homebrew's SDK links against its OpenSSL and Brotli libraries.
export LIBRARY_PATH="$(brew --prefix openssl@4)/lib:$(brew --prefix)/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
.venv/bin/python tools/build_xlate.py
.venv/bin/python tools/xlate_smoke.py
```

Verified locally: native ARM64 output, valid Vulkan 1.3 SPIR-V with and without
alpha discard, and service recovery after malformed input. This local Homebrew
build depends on installed libraries built for macOS 27; it is not a portable
release for earlier macOS versions. If restore stalls on unreachable IPv6 routes,
the process-local `DOTNET_SYSTEM_NET_DISABLEIPV6=1` setting allows an IPv4 retry.

CPU baseline on the M6 (Apple clang 21, 2026-11-28):

- `python tests/flags_check.py`: 2,000,484 operand pairs at each of `-O0` and
  `-O1`, zero mismatches.
- `python tests/diff/atomic.py --jobs 4`: all 198 instruction tests / 12,672
  cases pass against Unicorn. The harness uses Apple's linker on Darwin.
- `python tests/diff/simd.py --jobs 4`: 1,048,640 encoding-validity checks agree
  with Unicorn; 159,976 numerical cases across 20,000 encodings have no mismatches.
- `python tests/diff/lift.py --jobs 4`: 119,982 cases across 40,000 instructions
  sampled from the verified 1.0.1 modules have no mismatches.
- `python tests/diff/func.py --jobs 4`: 3,477 runnable cases across 537 functions
  have no mismatches; 5,799 attempted cases were skipped by the test harness.
- With `--abi-trust --ipa generated/v101/ipa.json`, the same function test passes
  3,316 runnable cases across 505 functions (5,072 skipped), covering the default
  build's register-analysis optimization.
- `tests/excl_mt.py` passes the changed-value and contention checks;
  `tests/excl_windows.py` finds no unsafe exclusive windows in either game module.
- `tests/jumptables_check.py` reports zero disagreements and zero unverified bases.
- `tests/vlocals_check.py` checks 175,592 synchronization points and 179 patch
  helper calls in the full generated output, with zero violations.

The existing C lowering now uses ARM64 FPCR control and NEON table lookups on
Apple Silicon, retaining the x86 implementation on x86 hosts. The 1.0.1 generator
emitted all 61,454 discovered functions with no unsupported instructions. All 123
generated chunks and both dispatch tables compile to Mach-O ARM64 objects at
`-O1`. This does not yet link a game executable or verify scheduling or gameplay.

Runtime sources for arm64 macOS (`clang -target arm64-apple-macos14.4`, 2026-11-28):
all C files compile but `vk_ffxfg.c` and `vk_fsr3.c`, which need the SPIR-V headers
the build generates. Host-function tables use Mach-O sections (`__DATA,ac_ovr`,
bounded by ld64's `section$start$`/`section$end$` symbols; checked across object
files, with `-dead_strip`, and with no entries). `ov_libc.c` runs its SSE2 paths on
the nine AdvSIMD equivalents it needs (`movemask` checked against SSE2 on 200,000
random vectors).

Asynchronous preemption (`kern_ap_stub`) has an ARM64 version. An ARM64 detour
cannot jump back without a free register (compiled code may keep values in x16,
x17 and x30; Darwin reserves x18), so the stub restores everything and branches to
`plat_intr_continue`, whose `brk` the platform's SIGTRAP handler serves: it sets pc
to the address stored under the red zone, and `sigreturn` restores every register,
the flags and the FP state. `tests/macos_platform_test.c` interrupts an assembly
loop that checks x16, x17, x30, v31 and the flags every iteration: about 6,900
detours in 0.3 s with none corrupted; a stub that clobbers x16 fails at once.

`tools/build_game.py` builds on macOS (Homebrew's clang-compatible Apple clang,
ld64, Ninja, SDL3, Vulkan loader and MoltenVK, Opus, libvpx, zstd).

## Address space on Apple Silicon

Guest addresses equal host addresses and must stay below 2^39 (IPC buffer
descriptors encode 39 bits). On Apple Silicon a process gets little of that range.
XNU (`osfmk/vm/vm_map.c`, `vm_map_range_configure`; `osfmk/mach/arm/vm_param.h`)
fixes the layout:

- shared region from 6 GiB (on this Mac it reaches about 63 GiB);
- commpage at 63-64 GiB;
- GPU carve-out at 64-448 GiB (`MACH_VM_MIN/MAX_GPU_CARVEOUT_ADDRESS`), unusable:
  a fixed allocation there fails with `KERN_NO_SPACE`, even after deallocating;
- before `main` runs, libmalloc reserves about 24 GiB, plus a 20 MiB region, at a
  random place in the 64 GiB left at 448-512 GiB.

A Mach-O zero-fill segment at a fixed address cannot claim the space earlier, as
Wine does on x86: arm64 executables must be PIE, so every segment slides with the
image, and exec kills a process whose image spans the shared region.

The module window (from `0x70_0000_0000`, with RomFS at its fixed address in the
game), the kernel region and the host-function window are at 448-460 GiB already.
The stack and heap block (6 GiB) and the 64 GiB alias region do not fit below the
modules. So memory.c then reserves the block and an alias region right after it in
one free hole anywhere below 2^39, shrinking the alias region in 4 GiB steps down
to 8 GiB, as MacNativeGames does for TOTK on macOS 27. The game maps about 3.3 GiB
there. A process that finds no place re-executes itself (`reexec_or_fail`) and gets
libmalloc's reservation elsewhere. The ASLR region then starts at the modules, so
it still holds all guest memory. Windows and Linux keep their layout, since the
search below the modules succeeds there.

`tests/macos_memory_test.c` runs memory.c on the platform layer without the game:
layout, QueryMemory around the module window, kernel region and alias region, the
heap, physical memory, a freeze thawed by a write and a hold fetched by a read.
Observed on the M6, 2026-11-28: every run found a layout, with the stack at
448-500 GiB and an alias region of 8-16 GiB (44 GiB once, when libmalloc's
reservation came after the free space), in about 0.3 s; 26 checks pass. Whether
Astral Chain itself accepts an alias region below 64 GiB is still to be verified
with the game.

## Darwin platform backend

`runtime/platform/macos.c` (scheduling), `macos_mem.c` (memory and faults) and
`macos_posix.c` (files and processes) implement `platform.h` for Apple Silicon,
macOS 14.4 or newer (`os_sync_wait_on_address`). What differs from Linux:

- Locks and condition variables are linux.c's one-word futex design on
  `os_sync_wait_on_address`: zero-initialised, no allocation, nothing to destroy.
- Priorities are QoS classes, which also choose performance or efficiency cores.
  Another thread can only be raised, with a QoS override that ends when the thread
  next sets its own priority (vk_pipeline.c raises a worker a waiter depends on).
- Thread interruption suspends the thread through Mach; no handler runs on the
  game's fiber stack.
- Guest pages are 4 KiB and host pages 16 KiB. A host page takes the most
  permissive state of its 4 KiB pages, so nothing that should run faults; the
  4 KiB state is kept in software for `plat_mem_state`. `plat_mem_grain` tells the
  runtime the host page size.
- There is no GetWriteWatch or userfaultfd. A committed, clean, read-write host
  page of a watched reservation is kept read-only; its first write faults, copies
  the page into a snapshot slot and opens it. A query with reset arms it again,
  then compares each 4 KiB page with the snapshot, so it reports exactly the 4 KiB
  pages whose bytes changed, as Linux and Windows report written pages (the Vulkan
  back end's mirror sync copies reported pages over GPU results, so a neighbour
  reported by mistake would not be harmless). Every protection change of a host
  page takes its lock bit, so a query's re-arm and a fault's disarm cannot
  interleave. A kernel write into an armed page fails instead of faulting, so
  `plat_file_pread` opens its destination first.
- `plat_mem_replace` moves the staging pages in with `mach_vm_remap`
  (`VM_FLAGS_OVERWRITE`), atomic like linux.c's `mremap`.
- Protection faults arrive as SIGBUS, and read, write or fetch comes from the
  exception syndrome (ESR) in the thread state.
- Data that must reach the disk uses `F_FULLFSYNC`; children get only the
  descriptors named in their file actions (`POSIX_SPAWN_CLOEXEC_DEFAULT`).

`tests/macos_platform_test.c` checks the semantics the runtime relies on, not just
the calls: fibers moving between host threads with their FP state, interruption,
4 KiB states inside a host page, read and write faults reaching the right hook at
the faulting byte, decommitted pages reading zero beside kept neighbours, replace,
and the write watch. Its stress part has four threads writing a watched range
while queries with reset run; a write that completed between two queries must be
in the second one's list. Build and run it as its header says.

Where the runtime needs a fault, it works in whole host pages, which changes
nothing where the grain is 4 KiB:

- memory.c refuses a hold that is not whole host pages; vk_memory.c holds the
  aligned middle of a range and copies the partial host pages at its ends back, as
  it does with any range it cannot hold.
- romfs.c's read-fault paging fills a whole host page. `ROMFS_BASE` is only 4 KiB
  aligned, so the image's first host page is shared with main's data and becomes
  accessible once main is committed: its RomFS pages are filled at start.
- libc_probe.c puts its inaccessible guard after a host page.

Observed on the M6, 2026-11-28: 124 checks pass; the stress part ran about 110,000
faulting writes against 6,000 queries with none missed. Removing the host-page
lock from the fault handler makes it report about 10 million missed writes, so the
stress part does detect that race.

## References

These are research sources; their reported results have not been reproduced here.
Pin source revisions and preserve per-file licenses and notices before adapting code.

- [MacNativeGames Astral Chain](https://github.com/ly05010419/MacNativeGames/tree/main/games/alc):
  1.0.1, native ARM64 with Metal. Its game-specific README and changelog report
  several chapters playable, including input, audio and saves. Its top-level status
  summary is older. Useful for service behavior, Apple memory handling and renderer
  workarounds; its execution architecture differs from this static C recompilation.
- [AC-Recomp](https://github.com/swaymun/AC-Recomp): 1.0.0, native ARM64 and MoltenVK.
  Reports the opening and motorcycle chase working, with incomplete gameplay and
  untested saves. Useful for MoltenVK-specific fixes; its patches are for another
  runtime and cannot be applied directly here.
- [LASRecomp](https://github.com/peiikai/LASRecomp): a source ancestor with an Apple
  Silicon/Metal target, useful for instruction lowering and platform design.
- [MoltenVK runtime guide](https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md):
  portability enumeration, extension support and deployment details.

## Runtime probes on the M6 (2026-11-28)

Pass: `--selftest`, `--probe-kernel`, `--probe-loader`, `--probe-libc` (40,017
checks), `--probe-fs`, `--probe-ipc`, `--probe-services`, `--probe-dlc`,
`--probe-nvn`, `--probe-nvn-tex`, `--probe-vkdevice`, `--probe-vkmem` (write watch exact,
file reads into watched pages seen), `--probe-window` (300
frames, vsync 60.0 Hz), `--boot`.

Not passing, and why:

- `--probe-astc`: its BC encode checks need the ASTC-to-BC pipelines, which the
  runtime builds only for a GPU without native ASTC or short of video memory; the
  M6 samples ASTC natively.
- `--probe-shader`, `--probe-vkexec`: need `build/v101/shaders/jobs.json` from
  `tools/shader_census.py`.
