# Building from source

Windows 10 or later, an x86-64 CPU with SSE4.2, and a GPU with Vulkan 1.3 and `VK_EXT_external_memory_host`. The
default build targets x86-64-v3 (AVX2/FMA); a CPU without them builds with `--cpu v2` instead and runs slower (see
"Build"). Allow about 60 GB for the work folder. Extraction wants about 55 GB free while it runs and keeps about 35 GB
afterwards.

Players who only want to play can use the [installer](installer.md) instead: it builds the game on their machine
without this toolchain.

## Toolchain

Versions are the ones the build was verified with; newer patch releases are fine.

| Tool | Version | Install | Used for |
|---|---|---|---|
| MSYS2 UCRT64 | clang/lld 23.1.3 | msys2.org installer to `C:\msys64`, packages below | compiling the generated C and the runtime |
| Visual Studio Build Tools | 2026, C++ workload, Windows SDK 10.0.26100 | winget `Microsoft.VisualStudio.BuildTools` with `--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended` | the linker .NET NativeAOT calls |
| .NET SDK | 10.0 | winget `Microsoft.DotNet.SDK.10` | the shader translator (`ac_xlate.exe`) |
| Vulkan SDK | 1.4.363 | winget `KhronosGroup.VulkanSDK` | validation layers, `vulkaninfo` |
| uv | 0.13 | standalone installer from astral.sh | the two Python environments |
| hactool | 1.4.0 @ `1d64a83` | `.venv\Scripts\python tools\build_hactool.py`, see below | NCA and update extraction |

MSYS2 packages, after `pacman -Syuu` (run it twice; the first pass updates the core and exits):

```sh
pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-{clang,lld,ninja,gcc,make,pkgconf,sdl3,opus,libvpx,zstd,vulkan-devel,\
vulkan-memory-allocator,vulkan-validation-layers,shaderc,spirv-tools} base-devel git unzip
```

`ffmpeg` from MSYS2 is a GPLv3 build and is never linked; the movies go through libvpx.

PATH:

- Add `C:\Program Files (x86)\Microsoft Visual Studio\Installer` to the user PATH: NativeAOT needs `vswhere.exe`, or
  the publish fails at the link step (MSB3073).
- Another `cmake` or `ninja` earlier on PATH (a 32-bit winlibs MinGW, for example) is picked up by accident. The build
  scripts call Ninja from `C:\msys64\ucrt64\bin` themselves; from a shell, put `C:\msys64\ucrt64\bin` first.

### hactool

`tools/build_hactool.py` fetches SciresM/hactool at `1d64a83` into `toolchain\src\hactool`, applies the three patches in
`tools/hactool/` and builds a static `toolchain\hactool\hactool.exe` with MSYS2 UCRT64 gcc (a few seconds):

- `portable.patch`: the parts of MacNativeGames' `build_hactool()` that run on Windows, the `MNG_TITLE_KEYS` variable and
  the two BKTR fixes in `nca.c` for the subsection table padding and the subsection layout of newer cumulative updates;
- `unicode.patch`: a manifest with `<activeCodePage>UTF-8</activeCodePage>`, so paths under a non-ASCII user profile
  reach it intact;
- `mbedtls-asm-clobbers.patch`: the `memory` and `cc` clobbers the bundled mbedtls 2.6.1's x86-64 multiply asm lacks.
  Without them a clang build (the Linux release) fails every RSA signature check and the update merge stops.

hactool reads `-k <prod.keys>`; MSYS2 shells set `HOME`, which hides its default key location, so pass `-k` there.

## Build

From the repository root, in PowerShell. Every step that depends on the game version takes it as `<ver>`: `v101` for
1.0.1. The tools default to `v101` when `--version` is left out. Saves load the same way.

1. Python environments (once). `.venv` runs the lifter, the build and the extraction scripts:

   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv/Scripts/python.exe -r tools/requirements.txt
   ```

2. Copy `config.example.toml` to `config.local.toml` and set the paths to your base game, the 1.0.1 update and your
   console's keys file. `config.local.toml` is ignored by git. The tools read the keys file themselves.

3. Extract and verify. This writes `build\extracted\v0` (the base ExeFS, checked as 1.0.0) and `build\extracted\v101`
   (the update's ExeFS and the RomFS merged through hactool), identified by its hash; a hash mismatch stops the run.
   The merge needs about 21 GiB free next to `build\extracted\v101` (the two Program NCAs and the merged image). With
   another `[work] dir` in `config.local.toml`, `extracted` goes there instead, and `build_game.py` reads the ExeFS and
   RomFS from there:

   ```powershell
   .venv\Scripts\python tools\extract.py
   ```

4. Shader translator (no game files; again after a change to its sources, and `tools\make_release.py` stops on an older
   one). This publishes `build\xlate\ac_xlate.exe`:

   ```powershell
   .venv\Scripts\python tools\build_xlate.py
   ```

5. Lift the modules to C, then compile and link `build\v101\astralchain.exe`:

   ```powershell
   .venv\Scripts\python recomp\gen.py --exefs build\extracted\v101\exefs --out generated\v101 --patches recomp\patches-v101.txt --version v101
   .venv\Scripts\python tools\build_game.py --version v101
   ```

   `--cpu v2` targets x86-64-v2 (SSE4.2, no AVX2/FMA) for older CPUs; the default `v3` needs AVX2. The v2 build links
   `runtime/fma.c`, a correctly rounded software fma the FP lowering needs, and runs slower.

   Lifting takes about 15 s, a clean compile about 200 s on 24 threads; later builds recompile only what changed.
   A compiler cache in front of clang shortens it further: with `SCCACHE` set to an sccache executable, `build_game.py`
   writes it into the ninja file ahead of `cc`, and every lift rewrites the generated chunks with the same contents, so
   a second build of the same version hits all of their objects:

   ```powershell
   $env:SCCACHE = "path\to\sccache.exe"
   .venv\Scripts\python tools\build_game.py --version v101
   ```
   `--patches` builds in the frame-rate, resolution and other patch sets, switchable at run time. The link copies
   `SDL3.dll`, `libiconv-2.dll` and `libopus-0.dll` next to the exe.
   For Xbox and PlayStation button prompts (the Controls tab's *Button prompts* row), build the button mods' overlays
   from the emulator mods with `tools\make_prompts_folders.py` (defaults read `controller-ui\`):
   they write the two `romfs-patches-*` folders, which `build_game.py` copies next to `astralchain.exe` like
   `romfs-patches`. The default `prompts = auto` loads the mod of the controller in use at the first run (no
   restart needed); the Controls tab shows and changes it. Each mod is drawn for one gamepad layout (a loaded mod
   forces it: Xbox for the Xbox mod, PlayStation for the PlayStation mod).

6. Prepare the shaders (once per build folder and translator). It reads the variants players met from
   `generated\shader-answers.bin` (recorded in play; `tools\prepare_shaders.py --recordings` turns a recording into it)
   and the shaders themselves from the extracted RomFS:

   ```powershell
   .venv-py314\Scripts\python tools\prepare_shaders.py build\v101 --version v101
   ```

   Without this step the first launch compiles shaders and pipelines during play and stutters until the caches fill.
   The AMD driver keeps its pipeline cache per exe path and holds a limited amount: preparing another build folder can
   evict this one's. `astralchain.exe --warm-shaders` rebuilds it from the shader cache.
   What the preparation leaves is the driver's own disk cache, and the driver decides where it lives: Mesa's drivers
   (RADV on the Steam Deck, under Proton as well) use the folder the launcher's environment names
   (`MESA_SHADER_CACHE_DIR`, which Steam can set per library entry), so the installer's run and a later start from
   another Steam entry can use different caches, and a driver update starts an empty one. `shader-prepared-driver.txt`
   next to the exe lists the caches a preparation filled. A start whose cache is not listed, or is listed but builds a
   sample of the prepared variants slowly (evicted or cleared), builds them into it in the background at idle priority
   (`vk_pipeline: driver shader cache:` in the log says which case applies, and the boot prebuild's report its timings).
   When the driver's shader compiler crashes on a variant (NVIDIA 615.78.08 on an RTX 5080 does, issue #30),
   `prepare_shaders.py` runs the driver step again with `--warm-shaders`, up to three times, without the variants the
   driver was building at the crash. `shader-prepare-skip.txt` next to the exe lists them with the GPU and driver
   version; they build at their first draw instead, another driver version builds them again, and so does deleting the
   file.

7. Run. The game keeps running when its window loses the focus (`assume_focus = 1` in `astralchain.ini`, the default; the
   Game tab switches it). With 0 it pauses, as it does on the Switch when Home is pressed.

   ```powershell
   build\<ver>\astralchain.exe
   ```

Settings live in `astralchain.ini` next to `astralchain.exe` ([astralchain.ini.example](../astralchain.ini.example) lists them). Saves go to
`build\profile-v101` (`build_game.py --profile` changes it).

### Portable folder

`.venv\Scripts\python tools\package_build.py --version <ver>` writes `build\<ver>-portable`, which runs on another of
your machines without MSYS2, Python or .NET. `romfs.bin` stays out unless you pass `--with-romfs`; otherwise put it in
`romfs\` next to the exe or point `AC_ROMFS` at its folder. Run `astralchain.exe --prepare-shaders` once in the copied
folder. The folder holds code and data derived from your game: for your own machines only.

## Linux (in progress)

The runtime builds and runs on x86-64 Linux with a window, input and menu on SDL3 (`*_sdl.c`) in place of the Win32
files (`*_win.c`); `runtime/platform/linux.c` implements `platform.h`. Status: the selftest and the probes pass; 1.2.1
plays at 60 fps on RADV (RX 7800 XT) and NVIDIA (RTX 3060); 1.4.3 builds, probes and plays on RADV (RX 6600, and the
Vega 7 of a Ryzen 5 4600G at about 34 fps). Long play sessions and the Steam Deck itself are not verified yet.

Requirements:

- Linux 6.7 or later: the write watch uses userfaultfd write protection with `PAGEMAP_SCAN`. SteamOS 3.8 qualifies.
- A Vulkan 1.3 driver with `VK_EXT_external_memory_host` (RADV, NVIDIA).
- Packages on Arch:

  ```sh
  pacman -S --needed clang lld ninja make sdl3 opus libvpx zstd vulkan-headers vulkan-icd-loader \
      vulkan-validation-layers shaderc spirv-tools dotnet-sdk uv git
  ```

  Vulkan Memory Allocator is a single header (AUR `vulkan-memory-allocator`). Without it, `build_game.py` fetches the
  version the Windows release pins (3.4.0) into `toolchain/vma/` on first use and checks its hash.

The steps are the Windows ones with POSIX paths: `.venv/bin/python` in place of `.venv\Scripts\python`, and `totk` in
place of `astralchain.exe`. `build_game.py` takes clang, lld and Ninja from PATH (or the ones in `<dir>/bin` with
`--toolchain <dir>`) and links against the system libraries, with `libstdc++` static and an `$ORIGIN` rpath;
`build_hactool.py` builds `hactool` with make. `tools/prepare_shaders.py` reads the recorded shader variants from
`generated/shader-answers.bin`.

The probes open a window, so from an SSH session they need the desktop session's display, or they stop at `gpu 0
skipped: no graphics+compute+present queue family`: on Wayland `XDG_RUNTIME_DIR=/run/user/$(id -u)
WAYLAND_DISPLAY=wayland-1 SDL_VIDEODRIVER=wayland` (the socket name is in `$XDG_RUNTIME_DIR`), on X11 `DISPLAY=:0`.

Memory: on Windows the GPU imports guest memory and `GetWriteWatch` tracks the CPU's writes to it. On Linux,
re-protecting imported memory invalidates the driver's userptr buffer on every submit (about 1.3 ms per 16 MiB on
amdgpu), and importing pins the pages, which drops their write protection. Each CPU-visible pool therefore gets a
host-visible copy (shadow storage, `runtime/vk_memory.c`): the pages the CPU wrote go in before each submit, and the GPU's
writes come back at its fence. In play that costs about 0.2 ms a frame and doubles the memory of those pools (about
1.1 GB). `AC_SHADOW=1` turns it on under Windows too. The Steam Deck's RADV has `VK_EXT_external_memory_host` as
well: the shadow copies are chosen for the measured cost above, not for a missing extension.

The Deck's GPU (Van Gogh) has no native ASTC, so its textures go through the compute decoder (`runtime/vk_astc.c`,
`AC_ASTC`) as on desktop AMD GPUs without it; frame times measured there include it.

CPU frequency: Linux picks a core's frequency from how busy that one core has been and wakes a thread on whichever
core is idle. The game's threads hand work to each other thousands of times a second and sleep in between, so every
core looks lightly loaded and stays slow: on a Core Ultra 285H (16 cores, `intel_pstate`) they ran at 1.3-1.8 GHz on
cores that hold 5 GHz, a third of the time on the low-power cores, and a scene Windows shows at 60 fps ran at 21. The
logs of a Steam Deck and of a GTX 1080 Ti in issue #8 show the same picture (the render thread busy for the whole
frame, the GPU waiting for the next submission most of it). `cpu_boost` (`AC_CPU_BOOST`, `plat_cpu_boost` in
`runtime/platform/linux.c`) answers with three things that need no privilege:

- the performance power profile, held over D-Bus while the game runs (`HoldProfile` of power-profiles-daemon and of
  tuned's `tuned-ppd`, what `powerprofilesctl launch` does): the daemon switches the governor or the energy preference
  to performance and switches back when the game exits or crashes. libsystemd's bus client is loaded at run time;
  without it or without the daemon the two below stand alone;
- a utilization clamp at the maximum for the game's threads (`sched_setattr`, Linux 5.3): the `schedutil` governor
  then runs them at the top frequency, and on processors with cores of different speeds the scheduler tries the fast
  ones first;
- where the processor still chooses the frequency itself (`intel_pstate` and `amd-pstate-epp` under their `powersave`
  governor with an energy preference other than `performance`) or the governor samples each core's idle time
  (`ondemand`, `conservative`), a thread about to wait polls for up to 1 ms first, in proportion to the work it just
  did, so the core stays awake across the gap.

`auto`, the default, asks for the profile, sets the clamp, and polls only under those governors (and under
`schedutil` on a kernel without the clamp), looking at the governor again every few seconds; `1` polls whatever the
governor, `0` turns all three off. Measured on the 285H, in play, same spot:

| | fps | 1% low | frames over 25 ms in 26 s | CPU time | cores at |
|---|---|---|---|---|---|
| `cpu_boost = 0` | 21-25 | 15-18 | all | 1.0 cores | 1.1-1.8 GHz |
| clamp and polling (no profile daemon) | 60 | 33-42 | 2-144 | 1.7 cores | 4.0-4.6 GHz |
| performance profile held | 60 | 41-47 | 4 | 1.5 cores | 4.2-4.7 GHz |

Limited to 4 of its fast cores the game holds 60 with the clamp and polling as well. Only that machine is measured.

The Steam Deck profile turns `cpu_boost` off: its CPU and GPU share one 15 W limit, and a 1.6.0 session in the
headquarters (the heaviest area) showed 25 fps with the CPU drawing 9.1 W and the GPU 2.3 W while 83% busy. There the
GPU runs out of power, and running the CPU faster would take more of it away. `cpu_boost = auto` in `astralchain.ini`
turns it back on for a comparison. The log says at start what was arranged (`cpu:` lines, the second one once the daemon has
answered) and `AC_TRACE=perf` adds, every 60 frames, the frequency of the cores under the render and replay threads
(`cpu frequency under the render thread`), which is what to compare between `cpu_boost = 0` and the default on
another machine.

Frame times in the performance overlay: its graph shows the time between the game's present calls, not between the
pictures on the display. The present thread sends each picture at its slot (16.67 ms apart, 0.01 ms of deviation in
the menus), while the game's calls alternate around that in the menus that play a video: its render thread decodes a
video frame in software before every other submission (5 ms at 4 GHz, 13 ms at 2 GHz). The graph therefore shows a
sawtooth whose height follows the CPU frequency: 6/27 ms at 2 GHz, 11/22 ms with the performance profile.

The dirty census (`mem: dirty` lines, `tools/perf_summary.py`) is the trace item `dirty`, not part of `perf`: its
write-watch queries read every page table entry of the buffer-bound pools at each submit, 15 ms a frame on the Deck.

## Checks

Before a commit:

- `python tools/check_content.py --staged`: nothing game-derived is staged. CI runs the same check;
  `git config core.hooksPath .githooks` runs it before each local commit.
- `build\v101\astralchain.exe --selftest`: layout, dispatch table and host-function registry, without game files.

A change to the button prompts (`tools/make_prompts_folders.py`, `runtime/romfs.c` overlay_init) passes
`astralchain.exe --probe-cfg`, which checks the prompts pick and folders (input_map.c); a change to the settings
passes it too.

The lifter (`recomp/`) and the CPU headers (`runtime/cpu.h`, `runtime/simd.h`) are checked against Unicorn. Every lifter
change passes these, all run with `.venv`:

| Test | Checks | Game files |
|---|---|---|
| `tests\diff\lift.py` | instructions sampled from the extracted modules; `--coverage` lists every word the lifter rejects or traps | yes |
| `tests\diff\simd.py` | the whole FP/SIMD encoding space | no |
| `tests\diff\atomic.py` | exclusive load/store sequences and ARMv8.1 atomics | no |
| `tests\diff\func.py` | whole functions as the generator emits them, loops, jump tables and calls included (calls go to a stub that checks the registers at the call) | yes |
| `tests\flags_check.py` | the function-local flag helpers of `runtime\cpu.h` against its packed ones | no |
| `tests\excl_mt.py` | the exclusive monitor helpers under contention and ABA, on several threads | no |
| `tests\jumptables_check.py` | `recomp/jumptables.py` against a Unicorn oracle; must report 0 disagreements, on 1.4.3 too | yes |

A `--cpu v2` build passes the same differential tests with `--cflags=-march=x86-64-v2`; they compile `runtime/fma.c`
too, so they check the software fma the v2 build links rather than the C runtime's.

Runtime probes, each a mode of `astralchain.exe` (the list is at the top of `runtime/main.c`): `--probe-kernel`,
`--probe-vkexec`, `--probe-nvn-tex` and `--probe-shader` cover most runtime changes. `--probe-shader` and the draw
part of `--probe-vkexec` translate shaders of the census below: they look for `build/<ver>/shaders/jobs.json` of the
exe's own version first, then of the other one. `--probe-xess` runs Intel XeSS once on synthetic images (it passes,
saying so, without `libxess.dll` or on a GPU XeSS cannot use); a change to `runtime/vk_dlaa.c` or `runtime/vk_xess.c`
also wants a scripted run into play with `AC_ANTI_ALIASING=xess`, whose log counts the frames anti-aliased and the history resets.
A change to the per-object motion code (`runtime/vk_mvprev.c`, the "per-object motion" section of
`runtime/vk_exec_draw.c`) wants the same run with `AC_ANTI_ALIASING=xess` to show the `mv_prev:` and `per-object motion:` log lines
with draws drawn again; `AC_DLAA_MVDEBUG=1` with `--shot` shows the vectors (x red, y green) in place of the picture.
`.venv\Scripts\python tools\xess_sdk.py` puts the pinned SDK's `libxess.dll` in `toolchain\xess\`, which
`build_game.py` then copies next to `astralchain.exe`. AMD FSR 3 (`AC_ANTI_ALIASING=fsr3`) needs no step of its own: when it compiles
`runtime/`, `build_game.py` calls `tools/ffx_sdk.py`, which fetches the pinned FidelityFX SDK files once into
`toolchain\ffx-sdk\` and compiles the passes with `glslc` into `fsr3_spv.h` in the build directory, again only when
the pins or the options change. A change to `runtime/vk_fsr3.c` wants the same scripted run with `AC_ANTI_ALIASING=fsr3`.

Shaders: `.venv-py314\Scripts\python tools\shader_census.py --version <ver> --translate` extracts every Maxwell shader
to `build\<ver>\shaders\` and runs it through the translator and `spirv-val`; failure groups go to `failures.txt`.
`tools\xlate_smoke.py` checks the translator alone, without game files.

## Reverse engineering (optional)

- Ghidra 12.1.2 with Adubbz's Ghidra-Switch-Loader in `toolchain\ghidra\` (JDK 21) opens the game modules (NSO).
- `tools\fetch_refs.sh` clones the upstream projects this port builds on into `refs\`, read-only.
