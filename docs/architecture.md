# Architecture

How the Switch build of *Tears of the Kingdom* becomes a native Windows x86-64 program. The upstream projects it builds on are listed under Credits in the [README](../README.md).

## Starting points

Two MIT projects cover different halves of the problem:

| | LAS Recomp | MacNativeGames |
|---|---|---|
| Game | *Link's Awakening* | *Tears of the Kingdom* 1.0.0–1.4.3 |
| CPU | ARM64 lifted to C, runs on x86-64 | Original ARM64 code runs directly on Apple silicon |
| OS boundary | IPC services and SVCs | ~163 `nn::*` SDK functions replaced by symbol, 38 SVCs |
| Graphics | Own NVN layer on Vulkan, own Maxwell translator | NVN front end + Metal 4 back end, Ryujinx translator |
| Useful to us | Lifter, NEON-to-C lowering, Windows memory layer, tests | Everything Astral Chain-specific: host behaviour, NVN front end, version tables, patches, shader list |

MacNativeGames cannot run on x86-64 as is: its CPU path executes ARM64 code natively. LAS Recomp's host only covers what *Link's Awakening* calls. The port takes the CPU half from LAS Recomp and the game half from MacNativeGames, and writes the parts neither has: a lifter that scales to Astral Chain, and a Vulkan back end for the NVN front end.

## Pipeline

### 1. Input

- Base 1.0.0 (`01007300020FA000`) and its 1.0.1 update (`01007300020FA800`), NSP, plus the keys of the user's console.
- ExeFS and RomFS extraction: MacNativeGames' Python extractor (`.venv`; the port's tools decode zstd through the `zstandard` package).
- Update merging: hactool at `1d64a83` with MacNativeGames' BKTR patches, built by `tools/build_hactool.py`.
- Verification: our `tools/data/expected-v*.json` hashes of the ExeFS and the merged RomFS.
- Primary version: 1.0.1, the only update. It replaces `main`; `sdk` and `rtld` are byte-identical to 1.0.0. It builds with `--version v101`, `recomp/patches-v101.txt`; the fixed RomFS address differs from 1.0.0 and is chosen by `recomp/layout.py`.
- Windows driver: `tools/extract.py` runs the vendored extractors (`tools/vendor/macnativegames/`) and identifies the update by the SHA-256 of its `main`. hactool has no Darwin `funopen` here, so the base Program NCA is copied out of the NSP before the BKTR merge (about 11 GB of temporary space).

### 2. CPU: lift ARM64 to C

- Modules: `main`, `sdk`, `subsdk0`. `rtld` is dropped and linking is done by the host, as MacNativeGames does.
- Base: LAS Recomp's hand-written decoder (`lifter.py`) and NEON/FP lowering (`simd_c.py`, `simd.h`), MIT. Output: one C function per guest function, direct calls for `BL`, an address-keyed table for indirect branches.
- Rewritten for Astral Chain scale:
  - generator runs in parallel processes, with bounded memory per worker;
  - jump tables resolved by our own symbolic pass instead of Unicorn, which keeps GPL code out of the shipped generator;
  - `LDXP`/`STXP` (128-bit exclusive pairs) supported;
  - no 64 MB limit on `main`; `subsdk*` loaded;
  - NZCV and the six most used general registers of each function kept in C locals instead of the `Cpu`
    (`recomp/flags.py`, `recomp/regs.py`): registers are stored back where they changed, before calls, SVCs and
    exits, and reloaded after them. A call reloads only X0-X18 and X30, as AAPCS64 has the callee preserve X19-X29 and
    SP; `gen.py --no-callee-saved-trust` reloads those too.
- Coverage oracle: MacNativeGames' NSO rewriter lists every site where game code touches the OS (SVCs, TLS reads, counter reads, `x18`, exclusive windows, jump tables): 68,968 sites in 1.4.3 `main`. The lifter must handle every one.
- Memory: guest address equals host address, all below 2^39 (IPC encodes 39-bit addresses). The heap, stacks and the alias region are reserved with `MEM_WRITE_WATCH`, so the graphics back end can ask which pages the CPU wrote since the last upload (`runtime/memory.c`).
- Indirect branches go through `rt_dispatch` (`runtime/dispatch.c`), which looks the target up in the address table and tail-calls it.
- Testing: LAS Recomp's differential harness, every lifted instruction class checked against Unicorn (test-only, never linked into the game).

### 3. Host

- Portable C from MacNativeGames: threads and synchronisation, IPC, memory services, fs/RomFS, time, account, applet, audio renderer and effects. The Apple pieces (mach VM, CommonCrypto, AudioToolbox, GameController) are replaced with Win32 and SDL3.
- SDK boundary at the `nn::*` function level, matching MacNativeGames. Only `audren:u` and `sm:` are real IPC services.
- Window, keyboard and mouse: Win32 (`gfx_window.c`, `input_win.c`). Gamepad and audio output: SDL3. Voice: libopus (BSD). Movies: the SDK's `movie::Decoder` is replaced by a host VP9 decoder over libvpx (BSD, static; `runtime/ov_movie.c`), because FFmpeg from MSYS2 is a GPL build.
- zstd: most RomFS files of 1.4.3 are zstd frames. The patch set `zstd` sends the game's two decoder entry points to
  libzstd on the host (`ov_zstd.c`, `AC_ZSTD_NATIVE`); the lifted decoder took almost three times as long.
- Settings and menu: every key with its default is in one table (`cfg.c`), read from `astralchain.ini` and `AC_*`
  variables and written back when the in-game menu changes it. The menu and the performance overlay are Dear ImGui,
  drawn into the swapchain image (`runtime/ui/`); their text is in `ui_text.c` (English, Russian, Brazilian Portuguese,
  Simplified Chinese). The Chinese glyphs come from a system font (Microsoft YaHei, `plat_font_path`) merged into the
  font atlas.
- OS calls the host makes go through `runtime/platform/` (`platform.h`, `win32.c`), so a Linux or macOS host adds one file instead of touching the rest. The split is not finished: `tools/check_platform.py` lists the Win32 uses left in the other files.

#### Threads and cores

Every guest thread is a host thread. Astral Chain assumes Horizon's scheduler: three application cores, a thread runs only on a core in its affinity mask, and a higher-priority thread takes a core from a lower-priority one. The game keeps per-core data without locks and relies on that, so the host models it (`runtime/kernel.c`):

- Three virtual cores. A guest thread runs guest code only while it holds a core; the others wait in a priority-ordered list.
- Preemption. When a waiting thread has a better priority than a holder on a core it may use, exactly one holder is flagged. The holder checks the flag at its next SVC and at every indirect call (`rt_dispatch`: two loads when nobody waits), checks again under the kernel lock that a better thread still waits, and only then hands its core over. The indirect-call check comes from SavageWindRecomp's BotW port; checking only at SVCs left a better thread waiting up to 26 ms.
- Asynchronous preemption (from MacNativeGames 8bf0893, on unless `AC_ASYNC_PREEMPT=0`). Game code can run for
  milliseconds with neither (`AC_CORE_DIAG=1` measures it): a waiter that has spun 250 us suspends the holder and,
  if its host rip is in lifted code at host depth 0 (not inside a host call or SVC), moves it to `kern_ap_stub`, which
  saves the volatile registers and gives the core up there. A holder of a guest mutex counts at the priority of its
  best waiter, and one inside an SDK spin lock (TLR+0x100 `disable_count`) is not preempted: its interrupt flag is set
  and the switch happens at SVC 0x36. `AC_ASYNC_PREEMPT=2` keeps those two without interrupting. The interruption is
  Win32 (SuspendThread, SetThreadContext); another host needs a signal handler there, as MacNativeGames has.
- Host work gives the core back. Queue submits, presents, GPU waits, shader translation, `nn::fs` calls and NVN memory pool setup run without a core (`NVN_BLOCKING`, `kern_core_away`), as the Switch blocks those threads in IPC, so a thread blocked in the driver or on a file does not hold up guest threads.
- Spin waits. Loops that only another thread can end (857 in 1.4.3, 864 in 1.2.1, found by `tools/spin_waits.py`, the
  nxnative search) send their back-edges through `rt_spin_gate`, which pauses the core once per pass and hands it to a
  waiter after 50 us of polling (`AC_SPIN`, `AC_SPIN_HANDOFF_US`).
- `free_cores = 1` (`AC_FREE_CORES=1`) turns the gate off, for experiments only: the game crashes or hangs, usually
  at start, because its job workers all report core 0 and share per-core memory (#212).

### 4. Graphics

- NVN front end from MacNativeGames (`nvn_api_*.c`, `nvn_obj.h`, texture layout and deswizzle). The game records command buffers through it; nothing reaches the GPU until the game submits.
- Vulkan 1.3 back end in place of MacNativeGames' Metal one. At `QueueSubmitCommands` the recorded NVN commands are replayed into Vulkan command buffers (`vk_exec*.c`).
- Memory pools. Apple's unified memory let MacNativeGames use the game's pools zero-copy. The back end imports the pools as host memory (`VK_EXT_external_memory_host`). On a discrete GPU, pools bound as buffers are mirrored in device-local memory and the pages the CPU wrote (write watch, above) are uploaded before a replay; idle pool regions are made read-only so the write-watch queries skip them (`vk_memory.c`). GPU-rendered textures are written back when the CPU reads them: on a pool invalidate, on a CPU copy, and for small targets once the CPU waits for a fence (`vk_texture.c`).
- Replay thread. `AC_REPLAY_ASYNC=1` (the default; `=0` turns it off) moves the replay off the submitting thread onto a queue at most one frame behind the game; reads of GPU results wait for the newest writer instead of draining the queue (`vk_backend.c`). It is on because an A/B run on the dev machine with a save load and gameplay showed identical throughput; the waits were under 160 ms over 3,472 frames.
- Recorder thread. The Vulkan calls the replay makes into its command buffer are made by a thread of their own (`vk_rec.c`, after Savage Wind Recomp and LAS Recomp): each `vkCmd*` copies its arguments into a packet of a ring and the recorder makes the calls in order; calls into other command buffers are made at once; the recorder also ends the command buffer, and the submit waits until it has. In the town on an RX 9070 the replay takes 17% less time on the submitting thread. It is off by default on AMD GPUs, where the driver loses the device or hangs the GPU at startup with it (`AC_VK_RECORDER=1` forces it on, `=0` makes every call on the submitting thread).
- Uniform snapshot. The game rewrites per-object uniform blocks on other threads soon after it submits the frame
  that reads them, while the replay still records it. At the start of a submission's replay every uniform range it
  binds from CPU-written pools is copied, and the draws bind the copy (`vk_usnap.c`, after MacNativeGames'
  `AC_UBO_SNAPSHOT`); without it, late draws read another object's matrix.
- Driver data. The NVN driver keeps a per-stage constant bank (c0) that shaders read for things like point size, dispatch size and base instance. The back end fills it the way the driver does.
- Shaders: the Ryujinx translator 1.3.2 with MacNativeGames' patches and ours (`tools/xlate/patches/`), built with .NET 10 NativeAOT as `ac_xlate.exe` and run as a separate process. Its SPIR-V goes to Vulkan directly. `tools/prepare_shaders.py` and `astralchain.exe --prepare-shaders` translate and compile the variants of MacNativeGames' shared list ahead of play (29,587 for 1.4.3); the rest are translated on first use and cached on disk.
- Pipelines: `VK_EXT_graphics_pipeline_library` where the driver has it. Shader stages are compiled ahead into libraries; a draw that meets a new state links them at once (fast link), and an optimised pipeline replaces the linked one when a background thread finishes it. Pipeline keys seen in play are saved and built at boot. A draw whose pipeline is not ready is skipped when one of its pass's targets was rendered in the previous frame, and waits otherwise, so one-shot passes are never lost; a full-screen draw is skipped as well rather than waiting (`AC_PIPE_SKIP`); the stage libraries a waiting draw needs are built on idle workers in parallel (`AC_PIPE_WAIT`). Without the extension, a missing pipeline is built on a worker and the draw waits or is skipped by the same rule.
- Resolution. Patch sets make the game read its output, swapchain and scene sizes from the patch page
  (MacNativeGames' sites); `resolution.c` sets them from `output_res` and `render_scale` and grows the texture heap
  and the application pool to match. Astral Chain 1.0.1 has no output set: there the `res` set does the same at the
  sites that hold the game's render size (`resolution.c` `res_pick`, `rt_res_*`), reading `output_res` from the patch
  page: any 16:9 size of 32n x 18n from 1280x720 to 3840x2160 (1920x1080 by default; `res_size`). 1280x800 lost the
  device, and crashed in handheld, where a 1280-wide output makes the ladder take the docked entries for the 1280x720
  targets; a size off the grid falls back to 1920x1080. The Steam Deck profile (`cfg.c` `DECK`) therefore defaults to
  1280x720, which the present fits into the 1280x800 screen with 40-row bars (`gfx_fit_rect`); the selection happens
  at start, in one exe for desktop and Deck. Decks that rendered 1080p before keep no stored 1080p:
  `cfg_migrate_deck_res` removes, once, an ini `output_res` that renders 1920x1080 and marks the file with
  `deck_res_migrated = 1`. The game's render targets and window
  textures come from one budget the graphics configuration sets (main+0x40e0b0: 310 MiB, of which 1080p uses 304),
  carved from the process heap; above 1080p's pixel count `rt_res_budget_w8` raises it with the pixels (`res_budget`:
  the measured 87 MiB and 110 bytes a pixel, 15 % to spare) and `res_heap_extra_bytes` grows the application pool by
  the same amount (TotK's HeapDef growth does not apply there). 1.0.1 has no `aspect` set and no HUD layouts for other
  shapes yet. The
  game always renders its picture at its own size: nothing enlarges its render targets from outside (the former
  `ssaa`). When the window is larger than the output, `vk_fsr.c` runs AMD FSR 1 (EASU, RCAS) on the present. An
  `output_res` that is not 16:9 (more than 1% off, `res_is_16x9`) also switches on the
  `aspect` patch set, which hands the output's ratio to the camera, and `romfs.c` `overlay_init` loads the
  `romfs-patches-<w>x<h>` overlay for it when it exists (`tools/make_romfs_patches.py --aspect`, laid out by
  `tools/ultrawide_layout.py`; the installer builds ten sizes: 21:9 2560x1080, 3440x1440, 3840x1600, 5120x2160, 32:9
  3840x1080, 5120x1440 and 16:10 1280x800, 1680x1050, 1920x1200, 2560x1600). A wider layout moves the HUD to the
  screen's corners, a narrower one to its top and bottom edges. Other sizes get the camera of their shape and the 16:9
  HUD stretched over it. The `aspect` setting (`fit`, the default; `stretch`) decides what the present does
  when the window's shape differs from the output's: `fit` letterboxes, where a 16:9 game in a wider window used to be
  stretched.
- HDR (`hdr = 1`, `vk_hdr.c`). The game tonemaps its scene (Uchimura's curve, or Hable's, then a 16³ colour LUT) into
  an R11G11B10F target at the output's size, sRGB-encoded; its SMAA reads that target and writes the 8-bit window
  texture it presents, and the HUD and menus draw into the window texture afterwards. With HDR on, the tonemap's
  fragment shader (recognised by its constants and resource names, not a hash) runs as `shaders/hdr_tonemap.frag`,
  which grades in SDR as the game does and, on an HDR10 swapchain, writes the picture with its highlights (up to the
  peak over the game's white) into the float target; `hdr_hue` (0 to 100, default 80, from TotK Recomp) sets how far
  highlights keep the SDR picture's whitened hue rather than the extended curve's saturated one. The present gets that scene beside the window texture; a compute
  pass takes each window pixel that still equals the clamped scene (or its SMAA blend with one neighbour) as the scene,
  at the game's white and with its highlights from the float target, and every other pixel (the HUD, menus, the
  game's dimming behind a pop-up) at the HUD's white, then turns the result into PQ code values in Rec. 2020, which
  FSR 1 or the blit then scale. A frame without the scene (movies, loading screens) is shown at the HUD's white. The
  in-game menu is drawn into an RGBA8 image and blended on at the HUD's white. Without an HDR10 surface format
  (Windows with "Use HDR" off, a compositor without colour management) the swapchain stays SDR and the tonemap writes
  the game's own result. `AC_SHOTS` reads the window texture, so its pictures clip highlights in HDR.
- Temporal anti-aliasing at the scene's own size (`vk_dlaa.c`, off by default): one setting, `anti_aliasing`, keeps the
  game's SMAA (`smaa`) or puts NVIDIA DLAA (`dlaa`, through the driver's NGX core loaded at run time, Windows only), Intel
  XeSS Native AA (`xess`, `vk_xess.c`, `libxess.dll` loaded at run time, Windows only) or AMD FSR 3 Native AA (`fsr3`,
  `vk_fsr3.c`, every OS) in its place; no SDK header or code of NVIDIA's
  or Intel's is compiled in. FSR 3's passes are AMD's GLSL, unmodified and not in the repository: `tools/ffx_sdk.py`
  fetches the pinned files of the FidelityFX SDK (commit and per-file SHA-256) into `toolchain/ffx-sdk/` and compiles
  them into `fsr3_spv.h` in the build directory, and `vk_fsr3.c` is our own host for them (internal images, constant
  blocks, the seven passes in AMD's order). Astral Chain renders forward: a depth-only prepass writes the scene
  depth (D32F_S8), the opaque draws and the transparents go into an R11G11B10F light target with it (two images,
  alternating per frame), the bloom chain reads it at half size and down, a composite tonemaps, SMAA anti-aliases and
  the HUD is drawn over the result. The AA arms on the first light-target pass after a prepass and runs, in place on
  the light target, at the first pass a quarter as wide (the bloom's third level), so the bloom, the composite and the
  HUD read the resolved picture; the game's SMAA edge and weight passes are skipped meanwhile. The scene's draws get a
  sub-pixel Halton viewport jitter. Motion vectors start from the camera: the camera block is the scene draws' vertex c3
  (256 bytes: view, projection, view-projection, inverse view), and one reprojection matrix (last frame's
  view-projection times the inverse of this frame's, in double precision on the CPU) takes every depth pixel to last
  frame's position. Over them comes a per-object pass (`vk_exec_draw.c`, "per-object motion"; `AC_DLAA_OBJMV=0` turns
  it off): the prepass's solid draws whose vertex position reads an object's uniform blocks besides c3 are drawn again
  at the AA point into the motion image, with a last-frame variant of their vertex shader (translator patch
  `0012-motion-previous-position.patch` recomputes the position from the previous blocks) and a small motion fragment
  shader, so enemies and anything else that moves on its own get their own vectors; objects whose blocks did not
  change and instanced draws (scenery whose instance list changes from frame to frame) keep the camera's. Astral Chain hands those blocks out at new addresses every frame, so `vk_mvprev.c` copies
  each candidate's blocks per frame into a host-visible arena and finds the object again next frame by its draw
  (vertex program, index buffer, first vertex, counts) and the translation of its first matrix. `AC_DLAA_MVDEBUG=1`
  shows the vectors in place of the anti-aliased picture. Camera cuts, a missed AA point for 6 presents, another scene
  size or another light target start a new history. DLSS is created with model K (the transformer) and given the
  measured frame time. After 600 presents without DLSS its features are released, and without any temporal AA the work
  images too, without waiting for the GPU; at exit NGX is shut down before the device goes. The performance overlay's
  Scaling line names the AA while it resolves frames.
- Occlusion queries never block the CPU; a result that is not ready counts as visible.
- ASTC textures are uploaded as they are where the device reports `textureCompressionASTC_LDR`, and decoded by a
  compute shader everywhere else (`vk_astc.c`, `AC_ASTC`). A discrete GPU with 6 GB of video memory or more keeps
  the decoded textures as RGBA8, their full quality (`AC_ASTC_BC=auto`, the default, resolves to `0`), and the
  texture cache's default bound grows to half the device budget for them. Below 6 GB and on integrated GPUs
  (`bc1`) they are re-encoded on the GPU: the alpha scan of the raw storage (`tex_astc.c`, sound, the swizzle only
  permutes 16-byte sectors) picks BC1 for the provably opaque textures — the same colour quality as BC3 at half its
  size — and BC3 for the rest. Both lose against RGBA8 (blockier gradients, less precise normal maps), which shows
  most on the environments' large tiled surfaces. `1` forces BC3 everywhere.
  --probe-astc holds both encoders and the scan to PSNR and exactness bounds. The CPU decoder (`tex_astc*.c`) is the fallback and the test oracle.
- Draw state. A draw whose program, bound state and textures repeat reuses the previous draw's shaders and skips rebuilding its descriptor sets; descriptor sets are written through update templates (`vk_exec_draw.c`, `vk_pipeline.c`).
- Allocation: Vulkan Memory Allocator. Debugging: Vulkan validation layers, per-pass frame dumps (`AC_DUMP`, `vk_dump.c`) and a flight recorder (`AC_FLIGHT`). RenderDoc cannot capture, because it hides `VK_EXT_external_memory_host` from the application.

Maxwell behaviour the back end has to match, and the Astral Chain bugs other ports fixed: [maxwell.md](maxwell.md).

Why Vulkan rather than D3D12 or wgpu:

- The translator emits SPIR-V, which Vulkan consumes as is. D3D12 would add a SPIR-V to DXIL step; wgpu would add SPIR-V to WGSL through Naga and drop features the game uses (bindless textures, some atomics).
- LAS Recomp's Vulkan back end is a working reference on Windows.
- The same back end runs on Linux.

### 5. Toolchain

- Generated C: MSYS2 UCRT64 Clang + LLD. The generated code uses GNU extensions (`__int128`, `musttail`, vector types) that `cl.exe` cannot compile.
- Host code: the same Clang, C11.
- Builds: `tools/build_game.py` writes a Ninja file and runs MSYS2 Ninja. Every object is hashed before linking, because a faulty RAM module once flipped bits in stored objects.
- Analysis: Ghidra with the Switch loader, Capstone, `llvm-objdump`.
- Players: the installer ([installer.md](installer.md)) builds the game code on the player's machine with a bundled llvm-mingw, so no recompiled code is ever distributed.

