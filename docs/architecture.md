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
  - `LDXP`/`STXP` (128-bit exclusive pairs) supported. The exclusive monitor is a compare-and-swap against the value
    the load exclusive read (`runtime/cpu.h`): it does not see a write that puts the same value back (X -> Y -> X),
    which fails the store on hardware. That only matters to a window that reads memory or calls between the load and
    the store, as a hand-written lock-free pop does; one that computes from registers alone gets from a successful
    compare-and-swap what hardware gives when the thread runs after the other writers. `tests/excl_windows.py`
    checks every window of 1.0.1: main 2290 and sdk 841, none reads memory or calls, 7 instructions at most
    (2026-11-20). dynarmic also clears the other cores' reservations on a store exclusive, under a global lock; it is
    not needed here, and would cost a lock on every store exclusive;
  - no 64 MB limit on `main`; `subsdk*` loaded;
  - NZCV and the general registers of each function kept in C locals instead of the `Cpu` (`recomp/flags.py`,
    `recomp/regs.py`), X18 excepted. A register analysis of every lifted function at once (`recomp/ipa.py`) tells what
    each function reads of the registers it is entered with, what its callers read of the ones it returns with, and
    what it always writes: a call stores only the changed registers the callee reads, a return those a caller reads
    and the callee-saved ones, and after a call a function loads again only what its own code reads. Nothing is taken
    from AAPCS64 between two functions joined by a direct call; the target of an indirect call is taken to read X0-X8,
    X19-X30 and SP, and an SVC or a patch helper every register. `gen.py --abi-poison` builds a test lift that puts a
    trap value in every register a sync leaves out. `gen.py --no-abi-trust` is the lift up to 1.19.1: the six most
    used registers in locals, stored before calls, SVCs and exits, and X0-X18 and X30 loaded again after every call
    (`--no-callee-saved-trust` reloads X19-X29 and SP too).
  - `gen.py --v-locals K` keeps the K most used V registers of each function in a local `u128 vr[32]` that clang
    holds in XMM registers (`recomp/vlocals.py`, the `RT_VLOCALS` accessors of `runtime/simd.h`): loaded at entry,
    stored where they may have changed before calls and exits, reloaded after calls; `--v-trust` does not reload
    V8-V15, whose low halves AAPCS64 has the callee preserve. K is 8 by default; K = 0 keeps every V register in the
    `Cpu`. The host helpers of the code patches (`@fn` / `@fnz`, `runtime/frame_rate.c`, `runtime/resolution.c`) read
    and write `c->v`, so their calls are sync points like guest calls; a function that calls anything else that is
    not known to stay away from `c->v` keeps every register in the `Cpu`. `tests\vlocals_check.py` checks the syncs
    on the generated chunks.
- Coverage oracle: MacNativeGames' NSO rewriter lists every site where game code touches the OS (SVCs, TLS reads, counter reads, `x18`, exclusive windows, jump tables): 68,968 sites in 1.4.3 `main`. The lifter must handle every one.
- Memory: guest address equals host address, all below 2^39 (IPC encodes 39-bit addresses). The heap, stacks and the alias region are reserved with `MEM_WRITE_WATCH`, so the graphics back end can ask which pages the CPU wrote since the last upload (`runtime/memory.c`).
- Indirect branches go through `rt_dispatch` (`runtime/dispatch.c`), which looks the target up and tail-calls it: first in the calling thread's own cache of 1024 targets, then in the address table (`AC_DISPATCH_CACHE=0` leaves the cache empty). On Windows (a Core Ultra 9 285H, standing in a scene at 60 fps) the dispatch takes 2.1-2.2% of the samples inside the executable with the cache and 3.5% without it; the frame time is the same. An import stub (`adrp; ldr; add; br` on a GOT slot) whose slot the generator resolves to a lifted function compares the slot with that function's address and calls it directly when they match (`rt_stub_hit` in `cpu.h`), which they do unless the runtime bound a host function to the name; in the town 15 to 22% of the guest threads' indirect branches were such calls.
- Testing: LAS Recomp's differential harness, every lifted instruction class checked against Unicorn (test-only, never linked into the game).

### 3. Host

- Portable C from MacNativeGames: threads and synchronisation, IPC, memory services, fs/RomFS, time, account, applet, audio renderer and effects. The Apple pieces (mach VM, CommonCrypto, AudioToolbox, GameController) are replaced with Win32 and SDL3.
- SDK boundary at the `nn::*` function level, matching MacNativeGames. Only `audren:u` and `sm:` are real IPC services.
- Window, keyboard and mouse: Win32 (`gfx_window.c`, `input_win.c`). Gamepad: SDL3. Audio output: WASAPI on Windows (`audio_out_win.c`), SDL3 elsewhere (`audio_out_sdl.c`). Voice: libopus (BSD). Movies: the SDK's `movie::Decoder` is replaced by a host VP9 decoder over libvpx (BSD, static; `runtime/ov_movie.c`), because FFmpeg from MSYS2 is a GPL build.
- Sound and 5.1: the game plays everything through `audout:u` (`ov_audout.c`); its Wwise 2019.1 sink asks for stereo,
  then mixes for the channel count `OpenAudioOut` answered, so `audio_output = 5.1` makes the game mix 5.1 itself
  (FL FR FC LFE SL SR). The effects, voices and reverbs follow the output; the music is authored stereo and stays in
  front; the LFE gets nothing. Measured on the new game's opening (radio dialogue over traffic, 2 s RMS, 2026-11-21):
  fronts 0.069-0.082 in stereo; in 5.1 fronts 0.049-0.059, centre 0.054-0.079, surrounds 0.005-0.026, LFE 0. The
  voices came out about 3.4 dB louder than in stereo (the mono 2D sounds go to the centre at full level instead of
  -3 dB per side) and a phone beep too loud on headphones, hence the centre's default of -2 dB. The player's mix runs
  on the output thread (`mix_frame`): a gain per speaker, presets that write them, the bass of the five other speakers
  to the LFE (120 Hz low-pass, 10 dB down), a speaker test (pink noise per speaker, as a receiver does) and the meters
  of the menu's speaker diagram. A device without a 5.1 path gets an ITU-R BS.775 fold-down. WASAPI rather than SDL on
  Windows: a headset with a spatial sound format on (Windows Sonic, Dolby Atmos, DTS Headphone:X) keeps a stereo mix
  format, and SDL3 opens the endpoint at that count (asked for 6 channels on a PRO X Wireless with Sonic on, it opened
  2), folding the 5.1 before the virtualiser sees it; WASAPI takes the 6 channel stream as it is. 7.1 ran without a
  fault in one test (the sink's table goes to 8 channels) and is left out for now.
- zstd: most RomFS files of 1.4.3 are zstd frames. The patch set `zstd` sends the game's two decoder entry points to
  libzstd on the host (`ov_zstd.c`, `AC_ZSTD_NATIVE`); the lifted decoder took almost three times as long.
- Settings and menu: every key with its default is in one table (`cfg.c`), read from `astralchain.ini` and `AC_*`
  variables and written back when the in-game menu changes it. The menu and the performance overlay are Dear ImGui,
  drawn into the swapchain image (`runtime/ui/`). Their text comes from `lang/*.po` catalogs, keyed by the stable ids in
  `ui_text.h` and loaded at startup (`ui_text.c`); English is built in as the final fallback, so no `en.po` is shipped,
  and `pt` uses the `pt-BR` catalog. A new language needs no rebuild. `tools/ui_text_catalogs.py` generates the catalogs
  from the previous compiled rows and checks their ids, metadata, formats and lengths; each entry carries the English
  source in a gettext extracted comment. `build_game.py` and the release builders run it automatically. Fonts named by a
  catalog's `X-Astral-Font-Lang` header come from a system font
  (`plat_font_path`) merged into the font atlas.
- OS calls the host makes go through `runtime/platform/` (`platform.h`, `win32.c`), so a Linux or macOS host adds one file instead of touching the rest. The split is not finished: `tools/check_platform.py` lists the Win32 uses left in the other files.

#### Threads and cores

Every guest thread is a host thread. Astral Chain assumes Horizon's scheduler: three application cores, a thread runs only on a core in its affinity mask, and a higher-priority thread takes a core from a lower-priority one. The game keeps per-core data without locks and relies on that, so the host models it (`runtime/kernel.c`):

- Three virtual cores. A guest thread runs guest code only while it holds a core; the others wait in a priority-ordered list.
- Preemption. When a waiting thread has a better priority than a holder on a core it may use, exactly one holder is flagged. The holder checks the flag at its next SVC and at every indirect call (`rt_dispatch`: two loads when nobody waits), checks again under the kernel lock that a better thread still waits, and only then hands its core over. The indirect-call check comes from SavageWindRecomp's BotW port; checking only at SVCs left a better thread waiting up to 26 ms.
- Asynchronous preemption (from MacNativeGames 8bf0893, on unless `AC_ASYNC_PREEMPT=0`). Game code can run for
  milliseconds with neither (`AC_CORE_DIAG=1` measures it): a waiter that has spun 250 us suspends the holder and,
  if its host rip is in lifted code (not inside a host call or SVC) at host depth 0 (not in guest code that a host
  function runs through `rt_call` or `rt_original`, the only ways host code may enter guest code), moves it to `kern_ap_stub`, which
  saves the volatile registers and gives the core up there. A holder of a guest mutex counts at the priority of its
  best waiter, and one inside an SDK spin lock (TLR+0x100 `disable_count`) is not preempted: its interrupt flag is set
  and the switch happens at SVC 0x36. `AC_ASYNC_PREEMPT=2` keeps those two without interrupting. The interruption is
  Win32 (SuspendThread, SetThreadContext); another host needs a signal handler there, as MacNativeGames has.
- Host work gives the core back. Queue submits, presents, GPU waits, shader translation, `nn::fs` calls and NVN memory pool setup run without a core (`NVN_BLOCKING`, `kern_core_away`), as the Switch blocks those threads in IPC, so a thread blocked in the driver or on a file does not hold up guest threads.
- Spin waits. Loops that only another thread can end (857 in 1.4.3, 864 in 1.2.1, found by `tools/spin_waits.py`, the
  nxnative search) send their back-edges through `rt_spin_gate`, which pauses the core once per pass and hands it to a
  waiter after 50 us of polling (`AC_SPIN`, `AC_SPIN_HANDOFF_US`). A wait that has lasted 200 us stops polling at full
  speed: each further pass leaves the core and sleeps an eighth of the time waited so far, 250 us at most
  (`AC_SPIN_SLEEP_US`, 0 to keep polling).
- `free_cores = 1` (`AC_FREE_CORES=1`) turns the gate off, for experiments only: the game crashes or hangs, usually
  at start, because its job workers all report core 0 and share per-core memory (#212).

### 4. Graphics

- NVN front end from MacNativeGames (`nvn_api_*.c`, `nvn_obj.h`, texture layout and deswizzle). The game records command buffers through it; nothing reaches the GPU until the game submits.
- Vulkan 1.3 back end in place of MacNativeGames' Metal one. At `QueueSubmitCommands` the recorded NVN commands are replayed into Vulkan command buffers (`vk_exec*.c`).
- Memory pools. Apple's unified memory let MacNativeGames use the game's pools zero-copy. The back end imports the pools as host memory (`VK_EXT_external_memory_host`). On a discrete GPU, pools bound as buffers are mirrored in device-local memory, by the 64 KiB chunks the replay binds, and the pages the CPU wrote in them (write watch, above) are uploaded before a replay; the large storage buffers the shaders write stay in the mirror until the CPU touches them; idle pool regions are made read-only so the write-watch queries skip them (`vk_memory.c`). GPU-rendered textures are written back when the CPU reads them: on a pool invalidate, on a CPU copy, and for small targets once the CPU waits for a fence (`vk_texture.c`).
- Replay thread. `AC_REPLAY_ASYNC=1` (the default; `=0` turns it off) moves the replay off the submitting thread onto a queue at most one frame behind the game; reads of GPU results wait for the newest writer instead of draining the queue (`vk_backend.c`). It is on because an A/B run on the dev machine with a save load and gameplay showed identical throughput; the waits were under 160 ms over 3,472 frames.
- Low latency (`low_latency = off|on|boost`, `vk_lowlat.c`, off by default). The pacer lets the game start a frame
  two periods before its display slot (three window textures), so the pad read for a frame happened 33 ms before its
  present at 60 fps and 67 ms at 30 fps. A frame starts when the game acquires its window texture
  (`nvnWindowAcquireTexture`, which reads the pad right after, on the same thread), and an ID follows it to its
  submissions (`ReplayItem.lid`, a thread-local on the replay and present threads) and its host present
  (`PresentJob.lid`, `flip_locked`). Three back ends, one at most: NVIDIA Reflex (`VK_NV_low_latency2` with
  `VK_KHR_present_id`: the driver's sleep before the frame, the six latency markers, the frame's ID on its
  submissions and present), AMD Anti-Lag 2 (`VK_AMD_anti_lag`: one call before the pad read, frames delimited by the
  present), else a built-in sleep until the slot minus the work of the last frames (start to blit done on the GPU,
  DXVK's median-of-three maximum over 32 frames) minus a margin that grows on a missed slot and shrinks back. XeLL
  has no Vulkan API. On an Arc 140T the built-in sleep took the pad read to present from 33.3 to about 23 ms at 60 fps
  and from 66.7 to about 33 ms at 30 fps (fifo, the Deck's present mode), with no more frames over 25 (50) ms than without
  it. `AC_TRACE=latency` logs every 600 frames where a frame's time went; `--probe-latency` runs the back end on a
  synthetic game.
- Recorder thread. The Vulkan calls the replay makes into its command buffer are made by a thread of their own (`vk_rec.c`, after Savage Wind Recomp and LAS Recomp): each `vkCmd*` copies its arguments into a packet of a ring and the recorder makes the calls in order; calls into other command buffers are made at once; the recorder also ends the command buffer, and the submit waits until it has. In the town on an RX 9070 the replay takes 17% less time on the submitting thread. It is off by default on AMD GPUs, where the driver loses the device or hangs the GPU at startup with it (`AC_VK_RECORDER=1` forces it on, `=0` makes every call on the submitting thread).
- Uniform snapshot. The game rewrites per-object uniform blocks on other threads soon after it submits the frame
  that reads them, while the replay still records it. At the start of a submission's replay every uniform range it
  binds from CPU-written pools is copied, and the draws bind the copy (`vk_usnap.c`, after MacNativeGames'
  `AC_UBO_SNAPSHOT`); without it, late draws read another object's matrix.
- Repeated uniform binds. The game binds all 14 uniform slots of a stage before nearly every draw (54,560 bindings
  a frame in the town, 87% of them the range the recording already holds at that stage and slot). The recorder
  leaves those out (`nvn_api_cmd.c` `ubo_table`; `AC_UBO_DEDUPE=0` records them all): they were 1.7 of the 2.6 MiB of
  records a frame, copied at EndRecording and walked at the submit and by the replay, which sets its binding at such
  a record and nowhere else. The command memory reported to the game counts them as before.
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
  `deck_res_migrated = 1`. The game's SMAA stays on there as elsewhere (1.19.0 and 1.19.1 turned it off on the Deck
  and left `deck_aa_migrated = 1` in the file, a marker nothing reads now). The game's render targets and window
  textures come from one budget the graphics configuration sets (main+0x40e0b0: 310 MiB, of which 1080p uses 304),
  carved from the process heap; above 1080p's pixel count `rt_res_budget_w8` raises it with the pixels (`res_budget`:
  the measured 87 MiB and 110 bytes a pixel, 15 % to spare) and `res_heap_extra_bytes` grows the application pool by
  the same amount (TotK's HeapDef growth does not apply there). The GRAPHIC BUFFER heap (143 MiB inside BUFFER ROOT,
  sized at main+0x575c94 and main+0x575a30) holds the constant buffers and a storage buffer of 6.97 bytes a pixel of
  the render size, so above 1080p it and its root grow by 8 bytes a pixel as well (`res_gbuf_extra`,
  `rt_res_gbuf_w1`, `rt_res_bufroot_w1`): at 3840x2160 the fixed size ran full in play and the allocator crashed.
  1.0.1 has no `aspect` set and no HUD layouts for other shapes; its wider pictures come from the `wide` set below. The
  game always renders its picture at its own size: nothing enlarges its render targets from outside (the former
  `ssaa`). When the window is larger than the output, `vk_fsr.c` runs AMD FSR 1 (EASU, RCAS) on the present. An
  `output_res` that is not 16:9 (more than 1% off, `res_is_16x9`) also switches on the
  `aspect` patch set, which hands the output's ratio to the camera, and `romfs.c` `overlay_init` loads the
  `romfs-patches-<w>x<h>` overlay for it when it exists (`tools/make_romfs_patches.py --aspect`, laid out by
  `tools/ultrawide_layout.py`; the installer builds ten sizes: 21:9 2560x1080, 3440x1440, 3840x1600, 5120x2160, 32:9
  3840x1080, 5120x1440 and 16:10 1280x800, 1680x1050, 1920x1200, 2560x1600). A wider layout moves the HUD to the
  screen's corners, a narrower one to its top and bottom edges. Other sizes get the camera of their shape and the 16:9
  HUD stretched over it. The `aspect` setting (`fit`, the default; `stretch`; `wide`) decides what the present does
  when the window's shape differs from the output's: `fit` letterboxes, where a 16:9 game in a wider window used to be
  stretched.
- Wide pictures on 1.0.1 (`aspect = wide`, the default; `resolution.c` `res_wide_tick`, set `wide` of
  `recomp/patches-v101.txt`, switched on the first time the window's shape is more than 1 % off 16:9;
  `cfg_migrate_aspect` removes, once, an ini `aspect = fit` or `stretch` from before the option and marks the file
  with `aspect_migrated = 1`). The game's targets stay 16:9, since sizes off the 32n x 18n grid lose the device; a
  window wider than 16:9 (up to 48:9) gets them drawn with a wider camera instead, and a taller one (16:10, the Steam
  Deck's 1280x800, down to 4:3) with a taller camera that keeps the 16:9 view's width: where the projection is built
  from the lens's vertical angle, `rt_wide_angle24_s0` (or the fov set's helper on the same site) widens that angle
  for the taller ratio, and the lens keeps the game's own. The present stretches the picture to the camera's shape, so
  the pixel count does not change. Each frame has a shape: the ratio its lenses were built with. `rt_wide_dirty_w8`,
  where the game builds a lens's matrices, hands every lens made 16:9 the ratio of the frame being built; the game's
  present (`vk_backend.c` `vk_be_present`) takes that frame's shape along to the present thread
  (`gfx_present.c` `gfx_fit_rect`) and decides the next frame's. The ratio is the window's, and 16:9 while a
  full-screen movie plays (`frame_rate.c` `rt_movie_showing`, counted in presents so it holds at 30 fps too): a movie
  is a 16:9 picture and is shown between black bars, with the camera back at 16:9 for those frames. The interface is
  drawn through a camera of its own (a 20 degree lens in front of the 1280x720 plane); with the frame's ratio in its
  lens too, the interface keeps its proportions in the middle 16:9 of the picture. Its matrices are built once by the
  game, so the helper calls the game's own build again when the ratio changes (`ui_camera_refresh`). World to screen
  (`rt_wide_view_x8`) then maps onto that wider plane, and the visibility cone (`rt_wide_cone_s0`) is widened to the
  picture's corners. The rest of the interface (the mission banner, the dialogue's lines, the civilians' markers) is
  drawn through a flat camera over the same 1280x720 plane, the first of three the game makes; it also draws the
  interface's 1280x720 textures (the minimap's picture) and the effects that cover the picture (the IRIS mode's
  veil), so it keeps its plane, and the interface layers drawn onto the screen bind the second one instead, which
  `rt_wide_flat_w8` keeps as that plane at the frame's width. The HUD's blocks then go to the picture's edges
  (`rt_wide_anchor`): each is an instance of a layout, a tree of parts whose world matrices the game makes every
  frame, and the top parts of a gameplay HUD layout (archive `hud2`) whose drawn parts sit in the left or right third
  of the plane are moved by half the plane's extra width (in a taller frame, those in the top or bottom third by half
  its extra height); the side is decided once per instance from the layout's
  own places, only instances placed at the plane's middle count (one placed elsewhere hangs on a place in the world
  or on a part of a block that has moved already), and menus keep their composition in the middle 16:9. Not done:
  widening a layout's own backdrops that span the screen. `--shot` and `AC_SHOTS` save the
  game's 16:9 texture; `AC_PRESENT_SHOT` saves the window's picture.
- Draw distance (`draw_distance`, 100 to 400 %, 100 by default and in the Steam Deck profile; `resolution.c`
  `res_drawdist_live`, set `drawdist` of `recomp/patches-v101.txt`). A model holds up to four level-of-detail rates
  in 0..1 (`DistRate0` to `DistRate2`, the steps between its four levels, and `LostDistRate`, below which it is not
  drawn). The set divides each rate by the percentage where the model's set-up stores the defaults and where its
  named parameters are read, so models keep their finer level, and small ones stay drawn, from further away. In
  Harmony Square at 400 % the alley yard shows about 15 debris pieces where 100 % shows 3. The percentage scales the
  rates, not a distance in metres, and a model takes its rates when it is made, so a change shows from the next
  loading screen. Two other distance checks were tried and left out: the visibility test of effect emitters
  (main+0x991220, one or two entries a frame in the city, nothing visible changed) and the radius placed objects are
  made within (20 m, main+0x4bb1ac).
- No HUD (`hud = 0`; F12 and the menu's Game tab switch and save it; `resolution.c` `res_hud_tick`, set `nohud` of
  `recomp/patches-v101.txt`). The game sends a layout instance's parts to the renderer from two functions
  (main+0x9177d0 for an instance with a layer of its own, main+0x915e50 for the queued ones); the set's site in
  each makes an instance of the gameplay HUD's archive (`hud2`) read as one with nothing to draw
  (`rt_hud_off23_x8`, `rt_hud_off22_x8`). The game keeps updating those layouts, so the HUD is back whole in the
  frame after the set goes off, and the menus, the title screen, the loading screens and the dialogue's lines,
  which are not `hud2` layouts, stay. The game's present switches the set when the setting changed.
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
- Temporal upscaling (`vk_dlaa.c`, off by default): `upscaling` puts NVIDIA DLSS (`dlss`, through the driver's NGX core
  loaded at run time, Windows only), Intel XeSS (`xess`, `vk_xess.c`, `libxess.dll` loaded at run time, Windows only),
  AMD FSR 3 (`fsr3`, `vk_fsr3.c`, every OS; `auto` is DLSS where it can run, else FSR 3) or AMD FSR 4 (`fsr4`,
  `vk_fsr4.c`, through AMD's FSR API, `ffxCreateContext` / `ffxDispatch`, loaded at run time: AMD's signed
  `amd_fidelityfx_loader_vk.dll` and `amd_fidelityfx_upscaler_vk.dll` of the FSR SDK 2.4.0 on every OS, pinned by
  `tools/fsr4_sdk.py` and shipped in the release's `bin/`; Linux, for which AMD publishes no library, runs the same
  DLLs through ffx-vk-linux, a PE loader with the Windows services and calling-convention bridges they need, vendored
  in `runtime/third_party/ffx-vk-linux/` and built into `libffx_vk_linux.so`, which exports the same entry points;
  the menu offers it only on an AMD GPU with `VK_KHR_ray_query`, i.e. RDNA 2 or newer, which is what AMD's
  FSR 4.1 supports; the API's structures are declared in `vk_fsr4.c`, no SDK header is compiled in) in place of the
  game's SMAA (`anti_aliasing = smaa | off`, for when none runs), in the mode `upscaling_quality` gives (`native`:
  DLAA, XeSS or FSR Native AA at the scene's own size; `quality` ... `ultra_performance`: a smaller scene, below); no
  SDK header or code of NVIDIA's or Intel's is compiled in. FSR 3's passes are AMD's GLSL, unmodified and not in the
  repository: `tools/ffx_sdk.py`
  fetches the pinned files of the FidelityFX SDK (commit and per-file SHA-256) into `toolchain/ffx-sdk/` and compiles
  them into `fsr3_spv.h` in the build directory, and `vk_fsr3.c` is our own host for them (internal images, constant
  blocks, the seven passes in AMD's order). FSR 4 takes the same inputs as FSR 3; its library reads the colour, depth
  and motion vectors in SHADER_READ_ONLY_OPTIMAL (the API's COMPUTE_READ state), so `vk_fsr4.c` moves them out of
  GENERAL around the dispatch and back (the recorder thread makes the barrier before the library records into the
  command buffer itself, as for XeSS), and passes a 1x1 R32F exposure image holding 1. AMD's library is also given
  the API version, which its documentation requires since SDK 2.1,
  the stencil usage of the D32F_S8 depth (its back end otherwise makes a D32F view of it, which is not valid) and, for
  the tonemapped composite, which is sRGB-encoded, the non-linear colour flags. AMD's loader makes its newest
  upscaler that takes the GPU, FSR 3.1 where FSR 4 does not run (provider 3.1.6 on an Arc 140T), without an error:
  `vk_fsr4.c` asks the context for its version and, when it is not FSR 4, destroys it and turns FSR 4 off for the
  run (`AC_FSR4_ANY_VERSION=1`, a test switch, keeps the context). The library picks FSR 4's model from the GPU
  family: the full network on the Radeon RX 7000 and 9000 series, a version of it for the RX 6000, FSR 4 Lite on its
  integrated GPUs. The SDK 2.4.0 DLL lists no family for the integrated RDNA 3 GPUs before the Radeon 890M (a Ryzen
  Z1 Extreme's 780M got FSR 3.1.6), so on Linux, where ffx-vk-linux answers the library's question about the family,
  an AMD integrated GPU that got FSR 3.1 is asked again for FSR 4 Lite, which the Z1 Extreme then runs (4.1.2).
  Wherever FSR 4 cannot run (one of its files is not beside the executable, the library cannot be loaded, it does not
  take the GPU, a call fails), the log says why, the menu's row names the missing file and what runs, and FSR 3 runs
  in its place, at the scene's own size too (`vk_dlaa.c` `pick`). One context is kept per scene size, output size
  and colour kind, four at most (AMD gives a working set of 81 MB at 1920x1080 and 318 MB at 3840x2160): the least
  recently used goes for a new one and all go after 600 presents without FSR 4, each destroyed once no frame in
  flight uses it. The library compiles its shaders when it makes a context (0.2 to 1 s for 1920x1080, 2.6 s for
  3840x2160 on the Z1 Extreme), so the context of the game's sizes is made while the game starts (`vk_dlaa_warm`,
  `boot.c`) and not at the first frame with a 3D scene, where it stalled the frame after a load and left a cutscene's
  audio ahead of its picture. With `upscaling = fsr4` at start the device also enables
  `VK_KHR_compute_shader_derivatives`, which the shaders of AMD's library declare, and `VK_KHR_cooperative_matrix`
  and `VK_EXT_shader_float8`, which its FP8 model for the RX 9000 names (not run here: no such GPU at hand), when the
  GPU has them. Astral Chain renders forward: a depth-only prepass
  writes the scene depth (D32F_S8), the opaque draws and the transparents go into an R11G11B10F light target with it (two images,
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
  A smaller scene (`resolution.c`, the `scene` patch set of `recomp/patches-v101.txt`): the six `csel` of the game's
  dynamic-resolution ladder (main+0x40f18c..0x40f1dc) take every level's size from the patch page, so the game makes
  its scene targets (depth prepass, light target, bloom, composite) at the mode's share of the game resolution, on the
  32n x 18n grid the game takes (off it the first 3D frame lost the device), at least 640x360 and in handheld at most
  1280x720; the output and HUD stay at the game resolution. Decided once at start from the settings and the files
  next to the exe (`scene_init`), and checked with the device up before the game runs (`res_settle`, `boot.c`): an
  upscaler that cannot run here gives the game's size back. Upscaled, the motion vectors are still made where the
  scene ends, and the upscaler runs at the begin of SMAA's blending pass, the game's only scaling pass, on the
  composite (LDR colour, or HDR with HDR output), into an image of the game resolution; passes wider than the
  composite that sample it then sample that image (`vk_exec_draw.c` `tex_info`, part of the descriptor build
  signature), so the blend copies the upscaled picture and the HUD is drawn over it unjittered. The jitter sequence
  has 8 x (output / scene)^2 phases (16 to 72), DLSS gets the quality value and model per mode, XeSS the quality
  setting whose input range holds the scene (`xessGetOptimalInputResolution`), FSR 3 its output-size images and an
  LDR build of its passes (`tools/ffx_sdk.py`). Textures keep the game's LOD bias while an upscaler
  scales: the bias of log2(scene / output) the upscalers' guides ask for made fine regular patterns (the road's dot
  grid in Harmony Square) shimmer in rings as the camera moved (`vk_texture.c` `sampler_for`). With `upscaler =
  fsr`, sharpness above 0, no HDR and a window that shows the picture 1:1, RCAS sharpens the upscaled picture before
  the HUD and the present copies the frame (`vk_fsr_rcas`, `vk_fsr_presharp_set`). A backend that fails during play
  with a smaller scene leaves the first that can of DLSS, XeSS and FSR 3 until a restart, so the game's plain scaling
  never shows. HDR output records the upscaled picture as the frame's scene (`vk_hdr.c` `vk_hdr_draw`).
  `output_res = auto` takes the screen's size, up to 1920x1080 without an upscaler and 3840x2160 with one (2560x1440 on
  a GPU with 4.5 GB of video memory or less).
- Ambient occlusion (`vk_ao.c`, `ao = off | low | high | gi`, live; off by default): the game has none.
  The first light-target pass with the scene depth after the game's R32F linear-depth pass that writes colour is the
  opaque lighting (in play, every other frame, a pass with its colour writes masked off comes before it); when
  the next pass with other targets begins (or a dispatch or copy comes first), `ao_prep.comp` takes the scene depth to a
  half-resolution view z, `ao_src.comp` the light target (without the fog) to the light the surfaces send out, at a
  quarter of the scene's resolution with coarser levels, `ao.comp` searches horizons along 2 or 3 directions within
  1.5 m and a quarter of the picture's height (GTAO's closed-form integral, a fixed 4x4 pattern), `ao_blur.comp`
  averages the pattern out at half resolution with depth-aware weights and turns the visibility into the factor on the
  light by a fixed curve (visibility from 0.9 up is left alone, the rest to the power 1.5, then lifted by the
  multi-bounce fit of Jimenez et al. 2016 with an albedo of 0.3: `AO_OPEN`, `AO_POWER`, `AO_MULTIBOUNCE`), and
  `ao_apply.frag` scales the factor up and multiplies the light target through a blend. Transparents, bloom, tonemap
  and HUD come after. The camera is the c3 block of the scene's draws, as for the motion vectors.
  The curve was softened from a plain square after players found the darkening too strong where no direct light
  falls: there the whole pixel is the ambient part, so the occlusion acts at full strength (measured on captured
  Harmony Square frames: 88 to 96% of the pixels under a factor of 0.5 were such pixels). The power and the multi-bounce
  fit took those pixels from 15.2% to 9.6% of the picture at `high`, keeping about three quarters of the contacts'
  darkening. `ao.comp` also adds 4 times the square of a sample's offset along the view direction to its squared
  distance (`AO_THIN`, after XeGTAO's thin-occluder compensation), so that component counts √5 times its length,
  towards or away from the camera (1.5 m of reach become 0.67 m straight along the view): the depth buffer has no
  thickness, and the open metal cage around a Harmony Square tree darkened 1.5 m of the grass behind it like a wall
  (that grass kept 89% of its light at `high` before, 97% after). It also weakens the darkening of a ground 0.25 to
  0.75 m from a wall seen face on by 35 to 45%; penalising only the samples on the camera's side gave the same
  picture. Judged in play in Harmony Square, simulated on captured frames. Left out: a lower strength or a
  floor (they took half the contacts' darkening), a shorter radius (shortens the reach under cars everywhere), a raised
  cap on the `gi` bounce (g rarely reaches it beyond 1 m of a source; the probes' shimmer gathers near sources).
  The occlusion takes only what the sky gives. The game's opaque shaders light a surface with `mask x albedo x
  ambient(N) + local lights + reflections + emission`, `ambient(N)` being the ambient cube map (fragment texture slot
  5) along the normal times c5 row 1 `.w` over pi; `vk_ao_draw` takes both from the opaque lighting's draws. The
  factor acts on `min(surface, 0.9 x ambient(N))` of each pixel only, so signs and surfaces under lamps keep the rest,
  and `ao.comp` gives a hidden arc back in proportion to the light of what hides it over the ambient light, per colour
  channel, from 1x up and whole at 3x (`AO_CREDIT_LO`, `AO_CREDIT_HI`): what stands above a lit ground line keeps that
  line's light and colour, and the contact under a car stays. Without those inputs a frame multiplies the whole
  surface and lets go of what glows (light several times the picture's typical brightness, which `ao_meter.comp`
  reads). Tried in play and left out: the screen-space sun mask in the bound (it is 0 all over the characters, which
  then lost all occlusion, and on the ground it took the contact off a car in its own shadow), the previous frame's
  light target as the light sent out (it holds the holographic HUD, a frame late), and the ambient light read along
  the open direction (no visible change in Harmony Square).
  The game's fog keeps its light: every opaque fragment shader ends with a distance fog, `mix(surface, cube map colour
  x scale, f)` with `f` from the distance to the camera, and that fog can be most of a pixel with a small `f` (a misty
  street 20 m away: 1.5 from the surface, 16 from the fog). Multiplying such pixels whole dug dark, blurred, shimmering
  stains into the mist around distant objects (38% of that picture lost more than 5% of its light) and moved the
  game's exposure. `vk_ao_draw` takes the fog parameters (fragment block c5, rows 2 and 4) and the fog cube map
  (fragment texture slot 12) of the opaque lighting's draws, `ao_blur.comp` computes the same fog per occlusion texel,
  and each colour channel takes the occlusion on the surface's share of the pixel only; a frame without usable fog
  parameters multiplies the whole pixel. In that street the frame-to-frame change of a still picture under FSR 3 went
  from 0.054 back to 0.030 (0.027 without occlusion). Half resolution and the fixed pattern hold with a temporal
  upscaler too: with FSR 3 a pattern turning per frame made the picture less steady, and the scene's full resolution cost 2.5 ms for no gain. Lengths on screen count
  in heights of the target, so ultrawide targets and `aspect = wide` (an ultrawide view in a 16:9 target) behave alike.
  On an Arc 140T, apply included: 0.72 ms (low) and 0.98 ms (high) on a 1472x828 scene (FSR 3 balanced for
  2560x1440), about 1.85 ms (low) at 2560x1440; occluding only what the sky gives and the light sent out added
  0.16 ms (low) to the first. `AC_GPU_PASSES` times each step ("ao"),
  `AC_AO_DEBUG=1` shows the factor alone, `2` the part of the scene the occlusion acts on (without the fog computed
  here and the light beyond what the sky gives: a check of both), `3` multiplies the fog too.
  `ao = gi` runs the occlusion as `high` and adds radiance cascades over the depth buffer (`vk_rc.c`), for the light
  the visible surfaces send each other at long range, only when the game's ambient light is known (without it the
  frame is `high` alone). It is the screen-space hierarchy Sannikov describes for Path of Exile 2: probes on a screen
  grid (cascade 0 every 4 occlusion texels; the spacing grows with the occlusion's width to keep about 184 probes
  across, so the reach follows the width: 61% of the height at 16:9, more on wider pictures and below 736 texels wide),
  8 screen directions doubling per cascade, intervals doubling in length over 6 cascades (252 texels at 736x414), one
  march of the view z per probe and direction serving every elevation through a 32-sector bitmask in 4 elevation
  bins. `rc_trace.comp` marches and merges each cascade from the top down, counting by default only what a surface
  sends beyond the sky it hides (the ambient cube map along the direction to it), and `rc_resolve.comp` turns cascade
  0 into g, the bounce over the ambient light along the normal. `ao_blur.comp` brings g to the occlusion's texels,
  caps it at 1, fades it out from 25 to 50 view units and makes the factor `1 + share x (g - (1 - k))`, which
  `ao_apply.frag` lets rise above 1; there, a pixel whose four texels all lie on another surface searches the 4x4
  texels around it for its own within the fade distance (with a factor far from 1 that band showed one-texel stairs
  along slanted depth edges; no measurable cost on the Arc 140T); the other levels keep their shaders' paths
  unchanged. `AC_RC_LIGHT=zero` counts
  the samples' whole light, `AC_RC_APPLY=0` computes without applying; `AC_RC_DEBUG=g` and `open` (with
  `AC_RC_DEBUG_CASCADE=<i>` for one cascade alone) show the bounce and the open share of the hemisphere through
  `ao_apply.frag`'s debug path. On an Arc 140T, 1472x828 scene, before the application: 0.68 ms for the cascades and
  0.05 ms for the resolve (`AC_GPU_PASSES` "ao step 6" and "7"), 9.5 MiB of images, made at the first frame at `gi`
  and given back after 600 presents without it; applied, `gi` cost 0.83 to 0.93 ms over `high` in single runs at
  1472x828 (FSR 3 or XeSS balanced for 2560x1440), 1.2 to 1.4 ms at 2560x1440 without an upscaler.
  On captured frames the shaders match a numpy oracle (ratio of the
  means 0.999 and 1.002, correlation 0.993 and 0.997); against a brute-force reference the cascades lose about 20% of
  the light, mostly from the back-face test on sparse samples. In motion the bounce near small bright sources shimmers
  slightly; on frames captured in play at such a spot the brute-force reference changes from frame to frame almost
  as much as the cascades (0.042 against 0.049 of the ambient light in the shimmering patch), so the cause is the
  screen-space input under the camera's motion and jitter, and sampling changes (more directions or steps, a light
  level set by the ray's footprint, a tent filter) gained 11% at most there; a 3x3 average of g had no visible effect in
  play. So the bounce is blended with last frame's at the probes, on by default at `gi` (outside the design's
  no-history rule; `AC_RC_TEMPORAL=0` turns it off as a debug switch), reprojected with the camera and kept on the same
  surface, dropped at a camera cut: offline 70% less frame-to-frame change in the shimmering patch, in game on a
  turning camera the probes' change from 0.23-0.33 to 0.05-0.06 of the mean bounce (about 80% less), for 0.02 ms and
  0.3 MiB (Arc 140T, 184x104 probes), made only at the first frame at `gi`, at the price of the bounce of moving things
  following them about 3 frames late. `vk_rc.c`'s header lists the measurements and the variants left out.
- The sun shadow map, one 2048x2048 DEPTH24_STENCIL8 texture on 1.0.1, is made `shadow_scale` times larger on each side
  (`vk_texture.c` `shadow_scale_of`; 1 by default, the game's size): its depth-only pass draws at that
  size and the screen-space sun mask that reads it samples in UV, so edges get finer at the same softness. At x2 it
  takes 85 MiB of its own on an Arc 140T (the pool storage it no longer aliases stays) and its pass about 0.5 ms more.
- Occlusion queries never block the CPU; a result that is not ready counts as visible.
- ASTC textures are uploaded as they are where the device reports `textureCompressionASTC_LDR`, and decoded by a
  compute shader everywhere else (`vk_astc.c`, `AC_ASTC`). A discrete GPU with 6 GB of video memory or more keeps
  the decoded textures as RGBA8, their full quality (`AC_ASTC_BC=auto`, the default, resolves to `0`), and the
  texture cache's default bound grows to half the device budget for them. Below 6 GB and on integrated GPUs
  (`bc1`) they are re-encoded on the GPU: the alpha scan of the raw storage (`tex_astc.c`, sound, the swizzle only
  permutes 16-byte sectors) picks BC1 for the provably opaque textures — the same colour quality as BC3 at half its
  size — and BC3 for the rest. The scan reads each block's header once per distinct header and only the two
  endpoint alphas of alpha-carrying modes; a thread that waits on it (the replay making an image the background scan
  has not reached) spreads it over the ASTC worker pool (`tex_astc_mt.c`). Both lose against RGBA8 (blockier gradients, less precise normal maps), which shows
  most on the environments' large tiled surfaces. `1` forces BC3 everywhere.
  Video memory can still run short: the RGBA8 textures outgrow a 6 GB card, or another program takes memory after the
  game started (a recorder) and the system lowers the game's budget. Two guards, both for the rest of the session.
  Before anything fails, once the device is over its whole budget, RGBA8 turns into BC3 for every ASTC texture made
  from then on and the texture cache's budget drops to what it holds less the overshoot past `AC_TEX_VRAM_PRESSURE`
  (`vk_texture.c` `over_budget`, `vk_astc.c` `vk_astc_low_memory`). When an ASTC decode finds no room for its scratch
  buffer anyway, the deferred objects that are done are freed (the scratch buffers of the decodes before it first) and
  the decode runs again; then the cache budget drops the same way, the entries that can come back intact are purged
  and the decode runs again; at last the scratch is taken from system memory, slower but never left out
  (`vk_texture.c` `tex_reclaim`, `vk_astc.c` `vk_astc_record_host`). A texture that still got nothing is decoded again
  at its next use instead of keeping stale memory.
  `AC_VRAM_LIMIT_MIB` makes the port act as if the card had that much video memory, to drive these paths on a large
  card or an integrated GPU: the budget the guards read is at most the limit, the texture cache's default bound and
  the mirror refusal follow it, and the start choices taken from the card's size (`AC_ASTC_BC=auto`, `output_res =
  auto`) take the limit as the size. On an integrated GPU the imported guest pools sit in the device-local heap; they
  leave the usage held against the limit, as a discrete card keeps them in system memory. The driver knows nothing of
  the limit, so no allocation fails past it (`vk_memory.c` `vk_mem_vram_limit`). The memory census at frames 600 and
  1800 logs the process and system commit and, per heap, VMA's blocks beside the driver's usage; crash reports print
  the commit too.
  --probe-astc holds both encoders and the scan to PSNR and exactness bounds. The CPU decoder (`tex_astc*.c`) is the fallback and the test oracle.
- Colour render targets are made `MUTABLE_FORMAT`, for the game's views in another format. Below GFX11 RADV turns
  colour compression off for such an image unless it carries a format list, so under RADV on an RDNA 2 GPU (the
  Steam Deck, RX 6000 cards, Ryzen 6000/7000 integrated GPUs) each one gets a list of its own format and its sRGB or
  UNORM twin (`AC_TARGET_COMPRESSION`, `vk_texture.c` `target_compression`): -3 to -4% GPU time per frame in Harmony
  Square in the Steam Deck profile, measured on a Z1 Extreme whose Mesa applies the GFX10.3 rule (`tools/deck_power.sh`
  loads it). A view in a format outside the list makes the target again without one, with its content (`unlist`).
- Draw state. A draw whose program, bound state and textures repeat reuses the previous draw's shaders and skips rebuilding its descriptor sets; descriptor sets are written through update templates (`vk_exec_draw.c`, `vk_pipeline.c`).
- Allocation: Vulkan Memory Allocator. Debugging: Vulkan validation layers, per-pass frame dumps (`AC_DUMP`, or F12 with `AC_DUMP_KEY=1`; `vk_dump.c`) and a flight recorder (`AC_FLIGHT`). RenderDoc cannot capture, because it hides `VK_EXT_external_memory_host` from the application.

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

