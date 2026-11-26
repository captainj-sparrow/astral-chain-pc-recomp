# Astral Chain Recomp

**Astral Chain Recomp** is a static recompilation of *Astral Chain* (Nintendo Switch, 2019) into a native x86-64
application for Windows, built on the pipeline of [TotK Recomp](https://github.com/valentderah/totk-pc-recomp)
([docs/architecture.md](docs/architecture.md)).

> [!IMPORTANT]
> **No game files are included.** You need your own legally obtained copy of *Astral Chain*, dumped from your own
> console: the base game (version 1.0.0) and the 1.0.1 update. Instructions for obtaining those files will not be
> provided.
>
> The game is built from your files on your machine, so the result contains the game. It is for your own use. Do not
> share it.

Special thanks to [@capibawa](https://github.com/capibawa) for their multiple contributions, the list is very long but it does include performances, code review, timing correctness, visuals, HDR and upscaling

## Status

The game is playable from the title screen to credits. The port has no known gameplay issue, including correct timings at 60FPS (perfect dodge, assault attack, 360 spin and so on) and no known visual issues. People enjoy this port for it's stutter-free experience, visuals, no audio desync (since 1.13.0 port's update) and good performances. The project goal is to make people discover this hidden gem from Platinum Games.

Report problems as issues, following [CONTRIBUTING.md](CONTRIBUTING.md).

## Features

- Native code, not emulation. It already runs on very old systems;
- Externally tested (Reddit, YouTube, ...): playable from intro to credits.
- [Flat 60FPS on Steam Deck](https://steamdeckhq.com/news/astral-chains-pc-port-60-fps-on-steam-deck/)
- HDR output (HDR10), with separate brightness for the game and the HUD (on Linux, under a Wayland compositor with HDR:
  start the game with `SDL_VIDEODRIVER=wayland`, since SDL may otherwise pick XWayland, which offers no HDR10 output)
- Temporal upscaling and anti-aliasing for the 3D scene up to 4K, from native (DLAA / Native AA) to Ultra Performance: NVIDIA
  DLSS on RTX cards (with `nvngx_dlss.dll` next to `astralchain.exe`, on Linux `libnvidia-ngx-dlss.so.<version>` next
  to `astralchain`), Intel XeSS on most recent GPUs, AMD FSR 3 on any
  GPU, AMD FSR 4 on AMD Radeon RX 6000 series or newer (Windows and Linux);
- NVIDIA DLSS Frame Generation (2x, and 3x or 4x on the RTX 50 series) on RTX 40 or newer cards, over a temporal upscaler
  (with `nvngx_dlssg.dll` next to `astralchain.exe`, on Linux `libnvidia-ngx-dlssg.so.<version>` next to `astralchain`),
  with SDR and HDR output;
- Anisotropic texture filtering up to 16x, for sharp ground into the distance
- Low latency mode: NVIDIA Reflex or AMD Anti-Lag 2 where the driver offers them, the port's own frame timing on any
  other GPU (Intel, Steam Deck); about a third less time from the controller to the screen at 60 fps, half at 30
- In-game settings menu (F1) that pauses the game while it is open, performance overlay (F10)
- Optional pause while the game's window is out of focus (the menu's Game tab, "Keep running out of focus")
- No-HUD mode for screenshots and videos: F12 (or the menu's Game tab) hides the gauges, map, prompts and markers
  during play and shows them again; the menus stay
- In-game menu, overlay and setup installer in English, Italian, Russian, Brazilian Portuguese and Simplified Chinese;
  the menu's translations are plain gettext files in `lang/` that anyone can add to (docs/building.md, "Texts and
  translations")
- Keyboard, mouse and gamepad, with remapping, rumble and gyro
- Xbox and PlayStation button prompts, each with its matching button layout (PlayStation mod from https://gamebanana.com/mods/604666)
- Shaders prepared at install instead of during play
- Ultrawide - From 16:10 to 48:9
- Ambient Occlusion, and at its top level bounced light: lit surfaces brighten and tint what faces them
- 5.1 surround sound

## Installing

Download `Astral Chain-Setup-<version>-windows.zip` from [Releases](../../releases), unpack it anywhere and start
`Astral Chain-Setup.exe`. It asks for:

- your own copy of the base game (version 1.0.0, XCI or NSP) and of the 1.0.1 update (NSP);
- the game language and the folder to install into.

It then extracts the game's data, merges the update into it, lifts the game's code to C, and builds `astralchain.exe` on
your machine: about a minute on a fast CPU, a few minutes on a slow one, about 14 GB for the installed folder and about
11 GB of temporary space while the update is merged. The setup carries no game data and no code derived from the game.

You need Windows 10 or 11 on x86-64 (SSE4.2 at least) and a Vulkan 1.3 graphics card. The exe is unsigned, so
SmartScreen warns on the first start.

## Credits

Astral Chain Recomp builds on the pipeline of [TotK Recomp](https://github.com/valentderah/totk-pc-recomp), which
supplied the lifter, the runtime, the graphics translation, the installer and most of the documentation this port
started from. What was taken from each project, with its license, is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [TotK Recomp](https://github.com/valentderah/totk-pc-recomp)
- [LAS Recomp](https://github.com/peiikai/LASRecomp)
- [MacNativeGames](https://github.com/ly05010419/MacNativeGames)
- [nxnative](https://github.com/seedds/nxnative)
- [Ryujinx](https://github.com/Ryujinx/Ryujinx)
- [deko3d](https://github.com/devkitPro/deko3d)
- [SDL 3](https://libsdl.org), [Dear ImGui](https://github.com/ocornut/imgui), [AMD FSR 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR), [AMD FSR 3](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), [Intel XeSS](https://github.com/intel/xess), [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator), [libopus](https://opus-codec.org), [Zstandard](https://github.com/facebook/zstd)
- [Switchbrew](https://switchbrew.org) and the Switch homebrew community

## License

The project's own code is under the [MIT License](LICENSE); third-party code keeps its own license
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The license grants nothing for Nintendo's game.

Astral Chain Recomp is an independent, non-commercial fan project, not affiliated with or endorsed by Nintendo.
*Astral Chain*, *PlatinumGames* and *Nintendo Switch* are trademarks of their respective owners.

## Building from source

[docs/building.md](docs/building.md) lists the toolchain and the steps. Everything the build writes stays under
`build\` and `generated\`, which git ignores; run `tools/check_content.py` before a commit.

The menu, overlay and keyboard texts are English in `runtime/ui_text.c` plus one gettext catalog per language,
`lang/<tag>.po`. `tools/build_game.py` updates the catalogs against the English, compiles them into `lang/*.mo`
(ignored by git) and copies those beside the executable, which reads them at start: a new string is an id in
`runtime/ui_text.h` and its English in `runtime/ui_text.c`, and a translation is an edit of a `.po` file. The details
are in [docs/building.md](docs/building.md), "Texts and translations".
