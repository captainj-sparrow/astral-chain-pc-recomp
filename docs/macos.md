# macOS port work

Status: feasibility work, not yet a runnable game. Target: native Apple Silicon,
initially with the existing Vulkan renderer through MoltenVK and game version 1.0.1.

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

- `recomp/lifter.py` contains an ARM64 assembly mode, but `recomp/gen.py` currently
  forces the C mode, and generated chunks include the x86-specific `simd.h`.
- `runtime/platform/` currently has Windows and Linux backends. Darwin needs its
  own memory tracking, thread interruption and fiber implementations; Linux's
  userfaultfd/PAGEMAP_SCAN cannot be used directly.
- The build selects x86 flags and Linux linking on every non-Windows host.
- `tools/build_xlate.py` selects linux-x64 on non-Windows hosts; macOS needs an
  osx-arm64 build and verification of its native dependencies.

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
