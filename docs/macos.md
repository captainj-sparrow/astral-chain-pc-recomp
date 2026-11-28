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

- `runtime/platform/` currently has Windows and Linux backends. Darwin needs its
  own memory tracking, thread interruption and fiber implementations; Linux's
  userfaultfd/PAGEMAP_SCAN cannot be used directly.
- The build selects x86 flags and Linux linking on every non-Windows host.

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
