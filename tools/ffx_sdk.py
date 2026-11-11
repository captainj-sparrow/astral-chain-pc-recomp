"""AMD's FidelityFX SDK, pinned: the GLSL passes of the FSR 3.1.4 upscaler, compiled into SPIR-V for runtime/vk_fsr3.c
(FSR 3 Native AA, fsr3 = 1).

    .venv\\Scripts\\python tools\\ffx_sdk.py [--out DIR]

Fetches FILES at COMMIT (the SDK's v1.1.4 tag) from raw.githubusercontent.com once into toolchain/ffx-sdk/<commit>/,
each checked against its SHA-256, then compiles the seven passes vk_fsr3.c runs with glslc into DIR/fsr3_spv.h (default
build/fsr3), as `static const uint32_t fsr3_<pass>_spv[]`. tools/build_game.py calls header() before it runs ninja, so
a build fetches and compiles on its own; the header records a hash of everything it was made from (the pins, the
options, glslc's target) and is written again only when that changes, so glslc is needed once per change, not per build.

Nothing of the SDK goes into the repository: 28 files of AMD's (MIT), about 600 KB, unmodified, the same in every
checkout because the commit and every file's hash are pinned here. Moving to another SDK version means a new COMMIT and
new hashes (--pin prints them for the files at COMMIT) and a look at the passes' bindings in vk_fsr3.c. The releases ship
the SDK's LICENSE.txt (license_path) in licenses/.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_ROOT = (ROOT / 'build').resolve().parent   # toolchain/ is a junction in a worktree
REPO = 'GPUOpen-LibrariesAndSDKs/FidelityFX-SDK'
VERSION = '1.1.4'
COMMIT = 'c6efa6bf7f2027b3ec94f28578bb5965eabb9e55'   # tag v1.1.4
DEST = MAIN_ROOT / 'toolchain' / 'ffx-sdk' / COMMIT
GPU = 'sdk/include/FidelityFX/gpu'
PASSES_DIR = 'sdk/src/backends/vk/shaders/fsr3upscaler'
# Every file the passes include (glslc -MD over the seven passes with OPTIONS), and the licence.
FILES = {
    'LICENSE.txt': '82cf74fc23885107c7f69249ac3f1dac55898ff68320eb2c753984cae6b2d78a',
    f'{GPU}/ffx_common_types.h': 'a19560f837c09f0c990e98b5925b731135d908e372af41c5895cbfa70cc9e5c8',
    f'{GPU}/ffx_core.h': 'e439aa5f516b2d2ebf070ff1d24561f04fa5f00262a2e4f760764a7d07de7997',
    f'{GPU}/ffx_core_glsl.h': '01afe5b65549b2c0951b494ca4bf570ee2aa89a64344e4c7d98ef86e6613ab42',
    f'{GPU}/ffx_core_gpu_common.h': '0e3c50cb0b4592c051d87e028a1dfa54fbd097942ebb3146964db801dec095d0',
    f'{GPU}/ffx_core_gpu_common_half.h': 'd10466cbf027c8c23858a25f8e8cbbbf521d29f373db663c9354221651e9a498',
    f'{GPU}/ffx_core_portability.h': 'a7c3a1c213ec083bec401181036cffe7a222af2c430c0bab6bd8c14e83e8a567',
    f'{GPU}/spd/ffx_spd.h': 'd276f32ee18948cb053c991f3c8821f9664c93f12cb2483dd2978f333bca67b2',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_accumulate.h': '7212a028c6ff3ca4a76f9d62b7f64f5f5adce0f108067d8995c7bc795234afbc',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_callbacks_glsl.h': '875c2b8d8df93716fb46271886c768a9ef40c5ccc11ade39397952b4a4e54321',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_common.h': '4898aa014fef875d3644b9b3bed03f31a963349e17ad553198f87ae8af82cb3f',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_luma_instability.h': '82d13d21e161732e48b267d28906f067a8df306ead5aeeed65a1ef66b3e06003',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_luma_pyramid.h': 'fa6eeaf8197b214ee1853d73da3c66cd0c8fc24444a65237b0023bd68f10d887',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_prepare_inputs.h': 'acae0499b8805c363c105895afc28fc90e549b38c8bffdcbec1cf9e2f842abb0',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_prepare_reactivity.h': 'ddf7a6d9cd6e357328e972e0b654da977edcb2e337c992b06d6256eeae339e03',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_reproject.h': 'd2d01ea2f59658e5494414448f11435879bdac871c3ad7c84b951248bc223702',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_resources.h': '10c6f7619a6c1ff9983fdff9bac565789b1cda52f97ef60b491498223e7b7097',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_sample.h': 'e8de656c71b15537ddd891b7485492b7829e6aec1dea96256fbc246c9a7c7366',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_shading_change.h': '13e772ff9bd32244fb990704055040f8dbb945c014cc6f0fdbf553188b300e7e',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_shading_change_pyramid.h': 'b1e94d7cbb1b7d6e4556e4a080577a62fde07efbc8379b52223d322ed4b75912',
    f'{GPU}/fsr3upscaler/ffx_fsr3upscaler_upsample.h': 'c51f3af61b43fd779a6685202886dbb5dab3bfea989f8b5060c8c9ab4a93207b',
    f'{PASSES_DIR}/ffx_fsr3upscaler_prepare_inputs_pass.glsl': 'bdf3b1ca0fdce968afb79f55d667960382139926eccdfdec46e08dbae3982657',
    f'{PASSES_DIR}/ffx_fsr3upscaler_luma_pyramid_pass.glsl': '3e637a702c2cc6b6a222df4327a4ed565f731c6190e958e0ffec5b43741f7ea4',
    f'{PASSES_DIR}/ffx_fsr3upscaler_shading_change_pyramid_pass.glsl': 'ce34582f454925499523d30d2619e19aefd8f1c091545c37d4ebd8feea91cf55',
    f'{PASSES_DIR}/ffx_fsr3upscaler_shading_change_pass.glsl': '246d24b76e292b7c8c5ba39dfc3a4dff1486701f5fccc6482284afac5a7daaf9',
    f'{PASSES_DIR}/ffx_fsr3upscaler_prepare_reactivity_pass.glsl': '176f6aa5f2f437487222c7fc057ce858795a6677d6ac3e937a1ce36521c6ddfc',
    f'{PASSES_DIR}/ffx_fsr3upscaler_luma_instability_pass.glsl': '423345d36c811237af7a41ef5e6220638f5159952e90fa084aa952ddd267581b',
    f'{PASSES_DIR}/ffx_fsr3upscaler_accumulate_pass.glsl': '73b5be6db479475559d2374957b2854aa6d7f7ddadba76ef5e4ed7f248ea9564',
}
# The passes in the order vk_fsr3.c records them (its P_* enum and SPV table follow this order).
PASSES = ('prepare_inputs', 'luma_pyramid', 'shading_change_pyramid', 'shading_change', 'prepare_reactivity',
          'luma_instability', 'accumulate')
# AMD's options (the SDK's CMakeCompileFSR3UpscalerShaders.txt) for what vk_fsr3.c feeds them: linear scene light (the
# light target, HDR colour input), motion vectors at the render size and without the jitter, depth near 0 to far 1
# (Astral Chain tests LEQUAL), the reference Lanczos in the reprojection (no LUT to bind), no sharpening in the
# accumulate pass, and no half precision, which not every GPU the port runs on has in its shaders.
OPTIONS = ('FFX_GPU=1', 'FFX_GLSL=1', 'FFX_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2',
           'FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=0',
           'FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=1',
           'FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1',
           'FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0',
           'FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=0',
           'FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING=0')
TARGET = 'vulkan1.1'
WIN = os.name == 'nt'
GLSL_BIN = Path(r'C:\msys64\ucrt64\bin')   # Windows (MSYS2 shaderc, spirv-tools); elsewhere glslc comes from PATH
HEADER = 'fsr3_spv.h'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def url_of(path, commit=COMMIT):
    return f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}'


def fetch():
    """DEST with every FILE in it, downloaded when missing or not the pinned file. Exits on a download or hash error."""
    for path, want in FILES.items():
        dst = DEST / path
        if dst.is_file() and sha256(dst.read_bytes()) == want:
            continue
        try:
            data = urllib.request.urlopen(url_of(path), timeout=60).read()
        except OSError as e:
            sys.exit(f'ffx_sdk: fetching {url_of(path)} failed ({e}); FSR 3 needs these files once '
                     f'(or copy the SDK v{VERSION} files into {DEST})')
        if sha256(data) != want:
            sys.exit(f'ffx_sdk: {url_of(path)}: sha256 {sha256(data)}, pinned {want}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return DEST


def license_path():
    """The SDK's licence text, fetched when missing: the releases ship it as FidelityFX-SDK-LICENSE.txt."""
    return fetch() / 'LICENSE.txt'


def recipe_hash():
    """What the header is made from: the pinned files, the passes, the options and the target."""
    h = hashlib.sha256()
    for part in (COMMIT, *sorted(f'{p} {s}' for p, s in FILES.items()), *PASSES, *OPTIONS, TARGET):
        h.update(part.encode() + b'\0')
    return h.hexdigest()


def tool(name):
    return str(GLSL_BIN / f'{name}.exe') if WIN else name


def spirv_words(src, sdk):
    gpu = sdk / GPU
    with tempfile.TemporaryDirectory() as tmp:
        spv = Path(tmp) / 'out.spv'
        cmd = [tool('glslc'), '-O', f'--target-env={TARGET}', '-fshader-stage=compute', '-I', str(gpu),
               '-I', str(gpu / 'fsr3upscaler'), *(f'-D{o}' for o in OPTIONS), '-o', str(spv), str(src)]
        try:
            subprocess.run(cmd, check=True)
            subprocess.run([tool('spirv-val'), '--target-env', TARGET, str(spv)], check=True)
        except FileNotFoundError:
            sys.exit(f'ffx_sdk: {cmd[0]} not found; FSR 3 needs glslc and spirv-val (MSYS2 UCRT64 packages '
                     f'mingw-w64-ucrt-x86_64-shaderc and spirv-tools; Debian/Ubuntu glslc and spirv-tools)')
        except subprocess.CalledProcessError as e:
            sys.exit(f'ffx_sdk: {Path(src).name}: {Path(e.cmd[0]).name} failed ({e.returncode})')
        data = spv.read_bytes()
    return [int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4)]


def header(out):
    """out/fsr3_spv.h, made (fetching and compiling) unless the one there was made from the same recipe."""
    dst = Path(out) / HEADER
    stamp = f'// recipe sha256 {recipe_hash()}'
    if dst.is_file() and stamp in dst.read_text(encoding='utf-8').splitlines()[:3]:
        return dst
    sdk = fetch()
    lines = [f'// Generated by tools/ffx_sdk.py from AMD FidelityFX SDK v{VERSION} (MIT) @ {COMMIT}; do not edit.',
             stamp, '#pragma once', '#include <stdint.h>']
    total = 0
    for p in PASSES:
        words = spirv_words(sdk / PASSES_DIR / f'ffx_fsr3upscaler_{p}_pass.glsl', sdk)
        total += len(words)
        lines.append(f'static const uint32_t fsr3_{p}_spv[] = {{')
        lines += ['    ' + ', '.join(f'0x{w:08x}' for w in words[i:i + 8]) + ',' for i in range(0, len(words), 8)]
        lines.append('};')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(dst)
    print(f'ffx_sdk: {dst}: FSR 3.1.4 passes, {total * 4 // 1024} KiB of SPIR-V')
    return dst


def pin(commit):
    """The FILES entries for the same paths at another commit, for moving the pin."""
    for path in FILES:
        data = urllib.request.urlopen(url_of(path, commit), timeout=60).read()
        print(f"    '{path}': '{sha256(data)}',")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'fsr3', help='where fsr3_spv.h goes')
    ap.add_argument('--pin', metavar='COMMIT', help='print the FILES hashes at another commit and exit')
    args = ap.parse_args()
    if args.pin:
        return pin(args.pin)
    header(args.out)


if __name__ == '__main__':
    main()
