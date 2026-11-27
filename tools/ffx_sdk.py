"""AMD's FidelityFX SDK, pinned: the GLSL passes of the FSR 3.1.4 upscaler, compiled into SPIR-V for runtime/vk_fsr3.c
(FSR 3 Native AA and upscaling, upscaling = fsr3), and those of FSR 3.1.4 frame generation (the optical flow and the frame
interpolation) for runtime/vk_ffxfg.c (frame_generation with the FSR 3 method, vk_dlssg.c).

    .venv\\Scripts\\python tools\\ffx_sdk.py [--out DIR]

Fetches FILES at COMMIT (the SDK's v1.1.4 tag) from raw.githubusercontent.com once into toolchain/ffx-sdk/<commit>/,
each checked against its SHA-256, then compiles the seven passes vk_fsr3.c runs with glslc into DIR/fsr3_spv.h (default
build/fsr3), as `static const uint32_t fsr3_<pass>_spv[]` (HDR colour input) and `fsr3_<pass>_ldr_spv[]` (LDR), and
the seventeen frame generation passes vk_ffxfg.c runs into DIR/ffxfg_spv.h, as `ffxfg_<pass>_spv[]` with each pass's
descriptor bindings as glslc left them (`ffxfg_<pass>_bind[]`: binding, kind; read back from the SPIR-V with
spirv-dis, so vk_ffxfg.c's push descriptor layouts hold exactly what the shader declares). tools/build_game.py calls
header() before it runs ninja, so a build fetches and compiles on its own; each header records a hash of everything it
was made from (the pins, the options, glslc's target) and is written again only when that changes, so glslc is needed
once per change, not per build.

Nothing of the SDK goes into the repository: 68 files of AMD's (MIT), about 1 MB, unmodified, the same in every
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
# FSR 3 frame generation (vk_ffxfg.c): the optical flow's and the frame interpolation's passes and every file they
# include beyond FILES (glslc -MD over the seventeen passes with FG_OPTIONS). The debug view pass is left out.
FG_DIR = 'sdk/src/backends/vk/shaders'
FG_FILES = {
    f'{GPU}/frameinterpolation/ffx_frameinterpolation.h': '9b9280a86a00850e2930fd6e19442d2984aa36964dd22a077a328a573f7b4830',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h': 'e1e132729a80fe8e812c83d527056a17d62e7d815a79511465bfd1b0f8483095',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_common.h': '3e0e8a8109e4ba19441216e1400d52370a37ab7a1520029d5456368df24a22cb',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid.h': '78b97b96fda75b0aa7c84dde5594eb63883e43c39dc098b8d05c648674a4b16e',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_compute_inpainting_pyramid.h': 'dd29e6171180b949342585bdfe70f8f8aec631058b19773432150f48da73d45b',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_disocclusion_mask.h': '5e7fe2cb7a60e838c6f0d2e009db2dfa5e09b2c2813e62726795cfb983bf0ea5',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_game_motion_vector_field.h': '8f998e6c2a71e70bf54dc02db63cd64df6131ee6508b5dde702cce5320d467b4',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_inpainting.h': '42cc6f6b11f98d4c2a827834da9707c43f5bc817f190aeb3b1a071c5b8ca1149',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_optical_flow_vector_field.h': '950e5736eb49d0886f9c5703fa0bccea1210c8834170243f29b3d8a07cd1a8c3',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_reconstruct_dilated_velocity_and_previous_depth.h': '80a8cc026cc9352185ffb7366fbc275f3957b1fef2e1fdf9345509e1188c5ff4',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_reconstruct_previous_depth.h': '64bdc3e1b6e5278955c1ef66ed65918e11e05b1452d7411c6601ea7bd77edbcc',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_resources.h': 'aa9daea4d768cdc014de54bd42890673ffdef2d7d8cf796c986c2b873ad98ac9',
    f'{GPU}/frameinterpolation/ffx_frameinterpolation_setup.h': 'b8c785a0b09a8f148f12a19ca928f5a8178e82429c1f32e176ef300ae5e87493',
    f'{GPU}/opticalflow/ffx_opticalflow_callbacks_glsl.h': 'a6105db7ddfee06ab636dcd9a54e783d14e63980b844df64806f642eddce646b',
    f'{GPU}/opticalflow/ffx_opticalflow_common.h': '9d4b9466bf831e7d4a9c1c3e5e61af4977582c4e24439751eca8b866feadb33a',
    f'{GPU}/opticalflow/ffx_opticalflow_compute_luminance_pyramid.h': 'c5cda3dbbceffe27b9837952b04440a3095fb16fbe15092f8568b5b78f6c6d69',
    f'{GPU}/opticalflow/ffx_opticalflow_compute_optical_flow_v5.h': '8360c1840cb067b47485cbef4853392eae5f8826e9de17fa50af746502acb7a0',
    f'{GPU}/opticalflow/ffx_opticalflow_compute_scd_divergence.h': 'c254d8777507fccfaffdcaad304d5bd484473c0b19f38e35a3d05396537aa7b6',
    f'{GPU}/opticalflow/ffx_opticalflow_filter_optical_flow_v5.h': 'b4818d2e0ad597d2dcfee9c3d4f2b513f483cebb94098012ececc01a7c126a61',
    f'{GPU}/opticalflow/ffx_opticalflow_generate_scd_histogram.h': '0c941d937d2dcb87e423016fb41fe24c39e6b4884c74445e41a276cb282f2351',
    f'{GPU}/opticalflow/ffx_opticalflow_prepare_luma.h': '6f3cec87a8ad0d68c2d8bd78e8b0862eea7143e708580946eea85cfbfe5a32a4',
    f'{GPU}/opticalflow/ffx_opticalflow_resources.h': 'cbb2a75305c6bb45d8efd2dd5b4e471529afed9005a8003eed39a330f2cfa079',
    f'{GPU}/opticalflow/ffx_opticalflow_scale_optical_flow_advanced_v5.h': '8e1f3d2e2b98a25e836e1dadb97fcbba5b0b126a2113f1551be45cdc1e696d4d',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid_pass.glsl': '0637a0d5ecf54a5e1b4161dc48db334c1cdf33a6cdd475b5262254d74d8dd506',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_compute_inpainting_pyramid_pass.glsl': '0641760d7ccc676c4cfbc8a5b958435933eb523bb736fcdd42d270d05e5cc180',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_disocclusion_mask_pass.glsl': '5e8a4eafecc3c4c72ba076e2c45ea066f9abf083f962a55bd62b3a4bdbdec9d1',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_game_motion_vector_field_pass.glsl': '30ba7806d9451d89f6568b670eee6f1c6cc0f3fb5d98a89f6fafb75f51595b37',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_inpainting_pass.glsl': '0e9e87f6f323c1e7284ea3dd86022035be1db37953a9b988feee57a45d4ffb38',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_optical_flow_vector_field_pass.glsl': '30279e09bba6c44051fd0e7d5c7e5e0606ff472c168985fba8250ab1eb6a874e',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_pass.glsl': 'd7787ed6241e4f5f3951997abf6a35389dd9f215923306c81e8c4b3977c6cd0f',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_reconstruct_and_dilate_pass.glsl': '4728c67aeaf2b480668e2a54a5c0acd2780303ccfe86f3056c150bc97a7f52d7',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_reconstruct_previous_depth_pass.glsl': 'b5fc5291656ed9770e22216fd90dddd354f17891b4ec84d300642c0584556cb5',
    f'{FG_DIR}/frameinterpolation/ffx_frameinterpolation_setup_pass.glsl': '23f1bd8bb68060c2a5f95cd0261bbc42dce74054690e884bdda33a3aea7aa57e',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_compute_luminance_pyramid_pass.glsl': '09c878c33ddac648cff4669a8544016e233a5d6784f9ab08fb39085f23a26d3d',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_compute_optical_flow_advanced_pass_v5.glsl': '43adca846366f513cd65f818de14147cca58f0534fc8f1513b7db2c3f0611bcf',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_compute_scd_divergence_pass.glsl': 'cb01c895ac97f986bcff9174574b2e7a37a913333f505f80f3d161b719a6bbfa',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_filter_optical_flow_pass_v5.glsl': 'a313c2174790efc245ee722c2899f3743fef96ede28709ab636626930b66d4b5',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_generate_scd_histogram_pass.glsl': 'aa831f7f27d7f49ed2da9476b2ad9e0daad79ea6ed59a5166c9ca5ba3ebc2d97',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_prepare_luma_pass.glsl': '4ce523b2c9596565f836cc1975404aedca0f3341f62d16316ddf9e4e4260218d',
    f'{FG_DIR}/opticalflow/ffx_opticalflow_scale_optical_flow_advanced_pass_v5.glsl': '82637bbf53fe960034e972fc00d0621dfbca84d91f608c737e2a6d74252fdf77',
}
# vk_ffxfg.c's passes (its FG_* enum follows this order): its name in the header, and the SDK's pass file
FG_PASSES = (
    ('of_prepare_luma', 'opticalflow/ffx_opticalflow_prepare_luma_pass'),
    ('of_luma_pyramid', 'opticalflow/ffx_opticalflow_compute_luminance_pyramid_pass'),
    ('of_scd_histogram', 'opticalflow/ffx_opticalflow_generate_scd_histogram_pass'),
    ('of_scd_divergence', 'opticalflow/ffx_opticalflow_compute_scd_divergence_pass'),
    ('of_search', 'opticalflow/ffx_opticalflow_compute_optical_flow_advanced_pass_v5'),
    ('of_filter', 'opticalflow/ffx_opticalflow_filter_optical_flow_pass_v5'),
    ('of_scale', 'opticalflow/ffx_opticalflow_scale_optical_flow_advanced_pass_v5'),
    ('fi_reconstruct_and_dilate', 'frameinterpolation/ffx_frameinterpolation_reconstruct_and_dilate_pass'),
    ('fi_setup', 'frameinterpolation/ffx_frameinterpolation_setup_pass'),
    ('fi_reconstruct_previous_depth', 'frameinterpolation/ffx_frameinterpolation_reconstruct_previous_depth_pass'),
    ('fi_game_motion_vector_field', 'frameinterpolation/ffx_frameinterpolation_game_motion_vector_field_pass'),
    ('fi_game_vector_field_pyramid',
     'frameinterpolation/ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid_pass'),
    ('fi_optical_flow_vector_field', 'frameinterpolation/ffx_frameinterpolation_optical_flow_vector_field_pass'),
    ('fi_disocclusion_mask', 'frameinterpolation/ffx_frameinterpolation_disocclusion_mask_pass'),
    ('fi_interpolation', 'frameinterpolation/ffx_frameinterpolation_pass'),
    ('fi_inpainting_pyramid', 'frameinterpolation/ffx_frameinterpolation_compute_inpainting_pyramid_pass'),
    ('fi_inpainting', 'frameinterpolation/ffx_frameinterpolation_inpainting_pass'),
)
# AMD's options for what vk_ffxfg.c feeds them (the SDK's CMakeCompile*Shaders.txt and ffx_fsr3.cpp): motion vectors at
# the render size and without the jitter, depth near 0 to far 1, no half precision; the optical flow without its HDR
# option, as FSR 3 creates it (the back buffer's transfer function, sRGB or PQ, is a constant read at run time).
FG_OPTIONS = ('FFX_GPU=1', 'FFX_GLSL=1', 'FFX_HALF=0',
              'FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=1',
              'FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS=0',
              'FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH=0',
              'FFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0',
              'FFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0',
              'FFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=0',
              'FFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0',
              'FFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2',
              'FFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT=0')
FG_HEADER = 'ffxfg_spv.h'
# a binding's kind in ffxfg_<pass>_bind[] (vk_ffxfg.c's K_* enum): sampled image, storage image, uniform buffer, storage
# buffer, sampler
KIND_SAMPLED, KIND_STORAGE, KIND_UNIFORM, KIND_SSBO, KIND_SAMPLER = range(5)

# The passes in the order vk_fsr3.c records them (its P_* enum and SPV table follow this order).
PASSES = ('prepare_inputs', 'luma_pyramid', 'shading_change_pyramid', 'shading_change', 'prepare_reactivity',
          'luma_instability', 'accumulate')
# AMD's options (the SDK's CMakeCompileFSR3UpscalerShaders.txt) for what vk_fsr3.c feeds them: motion vectors at the
# render size and without the jitter, depth near 0 to far 1 (Astral Chain tests LEQUAL), the reference Lanczos in the
# reprojection (no LUT to bind), no sharpening in the accumulate pass, and no half precision, which not every GPU the
# port runs on has in its shaders. Each pass is compiled twice (VARIANTS): HDR colour input for linear scene light (the
# light target, Native AA; the composite with HDR output), LDR for the tonemapped composite an upscaled scene gives.
OPTIONS = ('FFX_GPU=1', 'FFX_GLSL=1', 'FFX_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0',
           'FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2',
           'FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=0',
           'FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1',
           'FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0',
           'FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=0',
           'FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING=0')
# variant suffix of the arrays: HDR colour input (fsr3_<pass>_spv) or LDR (fsr3_<pass>_ldr_spv)
VARIANTS = (('', 'FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=1'), ('_ldr', 'FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=0'))
TARGET = 'vulkan1.1'
WIN = os.name == 'nt'
GLSL_BIN = Path(r'C:\msys64\ucrt64\bin')   # Windows (MSYS2 shaderc, spirv-tools); elsewhere glslc comes from PATH
HEADER = 'fsr3_spv.h'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def url_of(path, commit=COMMIT):
    return f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}'


def fetch(files=None):
    """DEST with every FILE (and FG_FILE) in it, downloaded when missing or not the pinned file. Exits on a download or
    hash error."""
    for path, want in (files or {**FILES, **FG_FILES}).items():
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


def fg_recipe_hash():
    """What the frame generation header is made from: its pinned files, passes and options, and the target."""
    h = hashlib.sha256()
    for part in (COMMIT, *sorted(f'{p} {s}' for p, s in {**FILES, **FG_FILES}.items()),
                 *(f'{n} {f}' for n, f in FG_PASSES), *FG_OPTIONS, TARGET, 'bind v1'):
        h.update(part.encode() + b'\0')
    return h.hexdigest()


def recipe_hash():
    """What the header is made from: the pinned files, the passes, the options and the target."""
    h = hashlib.sha256()
    for part in (COMMIT, *sorted(f'{p} {s}' for p, s in FILES.items()), *PASSES, *OPTIONS,
                 *(f'{suffix} {o}' for suffix, o in VARIANTS), TARGET):
        h.update(part.encode() + b'\0')
    return h.hexdigest()


def tool(name):
    return str(GLSL_BIN / f'{name}.exe') if WIN else name


def spirv_words(src, sdk, variant, includes=('fsr3upscaler',), options=OPTIONS, bindings=False):
    """The pass compiled and validated, as SPIR-V words; with bindings, also its descriptor bindings (bindings_of)."""
    gpu = sdk / GPU
    with tempfile.TemporaryDirectory() as tmp:
        spv = Path(tmp) / 'out.spv'
        cmd = [tool('glslc'), '-O', f'--target-env={TARGET}', '-fshader-stage=compute', '-I', str(gpu),
               *(a for i in includes for a in ('-I', str(gpu / i))),
               *(f'-D{o}' for o in (*options, *((variant,) if variant else ()))), '-o', str(spv), str(src)]
        try:
            subprocess.run(cmd, check=True)
            subprocess.run([tool('spirv-val'), '--target-env', TARGET, str(spv)], check=True)
            dis = subprocess.run([tool('spirv-dis'), '--raw-id', str(spv)], check=True, capture_output=True,
                                 text=True).stdout if bindings else ''
        except FileNotFoundError:
            sys.exit(f'ffx_sdk: {cmd[0]} not found; FSR 3 needs glslc, spirv-val and spirv-dis (MSYS2 UCRT64 '
                     f'packages mingw-w64-ucrt-x86_64-shaderc and spirv-tools; Debian/Ubuntu glslc and spirv-tools)')
        except subprocess.CalledProcessError as e:
            sys.exit(f'ffx_sdk: {Path(src).name}: {Path(e.cmd[0]).name} failed ({e.returncode})')
        data = spv.read_bytes()
    words = [int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4)]
    return (words, bindings_of(dis, src)) if bindings else words


def bindings_of(dis, src):
    """The descriptor bindings a compiled pass kept (set 0), as sorted (binding, KIND_*), from spirv-dis --raw-id."""
    import re
    binding, var, ptr, types, block = {}, {}, {}, {}, {}
    for line in dis.splitlines():
        line = line.strip()
        if m := re.match(r'OpDecorate %(\d+) Binding (\d+)', line):
            binding[m[1]] = int(m[2])
        elif m := re.match(r'OpDecorate %(\d+) DescriptorSet (\d+)', line):
            if m[2] != '0':
                sys.exit(f'ffx_sdk: {Path(src).name}: a binding in set {m[2]}')
        elif m := re.match(r'OpDecorate %(\d+) (Block|BufferBlock)', line):
            block[m[1]] = m[2]
        elif m := re.match(r'%(\d+) = OpTypePointer (\w+) %(\d+)', line):
            ptr[m[1]] = (m[2], m[3])
        elif m := re.match(r'%(\d+) = (OpType\w+)(.*)', line):
            types[m[1]] = (m[2], m[3].split())
        elif m := re.match(r'%(\d+) = OpVariable %(\d+) (\w+)', line):
            var[m[1]] = m[2]
    out = []
    for v, b in binding.items():
        storage, t = ptr[var[v]]
        kind, args = types[t]
        if kind == 'OpTypeImage':
            k = KIND_SAMPLED if args[5] == '1' else KIND_STORAGE
        elif kind == 'OpTypeSampler':
            k = KIND_SAMPLER
        elif kind == 'OpTypeStruct':
            k = KIND_UNIFORM if storage == 'Uniform' and block.get(t) == 'Block' else KIND_SSBO
        else:
            sys.exit(f'ffx_sdk: {Path(src).name}: binding {b} of a kind vk_ffxfg.c does not bind ({kind})')
        out.append((b, k))
    return sorted(out)


def fg_header(out):
    """out/ffxfg_spv.h, made (fetching and compiling) unless the one there was made from the same recipe."""
    dst = Path(out) / FG_HEADER
    stamp = f'// recipe sha256 {fg_recipe_hash()}'
    if dst.is_file() and stamp in dst.read_text(encoding='utf-8').splitlines()[:3]:
        return dst
    sdk = fetch()
    lines = [f'// Generated by tools/ffx_sdk.py from AMD FidelityFX SDK v{VERSION} (MIT) @ {COMMIT}; do not edit.',
             stamp, '#pragma once', '#include <stdint.h>',
             '// FSR 3.1.4 frame generation (runtime/vk_ffxfg.c): each pass\'s SPIR-V and the descriptor bindings it kept,',
             '// {binding, kind}: 0 sampled image, 1 storage image, 2 uniform buffer, 3 storage buffer, 4 sampler']
    total = 0
    for name, path in FG_PASSES:
        words, binds = spirv_words(sdk / FG_DIR / f'{path}.glsl', sdk, None, includes=('frameinterpolation', 'opticalflow',
                                   'spd'), options=FG_OPTIONS, bindings=True)
        total += len(words)
        lines.append(f'static const uint32_t ffxfg_{name}_spv[] = {{')
        lines += ['    ' + ', '.join(f'0x{w:08x}' for w in words[i:i + 8]) + ',' for i in range(0, len(words), 8)]
        lines.append('};')
        lines.append(f'static const uint16_t ffxfg_{name}_bind[][2] = {{' +
                     ', '.join(f'{{{b}, {k}}}' for b, k in binds) + '};')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(dst)
    print(f'ffx_sdk: {dst}: FSR 3.1.4 frame generation passes, {total * 4 // 1024} KiB of SPIR-V')
    return dst


def header(out):
    """out/fsr3_spv.h and out/ffxfg_spv.h, each made (fetching and compiling) unless the one there was made from the
    same recipe."""
    fg_header(out)
    dst = Path(out) / HEADER
    stamp = f'// recipe sha256 {recipe_hash()}'
    if dst.is_file() and stamp in dst.read_text(encoding='utf-8').splitlines()[:3]:
        return dst
    sdk = fetch()
    lines = [f'// Generated by tools/ffx_sdk.py from AMD FidelityFX SDK v{VERSION} (MIT) @ {COMMIT}; do not edit.',
             stamp, '#pragma once', '#include <stdint.h>']
    total = 0
    for suffix, variant in VARIANTS:
        for p in PASSES:
            words = spirv_words(sdk / PASSES_DIR / f'ffx_fsr3upscaler_{p}_pass.glsl', sdk, variant)
            total += len(words)
            lines.append(f'static const uint32_t fsr3_{p}{suffix}_spv[] = {{')
            lines += ['    ' + ', '.join(f'0x{w:08x}' for w in words[i:i + 8]) + ',' for i in range(0, len(words), 8)]
            lines.append('};')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    tmp.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(dst)
    print(f'ffx_sdk: {dst}: FSR 3.1.4 passes, {total * 4 // 1024} KiB of SPIR-V')
    return dst


def pin(commit):
    """The FILES and FG_FILES entries for the same paths at another commit, for moving the pin."""
    for path in {**FILES, **FG_FILES}:
        data = urllib.request.urlopen(url_of(path, commit), timeout=60).read()
        print(f"    '{path}': '{sha256(data)}',")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'fsr3', help='where fsr3_spv.h and ffxfg_spv.h go')
    ap.add_argument('--pin', metavar='COMMIT', help='print the FILES hashes at another commit and exit')
    args = ap.parse_args()
    if args.pin:
        return pin(args.pin)
    header(args.out)


if __name__ == '__main__':
    main()
