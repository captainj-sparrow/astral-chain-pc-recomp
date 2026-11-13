"""Where the installer finds what it runs: the one place that maps the release folder and a repository checkout.

Release (docs/installer.md "Release layout"), root = the unpacked folder:

    python/python.exe       runs every tool (numpy, lz4, cryptography, compression.zstd); its ._pth puts .., ../tools
                            and ../recomp on sys.path, so `python -m installer` imports from the release folder
    recomp/, tools/         the lifter and the scripts; recomp/patches-<v>.txt, tools/data/expected-v<version>.json for
                            each supported update (SUPPORTED)
    llvm/                   llvm-mingw (bin/clang.exe)
    runtime/                libacrt.a and libacrt-v2.a (x86-64-v3 / v2), the headers, install-side sources (romfs.c,
                            patches.c, ov_zstd.c) and astralchain.ico; runtime/deps/ holds
                            include/ + lib/ of Vulkan, SDL3, Opus, libvpx and VMA (build_game.py --deps)
    bin/                    ac_xlate.exe (+ ac_xlate.id), hactool.exe, ninja.exe, SDL3.dll, libopus-0.dll
    data/shader-answers.bin the shader variants players met (a shader-answers.bin recorded in play, tools/prepare_shaders.py)
    astralchain.ini.example
    VERSION                 the release version, one line
    temp/                   the default scratch folder (a successful install removes it; its run logs move to the
                            install folder's build-cache/logs/)

Linux release (the AppImage's folder, docs/installer.md "Linux release"): python/bin/python3 (python-build-standalone,
with tkinter), llvm/ with clang, lld and llvm-ar plus llvm/sysroot (the glibc headers and libraries the game links
against), runtime/deps/ with include/ and the static Opus, libvpx and zstd, bin/ without .exe and with libSDL3.so.0.

Dev mode, root = a repository checkout (no python/python.exe): build/py314-embed/py/python.exe (the embeddable
Python the release ships, with the same packages), else this Python; toolchain/llvm-mingw-*/ and toolchain/deps-mingw;
runtime/ as the sources with build/install-rtlib/libacrt.a and libacrt-v2.a (tools/build_game.py --make-runtime-lib
--out build/install-rtlib, once per --cpu); MSYS2 UCRT64 ninja.exe; toolchain/hactool; build/xlate/ac_xlate.exe; the
recorded shader variants in generated/; build/install-temp. The version is VERSION with -dev and the commit: 0.1.0-dev+973c626. On Linux,
dev mode takes clang, lld and ninja from PATH and the system's libraries (no deps folder, no SDL3 copy).
"""
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

WIN = os.name == 'nt'
EXE = '.exe' if WIN else ''
UCRT = Path(r'C:\msys64\ucrt64\bin')
SUPPORTED = ('1.0.1',)   # updates the installer builds; the check identifies the player's by hash


def version_tag(version):
    """'1.0.0' -> 'v0', '1.0.1' -> 'v101': the version name of build_game.py, recomp/gen.py and the patch tables."""
    return 'v0' if version == '1.0.0' else 'v' + version.replace('.', '')


@dataclass(frozen=True)
class Layout:
    root: Path
    dev: bool
    release: str
    python: Path          # runs the tools
    tools: Path
    recomp: Path
    llvm: Path            # llvm-mingw tree; on Linux the tree with bin/clang (a release's, or the system's /usr)
    deps: Path | None     # include/ and lib/ for the link; None: the system's (Linux dev mode)
    runtime: Path         # runtime headers and install-side sources
    runtime_lib: Path     # libacrt.a
    ninja: Path
    hactool: Path
    xlate: Path           # ac_xlate.exe; a ac_xlate.id beside it is copied too
    dlls: tuple = field(default_factory=tuple)
    shader_answers: Path | None = None   # the shader variants players met (tools/prepare_shaders.py)
    ini_example: Path | None = None
    temp: Path | None = None
    sysroot: Path | None = None       # Linux release: what the game links against, instead of the host's
    runtime_lib_v2: Path | None = None   # libacrt-v2.a: the runtime for CPUs without AVX2 (build_game.py --cpu v2)

    @property
    def clang(self):
        return self.llvm / 'bin' / ('clang' + EXE)

    def runtime_lib_for(self, cpu):
        """The prebuilt runtime library for cpu (v2 or v3). None in dev mode when it was not built: the runtime is
        then compiled from the sources in runtime/ instead."""
        lib = self.runtime_lib_v2 if cpu == 'v2' else self.runtime_lib
        if lib and lib.is_file():
            return lib
        return None if self.dev else lib

    def patches(self, version):
        """The code patch table for recomp/gen.py."""
        return self.recomp / f'patches-{version_tag(version)}.txt'

    def expected(self, version):
        """The expected hashes of the update."""
        return self.tools / 'data' / f'expected-v{version}.json'

    def missing(self, cpu='v3'):
        """Paths the steps need that do not exist. cpu picks the runtime library the host will use."""
        lib = self.runtime_lib_for(cpu)
        need = [self.python, self.tools / 'extract.py', self.tools / 'build_game.py', self.tools / 'build_shaders.py',
                self.tools / 'check_content.py', self.tools / 'romfs_index.py', self.tools / 'exefs_patches.py',
                self.tools / 'prepare_shaders.py', self.tools / 'shader_census.py', self.tools / 'pack_shader_cache.py',
                self.expected('1.0.0'),
                self.recomp / 'gen.py', self.clang,
                *((self.deps / 'include', self.deps / 'lib') if self.deps else ()), self.runtime, *([lib] if lib else ()),
                self.ninja, self.hactool,
                self.xlate, *self.dlls,
                *(f(v) for v in SUPPORTED for f in (self.patches, self.expected))]
        if self.shader_answers:
            need.append(self.shader_answers)
        if self.sysroot:
            need.append(self.sysroot)
        return [str(p) for p in need if not p.exists()]


def release_python(root):
    return root / 'python' / ('python.exe' if WIN else 'bin/python3')


def is_release(root):
    return release_python(root).is_file()


def release_layout(root):
    version = root / 'VERSION'
    return Layout(
        root=root, dev=False,
        release=version.read_text(encoding='utf-8').strip() if version.is_file() else 'unknown',
        python=release_python(root),
        tools=root / 'tools', recomp=root / 'recomp',
        llvm=root / 'llvm', deps=root / 'runtime' / 'deps', runtime=root / 'runtime',
        runtime_lib=root / 'runtime' / 'libacrt.a',
        runtime_lib_v2=root / 'runtime' / 'libacrt-v2.a',
        ninja=root / 'bin' / ('ninja' + EXE), hactool=root / 'bin' / ('hactool' + EXE),
        xlate=root / 'bin' / ('ac_xlate' + EXE),
        # libxess.dll: Intel XeSS (runtime/vk_xess.c), Windows only
        dlls=(root / 'bin' / 'SDL3.dll', root / 'bin' / 'libopus-0.dll', root / 'bin' / 'libxess.dll') if WIN
        else (root / 'bin' / 'libSDL3.so.0',),
        shader_answers=root / 'data' / 'shader-answers.bin', ini_example=root / 'astralchain.ini.example',
        # Linux: None, the install folder's .setup-temp (installer/cli.py): the AppImage is mounted read-only, and on
        # a Steam Deck /home is often too full for the generated C while the install folder is on the SD card
        temp=root / 'temp' if WIN else None,
        sysroot=None if WIN else root / 'llvm' / 'sysroot')


def dev_layout(root):
    if not WIN:
        return dev_layout_posix(root)
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    embed = root / 'build' / 'py314-embed' / 'py' / ('python' + EXE)
    trees = sorted(p.parent.parent for p in (root / 'toolchain').glob('llvm-mingw-*/llvm-mingw-*/bin/clang' + EXE))
    deps = root / 'toolchain' / 'deps-mingw'
    semver = (root / 'VERSION').read_text(encoding='utf-8').strip()
    return Layout(
        root=root, dev=True, release=f'{semver}-dev+{commit or "unknown"}',
        python=embed if embed.is_file() else Path(sys.executable),
        tools=root / 'tools', recomp=root / 'recomp',
        llvm=trees[-1] if trees else root / 'toolchain' / 'llvm-mingw-missing', deps=deps,
        runtime=root / 'runtime', runtime_lib=root / 'build' / 'install-rtlib' / 'libacrt.a',
        runtime_lib_v2=root / 'build' / 'install-rtlib' / 'libacrt-v2.a',
        ninja=UCRT / ('ninja' + EXE), hactool=root / 'toolchain' / 'hactool' / ('hactool' + EXE),
        xlate=root / 'build' / 'xlate' / ('ac_xlate' + EXE),
        dlls=tuple(sorted((deps / 'bin').glob('*.dll'))),
        shader_answers=root / 'generated' / 'shader-answers.bin', ini_example=root / 'astralchain.ini.example',
        temp=root / 'build' / 'install-temp')


def dev_commit(root):
    return subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                          text=True).stdout.strip()


def dev_layout_posix(root):
    clang, ninja = shutil.which('clang'), shutil.which('ninja')
    semver = (root / 'VERSION').read_text(encoding='utf-8').strip()
    return Layout(
        root=root, dev=True, release=f'{semver}-dev+{dev_commit(root) or "unknown"}',
        python=Path(sys.executable), tools=root / 'tools', recomp=root / 'recomp',
        llvm=Path(clang).parent.parent if clang else root / 'toolchain' / 'clang-missing', deps=None,
        runtime=root / 'runtime', runtime_lib=root / 'build' / 'install-rtlib' / 'libacrt.a',
        runtime_lib_v2=root / 'build' / 'install-rtlib' / 'libacrt-v2.a',
        ninja=Path(ninja) if ninja else root / 'toolchain' / 'ninja-missing',
        hactool=root / 'toolchain' / 'hactool' / 'hactool', xlate=root / 'build' / 'xlate' / 'ac_xlate',
        shader_answers=root / 'generated' / 'shader-answers.bin', ini_example=root / 'astralchain.ini.example',
        temp=root / 'build' / 'install-temp')


def running_appimage(layout):
    """The AppImage file this setup runs from, or None. Its runtime sets APPIMAGE (the file) and APPDIR (where it
    mounted it) for AppRun; both are checked against the layout, so a checkout started from a shell that some other
    AppImage opened is not taken for one."""
    image, mount = os.environ.get('APPIMAGE'), os.environ.get('APPDIR')
    if WIN or layout.dev or not image or not mount:
        return None
    try:
        if Path(mount).resolve() != Path(layout.root).resolve() or not Path(image).is_file():
            return None
    except OSError:
        return None
    return Path(image)


def find(root=None):
    """The layout of root (default: the folder above this package)."""
    root = Path(root or Path(__file__).resolve().parent.parent).resolve()
    return release_layout(root) if is_release(root) else dev_layout(root)
