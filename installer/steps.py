"""The six setup steps of docs/installer.md: check, extract, lift, compile, prepare, cleanup.

Extraction runs in this process through tools/extract.py (the package's sibling tools/, which ships with it); lift and
compile run the bundled Python on recomp/gen.py and tools/build_game.py, whose output goes to
build-cache/logs/<step>-<tool>.log and is read for progress. No Win32 call here: the host check is a small C program
built with the bundled clang (hostcheck.c). The port builds the game from the player's base game (1.0.0) and update NSP
(1.0.1): the update supplies the ExeFS and its BKTR section, which hactool merges with the base RomFS.

Install folder: exefs/, romfs/ (romfs.bin, index.json), profile/, astralchain.exe and its DLLs, romfs-index.bin,
ac_xlate.exe, astralchain.ini, astralchain-install.json, build-cache/ (gen/, build/, logs/, the status files and
rebuild.txt).
Temp folder (<temp>/<dest name>-<hash>): check/ (the ExeFS of both inputs and their manifests, the host check) and
logs/. A successful install deletes it, and the run logs move to build-cache/logs/. The generated C and the objects stay
in build-cache/gen and build-cache/build: the next lift rewrites only the chunks whose C changed, and Ninja recompiles
only those. A change of the mods' code cheats then costs the lift (about a minute) and the chunks it touches, not a full
compile.

build-cache/ also holds cheats.status (tools/cheats.py), exefs.status (tools/exefs_patches.py) and rebuild.txt, the
command that starts this setup in its rebuild mode on the install folder (installer/gui.py --rebuild), one argument per
line; astralchain.exe runs it at start when an enabled mod has code cheats the build lacks (runtime/cheats.c).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import ini
from . import lifeline
from .layout import EXE, SUPPORTED, running_appimage, version_tag
from .runner import InstallError, Step

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
try:
    import extract  # noqa: E402
    from extract import (DamagedFile, common_tickets, container_entries, identify_update, load_expected,  # noqa: E402
                         load_keys, merge_space, merge_update_romfs, nca_ctr_key, nca_inventory, program_nca)
    TOOLS_IMPORT_ERROR = None
except ImportError as exc:
    # A release folder whose tools/ is missing or partly copied: the check step names the missing files, so the setup
    # reports check.release_incomplete instead of stopping on an import before it reaches any step.
    extract = None
    DamagedFile = common_tickets = container_entries = identify_update = load_expected = None
    load_keys = merge_space = merge_update_romfs = nca_ctr_key = nca_inventory = program_nca = None
    TOOLS_IMPORT_ERROR = exc

GiB = 1 << 30
DEST_EXTRA = 4 * GiB      # beside the RomFS: shader and pipeline caches (about 2 GB), exefs, astralchain.exe, seed files
BUILD_NEED = 3 * GiB      # generated C (about 0.9 GB) and objects, in <install>/build-cache
TEMP_NEED = GiB // 2      # the ExeFS trees, manifests and host check of the version check
HOSTCHECK = Path(__file__).resolve().parent / 'hostcheck.c'
STAMP = '.install-key'
BASE_VERSION = '1.0.0'   # the base game the update is merged with
DEFAULT_VERSION = '1.0.1'   # the version a fresh install builds (SUPPORTED in installer/layout.py)


# ---- helpers

def file_sha256(path, progress=None):
    """SHA-256 hex of a file; progress(done, total) after each block."""
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, 'rb', buffering=0) as f:
        buf = bytearray(8 << 20)
        view = memoryview(buf)
        while n := f.readinto(buf):
            h.update(view[:n])
            done += n
            if progress:
                progress(done, total)
    return h.hexdigest()


def fingerprint(*parts):
    """SHA-256 over strings and the contents of files and directory trees (a missing path counts as missing)."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            if p.is_dir():
                for f in sorted(p.rglob('*')):
                    if f.is_file() and '__pycache__' not in f.parts:
                        h.update(f.relative_to(p).as_posix().encode() + b'\0' + bytes.fromhex(file_sha256(f)))
            elif p.is_file():
                h.update(p.name.encode() + b'\0' + bytes.fromhex(file_sha256(p)))
            else:
                h.update(b'missing\0' + p.name.encode())
        else:
            h.update(str(p).encode())
        h.update(b'\1')
    return h.hexdigest()


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def replace_file(src, dst):
    """Copy src over dst through a temporary name, so a cancelled copy never leaves half a file under dst's name."""
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True)
        return
    # run_tool starts each tool in a session of its own: it holds ninja, the compilers ninja puts in process groups of
    # their own, and the game
    lifeline.kill_session(proc.pid)
    proc.kill()


def linux_glibc():
    """(major, minor) of the C library this Linux process runs on, None elsewhere or when it is not glibc."""
    try:
        name = os.confstr('CS_GNU_LIBC_VERSION') if sys.platform.startswith('linux') else None
    except (ValueError, OSError):
        return None
    m = re.match(r'glibc (\d+)\.(\d+)', name or '')
    return (int(m.group(1)), int(m.group(2))) if m else None


# The bundled clang, lld and the cryptography wheel are built against glibc 2.34 (tools/make_release_linux.py
# GLIBC_FLOOR); on an older one they cannot start
GLIBC_NEEDED = (2, 34)


def linux_kernel():
    """(major, minor) of the running Linux kernel, None elsewhere."""
    if not sys.platform.startswith('linux'):
        return None
    m = re.match(r'(\d+)\.(\d+)', os.uname().release)
    return (int(m.group(1)), int(m.group(2))) if m else None


def run_tool(ctx, step, script, args, parse=None, cwd=None):
    """Run a tool script with the bundled Python. Its output goes to <install>/build-cache/logs/<step>-<script>.log and,
    line by line, to parse(line). Returns (exit code, log path).

    The log lives with the install, not in the temp folder, so a warning or an error that names it still points at the
    file after a successful run removed the temp folder."""
    logs = ctx.cache / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f'{step}-{Path(script).stem}.log'
    command = [str(ctx.layout.python), str(script), *map(str, args)]
    ctx.reporter.log('run ' + ' '.join(command[2:]))
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    with open(log, 'w', encoding='utf-8', newline='\n') as out:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd or ctx.layout.root,
                                env=env, text=True, encoding='utf-8', errors='replace', bufsize=1,
                                start_new_session=os.name != 'nt')
        lifeline.watch(proc.pid)
        try:
            for line in proc.stdout:
                out.write(line)
                if parse:
                    parse(line.rstrip())
            code = proc.wait()
        except BaseException:
            kill_tree(proc)
            raise
        finally:
            lifeline.unwatch(proc.pid)
    ctx.reporter.log(f'{Path(script).name}: exit code {code}; output in {log}')
    return code, log


def free_bytes(path):
    """Free bytes on the disk of path (its nearest existing parent) and that parent's device."""
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free, os.stat(p).st_dev


def size_of(paths):
    return sum(p.stat().st_size for p in paths if p.is_file())


def raise_if_damaged(error, name, path):
    """InstallError check.input_damaged when a reader's ValueError says the file's data fails its hashes or ends early."""
    if re.search(r'mismatch|truncat|bounds|short', str(error), re.IGNORECASE):
        raise InstallError('check.input_damaged', input=name, path=str(path), detail=str(error)) from None


def romfs_verified(romfs, expected):
    """Whether romfs/index.json records the expected size and hash and romfs.bin still has that size."""
    try:
        report = json.loads((romfs / 'index.json').read_text(encoding='utf-8'))
        return (report.get('romfs_size') == expected['romfs']['size']
                and report.get('romfs_sha256') == expected['romfs']['sha256']
                and (romfs / 'romfs.bin').stat().st_size == expected['romfs']['size'])
    except (OSError, ValueError):
        return False


def game_version(ctx):
    """The game version this install is built from: what the check identified, else the recorded one. An install made
    before update support records none (or 1.0.0) and is rebuilt from the update the player chooses."""
    return ctx.state.get('version', DEFAULT_VERSION)


def expected_game(ctx):
    """The expected hashes of the version this install builds: tools/data/expected-v<version>.json."""
    return load_expected()[game_version(ctx)]


def expected_base():
    """The expected hashes of the base game (1.0.0): the code modules the update is merged with."""
    return load_expected()[BASE_VERSION]


class Watch(threading.Thread):
    """Polls a measure() of bytes written while a long call runs and reports it as progress."""

    def __init__(self, reporter, step):
        super().__init__(daemon=True)
        self.reporter, self.step = reporter, step
        self.phase, self.measure, self.total = None, None, None
        self.stopped = threading.Event()

    def set(self, phase, measure=None, total=None):
        self.phase, self.measure, self.total = phase, measure, total
        self.reporter.progress(self.step, phase, 0 if total else None, total, 'bytes' if total else None, force=True)

    def run(self):
        while not self.stopped.wait(1.0):
            phase, measure, total = self.phase, self.measure, self.total
            if phase and measure:
                try:
                    done = min(measure(), total)
                except OSError:
                    continue
                self.reporter.progress(self.step, phase, done, total, 'bytes')

    def stop(self):
        self.stopped.set()


def requested_lang(ctx):
    """The language this run uses: --lang, else the one recorded for the install (Check canonicalizes it)."""
    return ini.canonical_language(getattr(ctx.options, 'lang', None) or ctx.state.get('lang'))


# ---- 1. check

UPDATE_TITLE_ID = '01007300020FA800'   # the update NSP of Astral Chain: the base id with bit 0x800 set


def base_misuse(records):
    """Name why a file chosen as the base game holds an update or a combined package, or return None.

    The Meta NCA (content type 1) carries the package's own title id: the update's id (...800) for an update NSP, and
    several Meta NCAs mean base, update and DLC repacked into one file.
    """
    metadata = [r['title_id'] for r, _ in records if r['content_type'] == 1]
    if len(metadata) > 1:
        return 'check.base_combined'
    if UPDATE_TITLE_ID in metadata:
        return 'check.base_is_update'
    return None


def update_misuse(records):
    """Name why a file chosen as the update holds a base game or a combined package, or return None."""
    metadata = [r['title_id'] for r, _ in records if r['content_type'] == 1]
    if len(metadata) > 1:
        return 'check.update_combined'
    if UPDATE_TITLE_ID not in metadata:
        return 'check.update_is_base'
    return None


class Check(Step):
    id = 'check'
    always = True

    def key(self, ctx):
        # The compile step fingerprints the CPU level, and every key is computed before a step runs: probe the host
        # here, once, so both steps see the same answer (the check step itself is always run). A failed probe is
        # reported by run(), where the error names this step.
        try:
            self.host(ctx)
        except (InstallError, OSError):
            # OSError covers a release folder without llvm/: the clang subprocess cannot start. run() then names the
            # missing files with check.release_incomplete.
            pass
        return ''

    def run(self, ctx):
        layout, rep, dest = ctx.layout, ctx.reporter, ctx.dest
        missing = layout.missing(getattr(ctx, 'cpu', 'v3'))
        if missing:
            raise InstallError('check.release_incomplete', paths=missing[:5])
        if TOOLS_IMPORT_ERROR is not None:
            raise InstallError('check.release_incomplete', paths=[f'tools ({TOOLS_IMPORT_ERROR})'])
        if dest.exists() and not dest.is_dir():
            raise InstallError('check.dest_not_folder', path=str(dest))
        own = (ctx.temp, *ctx.temp.parents)   # a temp folder inside the install folder (installer/cli.py) is ours
        if ctx.mode == 'install' and dest.is_dir() and any(p not in own for p in dest.iterdir()):
            raise InstallError('check.dest_not_empty', path=str(dest))
        lang = requested_lang(ctx)
        if not lang:
            raise InstallError('check.lang_missing')
        if lang not in ini.LANGUAGES:
            raise InstallError('check.lang_unknown', lang=lang, known=', '.join(ini.LANGUAGES))
        ctx.lang = lang
        ctx.temp.mkdir(parents=True, exist_ok=True)
        if game_version(ctx) not in SUPPORTED:
            # an install made before update support records 1.0.0 (or nothing): the game files are needed again to
            # rebuild it from the update, so the extraction cannot stay current
            ctx.plan(ctx.forced | {'extract'})

        rep.progress('check', 'host', force=True)
        self.host(ctx)
        rep.progress('check', 'verify', force=True)
        ctx.verify_kept('check')
        ncas = 0
        if ctx.will_run('extract'):
            rep.progress('check', 'inputs', force=True)
            ncas = self.inputs(ctx)
        self.space(ctx, ncas)
        if ctx.will_run('extract'):
            rep.progress('check', 'versions', force=True)
            self.versions(ctx)
        return {}

    def host(self, ctx):
        if getattr(ctx, 'host', None) is not None:
            return
        layout, work = ctx.layout, ctx.temp / 'check'
        glibc = linux_glibc()
        if glibc and glibc < GLIBC_NEEDED and not layout.dev:
            raise InstallError('check.glibc_old', version='.'.join(map(str, glibc)),
                               need='.'.join(map(str, GLIBC_NEEDED)))
        work.mkdir(parents=True, exist_ok=True)
        exe = work / ('hostcheck' + EXE)
        command = [str(layout.clang), '-O1', str(HOSTCHECK), '-o', str(exe)]
        if layout.deps:
            command += ['-I', str(layout.deps / 'include')]
        if layout.sysroot:
            command += [f'--sysroot={layout.sysroot}', '-fuse-ld=lld']
        if os.name != 'nt':
            command.append('-ldl')
        r = subprocess.run(command, capture_output=True, text=True, errors='replace')
        if r.returncode:
            ctx.reporter.log(r.stdout + r.stderr)
            raise InstallError('check.toolchain_broken', detail=(r.stderr or r.stdout).strip()[-400:])
        r = subprocess.run([str(exe)], capture_output=True, text=True, errors='replace')
        try:
            host = json.loads(r.stdout)
        except ValueError:
            raise InstallError('check.host_probe_failed', code=r.returncode) from None
        ctx.reporter.log(f'host: {r.stdout.strip()}')
        if not host['x86_64_v2']:
            raise InstallError('check.cpu_too_old')
        # v3 is the faster build; v2 runs on CPUs without AVX2/FMA (runtime/fma.c supplies the fused multiply-add).
        ctx.cpu = 'v3' if host['x86_64_v3'] else 'v2'
        if not host['loader']:
            raise InstallError('check.vulkan_missing')
        devices = host['devices']
        if not devices:
            raise InstallError('check.vulkan_no_device')
        if not any(tuple(d['api'][:2]) >= (1, 3) for d in devices):
            raise InstallError('check.vulkan_too_old',
                               devices=', '.join(f'{d["name"]} ({d["api"][0]}.{d["api"][1]})' for d in devices))
        kernel = linux_kernel()
        if kernel and kernel < (6, 7):
            # Without userfaultfd's asynchronous write protection nothing tracks the CPU's writes: the game copies whole
            # pools to the GPU at each submit and its texture checks see every page as changed (docs/building.md, Linux)
            ctx.reporter.warning('warn.kernel_old', version='.'.join(map(str, kernel)))
        ctx.host = host

    def inputs(self, ctx):
        """Open both files with the keys as the extraction will. Returns the bytes of both Program NCAs."""
        o = ctx.options
        for name in ('base', 'update', 'keys'):
            path = getattr(o, name)
            if not path:
                raise InstallError('check.inputs_needed')
            if not Path(path).is_file():
                raise InstallError('check.input_missing', input=name, path=str(path))
        try:
            keys = load_keys(Path(o.keys))
        except (ValueError, OSError, UnicodeDecodeError):
            raise InstallError('check.keys_unreadable') from None
        if 'header_key' not in keys:
            raise InstallError('check.keys_no_header_key')
        total = 0
        for name, is_update in (('base', False), ('update', True)):
            path = Path(getattr(o, name))
            with path.open('rb') as f:
                total += self.open_input(f, path, keys, name, is_update)
        del keys
        return total

    @staticmethod
    def open_input(f, path, keys, name, is_update):
        try:
            kind, entries = container_entries(f, path.stat().st_size)
        except ValueError as e:
            raise_if_damaged(e, name, path)
            raise InstallError('check.not_game_file', input=name) from None
        if is_update and kind != 'nsp':
            raise InstallError('check.update_not_nsp') from None
        try:
            tickets = common_tickets(f, entries)
        except ValueError as e:
            if 'Personalized' in str(e):
                raise InstallError('check.ticket_personalized', input=name) from None
            raise_if_damaged(e, name, path)
            raise InstallError('check.file_unreadable', input=name, detail=str(e)) from None
        try:
            records = nca_inventory(f, entries, keys)
        except ValueError as e:
            if 'NCA3' in str(e):
                raise InstallError('check.keys_wrong', input=name) from None
            raise_if_damaged(e, name, path)
            raise InstallError('check.file_unreadable', input=name, detail=str(e)) from None
        reason = update_misuse(records) if is_update else base_misuse(records)
        if reason:
            raise InstallError(reason)
        try:
            record, header = program_nca(records, update=is_update)
        except ValueError as e:
            if is_update:
                raise InstallError('check.update_not_astral', detail=str(e)) from None
            raise InstallError('check.base_not_astral', detail=str(e)) from None
        try:
            nca_ctr_key(header, keys, tickets)
        except ValueError as e:
            if 'ticket' in str(e):
                raise InstallError('check.ticket_missing', input=name) from None
            raise InstallError('check.keys_too_old', input=name,
                               generation=max(header[0x206], header[0x220])) from None
        return record['size']

    def space(self, ctx, ncas):
        dest_need = temp_need = 0
        if ctx.will_run('extract'):
            # the update is identified after this estimate: the larger RomFS of the supported ones, plus the two
            # Program NCAs the update merge copies into the install folder (about 10.5 GB for 1.0.1)
            versions = load_expected()
            dest_need += max(versions[v]['romfs']['size'] for v in SUPPORTED) + ncas
            romfs = ctx.dest / 'romfs' / 'romfs.bin'
            if romfs.is_file():   # a resumed extraction replaces it
                dest_need -= romfs.stat().st_size
        if ctx.will_run('extract'):
            dest_need += DEST_EXTRA
            temp_need += TEMP_NEED      # the ExeFS trees the version check extracts
        if ctx.will_run('lift') or ctx.will_run('compile'):
            dest_need += BUILD_NEED     # the generated C and the objects are in the install folder's build-cache
        dest_free, dest_dev = free_bytes(ctx.dest)
        temp_free, temp_dev = free_bytes(ctx.temp)
        ctx.reporter.log(f'space: install folder needs {dest_need / GiB:.1f} GiB of {dest_free / GiB:.1f} free, '
                         f'temp folder {temp_need / GiB:.1f} GiB of {temp_free / GiB:.1f}')
        if dest_dev == temp_dev:
            dest_need += temp_need
            temp_need = 0
        ctx.reporter.emit('space', dest=str(ctx.dest), need=dest_need, free=dest_free, temp=str(ctx.temp),
                          temp_need=temp_need, temp_free=temp_free, same_disk=dest_dev == temp_dev)
        for path, need, free in ((ctx.dest, dest_need, dest_free), (ctx.temp, temp_need, temp_free)):
            if need > free:
                raise InstallError('check.disk_space', path=str(path), need_gb=round(need / 1e9, 1),
                                   free_gb=round(free / 1e9, 1))

    def versions(self, ctx):
        """Extract both ExeFS into the temp folder (verifying every block), check the base game against the 1.0.0
        hashes and identify the update by the SHA-256 of its `main`.

        Only the code modules decide the version (main.npdm is the process's metadata, and dumps of the same build
        carry different ones). A differing code module is not a dead end: --accept-unknown-version carries on as a
        warning. An update that matches no supported version cannot be built from its patch table, so it needs the
        same explicit acceptance.
        """
        o, work = ctx.options, ctx.temp / 'check'
        accept = getattr(ctx.options, 'accept_unknown_version', False)
        for name, is_update in (('base', False), ('update', True)):
            out = work / name
            remove(out)
            path = Path(getattr(o, name))
            try:
                extract.extract_exefs(path, Path(o.keys), out / 'exefs', update=is_update)
            except DamagedFile as e:
                raise InstallError('check.input_damaged', input=name, path=str(e.path), detail=e.detail) from None
            except ValueError as e:
                raise_if_damaged(e, name, path)
                raise InstallError('check.file_unreadable', input=name, detail=str(e)) from None

        base_expected = expected_base()
        ok, problems = extract.exefs_ok(work / 'base' / 'exefs', base_expected)
        if not ok:
            if not accept:
                raise InstallError('check.base_wrong_version', modules=', '.join(problems))
            ctx.reporter.warning('warn.version_accepted', modules=', '.join(problems))
        if extract.sha256(work / 'base' / 'exefs' / 'main.npdm') != base_expected['exefs_sha256']['main.npdm']:
            ctx.reporter.warning('warn.npdm_differs', version=BASE_VERSION)

        versions = load_expected()
        try:
            found = extract.identify_update(work / 'update' / 'exefs', versions)
        except ValueError:
            found = None
        if found is not None and found['display_version'] not in SUPPORTED:
            if not accept:
                raise InstallError('check.update_wrong_version', found=found['display_version'],
                                   wanted=' / '.join(SUPPORTED))
            found = None
        if found is None:
            if not accept:
                raise InstallError('check.update_wrong_version', found='unknown', wanted=' / '.join(SUPPORTED))
            found = versions[DEFAULT_VERSION]
            ctx.reporter.warning('warn.update_accepted', version=found['display_version'], modules='main')
        ok, problems = extract.exefs_ok(work / 'update' / 'exefs', found)
        if not ok:
            if not accept:
                raise InstallError('check.update_wrong_version', found=found['display_version'],
                                   wanted=' / '.join(SUPPORTED))
            ctx.reporter.warning('warn.update_accepted', version=found['display_version'],
                                 modules=', '.join(problems))
        if extract.sha256(work / 'update' / 'exefs' / 'main.npdm') != found['exefs_sha256']['main.npdm']:
            ctx.reporter.warning('warn.npdm_differs', version=found['display_version'])

        version = found['display_version']
        if version != game_version(ctx):   # another update than the recorded one: every step builds again from it
            ctx.reporter.log(f'check: update {version} (recorded {game_version(ctx)})')
            ctx.state['version'] = version
            ctx.keys = {}
            ctx.plan(ctx.forced | {'extract', 'lift', 'compile'})
        ctx.state['version'] = version


# ---- 2. extract

class Extract(Step):
    id = 'extract'

    def key(self, ctx):
        # The output is whatever matches the expected hashes, whichever files and tools produced it.
        return fingerprint('extract-1', ctx.layout.expected(game_version(ctx)))

    def current(self, ctx, record):
        # Another base game or update file than the recorded one extracts again; the check then identifies it.
        for name in ('base', 'update'):
            path, known = getattr(ctx.options, name, None), ctx.state['inputs'].get(name)
            if path and known and Path(path).is_file():
                st = Path(path).stat()
                if (st.st_size, st.st_mtime_ns) != (known.get('size'), known.get('mtime_ns')) \
                        and file_sha256(Path(path)) != known.get('sha256'):
                    return False
        expected = expected_game(ctx)
        romfs = ctx.dest / 'romfs'
        if not extract.exefs_ok(ctx.dest / 'exefs', expected)[0] or not romfs_verified(romfs, expected):
            return False
        st = (romfs / 'romfs.bin').stat()
        return st.st_size == record.get('romfs_size') and st.st_mtime_ns == record.get('romfs_mtime_ns')

    def verify(self, ctx, record):
        image = ctx.dest / 'romfs' / 'romfs.bin'
        rep = ctx.reporter
        digest = file_sha256(image, lambda d, t: rep.progress('check', 'verify_romfs', d, t, 'bytes'))
        return digest == expected_game(ctx)['romfs']['sha256']

    def run(self, ctx):
        o, dest, rep = ctx.options, ctx.dest, ctx.reporter
        expected = expected_game(ctx)
        base, update, keys = Path(o.base), Path(o.update), Path(o.keys)
        inputs = {}
        for name, path in (('base', base), ('update', update)):
            known = ctx.state['inputs'].get(name, {})
            st = path.stat()
            if known.get('size') == st.st_size and known.get('mtime_ns') == st.st_mtime_ns and known.get('sha256'):
                inputs[name] = known
                continue
            digest = file_sha256(path, lambda d, t, n=name: rep.progress('extract', 'hash_' + n, d, t, 'bytes'))
            inputs[name] = {'name': path.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': digest}
        ctx.state['inputs'] = inputs
        ctx.save()

        dest.mkdir(parents=True, exist_ok=True)
        for stale in dest.glob('update-merge-*'):   # a killed merge leaves its NCA copies behind
            remove(stale)
        rep.progress('extract', 'exefs', force=True)
        staged = ctx.temp / 'check' / 'update' / 'exefs'
        if not staged.is_dir():
            remove(ctx.temp / 'check' / 'update')
            try:
                extract.extract_exefs(update, keys, staged, update=True)
            except DamagedFile as e:
                raise InstallError('check.input_damaged', input='update', path=str(e.path), detail=e.detail) from None
            except ValueError as e:
                raise_if_damaged(e, 'update', update)
                raise InstallError('check.file_unreadable', input='update', detail=str(e)) from None
        remove(dest / 'exefs')
        shutil.move(str(staged), str(dest / 'exefs'))
        ok, problems = extract.exefs_ok(dest / 'exefs', expected)
        if not ok:
            if not getattr(ctx.options, 'accept_unknown_version', False):
                raise InstallError('extract.exefs_mismatch', version=game_version(ctx), modules=', '.join(problems))
            # The check already warned about the same files; the copy here is what the check read.
        if extract.sha256(dest / 'exefs' / 'main.npdm') != expected['exefs_sha256']['main.npdm']:
            rep.warning('warn.npdm_differs', version=game_version(ctx))

        romfs = dest / 'romfs'
        if self.id in ctx.forced:
            remove(romfs)   # the full hash failed: index.json still claims the extraction's hash
        if not romfs_verified(romfs, expected):
            remove(romfs)
            rep.progress('extract', 'copy', force=True)
            watch = Watch(rep, 'extract')
            watch.start()
            try:
                copy_total = merge_space(base, update, keys)

                def phase(name):
                    if name == 'copy':
                        watch.set('copy', lambda: size_of(dest.glob('update-merge-*/*.nca')), copy_total)
                    elif name == 'merge':
                        watch.set('merge', lambda: size_of(dest.glob('update-merge-*/result/romfs.bin')),
                                  expected['romfs']['size'])
                    else:
                        watch.set(name)

                extract.merge_update_romfs(base, update, keys, ctx.layout.hactool, romfs, ctx.temp / 'logs',
                                           expected['romfs']['size'], phase=phase)
            except DamagedFile as e:
                raise InstallError('check.input_damaged', input='base' if e.path == base else 'update',
                                   path=str(e.path), detail=e.detail) from None
            except ValueError as e:
                raise InstallError('extract.romfs_merge_failed', detail=str(e),
                                   log=str(ctx.temp / 'logs' / f'merge_romfs-{dest.name}.log')) from None
            finally:
                watch.stop()
        report = json.loads((romfs / 'index.json').read_text(encoding='utf-8'))
        if report['romfs_size'] != expected['romfs']['size'] or report['romfs_sha256'] != expected['romfs']['sha256']:
            if not getattr(ctx.options, 'accept_unknown_version', False):
                raise InstallError('extract.romfs_mismatch', version=game_version(ctx), size=report['romfs_size'],
                                   sha256=report['romfs_sha256'][:16])
            rep.warning('warn.data_accepted', size=report['romfs_size'])
        (dest / 'profile').mkdir(exist_ok=True)
        st = (romfs / 'romfs.bin').stat()
        return {'romfs_size': st.st_size, 'romfs_mtime_ns': st.st_mtime_ns}


# ---- 4. lift

EMITTED = re.compile(r'^emitted (\d+)/(\d+) chunks')
DISCOVERED = re.compile(r'^(\w+): \d+ functions \(.*discovery')


CHEAT_PLACES = ('cheats', 'atmosphere/contents/0100F2C0115B6000/cheats', 'contents/0100F2C0115B6000/cheats',
                '0100F2C0115B6000/cheats')


EXEFS_PLACES = ('exefs', 'atmosphere/contents/01007300020FA000/exefs', 'contents/01007300020FA000/exefs',
                '01007300020FA000/exefs')


def exefs_files(ctx):
    """The ExeFS patch files of the mods folder (.ips and .pchtxt in each mod's exefs places and its
    atmosphere/exefs_patches/<name>/, as tools/exefs_patches.py finds them), for the lift key."""
    mods = ctx.dest / 'mods'
    if not mods.is_dir():
        return []
    out = []
    for mod in (m for m in mods.iterdir() if m.is_dir()):
        folders = [mod / w for w in EXEFS_PLACES if (mod / w).is_dir()]
        patches = mod / 'atmosphere' / 'exefs_patches'
        folders += [d for d in patches.iterdir() if d.is_dir()] if patches.is_dir() else []
        out += [f for d in folders for f in d.iterdir() if f.is_file() and f.suffix.lower() in ('.ips', '.pchtxt')]
    return sorted(out)


def exefs_status(ctx):
    """The build's ExeFS patch status file (tools/exefs_patches.py --status), which runtime/mods.c reads."""
    return ctx.dest / 'build-cache' / 'exefs.status'


class Lift(Step):
    id = 'lift'
    transient = True

    def key(self, ctx):
        return fingerprint('lift-5', ctx.keys['extract'], ctx.layout.recomp, ctx.layout.patches(game_version(ctx)),
                           ctx.layout.tools / 'exefs_patches.py', *exefs_files(ctx))

    def current(self, ctx, record):
        gen = ctx.cache / 'gen'
        stamp = gen / STAMP
        if not ((gen / 'layout.json').is_file() and stamp.is_file() and stamp.read_text().strip() == ctx.keys['lift']):
            return False
        # A previous run can leave zeroed or truncated manifests behind (killed process, disk hiccup): the compile
        # would die on them, so treat them as not current and lift again.
        for mod in ('main', 'sdk'):
            try:
                entries = json.loads((gen / mod / 'manifest.json').read_text())
                entries = entries if isinstance(entries, list) else entries.get('chunks', entries.get('files'))
                if not entries:
                    return False
            except (OSError, ValueError):
                return False
        return True

    def run(self, ctx):
        # gen/ of the last run stays in the install folder: gen.py rewrites only the chunks whose C changed (and removes
        # the ones a smaller partition no longer has), so Ninja recompiles only those
        layout, rep, gen = ctx.layout, ctx.reporter, ctx.cache / 'gen'
        (gen / STAMP).unlink(missing_ok=True)
        rep.progress('lift', 'discover', force=True)

        def parse(line):
            m = EMITTED.match(line)
            if m:
                rep.progress('lift', 'emit', int(m.group(1)), int(m.group(2)), 'chunks')

        # The mods' ExeFS patch files become a patch set (tools/exefs_patches.py); its status file tells the runtime
        # what the build has. They are the player's downloads, so files the tool refuses cost those patches, not the
        # install: the game is built without them and the setup warns.
        plain = layout.patches(game_version(ctx))
        stages = [('exefs_patches.py', 'patches-exefs.txt', exefs_status(ctx), exefs_files(ctx), 'warn.exefs_skipped')]
        patches, skipped = plain, {}
        for tool, out, status, inputs, warn in stages:
            status.unlink(missing_ok=True)   # no status: the runtime treats every such file as not built
            if not inputs:
                continue
            code, log = run_tool(ctx, 'lift', layout.tools / tool,
                                 ['--exefs', ctx.dest / 'exefs', '--mods', ctx.dest / 'mods', '--patches', patches,
                                  '--out', ctx.temp / out, '--status', status])
            if code:
                status.unlink(missing_ok=True)
                skipped[warn] = log
            else:
                patches = ctx.temp / out
        args = ['--exefs', ctx.dest / 'exefs', '--out', gen, '--version', version_tag(game_version(ctx))]
        code, log = run_tool(ctx, 'lift', layout.recomp / 'gen.py', args + ['--patches', patches], parse)
        if code and patches != plain:
            for tool, out, status, inputs, warn in stages:
                if inputs and warn not in skipped:
                    status.unlink(missing_ok=True)
                    skipped[warn] = log
            code, log = run_tool(ctx, 'lift', layout.recomp / 'gen.py', args + ['--patches', plain], parse)
        if code:
            raise InstallError('lift.failed', code=code, log=str(log))
        # The compile reads <mod>/manifest.json next: a zeroed or truncated file (killed run, disk hiccup) would
        # only fail there, so validate now and fail at this step with its log instead.
        for mod in ('main', 'sdk'):
            try:
                entries = json.loads((gen / mod / 'manifest.json').read_text())
                entries = entries if isinstance(entries, list) else entries.get('chunks', entries.get('files'))
                if not entries:
                    raise ValueError('no chunks')
            except (OSError, ValueError) as e:
                raise InstallError('lift.failed', code=0, log=str(log)) from e
        for warn, log in skipped.items():
            rep.warning(warn, log=str(log))
        (gen / STAMP).write_text(ctx.keys['lift'] + '\n')
        return {}


# ---- 5. compile

NINJA_LINE = re.compile(r'^\[(\d+)/(\d+)\] ')


def runtime_key(ctx):
    """Fingerprint of what astralchain.exe is built from besides the lifted code: the runtime, the toolchain and the build
    tools. Shader preparation depends on these, not on the code cheats a lift adds."""
    layout, cpu = ctx.layout, getattr(ctx, 'cpu', 'v3')
    try:
        clang = subprocess.run([str(layout.clang), '--version'], capture_output=True, text=True).stdout
    except OSError:
        # No clang in this folder: every key is computed before any step runs, so return a marker and let the check
        # step report the missing files instead of failing here.
        clang = 'no-clang'
    tools = [layout.tools / n for n in ('build_game.py', 'romfs_index.py', 'check_content.py')]
    return fingerprint('runtime-1', cpu, layout.runtime_lib_for(cpu), layout.runtime, layout.deps or 'system',
                       layout.sysroot or 'host', *tools, clang, *[d.name for d in layout.dlls])


class Compile(Step):
    id = 'compile'

    def key(self, ctx):
        return fingerprint('compile-4', ctx.keys['lift'], runtime_key(ctx))

    def current(self, ctx, record):
        exe = ctx.dest / ('astralchain' + EXE)
        if not exe.is_file() or not (ctx.dest / 'romfs-index.bin').is_file():
            return False
        if not (ctx.dest / ctx.layout.xlate.name).is_file():
            return False
        if not all((ctx.dest / d.name).is_file() for d in ctx.layout.dlls):
            return False
        return file_sha256(exe) == record.get('exe_sha256')

    def run(self, ctx):
        layout, rep, dest = ctx.layout, ctx.reporter, ctx.dest
        build = ctx.cache / 'build'
        phase = ['compile']

        def parse(line):
            m = NINJA_LINE.match(line)
            if m:
                rep.progress('compile', phase[0], int(m.group(1)), int(m.group(2)), 'objects')
            elif line.startswith('compile: '):
                phase[0] = 'link'
                rep.progress('compile', 'link', force=True)

        rep.progress('compile', 'compile', force=True)
        cpu = getattr(ctx, 'cpu', 'v3')
        lib = layout.runtime_lib_for(cpu)
        # The exe verifies every ExeFS file it loads against hashes baked into it (build_game.py --expected,
        # runtime/boot.c ldr_read_verified). They must be the files this install holds, not the reference dump: a
        # player's main.npdm legitimately differs from ours, and the version check above already compared the code.
        exefs = dest / 'exefs'
        hashes = {name: extract.sha256(exefs / name) for name in ('main', 'main.npdm', 'rtld', 'sdk')
                  if (exefs / name).is_file()}
        expected_install = ctx.temp / 'expected-install.json'
        expected_install.write_text(json.dumps({'exefs_sha256': hashes}, indent=2) + '\n', encoding='utf-8',
                                    newline='\n')
        code, log = run_tool(ctx, 'compile', layout.tools / 'build_game.py', [
            '--gen', ctx.cache / 'gen', '--out', build, '--toolchain', layout.llvm, '--cpu', cpu,
            *(['--deps', layout.deps] if layout.deps else []), *(['--sysroot', layout.sysroot] if layout.sysroot else []),
            '--runtime', layout.runtime, *(['--runtime-lib', lib] if lib else ()), '--ninja', layout.ninja,
            '--version', version_tag(game_version(ctx)), '--expected', expected_install,
            '--exefs', dest / 'exefs', '--romfs', dest / 'romfs',
            '--profile', dest / 'profile'], parse)
        exe = build / ('astralchain' + EXE)
        if code or not exe.is_file():
            raise InstallError('compile.failed', code=code, log=str(log))
        try:
            replace_file(exe, dest / exe.name)
        except PermissionError:
            raise InstallError('compile.exe_in_use', path=str(dest / exe.name)) from None
        replace_file(build / 'romfs-index.bin', dest / 'romfs-index.bin')
        # The translator translates the game's shaders at run time: without it next to the game, nothing is drawn.
        try:
            replace_file(layout.xlate, dest / layout.xlate.name)
            xlate_id = layout.xlate.with_suffix('.id')
            if xlate_id.is_file():
                replace_file(xlate_id, dest / xlate_id.name)
        except PermissionError:
            raise InstallError('compile.exe_in_use', path=str(dest / layout.xlate.name)) from None
        for dll in layout.dlls:
            replace_file(dll, dest / dll.name)
        # The controller button mods (controller-ui/, tools/make_prompts_folders.py): repacked ui.pkz overlays
        # next to the installed exe, picked up through AC_PROMPTS (auto by default). A failure only warns: the
        # game keeps the Switch prompts.
        prompts = layout.root / 'controller-ui'
        if prompts.is_dir():
            code, log = run_tool(ctx, 'compile', layout.tools / 'make_prompts_folders.py',
                                  ['--xbox-mod', prompts / 'xbox', '--playstation-mod', prompts / 'playstation',
                                   '--romfs', dest / 'romfs', '--out', dest])
            if code:
                rep.warning('warn.xbox_prompts_failed', code=code, log=str(log))
        return {'exe_sha256': file_sha256(dest / exe.name), 'cpu': cpu}


# ---- 6. prepare

TRANSLATED = re.compile(r'gfx_shader: prepare: (\d+) of (\d+) pairs')
# runtime/vk_pipeline.c lb_warm_prepared: variants the driver crashed on in an earlier run (tools/prepare_shaders.py
# runs the driver step again after such a crash)
SKIPPED = re.compile(r'prepare-shaders: (\d+) variants skipped: the graphics driver crashed')


class Prepare(Step):
    id = 'prepare'

    def key(self, ctx):
        layout = ctx.layout
        xlate_id = layout.xlate.with_suffix('.id')
        tools = [layout.tools / n for n in ('prepare_shaders.py', 'shader_census.py', 'pack_shader_cache.py')]
        # not the compile key: a rebuild for the mods' code cheats keeps the prepared shaders
        return fingerprint('prepare-2', ctx.keys['extract'], runtime_key(ctx), layout.xlate, xlate_id,
                           layout.shader_answers, *tools)

    def current(self, ctx, record):
        dest = ctx.dest
        xlate = dest / ctx.layout.xlate.name
        return ((dest / 'shader-prepared.bin').is_file() and (dest / 'shader-cache' / 'shader-cache.pack').is_file()
                and xlate.is_file() and file_sha256(xlate) == record.get('xlate_sha256'))

    def run(self, ctx):
        layout, rep, dest = ctx.layout, ctx.reporter, ctx.dest
        xlate = dest / layout.xlate.name
        if xlate.is_file() and file_sha256(xlate) != file_sha256(layout.xlate):
            # Another translator: its cache keys differ, so the old translations would only take space.
            for name in ('shader-cache', 'shader-prepared.bin'):
                remove(dest / name)
        # astralchain.exe starts ac_xlate.exe as a child while it prepares shaders; a preparation that was stopped from
        # outside leaves that child holding this file, and Windows then refuses to replace it.
        try:
            replace_file(layout.xlate, xlate)
        except PermissionError:
            raise InstallError('prepare.xlate_in_use', path=str(xlate)) from None
        xlate_id = layout.xlate.with_suffix('.id')
        if xlate_id.is_file():
            try:
                replace_file(xlate_id, xlate.with_suffix('.id'))
            except PermissionError:
                raise InstallError('prepare.xlate_in_use', path=str(xlate.with_suffix('.id'))) from None
        else:
            remove(xlate.with_suffix('.id'))

        skipped = [0]

        def parse(line):
            m = TRANSLATED.search(line)
            if m:
                rep.progress('prepare', 'translate', int(m.group(1)), int(m.group(2)), 'shaders')
            elif SKIPPED.search(line):
                skipped[0] = int(SKIPPED.search(line).group(1))
            elif 'variants' in line:
                rep.progress('prepare', 'translate', force=True)
            elif 'gfx_shader: prepare:' in line and 'translated' in line and 'programs' in line:
                rep.progress('prepare', 'libraries', force=True)

        rep.progress('prepare', 'seed', force=True)
        code, log = run_tool(ctx, 'prepare', layout.tools / 'prepare_shaders.py',
                             [dest, '--version', version_tag(game_version(ctx)), '--romfs', dest / 'romfs',
                              '--answers', layout.shader_answers], parse)
        if code or not (dest / 'shader-prepared.bin').is_file():
            raise InstallError('prepare.shaders_failed', code=code, log=str(log))
        if skipped[0]:
            rep.warning('warn.shaders_skipped', count=skipped[0], log=str(log))

        # One shader-cache.pack instead of four files per shader (~117,000): astralchain.exe reads the pack first.
        rep.progress('prepare', 'pack', force=True)
        code, log = run_tool(ctx, 'prepare', layout.tools / 'pack_shader_cache.py', [dest / 'shader-cache'])
        if code:
            raise InstallError('prepare.pack_failed', code=code, log=str(log))
        return {'xlate_sha256': file_sha256(xlate)}


# ---- 7. clean up

class Cleanup(Step):
    id = 'cleanup'
    always = True

    def run(self, ctx):
        rep, dest = ctx.reporter, ctx.dest
        rep.progress('cleanup', 'ini', force=True)
        try:
            ini.write(dest, ctx.lang, ctx.layout.ini_example)
        except OSError as e:
            raise InstallError('cleanup.ini_failed', detail=str(e)) from None
        ctx.state['lang'] = ctx.lang
        (dest / 'profile').mkdir(exist_ok=True)
        rep.progress('cleanup', 'temp', force=True)
        for stale in dest.glob('update-merge-*'):
            remove(stale)
        # The temp folder itself goes once the reporter closed its log (installer/cli.py finish_temp): the generated C,
        # the objects and the logs live in build-cache/.
        for name in ('check', 'shared-fonts', 'expected-install.json'):
            remove(ctx.temp / name)
        romfs_problem(dest / 'romfs', expected_game(ctx))
        write_rebuild_command(ctx)
        return {}


# Puts the release folder (the argument after -c) on sys.path and runs installer.gui with the rest. `-m installer.gui`
# finds the package only through the release Python's ._pth: the Linux release's Python and the dev-mode embeddable one
# lack that entry, and astralchain.exe starts the command in the install folder.
REBUILD_BOOT = ('import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); '
                "runpy.run_module('installer.gui', run_name='__main__', alter_sys=True)")


def rebuild_command(ctx):
    """The command that starts the setup window in its rebuild mode on this install folder (installer/gui.py
    --rebuild): pythonw beside the setup's Python on Windows, so no console opens. In a repository that is the Python
    running this setup: the tools' embeddable Python there has no tkinter. From an AppImage it is the AppImage file
    itself (its AppRun passes the arguments to the window): the folder it is mounted on, where its Python is, goes
    away when this setup ends."""
    layout = ctx.layout
    image = running_appimage(layout)
    if image:
        return [str(image), '--rebuild', str(ctx.dest)]
    python = Path(sys.executable) if layout.dev else layout.python
    if os.name == 'nt' and python.with_name('pythonw.exe').is_file():
        python = python.with_name('pythonw.exe')
    return [str(python), '-c', REBUILD_BOOT, str(layout.root), '--root', str(layout.root), '--rebuild', str(ctx.dest)]


def write_rebuild_command(ctx):
    """build-cache/rebuild.txt in the install folder: rebuild_command, one argument per line, for astralchain.exe."""
    cache = ctx.dest / 'build-cache'
    cache.mkdir(exist_ok=True)
    tmp = cache / 'rebuild.txt.tmp'
    tmp.write_text('\n'.join(rebuild_command(ctx)) + '\n', encoding='utf-8', newline='\n')
    os.replace(tmp, cache / 'rebuild.txt')


def finish_temp(ctx, log):
    """Delete the temp folder of a finished install, keeping its run logs in build-cache/logs/.

    installer/cli.py calls this after the reporter closed the log file (Windows cannot remove an open file) and names
    the final path in the end event, which the setup window shows."""
    logs = ctx.temp / 'logs'
    if logs.is_dir():
        log.parent.mkdir(parents=True, exist_ok=True)
        for p in logs.iterdir():
            target = log.parent / p.name
            if target.is_dir():
                remove(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(p), str(target))
    for _ in range(10):   # a scanner or an editor may still hold a file for a moment
        try:
            remove(ctx.temp)
            return
        except OSError:
            time.sleep(0.2)
    remove(ctx.temp)


def romfs_problem(romfs, expected):
    """Raise cleanup.romfs_unreadable unless romfs/ still holds the verified image. The game opens romfs.bin at start;
    an install whose romfs/ went empty and unreadable after the extraction verified it (a folder Windows was still
    deleting, #107) reported success and the game stopped at its first file read."""
    try:
        ok = romfs_verified(romfs, expected)
        if ok:
            with open(romfs / 'romfs.bin', 'rb') as f:
                f.read(1)
    except (OSError, ValueError):
        ok = False
    if not ok:
        raise InstallError('cleanup.romfs_unreadable', path=str(romfs))


def all_steps():
    return [Check(), Extract(), Lift(), Compile(), Prepare(), Cleanup()]
