"""Run a build of the game headless on a copy of the user's profile with a backed-up save and scripted input.

The user's profile (--profile, else build/profile-<exe dir name>, build_game.py's default, else build/profile) is never written: each
run copies it to <exe dir>/prof-<tag>, swaps in the save backup, and deletes the copy afterwards. The log records how
many other astralchain.exe and ninja.exe processes ran, because timings taken next to them are not comparable.

    python tools/run_game.py build/v143 --save <backup> --seconds 15 --tag idle --env AC_TRACE=perf
    python tools/run_game.py build/v143 --save <backup> --script <input.txt> --seconds 18 --tag fight --quiet \
        --env AC_TRACE=perf,sched

    python tools/run_game.py build/v101 --fresh --script tools/input/new_game.txt --seconds 90 --tag new-game \
        --shot build/new-game.png                      # title to the motorway on an empty profile

<backup> is a folder of build/save-backup; --script defaults to tools/input/load_save.txt (title screen to the first
save slot), and a recording made with AC_INPUT_RECORD replays the same way.
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

ROOT = Path(__file__).resolve().parent.parent
SAVES = ROOT / 'build' / 'save-backup'
SLOTS = ROOT / 'build' / 'run-slots'
GPU_LOCK = ROOT / 'build' / '.gpu-run.lock'
LOCK_OFFSET = 1 << 16   # Windows byte-range lock sits past the holder text, so other processes can still read it
EXE = 'astralchain.exe' if os.name == 'nt' else 'astralchain'
NINJA = 'ninja.exe' if os.name == 'nt' else 'ninja'
MAX_RUNS = int(os.environ.get('AC_MAX_RUNS', '3'))   # more parallel runs exhaust device memory (16 GB RX 9070)


def count(image):
    if os.name != 'nt':
        out = subprocess.run(['pgrep', '-x', image], capture_output=True, text=True).stdout
        return len(out.split())
    out = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {image}', '/NH'], capture_output=True, text=True).stdout
    return sum(line.lower().startswith(image) for line in out.splitlines())


def busy():
    return count(EXE) + count(NINJA)


def keep_in_background(proc):
    """Push the game window behind the user's windows as soon as it appears, for builds older than
    AC_BACKGROUND_WINDOW."""
    user32 = ctypes.windll.user32
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    wintypes.UINT)
    found = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def match(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == proc.pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    callback = enum_proc(match)
    while proc.poll() is None and not found:
        user32.EnumWindows(callback, 0)
        time.sleep(0.05)
    for hwnd in found:
        user32.SetWindowPos(hwnd, 1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)   # HWND_BOTTOM; NOSIZE | NOMOVE | NOACTIVATE


def take_slot():
    """Block until one of MAX_RUNS run slots is free and hold it (a locked file) until this process exits."""
    SLOTS.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(MAX_RUNS):
            f = open(SLOTS / f'slot{i}.lock', 'a+b')
            if try_lock(f):
                return f
            f.close()
        time.sleep(5)


def try_lock(f):
    """Take an exclusive OS lock on the open file without blocking; False if another process holds it."""
    try:
        if os.name == 'nt':
            f.seek(LOCK_OFFSET)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def holder_of(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace').strip() or 'unknown'
    except OSError:
        return 'unknown'


def take_gpu_lock(exe_dir, poll=2.0):
    """Block until this process holds the machine-wide measured-run lock (build/.gpu-run.lock) and return the open
    file. The OS drops the lock when the process exits or dies, so a stale file never blocks. The file text names the
    holder (pid and build path) for the waiters."""
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    f = open(GPU_LOCK, 'a+b')
    waiting = False
    while not try_lock(f):
        if not waiting:
            print(f'run_game: --quiet waits for the GPU run lock held by {holder_of(GPU_LOCK)}', flush=True)
            waiting = True
        time.sleep(poll)
    f.seek(0)
    f.truncate()
    f.write(f'pid {os.getpid()}, build {exe_dir}'.encode())
    f.flush()
    return f


def release_gpu_lock(f):
    if f is None:
        return
    try:
        if os.name == 'nt':
            f.seek(LOCK_OFFSET)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    f.close()


def profile_of(exe_dir, given=None):
    """The profile the build plays with: build_game.py makes build/profile-<version> the default of its exe. --profile
    names another one, which is copied the same way and never written."""
    if given:
        if given.is_dir():
            return given.resolve()
        sys.exit(f'run_game: no profile at {given}')
    for p in (ROOT / 'build' / f'profile-{exe_dir.name}', ROOT / 'build' / 'profile'):
        if p.is_dir():
            return p
    sys.exit(f'run_game: no profile at build/profile-{exe_dir.name} or build/profile')


def profile_copy(src, dest, save):
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    if save:
        users = [d for d in (dest / 'save').iterdir() if d.is_dir()]
        if len(users) != 1:
            sys.exit(f'run_game: expected one save directory in {dest / "save"}, found {len(users)}')
        shutil.rmtree(users[0])
        shutil.copytree(SAVES / save, users[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('exe_dir', type=Path, help='build directory holding astralchain.exe')
    ap.add_argument('--profile', type=Path, help='profile to copy; default: build/profile-<exe dir name>, else build/profile')
    ap.add_argument('--fresh', action='store_true',
                    help='start from an empty profile, no save and no settings (tools/input/new_game.txt)')
    ap.add_argument('--save', help='directory name under build/save-backup; default: the profile as it is')
    ap.add_argument('--script', type=Path, default=ROOT / 'tools' / 'input' / 'load_save.txt', help='AC_INPUT_SCRIPT')
    ap.add_argument('--seconds', type=float, required=True,
                    help='game time to run: the run ends once the world clock (game time since the save loaded) passed '
                         'this many seconds (shot is taken of that frame)')
    ap.add_argument('--tag', default='run', help='names the profile copy and the log')
    ap.add_argument('--shot', type=Path, help='PNG of the last frame; turns async pipeline builds off (boot.c)')
    ap.add_argument('--log', type=Path, help='default: <exe dir>/logs/<tag>.log')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='extra environment, repeatable')
    ap.add_argument('--quiet', action='store_true', help='wait until no other astralchain.exe or ninja.exe runs (benchmarks)')
    ap.add_argument('--show', action='store_true', help='open the window on top; by default it opens behind other windows')
    a = ap.parse_args()

    exe_dir = a.exe_dir.resolve()
    log = (a.log or exe_dir / 'logs' / f'{a.tag}.log').resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    gpu = take_gpu_lock(exe_dir) if a.quiet else None
    try:
        slot = take_slot()
        if a.quiet:
            while busy():
                time.sleep(10)
        prof = exe_dir / f'prof-{a.tag}'
        if a.fresh:
            shutil.rmtree(prof, ignore_errors=True)
            prof.mkdir(parents=True)
        else:
            profile_copy(profile_of(exe_dir, a.profile), prof, a.save)

        env = dict(os.environ, AC_ASSUME_FOCUS='1', AC_INPUT='none', AC_INPUT_SCRIPT=str(a.script.resolve()))
        if not a.show:
            env['AC_BACKGROUND_WINDOW'] = '1'
        for kv in a.env:
            k, _, v = kv.partition('=')
            env[k] = v
        args = [str(exe_dir / EXE), '--profile', str(prof), '--seconds', str(a.seconds)]
        if a.shot:
            args += ['--shot', str(a.shot.resolve())]
        try:
            with open(log, 'w', encoding='utf-8', errors='replace') as f:
                f.write(f'run_game: others running before: {EXE} {count(EXE)}, {NINJA} {count(NINJA)}\n')
                f.flush()
                start = time.time()
                proc = subprocess.Popen(args, cwd=exe_dir, env=env, stdout=f, stderr=subprocess.STDOUT)
                if not a.show and os.name == 'nt':   # elsewhere AC_BACKGROUND_WINDOW alone
                    threading.Thread(target=keep_in_background, args=(proc,), daemon=True).start()
                rc = proc.wait()
                f.write(f'run_game: rc={rc} wall={time.time() - start:.0f}s; others running after: '
                        f'{EXE} {count(EXE)}, {NINJA} {count(NINJA)}\n')
        finally:
            shutil.rmtree(prof, ignore_errors=True)
    finally:
        release_gpu_lock(gpu)
    print(f'{log}: rc={rc}')
    sys.exit(rc)


if __name__ == '__main__':
    main()
