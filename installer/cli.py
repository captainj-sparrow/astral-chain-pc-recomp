"""Command line: python -m installer --base FILE --update FILE --keys FILE --lang CODE --dest DIR
                                     [--root DIR] [--temp DIR] [--verify] [--progress-lines]

A first install needs every option: the base game (1.0.0) and the update NSP (1.0.1). On a folder with
astralchain-install.json (resume after a cancel or failure, or update with a newer release) only --dest is needed:
--lang defaults to the recorded language, and the game files are needed only when the extraction has to run again.
Prints JSON events on stdout (events.py). Exit code: 0 done, 1 failed, 2 bad command line, 3 cancelled (Ctrl+C or
Ctrl+Break).

A launcher runs the same command without a window (recomplauncher.json `prepare`, see progress.py):
--progress-lines prints `progress <0..1> <text>` lines instead of the events (progress.py) and ends the tools when
the launcher kills this process (lifeline.py), and --lang auto needs no language from the player (auto_language).
"""
import argparse
import hashlib
import signal
import sys
import time
from pathlib import Path

from . import ini
from . import layout as layouts
from . import lifeline
from . import state as statefile
from .events import Reporter
from .guilogic import system_game_language
from .progress import ProgressLines
from .runner import Context, run_steps
from .steps import all_steps, finish_temp

EXIT = {'ok': 0, 'failed': 1, 'cancelled': 3}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        Reporter().error('cli.usage', {'detail': message})
        sys.exit(2)


# The temp folder inside the install folder when the layout names none (the Linux release); the check step does not
# count it when it requires an empty install folder
TEMP_IN_DEST = '.setup-temp'


def temp_folder(base, dest):
    """One temp folder per install folder, so two installs never share generated files."""
    tag = hashlib.sha256(str(dest).lower().encode()).hexdigest()[:8]
    return base / f'{dest.name or "root"}-{tag}'


def auto_language(dest, state):
    """The language of --lang auto: the one astralchain.ini of the install folder has (the player may have changed it
    on the launcher's game page or in the file since the install), else None for the recorded one, else, on a first
    install, the game's language for this system's display language (English when the game has none for it)."""
    try:
        chosen = ini.canonical_language(ini.get_language((dest / 'astralchain.ini').read_text(encoding='utf-8-sig')))
    except (OSError, UnicodeDecodeError):
        chosen = None
    if chosen in ini.LANGUAGES:
        return chosen
    if state and state.get('lang'):
        return None
    return system_game_language()


def cancel(signum, frame):
    raise KeyboardInterrupt


def build_parser():
    ap = Parser(prog='python -m installer', description=__doc__.splitlines()[0])
    ap.add_argument('--base', type=Path, help='base game, XCI or NSP (version 1.0.0)')
    ap.add_argument('--update', type=Path, help='the update NSP (version 1.0.1)')
    ap.add_argument('--keys', type=Path, help='prod.keys')
    ap.add_argument('--lang', help='game language, a Switch language code (en-US, en-GB, ja, de, ru, ...), or auto: '
                                   "the install folder's own, else the system's")
    ap.add_argument('--dest', type=Path, required=True, help='install folder')
    ap.add_argument('--root', type=Path, help='release folder or repository (default: the folder above installer/)')
    ap.add_argument('--temp', type=Path, help='temp folder (default: temp/ of the release on Windows, .setup-temp in '
                                              'the install folder on Linux, build/install-temp in a repository)')
    ap.add_argument('--verify', action='store_true', help='hash every kept file, also those this release made')
    ap.add_argument('--accept-unknown-version', action='store_true',
                    help='carry on when the game files do not match the reference hashes (the window asks first); '
                         'the game may not run')
    ap.add_argument('--progress-lines', action='store_true',
                    help='print "progress <0..1> <text>" lines instead of JSON events, for a launcher')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = None
    if args.progress_lines:
        lifeline.arm()   # before the reporter and any tool: the POSIX watcher is a fork of this process
        if hasattr(sys.stdout, 'reconfigure'):
            # A pipe gets the system code page on Windows; the launcher reads UTF-8 and stops at a line that is not
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        out = ProgressLines(sys.stdout)

    # The window cancels with Ctrl+Break to the installer's process group (CREATE_NEW_PROCESS_GROUP), which also reaches
    # the tools it runs; Python's default for it ends the process without recording anything.
    for name in ('SIGBREAK', 'SIGTERM'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), cancel)
    layout = layouts.find(args.root)
    dest = args.dest.resolve()
    base = args.temp or layout.temp
    temp = temp_folder(base.resolve(), dest) if base else dest / TEMP_IN_DEST
    log = temp / 'logs' / time.strftime('install-%Y%m%d-%H%M%S.log')
    rep = Reporter(out=out, log_path=log)
    t0 = time.monotonic()
    try:
        state = statefile.load(dest)
    except statefile.StateError as e:
        rep.error(e.id, {'path': str(e.path)})
        rep.emit('end', result='failed', seconds=0)
        return 1
    if args.lang == 'auto':
        args.lang = auto_language(dest, state)
    ctx = Context(layout, dest, temp, rep, args, state=state, verify=args.verify)
    steps = all_steps()
    rep.emit('start', release=layout.release, mode=ctx.mode, dev=layout.dev, dest=str(dest), log=str(log),
             steps=[s.id for s in steps])
    result = run_steps(steps, ctx)
    seconds = round(time.monotonic() - t0, 1)
    # A finished install keeps its log in build-cache/logs/ and leaves no temp folder behind; the window shows
    # that path.
    final_log = dest / 'build-cache' / 'logs' / log.name if result == 'ok' else None
    rep.emit('end', result=result, seconds=seconds, **({'log': str(final_log)} if final_log else {}))
    rep.close()
    if final_log:
        try:
            finish_temp(ctx, final_log)
        except OSError:
            pass   # the install is done; a file another program still holds keeps the folder for the next run
        if base and args.temp is None:
            try:
                base.rmdir()   # the release's own temp folder, empty once the run's folder is gone
            except OSError:
                pass
    return EXIT[result]
