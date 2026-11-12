"""Measure how well a cutscene's picture keeps to its sound on a machine that is slow, stutters or starves the sound.

The game's sound plays on the audio device's clock whatever the picture does, so a cutscene or a movie whose frames
come late has to make up for them or its picture ends up behind its sound. This runs one scene once per
scenario, each scenario switching on one of the runtime's test knobs (runtime/frame_rate.c AC_TEST_HITCH and
AC_TEST_FREEZE, runtime/ov_audout.c AC_TEST_AUDIO_STALL), and prints what the log's `cutscene run`, `movie` and
`audout` lines measured: picture minus sound, in ms, at the end of the scene and its range while it played.

    python tools/av_sync_test.py build/v101 movie
    python tools/av_sync_test.py build/v101 cutscene --profile build/profile-cutscene
    python tools/av_sync_test.py build/v101 movie --only slow,stutter-300

Scenes:
  movie     the opening movie (movie/ev0010.bk2, a cutscene held to 30 fps with speech) on a new empty profile;
            tools/input/opening_movie.txt starts it from the title screen.
  cutscene  an in-engine cutscene at 60 fps: --profile names a profile whose first save opens on one, and
            tools/input/load_save.txt (or --script) loads it.

A movie run takes --seconds of game time (world clock); a cutscene run ends as soon as the log has the line
of a cutscene run of 10 s or more, and after --seconds at the latest. The window opens behind the others unless
--show is given. Nothing else should run on the machine meanwhile.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'tools' / 'input'

# name, the scenes it applies to, the environment, what it stands for
SCENARIOS = [
    ('baseline', 'movie cutscene', {}, 'nothing held up'),
    ('slow', 'movie cutscene', {'AC_TEST_HITCH': '27'},
     'every cutscene frame 27 ms longer: about 26 fps, a machine too slow for the scene'),
    ('slow-jitter', 'movie cutscene', {'AC_TEST_HITCH': '5:0:45'}, 'every cutscene frame 5 to 50 ms longer'),
    ('very-slow', 'movie cutscene', {'AC_TEST_HITCH': '70'},
     'every cutscene frame 70 ms longer: about 12 fps, below the 15 fps a step of 2.0 covers'),
    ('stutter-80', 'movie cutscene', {'AC_TEST_HITCH': '80:1000'}, 'an 80 ms hitch every second'),
    ('stutter-300', 'movie cutscene', {'AC_TEST_HITCH': '300:3000'}, 'a 300 ms hitch every 3 s'),
    ('stutter-900', 'movie cutscene', {'AC_TEST_HITCH': '900:6000'}, 'a 900 ms hitch every 6 s'),
    ('stutter-2500', 'movie cutscene', {'AC_TEST_HITCH': '2500:12000'},
     'a 2.5 s hitch every 12 s: longer than the cutscene catches up'),
    ('sound-starved', 'movie cutscene', {'AC_TEST_AUDIO_STALL': '200:3000'},
     "the game's sound thread held up 200 ms every 3 s: the output runs dry"),
    ('freeze', 'movie cutscene', {'AC_TEST_FREEZE': '700:5000'},
     'picture and sound both stand still 700 ms every 5 s: a suspended process'),
    ('fps30', 'movie cutscene', {'AC_FPS': '30'}, 'the 30 fps mode, nothing held up: the game as shipped'),
    ('fps30-slow', 'movie cutscene', {'AC_FPS': '30', 'AC_TEST_HITCH': '27'},
     'the 30 fps mode, every cutscene frame 27 ms longer'),
    ('fps30-stutter', 'movie cutscene', {'AC_FPS': '30', 'AC_TEST_HITCH': '300:3000'},
     'the 30 fps mode, a 300 ms hitch every 3 s'),
    ('fps30-starved', 'movie cutscene', {'AC_FPS': '30', 'AC_TEST_AUDIO_STALL': '200:3000'},
     "the 30 fps mode, the game's sound thread held up 200 ms every 3 s"),
]

AV = r'picture minus sound ([+-]\d+) ms, ([+-]\d+)\.\.([+-]\d+) ms'
RUN = re.compile(r'frame rate: cutscene run (at 60|held to 30)( so far)?: (\d+) frames, ([\d.]+) s measured, ([\d.]+) s '
                 r'stepped, (\d+) catch-up frames, peak debt ([\d.]+) ms, ([\d.]+) ms left out(?:; ' + AV + ')?')
MOVIE = re.compile(r'frame rate: movie (?:slot (\d) at frame (\d+) of (\d+): |closed, \d+x\d+ at frame (\d+) of (\d+).*?; )'
                   r'(?:' + AV + r'[^;]*; )?(\d+) frames skipped to keep up, (\d+) delays too long gave up (\d+) ms'
                   r'(?:; ' + AV + ')?')
DRY = re.compile(r'audout: the output ran dry .*?; (\d+) ms in (\d+) gaps since the start')
VOICE = re.compile(r'frame rate: voice sync .*?audio minus picture ([+-][\d.]+) ms mean, ([+-][\d.]+)\.\.([+-][\d.]+) ms '
                   r'over (\d+) checks')
BAD = re.compile(r'FATAL|device lost')
ENDED = re.compile(rb'frame rate: cutscene run (?:at 60|held to 30): \d+ frames, (\d+)\.\d+ s measured')
KEPT_MS = 90   # picture and sound within this of each other count as kept (ITU-R BT.1359's acceptable lead of sound)


def read_log(path, scene):
    """What one run measured: its longest cutscene run, the longest movie, the dry output."""
    runs = {}        # (start line of the run) -> its last line's fields; a run's "so far" lines precede its end line
    current = None
    movie, dry, voice, bad = None, (0, 0), [], None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = RUN.search(line)
        if m:
            if current is None:
                current = len(runs)
            runs[current] = m
            if not m.group(2):
                current = None
            continue
        m = MOVIE.search(line)
        if m:
            total = int(m.group(3) or m.group(5))
            if movie is None or total >= movie[0]:
                movie = (total, m)
            continue
        m = DRY.search(line)
        if m:
            dry = (int(m.group(1)), int(m.group(2)))
        m = VOICE.search(line)
        if m:
            voice.append(m)
        if BAD.search(line) and bad is None:
            bad = line.strip()[:120]
    run = max(runs.values(), key=lambda m: int(m.group(3)), default=None)
    return run, movie[1] if movie else None, dry, voice, bad


def av_of(m, first):
    """(end, low, high) of a match's picture-minus-sound groups starting at group `first`, or None."""
    return tuple(int(m.group(first + i)) for i in range(3)) if m and m.group(first) else None


def verdict(av):
    if av is None:
        return '?'
    return 'kept' if abs(av[0]) <= KEPT_MS else 'LOST'


def one_run(a, scene, name, env_extra, log):
    prof = a.exe_dir / 'prof-avsync'
    if prof.exists():
        shutil.rmtree(prof)
    if a.profile:
        shutil.copytree(a.profile, prof)
    else:
        prof.mkdir()
    env = dict(os.environ, AC_ASSUME_FOCUS='1', AC_INPUT='none', AC_INPUT_SCRIPT=str(a.script), AC_FULLSCREEN='0',
               AC_OUTPUT_RES=a.res, AC_TRACE='perf')
    if not a.show:
        env['AC_BACKGROUND_WINDOW'] = '1'
    env.update(env_extra)
    try:
        with open(log, 'w', encoding='utf-8', errors='replace') as f:
            f.write(f'av_sync_test: scene {scene}, scenario {name}: {env_extra}\n')
            f.flush()
            proc = subprocess.Popen([str(a.exe_dir / a.exe), '--profile', str(prof), '--seconds', str(a.seconds)],
                                    cwd=a.exe_dir, env=env, stdout=f, stderr=subprocess.STDOUT)
            seen = 0
            while proc.poll() is None:
                time.sleep(0.5)
                if scene != 'cutscene':
                    continue
                with open(log, 'rb') as r:   # the lines since the last look, from the start of the first one
                    r.seek(seen)
                    text = r.read()
                seen += text.rfind(b'\n') + 1
                if any(int(m.group(1)) >= 10 for m in ENDED.finditer(text)):
                    time.sleep(1.0)   # the lines that follow the run's own reach the log
                    proc.terminate()
            proc.wait()
    finally:
        shutil.rmtree(prof, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('exe_dir', type=Path, help='build directory holding astralchain.exe')
    ap.add_argument('scene', choices=('movie', 'cutscene'))
    ap.add_argument('--profile', type=Path, help='the profile to copy for each run; default for movie: a new empty one')
    ap.add_argument('--script', type=Path, help='AC_INPUT_SCRIPT; default: opening_movie.txt or load_save.txt of tools/input')
    ap.add_argument('--seconds', type=float, help='game time of each run; default 60 (movie) or 90 (cutscene)')
    ap.add_argument('--only', help='comma-separated scenario names; default: all that apply to the scene')
    ap.add_argument('--res', default='1920x1080', help='AC_OUTPUT_RES of the runs')
    ap.add_argument('--exe', default='astralchain.exe' if os.name == 'nt' else 'astralchain')
    ap.add_argument('--show', action='store_true', help='open the window on top')
    ap.add_argument('--parse', action='store_true', help='run nothing: print the table from the logs of an earlier run')
    a = ap.parse_args()
    a.exe_dir = a.exe_dir.resolve()
    if a.scene == 'cutscene' and not a.profile:
        sys.exit('av_sync_test: the cutscene scene needs --profile, a profile whose first save opens on a cutscene')
    a.script = (a.script or INPUT / ('opening_movie.txt' if a.scene == 'movie' else 'load_save.txt')).resolve()
    a.seconds = a.seconds or (60 if a.scene == 'movie' else 90)
    only = set(a.only.split(',')) if a.only else None
    unknown = (only or set()) - {s[0] for s in SCENARIOS}
    if unknown:
        sys.exit(f'av_sync_test: unknown scenario {", ".join(sorted(unknown))}')
    logs = a.exe_dir / 'logs'
    logs.mkdir(exist_ok=True)

    print(f'{"scenario":<14} {"frames":>6} {"measured":>9} {"stepped":>8} {"picture - sound":>16} '
          f'{"range":>13} {"":>4}  {"movie":<34} sound dry')
    for name, scenes, env_extra, what in SCENARIOS:
        if a.scene not in scenes.split() or (only and name not in only):
            continue
        for _ in (0,):
            log = logs / f'avsync-{a.scene}-{name}.log'
            if not a.parse:
                one_run(a, a.scene, name, env_extra, log)
            if not log.exists():
                continue
            run, movie, dry, voice, bad = read_log(log, a.scene)
            av = av_of(run, 9)
            cols = [f'{name:<14}']
            if run:
                cols += [f'{run.group(3):>6}', f'{float(run.group(4)):>7.2f} s', f'{float(run.group(5)):>6.2f} s']
            else:
                cols += [f'{"-":>6}', f'{"-":>9}', f'{"-":>8}']
            cols += [f'{av[0]:>+13d} ms' if av else f'{"-":>16}', f'{av[1]:>+5d}..{av[2]:<+6d}' if av else f'{"-":>13}',
                     f'{verdict(av):>4}']
            if movie:
                mav = av_of(movie, 12) or av_of(movie, 6)
                text = f'{mav[0]:+d} ms ({mav[1]:+d}..{mav[2]:+d})' if mav else '-'
                text += f', {movie.group(9)} skipped'
                if int(movie.group(10)):
                    text += f', gave up {movie.group(11)} ms'
                cols.append(f' {text:<34}')
            else:
                cols.append(f' {"-":<34}')
            cols.append(f'{dry[0]} ms in {dry[1]}' if dry[1] else '-')
            print(' '.join(cols), flush=True)
            for v in voice:
                print(f'{"":<14} the game\'s own voice check: sound minus picture {v.group(1)} ms mean, '
                      f'{v.group(2)}..{v.group(3)} ms over {v.group(4)} checks')
            if bad:
                print(f'{"":<14} {bad}')
        if not a.parse:
            print(f'{"":<14} ({what})', flush=True)


if __name__ == '__main__':
    main()
