"""Progress lines for a launcher: `progress <0..1> <text>` on stdout in place of the JSON events (--progress-lines).

RecompLauncher (recomplauncher.com) installs and updates the port from the GitHub releases, beside the players who
download a release themselves; recomplauncher.json at the repository root, copied into each release
(tools/make_release.py build_sources), tells it how. Its format is the launcher's own: docs/DEVELOPERS.md and
docs/SETUP-INPUTS.md of elripalda/RecompLauncher.

    {install}/              the launcher's folder for the game: each release is unpacked over it (python/, llvm/,
                            tools/, Astral Chain-Setup.exe, recomplauncher.json; on Linux the AppImage alone)
      Astral Chain/         the built game (guilogic.GAME_FOLDER). It is in no release, so an update never unpacks
                            over it, and build-cache/rebuild.txt keeps pointing at {install}, which stays.

    inputs     the base game, the update and the keys (the launcher's shared switch.prod-keys, passed as a path)
    prepare    python -m installer ... --lang auto --dest "{install}/Astral Chain" --progress-lines, without a window
               (python/python.exe on Windows; the AppImage on Linux, whose AppRun passes "-m ..." to its Python). The
               launcher runs it again after each update with the files the player chose: the update mode of the
               setup, which skips the extraction and rebuilds the game.
    start      the built game, else the setup window, for a launcher too old for inputs and prepare. The window then
               sees RECOMPLAUNCHER=1 and starts with the same folder (guilogic.launcher_game_folder).
    settings   keys of Astral Chain/astralchain.ini as section/key, with the defaults of runtime/cfg.c
               (tests/installer_test.py ManifestTest keeps the two in step)
    saves      profile/save and profile/account.uid, which the save folder is named after

The launcher reads the prepare step's stdout only, shows the number as the bar and the text under it, and reports a
failure as the exit code plus the last text. So everything the player must read goes into a progress line, the error
included: stderr is not shown.

ProgressLines is the `out` of events.Reporter: it takes each JSON event line, folds it through the window's Model
(guilogic.py) and writes one line with the English text the window would show. The log file is written by the Reporter
as before.

The fraction covers the whole run. Each step has a share (WEIGHTS, roughly its part of the time of a first install); a
finished or skipped step counts in full, and the running step counts by its phase: the phases of a step in the
catalog's order split its share evenly, and a phase with a known amount advances inside its part. The fraction never
goes back, and reaches 1 only with a successful end.
"""
from .events import load_catalog
from .guilogic import DONE, RUNNING, SKIPPED, Catalog, Model, describe_event, parse_line

WEIGHTS = {'check': 3, 'extract': 35, 'lift': 12, 'compile': 28, 'prepare': 20, 'cleanup': 2}
OTHER_WEIGHT = 2      # a step this table does not name
BEFORE_END = 0.99     # the most a run shows before its end event


def step_phases(catalog):
    """{step: [phase, ...]} in the order the catalog lists them, which is the order the steps go through them."""
    phases = {}
    for key in catalog:
        kind, _, rest = key.partition('.')
        step, _, phase = rest.partition('.')
        if kind == 'phase' and phase:
            phases.setdefault(step, []).append(phase)
    return phases


class ProgressLines:
    def __init__(self, stream, cat=None):
        self.stream = stream
        self.cat = cat or Catalog('en')
        self.model = Model()
        self.phases = step_phases(load_catalog())
        self.best = 0.0
        self.text = ''

    # ---- the file interface events.Reporter writes to

    def write(self, text):
        for line in text.splitlines():
            event = parse_line(line)
            if event:
                self.event(event)
        return len(text)

    def flush(self):
        self.stream.flush()

    # ---- events -> lines

    def within(self, step):
        """0..1 of a running step, from its current phase."""
        m, phases = self.model, self.phases.get(step, [])
        if m.current != step or m.phase not in phases:
            return 0.0
        return (phases.index(m.phase) + (m.fraction() or 0.0)) / len(phases)

    def fraction(self):
        m = self.model
        weights = {s: WEIGHTS.get(s, OTHER_WEIGHT) for s in m.steps}
        done = 0.0
        for s, weight in weights.items():
            state = m.states.get(s)
            if state in (DONE, SKIPPED):
                done += weight
            elif state == RUNNING:
                done += weight * self.within(s)
        return done / (sum(weights.values()) or 1)

    def error_text(self):
        m = self.model
        id, params, _step = m.error
        text = self.cat(id, **params)
        if m.log_path and m.log_path not in text:
            text += ' ' + self.cat('ui.log_file', path=m.log_path)
        return text

    def event(self, e):
        m = self.model
        m.apply(e)
        kind = e['type']
        limit = BEFORE_END
        if kind == 'step' and e.get('state') == 'started':
            self.text = self.cat('step.' + e['step'])
        elif kind == 'progress':
            self.text = describe_event(self.cat, e)
        elif kind == 'error':
            self.text = self.error_text()
        elif kind == 'end':
            if m.result == 'ok':
                limit = self.best = 1.0
                self.text = self.cat('ui.result_updated' if m.mode == 'update' else 'ui.result_ok')
            elif m.result == 'cancelled':
                self.text = self.cat('ui.result_cancelled')
            else:   # the error is the last line, so it is the text the launcher leaves on screen
                self.text = self.error_text() if m.error else self.cat('ui.result_failed')
        elif kind not in ('step', 'plan'):
            return   # start, space, warning: nothing the bar or its text shows
        self.best = max(self.best, min(limit, self.fraction()))
        if not self.text:
            return   # the plan, before the first step: nothing to say yet
        self.stream.write(f'progress {self.best:.3f} {" ".join(self.text.split())}\n')
        self.stream.flush()
