"""installer/ bookkeeping with fake steps: fresh install, resume after a failure or cancel, transient steps, update
with a newer release, astralchain.ini, the event stream, and what a launcher uses: progress lines, --lang auto and
recomplauncher.json. No game files, no tools; stdlib only.

Run: python tests/installer_test.py
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from installer import cli, ini, runner, state, steps  # noqa: E402
from installer import layout as layouts  # noqa: E402
from installer.events import Reporter  # noqa: E402
from installer.layout import Layout  # noqa: E402
from installer.progress import ProgressLines  # noqa: E402


class Fake(runner.Step):
    """A step whose output is <dir>/<id>.out holding its key; dir is the temp folder for a transient step."""

    def __init__(self, id, transient=False, always=False, previous=None):
        self.id, self.transient, self.always, self.previous = id, transient, always, previous

    def out(self, ctx):
        return (ctx.temp if self.transient else ctx.dest) / f'{self.id}.out'

    def key(self, ctx):
        upstream = ctx.keys[self.previous] if self.previous else ''
        return f'{self.id}:{ctx.options.inputs.get(self.id, "v1")}:{upstream}'

    def current(self, ctx, record):
        p = self.out(ctx)
        return p.is_file() and p.read_text() == ctx.keys[self.id]

    def verify(self, ctx, record):
        ctx.options.verified.append(self.id)
        return ctx.options.verify_ok.get(self.id, True)

    def run(self, ctx):
        o = ctx.options
        o.ran.append(self.id)
        if self.id in ctx.forced:
            o.forced.append(self.id)
        if self.id == 'check':
            ctx.verify_kept(self.id)
        failure = o.fail.pop(self.id, None)
        if failure == 'error':
            raise runner.InstallError('fake.failed', step=self.id)
        if failure == 'cancel':
            raise KeyboardInterrupt
        if self.id == 'cleanup':
            for p in ctx.temp.glob('*.out'):
                p.unlink()
            return {}
        p = self.out(ctx)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(ctx.keys[self.id])
        return {'note': self.id}


def fake_steps():
    return [Fake('check', always=True), Fake('extract'), Fake('lift', transient=True, previous='extract'),
            Fake('compile', previous='lift'), Fake('prepare', previous='compile'), Fake('cleanup', always=True)]


class Bed:
    """A dest and temp folder plus one run of the fake steps per call."""

    def __init__(self, base):
        self.dest, self.temp = base / 'dest', base / 'temp'
        self.temp.mkdir()

    def run(self, release='1.0', inputs=None, fail=None, verify_ok=None, verify=False):
        layout = Layout(root=Path('.'), dev=True, release=release, python=Path('python'), tools=Path('tools'),
                        recomp=Path('recomp'), llvm=Path('llvm'),
                        deps=Path('deps'), runtime=Path('rt'), runtime_lib=Path('lib'), ninja=Path('ninja'),
                        hactool=Path('hactool'), xlate=Path('xlate'))
        options = SimpleNamespace(inputs=inputs or {}, fail=dict(fail or {}), verify_ok=verify_ok or {}, ran=[],
                                  verified=[], forced=[])
        out = io.StringIO()
        ctx = runner.Context(layout, self.dest, self.temp, Reporter(out=out), options, state=state.load(self.dest),
                             verify=verify)
        mode = ctx.mode
        result = runner.run_steps(fake_steps(), ctx)
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        return SimpleNamespace(result=result, ran=options.ran, verified=options.verified, forced=options.forced,
                               events=events, mode=mode,
                               state=state.load(self.dest))

    @staticmethod
    def skipped(r):
        return {e['step']: e['reason'] for e in r.events if e['type'] == 'step' and e['state'] == 'skipped'}


ALL = ['check', 'extract', 'lift', 'compile', 'prepare', 'cleanup']


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bed = Bed(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_install_runs_everything_and_records_it(self):
        r = self.bed.run()
        self.assertEqual((r.result, r.mode, r.ran), ('ok', 'install', ALL))
        self.assertEqual(set(r.state['steps']), {'extract', 'lift', 'compile', 'prepare'})
        self.assertEqual(r.state['steps']['compile']['note'], 'compile')
        self.assertEqual(r.state['release'], '1.0')
        started = [e['step'] for e in r.events if e['type'] == 'step' and e['state'] == 'started']
        done = [e['step'] for e in r.events if e['type'] == 'step' and e['state'] == 'done']
        self.assertEqual((started, done), (ALL, ALL))

    def test_rerun_skips_finished_steps_and_unneeded_transient(self):
        self.bed.run()
        r = self.bed.run()
        self.assertEqual((r.result, r.mode, r.ran), ('ok', 'resume', ['check', 'cleanup']))
        self.assertEqual(self.bed.skipped(r), {'extract': 'current', 'lift': 'not_needed', 'compile': 'current',
                                               'prepare': 'current'})
        self.assertEqual(r.verified, [])   # same release: no full checks

    def test_resume_after_failure(self):
        r = self.bed.run(fail={'compile': 'error'})
        self.assertEqual((r.result, r.ran), ('failed', ['check', 'extract', 'lift', 'compile']))
        error = [e for e in r.events if e['type'] == 'error'][0]
        self.assertEqual((error['id'], error['step'], error['params']), ('fake.failed', 'compile', {'step': 'compile'}))
        self.assertEqual(set(r.state['steps']), {'extract', 'lift'})
        r = self.bed.run()
        # lift's output is still in the temp folder (no cleanup ran), so only compile and later run.
        self.assertEqual((r.result, r.mode, r.ran), ('ok', 'resume', ['check', 'compile', 'prepare', 'cleanup']))

    def test_resume_after_cancel_with_transient_output_gone(self):
        r = self.bed.run(fail={'prepare': 'cancel'})
        self.assertEqual(r.result, 'cancelled')
        self.assertNotIn('prepare', r.state['steps'])
        for p in self.bed.temp.glob('*.out'):
            p.unlink()
        r = self.bed.run()
        self.assertEqual(r.ran, ['check', 'prepare', 'cleanup'])
        self.assertEqual(self.bed.skipped(r)['lift'], 'not_needed')

    def test_failed_check_writes_nothing(self):
        r = self.bed.run(fail={'check': 'error'})
        self.assertEqual(r.result, 'failed')
        self.assertFalse(self.bed.dest.exists())

    def test_removed_output_runs_again(self):
        self.bed.run()
        (self.bed.dest / 'compile.out').unlink()
        r = self.bed.run()
        self.assertEqual(r.ran, ['check', 'lift', 'compile', 'cleanup'])

    def test_update_with_new_lifter_keeps_extraction(self):
        self.bed.run(release='1.0')
        r = self.bed.run(release='1.1', inputs={'lift': 'v2'})
        self.assertEqual((r.result, r.mode), ('ok', 'update'))
        self.assertEqual(r.ran, ['check', 'lift', 'compile', 'prepare', 'cleanup'])
        self.assertEqual(r.verified, ['extract'])   # kept from another release: full check
        self.assertEqual(r.state['steps']['extract']['release'], '1.0')
        self.assertEqual(r.state['steps']['compile']['release'], '1.1')
        self.assertEqual(r.state['release'], '1.1')

    def test_update_that_changes_nothing_skips_the_rebuild(self):
        self.bed.run(release='1.0')
        r = self.bed.run(release='1.1')
        self.assertEqual(r.ran, ['check', 'cleanup'])
        self.assertEqual(r.verified, ['extract', 'compile', 'prepare'])

    def test_update_with_changed_kept_files_extracts_again(self):
        self.bed.run(release='1.0')
        r = self.bed.run(release='1.1', verify_ok={'extract': False})
        self.assertEqual(r.ran, ['check', 'extract', 'cleanup'])
        self.assertEqual(r.forced, ['extract'])   # the step knows not to trust what it left behind
        warnings = [e for e in r.events if e['type'] == 'warning']
        self.assertEqual([(w['id'], w['params']) for w in warnings], [('state.output_changed', {'step': 'extract'})])

    def test_verify_option_checks_same_release(self):
        self.bed.run()
        r = self.bed.run(verify=True)
        self.assertEqual(r.verified, ['extract', 'compile', 'prepare'])

    def test_state_errors(self):
        self.bed.dest.mkdir()
        (self.bed.dest / state.NAME).write_text('{not json')
        with self.assertRaises(state.StateError) as e:
            state.load(self.bed.dest)
        self.assertEqual(e.exception.id, 'state.unreadable')
        (self.bed.dest / state.NAME).write_text(json.dumps({'format': 99, 'steps': {}}))
        with self.assertRaises(state.StateError) as e:
            state.load(self.bed.dest)
        self.assertEqual(e.exception.id, 'state.newer_format')


class ReleaseFilesTest(unittest.TestCase):
    def test_release_ships_every_tool_the_steps_run(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        import make_release
        text = (ROOT / 'installer' / 'steps.py').read_text(encoding='utf-8')
        used = set(re.findall(r"layout\.tools / '(\w+\.py)'", text))
        for names in re.findall(r"layout\.tools / n for n in \((.*?)\)", text, re.S):
            used |= set(re.findall(r"'(\w+\.py)'", names))
        self.assertEqual(used - set(make_release.TOOLS_FILES), set())


class LiftKeyTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dest = Path(tmp.name)
        layout = SimpleNamespace(recomp=self.dest / 'no-recomp', tools=self.dest / 'no-tools',
                                 patches=lambda version: self.dest / 'no-patches')
        self.ctx = SimpleNamespace(dest=self.dest, layout=layout, keys={'extract': 'x'}, state={'version': '1.4.3'})

    def key(self):
        return steps.Lift().key(self.ctx)

    def add_cheats(self, name, text, mod='Cheats', where='cheats'):
        folder = self.dest / 'mods' / mod / where
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)

    def test_extracting_main_keeps_the_key(self):
        before = self.key()
        (self.dest / 'exefs').mkdir()
        (self.dest / 'exefs' / 'main').write_bytes(bytes(0x48))
        self.assertEqual(self.key(), before)

    def test_exefs_patch_files_change_the_key(self):
        before = self.key()
        self.add_cheats('dynamic-fps.pchtxt', '@nsobid-277178B7DBA1B6D4', mod='Fps', where='exefs')
        added = self.key()
        self.assertNotEqual(added, before)
        self.add_cheats('277178B7DBA1B6D4.ips', 'PATCHEOF', mod='Ips', where='atmosphere/exefs_patches/x')
        both = self.key()
        self.assertNotEqual(both, added)
        self.add_cheats('main.npdm', 'not a patch', mod='Fps', where='exefs')
        self.assertEqual(self.key(), both)

    def test_status_and_other_files_keep_the_key(self):
        self.add_cheats('dynamic-fps.pchtxt', '@nsobid-277178B7DBA1B6D4', mod='Fps', where='exefs')
        before = self.key()
        (self.dest / 'build-cache').mkdir()
        (self.dest / 'build-cache' / 'exefs.status').write_text('written by lift')
        self.add_cheats('readme.md', 'not a patch file', mod='Fps', where='exefs')
        (self.dest / 'mods' / 'Fps' / 'romfs').mkdir()
        (self.dest / 'mods' / 'Fps' / 'romfs' / 'a.txt').write_text('a RomFS file')
        self.assertEqual(self.key(), before)


class IniTest(unittest.TestCase):
    EXAMPLE = '[display]\n# vk_present = mailbox\n\n[language]\n# locale = en-US\n# lang = en-US\n# x = 1\n'

    def test_example_line_is_activated(self):
        out = ini.set_language(self.EXAMPLE, 'ru')
        self.assertIn('\nlang = ru\n', out)
        self.assertNotIn('# lang', out)
        self.assertIn('# locale = en-US', out)

    def test_active_line_is_rewritten_and_the_rest_kept(self):
        text = '[display]\nvk_present = fifo\n[language]\n  LANG =  ja  \nfps60 = 0\n'
        out = ini.set_language(text, 'de')
        self.assertEqual(out, '[display]\nvk_present = fifo\n[language]\n  lang =  de\nfps60 = 0\n')
        self.assertEqual(ini.get_language(out), 'de')

    def test_missing_section_is_added(self):
        self.assertEqual(ini.set_language('fps60 = 0\n', 'fr'), 'fps60 = 0\n\n[language]\nlang = fr\n')
        self.assertEqual(ini.set_language('[language]\nlocale = de-DE\n', 'fr'),
                         '[language]\nlang = fr\nlocale = de-DE\n')

    def test_old_chinese_codes_are_written_as_the_canonical_ones(self):
        self.assertEqual(ini.canonical_language('zh-CN'), 'zh-Hans')
        self.assertEqual(ini.canonical_language('zh-TW'), 'zh-Hant')
        self.assertEqual(ini.canonical_language('ja'), 'ja')
        self.assertNotIn('zh-CN', ini.LANGUAGES)
        self.assertNotIn('zh-TW', ini.LANGUAGES)
        out = ini.set_language(self.EXAMPLE, 'zh-CN')
        self.assertEqual(ini.get_language(out), 'zh-Hans')
        self.assertEqual(ini.get_language(ini.set_language(out, 'zh-TW')), 'zh-Hant')

    def test_write_keeps_user_lines(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            example = d / 'astralchain.ini.example'
            example.write_text(self.EXAMPLE)
            path = ini.write(d / '', 'en-GB', example)
            self.assertEqual(ini.get_language(path.read_text()), 'en-GB')
            path.write_text(path.read_text().replace('# vk_present = mailbox', 'vk_present = fifo'))
            ini.write(d, 'ja', example)
            text = path.read_text()
            self.assertIn('vk_present = fifo', text)
            self.assertEqual(ini.get_language(text), 'ja')


class CleanupTest(unittest.TestCase):
    """The real Cleanup step on a finished install whose romfs/ is intact, emptied or gone (#107)."""

    def setUp(self):
        self.steps = steps
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.dest, temp = base / 'dest', base / 'temp'
        (self.dest / 'romfs').mkdir(parents=True)
        temp.mkdir()
        image = b'romfs image'
        (self.dest / 'romfs' / 'romfs.bin').write_bytes(image)
        digest = steps.extract.sha256(self.dest / 'romfs' / 'romfs.bin')
        (self.dest / 'romfs' / 'index.json').write_text(json.dumps({'romfs_size': len(image), 'romfs_sha256': digest}))
        self.expected = {'romfs': {'size': len(image), 'sha256': digest}}
        self.ctx = SimpleNamespace(reporter=Reporter(out=io.StringIO(), log_path=base / 'x.log'), dest=self.dest,
                                   temp=temp, cache=self.dest / 'build-cache', lang='en-US', state={},
                                   layout=SimpleNamespace(ini_example=ROOT / 'astralchain.ini.example', dev=False,
                                                          python=base / 'python' / 'python.exe', root=base / 'release'))
        self.saved = steps.expected_game
        steps.expected_game = lambda ctx: self.expected

    def tearDown(self):
        self.steps.expected_game = self.saved
        self.ctx.reporter.close()
        self.tmp.cleanup()

    def run_cleanup(self):
        try:
            self.steps.Cleanup().run(self.ctx)
            return 'ok'
        except self.steps.InstallError as e:
            return e.id

    def test_intact_romfs_passes(self):
        self.assertEqual(self.run_cleanup(), 'ok')

    def test_removes_the_scratch_and_writes_rebuild_command(self):
        for name in ('check', 'shared-fonts'):
            (self.ctx.temp / name).mkdir()
        (self.ctx.temp / 'expected-install.json').write_text('{}')
        self.assertEqual(self.run_cleanup(), 'ok')
        self.assertEqual(list(self.ctx.temp.iterdir()), [])
        lines = (self.dest / 'build-cache' / 'rebuild.txt').read_text(encoding='utf-8').splitlines()
        self.assertEqual(lines[1:], ['-c', steps.REBUILD_BOOT, str(self.ctx.layout.root), '--root',
                                     str(self.ctx.layout.root), '--rebuild', str(self.dest)])

    def test_keeps_the_build_cache(self):
        for name in ('gen', 'build'):
            (self.ctx.cache / name).mkdir(parents=True)
        self.assertEqual(self.run_cleanup(), 'ok')
        self.assertTrue((self.ctx.cache / 'gen').is_dir() and (self.ctx.cache / 'build').is_dir())

    def test_finish_temp_keeps_the_log_in_the_install_folder(self):
        (self.ctx.temp / 'logs').mkdir(parents=True)
        (self.ctx.temp / 'logs' / 'install-1.log').write_text('log')
        (self.ctx.temp / 'check').mkdir()
        log = self.ctx.cache / 'logs' / 'install-1.log'
        steps.finish_temp(self.ctx, log)
        self.assertFalse(self.ctx.temp.exists())
        self.assertEqual(log.read_text(encoding='utf-8'), 'log')

    def test_emptied_romfs_fails(self):
        for p in (self.dest / 'romfs').iterdir():
            p.unlink()
        self.assertEqual(self.run_cleanup(), 'cleanup.romfs_unreadable')

    def test_missing_romfs_fails(self):
        self.steps.remove(self.dest / 'romfs')
        self.assertEqual(self.run_cleanup(), 'cleanup.romfs_unreadable')

    def test_romfs_of_another_size_fails(self):
        (self.dest / 'romfs' / 'romfs.bin').write_bytes(b'short')
        self.assertEqual(self.run_cleanup(), 'cleanup.romfs_unreadable')


class BaseGameCheckTest(unittest.TestCase):
    """A file that holds an update or a combined package is named as such, not reported as another game."""
    BASE, UPDATE = '01007300020FA000', '01007300020FA800'

    @staticmethod
    def nca(content_type, title_id):
        return ({'name': 'x.nca', 'nca_offset': 0, 'size': 0xc00, 'content_type': content_type, 'title_id': title_id}, b'')

    def test_update_package_names_the_update(self):
        # An update NSP keeps the base title id on its Program NCA; only its Meta NCA carries the update id.
        records = [self.nca(0, self.BASE), self.nca(1, self.UPDATE)]
        self.assertEqual(steps.base_misuse(records), 'check.base_is_update')

    def test_combined_package_is_named(self):
        records = [self.nca(0, self.BASE), self.nca(1, self.BASE), self.nca(1, self.UPDATE)]
        self.assertEqual(steps.base_misuse(records), 'check.base_combined')

    def test_plain_base_game_passes(self):
        records = [self.nca(0, self.BASE), self.nca(1, self.BASE), self.nca(3, self.BASE)]
        self.assertIsNone(steps.base_misuse(records))
        self.assertEqual(steps.program_nca(records)[0]['title_id'], self.BASE)

    def test_another_title_is_rejected(self):
        with self.assertRaises(ValueError):
            steps.program_nca([self.nca(0, '0100F2C0115B6000')])

    def test_update_slot_accepts_the_update(self):
        records = [self.nca(0, self.BASE), self.nca(1, self.UPDATE)]
        self.assertIsNone(steps.update_misuse(records))
        self.assertEqual(steps.program_nca(records, update=True)[0]['title_id'], self.BASE)

    def test_update_slot_names_a_base_game(self):
        records = [self.nca(0, self.BASE), self.nca(1, self.BASE)]
        self.assertEqual(steps.update_misuse(records), 'check.update_is_base')

    def test_update_slot_names_a_combined_package(self):
        records = [self.nca(0, self.BASE), self.nca(1, self.BASE), self.nca(1, self.UPDATE)]
        self.assertEqual(steps.update_misuse(records), 'check.update_combined')


class EventsTest(unittest.TestCase):
    def test_lines_are_json_and_the_log_is_english(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            rep = Reporter(out=out, log_path=Path(d) / 'x.log')
            rep.step('check', 'started')
            rep.progress('extract', 'copy', 0, 10, 'bytes')
            rep.progress('extract', 'copy', 5, 10, 'bytes')    # throttled
            rep.progress('extract', 'copy', 10, 10, 'bytes')   # last one always goes out
            rep.error('check.disk_space', {'path': 'D:\\x', 'need_gb': 40, 'free_gb': 3}, step='check')
            rep.close()
            events = [json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual([e['type'] for e in events], ['step', 'progress', 'progress', 'error'])
            self.assertEqual([e.get('done') for e in events[1:3]], [0, 10])
            self.assertTrue(all('t' in e for e in events))
            log = (Path(d) / 'x.log').read_text(encoding='utf-8')
            self.assertIn('40 GB needed, 3 GB free', log)
            self.assertIn('[check.disk_space]', log)

    def test_catalog_has_every_id_the_code_uses(self):
        catalog = json.loads((ROOT / 'installer' / 'text' / 'en.json').read_text(encoding='utf-8'))
        used = set()
        for path in (ROOT / 'installer').glob('*.py'):
            used |= set(re.findall(r"InstallError\('([a-z_.]+)'", path.read_text(encoding='utf-8')))
            used |= set(re.findall(r"warning\('([a-z_.]+)'", path.read_text(encoding='utf-8')))
            used |= {f'phase.{s}.{p}' for s, p in re.findall(r"progress\('(\w+)', '(\w+)'[,)]",
                                                                  path.read_text(encoding='utf-8'))}
        self.assertEqual(sorted(i for i in used if i not in catalog), [])


def progress_lines(text):
    """[(fraction, text)] of the `progress` lines in text; any other line fails the test that reads them."""
    rows = []
    for line in text.splitlines():
        word, number, rest = line.split(' ', 2)
        assert word == 'progress', line
        rows.append((float(number), rest))
    return rows


class ProgressLinesTest(unittest.TestCase):
    """--progress-lines: the events of a run as the `progress <0..1> <text>` lines a launcher reads (progress.py)."""

    def lines(self, fill):
        out = io.StringIO()
        rep = Reporter(out=ProgressLines(out))
        fill(rep)
        return progress_lines(out.getvalue())

    @staticmethod
    def first_install(rep):
        rep.emit('start', release='1.0', mode='install', dest='D:\\g', log='D:\\g.log', steps=ALL)
        rep.emit('plan', run=ALL, skip={})
        for step, phases in (('check', ('host', 'inputs')), ('extract', ('exefs', 'copy', 'merge')),
                             ('lift', ('discover', 'emit')), ('compile', ('compile', 'link')),
                             ('prepare', ('seed', 'translate', 'pack')), ('cleanup', ('ini', 'temp'))):
            rep.step(step, 'started')
            for phase in phases:
                rep.progress(step, phase, 0, 10, 'objects', force=True)
                rep.progress(step, phase, 10, 10, 'objects')
            rep.step(step, 'done', seconds=1.0)
        rep.emit('end', result='ok', seconds=6.0)

    def test_fractions_go_from_zero_to_one_and_never_back(self):
        rows = self.lines(self.first_install)
        fractions = [f for f, _ in rows]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual((fractions[0], fractions[-1]), (0.0, 1.0))
        self.assertTrue(all(f < 1.0 for f in fractions[:-1]))
        self.assertGreater(len(set(fractions)), 12)   # it moves inside the steps, not only between them
        self.assertEqual(rows[-1][1], 'The game is ready')
        self.assertTrue(any(text.startswith('Compiling') for _, text in rows))

    def test_skipped_steps_count_as_done(self):
        def update(rep):
            rep.emit('start', release='1.1', mode='update', dest='D:\\g', log='D:\\g.log', steps=ALL)
            rep.emit('plan', run=['check', 'lift', 'compile', 'cleanup'], skip={'extract': 'current',
                                                                                'prepare': 'current'})
            rep.step('check', 'started')
            rep.step('check', 'done', seconds=0.1)
            rep.step('extract', 'skipped', reason='current')
            rep.step('lift', 'started')
        rows = self.lines(update)
        self.assertGreater(rows[0][0], 0.5)   # extract and prepare, more than half of a first install, are behind
        self.assertEqual(rows[-1][1], 'Translating the game code to C')

    def test_an_error_is_the_last_line(self):
        def failing(rep):
            rep.emit('start', release='1.0', mode='install', dest='D:\\g', log='D:\\run.log', steps=ALL)
            rep.emit('plan', run=ALL, skip={})
            rep.step('check', 'started')
            rep.error('check.disk_space', {'path': 'D:\\g', 'need_gb': 40, 'free_gb': 3}, step='check')
            rep.emit('end', result='failed', seconds=0.2)
        rows = self.lines(failing)
        self.assertIn('40 GB needed, 3 GB free', rows[-1][1])
        self.assertIn('D:\\run.log', rows[-1][1])   # the launcher shows no stderr: the log's place is in the line
        self.assertLess(rows[-1][0], 1.0)

    def test_the_cli_prints_only_progress_lines_and_fails_with_exit_code_1(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PYTHONIOENCODING='cp1252')   # the launcher's pipe: the lines stay UTF-8
            r = subprocess.run([sys.executable, '-m', 'installer', '--lang', 'auto', '--dest', str(Path(d) / 'game'),
                                '--temp', str(Path(d) / 'temp'), '--progress-lines'], cwd=ROOT, env=env,
                               capture_output=True, timeout=600)
            rows = progress_lines(r.stdout.decode('utf-8'))   # no game files given: the check step stops the run
            self.assertEqual(r.returncode, 1, r.stderr.decode(errors='replace'))
            self.assertTrue(rows and rows[-1][1], rows)
            self.assertLess(rows[-1][0], 1.0)


LIFELINE_PARENT = """
import subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from installer import lifeline
assert lifeline.arm()
tool = subprocess.Popen([sys.executable, '-c', sys.argv[2]], stdout=subprocess.PIPE, text=True,
                        start_new_session=sys.platform != 'win32')
lifeline.watch(tool.pid)
print(tool.pid, tool.stdout.readline(), flush=True)
time.sleep(120)
"""

# A tool that starts a program in a process group of its own, as ninja starts the compilers
LIFELINE_TOOL = """
import os, subprocess, sys, time
job = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                       **({} if sys.platform == 'win32' else {'preexec_fn': os.setpgrp}))
print(job.pid, flush=True)
time.sleep(120)
"""


def process_alive(pid):
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        # a killed child of a dead parent stays as a zombie until init reaps it: that is not running
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except OSError:
            return False
    out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH', '/FO', 'CSV'], capture_output=True,
                         text=True).stdout
    return f'"{pid}"' in out


class LifelineTest(unittest.TestCase):
    """A launcher stops the setup by killing its process, with no signal to handle: the tools must not outlive it
    (lifeline.py)."""

    def test_the_tools_end_when_the_setup_is_killed(self):
        parent = subprocess.Popen([sys.executable, '-c', LIFELINE_PARENT, str(ROOT), LIFELINE_TOOL],
                                  stdout=subprocess.PIPE, text=True)
        try:
            tool, job = map(int, parent.stdout.readline().split())
            self.assertTrue(process_alive(tool) and process_alive(job))
            parent.kill()   # TerminateProcess on Windows, SIGKILL elsewhere: what the launcher does
            parent.wait()
            for _ in range(50):
                if not (process_alive(tool) or process_alive(job)):
                    break
                time.sleep(0.1)
            self.assertFalse(process_alive(tool), 'the tool outlived the setup')
            self.assertFalse(process_alive(job), 'a program the tool started in its own process group outlived it')
        finally:
            parent.kill()
            parent.stdout.close()


class AutoLanguageTest(unittest.TestCase):
    """--lang auto (cli.auto_language): the launcher has no language to pass."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_first_install_takes_the_system_language(self):
        self.assertIn(cli.auto_language(self.dest, None), ini.LANGUAGES)

    def test_the_system_language_maps_to_a_language_of_the_game(self):
        from installer.guilogic import game_language_for_langid, game_language_for_locale
        langids = {0x040c: 'fr', 0x0c0c: 'fr-CA', 0x0407: 'de', 0x0410: 'it', 0x0c0a: 'es', 0x080a: 'es-419',
                   0x0413: 'nl', 0x0416: 'pt-BR', 0x0816: 'pt', 0x0419: 'ru', 0x0411: 'ja', 0x0409: 'en-US',
                   0x0809: 'en-GB', 0x0804: 'en-US', 0x0412: 'en-US', 0x041f: 'en-US'}
        self.assertEqual({k: game_language_for_langid(k) for k in langids}, langids)
        locales = {'fr_FR.UTF-8': 'fr', 'fr_CA.UTF-8': 'fr-CA', 'de_AT': 'de', 'es_ES.UTF-8': 'es', 'es_MX': 'es-419',
                   'pt_BR.UTF-8': 'pt-BR', 'pt_PT': 'pt', 'en_GB.UTF-8': 'en-GB', 'en_US.UTF-8': 'en-US',
                   'ja_JP.eucJP': 'ja', 'zh_CN.UTF-8': 'en-US', 'ko_KR': 'en-US', 'C': 'en-US', '': 'en-US',
                   'ru_RU@euro': 'ru', 'nl-NL': 'nl'}
        self.assertEqual({k: game_language_for_locale(k) for k in locales}, locales)
        every = {game_language_for_langid(k) for k in langids} | {game_language_for_locale(k) for k in locales}
        self.assertEqual(every - (set(ini.LANGUAGES) - set(ini.NEED_SHARED_FONTS)), set())

    def test_an_installed_game_keeps_its_recorded_language(self):
        self.assertIsNone(cli.auto_language(self.dest, {'lang': 'de'}))

    def test_the_language_set_in_the_ini_wins(self):
        (self.dest / 'astralchain.ini').write_text('[language]\nlang = fr\n', encoding='utf-8')
        self.assertEqual(cli.auto_language(self.dest, {'lang': 'de'}), 'fr')

    def test_an_unknown_language_in_the_ini_is_ignored(self):
        (self.dest / 'astralchain.ini').write_text('[language]\nlang = xx\n', encoding='utf-8')
        self.assertIsNone(cli.auto_language(self.dest, {'lang': 'de'}))


class AppImageRebuildTest(unittest.TestCase):
    """rebuild.txt from an AppImage names the AppImage file, not the mount that goes away with the setup."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.image, self.mount = base / 'Setup.AppImage', base / 'mount'
        self.image.write_bytes(b'image')
        self.mount.mkdir()
        self.saved = (layouts.WIN, {k: os.environ.get(k) for k in ('APPIMAGE', 'APPDIR')})
        layouts.WIN = False
        os.environ.update(APPIMAGE=str(self.image), APPDIR=str(self.mount))

    def tearDown(self):
        layouts.WIN = self.saved[0]
        for key, value in self.saved[1].items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def ctx(self, root, dev=False):
        return SimpleNamespace(dest=Path('/games/Astral Chain'),
                               layout=SimpleNamespace(root=root, dev=dev, python=root / 'python' / 'bin' / 'python3'))

    def test_the_appimage_is_the_rebuild_command(self):
        self.assertEqual(steps.rebuild_command(self.ctx(self.mount)),
                         [str(self.image), '--rebuild', str(Path('/games/Astral Chain'))])

    def test_the_appimage_of_another_program_is_not_taken(self):
        for ctx in (self.ctx(Path(self.tmp.name) / 'unpacked'), self.ctx(self.mount, dev=True)):
            self.assertNotIn(str(self.image), steps.rebuild_command(ctx))


class ManifestTest(unittest.TestCase):
    """recomplauncher.json against the code it drives: the CLI's options, the settings table of runtime/cfg.c and the
    languages of installer/ini.py."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'recomplauncher.json').read_text(encoding='utf-8'))
        text = (ROOT / 'runtime' / 'cfg.c').read_text(encoding='utf-8')
        cls.sections, cls.metas = {}, {}
        for key, section in re.findall(r'\{"([A-Z0-9_]+)", "([a-z ]+)"\}', text):
            cls.sections.setdefault(key, section)   # the section table comes first in the file
        for key, default, choices in re.findall(r'\{"([A-Z0-9_]+)", "([^"]*)", (?:"([^"]*)"|NULL), (?:LIVE|RESTART)\}',
                                                text):
            cls.metas[key] = (default, choices.split('|'))

    def test_the_prepare_arguments_are_those_of_the_cli(self):
        prepare = self.manifest['prepare']
        self.assertEqual(prepare['args'][:2], ['-m', 'installer'])
        args = cli.build_parser().parse_args([a.replace('{install}', 'D:/x') for a in prepare['args'][2:]])
        self.assertEqual((args.lang, args.progress_lines, args.dest), ('auto', True, Path('D:/x/Astral Chain')))
        self.assertTrue(all((args.base, args.update, args.keys)))
        used = set(re.findall(r'\{input:([\w-]+)\}', ' '.join(prepare['args'])))
        self.assertEqual(used, {i['id'] for i in self.manifest['inputs']})

    def test_the_game_lives_in_the_folder_the_window_suggests(self):
        from installer.guilogic import GAME_FOLDER
        game = '{install}/' + GAME_FOLDER
        self.assertIn(game, self.manifest['prepare']['args'])
        self.assertEqual(self.manifest['start']['program'][:2], [GAME_FOLDER + '/astralchain.exe',
                                                                 GAME_FOLDER + '/astralchain'])
        places = [self.manifest['config'][0]['file'][0]['path'], *self.manifest['saves'],
                  *(p[0]['path'] for p in self.manifest['folders'].values())]
        self.assertTrue(all(p.startswith(game + '/') for p in places), places)

    def test_every_setting_is_a_key_of_the_runtime_with_its_values_and_default(self):
        for setting in self.manifest['settings']:
            (name, default), = setting['defaults'].items()
            section, _, key = name.partition('/')
            key = key.upper()
            self.assertEqual(self.sections.get(key), section, name)
            meta_default, allowed = self.metas[key]
            values = [c['set'][name] for c in setting['choices']]
            self.assertIn(default, values, name)
            if key == 'LANG':   # no default in the table: the setup always writes the line
                self.assertEqual(set(values), set(ini.LANGUAGES) - set(ini.NEED_SHARED_FONTS))
                continue
            self.assertEqual(default, meta_default, name)
            if key != 'OUTPUT_RES':   # a size, not a list of choices
                self.assertEqual(set(values) - set(allowed), set(), name)

    def test_the_launcher_switches_on_the_line_the_setup_wrote(self):
        """The launcher edits astralchain.ini line by line (its profile.rs File::set): inside the key's section it
        rewrites the first active `key = value` line, else the first commented line that starts with `key =`, else it
        adds a line at the section's end. The file the setup writes starts as astralchain.ini.example, whose lines are
        all commented, so for every setting that first commented line must be the setting's own `# key = default`
        line, not a sentence of the comments above it that happens to start with the key."""
        lines = (ROOT / 'astralchain.ini.example').read_text(encoding='utf-8').splitlines()
        for setting in self.manifest['settings']:
            (name, default), = setting['defaults'].items()
            section, _, key = name.partition('/')
            start = next(i for i, l in enumerate(lines) if l.strip().lower() == f'[{section}]') + 1
            end = next((i for i in range(start, len(lines)) if lines[i].strip().startswith('[')), len(lines))
            body = lines[start:end]
            self.assertFalse([l for l in body if re.match(rf'^\s*{re.escape(key)}\s*=', l)], name)
            commented = [l for l in body if re.match(rf'^\s*[;#]+\s*{re.escape(key)}\s*=', l)]
            self.assertTrue(commented, f'{name}: no commented line, the launcher would add one at the section end')
            if key != 'lang':   # the example shows en-US; the setup writes the language itself
                self.assertEqual(re.sub(r'^\s*[;#]+\s*', '', commented[0]).replace(' ', ''), f'{key}={default}', name)


if __name__ == '__main__':
    unittest.main()
