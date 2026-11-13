"""The installer window: event model, form, catalogs, cancel, a self-closing Tk run and one end-to-end run.

    .venv-py314\\Scripts\\python tests\\installer_gui_test.py

The end-to-end test runs the real CLI in --verify mode on a finished install (every step skips) and folds its events
through the same code the window uses (the fixture's step keys are recomputed for the checkout under test, because the keys fingerprint recomp/, tools/ and runtime/). The install is build/inst-gui/e2e-install: hard links to build/install-test, so
nothing there is written, except the state file and astralchain.ini, which a run rewrites and which are copied. The run hashes
the 17 GB RomFS; it is skipped when the hash differs from the expected one in all three tries. On the developer machine
(RAM with a bad bit) a buffered read of a file sometimes returns flipped bits from the page cache while the file on
disk is intact (an unbuffered read of install-test's romfs.bin gives the expected hash), so such a skip is the machine,
not the installer. It is also skipped when build/install-test is absent.
"""
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from installer import layout as layouts  # noqa: E402
from installer import state as statefile  # noqa: E402
from installer.events import Reporter  # noqa: E402
from installer.runner import Context  # noqa: E402
from installer.steps import all_steps  # noqa: E402
from installer.guilogic import (BUILT_STEPS, DONE, FAILED, PENDING, RUNNING, SKIPPED,  # noqa: E402
                                UI_LANGUAGE_LABELS, UI_LANGUAGES, UPDATE, Catalog, Form, Model, Process, amount,
                                default_game_language, describe_event, find_keys, inspect_folder, install_subfolder,
                                keys_places, launcher_game_folder, parse_line, sibling_game_files,
                                ui_language_for_langid, windows_ui_language)

INSTALLED = ROOT / 'build' / 'install-test'
FIXTURE = ROOT / 'build' / 'inst-gui' / 'e2e-install'
COPIED = ('astralchain.ini', statefile.NAME)   # written by a run, so never linked
START = {'type': 'start', 'release': 'r1', 'mode': 'install', 'dest': 'D:\\T', 'log': 'D:\\T.log',
         'steps': ['check', 'extract', 'lift', 'compile', 'cleanup']}


def fold(*events):
    m = Model()
    for e in events:
        m.apply(e)
    return m


class ParseTest(unittest.TestCase):
    def test_event_line(self):
        self.assertEqual(parse_line('{"type": "step", "step": "check", "state": "started"}\n')['step'], 'check')

    def test_noise_is_not_an_event(self):
        for line in ('', 'Traceback (most recent call last):', '{broken', '[1, 2]', '{"no_type": 1}', '{"type": 3}'):
            self.assertIsNone(parse_line(line), line)

    def test_unicode_survives(self):
        e = parse_line(json.dumps({'type': 'warning', 'id': 'x', 'params': {'p': 'Привет'}}, ensure_ascii=False))
        self.assertEqual(e['params']['p'], 'Привет')


class ModelTest(unittest.TestCase):
    def test_steps_follow_the_events(self):
        m = fold(START, {'type': 'plan', 'run': ['check', 'extract'], 'skip': {'lift': 'current', 'compile': 'current'}},
                 {'type': 'step', 'step': 'check', 'state': 'started'})
        self.assertEqual([m.states[s] for s in m.steps],
                         [RUNNING, PENDING, SKIPPED, SKIPPED, PENDING])
        self.assertEqual(m.current, 'check')
        m.apply({'type': 'step', 'step': 'check', 'state': 'done', 'seconds': 1.0})
        self.assertEqual(m.states['check'], DONE)
        self.assertIsNone(m.current)
        m.apply({'type': 'step', 'step': 'extract', 'state': 'skipped', 'reason': 'current'})
        self.assertEqual(m.states['extract'], SKIPPED)

    def test_progress_fraction(self):
        m = fold(START, {'type': 'step', 'step': 'extract', 'state': 'started'},
                 {'type': 'progress', 'step': 'extract', 'phase': 'copy', 'done': 5, 'total': 20, 'unit': 'bytes'})
        self.assertEqual((m.current, m.phase, m.fraction()), ('extract', 'copy', 0.25))
        m.apply({'type': 'progress', 'step': 'extract', 'phase': 'merge'})
        self.assertIsNone(m.fraction())   # a phase without size is indeterminate
        m.apply({'type': 'progress', 'step': 'extract', 'phase': 'copy', 'done': 30, 'total': 20})
        self.assertEqual(m.fraction(), 1.0)

    def test_a_new_step_resets_progress(self):
        m = fold(START, {'type': 'progress', 'step': 'check', 'phase': 'inputs', 'done': 1, 'total': 2},
                 {'type': 'step', 'step': 'extract', 'state': 'started'})
        self.assertIsNone(m.phase)
        self.assertIsNone(m.fraction())

    def test_error_marks_the_step(self):
        m = fold(START, {'type': 'step', 'step': 'check', 'state': 'started'},
                 {'type': 'error', 'id': 'check.disk_space', 'step': 'check', 'params': {'path': 'D:\\T'}},
                 {'type': 'end', 'result': 'failed', 'seconds': 2})
        self.assertEqual(m.states['check'], FAILED)
        self.assertEqual(m.error[0], 'check.disk_space')
        self.assertEqual(m.result, 'failed')

    def test_space_text(self):
        m = fold({'type': 'space', 'dest': 'D:\\T', 'need': 36_400_000_000, 'free': 100_000_000_000, 'temp': 'C:\\t',
                  'temp_need': 0, 'temp_free': 5, 'same_disk': True})
        text = m.need_text(Catalog('en'))
        self.assertIn('36.4', text)
        self.assertIn('100.0', text)
        self.assertEqual(Model().need_text(Catalog('en')), '')
        m = fold({'type': 'space', 'dest': 'D:\\T', 'need': 20_000_000_000, 'free': 1, 'temp': 'C:\\t',
                  'temp_need': 2_000_000_000, 'temp_free': 3, 'same_disk': False})
        self.assertIn('C:\\t', m.need_text(Catalog('en')))

    def test_warnings_collect(self):
        m = fold(START, {'type': 'warning', 'id': 'warn.lang_needs_fonts', 'params': {'lang': 'ko'}})
        self.assertEqual(m.warnings, [('warn.lang_needs_fonts', {'lang': 'ko'})])

    def test_finish_without_end_event(self):
        m = fold(START, {'type': 'step', 'step': 'lift', 'state': 'started'})
        m.finish(-1073741510)
        self.assertEqual(m.result, 'failed')
        self.assertEqual(m.states['lift'], FAILED)
        self.assertEqual(m.error[0], 'ui.process_died')

    def test_finish_after_cancel(self):
        m = fold(START, {'type': 'step', 'step': 'lift', 'state': 'started'}, {'type': 'end', 'result': 'cancelled'})
        m.finish(3)
        self.assertEqual(m.result, 'cancelled')
        self.assertIsNone(m.error)
        self.assertEqual(m.states['lift'], PENDING)   # not finished; the next run resumes here

    def test_exit_code_3_alone_is_cancelled(self):
        m = Model()
        m.finish(3)
        self.assertEqual(m.result, 'cancelled')

    def test_end_names_the_final_log(self):
        m = fold(START, {'type': 'end', 'result': 'ok', 'seconds': 1,
                         'log': 'D:\\T\\build-cache\\logs\\install-1.log'})
        self.assertEqual(m.log_path, 'D:\\T\\build-cache\\logs\\install-1.log')
        m = fold(START, {'type': 'end', 'result': 'failed', 'seconds': 1})
        self.assertEqual(m.log_path, 'D:\\T.log')   # a failed run keeps its log in the temp folder

    def test_unknown_event_is_ignored(self):
        m = fold(START, {'type': 'future_thing', 'x': 1})
        self.assertEqual(m.states['check'], PENDING)


class RunStatesTest(unittest.TestCase):
    OK = [START, {'type': 'plan', 'run': ['check', 'extract'], 'skip': {'lift': 'current', 'compile': 'current',
                                                                     'cleanup': 'not_needed'}},
          {'type': 'step', 'step': 'check', 'state': 'started'}, {'type': 'step', 'step': 'check', 'state': 'done'},
          {'type': 'step', 'step': 'extract', 'state': 'started'}, {'type': 'step', 'step': 'extract', 'state': 'done'},
          {'type': 'step', 'step': 'lift', 'state': 'skipped'}, {'type': 'step', 'step': 'compile', 'state': 'skipped'},
          {'type': 'step', 'step': 'cleanup', 'state': 'done'},
          {'type': 'end', 'result': 'ok', 'seconds': 1}]
    FAILED = [START, {'type': 'step', 'step': 'check', 'state': 'started'}, {'type': 'step', 'step': 'check', 'state': 'done'},
              {'type': 'step', 'step': 'extract', 'state': 'started'},
              {'type': 'error', 'id': 'extract.romfs_mismatch', 'step': 'extract', 'params': {}},
              {'type': 'warning', 'id': 'warn.lang_needs_fonts', 'params': {'lang': 'ko'}},
              {'type': 'end', 'result': 'failed', 'seconds': 1}]

    def test_ok_run_has_only_done_and_skipped_steps(self):
        m = fold(*self.OK)
        m.finish(0)
        self.assertEqual(m.result, 'ok')
        self.assertIsNone(m.error)
        self.assertEqual({m.states[s] for s in m.steps}, {DONE, SKIPPED})

    def test_a_new_run_keeps_nothing_of_the_failed_one(self):
        m = fold(*self.FAILED)
        m.finish(1)
        self.assertEqual(m.states['extract'], FAILED)
        for e in self.OK:   # the same model object fed a second run (the window makes a new one, but even then)
            m.apply(e)
        m.finish(0)
        self.assertEqual({m.states[s] for s in m.steps}, {DONE, SKIPPED})
        self.assertIsNone(m.error)
        self.assertEqual(m.warnings, [])
        self.assertEqual(m.result, 'ok')


class FormTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ('base.nsp', 'upd.nsp', 'prod.keys'):
            (self.dir / name).write_bytes(b'x')

    def tearDown(self):
        self.tmp.cleanup()

    def form(self, **kw):
        values = dict(base=str(self.dir / 'base.nsp'), update=str(self.dir / 'upd.nsp'),
                      keys=str(self.dir / 'prod.keys'), lang='en-US', dest=str(self.dir / 'out'))
        values.update(kw)
        return Form(**values)

    def test_complete_form(self):
        self.assertEqual(self.form().problems(), [])

    def test_empty_form(self):
        self.assertEqual(Form().problems(),
                         ['ui.need_dest', 'ui.need_base', 'ui.need_update', 'ui.need_keys', 'ui.need_lang'])

    def test_missing_files(self):
        self.assertEqual(self.form(base=str(self.dir / 'no.nsp'), update=str(self.dir / 'no-upd.nsp'),
                                   keys=str(self.dir / 'no.keys')).problems(),
                         ['ui.missing_base', 'ui.missing_update', 'ui.missing_keys'])

    def test_unknown_language(self):
        self.assertEqual(self.form(lang='xx').problems(), ['ui.need_lang'])

    def test_command(self):
        cmd = self.form().command('py.exe', root=ROOT, temp=self.dir / 't', verify_hashes=True)
        self.assertEqual(cmd[:3], ['py.exe', '-m', 'installer'])
        for flag in ('--base', '--update', '--keys', '--lang', '--dest', '--root', '--temp', '--verify'):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index('--lang') + 1], 'en-US')

    def test_continue_anyway_carries_the_flag(self):
        # The window's "Continue anyway" runs the same command with the version check as a warning.
        self.assertNotIn('--accept-unknown-version', self.form().command('py.exe'))
        cmd = self.form().command('py.exe', accept_version=True)
        self.assertEqual(cmd[-1], '--accept-unknown-version')

    BUILT = {s: {'key': 'k'} for s in BUILT_STEPS}

    def installed(self, name='out', **fields):
        out = self._mkdir(self.dir / name)
        statefile.save(out, statefile.empty('r0') | {'lang': 'ru', 'version': '1.0.1', 'steps': dict(self.BUILT)}
                       | fields)
        return out

    def test_installed_folder_needs_only_itself(self):
        out = self.installed()
        form = Form(dest=str(out), mode=UPDATE)
        self.assertTrue(form.update_mode())
        self.assertFalse(form.needs_game_files())
        self.assertEqual(form.problems(), [])
        cmd = form.command('py.exe')
        self.assertEqual(cmd, ['py.exe', '-m', 'installer', '--dest', str(out)])

    def test_old_install_asks_for_the_game_files_again(self):
        out = self.installed(version='1.0.0')
        self.assertEqual(Form(dest=str(out), mode=UPDATE).problems(),
                         ['ui.need_base', 'ui.need_update', 'ui.need_keys', 'ui.need_lang'])

    def test_update_of_an_install_that_never_extracted_asks_for_the_game_files(self):
        # a first setup that stopped before the extraction finished: the next run extracts, so it reads the files
        out = self.installed(steps={})
        self.assertEqual(Form(dest=str(out), lang='ru', mode=UPDATE).problems(),
                         ['ui.need_base', 'ui.need_update', 'ui.need_keys'])
        out = self.installed('out2', steps={'extract': {'key': 'k'}})
        self.assertEqual(Form(dest=str(out), lang='ru', mode=UPDATE).problems(), [])

    def test_installed_folder_still_checks_given_files(self):
        out = self.installed()
        self.assertEqual(Form(dest=str(out), base=str(self.dir / 'no.nsp'), mode=UPDATE).problems(),
                         ['ui.missing_base'])

    def test_update_needs_an_installation(self):
        self.assertEqual(Form(mode=UPDATE).problems(), ['ui.need_install_folder'])
        self.assertEqual(Form(dest=str(self.dir / 'nothing'), mode=UPDATE).problems(), ['ui.no_install'])
        self.assertEqual(Form(dest=str(self.dir), mode=UPDATE).problems(), ['ui.no_install'])

    def test_first_install_refuses_a_finished_installation(self):
        self.assertEqual(self.form(dest=str(self.installed())).problems(), ['ui.dest_has_install'])

    def test_first_install_continues_a_stopped_setup(self):
        out = self.installed(steps={'extract': {'key': 'k'}})
        self.assertEqual(self.form(dest=str(out)).problems(), [])

    def test_first_install_wants_an_empty_folder(self):
        self.assertEqual(self.form(dest=str(self.dir)).problems(), ['ui.dest_not_empty'])
        self.assertEqual(self.form(dest=str(self.dir / 'prod.keys')).problems(), ['ui.dest_is_file'])
        self.assertEqual(self.form(dest=str(self._mkdir(self.dir / 'empty'))).problems(), [])

    def test_broken_state_is_a_new_install(self):
        out = self._mkdir(self.dir / 'out')
        (out / statefile.NAME).write_text('{nope', encoding='utf-8')
        self.assertFalse(Form(dest=str(out)).update_mode())
        self.assertEqual(inspect_folder(str(out)).kind, 'broken')
        self.assertEqual(Form(dest=str(out), mode=UPDATE).problems(), ['ui.install_unreadable'])

    def test_folder_kinds(self):
        self.assertEqual(inspect_folder('').kind, '')
        self.assertEqual(inspect_folder(str(self.dir / 'new')).kind, 'new')
        self.assertEqual(inspect_folder(str(self.dir)).kind, 'other')
        folder = inspect_folder(str(self.installed(version='1.0.1', lang='zh-CN')))
        self.assertEqual((folder.kind, folder.version, folder.lang, folder.finished), ('install', '1.0.1', 'zh-Hans', True))

    def test_install_subfolder(self):
        # a folder that holds other files gets the game in a new folder inside it; a new or empty one is used as is
        self.assertEqual(install_subfolder(str(self.dir)), os.path.join(str(self.dir), 'Astral Chain'))
        self.assertEqual(install_subfolder(str(self.dir / 'new')), str(self.dir / 'new'))
        empty = str(self._mkdir(self.dir / 'empty'))
        self.assertEqual(install_subfolder(empty), empty)

    @staticmethod
    def _mkdir(path):
        path.mkdir(parents=True, exist_ok=True)
        return path


class LauncherFolderTest(unittest.TestCase):
    """The folder the window starts with when RecompLauncher opened it (guilogic.launcher_game_folder)."""

    RELEASE = SimpleNamespace(dev=False, root=Path('C:/Launcher/games/astral-chain'))

    def test_the_game_goes_inside_the_folder_of_the_launcher(self):
        self.assertEqual(launcher_game_folder(self.RELEASE, {'RECOMPLAUNCHER': '1'}),
                         str(self.RELEASE.root / 'Astral Chain'))

    def test_nothing_is_suggested_without_the_launcher(self):
        self.assertEqual(launcher_game_folder(self.RELEASE, {}), '')
        self.assertEqual(launcher_game_folder(self.RELEASE, {'RECOMPLAUNCHER': '0'}), '')

    def test_nothing_is_suggested_in_a_checkout(self):
        checkout = SimpleNamespace(dev=True, root=Path('C:/src/astral'))
        self.assertEqual(launcher_game_folder(checkout, {'RECOMPLAUNCHER': '1'}), '')


class FindersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, *names):
        for name in names:
            (self.dir / name).write_bytes(b'x')
        return [str(self.dir / n) for n in names]

    def test_dumper_names(self):
        base, update = self.touch('Astral Chain [01007300020FA000][v0].nsp',
                                  'Astral Chain [UPD][01007300020FA800][v65536].nsp')
        self.touch('notes.txt')
        self.assertEqual(sibling_game_files(base), {'base': base, 'update': update})
        self.assertEqual(sibling_game_files(update), {'base': base, 'update': update})

    def test_xci_and_version_tag(self):
        base, update = self.touch('astral.xci', 'astral v65536.nsp')
        self.assertEqual(sibling_game_files(base), {'base': base, 'update': update})

    def test_several_candidates_fill_nothing(self):
        first, _ = self.touch('a [01007300020FA000].nsp', 'b [01007300020FA000].nsp')
        self.assertEqual(sibling_game_files(first), {'base': None, 'update': None})

    def test_keys_beside_the_game_come_first(self):
        game, keys = self.touch('game.xci', 'prod.keys')
        self.assertEqual(keys_places([game])[0], Path(keys))
        self.assertEqual(find_keys([game]), keys)

    def test_emulator_folders(self):
        env = 'APPDATA' if os.name == 'nt' else 'XDG_DATA_HOME'
        old = os.environ.get(env)
        os.environ[env] = str(self.dir)
        try:
            self.assertEqual(find_keys(), '')
            ryujinx = self.dir / 'Ryujinx' / 'system'
            ryujinx.mkdir(parents=True)
            (ryujinx / 'prod.keys').write_bytes(b'x')
            self.assertEqual(find_keys(), str(ryujinx / 'prod.keys'))
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old


class CatalogTest(unittest.TestCase):
    def test_every_language_covers_english_with_the_same_placeholders(self):
        en = Catalog('en').english
        for lang in UI_LANGUAGES[1:]:
            with self.subTest(lang=lang):
                other = Catalog(lang).texts
                self.assertEqual(sorted(set(en) - set(other)), [])
                self.assertEqual(sorted(set(other) - set(en)), [])
                holes = [k for k in en
                         if sorted(re.findall(r'\{\w+\}', en[k])) != sorted(re.findall(r'\{\w+\}', other[k]))]
                self.assertEqual(holes, [])

    def test_language_names_tell_the_codes_apart(self):
        # gui.App.selected_lang maps the shown name back to its code across every window language
        names = {}
        for lang in UI_LANGUAGES:
            cat = Catalog(lang)
            for id in cat.english:
                if id.startswith('lang.'):
                    names.setdefault(cat(id), set()).add(id)
        self.assertEqual({n: ids for n, ids in names.items() if len(ids) > 1}, {})
        self.assertEqual(sorted(UI_LANGUAGE_LABELS), sorted(UI_LANGUAGES))

    def test_every_id_the_window_uses_exists(self):
        cat = Catalog('en')
        ids = ['ui.' + n for n in ('title', 'do_install', 'do_update', 'cancel', 'open_folder', 'notice', 'space_one',
                                   'space_two', 'process_died', 'result_ok', 'result_updated', 'result_failed',
                                   'result_cancelled', 'play', 'open_log', 'choice_install', 'choice_update')]
        ids += [f'ui.{n}_hint' for n in ('base', 'update', 'keys')]
        ids += [f'step.{s}' for s in ('check', 'fonts', 'extract', 'lift', 'compile', 'prepare', 'cleanup')]
        ids += ['ui.state_' + n for n in ('started', 'done', 'skipped', 'pending', 'running', 'failed')]
        for lang in UI_LANGUAGES:
            cat.set_language(lang)
            self.assertEqual(cat.lang, lang)
            self.assertEqual([i for i in ids if not cat.has(i)], [])

    def test_codes_become_names(self):
        for lang, korean in (('en', 'Korean'), ('ru', 'Корейский'), ('pt-BR', 'Coreano'), ('zh-Hans', '韩语')):
            cat = Catalog(lang)
            self.assertIn(korean, cat('warn.lang_needs_fonts', lang='ko'))
            self.assertNotIn(' ko ', cat('warn.lang_needs_fonts', lang='ko'))
            self.assertNotIn('extract', cat('state.output_changed', step='extract'))
            for word in ('base', 'update', 'keys'):
                self.assertIn(cat('param.input.' + word), cat('check.keys_wrong', input=word))
            self.assertNotIn('en-US', cat('check.lang_unknown', lang='xx', known='ja, en-US'))
        self.assertEqual(Catalog('en')('check.lang_unknown', lang='xx', known='ja, en-US'),
                         "Unknown language 'xx'. Known: Japanese, English (US).")

    def test_fallback_and_params(self):
        cat = Catalog('ru')
        self.assertIn('36.4', cat('ui.log_space', need='36.4', free='1.0'))
        self.assertEqual(cat('no.such.id'), 'no.such.id')
        self.assertIn('a, b', cat('check.release_incomplete', paths=['a', 'b']))
        cat.set_language('de')   # unknown language falls back to English
        self.assertEqual(cat.lang, 'en')

    def test_describe_and_amount(self):
        cat = Catalog('en')
        self.assertEqual(amount(cat, 6e9, 16e9, 'bytes'), '6.0 / 16.0 GB')
        self.assertEqual(amount(cat, 1, 9, 'shaders'), '1 / 9 shaders')
        self.assertEqual(amount(cat, 1, 9, 'widgets'), '1 / 9 widgets')
        line = describe_event(cat, {'type': 'step', 'step': 'check', 'state': 'done', 'seconds': 1.5})
        self.assertIn('1.5 s', line)
        self.assertEqual(describe_event(cat, {'type': 'start'}), '')

    def test_windows_language_is_a_known_value(self):
        self.assertIn(windows_ui_language(), UI_LANGUAGES)

    def test_windows_langid_picks_the_window_language(self):
        for langid, lang in ((0x0419, 'ru'), (0x0819, 'ru'), (0x0416, 'pt-BR'), (0x0816, 'en'),
                             (0x0804, 'zh-Hans'), (0x1004, 'zh-Hans'),
                             (0x0404, 'en'), (0x0c04, 'en'), (0x1404, 'en'), (0x0409, 'en'), (0x0411, 'en')):
            self.assertEqual(ui_language_for_langid(langid), lang, hex(langid))

    def test_chinese_window_starts_with_a_game_language_that_needs_no_shared_fonts(self):
        self.assertEqual(default_game_language('zh-Hans'), 'en-US')
        self.assertEqual(default_game_language('ru'), 'ru')


BREAK_CHILD = 'import time\nprint("ready", flush=True)\ntime.sleep(60)\n'


class CancelTest(unittest.TestCase):
    def test_ctrl_break_ends_the_process_group(self):
        p = Process([sys.executable, '-c', BREAK_CHILD])
        try:
            self.assertEqual(p.lines.get(timeout=20).strip(), 'ready')
            t0 = time.monotonic()
            p.cancel()
            code = None
            while code is None and time.monotonic() - t0 < 20:
                code = p.poll()
                time.sleep(0.05)
            self.assertIsNotNone(code, 'the child ignored Ctrl+Break')
            self.assertNotEqual(code, 0)
        finally:
            p.kill()


class SmokeTest(unittest.TestCase):
    def test_every_screen_in_both_languages(self):
        try:
            import tkinter  # noqa: F401
        except ImportError:
            self.skipTest('no tkinter')
        r = subprocess.run([sys.executable, '-m', 'installer.gui', '--selftest'], cwd=ROOT, timeout=120,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])


def make_fixture():
    """build/inst-gui/e2e-install: links to install-test (see the module docstring), with the step keys of the state
    set to the ones this checkout computes, so the install counts as current for it."""
    if not (FIXTURE / statefile.NAME).is_file():
        shutil.rmtree(FIXTURE, ignore_errors=True)
        for folder, _dirs, files in os.walk(INSTALLED):
            target = FIXTURE / Path(folder).relative_to(INSTALLED)
            target.mkdir(parents=True, exist_ok=True)
            for name in files:
                src, dst = Path(folder) / name, target / name
                if name in COPIED and target == FIXTURE:
                    shutil.copy2(src, dst)
                else:
                    os.link(src, dst)
    layout = layouts.find(ROOT)
    state = statefile.load(FIXTURE)
    options = SimpleNamespace(base=None, update=None, keys=None, lang=None)
    ctx = Context(layout, FIXTURE, ROOT / 'build' / 'inst-gui' / 'e2e-temp', Reporter(out=io.StringIO()), options, state=state)
    ctx.steps = all_steps()
    ctx.plan()
    for step_id, record in state['steps'].items():
        record['key'] = ctx.keys[step_id]
    statefile.save(FIXTURE, state)
    return FIXTURE


@unittest.skipUnless((INSTALLED / statefile.NAME).is_file(), 'build/install-test is absent')
class EndToEndTest(unittest.TestCase):
    @staticmethod
    def run_cli(form, temp, tries=1):
        """Runs the CLI in --verify mode through Process, up to `tries` times while the RomFS hash reads wrong."""
        for attempt in range(tries):
            p = Process(form.command(sys.executable, root=ROOT, temp=temp, verify_hashes=True), cwd=ROOT)
            model, stray = Model(), []
            t0, code = time.monotonic(), None
            while code is None and time.monotonic() - t0 < 600:
                try:
                    line = p.lines.get(timeout=0.2)
                except queue.Empty:
                    line = ''
                if line:
                    event = parse_line(line)
                    if event:
                        model.apply(event)
                    elif line.strip():
                        stray.append(line)
                code = p.poll()
            p.kill()
            model.finish(code)
            if not any(w[0] == 'state.output_changed' for w in model.warnings):
                break
        return code, model, stray, time.monotonic() - t0

    def test_verify_run_skips_every_step(self):
        form = Form(dest=str(make_fixture()))
        self.assertTrue(form.update_mode())
        code, model, stray, elapsed = self.run_cli(form, ROOT / 'build' / 'inst-gui' / 'e2e-temp', tries=3)
        print(f'\nend-to-end: exit {code}, result {model.result}, {elapsed:.1f} s, states {model.states}')
        if any(w[0] == 'state.output_changed' for w in model.warnings):
            self.skipTest('the RomFS hash read wrong three times (page cache of the faulty RAM, see the module docstring)')
        self.assertEqual(code, 0, stray[-5:] or model.error)
        self.assertEqual(model.result, 'ok')
        self.assertIn(model.mode, ('resume', 'update'))
        self.assertEqual({s: model.states[s] for s in model.steps if s not in ('check', 'cleanup')},
                         {s: SKIPPED for s in model.steps if s not in ('check', 'cleanup')})
        self.assertIsNotNone(model.space)
        self.assertEqual(stray, [])

    def test_cancel_during_the_check(self):
        cmd = Form(dest=str(make_fixture())).command(sys.executable, root=ROOT, temp=ROOT / 'build' / 'inst-gui' / 'e2e-temp',
                                                     verify_hashes=True)
        p = Process(cmd, cwd=ROOT)
        model, t0, code, cancelled = Model(), time.monotonic(), None, False
        while code is None and time.monotonic() - t0 < 120:
            try:
                event = parse_line(p.lines.get(timeout=0.2))
            except queue.Empty:
                event = None
            if event:
                model.apply(event)
                if event.get('phase') == 'verify_romfs' and not cancelled:
                    cancelled = True
                    p.cancel()
            code = p.poll()
        p.kill()
        model.finish(code)
        self.assertTrue(cancelled)
        self.assertEqual((code, model.result), (3, 'cancelled'))
        self.assertIsNone(model.error)


if __name__ == '__main__':
    unittest.main(verbosity=2)
