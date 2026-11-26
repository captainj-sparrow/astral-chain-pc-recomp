"""tools/ui_text_catalogs.py without game files: PO parsing, the msgmerge-style update, the checks and the MO output.

    python tests/ui_text_catalogs_test.py

Each test builds a small checkout (runtime/ui_text.h, runtime/ui_text.c, lang/) in a temporary folder, so it covers
the tool's own rules rather than the committed catalogs (`tools/ui_text_catalogs.py --check` covers those).
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import ui_text_catalogs as u  # noqa: E402

HEADER_H = '''typedef enum {
    UI_SWKBD_TITLE, UI_SWKBD_PROMPT, UI_SWKBD_DEFAULT_NAME, UI_SWKBD_OK, UI_SWKBD_CANCEL, UI_SWKBD_PAD_HINT,
    UI_MENU_FIRST,
    UI_MENU_TITLE = UI_MENU_FIRST,
    UI_MENU_RATE,%s
    UI_MENU_LAST_ID = UI_MENU_RATE,
    UI_TEXT_COUNT
} UiTextId;
'''
SOURCE_C = '''#define SWKBD(title, prompt, name, ok, cancel, hint) x
static const char *const text_en[UI_TEXT_COUNT] = {
    SWKBD("Text entry", "Enter a name", "Horse", "OK", "Cancel", "Controller: A = OK, B = Cancel"),
    [UI_MENU_TITLE] = "Settings",
    [UI_MENU_RATE] = "%s",%s
};
'''


def po_header(lang, font=''):
    lines = [f'Language: {lang}\\n', 'Content-Type: text/plain; charset=UTF-8\\n', 'X-Astral-Language-Name: Test\\n']
    if font:
        lines.append(f'X-Astral-Font-Lang: {font}\\n')
    return 'msgid ""\nmsgstr ""\n' + ''.join(f'"{line}"\n' for line in lines) + '\n'


def entry(key, msgid, msgstr, pre=''):
    return f'{pre}msgctxt "{key}"\nmsgid "{msgid}"\nmsgstr "{msgstr}"\n\n'


class Checkout:
    def __init__(self, rate='%.0f fps', extra_id=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'runtime').mkdir()
        (self.root / 'lang').mkdir()
        self.write_sources(rate, extra_id)
        self.paths = u.paths_for(self.root)

    def write_sources(self, rate, extra_id=None):
        enum_extra = f'\n    {extra_id},' if extra_id else ''
        text_extra = f'\n    [{extra_id}] = "New text",' if extra_id else ''
        (self.root / 'runtime' / 'ui_text.h').write_text(HEADER_H % enum_extra, encoding='utf-8')
        (self.root / 'runtime' / 'ui_text.c').write_text(SOURCE_C % (rate, text_extra), encoding='utf-8')

    def po(self, name, text):
        (self.root / 'lang' / name).write_text(text, encoding='utf-8')

    def read(self, name):
        return (self.root / 'lang' / name).read_text(encoding='utf-8')

    def close(self):
        self.tmp.cleanup()


def menu_po(lang='it', rate_str='%.0f fps', rate_tr='%.0f fps', title_tr='Impostazioni', pre_rate=''):
    return (po_header(lang) + entry('UI_SWKBD_TITLE', 'Text entry', 'Inserimento testo') +
            entry('UI_SWKBD_PROMPT', 'Enter a name', '') + entry('UI_SWKBD_DEFAULT_NAME', 'Horse', 'Cavallo') +
            entry('UI_SWKBD_OK', 'OK', 'OK') + entry('UI_SWKBD_CANCEL', 'Cancel', 'Annulla') +
            entry('UI_SWKBD_PAD_HINT', 'Controller: A = OK, B = Cancel', '') +
            entry('UI_MENU_TITLE', 'Settings', title_tr) + entry('UI_MENU_RATE', rate_str, rate_tr, pre_rate))


def mo_strings(data):
    count, orig, trans = struct.unpack_from('<III', data, 8)
    def get(table, i):
        n, off = struct.unpack_from('<II', data, table + 8 * i)
        return data[off:off + n].decode('utf-8')
    return {get(orig, i): get(trans, i) for i in range(count)}


class ParseTest(unittest.TestCase):
    def test_a_fuzzy_flag_belongs_to_the_entry_after_it(self):
        text = po_header('it') + entry('A', 'a', 'aa') + entry('B', 'b', 'bb', '#, fuzzy\n') + entry('C', 'c', 'cc')
        _, _, entries, errors = u.parse_po(text)
        self.assertEqual(errors, [])
        self.assertEqual({k: (e.msgstr, e.fuzzy) for k, e in entries.items()},
                         {'A': ('aa', False), 'B': ('bb', True), 'C': ('cc', False)})

    def test_comments_and_the_previous_source_stay_with_their_entry(self):
        text = po_header('it') + entry('A', 'a', 'aa', '# a translator note\n#, fuzzy\n#| msgid "old a"\n')
        _, _, entries, _ = u.parse_po(text)
        e = entries['A']
        self.assertEqual((e.comments, e.fuzzy, e.previous), (['# a translator note'], True, 'old a'))
        self.assertEqual(u.render_po(*u.parse_po(text)[:3], ['A']), text.rstrip('\n') + '\n')

    def test_continuation_lines_and_escapes(self):
        text = po_header('it') + 'msgctxt "A"\nmsgid ""\n"line one\\n"\n"line \\"two\\""\nmsgstr "x"\n'
        _, _, entries, errors = u.parse_po(text)
        self.assertEqual(errors, [])
        self.assertEqual(entries['A'].msgid, 'line one\nline "two"')

    def test_malformed_files_are_reported(self):
        for text in ('msgctxt "A"\nmsgid "a"\nmsgstr "aa"\n',                      # no header
                     po_header('it') + 'msgid "a"\nmsgstr "aa"\n',                  # no msgctxt
                     po_header('it') + entry('A', 'a', 'x') + entry('A', 'a', 'y'),  # duplicate
                     po_header('it') + 'msgctxt "A"\nmsgid "a"\nmsgstr[0] "x"\n'):  # plural
            self.assertTrue(u.parse_po(text)[3], text)


class UpdateTest(unittest.TestCase):
    def setUp(self):
        self.c = Checkout()

    def tearDown(self):
        self.c.close()

    def test_a_clean_catalog_compiles_its_translated_entries(self):
        self.c.po('it.po', menu_po())
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)
        mo = mo_strings((self.c.root / 'lang' / 'it.mo').read_bytes())
        self.assertEqual(mo['UI_MENU_TITLE\x04Settings'], 'Impostazioni')
        self.assertNotIn('UI_SWKBD_PROMPT\x04Enter a name', mo)   # untranslated: English at run time
        self.assertIn('Language: it', mo[''])

    def test_a_changed_english_source_turns_the_translation_fuzzy_and_keeps_it(self):
        self.c.po('it.po', menu_po())
        self.c.write_sources('%.0f frames per second')
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)   # the build goes on
        _, _, entries, _ = u.parse_po(self.c.read('it.po'))
        e = entries['UI_MENU_RATE']
        self.assertEqual((e.msgid, e.msgstr, e.fuzzy, e.previous),
                         ('%.0f frames per second', '%.0f fps', True, '%.0f fps'))
        mo = mo_strings((self.c.root / 'lang' / 'it.mo').read_bytes())
        self.assertFalse(any(k.startswith('UI_MENU_RATE') for k in mo))   # fuzzy: left out

    def test_a_new_id_is_added_untranslated(self):
        self.c.po('it.po', menu_po())
        self.c.write_sources('%.0f fps', 'UI_MENU_NEW')
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)
        _, _, entries, _ = u.parse_po(self.c.read('it.po'))
        self.assertEqual((entries['UI_MENU_NEW'].msgid, entries['UI_MENU_NEW'].msgstr), ('New text', ''))
        self.assertIn('msgctxt "UI_MENU_NEW"', self.c.read('messages.pot'))

    def test_a_keyboard_catalog_gets_no_menu_ids(self):
        kbd = po_header('ja', 'ja') + entry('UI_SWKBD_DEFAULT_NAME', 'Horse', 'ウマ')
        self.c.po('ja.po', kbd)
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)
        _, _, entries, _ = u.parse_po(self.c.read('ja.po'))
        self.assertEqual(sorted(entries), sorted(u.SWKBD_KEYS))

    def test_untranslated_entries_with_conversions_do_not_fail(self):
        self.c.po('it.po', menu_po(rate_tr=''))
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)

    def test_what_the_runtime_would_refuse_fails(self):
        for po in (menu_po(rate_tr='%d fps'),                                   # another conversion
                   menu_po(rate_tr='100%'),                                     # a lone % at the end
                   menu_po().replace('"Cavallo"', '"Cavallo@Cavallo"'),         # keyboard name rules
                   menu_po(title_tr='x' * 1800),                                # description pane
                   menu_po().replace('Language: it', 'Language: fr')):          # header
            self.c.po('it.po', po)
            with self.subTest(po=po[-80:]):
                self.assertEqual(u.refresh(self.c.root, quiet=True), 1)

    def test_an_id_gone_from_the_enum_is_dropped_and_check_names_it(self):
        self.c.po('it.po', menu_po() + entry('UI_MENU_GONE', 'x', 'y'))
        keys, english, catalogs = u.load(self.c.paths)
        errors, _ = u.check(self.c.paths, keys, english, catalogs, check_mo=False)
        self.assertIn('it.po: unknown msgctxt UI_MENU_GONE', errors)
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)
        self.assertNotIn('UI_MENU_GONE', self.c.read('it.po'))

    def test_check_finds_an_unmerged_catalog_and_writes_nothing(self):
        self.c.po('it.po', menu_po())
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)
        self.c.write_sources('%.0f frames per second')
        before = self.c.read('it.po')
        keys, english, catalogs = u.load(self.c.paths)
        errors, _ = u.check(self.c.paths, keys, english, catalogs, check_mo=False)
        self.assertTrue(any('not merged' in e for e in errors))
        self.assertEqual(self.c.read('it.po'), before)

    def test_install_copies_the_catalogs_and_drops_stale_ones(self):
        self.c.po('it.po', menu_po())
        self.assertEqual(u.refresh(self.c.root, quiet=True), 0)
        dest = self.c.root / 'out' / 'lang'
        dest.mkdir(parents=True)
        (dest / 'xx.mo').write_bytes(b'old')
        self.assertEqual(u.install(self.c.paths, dest), 1)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ['it.mo'])
        self.assertEqual(u.install(self.c.paths, dest), 0)   # nothing changed


class MoTest(unittest.TestCase):
    def test_the_layout_gettext_reads(self):
        data = u.render_mo('Language: it\n', {'B': 'bb', 'A': 'aa'}, {'A': 'a', 'B': 'b'})
        magic, rev, count, orig, trans = struct.unpack_from('<5I', data)
        self.assertEqual((magic, rev, count, orig, trans), (0x950412DE, 0, 3, 28, 28 + 8 * 3))
        self.assertEqual(list(mo_strings(data)), ['', 'A\x04a', 'B\x04b'])   # header first, then sorted
        self.assertEqual(u.render_mo('Language: it\n', {'B': 'bb', 'A': 'aa'}, {'A': 'a', 'B': 'b'}), data)


if __name__ == '__main__':
    unittest.main()
