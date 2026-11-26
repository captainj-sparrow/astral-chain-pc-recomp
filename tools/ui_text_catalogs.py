"""Generate, update and check the external text catalogs of runtime/ui_text.c (lang/*.po -> lang/*.mo).

    python tools/ui_text_catalogs.py            update the catalogs and compile them (build_game.py runs this)
    python tools/ui_text_catalogs.py --check    validate the committed files without writing anything
    python tools/ui_text_catalogs.py --root DIR another checkout

The English text is compiled into the runtime (text_en in runtime/ui_text.c) and is the fallback of every id; the ids
are the UiTextId enum of runtime/ui_text.h. Each lang/<tag>.po is one language: msgctxt is the id, msgid the English
source, msgstr the translation. The runtime reads lang/<tag>.mo, which this tool compiles; the .mo files are build
outputs (.gitignore), never committed.

An update works like gettext's msgmerge, so that changing the English text never stops a build:
- runtime/ui_text_keys.inc (the id names the runtime matches msgctxt against) and lang/messages.pot (the template a
  new language starts from) are written from the enum and the English text;
- an entry whose English source changed keeps its translation, marked fuzzy with the old source on a `#|` line: the
  .mo leaves it out, so the menu shows the new English until a translator checks it and removes the flag;
- an id new in the enum is added with an empty msgstr (English shows until it is translated);
- an id gone from the enum is dropped; translator comments (`#`, `#.`) stay with their entry.
A catalog that holds only the software keyboard ids (UI_SWKBD_*) is a keyboard catalog: the game's name-entry keyboard
uses it in that game language, and the menu does not offer it. Every other catalog gets every id.

Untranslated and fuzzy entries are reported as warnings. Errors, which fail the build: a malformed file, a header
without Language (equal to the file name), UTF-8 Content-Type or X-Astral-Language-Name, an unknown msgctxt, or a
translation whose printf conversions or length the runtime would reject (runtime/ui_text.c text_ok checks the same).
"""
import argparse
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_IDS = {'UI_MENU_FIRST', 'UI_MENU_LAST_ID', 'UI_TEXT_COUNT'}
SWKBD_KEYS = ('UI_SWKBD_TITLE', 'UI_SWKBD_PROMPT', 'UI_SWKBD_DEFAULT_NAME', 'UI_SWKBD_OK', 'UI_SWKBD_CANCEL',
              'UI_SWKBD_PAD_HINT')
# The system font each script needs (runtime/ui/menu.cpp fonts_up merges it into the menu's font atlas)
FONT_LANGUAGES = {'ja': 'ja', 'ko': 'ko', 'zh-Hans': 'zh-Hans', 'zh-Hant': 'zh-Hant'}
FORMAT = re.compile(r'%[0-9]*(?:\.[0-9]+)?[sdiufxX]')
MO_MAGIC = 0x950412DE
# runtime/ui_text.h UI_DESC_BYTES, less the 256 bytes the description pane keeps for a note
DESC_LIMIT = 2048 - 256
SWKBD_LIMITS = {'UI_SWKBD_TITLE': 128, 'UI_SWKBD_PROMPT': 258, 'UI_SWKBD_OK': 64, 'UI_SWKBD_CANCEL': 64,
                'UI_SWKBD_PAD_HINT': 160}


class Paths:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.header = self.root / 'runtime' / 'ui_text.h'
        self.source = self.root / 'runtime' / 'ui_text.c'
        self.keys = self.root / 'runtime' / 'ui_text_keys.inc'
        self.lang = self.root / 'lang'
        self.pot = self.lang / 'messages.pot'

    def rel(self, path):
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)


def paths_for(root=None):
    return Paths(ROOT if root is None else root)


# ---- the C sources

def strip_comments(text):
    """text without its C comments, string literals kept (a '//' inside a string is text)."""
    out, i, state = [], 0, None
    while i < len(text):
        c, n = text[i], text[i + 1] if i + 1 < len(text) else ''
        if state == 'line':
            if c == '\n':
                state = None
                out.append(c)
            i += 1
        elif state == 'block':
            if c == '*' and n == '/':
                state, i = None, i + 2
            else:
                if c == '\n':
                    out.append(c)
                i += 1
        elif state == '"':
            out.append(c)
            if c == '\\' and n:
                out.append(n)
                i += 2
                continue
            if c == '"':
                state = None
            i += 1
        elif c == '/' and n in '/*' and n:
            state, i = ('line' if n == '/' else 'block'), i + 2
        else:
            if c == '"':
                state = '"'
            out.append(c)
            i += 1
    return ''.join(out)


def enum_keys(paths):
    """The UiTextId names in enum order, without the range markers."""
    text = strip_comments(paths.header.read_text(encoding='utf-8'))
    start = text.index('typedef enum')
    body = text[start:text.index('} UiTextId;', start)]
    body = body[body.index('{') + 1:]
    keys = []
    for item in body.split(','):
        m = re.search(r'\b([A-Z][A-Z0-9_]*)\b', item)
        if m and m.group(1) not in SKIP_IDS:
            keys.append(m.group(1))
    if len(keys) != len(set(keys)):
        raise ValueError(f'{paths.rel(paths.header)}: duplicate UiTextId names')
    return keys


C_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\'}


def read_quoted(text, pos):
    """The C string literal (adjacent literals joined) at pos, and the position after it."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    parts = []
    while pos < len(text) and text[pos] == '"':
        pos += 1
        out = []
        while True:
            if pos >= len(text):
                raise ValueError('unterminated string')
            c = text[pos]
            if c == '"':
                pos += 1
                break
            if c == '\\':
                n = text[pos + 1] if pos + 1 < len(text) else ''
                if n not in C_ESCAPES:
                    raise ValueError(f'unknown escape \\{n}')
                out.append(C_ESCAPES[n])
                pos += 2
                continue
            out.append(c)
            pos += 1
        parts.append(''.join(out))
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if not parts:
        raise ValueError('expected a string')
    return ''.join(parts), pos


def split_args(args):
    parts, start, depth, quoted, i = [], 0, 0, False, 0
    while i < len(args):
        c = args[i]
        if quoted:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(args[start:i].strip())
            start = i + 1
        i += 1
    parts.append(args[start:].strip())
    return parts


def call_args(text, pos):
    if text[pos] != '(':
        raise ValueError('expected (')
    depth, quoted, i = 0, False, pos + 1
    while i < len(text):
        c = text[i]
        if quoted:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == '(':
            depth += 1
        elif c == ')':
            if depth == 0:
                return split_args(text[pos + 1:i])
            depth -= 1
        i += 1
    raise ValueError('unterminated call')


def block_at(text, pos):
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if not depth:
                return text[pos + 1:i]
    raise ValueError('unterminated block')


def english_text(paths):
    """{id: English text} from text_en in runtime/ui_text.c, the language names (LANG_NAMES) and the SWKBD row."""
    text = strip_comments(paths.source.read_text(encoding='utf-8'))
    entries = {}
    lm = re.search(r'#define\s+LANG_NAMES\b', text)
    if lm:
        macro = text[lm.end():text.index('\n\n', lm.end())]
        for m in re.finditer(r'\[([A-Z0-9_]+)\]\s*=\s*', macro):
            entries[m.group(1)] = read_quoted(macro, m.end())[0]
    m = re.search(r'static const char \*const text_en\[UI_TEXT_COUNT\]\s*=\s*\{', text)
    if not m:
        raise ValueError(f'{paths.rel(paths.source)}: text_en is missing')
    body = block_at(text, m.end() - 1)
    sw = re.search(r'\bSWKBD\s*\(', body)
    if sw:
        args = call_args(body, sw.end() - 1)
        if len(args) != len(SWKBD_KEYS):
            raise ValueError('text_en: SWKBD takes six strings')
        for key, arg in zip(SWKBD_KEYS, args):
            entries[key] = read_quoted(arg, 0)[0]
    for dm in re.finditer(r'\[([A-Z0-9_]+)\]\s*=\s*', body):
        entries[dm.group(1)] = read_quoted(body, dm.end())[0]
    return {k: v for k, v in entries.items() if v}


# ---- PO files

@dataclass
class Entry:
    msgid: str
    msgstr: str = ''
    fuzzy: bool = False
    previous: str | None = None             # the English source a fuzzy translation was made from (#| msgid)
    comments: list = field(default_factory=list)   # translator and extracted comment lines, kept as written


def po_unquote(line):
    """The string of a PO line holding one quoted literal (PO escapes are C's)."""
    value, pos = read_quoted(line, 0)
    if pos != len(line):
        raise ValueError(f'text after the string: {line!r}')
    return value


def parse_po(text):
    """(header, header comments, {msgctxt: Entry} in file order, errors). Flags, `#|` and comments belong to the entry
    that follows them; a comment or blank line after a msgstr ends that entry."""
    header, header_comments, entries, errors = None, [], {}, []
    cur = {}         # the entry being read: ctxt, id, str, and what preceded it
    pending = {'comments': [], 'fuzzy': False, 'previous': None}
    state = None     # the field the next continuation string extends

    def finish():
        nonlocal header, header_comments, state, cur
        if not cur:
            return
        ctxt, msgid, msgstr = cur.get('ctxt'), cur.get('id'), cur.get('str')
        if msgid is None or msgstr is None:
            errors.append(f'{ctxt or "entry"}: incomplete entry')
        elif ctxt is None and msgid == '':
            if header is not None:
                errors.append('two headers')
            header, header_comments = msgstr, cur['comments']
        elif ctxt is None:
            errors.append(f'entry without msgctxt: {msgid[:40]!r}')
        elif ctxt in entries:
            errors.append(f'duplicate msgctxt {ctxt}')
        else:
            entries[ctxt] = Entry(msgid, msgstr, cur['fuzzy'], cur['previous'], cur['comments'])
        cur, state = {}, None

    def start():
        nonlocal cur
        if cur and 'str' in cur:
            finish()
        if not cur:
            cur = {'comments': pending['comments'], 'fuzzy': pending['fuzzy'], 'previous': pending['previous']}
            pending.update(comments=[], fuzzy=False, previous=None)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            if 'str' in cur:
                finish()
            if line.startswith('#,'):
                pending['fuzzy'] |= 'fuzzy' in (f.strip() for f in line[2:].split(','))
            elif line.startswith('#| msgid '):
                try:
                    pending['previous'] = po_unquote(line[9:].strip())
                except ValueError as exc:
                    errors.append(str(exc))
            elif line.startswith('#|') or line.startswith('#~') or line.startswith('#:'):
                pass   # previous context, obsolete entries and source references are not kept
            elif line.startswith('#'):
                pending['comments'].append(line)
            continue
        try:
            if line.startswith('msgctxt '):
                start()
                cur['ctxt'], state = po_unquote(line[8:].strip()), 'ctxt'
            elif line.startswith('msgid_plural') or line.startswith('msgstr['):
                errors.append(f'plural forms are not used: {line[:40]!r}')
            elif line.startswith('msgid '):
                if 'ctxt' not in cur:
                    start()
                cur['id'], state = po_unquote(line[6:].strip()), 'id'
            elif line.startswith('msgstr '):
                if 'id' not in cur:
                    errors.append('msgstr without msgid')
                    continue
                cur['str'], state = po_unquote(line[7:].strip()), 'str'
            elif line.startswith('"'):
                if state is None:
                    errors.append('string outside an entry')
                else:
                    cur[state] += po_unquote(line)
            else:
                errors.append(f'unknown line {line[:40]!r}')
        except ValueError as exc:
            errors.append(str(exc))
    finish()
    if header is None:
        errors.append('no header entry (msgid "")')
    return header or '', header_comments, entries, errors


def parse_header(header):
    values = {}
    for line in header.splitlines():
        name, sep, value = line.partition(':')
        if sep:
            values[name.strip()] = value.strip()
    return values


def po_quote(value):
    out = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return f'"{out}"'


def render_header(header, comments):
    out = [c + '\n' for c in comments]
    out.append('msgid ""\nmsgstr ""\n')
    out += [po_quote(line) + '\n' for line in header.splitlines(keepends=True)]
    return ''.join(out) + '\n'


def render_po(header, header_comments, entries, keys):
    out = [render_header(header, header_comments)]
    for key in keys:
        e = entries.get(key)
        if e is None:
            continue
        out += [c + '\n' for c in e.comments]
        if e.fuzzy:
            out.append('#, fuzzy\n')
            if e.previous is not None and e.previous != e.msgid:
                out.append(f'#| msgid {po_quote(e.previous)}\n')
        out.append(f'msgctxt {po_quote(key)}\nmsgid {po_quote(e.msgid)}\nmsgstr {po_quote(e.msgstr)}\n\n')
    return ''.join(out).rstrip('\n') + '\n'   # one newline at the end, as Poedit and msgmerge write


def render_pot(english, keys):
    header = ('Project-Id-Version: astral-chain-pc-recomp\nLanguage: \nMIME-Version: 1.0\n'
              'Content-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: 8bit\n'
              'X-Astral-Language-Name: \n')
    comments = ['# The template of a new language for the in-game menu, the overlay and the game\'s name-entry keyboard.',
                '# Copy it to lang/<tag>.po, fill in Language and X-Astral-Language-Name (and X-Astral-Font-Lang for',
                '# ja, ko, zh-Hans or zh-Hant), translate the msgstr lines, and keep msgctxt and msgid as they are.']
    return render_po(header, comments, {k: Entry(english[k]) for k in keys}, keys)


def render_keys(keys):
    return '// Generated by tools/ui_text_catalogs.py from the UiTextId enum (runtime/ui_text.h); do not edit.\n' + \
        ''.join(f'[{key}] = "{key}",\n' for key in keys)


def is_keyboard_catalog(entries):
    return bool(entries) and all(k in SWKBD_KEYS for k in entries)


def merge(entries, english, keys):
    """The entries a catalog holds after an update against the English text (msgmerge)."""
    scope = [k for k in keys if k in SWKBD_KEYS] if is_keyboard_catalog(entries) else keys
    out = {}
    for key in scope:
        e = entries.get(key)
        src = english[key]
        if e is None:
            out[key] = Entry(src)
        elif e.msgid == src:
            out[key] = e
        else:
            # The English changed under the translation: keep it, flagged, with the source it was made from
            out[key] = Entry(src, e.msgstr, bool(e.msgstr), e.previous if e.fuzzy and e.previous else e.msgid,
                             e.comments)
            if not e.msgstr:
                out[key].previous = None
    return out


# ---- checks shared with the runtime (runtime/ui_text.c text_ok)

def format_specs(value):
    """The printf conversions of value in order (%% left out), or None for a '%' that starts none."""
    specs, i = [], 0
    while i < len(value):
        if value[i] != '%':
            i += 1
        elif value[i + 1:i + 2] == '%':
            i += 2
        else:
            m = FORMAT.match(value, i)
            if not m:
                return None
            specs.append(m.group(0))
            i = m.end()
    return specs


def utf16_units(value):
    return sum(2 if ord(c) > 0xFFFF else 1 for c in value)


def entry_errors(key, value, english):
    """What would make the runtime refuse this translation."""
    errors = []
    if format_specs(value) != format_specs(english):
        errors.append(f'{key}: printf conversions differ from the English {format_specs(english)}')
    if key == 'UI_SWKBD_DEFAULT_NAME' and (utf16_units(value) > 9 or any(c in value for c in '@%\\')):
        errors.append(f'{key}: at most 9 UTF-16 units, without @ % \\')
    if key in SWKBD_LIMITS and utf16_units(value) > SWKBD_LIMITS[key]:
        errors.append(f'{key}: longer than the software keyboard takes ({SWKBD_LIMITS[key]} UTF-16 units)')
    if key not in SWKBD_KEYS and len(value.encode('utf-8')) >= DESC_LIMIT:
        errors.append(f'{key}: too long for the description pane ({DESC_LIMIT} bytes)')
    return errors


def header_errors(name, header):
    meta = parse_header(header)
    lang = name[:-3]
    errors = []
    if meta.get('Language') != lang:
        errors.append(f'{name}: Language is {meta.get("Language")!r}, not the file name {lang!r}')
    if 'UTF-8' not in meta.get('Content-Type', '').upper():
        errors.append(f'{name}: Content-Type is not UTF-8')
    if not meta.get('X-Astral-Language-Name'):
        errors.append(f'{name}: no X-Astral-Language-Name')
    font = meta.get('X-Astral-Font-Lang', '')
    if font != FONT_LANGUAGES.get(lang, ''):
        errors.append(f'{name}: X-Astral-Font-Lang should be {FONT_LANGUAGES.get(lang, "absent")!r}, not {font!r}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,14}', lang):
        errors.append(f'{name}: the file name is not a language tag of at most 15 characters')
    return errors


def compiled(entries, english):
    """The translations a .mo carries: translated, not fuzzy, made from the current English."""
    return {k: e.msgstr for k, e in entries.items()
            if k in english and e.msgstr and not e.fuzzy and e.msgid == english[k]}


def render_mo(header, translations, english):
    """GNU MO, little endian, without a hash table; msgid is msgctxt EOT English, sorted bytewise as gettext does."""
    pairs = [('', header)] + sorted(((f'{k}\x04{english[k]}', v) for k, v in translations.items()),
                                    key=lambda kv: kv[0].encode('utf-8'))
    count = len(pairs)
    orig, trans = 28, 28 + 8 * count
    strings = trans + 8 * count
    table, blob = bytearray(), bytearray()
    for column in (0, 1):
        for pair in pairs:
            value = pair[column].encode('utf-8')
            table += struct.pack('<II', len(value), strings + len(blob))
            blob += value + b'\0'
    return struct.pack('<7I', MO_MAGIC, 0, count, orig, trans, 0, 0) + bytes(table) + bytes(blob)


# ---- the steps

def write_if_changed(path, data):
    old = path.read_bytes() if path.is_file() else None
    if old == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def load(paths):
    keys = enum_keys(paths)
    english = english_text(paths)
    missing = [k for k in keys if k not in english]
    if missing:
        raise ValueError(f'{paths.rel(paths.source)}: no English text for {", ".join(missing[:5])}')
    catalogs = {}
    for po in sorted(paths.lang.glob('*.po')) if paths.lang.is_dir() else []:
        catalogs[po.name] = parse_po(po.read_text(encoding='utf-8'))
    return keys, english, catalogs


def check(paths, keys, english, catalogs, check_mo):
    """(errors, warnings) of the files as they are."""
    errors, warnings = [], []
    if not paths.keys.is_file() or paths.keys.read_text(encoding='utf-8') != render_keys(keys):
        errors.append(f'{paths.rel(paths.keys)} is out of date')
    if not paths.pot.is_file() or paths.pot.read_text(encoding='utf-8') != render_pot(english, keys):
        errors.append(f'{paths.rel(paths.pot)} is out of date')
    for name, (header, _, entries, parse_errors) in catalogs.items():
        errors += [f'{name}: {e}' for e in parse_errors]
        errors += header_errors(name, header)
        errors += [f'{name}: unknown msgctxt {k}' for k in entries if k not in english]
        scope = SWKBD_KEYS if is_keyboard_catalog(entries) else keys
        if any(k not in entries or entries[k].msgid != english[k] for k in scope) or \
                any(k not in scope for k in entries if k in english):
            errors.append(f'{name}: not merged with the current English text (run tools/ui_text_catalogs.py)')
        for key, e in entries.items():
            if key in english and e.msgstr and not e.fuzzy:
                errors += [f'{name}: {msg}' for msg in entry_errors(key, e.msgstr, english[key])]
        untranslated = sum(1 for k in scope if k in entries and not entries[k].msgstr)
        fuzzy = sum(1 for k in scope if k in entries and entries[k].msgstr and entries[k].fuzzy)
        if untranslated or fuzzy:
            warnings.append(f'{name}: {untranslated} untranslated and {fuzzy} fuzzy entries show in English')
        if check_mo:
            mo = paths.lang / (name[:-3] + '.mo')
            if not mo.is_file() or mo.read_bytes() != render_mo(header, compiled(entries, english), english):
                errors.append(f'lang/{mo.name} is out of date')
    return errors, warnings


def update(paths, keys, english, catalogs):
    """Writes ui_text_keys.inc, messages.pot, the merged .po files and their .mo; returns what it changed."""
    changed = []
    if write_if_changed(paths.keys, render_keys(keys).encode('utf-8')):
        changed.append(paths.rel(paths.keys))
    if write_if_changed(paths.pot, render_pot(english, keys).encode('utf-8')):
        changed.append(paths.rel(paths.pot))
    names = set()
    for name, (header, header_comments, entries, parse_errors) in catalogs.items():
        if parse_errors:
            continue   # left as it is: check() reports why
        merged = merge(entries, english, keys)
        po = paths.lang / name
        if write_if_changed(po, render_po(header, header_comments, merged, keys).encode('utf-8')):
            changed.append(paths.rel(po))
        mo = paths.lang / (name[:-3] + '.mo')
        if write_if_changed(mo, render_mo(header, compiled(merged, english), english)):
            changed.append(paths.rel(mo))
        names.add(mo.name)
        catalogs[name] = (header, header_comments, merged, parse_errors)
    for stale in paths.lang.glob('*.mo') if paths.lang.is_dir() else []:
        if stale.name not in names:
            stale.unlink()
            changed.append(paths.rel(stale))
    return changed


def install(paths, dest):
    """Copies the compiled catalogs into dest (lang/ beside an astralchain executable) and removes the .mo files there
    that no catalog makes any more. Returns how many files it wrote."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {p.name for p in paths.lang.glob('*.mo')} if paths.lang.is_dir() else set()
    for old in dest.glob('*.mo'):
        if old.name not in wanted:
            old.unlink()
    n = 0
    for name in sorted(wanted):
        src, dst = paths.lang / name, dest / name
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
            n += 1
    return n


def refresh(root=None, quiet=False):
    """The update the builds run: 0 when the catalogs are usable, 1 (after printing why) when one is not."""
    paths = paths_for(root)
    keys, english, catalogs = load(paths)
    changed = update(paths, keys, english, catalogs)
    errors, warnings = check(paths, keys, english, catalogs, check_mo=True)
    if changed and not quiet:
        print(f'ui text catalogs: wrote {", ".join(changed)}')
    for w in warnings if not quiet else ():
        print(f'ui text catalogs: {w}')
    for e in errors:
        print(f'ui text catalogs: {e}')
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='validate the committed files without writing anything')
    ap.add_argument('--root', type=Path, help='checkout root (default: the repository of this script)')
    args = ap.parse_args()
    if not args.check:
        return refresh(args.root)
    paths = paths_for(args.root)
    keys, english, catalogs = load(paths)
    errors, warnings = check(paths, keys, english, catalogs, check_mo=False)
    for line in warnings + errors:
        print(line)
    if not errors:
        print(f'ok: {len(keys)} ids, {len(catalogs)} catalogs')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
