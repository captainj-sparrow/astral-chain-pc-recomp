"""Generate and check the external catalogs of runtime/ui_text.c.

    python tools/ui_text_catalogs.py
    python tools/ui_text_catalogs.py --check
    python tools/ui_text_catalogs.py --root /path/to/checkout

The first form rebuilds runtime/ui_text_keys.inc, lang/messages.pot, lang/*.po and lang/*.mo from the compiled English
fallback in runtime/ui_text.c and the translations already in lang/. Every entry carries the English source in a
gettext extracted comment. English is built in, so no en.po is written; pt uses pt-BR. The second validates the
committed files against the enum in runtime/ui_text.h and needs no C rows.
"""
import argparse
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Paths:
    def __init__(self, root):
        self.root = root.resolve()
        self.header = self.root / 'runtime' / 'ui_text.h'
        self.source = self.root / 'runtime' / 'ui_text.c'
        self.keys = self.root / 'runtime' / 'ui_text_keys.inc'
        self.lang = self.root / 'lang'

    def rel(self, path):
        try:
            return path.relative_to(self.root)
        except ValueError:
            return path


def paths_for(root=None):
    return Paths(ROOT if root is None else Path(root))

SKIP_IDS = {'UI_MENU_FIRST', 'UI_MENU_LAST_ID', 'UI_TEXT_COUNT'}
SWKBD_KEYS = ('UI_SWKBD_TITLE', 'UI_SWKBD_PROMPT', 'UI_SWKBD_DEFAULT_NAME', 'UI_SWKBD_OK', 'UI_SWKBD_CANCEL',
              'UI_SWKBD_PAD_HINT')
C_LANGUAGES = {
    'text_en': 'en', 'text_it': 'it', 'text_ru': 'ru', 'text_pt_br': 'pt-BR', 'text_zh_hans': 'zh-Hans',
    'text_ja': 'ja', 'text_fr': 'fr', 'text_de': 'de', 'text_es': 'es', 'text_nl': 'nl',
    'text_zh_hant': 'zh-Hant', 'text_ko': 'ko',
}
CATALOG_LANGS = tuple(lang for lang in C_LANGUAGES.values() if lang != 'en')
FULL_LANGUAGES = ('it', 'ru', 'pt-BR', 'zh-Hans')
FONT_LANGUAGES = {'ja': 'ja', 'ko': 'ko', 'zh-Hans': 'zh-Hans', 'zh-Hant': 'zh-Hant'}
LANGUAGE_NAME_IDS = {
    'en': 'UI_MENU_LANGNAME_EN', 'ja': 'UI_MENU_LANGNAME_JA', 'fr': 'UI_MENU_LANGNAME_FR',
    'de': 'UI_MENU_LANGNAME_DE', 'it': 'UI_MENU_LANGNAME_IT', 'es': 'UI_MENU_LANGNAME_ES',
    'nl': 'UI_MENU_LANGNAME_NL', 'ru': 'UI_MENU_LANGNAME_RU', 'zh-Hans': 'UI_MENU_LANGNAME_ZH_HANS',
    'zh-Hant': 'UI_MENU_LANGNAME_ZH_HANT', 'ko': 'UI_MENU_LANGNAME_KO', 'pt-BR': 'UI_MENU_LANGNAME_PT_BR',
}
FORMAT = re.compile(r'%[0-9]*(?:\.[0-9]+)?[sdiufxX%]')
MO_MAGIC = 0x950412DE


def strip_comments(text):
    out = []
    i = 0
    state = None
    while i < len(text):
        c = text[i]
        n = text[i + 1] if i + 1 < len(text) else ''
        if state == 'line':
            if c == '\n':
                state = None
                out.append(c)
            i += 1
            continue
        if state == 'block':
            if c == '*' and n == '/':
                state = None
                i += 2
            else:
                if c == '\n':
                    out.append(c)
                i += 1
            continue
        if state == '"':
            out.append(c)
            if c == '\\' and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                state = None
            i += 1
            continue
        if c == '/' and n == '/':
            state = 'line'
            i += 2
            continue
        if c == '/' and n == '*':
            state = 'block'
            i += 2
            continue
        if c == '"':
            state = '"'
        out.append(c)
        i += 1
    return ''.join(out)


def enum_keys(paths):
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
        raise ValueError('duplicate enum keys')
    return keys


def read_quoted(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    parts = []
    while pos < len(text) and text[pos] == '"':
        pos += 1
        out = []
        while pos < len(text):
            c = text[pos]
            if c == '"':
                pos += 1
                break
            if c == '\\':
                if pos + 1 >= len(text):
                    raise ValueError('truncated escape')
                n = text[pos + 1]
                if n == 'n':
                    out.append('\n')
                elif n == 't':
                    out.append('\t')
                elif n == 'r':
                    out.append('\r')
                elif n in ('"', '\\'):
                    out.append(n)
                else:
                    raise ValueError(f'unknown escape \\{n}')
                pos += 2
                continue
            out.append(c)
            pos += 1
        else:
            raise ValueError('unterminated string')
        parts.append(''.join(out))
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if not parts:
        raise ValueError('expected a string')
    return ''.join(parts), pos


def split_args(args):
    parts = []
    start = 0
    depth = 0
    quoted = False
    i = 0
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
    depth = 0
    start = pos + 1
    quoted = False
    i = start
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
                return split_args(text[start:i]), i + 1
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
            if depth == 0:
                return text[pos + 1:i], i + 1
    raise ValueError('unterminated block')


def compiled_catalogs(paths):
    text = strip_comments(paths.source.read_text(encoding='utf-8'))
    lm = re.search(r'#define\s+LANG_NAMES\b', text)
    lang_names = {}
    if lm:
        end = text.index('\n\n', lm.end())
        macro = text[lm.end():end]
        for m in re.finditer(r'\[([A-Z0-9_]+)\]\s*=\s*', macro):
            value, _ = read_quoted(macro, m.end())
            lang_names[m.group(1)] = value

    catalogs = {}
    for m in re.finditer(r'static const char \*const (\w+)\[UI_TEXT_COUNT\]\s*=\s*\{', text):
        name = m.group(1)
        if name not in C_LANGUAGES:
            continue
        body, _ = block_at(text, m.end() - 1)
        entries = dict(lang_names)
        sw = re.search(r'\bSWKBD\s*\(', body)
        if sw:
            args, _ = call_args(body, sw.end() - 1)
            if len(args) != len(SWKBD_KEYS):
                raise ValueError(f'{name}: bad SWKBD arguments')
            for key, arg in zip(SWKBD_KEYS, args):
                value, _ = read_quoted(arg, 0)
                entries[key] = value
        for dm in re.finditer(r'\[([A-Z0-9_]+)\]\s*=\s*', body):
            value, _ = read_quoted(body, dm.end())
            entries[dm.group(1)] = value
        catalogs[C_LANGUAGES[name]] = {k: v for k, v in entries.items() if v}

    for m in re.finditer(r'\bSWKBD_ONLY\s*\(', text):
        args, _ = call_args(text, m.end() - 1)
        name = args[0]
        if name not in C_LANGUAGES:
            continue
        values = args[1:]
        if len(values) != len(SWKBD_KEYS):
            raise ValueError(f'{name}: bad SWKBD_ONLY arguments')
        entries = dict(lang_names)
        for key, arg in zip(SWKBD_KEYS, values):
            value, _ = read_quoted(arg, 0)
            entries[key] = value
        catalogs[C_LANGUAGES[name]] = {k: v for k, v in entries.items() if v}

    if 'en' not in catalogs:
        raise ValueError('the compiled English row is missing')
    return catalogs


def po_quote(value):
    out = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    out = out.replace('\t', '\\t')
    return f'"{out}"'


def header_text(lang, name, font):
    lines = ['Project-Id-Version: astral-chain-pc-recomp\n', f'Language: {lang}\n',
             'Content-Type: text/plain; charset=UTF-8\n', f'X-Astral-Language-Name: {name}\n']
    if font:
        lines.append(f'X-Astral-Font-Lang: {font}\n')
    return ''.join(lines)


def english_comment(value):
    return f'#. English: {po_quote(value)}\n'


def render_po(lang, entries, keys, metadata, english):
    out = ['msgid ""\nmsgstr ""\n']
    for line in metadata.splitlines(keepends=True):
        out.append(po_quote(line) + '\n')
    out.append('\n')
    for key in keys:
        if key in entries:
            out.append(english_comment(english[key]))
            out.append(f'msgid {po_quote(key)}\nmsgstr {po_quote(entries[key])}\n\n')
    return ''.join(out)


def render_pot(entries, keys):
    metadata = 'Content-Type: text/plain; charset=UTF-8\nX-Astral-Language-Name: English\n'
    out = ['msgid ""\nmsgstr ""\n']
    for line in metadata.splitlines(keepends=True):
        out.append(po_quote(line) + '\n')
    out.append('\n')
    for key in keys:
        out.append(english_comment(entries[key]))
        out.append(f'msgid {po_quote(key)}\nmsgstr ""\n\n')
    return ''.join(out)


def render_keys(keys):
    return ''.join(f'[{key}] = "{key}",\n' for key in keys)


def parse_po(text):
    entries = {}
    header = ''
    key = None
    buf = []
    state = None
    fuzzy = False
    errors = []

    def finish():
        nonlocal header, key, buf, state, fuzzy
        value = ''.join(buf)
        if key == '':
            header = value
        elif key:
            if key in entries:
                errors.append(f'duplicate msgid {key}')
            elif not fuzzy:
                entries[key] = value
        key = None
        buf = []
        state = None
        fuzzy = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('~'):
            errors.append('obsolete entry')
            continue
        if line.startswith('#'):
            if line.startswith('#,') and 'fuzzy' in line:
                fuzzy = True
            continue
        if line.startswith('msgid_plural') or line.startswith('msgctxt') or line.startswith('msgstr['):
            errors.append(f'unsupported line {line!r}')
            continue
        try:
            if line.startswith('msgid '):
                if key is not None:
                    finish()
                key = read_quoted(line[6:], 0)[0]
                buf = []
                state = 'id'
            elif line.startswith('msgstr '):
                if key is None:
                    errors.append('msgstr without msgid')
                    continue
                buf = [read_quoted(line[7:], 0)[0]]
                state = 'str'
            elif line.startswith('"'):
                if state is None:
                    errors.append('string outside an entry')
                    continue
                buf.append(read_quoted(line, 0)[0])
            else:
                errors.append(f'unknown line {line!r}')
        except ValueError as exc:
            errors.append(str(exc))
    if key is not None:
        finish()
    return header, entries, errors


def english_comments(text):
    comments = {}
    pending = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith('#. English: '):
            pending = line[len('#. English: '):]
        elif line.startswith('msgid '):
            if pending is not None:
                try:
                    comments[read_quoted(line[6:], 0)[0]] = read_quoted(pending, 0)[0]
                except ValueError:
                    pass
            pending = None
        elif line.startswith('#') or line.startswith('"') or not line:
            continue
        else:
            pending = None
    return comments


def parse_header(header):
    values = {}
    for line in header.splitlines():
        name, sep, value = line.partition(':')
        if sep:
            values[name.strip()] = value.strip()
    return values


def format_specs(value):
    specs = []
    i = 0
    while i < len(value):
        if value[i] != '%':
            i += 1
            continue
        if i + 1 < len(value) and value[i + 1] == '%':
            i += 2
            continue
        m = FORMAT.match(value, i)
        if not m:
            return None
        specs.append(m.group(0))
        i = m.end()
    return specs


def utf16_units(value):
    return sum(2 if ord(c) > 0xFFFF else 1 for c in value)


def validate_entry(key, value, english, errors):
    expected = format_specs(english)
    actual = format_specs(value)
    if expected != actual:
        errors.append(f'{key}: printf conversions differ from English')
    if key == 'UI_SWKBD_DEFAULT_NAME':
        if utf16_units(value) > 9 or any(c in value for c in '@%\\'):
            errors.append(f'{key}: invalid software keyboard default name')
    limits = {'UI_SWKBD_TITLE': 128, 'UI_SWKBD_PROMPT': 258, 'UI_SWKBD_OK': 64, 'UI_SWKBD_CANCEL': 64,
              'UI_SWKBD_PAD_HINT': 160}
    if key in limits and utf16_units(value) > limits[key]:
        errors.append(f'{key}: too long for the software keyboard')
    if key.startswith('UI_MENU_') and len(value.encode('utf-8')) + 256 >= 2048:
        errors.append(f'{key}: too long for the description pane')


def write_if_changed(path, text):
    if path.is_file() and path.read_text(encoding='utf-8') == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    return True


def write_bytes_if_changed(path, data):
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def render_mo(header, entries):
    pairs = [('', header)] + sorted(entries.items(), key=lambda item: item[0].encode('utf-8'))
    count = len(pairs)
    orig_table = 28
    trans_table = orig_table + 8 * count
    strings = trans_table + 8 * count
    table = bytearray()
    blob = bytearray()
    for msgid, _ in pairs:
        value = msgid.encode('utf-8')
        table += struct.pack('<II', len(value), strings + len(blob))
        blob += value + b'\0'
    for _, msgstr in pairs:
        value = msgstr.encode('utf-8')
        table += struct.pack('<II', len(value), strings + len(blob))
        blob += value + b'\0'
    head = struct.pack('<7I', MO_MAGIC, 0, count, orig_table, trans_table, 0, strings)
    return head + bytes(table) + bytes(blob)


def compile_catalogs(paths):
    changed = False
    expected = {f'{lang}.mo' for lang in CATALOG_LANGS}
    for path in sorted(paths.lang.glob('*.mo')):
        if path.name not in expected:
            path.unlink()
            changed = True
    for lang in CATALOG_LANGS:
        po = paths.lang / f'{lang}.po'
        mo = paths.lang / f'{lang}.mo'
        if not po.is_file():
            if mo.is_file():
                mo.unlink()
                changed = True
            continue
        header, entries, errors = parse_po(po.read_text(encoding='utf-8'))
        if errors:
            raise ValueError(f'{po.name}: {errors[0]}')
        changed = write_bytes_if_changed(mo, render_mo(header, entries)) or changed
    return changed


def validate(keys, paths, check_mo=True):
    errors = []
    keys_file = render_keys(keys)
    if not paths.keys.is_file() or paths.keys.read_text(encoding='utf-8') != keys_file:
        errors.append(f'{paths.rel(paths.keys)} is out of date')

    files = sorted(p.name for p in paths.lang.glob('*.po')) if paths.lang.is_dir() else []
    if set(files) != {f'{lang}.po' for lang in CATALOG_LANGS}:
        errors.append(f'lang/ has unexpected catalogs: {files}')
    mo_files = sorted(p.name for p in paths.lang.glob('*.mo')) if paths.lang.is_dir() else []
    if check_mo and set(mo_files) != {f'{lang}.mo' for lang in CATALOG_LANGS}:
        errors.append(f'lang/ has unexpected compiled catalogs: {mo_files}')

    catalogs = {}
    comments = {}
    compiled = {}
    for name in files:
        text = (paths.lang / name).read_text(encoding='utf-8')
        header, entries, parse_errors = parse_po(text)
        comments[name] = english_comments(text)
        errors.extend(f'{name}: {e}' for e in parse_errors)
        meta = parse_header(header)
        lang = meta.get('Language', name[:-3])
        if lang != name[:-3]:
            errors.append(f'{name}: Language {lang!r} does not match the file name')
        if 'UTF-8' not in meta.get('Content-Type', '').upper():
            errors.append(f'{name}: Content-Type is not UTF-8')
        if not meta.get('X-Astral-Language-Name'):
            errors.append(f'{name}: no X-Astral-Language-Name')
        font = meta.get('X-Astral-Font-Lang', '')
        if font and font != FONT_LANGUAGES.get(lang):
            errors.append(f'{name}: unexpected X-Astral-Font-Lang {font!r}')
        if FONT_LANGUAGES.get(lang) and not font:
            errors.append(f'{name}: missing X-Astral-Font-Lang')
        unknown = sorted(set(entries) - set(keys))
        errors.extend(f'{name}: unknown msgid {k}' for k in unknown)
        catalogs[lang] = (meta, entries)
        compiled[lang] = render_mo(header, entries)

    if check_mo:
        for lang in sorted(compiled):
            path = paths.lang / f'{lang}.mo'
            if not path.is_file() or path.read_bytes() != compiled[lang]:
                errors.append(f'{lang}.mo is missing or out of date')

    english = {k: v for k, v in compiled_catalogs(paths)['en'].items() if k in set(keys)}
    if set(english) != set(keys):
        errors.append(f'compiled English: {len(set(keys) - set(english))} missing keys')

    for name in files:
        lang = name[:-3]
        meta, entries = catalogs.get(lang, ({}, {}))
        source_comments = comments.get(name, {})
        for key, value in entries.items():
            if key in english:
                if source_comments.get(key) != english[key]:
                    errors.append(f'{name}: {key}: English source comment is missing or wrong')
                validate_entry(key, value, english[key], errors)
        if lang in FULL_LANGUAGES:
            missing = [k for k in keys if k.startswith('UI_MENU_') and k not in entries]
            if missing:
                errors.append(f'{name}: {len(missing)} missing menu keys')
        if set(SWKBD_KEYS) - set(entries):
            errors.append(f'{name}: missing software keyboard keys')

    pot_file = paths.lang / 'messages.pot'
    if not pot_file.is_file():
        errors.append('lang/messages.pot is missing')
    else:
        pot_text = pot_file.read_text(encoding='utf-8')
        _, pot, pot_errors = parse_po(pot_text)
        pot_comments = english_comments(pot_text)
        errors.extend(f'messages.pot: {e}' for e in pot_errors)
        if set(pot) != set(english):
            errors.append('messages.pot does not match the compiled English fallback')
        for key in english:
            if pot_comments.get(key) != english[key]:
                errors.append(f'messages.pot: {key}: English source comment is missing or wrong')
    return errors


def generate(keys, paths):
    source = compiled_catalogs(paths)
    paths.lang.mkdir(parents=True, exist_ok=True)
    changed = False
    for name in ('en.po', 'pt.po'):
        path = paths.lang / name
        if path.is_file():
            path.unlink()
            changed = True
    keyset = set(keys)
    english = {k: v for k, v in source['en'].items() if k in keyset}
    catalogs = {}
    for path in sorted(paths.lang.glob('*.po')):
        header, entries, errors = parse_po(path.read_text(encoding='utf-8'))
        if errors:
            raise ValueError(f'{path.name}: {errors[0]}')
        catalogs[path.stem] = (header, {k: v for k, v in entries.items() if k in keyset})
    for lang, entries in source.items():
        if lang == 'en' or lang in catalogs:
            continue
        name = entries[LANGUAGE_NAME_IDS[lang]]
        catalogs[lang] = (header_text(lang, name, FONT_LANGUAGES.get(lang, '')), entries)
    changed = write_if_changed(paths.keys, render_keys(keys)) or changed
    changed = write_if_changed(paths.lang / 'messages.pot', render_pot(english, keys)) or changed
    for lang, (header, entries) in catalogs.items():
        text = render_po(lang, entries, keys, header, english)
        changed = write_if_changed(paths.lang / f'{lang}.po', text) or changed
    if changed:
        print(f'wrote {paths.rel(paths.keys)}, lang/messages.pot and {len(catalogs)} catalogs')
    return changed


def report(keys, paths):
    errors = validate(keys, paths)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f'ok: {len(keys)} ids, {len(list(paths.lang.glob("*.po")))} catalogs')
    return 0


def refresh(root=None):
    paths = paths_for(root)
    keys = enum_keys(paths)
    generate(keys, paths)
    errors = validate(keys, paths, check_mo=False)
    if errors:
        for error in errors:
            print(error)
        return 1
    if compile_catalogs(paths):
        print(f'wrote {len(CATALOG_LANGS)} compiled catalogs in lang/*.mo')
    return report(keys, paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='validate the committed files without regenerating them')
    parser.add_argument('--root', type=Path, help='checkout root (default: the repository of this script)')
    args = parser.parse_args()
    if not args.check:
        return refresh(args.root)
    paths = paths_for(args.root)
    keys = enum_keys(paths)
    return report(keys, paths)


if __name__ == '__main__':
    sys.exit(main())
