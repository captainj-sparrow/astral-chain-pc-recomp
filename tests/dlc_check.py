"""The DLCs (docs/dlc.md): that they are off unless the player asks, that the repository's DLCs are well formed, and that
the builder of astralchain.exe (runtime/dlc.c) writes what a second implementation writes.

    .venv\\Scripts\\python tests/dlc_check.py [--exe build/v101/astralchain.exe] [--romfs build/extracted/v101/romfs]

Three parts, each reported; a part whose inputs are missing says so and is skipped (the exit code is then still 0:
read the output).

1. No game files, no build. Off by default everywhere a default lives: the settings table, astralchain.ini.example,
   the two patch sets, which must follow the setting with the default 0 and hold every site that calls a quest helper.
   The release ships dlc/ and the installer copies it. Each DLC of dlc/: only XML and Markdown files, a manifest the
   port understands, ids in the ranges the game leaves free, no id used twice, a text for every language (a <t>
   without lang).
2. The build, no game files: astralchain.exe --probe-dlc (archives and DLC folders of the probe's own, broken ones
   too).
3. The build and the game files: the exe builds quest/quest.pkz and Text/Text.pkz of the player's RomFS with dlc/
   (AC_DLC_PROBE_DIR), and the result is compared with the writers of this file, which share no code with dlc.c:
   every entry of the game keeps its compressed bytes, every added or changed entry unpacks to the same bytes as this
   file's DAT, BXM and text writers give, and the archive is laid out as the game's (written back from its entries, it
   is the same file). The work folder is build/dlc-check/ (git ignores it: it holds the game's archives).
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ('USen', 'EUfr', 'EUde', 'EUes', 'EUit', 'EUru', 'JPja', 'KRko', 'CNzh', 'TWzh')
failures = []


def report(ok, what):
    print(('OK    ' if ok else 'FAIL  ') + what)
    if not ok:
        failures.append(what)


def unzstd(data, size):
    try:
        from compression import zstd   # Python 3.14
        return zstd.decompress(data)
    except ImportError:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)


def game_hash(name):
    return zlib.crc32(name.lower().encode()) & 0x7FFFFFFF


# ---- 1. off by default, the release, the repository's DLCs

def manifests():
    return sorted((ROOT / 'dlc').glob('*/dlc.xml'))


def part_static():
    cfg = (ROOT / 'runtime' / 'cfg.c').read_text(encoding='utf-8')
    m = re.search(r'\{"DLC", "([^"]*)", "([^"]*)", (\w+)\}', cfg)
    report(bool(m) and m.groups() == ('0', '0|1', 'RESTART'),
           f'runtime/cfg.c: the setting dlc is off by default and read at start ({m.groups() if m else "no row"})')
    deck = re.search(r'DECK\b[^;]*?\{(.*?)\};', cfg, re.S)
    report('"DLC"' not in (deck.group(1) if deck else ''), 'runtime/cfg.c: the Steam Deck profile does not switch it on')
    ini = (ROOT / 'astralchain.ini.example').read_text(encoding='utf-8')
    lines = [l for l in ini.splitlines() if re.match(r'^\s*#?\s*dlc\s*=', l)]
    report(lines == ['# dlc = 0'], f'astralchain.ini.example: one line, "# dlc = 0" ({lines})')

    table = (ROOT / 'recomp' / 'patches-v101.txt').read_text(encoding='utf-8')
    sets = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r'^set\s+(\w+)\s+(\w+)\s+(\d+)', table, re.M)}
    for name in ('quest', 'questmenu'):
        report(sets.get(name) == ('AC_DLC', '0'), f'patches-v101.txt: set {name} follows AC_DLC, default 0 ({sets.get(name)})')
    helpers = re.findall(r'^site\s+(\w+)\s+.*@fnz?:(rt_quest_\w+)', table, re.M)
    stray = sorted({h for s, h in helpers if s not in ('quest', 'questmenu')})
    report(helpers and not stray, f'patches-v101.txt: the {len(helpers)} sites that call a quest helper are in those two sets '
                                  f'({stray or "none elsewhere"})')
    dlc_c = (ROOT / 'runtime' / 'dlc.c').read_text(encoding='utf-8')
    report('started_on = cfg_bool("AC_DLC")' in dlc_c and 'scan_dir(have ? dir : NULL)' in dlc_c,
           'runtime/dlc.c: the scan reads the folder only with the setting on')

    release = (ROOT / 'tools' / 'make_release.py').read_text(encoding='utf-8')
    report("copy_tree(ROOT / 'dlc', out / 'dlc'" in release, 'tools/make_release.py ships dlc/ (both releases: build_sources)')
    report("'--probe-dlc'" in release, 'tools/make_release.py --prove runs --probe-dlc')
    steps = (ROOT / 'installer' / 'steps.py').read_text(encoding='utf-8')
    report("install_dlcs(ctx.layout.root / 'dlc', dest / 'dlc')" in steps, 'installer/steps.py copies dlc/ into the install folder')

    files = [p for p in (ROOT / 'dlc').rglob('*') if p.is_file()]
    odd = [str(p.relative_to(ROOT)) for p in files if p.suffix not in ('.xml', '.md')]
    report(not odd, f'dlc/ holds only .xml and .md files ({len(files)} files; others: {odd or "none"})')
    big = [str(p.relative_to(ROOT)) for p in files if p.stat().st_size > 256 * 1024]
    report(not big, f'no file of dlc/ is above 256 KiB ({big or "none"})')
    crlf = [str(p.relative_to(ROOT)) for p in files if b'\r' in p.read_bytes()]
    report(not crlf, f'the files of dlc/ have LF line endings ({crlf or "none with CR"})')
    report((ROOT / 'dlc' / 'README.md').is_file(), 'dlc/README.md, the guide, is there')

    actions = set(re.findall(r'"([A-Z_]+)"', re.search(r'action_names?\[[^\]]*\]\s*=\s*\{(.*?)\};',
                                                     (ROOT / 'runtime' / 'input_map.c').read_text(encoding='utf-8'),
                                                     re.S).group(1))) if (ROOT / 'runtime' / 'input_map.c').is_file() else set()
    seen = {'quest': {}, 'window': {}, 'training': {}, 'text': {}}
    live = 0
    report(bool(manifests()), f'dlc/ holds {len(manifests())} DLC(s)')
    for path in manifests():
        name = path.parent.name
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            report(False, f'{name}: dlc.xml parses ({e})')
            continue
        problems = []
        if root.tag != 'dlc':
            problems.append(f'root <{root.tag}>')
        for k in root:
            if k.tag == 'quest':
                qid, base = k.get('id', ''), k.get('base')
                if not re.fullmatch(r'c0[0-9a-f]{2}', qid) or qid == 'c050':
                    problems.append(f'quest id {qid!r} is not c0xx (a training) or is c050')
                if base is not None and not re.fullmatch(r'[0-9a-f]{4}', base):
                    problems.append(f'quest {qid}: base {base!r}')
                if base is None and not (path.parent / 'quest' / qid).is_dir():
                    problems.append(f'quest {qid}: neither a base nor a folder')
                key = ('quest', qid)
            elif k.tag == 'window':
                num = k.get('number', '')
                if not re.fullmatch(r'[a-f][0-9a-f]{3}', num):
                    problems.append(f'window number {num!r} is not a000..ffff')
                key = ('window', num)
            elif k.tag == 'training':
                if k.get('value') not in ('2', '7'):
                    problems.append(f'training value {k.get("value")!r} is not free in the game\'s menu (2, 7)')
                if not k.get('text') or not re.fullmatch(r'[0-9a-f]{4}', k.get('quest', '')):
                    problems.append('training without a text or a quest')
                key = ('training', k.get('value'))
            elif k.tag == 'text':
                ts = k.findall('t')
                if not k.get('file') or not k.get('id'):
                    problems.append('text without a file or an id')
                if sum(t.get('lang') is None for t in ts) != 1:
                    problems.append(f'text {k.get("id")}: one <t> without lang is needed (the text of every language '
                                    f'that has none of its own)')
                other = [t.get('lang') for t in ts if t.get('lang') and t.get('lang') not in LANGS]
                if other:
                    problems.append(f'text {k.get("id")}: unknown languages {other}')
                if k.get('live') == '1':
                    live += 1
                    for t in ts:
                        for field in re.findall(r'\{(\w+)\}', t.text or ''):
                            if actions and field.upper() not in actions:
                                problems.append(f'text {k.get("id")}: {{{field}}} is not an action name')
                key = ('text', (k.get('file'), (k.get('id') or '').lower()))
            else:
                problems.append(f'unknown element <{k.tag}>')
                continue
            if key[1] in seen[key[0]]:
                problems.append(f'{key[0]} {key[1]} is also in {seen[key[0]][key[1]]}')
            seen[key[0]][key[1]] = name
        for quest in sorted((path.parent / 'quest').glob('*/QuestData.xml')):
            try:
                q = ET.parse(quest).getroot()
            except ET.ParseError as e:
                problems.append(f'{quest.parent.name}/QuestData.xml: {e}')
                continue
            for c in q.iter('CommandList'):
                tags = [x.tag for x in c]
                if tags[:1] != ['typeIF'] or 'typeEXEC' not in tags:
                    problems.append(f'{quest.parent.name}: a command without typeIF first and a typeEXEC')
                    break
                typ = int(c.findtext('typeIF'))
                if typ > 38 and not 0x10000 <= typ < 0x50000:
                    problems.append(f'{quest.parent.name}: typeIF {typ} is neither the game\'s nor a host condition')
        report(not problems, f'{name}: manifest and quest scripts' + (': ' + '; '.join(problems) if problems else ''))
    report(len(seen['training']) <= 1, f'at most one training row over all DLCs ({len(seen["training"])}; the menu takes one)')
    report(len(seen['window']) <= 16 and live <= 4, f'{len(seen["window"])} windows (16 at most), {live} live texts (4 at most)')


# ---- 2. the probe

def part_probe(exe):
    if not exe.is_file():
        print(f'SKIP  --probe-dlc: {exe} is not built')
        return False
    # AC_DLC=0: an astralchain.ini next to the exe that switches DLCs on would skip the probe's checks of the off state
    # (the default itself is the settings table's, which the probe and part 1 read)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('AC_')}
    env['AC_DLC'] = '0'
    raw = subprocess.run([str(exe), '--probe-dlc'], capture_output=True, env=env, cwd=exe.parent, timeout=600)
    r = subprocess.CompletedProcess(raw.args, raw.returncode, raw.stdout.decode('utf-8', 'replace'),
                                    raw.stderr.decode('utf-8', 'replace'))
    tail = (r.stdout.strip().splitlines() or [''])[-2:]
    report(r.returncode == 0 and tail[-1] == 'probe-dlc: ok', f'--probe-dlc: exit {r.returncode}: {" / ".join(tail)}')
    # the probe feeds the builder damaged archives and manifests: what the builder logs of them must still be UTF-8,
    # which the Linux release's prove decodes strictly (a name or an XML excerpt of a damaged file is shown with '?')
    try:
        raw.stdout.decode('utf-8'), raw.stderr.decode('utf-8')
        report(True, f'--probe-dlc: its output ({len(raw.stderr) // 1024} KiB of log) is valid UTF-8')
    except UnicodeDecodeError as e:
        report(False, f'--probe-dlc: its output is not valid UTF-8 ({e})')
    report('probe-dlc: dlc = 0 here: checked that nothing is read' in r.stdout, '--probe-dlc checked the off state')
    for line in r.stderr.splitlines():
        if line.startswith('FAIL'):
            print('      ' + line)
    return r.returncode == 0


# ---- 3. the second implementation (the writers the study of the formats used, ported apart from dlc.c)

def pkz_raw(p):
    magic, ver, _, count, desc, names_len = struct.unpack_from('<4sIQIIQ', p, 0)
    names = p[desc + 32 * count:desc + 32 * count + names_len]
    out = []
    for i in range(count):
        name, comp_name, size, off, comp = struct.unpack_from('<IIQQQ', p, desc + 32 * i)
        assert names[comp_name:names.find(b'\0', comp_name)] == b'ZStandard'
        out.append((names[name:names.find(b'\0', name)].decode(), size, p[off:off + comp]))
    return magic, ver, out


def pkz_build(magic, ver, ents):
    def padded(s):
        s += b'\0'
        return s + b'\0' * (-len(s) % 8)

    blob = bytearray(padded(b'ZStandard'))
    name_off = []
    for e in ents:
        name_off.append(len(blob))
        blob += padded(e[0].encode())
    out = bytearray(0x20 + 32 * len(ents)) + blob
    for i, e in enumerate(ents):
        out += b'\0' * (-len(out) % 64)
        struct.pack_into('<IIQQQ', out, 0x20 + 32 * i, name_off[i], 0, e[1], len(out), len(e[2]))
        out += e[2]
    out += b'\0' * (-len(out) % 64)
    struct.pack_into('<4sIQIIQ', out, 0, magic, ver, len(out), len(ents), 0x20, len(blob))
    return bytes(out)


def dat_list(d):
    count, offs_t, _, names_t, sizes_t = struct.unpack_from('<5I', d, 4)
    width = struct.unpack_from('<I', d, names_t)[0]
    names = [d[names_t + 4 + i * width:names_t + 4 + (i + 1) * width].split(b'\0')[0].decode() for i in range(count)]
    offs = struct.unpack_from(f'<{count}I', d, offs_t)
    sizes = struct.unpack_from(f'<{count}I', d, sizes_t)
    return [(n, d[o:o + s]) for n, o, s in zip(names, offs, sizes)]


def dat_build(files):
    n = len(files)
    names = [f[0] for f in files]
    width = max(len(s) for s in names) + 1
    out = bytearray(0x20)
    offs_t = len(out)
    out += b'\0' * (4 * n)
    ext_t = len(out)
    for s in names:
        out += (s.rsplit('.', 1)[-1].encode() + b'\0\0\0\0')[:4]
    names_t = len(out)
    out += struct.pack('<I', width)
    for s in names:
        out += s.encode().ljust(width, b'\0')
    out += b'\0' * (-len(out) % 4)
    sizes_t = len(out)
    out += b'\0' * (4 * n)
    hash_t = len(out)
    p2 = 1
    while p2 < n:
        p2 <<= 1
    shift = min(31, 32 - (p2.bit_length() - 1))
    hashes = [game_hash(s) for s in names]
    order = sorted(range(n), key=lambda i: (hashes[i] >> shift, i))
    nb = 1 << (31 - shift)
    buckets = [0xFFFF] * nb
    for k, i in enumerate(order):
        if buckets[hashes[i] >> shift] == 0xFFFF:
            buckets[hashes[i] >> shift] = k
    out += struct.pack('<4I', shift, 0x10, 0x10 + 2 * nb, 0x10 + 2 * nb + 4 * n)
    out += struct.pack(f'<{nb}H', *buckets) + struct.pack(f'<{n}I', *(hashes[i] for i in order))
    out += struct.pack(f'<{n}H', *order)
    offs, sizes = [], []
    for _, body in files:
        if not body:
            offs.append(0), sizes.append(0)
            continue
        out += b'\0' * (-len(out) % 16)
        offs.append(len(out)), sizes.append(len(body))
        out += body
    out += b'\0' * (-len(out) % 16)
    struct.pack_into('<4s7I', out, 0, b'DAT\0', n, offs_t, ext_t, names_t, sizes_t, hash_t, 0)
    struct.pack_into(f'<{n}I', out, offs_t, *offs)
    struct.pack_into(f'<{n}I', out, sizes_t, *sizes)
    return bytes(out)


def bxm_write(tree, magic):
    order, first, i = [tree], {}, 0
    while i < len(order):
        first[id(order[i])] = len(order)
        order.extend(order[i][3])
        i += 1
    strings, str_off = bytearray(), {}

    def sref(s):
        if s == '':
            return 0xFFFF
        if s not in str_off:
            str_off[s] = len(strings)
            strings.extend(s.encode('utf-8') + b'\0')
        return str_off[s]

    rows, row_of, node_rows = [], {}, []
    for name, value, attrs, _ in order:
        key = (name, value, tuple(attrs.items()))
        if key not in row_of:
            row_of[key] = len(rows)
            rows.append((name, value))
            rows.extend(attrs.items())
        node_rows.append(row_of[key])
    packed = [(sref(a), sref(b)) for a, b in rows]
    out = bytearray(magic + b'\0\0\0\0') + struct.pack('>HHI', len(order), len(rows), len(strings))
    for n, r in zip(order, node_rows):
        out += struct.pack('>HHHH', len(n[3]), first[id(n)] if n[3] else len(order), len(n[2]), r)
    for a, b in packed:
        out += struct.pack('>HH', a, b)
    return bytes(out + strings)


def tree_of(e):
    return (e.tag, (e.text or '').strip(), dict(e.attrib), [tree_of(k) for k in e])


PTD_KEY = 0x26


def ptd_rows(d, p, n):
    out = []
    for i in range(n):
        r = p + 16 * i
        h, off, _, nb = struct.unpack_from('<4I', d, r)
        out.append((h, bytes((b - PTD_KEY) & 0xFF for b in d[r + off:r + off + nb]).decode('utf-16-le')[:-1]))
    return out


def ptd_read(d):
    magic, ver, key, n_ids, ids_at, n_sec, sec_at = struct.unpack_from('<4sIIIIII', d, 0)
    assert magic == b'PTD\0' and ver == 2 and key == PTD_KEY and ids_at == 0x1C
    sections = []
    for k in range(n_sec):
        s = sec_at + 20 * k
        h, n_lists, lists_at, n_cols, cols_at = struct.unpack_from('<5I', d, s)
        lists, cols = [], []
        for i in range(n_lists):
            r = s + lists_at + 12 * i
            lh, cnt, off = struct.unpack_from('<3I', d, r)
            lists.append((lh, list(struct.unpack_from(f'<{cnt}I', d, r + off))))
        for i in range(n_cols):
            r = s + cols_at + 12 * i
            ch, cnt, off = struct.unpack_from('<3I', d, r)
            cols.append((ch, ptd_rows(d, r + off, cnt)))
        sections.append((h, lists, cols))
    return {'ids': ptd_rows(d, ids_at, n_ids), 'sections': sections}


def ptd_put_rows(out, rows):
    at = len(out)
    out += b'\0' * (16 * len(rows))
    for i, (h, s) in enumerate(rows):
        e = bytes((b + PTD_KEY) & 0xFF for b in (s + '\0').encode('utf-16-le'))
        r = at + 16 * i
        struct.pack_into('<4I', out, r, h, len(out) - r, len(e) // 2, len(e))
        out += e


def ptd_write(t):
    ids, sections = t['ids'], t['sections']
    out = bytearray(0x1C)
    ptd_put_rows(out, ids)
    out += b'\0' * (-len(out) % 2)
    sec_at = len(out)
    struct.pack_into('<4sIIIIII', out, 0, b'PTD\0', 2, PTD_KEY, len(ids), 0x1C, len(sections), sec_at)
    out += b'\0' * (20 * len(sections))
    for k, (h, lists, cols) in enumerate(sections):
        s = sec_at + 20 * k
        lists_at = len(out)
        out += b'\0' * (12 * len(lists))
        for i, (lh, vals) in enumerate(lists):
            r = lists_at + 12 * i
            struct.pack_into('<3I', out, r, lh, len(vals), len(out) - r)
            out += struct.pack(f'<{len(vals)}I', *vals)
        cols_at = len(out)
        out += b'\0' * (12 * len(cols))
        for i, (ch, rows) in enumerate(cols):
            out += b'\0' * (-len(out) % 2)
            r = cols_at + 12 * i
            struct.pack_into('<3I', out, r, ch, len(rows), len(out) - r)
            ptd_put_rows(out, rows)
        struct.pack_into('<5I', out, s, h, len(lists), lists_at - s, len(cols), cols_at - s)
    return bytes(out)


def ptd_add(t, text_id, text):
    h = game_hash(text_id)
    t['ids'] = sorted([r for r in t['ids'] if r[0] != h] + [(h, text_id)])
    h0, lists, cols = t['sections'][0]
    cols = [(ch, sorted([r for r in rows if r[0] != h] + [(h, text if i == 0 else '')])) for i, (ch, rows) in enumerate(cols)]
    lists = [(lh, [v for v in vals if v != h] + [h]) for lh, vals in lists]
    t['sections'][0] = (h0, lists, cols)


def expected_quest(manifest_dir, quest, game):
    """[(name, bytes)] of the quest's DAT: the base's files under the new id, then the folder's over them."""
    qid, base = quest.get('id'), quest.get('base')
    files = []
    if base:
        files = [(n.replace(f'_{base}.', f'_{qid}.'), b) for n, b in dat_list(game[f'quest{base}.dat'])]
    folder = manifest_dir / 'quest' / qid
    for path in sorted(folder.iterdir(), key=lambda p: p.name.encode()) if folder.is_dir() else []:
        if not path.is_file():
            continue
        name = path.name[:-4] + '.bxm' if path.suffix.lower() == '.xml' else path.name
        at = next((i for i, f in enumerate(files) if f[0].lower() == name.lower()), None)
        if path.suffix.lower() == '.xml':
            magic = b'BXM\0' if at is not None and files[at][1][:4] == b'BXM\0' else b'XML\0'
            body = bxm_write(tree_of(ET.parse(path).getroot()), magic)
        else:
            body = path.read_bytes()
        if at is None:
            files.append((name, body))
        else:
            files[at] = (files[at][0], body)
    return files


def part_differential(exe, romfs):
    index = romfs / 'index.json'
    if not exe.is_file() or not index.is_file() or not (romfs / 'romfs.bin').is_file():
        print(f'SKIP  the builder against this file\'s writers: needs {exe} and the game files in {romfs}')
        return
    files = {f['path']: f for f in json.loads(index.read_text(encoding='utf-8'))['files']}
    work = ROOT / 'build' / 'dlc-check'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copytree(ROOT / 'dlc', work / 'dlc')
    game = {}
    with open(romfs / 'romfs.bin', 'rb') as rom:
        for rel in ('quest/quest.pkz', 'Text/Text.pkz'):
            rom.seek(files[rel]['offset'])
            game[rel] = rom.read(files[rel]['size'])
            (work / rel.split('/')[1]).write_bytes(game[rel])
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('AC_')}
    env['AC_DLC_PROBE_DIR'] = str(work)
    r = subprocess.run([str(exe), '--probe-dlc'], capture_output=True, encoding='utf-8', errors='replace', env=env,
                       cwd=exe.parent, timeout=600)
    report(r.returncode == 0, f'astralchain.exe builds both archives with dlc/ (exit {r.returncode})')
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return
    roots = [(p.parent, ET.parse(p).getroot()) for p in manifests()]

    # quests
    built = (work / 'out' / 'quest.pkz').read_bytes()
    o_raw, b_raw = pkz_raw(game['quest/quest.pkz']), pkz_raw(built)
    game_dats = {n: (s, d) for n, s, d in o_raw[2]}

    class Unpacked(dict):   # a quest of the game, unpacked when a DLC names it as its base
        def __missing__(self, name):
            self[name] = unzstd(game_dats[name][1], game_dats[name][0])
            return self[name]

    unpacked, want = Unpacked(), {}
    for folder, root in roots:
        for q in root.iter('quest'):
            want[f'quest{q.get("id")}.dat'] = dat_build(expected_quest(folder, q, unpacked))
    names = [e[0] for e in b_raw[2]]
    report(names == sorted(names, key=str.lower) and len(set(names)) == len(names), 'quest.pkz: entries sorted, none twice')
    report(set(names) == set(game_dats) | set(want), f'quest.pkz: the game\'s {len(game_dats)} quests and the DLCs\' '
                                                      f'{len(want)} ({len(names)} entries)')
    kept = [e for e in b_raw[2] if e[0] not in want]
    report(kept == [e for e in o_raw[2] if e[0] not in want], 'quest.pkz: every quest of the game keeps its compressed bytes')
    for name, body in want.items():
        have = next((unzstd(d, s) for n, s, d in b_raw[2] if n == name), None)
        report(have == body, f'quest.pkz: {name} is byte for byte this file\'s DAT ({len(body)} bytes'
                             + ('' if have == body else f', the exe\'s {len(have) if have else "missing"}') + ')')
    report(pkz_build(*b_raw) == built, 'quest.pkz: laid out as the game\'s archives')

    # texts
    built = (work / 'out' / 'Text.pkz').read_bytes()
    o_raw, b_raw = pkz_raw(game['Text/Text.pkz']), pkz_raw(built)
    texts = [(t.get('file'), t.get('id'), {k.get('lang'): (k.text or '').strip().replace('\\n', '\r\n') for k in t.findall('t')})
             for _, root in roots for t in root.iter('text')]
    report([e[0] for e in b_raw[2]] == [e[0] for e in o_raw[2]], f'Text.pkz: the same {len(o_raw[2])} entries, in the same order')
    same = changed = kept_n = 0
    wrong = []
    for (name, size, data), (_, bsize, bdata) in zip(o_raw[2], b_raw[2]):
        stem, lang = name[:-4].rsplit('_', 1)
        mine = [t for t in texts if t[0] == stem]
        if not mine:
            kept_n += (size, data) == (bsize, bdata)
            continue
        changed += 1
        t = ptd_read(unzstd(data, size))
        for _, tid, by_lang in mine:
            ptd_add(t, tid, by_lang.get(lang, by_lang.get(None, '')))
        if ptd_write(t) == unzstd(bdata, bsize):
            same += 1
        else:
            wrong.append(name)
    report(same == changed and changed > 0, f'Text.pkz: the {changed} text files the DLCs add to are byte for byte this '
                                            f'file\'s ({wrong or "none differs"})')
    report(kept_n == len(o_raw[2]) - changed, f'Text.pkz: the {len(o_raw[2]) - changed} other files keep their compressed '
                                              f'bytes ({kept_n})')
    report(pkz_build(*b_raw) == built, 'Text.pkz: laid out as the game\'s archives')
    shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--exe', type=Path, default=ROOT / 'build' / 'v101' / ('astralchain.exe' if os.name == 'nt' else 'astralchain'))
    ap.add_argument('--romfs', type=Path, default=ROOT / 'build' / 'extracted' / 'v101' / 'romfs')
    a = ap.parse_args()
    print('== 1. off by default, the release, the repository\'s DLCs')
    part_static()
    print('== 2. --probe-dlc')
    part_probe(a.exe)
    print('== 3. the builder against a second implementation (game files)')
    part_differential(a.exe, a.romfs)
    if failures:
        print(f'\n{len(failures)} check(s) failed')
        sys.exit(1)
    print('\nall checks passed')


if __name__ == '__main__':
    main()
