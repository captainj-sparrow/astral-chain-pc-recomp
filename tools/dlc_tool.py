"""Reads the player's own game files for writing a DLC (dlc/README.md, docs/dlc.md): the quests and texts of Astral
Chain 1.0.1 as plain XML and text, in the form runtime/dlc.c builds back into the game's archives.

    python tools/dlc_tool.py quests                      the quests of quest/quest.pkz, with their files
    python tools/dlc_tool.py quest c00b                  a readable listing of one quest's script
    python tools/dlc_tool.py quest c00b --out DIR        the quest's files into DIR: <name>.bxm as <name>.xml (the form a
                                                         DLC's quest/<id>/ folder takes), the others as they are
    python tools/dlc_tool.py commands                    every condition and action type of the 414 quests: its
                                                         parameters, how often it is used, its most frequent values
    python tools/dlc_tool.py texts Core_text TUTO_W      the texts of a text file whose id or text holds the pattern
                                                         (--lang USen by default)
    python tools/dlc_tool.py hash CORE_TUTO_W_TITLE_af00 the game's hash of a name (text ids, files of a DAT, flags)
    python tools/dlc_tool.py check dlc/pc-controls       a DLC folder's XML files parsed as the port parses them

--romfs names the folder with romfs.bin and index.json: the install folder's romfs/, or build/extracted/v101/romfs of a
checkout (the default). What `quest --out`, `quest` and `texts` print is the game's own content: keep it out of the
repository and of anything published (write it under build/ or notes/, which git ignores). A DLC holds only what its
author wrote; what it needs of the game is named in dlc.xml (base="...") and read from each player's files at start.

The formats (documented in runtime/dlc.c, whose writers were ported from these functions): PKZ, a zstd archive; DAT, a
file container; BXM, big-endian binary XML; PTD, a text table whose strings are UTF-16 with 0x26 added to every byte.
A hash is the crc32 of the lower-case name, 31 bits.
"""
import argparse
import json
import re
import struct
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

ROOT = Path(__file__).resolve().parent.parent
PTD_KEY = 0x26


def unzstd(data, size):
    try:
        from compression import zstd   # Python 3.14
        return zstd.decompress(data)
    except ImportError:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=size)


def game_hash(name):
    return zlib.crc32(name.lower().encode()) & 0x7FFFFFFF


class Rom:
    def __init__(self, folder):
        self.folder = Path(folder)
        index = self.folder / 'index.json'
        if not index.is_file() or not (self.folder / 'romfs.bin').is_file():
            sys.exit(f'{self.folder} holds no romfs.bin with its index.json: pass --romfs <the install folder>/romfs')
        self.files = {f['path']: f for f in json.loads(index.read_text(encoding='utf-8'))['files']}

    def read(self, rel):
        f = self.files.get(rel)
        if f is None:
            sys.exit(f'{rel} is not in the RomFS')
        with open(self.folder / 'romfs.bin', 'rb') as rom:
            rom.seek(f['offset'])
            return rom.read(f['size'])


def pkz_entries(p):
    """[(name, size, compressed bytes)] of a PKZ: header "pkzl", version, u64 size, count, the descriptors' offset,
    u64 string bytes; a descriptor is the name's and the compression name's offsets in the strings, then u64 size,
    data offset and compressed size."""
    magic, _, _, count, desc, names_len = struct.unpack_from('<4sIQIIQ', p, 0)
    if magic != b'pkzl':
        sys.exit('not a PKZ archive')
    names = p[desc + 32 * count:desc + 32 * count + names_len]
    out = []
    for i in range(count):
        name, _, size, off, comp = struct.unpack_from('<IIQQQ', p, desc + 32 * i)
        out.append((names[name:names.find(b'\0', name)].decode(), size, p[off:off + comp]))
    return out


def dat_files(d):
    """[(name, bytes)] of a DAT container, in file order."""
    if d[:4] != b'DAT\0':
        return []
    count, offs, _, names, sizes = struct.unpack_from('<5I', d, 4)
    width = struct.unpack_from('<I', d, names)[0]
    out = []
    for i in range(count):
        name = d[names + 4 + i * width:names + 4 + (i + 1) * width].split(b'\0')[0].decode('ascii', 'replace')
        off, size = struct.unpack_from('<I', d, offs + 4 * i)[0], struct.unpack_from('<I', d, sizes + 4 * i)[0]
        out.append((name, d[off:off + size]))
    return out


def bxm(b):
    """A BXM file as nested (name, value, attributes, children), or None."""
    if b[:4] not in (b'BXM\0', b'XML\0'):
        return None
    node_count, data_count, _ = struct.unpack_from('>HHI', b, 8)
    data_at = 16 + node_count * 8
    str_at = data_at + data_count * 4

    def s(off):
        return '' if off == 0xFFFF else b[str_at + off:b.find(b'\0', str_at + off)].decode('utf-8', 'replace')

    def node(i):
        kids, first, attr_count, row = struct.unpack_from('>HHHH', b, 16 + i * 8)
        name, value = struct.unpack_from('>HH', b, data_at + row * 4)
        attrs = {}
        for a in range(attr_count):
            k, v = struct.unpack_from('>HH', b, data_at + (row + 1 + a) * 4)
            attrs[s(k)] = s(v)
        return s(name), s(value), attrs, [node(first + k) for k in range(kids)]

    return node(0)


def to_xml(tree):
    """The tree as the XML runtime/dlc.c reads back: an element's text is its value, trimmed."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>']

    def emit(n, depth):
        name, value, attrs, kids = n
        pad = '  ' * depth
        a = ''.join(f' {k}={quoteattr(v)}' for k, v in attrs.items())
        if not kids:
            out.append(f'{pad}<{name}{a}>{escape(value)}</{name}>' if value else f'{pad}<{name}{a}/>')
            return
        out.append(f'{pad}<{name}{a}>{escape(value)}')
        for k in kids:
            emit(k, depth + 1)
        out.append(f'{pad}</{name}>')

    emit(tree, 0)
    return '\n'.join(out) + '\n'


def ptd_rows(d, at, count):
    out = []
    for i in range(count):
        r = at + 16 * i
        h, off, _, nbytes = struct.unpack_from('<4I', d, r)
        raw = bytes((b - PTD_KEY) & 0xFF for b in d[r + off:r + off + nbytes])
        out.append((h, raw.decode('utf-16-le').rstrip('\0')))
    return out


def ptd_texts(d):
    """[(id, text)] of a PTD text file: the id rows and the first column of the first section, both by hash."""
    magic, ver, key, n_ids, ids_at, n_sec, sec_at = struct.unpack_from('<4sIIIIII', d, 0)
    if magic != b'PTD\0' or ver != 2 or key != PTD_KEY:
        sys.exit('not a text file this tool reads')
    ids = dict(ptd_rows(d, ids_at, n_ids))
    if not n_sec:
        return []
    _, _, _, n_cols, cols_at = struct.unpack_from('<5I', d, sec_at)
    if not n_cols:
        return []
    r = sec_at + cols_at
    _, count, off = struct.unpack_from('<3I', d, r)
    return [(ids.get(h, f'#{h:08x}'), text) for h, text in ptd_rows(d, r + off, count)]


def quests(rom):
    for name, size, data in pkz_entries(rom.read('quest/quest.pkz')):
        yield name[5:-4], dat_files(unzstd(data, size))


def commands_of(tree):
    if tree[0] == 'CommandList':
        yield tree
        return
    for k in tree[3]:
        yield from commands_of(k)


def split_command(c):
    """((typeIF, [(parameter, value)]), (typeEXEC, [(parameter, value)])) of a CommandList."""
    tags = [(k[0], k[1]) for k in c[3]]
    at = [t[0] for t in tags].index('typeEXEC')
    return (tags[0][1], tags[1:at]), (tags[at][1], tags[at + 1:])


def child(n, name):
    return next((k for k in n[3] if k[0] == name), None)


def cmd_quests(rom, args):
    for qid, files in quests(rom):
        print(qid, ' '.join(f'{n}({len(b)})' for n, b in files))


def cmd_quest(rom, args):
    want = args.id.lower()
    files = next((f for qid, f in quests(rom) if qid == want), None)
    if files is None:
        sys.exit(f'quest {want} is not in quest/quest.pkz')
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, body in files:
            tree = bxm(body) if name.endswith('.bxm') else None
            if tree:
                (out / (name[:-4] + '.xml')).write_text(to_xml(tree), encoding='utf-8', newline='\n')
            else:
                (out / name).write_bytes(body)
        print(f'{len(files)} files of quest {want} in {out} (the game\'s content: keep it out of the repository)')
        return
    data = dict(files)
    t = bxm(data.get('QuestData.bxm', b''))
    if not t:
        sys.exit(f'quest {want} has no QuestData.bxm')
    sub = child(t, 'QuestSub')
    print(f'quest {want}: files', ', '.join(n for n, _ in files))
    print('QuestSub', sub[2], ' '.join(f'{k[0]}={k[1]}' for k in sub[3]))
    for lst, attr in (('QuestFlagNameList', 'QuestFlagName'), ('SaveFlagNameList', 'SaveFlagName')):
        named = [f'{i}:{k[2].get(attr, "")}' for i, k in enumerate(child(t, lst)[3])
                 if not re.fullmatch(r'(Save)?Flag\d\d', k[2].get(attr, ''))]
        print(f'{lst}: named {named}')
    for g in child(t, 'AreaList')[3]:
        print('AreaGroup', g[2])
        for w in g[3]:
            print('    Work', ' '.join(f'{k[0]}={k[1]}' for k in w[3]))
    for ti, task in enumerate(k for k in child(t, 'QuestData')[3] if k[0] == 'TaskList'):
        print(f'TASK {ti}', task[2], ' '.join(f'{k[0]}={k[1]}' for k in task[3] if k[0] != 'LineListTree'))
        for li, line in enumerate(child(task, 'LineListTree')[3]):
            print(f'  LINE {li}')
            for c in line[3]:
                (it, ip), (et, ep) = split_command(c)
                cond = 'always' if it == '0' else f'IF{it}(' + ', '.join(f'{k}={v}' for k, v in ip) + ')'
                print(f'      {cond:60s} -> EXEC{et}(' + ', '.join(f'{k}={v}' for k, v in ep) + ')')
    for name, body in files:
        if name == 'QuestData.bxm' or not body:
            continue
        print('----', name, f'({len(body)} bytes)')
        tree = bxm(body) if name.endswith('.bxm') else None
        if tree:
            print('\n'.join(to_xml(tree).splitlines()[1:args.lines + 1]))
        elif name.endswith('.csv'):
            print('\n'.join('    ' + l for l in body.decode('utf-8', 'replace').splitlines()[:args.lines]))


def cmd_commands(rom, args):
    count = {'IF': Counter(), 'EXEC': Counter()}
    params = {'IF': defaultdict(Counter), 'EXEC': defaultdict(Counter)}
    values = {'IF': defaultdict(lambda: defaultdict(Counter)), 'EXEC': defaultdict(lambda: defaultdict(Counter))}
    n = 0
    for _, files in quests(rom):
        t = bxm(dict(files).get('QuestData.bxm', b''))
        if not t:
            continue
        n += 1
        for c in commands_of(t):
            for kind, (typ, ps) in zip(('IF', 'EXEC'), split_command(c)):
                count[kind][typ] += 1
                for k, v in ps:
                    params[kind][typ][k] += 1
                    values[kind][typ][k][v] += 1
    print(f'{n} quests, {sum(count["IF"].values())} commands')
    for kind in ('IF', 'EXEC'):
        print(f'\n== type{kind}')
        for typ in sorted(count[kind], key=int):
            print(f'  {kind} {typ:>3s}  used {count[kind][typ]} times')
            for k, c in params[kind][typ].most_common():
                top = ', '.join(f'{v}({m})' for v, m in values[kind][typ][k].most_common(6))
                print(f'        {k:26s} {c:>6d}  {len(values[kind][typ][k]):>5d} values: {top}')


def cmd_texts(rom, args):
    entry = f'{args.file}_{args.lang}.bin'.lower()
    found = next((e for e in pkz_entries(rom.read('Text/Text.pkz')) if e[0].lower() == entry), None)
    if found is None:
        names = sorted({e[0].rsplit('_', 1)[0] for e in pkz_entries(rom.read('Text/Text.pkz'))})
        sys.exit(f'{entry} is not in Text/Text.pkz; the text files: {", ".join(names)}')
    pattern = (args.pattern or '').lower()
    for tid, text in ptd_texts(unzstd(found[2], found[1])):
        if pattern in tid.lower() or pattern in text.lower():
            print(f'{tid} = {text!r}')


def cmd_hash(rom, args):
    for name in args.names:
        print(f'{name}: {game_hash(name)} ({game_hash(name):#010x})')


def cmd_check(rom, args):
    """A DLC folder's XML read with the standard parser: a file the port's parser (runtime/dlc.c: elements,
    attributes, text, comments, the five entities) would refuse is refused here too."""
    import xml.etree.ElementTree as ET
    folder = Path(args.folder)
    manifest = folder / 'dlc.xml'
    if not manifest.is_file():
        sys.exit(f'{manifest} is missing')
    bad = 0
    for path in [manifest, *sorted(folder.glob('quest/*/*.xml'))]:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            print(f'{path}: {e}')
            bad += 1
            continue
        note = ''
        if path == manifest:
            if root.tag != 'dlc':
                print(f'{path}: the root element is <{root.tag}>, not <dlc>')
                bad += 1
            known = {'quest', 'training', 'window', 'text'}
            other = sorted({k.tag for k in root if k.tag not in known})
            note = '; '.join(f'{sum(k.tag == t for k in root)} {t}' for t in sorted(known))
            if other:
                print(f'{path}: elements the port does not know: {other}')
                bad += 1
            for q in root.iter('quest'):
                qdir = folder / 'quest' / (q.get('id') or '')
                if not q.get('base') and not qdir.is_dir():
                    print(f'{path}: quest {q.get("id")} has neither a base nor a folder {qdir}')
                    bad += 1
        elif path.name == 'QuestData.xml':
            cmds = list(root.iter('CommandList'))
            host = [c.findtext('typeIF') for c in cmds if int(c.findtext('typeIF') or 0) >= 0x10000]
            note = f'{len(list(root.iter("TaskList")))} tasks, {len(cmds)} commands, {len(host)} host conditions'
        print(f'{path}: ok' + (f' ({note})' if note else ''))
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--romfs', default=str(ROOT / 'build' / 'extracted' / 'v101' / 'romfs'),
                    help='folder with romfs.bin and index.json (an install folder\'s romfs/)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('quests')
    p = sub.add_parser('quest')
    p.add_argument('id')
    p.add_argument('--out')
    p.add_argument('--lines', type=int, default=60, help='lines shown of each other file of the quest')
    sub.add_parser('commands')
    p = sub.add_parser('texts')
    p.add_argument('file')
    p.add_argument('pattern', nargs='?')
    p.add_argument('--lang', default='USen')
    p = sub.add_parser('hash')
    p.add_argument('names', nargs='+')
    p = sub.add_parser('check')
    p.add_argument('folder')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    rom = None if args.cmd in ('hash', 'check') else Rom(args.romfs)
    try:
        {'quests': cmd_quests, 'quest': cmd_quest, 'commands': cmd_commands, 'texts': cmd_texts, 'hash': cmd_hash,
         'check': cmd_check}[args.cmd](rom, args)
        sys.stdout.flush()
    except (BrokenPipeError, OSError) as e:   # the reader of a pipe closed it (| head): Windows reports EINVAL
        if isinstance(e, BrokenPipeError) or getattr(e, 'errno', 0) == 22:
            sys.stderr.close()
            sys.exit(0)
        raise


if __name__ == '__main__':
    main()
