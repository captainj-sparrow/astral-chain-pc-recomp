# DLCs

A DLC is an addition to the game written as plain XML: a quest, the texts it shows, a row of the hub's training menu,
a tutorial window. It holds no file of the game. With the setting `dlc = 1` the port reads the DLC folders next to
`astralchain.exe` at start, builds the game's archives with their entries from the player's own RomFS, and serves
those in place of the game's.
`runtime/dlc.c` is the builder and documents the binary formats it writes; `runtime/quest.c` documents what the game
does with a quest script and the parts the port adds to it.

`dlc/pc-controls/` is the first one: the training "DLC: PC controls". `dlc/README.md` is the guide for writing one:
the quest script's conditions and actions, the texts, the file formats, the tool (`tools/dlc_tool.py`) and the ids
in use. It stands alone, as the installer copies it next to the DLCs.

## Off by default

The setting `dlc` (`astralchain.ini` `[mods]`, `AC_DLC`; the Mods tab's "Load DLC") is 0 by default and read at
start. With 0 nothing of this page runs: `dlc_scan` reads no folder, no archive is built or served, and the patch
sets `quest` and `questmenu` (`recomp/patches-v101.txt`), which follow the same setting, are off, so the game runs
its own code at each of their eight sites. The Mods tab shows the DLCs the start loaded and opens the folder.

## Layout

```
dlc/<name>/
  dlc.xml                    the manifest
  quest/<id>/QuestData.xml   a quest's script; any other file of the quest next to it
```

A release ships the repository's `dlc/` folder (`tools/make_release.py` `build_sources`, both systems), and the
setup's last step copies it next to `astralchain.exe` at every install and update (`installer/steps.py`
`install_dlcs`): each DLC folder of the release replaces the folder of its name whole, and a folder of another name,
the player's own DLC, is left alone. A copy that fails only warns.

## dlc.xml

```xml
<dlc>
  <quest id="c0f0" base="c00b"/>
  <training value="2" text="MENU_TUTO_NAME_DLC_00" quest="c0f0"/>
  <window number="af00"/>
  <text file="Core_text" id="CORE_TUTO_W_NAME_af00_01" live="1">
    <t>{ZL} + {ZR}</t>
    <t lang="EUfr">...</t>
  </text>
</dlc>
```

- **quest**: adds the quest `id` (four hex digits) to `quest/quest.pkz` as `quest<id>.dat`. Its files are those of
  `quest/<id>/`: `<name>.xml` becomes `<name>.bxm`, the game's binary XML (an element is a node: its name, its
  trimmed text as the value, its attributes, its children); any other file is taken as it is. With `base`, the files
  the folder does not give are the base quest's, read from the player's archive, with `_<base>.` in their names
  changed to `_<id>.`. A training's id must be `c0xx` (not `c050`): the game takes those for trainings.
- **training**: a row of the training menu that starts `quest`, named by the text `text` of `menu_text`. Its `value`
  must be free in the game's menu: 2 and 7 are. The menu takes one added row for now (the patch sites count 30 rows).
- **window**: the number (from `a000`) of a tutorial window, which action 18 of a quest script shows. Its texts are
  `CORE_TUTO_W_TITLE_<number>`, and per page `CORE_TUTO_W_NAME_<number>_<page>` and `CORE_TUTO_W_HELP_<number>_<page>`
  (pages from `01`), in `Core_text`.
- **text**: adds the text `id` to `Text/Text.pkz`, to `<file>_<language>.bin` of every language of the game. Each
  language takes the `<t>` of its name (`USen`, `EUfr`, `EUde`, `EUes`, `EUit`, `EUru`, `JPja`, `KRko`, `CNzh`,
  `TWzh`), else the `<t>` without `lang`. `\n` is a line break. With `live="1"`, each `{ACTION}` field of the text
  (`{ZL}`, `{A}`, `{PLUS}`...: the action names of `astralchain.ini`'s bind lines) shows the inputs bound to that
  button, as the Controls tab of the port's menu labels them, whenever the game reads the text.

An entry the port cannot use is logged (`dlc: ...` in `logs/astralchain.log`) and left out; it never stops the start.

## Quest scripts

`QuestData.xml` is the quest's `QuestRoot`: `QuestData` holds the tasks (`TaskList`), a task its lines (`LineList`), a
line its commands (`CommandList`): a condition (`typeIF` and its parameters) and an action (`typeEXEC` and its
parameters). `dlc/pc-controls/quest/c0f0/QuestData.xml` is a commented example. The port adds conditions on the
player's inputs, which the game has none of (`runtime/quest.c`): `typeIF` 0x10000 + buttons is true while all those
buttons are held, 0x20000 + buttons while one of them is not, whatever input the buttons are bound to.

## The build and its cache

The archives are built from what the game would read: the RomFS image's archive, or a player mod's that replaces it,
so a DLC and a mod of the same archive add up. The result is kept in `<profile>/dlc/` with a key file; a later start
serves it as it is while the key matches: the SHA-256 of every DLC source file and of the builder's version, with the
identity of the archive underneath. On a Core Ultra 9 285H the first start built `quest.pkz` (415 entries, 5.6 MB) in
23 ms and `Text.pkz` (20 of 90 text files rewritten, 11.5 MB) in 111 ms; a later start reads neither archive.

## Checks

- `astralchain.exe --probe-dlc` (no game files; `make_release.py --prove` runs it on both systems): a DLC folder and
  two archives of the probe's own through `dlc_build`, read back with readers written apart from the writers (40
  checks: a quest with and without a base, texts per language, the game's entries untouched, DLCs that must be left
  out, the key), then 2 000 archives and 300 manifests cut short or with bytes changed, which must all be answered:
  a DLC is written by anyone, and a bad one must never stop the game. 3 s.
- `tests/dlc_check.py`: part 1 (no build) checks that DLCs are off by default wherever a default lives, that every
  site calling a quest helper is in the two sets, that the release ships `dlc/` and the installer copies it, and
  each DLC of `dlc/` (only `.xml` and `.md`, ids in the free ranges and used once, a text for every language).
  Part 2 runs the probe. Part 3 (game files) has the build make both archives from the player's with `dlc/`
  (`AC_DLC_PROBE_DIR`) and compares them with the test's own DAT, BXM, text and archive writers: on 1.0.1 the 414
  quests and the 70 untouched text files keep their compressed bytes, and `questc0f0.dat` and the 20 text files
  are the same bytes.
- `tests/installer_test.py` `CleanupTest`: the setup's copy replaces a shipped DLC whole and keeps the player's.
