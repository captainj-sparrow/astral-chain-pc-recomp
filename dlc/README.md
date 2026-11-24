# Writing a DLC for Astral Chain Recomp

This folder holds the DLCs of the port: additions to the game written as plain XML by the community. This guide is
meant to be all a coding agent, or a person, needs to write one, or to extend what a DLC can do. It stands alone:
where it names a source file, that file's header comment holds the same facts next to the code.

`pc-controls/` is the first DLC, a training added to the hub's training menu, and the example to copy.

The paths of this guide (`tools/`, `tests/`, `runtime/`, `recomp/`, `docs/`) are those of the port's repository,
https://github.com/captainj-sparrow/astral-chain-pc-recomp. An installed game holds only this folder, next to
`astralchain.exe`: writing and trying a DLC needs no more than that and a text editor, and the tool and the checks
need a clone of the repository.

How each statement is known is marked. **Seen**: read in the game's data or code, or checked in game. **Inferred**:
deduced from how the game's 414 quests use it, not yet checked in game. Check an inferred fact with a small quest
before building on it, and correct this guide when you learn more.

Contents: [The rules](#the-rules) · [What a DLC can do](#what-a-dlc-can-do-today) · [Start here](#start-here) ·
[How it works](#how-it-works) · [dlc.xml](#dlcxml) · [Quests](#quests) · [The script](#the-script-questdataxml) ·
[Conditions](#conditions-typeif) · [Actions](#actions-typeexec) · [Tutorial windows](#tutorial-windows-and-help-panels) ·
[Objects](#what-a-quest-places-enemysetxml) · [Dialogues](#dialogues-talkscript) · [Texts](#texts) ·
[The training menu](#the-training-menu) · [The file formats](#the-file-formats) · [The tool](#the-tool) ·
[Checking](#checking-a-dlc) · [The log](#the-log) · [Known gaps](#known-gaps) ·
[Extending the port](#extending-the-port) · [Contributing](#contributing) · [Ids in use](#ids-in-use)

## The rules

1. **No game content.** A DLC holds only what its author wrote: XML and Markdown. It never holds a file of the game, a
   piece of one, or a text, script, picture or sound taken from one. What a DLC needs of the game it names
   (`base="c00b"`), and the port reads it from each player's own files at start. Facts *about* the game are fine:
   ids, hashes, element names, addresses, what a command does. `tests/dlc_check.py` refuses any file of `dlc/` that is
   not `.xml` or `.md`.
2. **What the tool prints is the game's.** `tools/dlc_tool.py quest` and `texts` show the player's own game files.
   Keep their output under `build/` or `notes/`, which git ignores, and out of commits, issues, pull requests and
   chats.
3. **Off unless the player asks.** DLCs load only with the setting `dlc = 1` (Settings, Mods tab, "Load DLC"; off by
   default). With it off the port reads no DLC folder, builds no archive and runs none of the code this guide
   describes: the game runs as it is. A DLC must never be needed for the game to start, and one the port cannot use
   is left out with a line in the log.
4. **English** in XML comments, file names and ids, as everywhere in the repository. Texts shown to the player can be
   given per language. Unix line endings (LF) in every file.

## What a DLC can do today

| It can | With |
|---|---|
| Add a quest to the game | `<quest>` and a `QuestData.xml` script |
| Reuse the files of one of the game's quests (its objects, dialogue tables, notices) | `base="<id>"` |
| Replace or add any file of that quest (objects, dialogues) | a file of the same name in `quest/<id>/` |
| Add a row to the hub's training menu that starts the quest | `<training>` |
| Show a tutorial window with a title and pages of its own | `<window>`, action 18 |
| Add or replace a text of the game in any language | `<text>` |
| Show the inputs the player bound, as the Controls tab names them | `live="1"` and `{ZL}`-style fields |
| Wait for buttons, keys or mouse buttons in a script | the host conditions |

Not yet: more than one added training row; new pictures, models, sounds or voices; a quest started from anywhere
but the training menu; archives other than `quest/quest.pkz` and `Text/Text.pkz`. Each needs work in the port, which
is welcome too: see [Extending the port](#extending-the-port).

## Start here

```
dlc/<name>/
  dlc.xml                    the manifest
  quest/<id>/QuestData.xml   the quest's script; any other file of the quest next to it
```

1. Copy `pc-controls/` to a folder of your own name (lower case, digits, `-`).
2. Pick ids no other DLC uses ([Ids in use](#ids-in-use)) and put them in `dlc.xml`.
3. Write the script. Read quests of the game for models:
   `python tools/dlc_tool.py quest c003` prints one as a listing;
   `python tools/dlc_tool.py quest c003 --out build/quests/c003` writes its files as the XML a DLC folder takes;
   `python tools/dlc_tool.py commands` lists every condition and action with its parameters.
   Add `--romfs "<install folder>/romfs"` when you work from an installed game and not from a checkout with
   extracted files. The tool needs Python 3.14, or an older one with the `zstandard` package.
4. `python tools/dlc_tool.py check dlc/<name>` parses the folder as the port will.
5. Put the folder next to the game: `<install folder>/dlc/<name>/` (the Mods tab's "Open DLC folder" shows it; in a
   checkout, `build/v101/dlc/`). Switch "Load DLC" on and start the game again. `AC_DLC=1` in the environment does
   the same for one run.
6. Read `logs/astralchain.log` next to the executable: every `dlc:` and `quest:` line ([The log](#the-log)).
7. Before a pull request: `python tests/dlc_check.py` and `python tools/check_content.py --staged`
   ([Contributing](#contributing)).

## How it works

- At start, with `dlc = 1`, the port reads every `dlc/<name>/dlc.xml` next to `astralchain.exe`, in the order of the
  folder names. A file next to the DLC folders (this guide) is ignored.
- It builds two archives the game reads: `quest/quest.pkz` with the DLCs' quests added, `Text/Text.pkz` with their
  texts added to the text files they name. It starts from what the game would read: the archive of the player's
  RomFS image, or a player mod's that replaces it, so a mod and a DLC of the same archive add up. Every entry the
  DLCs do not touch keeps its compressed bytes.
- The result is kept in `<profile>/dlc/` (`quest.pkz`, `Text.pkz`, each with a `.key` file). A later start serves it
  as it is while the key matches: the SHA-256 of every DLC source file and of the builder's version, with the
  identity of the archive underneath. Change a source and the next start builds again; nothing is to be deleted by
  hand. Measured on a Core Ultra 9 285H: `quest.pkz` (415 entries, 5.6 MB) in 23 ms, `Text.pkz` (20 of 90 text files
  rewritten, 11.5 MB) in 111 ms; a later start reads neither archive.
- The game then loads the quest and the texts as its own. Four things the game cannot do from data are done by the
  port's code, each only with `dlc = 1`: a training menu row for a quest the menu does not know, a tutorial window
  whose number the game does not know, texts filled in with the bound inputs, and script conditions on the player's
  inputs.
- `runtime/dlc.c` is the builder (manifest, formats, cache key); `runtime/romfs.c` serves the built archives;
  `runtime/quest.c` holds the four additions; `recomp/patches-v101.txt` holds the places in the game's code that call
  them (the sets `quest` and `questmenu`).

## dlc.xml

```xml
<dlc>
  <quest id="c0f0" base="c00b"/>
  <training value="2" text="MENU_TUTO_NAME_DLC_00" quest="c0f0"/>
  <window number="af00"/>
  <text file="Core_text" id="CORE_TUTO_W_TITLE_af00">
    <t>PC controls</t>
    <t lang="EUfr">Commandes PC</t>
  </text>
  <text file="Core_text" id="CORE_TUTO_W_NAME_af00_01" live="1">
    <t>{ZL} + {ZR}</t>
  </text>
</dlc>
```

The root element is `<dlc>`. Its children, in any order and number:

- **`<quest id base>`**: adds the quest `id` (four hex digits) to `quest/quest.pkz` as `quest<id>.dat`, or replaces
  the game's quest of that id. Its files are those of `quest/<id>/` next to `dlc.xml`: `<name>.xml` becomes
  `<name>.bxm`, the game's binary XML; any other file is taken as it is. With `base` (four hex digits), the files the
  folder does not give are the base quest's, read from the player's archive, with `_<base>.` in their names changed
  to `_<id>.` (`TalkScript_c00b.bxm` becomes `TalkScript_c0f0.bxm`). Without `base` the folder must hold every file
  the quest needs. A quest holds 64 files at most. *Seen.*
- **`<training value text quest>`**: a row of the hub's training menu, named by the text `text` (an id of
  `menu_text`), that starts `quest`. `value` must be 2 or 7, the two values the game's menu leaves free. One added
  row over all DLCs for now; a second one is left out and logged. *Seen.*
- **`<window number>`** (four hex digits, `a000` to `ffff`): a tutorial window that action 18 can show
  ([Tutorial windows](#tutorial-windows-and-help-panels)). *Seen.*
- **`<text file id live>`** with `<t>` children: adds the text `id` to `<file>_<language>.bin` of `Text/Text.pkz`, for
  every language of the game. Each language takes the `<t lang="...">` of its name, else the `<t>` without `lang`:
  always give one without `lang`. Languages: `USen`, `EUfr`, `EUde`, `EUes`, `EUit`, `EUru`, `JPja`, `KRko`, `CNzh`,
  `TWzh`. `\n` (the two characters) is a line break. A text whose id the file already has replaces it, in that file
  only. `live="1"`: see [Texts](#texts). *Seen.*

The XML the port reads: elements, attributes in single or double quotes, text, comments, an XML declaration, the
entities `&lt; &gt; &amp; &quot; &apos;` and character references, UTF-8. No CDATA, DTD or namespaces. An element's
value is the text before its first child, with the white space around it removed.

Limits of one start, over all DLCs: 32 DLCs, 32 quests, 64 files per quest, 512 texts, 16 languages per text, 16
windows, 4 live texts of 255 characters each once filled in, 1 training row. A BXM file holds at most 65 534 nodes.
An entry past a limit is left out and logged.

## Quests

A quest is `quest<id>.dat` in `quest/quest.pkz`, the id four hex digits (`questc003.dat` is the quest 0xc003, 49155).
The game opens it by that name. Its files (*seen*):

| File | What it holds |
|---|---|
| `QuestData.bxm` | the script: tasks, lines, commands |
| `EnemySet.bxm` | what the script can place: enemies, objects, triggers, in numbered groups |
| `TalkScript_<id>.bxm` | dialogues: an object id and the lines of text ids it says |
| `TalkScript_speech_<id>.bxm` | speech balloons (empty in the trainings) |
| `ExData.csv` | the kind of quest and its parameters |
| `BezierData.bxm` | paths (`BEZEIR` / `FILE` / `POINTLIST`, `CLUSTERLIST`); often empty |
| `SignboardData.bxm` | signs in the world (`SignboardData`: `Text`, `Pos`, `FontSize`, `Near`, `Far`, `Enable`...); often empty |
| `TalkData_<id>.csv`, `TalkCondition_<id>.csv`, `TalkFlag_<id>.csv`, `SubtitleInfo_<id>.csv`, `SpeechBalloon_<id>.csv`, `ReliefSupplies.csv`, `ResultData.csv` | dialogue and result tables; often empty |

The first cell of `ExData.csv` is the kind: `Battle` (most quests and most trainings), `Training`, `ShootingTraining`,
`Escort`, `Repair`, `Listening`, `Shadow`, `Stealth`, `ScoreAttack`, `Sokoban`, `BikeShooting`, `LegionDash`,
`Transport`... What the other cells mean is not studied. *Seen* for the names.

The game's ids: `0xxx` (11 quests), `1xxx` (28), `2xxx` (200, the hub and side missions), `3xxx` (88), `6xxx` (34),
`cxxx` (53). The game takes a quest for a training by its id, `c0xx`; `c050` is the free-combat quest of the training
room. Its own `c0xx` quests are `c000` to `c00c`, `c030` to `c034`, `c050`, `c055` and `c060` to `c06b` (the
sparring fights, one per enemy). *Seen.*

The simplest quest reuses a base: `pc-controls` is `base="c00b"` (the IRIS training) with a `QuestData.xml` of its
own, and takes from the base the "training started" and "training finished" notices its script places.

## The script (QuestData.xml)

```
QuestRoot
  Version               16
  QuestData
    TaskList TaskName="..." TemplateName=""    a task; every task runs side by side
      TaskEnable        1
      WorkInAdvanced    0
      TaskColor         0
      LineListTree
        LineList                               a line: one step of the task
          CommandList                          a command: a condition and an action
            typeIF      the condition's type, then its parameters
            typeEXEC    the action's type, then its parameters
  QuestSub SetMainScenarioValueStr="..."
    MinLevel, MaxLevel, bSetPlStartPos, PlStartPos ("x y z w"), PlStartRotY, bSetMainScenario
  QuestFlagNameList     64 x  Name QuestFlagName="..."     the quest's flags; the names are only labels
  SaveFlagNameList      64 x  Name SaveFlagName="..."      the saved flags
  AreaList
    GroupList GroupIndex GroupName GroupDebugDisp          a group of areas
      WorkList: AreaIndex, AreaType, Center, Height, DebugDisp, then Pos1..Pos4 (a box) or Radius (a circle)
```

Keep every element of this shape, in this order, and both name lists at 64 entries; `pc-controls` is the smallest
complete file. A parameter is an element named as the tables below name it, its value the element's text: the
condition's between `typeIF` and `typeEXEC`, the action's after `typeEXEC`. Numbers are decimal (`49155`, not
`0xc003`); a real number is written `1.000000`. Every one of the game's 64 202 commands has this shape. *Seen.*

### How a line runs

- The commands of a line run in order, top to bottom. An action runs when its condition is true; `typeIF` 0 is
  "always". *Seen.*
- **Action 1 waits.** It holds its line for as long as its condition is true (the line starts again on the next
  frame), and the line goes on when the condition turns false. "Wait until X" is therefore the condition "not X" with
  action 1. Most conditions have a `bCheck` / `IFbCheck` parameter that says which way they test. *Seen* in the
  game's code and in game.
- **Action 2 ends the line.** `EXECIsNextLine` 1 goes to the next line; `EXECIsNextLine` 0 with `EXECLine` n jumps to
  line n; `EXECLine` -1 on the last line ends the task. Action 3 is the same under its condition. *Seen* for the
  next line and the end of the task; *inferred* for the jump.
- A line without action 2 at its end, and what a false condition does to an action other than 1 (the usage says it
  skips that action): *inferred*.
- Tasks talk to each other through the quest's flags: one sets a flag with action 12, another waits for it with
  condition 7.
- A type outside 1 to 38 answers true in the game. The port answers the types from 0x10000 itself.

The script of `pc-controls`, as a listing:

```
start   line 0   always        -> 53(Type 2, bSet 1)          a lock while the notices show
                 always        -> 5(group 0), 5(group 4)      the base's "training started" notice
                 always        -> 35(Index 10, bOn 0)
                 always        -> 2(next line)                the last line: the task ends
intro   line 0   always        -> 35(Index 0, bOn 1)          controls off
                 always        -> 4(1 second)
                 always        -> 2(next line)
        line 1   always        -> 53(Type 2, bSet 0)
                 always        -> 18(enable 1, number 44800)  the DLC's tutorial window, 0xaf00
                 always        -> 2(next line)
        line 2   33(bCheck 1)  -> 1                           waits while the window is open
                 always        -> 35(Index 0, bOn 0)          controls on
                 always        -> 12(flag 0, 1)               tells the task "press"
                 always        -> 2(next line, EXECLine -1)
press   line 0   7(flag 0, 0)  -> 1                           waits for flag 0
                 131840        -> 1                           waits while ZL and ZR are not both held
                 always        -> 12(flag 1, 1)               tells the task "end"
                 always        -> 2(next line, EXECLine -1)
end     line 0   7(flag 1, 0)  -> 1                           waits for flag 1
                 always        -> 18(enable 0, number -1)     closes the help
                 always        -> 35(Index 0, bOn 1), 4(0.5 second)
                 always        -> 5(group 2)                  the base's "training finished" notice
                 always        -> 46                          ends the quest
                 always        -> 2(next line, EXECLine -1)
```

## Conditions (typeIF)

### Host conditions (the port's own)

The game's conditions test its own state only; none reads an input. The port adds four, with the argument in the
type itself (they take no parameter element):

| typeIF | True while |
|---|---|
| `0x10000 + buttons` | all those controller buttons are held |
| `0x20000 + buttons` | one of them is not held |
| `0x30000 + key1 * 256 + key2` | both keys are held (`key2` 0: `key1` alone) |
| `0x40000 + key1 * 256 + key2` | one of them is not held |

`buttons` adds the bits of the game's buttons: A 1, B 2, X 4, Y 8, left stick click 16, right stick click 32, L 64,
R 128, ZL 256, ZR 512, Plus 1024, Minus 2048, D-pad left 4096, up 8192, right 16384, down 32768. They are true
whatever holds them: a gamepad, or the keys and mouse buttons the player bound to them. **Prefer buttons to keys**: a
script that names a key breaks for a player who bound another one, and for one who plays with a gamepad.

Key codes: 1 to 5 the mouse's left, right, middle and two side buttons, 9 Tab, 13 Enter, 16 left Shift, 17 left
Ctrl, 32 Space, 48 to 57 the digits, 65 to 90 the letters, whatever they are bound to.

Nothing counts as held while the port's menu or a text entry is open, or while another window is in front.
"Wait for ZL and ZR together" is `<typeIF>131840</typeIF>` (`0x20000 + 256 + 512`) with action 1. *Seen.*

### The game's conditions

All 29 types the game's quests use, with their parameter elements in order. The count is how many of the 64 202
commands use the type.

| Type | Uses | Parameters | True when | Known |
|---|---|---|---|---|
| 0 | 42 988 | | always | seen |
| 1 | 3 357 | `IFMultiple`, `IFGroupNo`, `IFIndexNo`, `IFbCheck` | a member of a group (state not studied) | |
| 2 | 4 194 | `IfType`, `IFGroupNo`, `IFCommand`, `IFValue` | a group's state; `IFCommand` 5 with `IFValue` 0: the group is defeated | inferred |
| 3 | 307 | `IFHash`, `IFbCheck` | a named global flag (the hash of its name) | inferred |
| 4 | 4 189 | `IFQuestId`, `IFFlagNo`, `IFbCheck` | a flag of the line (`IFFlagNo` 0 to 4), the one action 9 sets | inferred |
| 5 | 93 | `IFCntNo`, `IFValue`, `IFCommand` | a counter compares so | inferred |
| 7 | 2 280 | `IFQuestId` (0: this quest), `IFFlagNo`, `IFbCheck` | the quest flag equals `IFbCheck` | seen |
| 8 | 288 | `IFCntNo`, `IFValue`, `IFCommand` | a counter compares so | inferred |
| 10 | 492 | `EventType`, `EventNo`, `Condition`, `EventState` | a cutscene's state | inferred |
| 11 | 3 765 | `IFHash`, `IFValueHash`, `IFValue2Hash`, `IFCondition` | a named counter compares so | inferred |
| 14 | 38 | `HasItemId`, `HasItemValue`, `HasItemCondition` | the player has an item | inferred |
| 15 | 53 | `IfIsGroup`, `IfEmGroupNo`, `IfEmSetNo`, `IfAreaGroup`, `IfAreaIndex`, `IfCheck` | an actor is in an area of `AreaList` | inferred |
| 16 | 260 | `IfIsGroup`, `IfEmGroupNo`, `IfEmSetNo`, `IfValue`, `IfCondition` | an enemy's health ratio | inferred |
| 17 | 87 | `IfQuestId`, `IfCondition`, `IfbCheck` | another quest's state | inferred |
| 20 | 5 | `IfSubQuestCompQuestId`, `IfSubQuestCompScore`, `IfSubQuestCondition` | a side quest's score | inferred |
| 22 | 581 | `bCheck` | a dialogue is running | inferred |
| 26 | 36 | `IsCombatBarrier` | a combat barrier is up | inferred |
| 27 | 259 | `Type`, `bCheck` | not studied | |
| 28 | 645 | `IFQuestId`, `IFFlagNo`, `IFbCheck` | a saved flag equals `IFbCheck` | inferred |
| 29 | 6 | `Condition`, `SecTime` | a time compares so | inferred |
| 30, 32, 35, 36 | 105, 13, 42, 23 | `bCheck` | not studied | |
| 31 | 9 | `Id`, `bCheck` | not studied | |
| 33 | 28 | `bCheck` | a tutorial window is open (`bCheck` 1) | seen |
| 34 | 18 | `Type`, `bCheck` | not studied | |
| 37 | 12 | `IfIsGroup`, `IfEmGroupNo`, `IfEmSetNo`, `IfValue`, `IfCondition` | as 16, another value | inferred |
| 38 | 29 | `bCheck`, `GroupNo` | not studied | |

Condition 7 with `IFbCheck` 0 and action 1 waits until the flag is set: the condition is true, and the line held,
while the flag is 0. *Seen.*

## Actions (typeEXEC)

All 50 types the game's quests use.

| Type | Uses | Parameters | What it does | Known |
|---|---|---|---|---|
| 0 | 320 | | nothing | inferred |
| 1 | 9 844 | | waits while the command's condition is true | seen |
| 2 | 14 405 | `EXECIsNextLine`, `EXECLine` | next line, jump, end of the task | seen |
| 3 | 3 370 | `EXECIsNextLine`, `EXECLine` | the same under its condition | inferred |
| 4 | 2 654 | `EXECTimer` | waits that many seconds | inferred |
| 5 | 6 471 | `EXECGroupNo`, `EXECSetMax` | places group n of `EnemySet` | inferred |
| 6 | 2 770 | `EXECGroupNo` | takes group n away | inferred |
| 7 | 571 | `EXECEventType`, `EXECEventNo`, `EXECEventPhaseNo`, and 23 more `EXECEvent...` | plays a cutscene | inferred |
| 8 | 317 | `ExecHash`, `ExecbCheck` | sets a named global flag | inferred |
| 9 | 8 875 | `ExecFlagNo`, `ExecbCheck` | sets a flag of the line | inferred |
| 10, 13 | 110, 120 | `ExecCntNo`, `ExecValue1`, `ExecValue2`, `ExecCommand` | changes a counter | inferred |
| 12 | 2 054 | `ExecFlagNo`, `ExecbCheck` | sets a quest flag | seen |
| 15 | 290 | `ExecPosX`, `ExecPosY`, `ExecPosZ`, `ExecRotY`, `ExecUseFade`, `ExecFadeColor` | moves the player | inferred |
| 16, 50 | 28, 16 | `ExecPhaseNo`, `ExecSubPhaseIndex`, `ExecQuestId`, `ExecLoadingType`, positions | goes to another phase of the story | inferred |
| 18 | 168 | `ExecDispTipsEnable`, `ExecDispTipsNo` | shows or closes a help panel or a tutorial window | seen |
| 19 | 1 | `ExecMClear...` (24 parameters) | clears a mission | inferred |
| 20 | 1 651 | `ExecCounterType`, `ExecCounterHash`, `ExecCounterValueHash` | sets a named counter | inferred |
| 21 | 1 760 | `EXECGroupNo` | deletes group n | inferred |
| 22 | 519 | `EXECGroupNo` | prepares group n (waves) | inferred |
| 23 | 360 | `ExecCallSubTitleNo`, `ExecCallSubTitleQuestId` | shows a result message | inferred |
| 28 | 21 | `EventType`, `EventNo`, `ExecSubQuestJumpPhaseNo`, `ExecSubQuestJumpQuestId` | jumps to a side quest | inferred |
| 30 | 612 | `ExecType`, `ExecTaskNo` | starts, stops or restarts another task | inferred |
| 31 | 606 | `QuestId`, `ExistSpeaker`, `GroupNo`, `SetNo`, `TalkId` | starts a dialogue: `TalkId` is the `ObjId` of the TalkScript of quest `QuestId` | inferred |
| 35 | 736 | `Index`, `bOn` | a "sequence" switch; `Index` 0 with `bOn` 1 takes the controls and the HUD away, `bOn` 0 gives them back | inferred |
| 36 | 92 | `GroupNo`, `SetNo`, `PosX`, `PosY`, `PosZ`, `RotY` | moves a placed object | inferred |
| 37 | 3 681 | | not studied | |
| 39 | 63 | `Type`, `QuestId`, `InNoticeType`, `InNoticeNo`, `ObjGroupNo`, and 10 more | a notice with a return point | inferred |
| 45 | 238 | | starts the quest's measure (time, score) | inferred |
| 46 | 343 | | ends the quest; every training ends with it | seen |
| 47 | 199 | `Type`, `Color`, `bKeyStop` | fades to a colour | inferred |
| 48 | 210 | | fades back | inferred |
| 49 | 373 | `ExecFlagNo`, `ExecbCheck` | sets a saved flag | inferred |
| 51 | 33 | `Type`, `SecTime`, `NoticeType`, `NoticeNo` | a timed notice | inferred |
| 52 | 21 | `Mode` | not studied (follows a dialogue in the trainings) | |
| 53 | 103 | `Type`, `bSet` | a lock on the player; `Type` 2 around the training notices | inferred |
| 54 | 9 | `bLock`, `Type` | a lock on the Legion | inferred |
| 55 | 11 | | heals | inferred |
| 61 | 86 | `bRead`, `GroupNo` | loads a group ahead | inferred |
| 26, 27, 38, 56, 59, 63 | 22, 17, 2, 4, 23, 4 | | not studied | |
| 57, 60 | 3, 6 | `Type` | not studied | |
| 58 | 8 | `Type`, `bFlag` | not studied | |
| 62 | 2 | `Type`, `Begin`, `End` | not studied | |

`python tools/dlc_tool.py commands` prints this list from your own game files with, for each parameter, its most
frequent values: the fastest way to learn what a parameter takes.

## Tutorial windows and help panels

Action 18 shows two kinds of help, by `ExecDispTipsNo` (decimal in the XML: `0xaf00` is `44800`):

- **Below `0xa000`: a panel of button hints.** Each number is a fixed list of lines in the game's code, and each
  line's text and its "done" test are code too. A DLC can show one of the game's panels, not add one. *Seen.*
- **From `0xa000`: a tutorial window.** The game shows it only when its number is one of its own 215 or of a DLC's
  `<window>`; any other number closes the window at once. Its texts, all in `Core_text`:

  | Text id | Shown as |
  |---|---|
  | `CORE_TUTO_W_TITLE_<number>` | the window's title |
  | `CORE_TUTO_W_NAME_<number>_<page>` | the page's headline |
  | `CORE_TUTO_W_HELP_<number>_<page>` | the page's help text |

  `<number>` is four hex digits in lower case, `<page>` two digits from `01`. The game counts the pages by the
  `HELP` texts there are. The page's picture is `tutorial_thumb_<number + page>`, which a DLC cannot add yet: a DLC's
  window shows the game's placeholder logo. *Seen.*

`ExecDispTipsEnable` 1 opens, 0 closes. To wait for the player to close a window: condition 33 with `bCheck` 1 and
action 1. A window of a DLC is in no chapter of the game's tutorial archive (the menu that lists the tutorials read).
The game's own numbers are `a000` to `a7ff`, `b240` and `c000` to `e500`; take a DLC's from `af00` to `afff`.

## What a quest places (EnemySet.xml)

```
EmSetRoot
  EmSetList
    GroupList number name GroupNameHash CanSet easy normal hard very_hard     a group; action 5 places it by number
      SetInfo                                                                  one object of the group
        Id, SetNo, Trans ("x y z w"), Rotation, BaseRot, SetType, Type, SetRtn, SetFlag, NoticeNo, ItemId,
        SetTimer, SetCounter, SetRadius, LvMin, LvMax, PathNo, EscapeNo, ParentId, PartsNo, HashNo,
        ExSetTypeA..D, ExSetAttr, ExSetRtn, ExSetFlag, SetWait, GridDisp, EventSuspend, ...
  GroupPosList
    GroupList > GroupPos (SetNo, Trans, Rotation, ...)                         positions shared by a group
```

`GroupNameHash` is the hash of `name`. *Seen* for the shape. A training has few enemies; its groups mostly hold
trigger objects. The ids the trainings use (*seen*; their effect *inferred*):

| `Id` | Used as |
|---|---|
| `0x20000` | the hologram enemy of the training room |
| `0xca02a` | a notice object: "training started", "training finished", "failed", controls off, Legion sealed, scenario flags reset... which one by its `SetRtn` / `SetFlag` |
| `0xca02d` | a target to walk to (a circle on the floor) |
| `0xca113` | a camera object |
| `0xfb040` | an item box |
| `0xfa540`, `0xfa590` | a floating switch, a wall to cut |

Every training of the game follows one plan: a "started" group; the explanation (action 18, then a dialogue with
action 31); the exercise's objects; a task that watches for success (a flag, an area, a defeated group); a
"finished" group and action 46. In `c00b`, groups 0 and 4 are the start notice and group 2 the finish notice.
`c000` (movement) and `c003` (evasion) are the fullest models. To change what a quest places, write its
`EnemySet.xml` from `dlc_tool.py quest <base> --out`, edited: that file is then yours only as far as you wrote it, so
ship a small one of your own making rather than a copy of the game's.

## Dialogues (TalkScript)

```
Root
  Info: ObjId, SetType, QuestId, TriggerType
    StateInfo: No, Priority, Trigger > Conditions
      CommandInfo: Command, Param1, Param2, Param3, Message     a line: Message is a text id
```

Action 31 starts the `Info` whose `ObjId` is its `TalkId`. A `Message` is a text id of `TalkSubtitleMessage` (the
game's look like `NP_c0003_...`). The line's voice is a sound of the game: a DLC brings none, so a new line is silent.
Dialogues from a DLC have not been tried in game: *inferred* throughout.

## Texts

The game's texts are `Text/Text.pkz`: 90 files `<name>_<language>.bin`, nine names in ten languages.

| Text | `file` | Id |
|---|---|---|
| A row of the training menu | `menu_text` | any id; the game's are `MENU_TUTO_NAME_<nn>` |
| A tutorial window | `Core_text` | `CORE_TUTO_W_TITLE_<number>`, `CORE_TUTO_W_NAME_<number>_<page>`, `CORE_TUTO_W_HELP_<number>_<page>` |
| A dialogue line | `TalkSubtitleMessage` | the `Message` id of the TalkScript |
| A line of a help panel | `Core_text` | the game's `CORE_TUTORIAL_BTN_*` (38 texts) |

`python tools/dlc_tool.py texts <file> [pattern] [--lang EUfr]` shows the game's own; with a file name it does not
know it lists the nine names. The game finds a text by the hash of its id, so ids do not depend on case; give each
DLC's ids a prefix or a number of its own. *Seen.*

**Live texts.** With `live="1"`, each `{ACTION}` field of the text is replaced, every time the game reads the text,
by the inputs the player bound to that button, with the labels of the Controls tab: `{ZL} + {ZR}` reads
`Left mouse / Z + Right mouse / X`, and follows a change of the bindings the next time the window opens. The names:
`A`, `B`, `X`, `Y`, `LCLICK`, `RCLICK`, `L`, `R`, `ZL`, `ZR`, `PLUS`, `MINUS`, `LEFT`, `UP`, `RIGHT`, `DOWN`,
`LSTICK_LEFT`, `LSTICK_UP`, `LSTICK_RIGHT`, `LSTICK_DOWN`, `RSTICK_LEFT`, `RSTICK_UP`, `RSTICK_RIGHT`,
`RSTICK_DOWN`. A field of another name stays as written. Never write a key's name in a text: use a field. A text
that opens with a field but lacks `live="1"` is shown as written and logged once. Four live texts over all DLCs, 255
characters each once filled in. *Seen.*

## The training menu

- The menu of the hub's training room lists 29 rows the game holds in its data: a value, a text of `menu_text`, a
  group. Choosing a row stores its value; the start reads the quest of a value 1 to 17 from a table of 17, of which
  the slots of the values 2 and 7 are empty. A `<training>` takes one of those two. *Seen.*
- The added row shows at the end of the main list (17 rows with it; the sparring list has 13). *Seen.*
- An added row always starts in the training zone: the port asks the game for its warp to the zone's start point at
  each start. *Seen* for the first start of a session; see [Known gaps](#known-gaps).
- The game marks a row "done" only for its own 29 rows, so an added row gets no mark and no "training complete"
  window. *Seen.*

## The file formats

For reading the tool's output or extending the builder. Every multi-byte value is little endian, except in BXM.
A **hash** is the crc32 of the lower-case name with the top bit cleared (31 bits); it names text ids, the files of a
DAT, flags, counters and groups. All four formats are checked against the game's files: every archive, DAT and text
file of 1.0.1 written back gives the same bytes; the 1 669 BXM files give the same trees.

**PKZ**, an archive (`quest/quest.pkz`, `Text/Text.pkz`): `"pkzl"`, u32 version, u64 file size, u32 entry count, u32
0x20 (the descriptors' offset), u64 string bytes. Per entry a descriptor of 32 bytes: u32 name offset and u32
compression-name offset into the strings, u64 size, u64 data offset, u64 compressed size. The strings: `"ZStandard"`
first (padded to 16 bytes), then the names in the entries' order, each padded to 8 bytes with at least one 0. Each
entry is one zstd frame at a multiple of 64. The entries are sorted by name without case. The game skips an entry
whose compression name is neither `None` nor `ZStandard`.

**DAT**, a file container (a quest): `"DAT\0"`, u32 count, then the offsets of five tables: the files' offsets, the
extensions (4 bytes each), the names (u32 width, then names of that width), the sizes, the hash table; u32 0. The
files' data follow, each at a multiple of 16; an empty file has offset 0 and size 0. The hash table: u32 shift, u32
0x10, u32 the hashes' offset, u32 the indexes' offset; then a u16 per bucket (the first slot whose hash's top bits
are the bucket, 0xffff for none), the hashes and the u16 file indexes, both sorted by bucket. The shift is 32 minus
log2 of the next power of two of the count, at most 31.

**BXM**, binary XML (big endian): `"XML\0"` or `"BXM\0"`, u32 0, u16 node count, u16 data row count, u32 string
bytes. Per node: u16 child count, u16 index of its first child (the node count for none), u16 attribute count, u16
data row. Per data row: u16 name offset and u16 value offset into the strings (0xffff for an empty one). The strings
are UTF-8, each with its 0. The nodes are breadth first, a node's children together; one data row per distinct
(name, value, attributes), its attributes' rows right after it; one string per distinct text. In XML: an element is
a node, its text the value.

**PTD**, a text file: `"PTD\0"`, u32 2, u32 0x26 (the key), u32 id count, u32 0x1c, u32 section count, u32 the
sections' offset. The id rows (16 bytes each, sorted by hash: hash, string offset from the row, characters, bytes)
and their strings; per section (20 bytes) a hash, list count, lists' offset, column count, columns' offset, from the
section. A list is a hash, a count and the offset of that many u32 (text hashes); a column is a hash, a row count
and the offset of its rows, which are as the id rows. Strings are UTF-16 with 0x26 added to every byte, the ending
0 included; the game takes the 0x26 off when it loads the file. A text is its id among the id rows, a row in each
column of the first section (the first column holds the text, the others an empty string) and its hash at the end
of each of that section's lists.

## The tool

`python tools/dlc_tool.py [--romfs DIR] <command>`:

| Command | Prints |
|---|---|
| `quests` | the quests of `quest/quest.pkz` with their files and sizes |
| `quest <id>` | a listing of the quest's script, then the start of its other files |
| `quest <id> --out DIR` | writes the quest's files into DIR, each `.bxm` as `.xml`: the form `quest/<id>/` takes |
| `commands` | every condition and action type: uses, parameters, most frequent values |
| `texts <file> [pattern] [--lang USen]` | the texts whose id or text holds the pattern |
| `hash <name>...` | the game's hash of each name |
| `check <DLC folder>` | parses `dlc.xml` and the quest XML; needs no game files |

`--romfs` is the folder with `romfs.bin` and `index.json`: an install folder's `romfs/`, or by default
`build/extracted/v101/romfs` of a checkout.

## Checking a DLC

| Command | What it checks | Needs |
|---|---|---|
| `python tools/dlc_tool.py check dlc/<name>` | the XML parses; the manifest's elements; the script's commands | nothing |
| `python tests/dlc_check.py`, part 1 | ids in the free ranges and used once; a `<t>` without `lang`; fields that are action names; only `.xml` and `.md`, LF, under 256 KiB; and that DLCs stay off by default (the setting, the ini example, the patch sets) | nothing |
| part 2 | `astralchain.exe --probe-dlc`: the builder on archives and DLC folders of its own, with broken manifests and archives | a build |
| part 3 | the two archives the build makes from the player's game with `dlc/`, against a second implementation in the test, byte for byte | a build and the game files |
| The game with "Load DLC" on | the `dlc:` and `quest:` log lines, then the DLC itself | an install |

`AC_TRACE=fs` in the environment adds a log line per file and per archive entry the game reads: it shows whether the
quest was loaded and which text files were read.

## The log

`logs/astralchain.log` next to the executable (the last run; `astralchain.1.log` the one before). The lines to know:

| Line | Meaning |
|---|---|
| `patches: ... quest on, questmenu on` | the port's quest code is active (`off` without `dlc = 1`) |
| `dlc: off (dlc = 0, ...)` | the setting is off: nothing was read |
| `dlc: on, but <dir> holds no DLC folder` | the folder is missing or empty |
| `dlc: <name>: N quests, N texts (N with input fields), N training rows, N tutorial windows` | the DLC's manifest as read |
| `dlc: <name>: no dlc.xml; left out` / `dlc.xml: <what> near "<text>"; left out` | the DLC was not loaded, and why |
| `dlc: <name>: dlc.xml: <element> is not understood, lacks an attribute or is one too many; left out` | that element only |
| `dlc: <name>: quest<id>.dat (N bytes) added to quest/quest.pkz` | the quest was built |
| `dlc: <name>: quest <id>: its base ... is not in the game's archive or is not a DAT; left out` | a wrong `base` |
| `dlc: <name>: quest <id>: <file>: <what>; the quest is left out` | a quest XML file that does not parse |
| `dlc: N texts added to N files of Text/Text.pkz` | the texts were built |
| `RomFS overlay: <file> built with the DLCs' entries in N ms` | the archive was built at this start (no such line: the cache was used) |
| `RomFS overlay: <file> (N bytes, the DLCs' entries) replaces the original` | the game reads the built archive |
| `quest: training menu row 29 added: <text> (value N) starts quest <id>` | the menu row is in place |
| `quest: training row ... left out: ...` | a value that is not free, or a second row |
| `quest: training value N starts quest <id> (...)` | the player started a training |
| `quest: host condition 0x... is true / false` | each change of a host condition's answer |
| `quest: text <id> filled in with the inputs bound in the Controls tab` | a live text was read |
| `quest: text 0x... opens with a field but no DLC names it with live="1"` | a missing `live="1"` |

## Known gaps

- **The warp to the training zone.** The first start of an added row warps the player into the zone. A second start
  in the same session fades but leaves the player in the hub; the quest still runs. Not explained: the game's warp
  runs, and its last step moves the player only when a table of the game holds no entry in state 3
  (`runtime/quest.c` logs the warp's state at each change, `quest: warp watch`).
- **No "training complete" mark** for an added row.
- **Dialogues and subtitles** from a DLC have not been tried in game.
- **No picture** in a DLC's tutorial window.
- **One added training row.** The patch sites count 30 rows in their instruction words, and the game has two free
  values.
- Many conditions and actions are only inferred, some not studied at all. A quest that tries one and a line in
  these tables is a welcome contribution.

## Extending the port

A DLC is data; what the game does with it is the port's code, in C, and the game's own code is patched only through
a table. The port is a static recompilation: the game's ARM64 code is lifted to C once (`recomp/`), so it cannot be
patched at run time.

- **`runtime/dlc.c`**, the builder: `manifest()` reads `dlc.xml`; `bxm_write`, `dat_build`, `pkz_write`, `ptd_write`
  are the four writers; `dlc_build` builds one archive; the cache key covers every source file. Another archive
  (`ui/ui.pkz` for pictures, say) is a case in `dlc_changes`, a builder, and one more `dlc_archive` call in
  `runtime/romfs.c`. Bump `DLC_BUILDER` when the builder writes other bytes for the same sources, so every cache is
  built again.
- **`runtime/quest.c`**, what the game cannot do from data. Its header gives the addresses in the game's `main` module
  (1.0.1, build id `4b159f0f7a360669738390ee8657b4f0`):

  | What | Where in `main` |
  |---|---|
  | A condition is evaluated (the command's `typeIF` at +4, the answer in w20) | 0x47f370; range check at 0x47f398 |
  | The two line runners | 0x47d700, 0x47db60 |
  | The script parser | 0x474d10 |
  | Action 18 | 0x5b8d30, then 0x6133c0; a tutorial window opens at 0x6101c0, changes at 0x611870 |
  | The table of the 215 tutorial windows ({number, chapter or -1}) | 0xd81018, through the GOT slot at 0xdbebe8 |
  | A help panel's lines, their texts and "done" tests | 0x613a00, 0x611e80, 0x612430 |
  | Every text lookup (x1 the result {bytes, characters, string at +8}, w3 the hash) | 0x8fbf60 |
  | The training menu's 29 rows ({value, text hash, group}) | 0xd71630, through the GOT slot at 0xdbbdc0 |
  | The quest of a menu value (17 slots, value - 1) | 0xbe30e0 |
  | The start of a training | 0x4bc8c0, 0x4bd3e0 |
  | Entering the training zone; the quest manager's update that warps | 0x491230, 0x48b268 |
  | A row marked done (rows 0 to 28 only) | 0x489a70 |
  | The quest manager (loaded quests at +0x248, warp state at +0x930, zone-warp request at +0xd10) | the GOT slot at 0xdb8460 |
  | An archive entry's compression name is checked | 0x9a6400 |

- **`recomp/patches-v101.txt`**, the patch table. A `site` line names a set, the module, an address, the instruction
  word expected there and its replacement: other words, or `@fn:<helper>`, a call of `void helper(Cpu *c)` in the
  runtime, which reads and writes the guest's registers as `c->x[n]`. The syntax is in the docstring of
  `recomp/patches.py`. A new site for DLCs goes into the set `quest` or `questmenu`: both follow the setting `dlc`
  and are off without it, so a player without DLCs runs the game's own code at every site. After a change to the
  table the game is lifted again (`recomp/gen.py`, about two minutes) and built (`tools/build_game.py`);
  `docs/building.md` has the toolchain and the commands.
- **A new manifest element** gets: its parsing in `manifest()`, a case in `runtime/dlc_probe.c` (well formed and
  broken), a case in `tests/dlc_check.py`, a paragraph in `docs/dlc.md` and here.

What a change to the port must keep:

- Nothing changes for a player with `dlc = 0`. `tests/dlc_check.py` part 1 checks the defaults and that every site
  calling a quest helper is in the two sets.
- Bad input never stops the game: a DLC is written by anyone. Check every offset and count read from a file, leave
  the entry out, and log one line that names the DLC, the file and the reason. `--probe-dlc` feeds the builder 2 000
  damaged archives and 300 damaged manifests.
- Code that blocks (file reads, compression) runs at start, before the game's threads, as the builder does.
- Comments and docs are plain, factual English: what the code does and why, with the measured numbers. Every source
  file opens with a header comment that says what it does.

## Contributing

- One folder per DLC in `dlc/`, with its row in [Ids in use](#ids-in-use), in the same pull request.
- Checks before a commit: `python tests/dlc_check.py` (say which parts ran: parts 2 and 3 need a build and the game
  files), `python tools/check_content.py --staged` (no game content; the project's CI runs it too), and for a change
  to the port `astralchain.exe --selftest` and `--probe-dlc`.
- Say in the pull request what you tried in game and what you only inferred.
- Commit messages are one plain sentence saying what changed and why, without a prefix such as `feat:`.
- Releases ship this folder; the installer copies it next to the game and replaces each DLC folder it ships whole. A
  DLC folder of another name, the player's own, is left alone.

## Ids in use

Take ids no row of this table has, and add yours.

| DLC | Quests | Windows | Training value | Text ids |
|---|---|---|---|---|
| `pc-controls` | `c0f0` | `af00` | 2 | `MENU_TUTO_NAME_DLC_00`, `CORE_TUTO_W_TITLE_af00`, `CORE_TUTO_W_NAME_af00_01`, `CORE_TUTO_W_HELP_af00_01` |

Free ranges to take from first: quests `c0f1` to `c0ff` (a training must be `c0xx`), windows `af01` to `afff`, text
ids with your DLC's name in them. The one free training value left is 7.
