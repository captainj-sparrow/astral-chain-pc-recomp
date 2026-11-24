// Quest script conditions answered by the host (recomp/patches-v101.txt, set quest).
//
// Everything in this file serves the DLCs (dlc.c, docs/dlc.md) and runs only with the setting dlc = 1: the sets quest
// and questmenu follow that setting and are off by default, so without it the game runs its own code at each of their
// sites and none of the helpers below is called.
//
// A quest (quest<id>.dat in quest/quest.pkz, its QuestData.bxm) is tasks of lines of commands, and a command is a
// condition (typeIF and its parameters) with an action (typeEXEC and its parameters). In Astral Chain 1.0.1 the game
// evaluates a condition at main+0x47f370 (quest, task, command): the command holds typeIF at +4 and the parameters
// from +8, the answer is built in w20, and a type outside 1..38 answers true (w20 = 1 before the range check at
// main+0x47f398). The game's 29 types test its own state only: flags, counters, enemy groups, areas, items. None reads
// the controller, the keyboard or the mouse.
//
// The site at main+0x47f398 calls rt_quest_if before the range check's branch, so a type the game does not know gets
// its answer here. The parser (main+0x474d10) keeps any typeIF as written and reads no parameter for a type it does
// not know, so a host condition carries its argument in the type itself:
//
//   0x10000 + buttons             true while all of these controller buttons are held
//   0x20000 + buttons             true while one of them is not
//   0x30000 + key1 * 256 + key2   true while both keys are held (key2 = 0: key1 alone)
//   0x40000 + key1 * 256 + key2   true while one of them is not
//
// buttons are the low 16 bits of the npad state (A 0, B 1, X 2, Y 3, left and right stick click 4 5, L 6, R 7, ZL 8,
// ZR 9, Plus 10, Minus 11, D-pad left, up, right, down 12..15) as the game last read them (ov_pad_buttons), whatever
// held them: a gamepad, or the keys bound to them.
// A key is a key of the keyboard or a button of the mouse whatever it is bound to (input_key_held), by a code of this
// file's own, kept when the runtime's key ids change (quest_key): 1..5 the mouse's left, right, middle and two side
// buttons, 9 Tab, 13 Enter, 16 left Shift, 17 left Ctrl, 32 Space, '0'..'9' and 'A'..'Z' the digit and letter keys.
// Nothing counts as held while the menu or a text entry is open, or while another window is in front.
//
// The game's action 1 holds its line for as long as its condition is true (both line runners, main+0x47d700 and
// main+0x47db60, return at once and start the line again on the next frame), so a line waits for a press with
// "typeIF 0x20000 + buttons (or 0x40000 + keys), typeEXEC 1".
//
// Checked with a quest written for it, in place of the IRIS training (questc00b): its line waited on 0x20300 and went on
// when ZL and ZR were held together, and the quest ended.
//
// The helper names its registers by number and passes the Cpu to nothing, which recomp/ipa.py needs to keep the
// evaluator's other registers in locals across the call. The log names each change of a host condition's answer.
//
// Training menu rows added by the port (set questmenu). The menu of the hub's training room lists the rows of a table
// in main's data, main+0xd71630: 29 rows of {menu value, text hash, group}, the text a row of
// Text/menu_text_<language>.bin by the crc32 of its lower-case id (31 bits), the group 2 or 3 for the rows a scenario
// flag hides (main+0x4bd260; any other group shows). The game reaches the table through its GOT slot at main+0xdbbdc0
// and counts 29 rows in five places: the widget's fill (main+0x6bbb80, a loop to 29), the count of the rows shown
// (main+0x4bcff0, unrolled), the value's row (main+0x4bc5c0, unrolled), the row of the chosen value (main+0x4bcd50, a
// loop to 29). Choosing a row stores its value; the start of the training (main+0x4bc8c0, main+0x4bd3e0) reads the
// quest of a value 1..17 from the table main+0xbe30e0 (read-only) at value - 1, 0 for none.
// quest_init copies the 29 rows to the patch page (+0x400) with the added rows after them and points the GOT slot at
// the copy; the set's sites count 30 rows, find the added row's value, count it among the rows of the main list (its
// group 0 always shows) and hand the added row's quest to the start. The added row takes the value 2, whose slot in
// the quest table is empty (the "Dashing" row of the menu's texts, which no row uses). Its done mark is the word after
// the 29 marks of the save's table (+0xb4118 of the save data, row index * 4), which no code of the game reads or
// writes (the next field the code names is at +0xb41b4).
// The widget holds 29 rows (+0x360 and +0x3d4 of the menu, 29 words each); the main list shows 17 rows with the
// added one, the sparring list 13.
// The game sets a row's done mark only for its 29 rows (main+0x489a70 returns for a row above 28, so an added row
// also gets no "training complete" window); choosing a row sets the top bit of its word (main+0x6bbf20).
// An added row always starts in the training zone: the start asks the quest manager (main's GOT slot at
// main+0xdb8460) for its warp to the zone's start point, +0xd10 = 1, which the manager's update (main+0x48b268) does
// with a fade once no event runs. The game sets it itself (main+0x491230) only on some of its paths: with the first
// version of the added row the first start from the hub's terminal warped and the second one left the player in the
// hub. Each start logs the manager's state (the modes at +0xd04 and +0xd08, the warp at +0x930, the quests loaded at
// +0x248, 24 slots), since that difference was not explained.
//
// Tutorial windows added by the port (set quest). Action 18 of a quest script (DispTipsEnable, DispTipsNo; main
// +0x5b8d30) shows two kinds of help. A number below 0xa000 is a panel of button hints, each number a fixed list of
// lines in the code (main+0x6133c0) and each line's text and "done" test code of its own (main+0x611e80,
// main+0x612430): nothing to add from a file. A number from 0xa000 is a tutorial window (main+0x6101c0; the same
// lookup at main+0x611870 when a shown window changes): title CORE_TUTO_W_TITLE_<number>, per page
// CORE_TUTO_W_NAME_<number>_<page> and CORE_TUTO_W_HELP_<number>_<page> (pages from 01; the game counts the HELP
// texts there are, main+0x6110e0) of Text/Core_text_<language>.bin and a picture tutorial_thumb_<number + page>. The
// game shows a window only when its number is in the table at main+0xd81018 (215 rows of {number, the chapter that
// lists it in the archive or -1}, through the GOT slot at main+0xdbebe8, which six other functions read with the
// count 215 in their code); another number closes the window at once. Both lookups end with "mov w20, #0xffff" when
// the number is not found: the sites call rt_quest_window there and branch back to the found path (with w21 = -1,
// in no chapter) for a number of windows[] below (the DLCs' <window>, dlc.c). A quest waits for the window to close with condition 33, as the
// game's quests do.
//
// Texts the port fills in (set quest). The game finds a text by its id's hash in the language's table
// (main+0x8fbf60: x1 the result {u32 bytes, u32 chars, string}, w3 the hash; every text lookup ends there), the
// string UTF-16 (the game takes the file's + 0x26 off every byte when it loads the table; its callers measure the
// string up to a 0, main+0x579a10). The site after its store of the string calls
// rt_quest_text, which for an id of live_ids[] (the DLCs' texts with live="1", dlc.c) puts in its place the file's text with each {ACTION} field (an action
// name of input_map.h: ZL, ZR, A, PLUS...) replaced by the inputs bound to it, with the labels of the menu's Controls
// tab (ui_bind_label): "{ZL} + {ZR}" reads "Left mouse / Z + Right mouse / X". The result goes to a slot of the
// patch page (+0x600, 0x200 bytes each) and follows a change of the bindings at the next lookup (the window reads its
// texts when it opens).
//
// The warp watch: rt_quest_if, which runs at every condition of every quest, logs each change of the quest manager's
// warp fields (the zone warp request +0xd10, the warp state +0x930 and its fade +0x93c, the tutorial and free combat
// modes +0xd04 and +0xd08), so a log tells what a training start did with the warp it asked for.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cpu.h"
#include "dlc.h"
#include "host.h"
#include "input.h"
#include "input_map.h"
#include "layout.h"
#include "ov.h"
#include "ui_text.h"

#if __has_include("patches.h")
#include "patches.h"
#endif

enum { QUEST_IF_PAD = 1, QUEST_IF_PAD_NOT = 2, QUEST_IF_KEYS = 3, QUEST_IF_KEYS_NOT = 4 };

// A key code of the quest files as the runtime's key id, KEY_NONE for a code without a key.
static int quest_key(uint32_t code) {
    if (code >= 1 && code <= 5) return KEY_MOUSE_L + (int)code - 1;
    if (code >= '0' && code <= '9') return KEY_0 + (int)(code - '0');
    if (code >= 'A' && code <= 'Z') return KEY_A + (int)(code - 'A');
    switch (code) {
    case 9: return KEY_TAB;
    case 13: return KEY_ENTER;
    case 16: return KEY_LSHIFT;
    case 17: return KEY_LCTRL;
    case 32: return KEY_SPACE;
    default: return KEY_NONE;
    }
}

static int keys_held(uint32_t arg) {
    uint32_t k1 = arg >> 8, k2 = arg & 0xff;
    return input_key_held(quest_key(k1)) && (!k2 || input_key_held(quest_key(k2)));
}

#define QUEST_MGR_GOT (MOD_MAIN_BASE + 0xdb8460ull)
enum { MGR_SLOTS = 0x248, MGR_SLOT_COUNT = 24, MGR_WARP = 0x930, MGR_WARP_FADE = 0x93c, MGR_TUTORIAL = 0xd04,
       MGR_FREE_COMBAT = 0xd08, MGR_ZONE_WARP = 0xd10 };
static volatile uint32_t *quest_mgr(void) {
    return (volatile uint32_t *)(uintptr_t)*(volatile uint64_t *)(uintptr_t)QUEST_MGR_GOT;
}

// The quest manager's warp fields, logged when one changes (the warp watch in the header).
static void warp_watch(void) {
    static uint32_t last[5] = {~0u, ~0u, ~0u, ~0u, ~0u};
    volatile uint32_t *m = quest_mgr();
    uint32_t now[5] = {m[MGR_ZONE_WARP / 4], m[MGR_WARP / 4], m[MGR_WARP_FADE / 4], m[MGR_TUTORIAL / 4],
                       m[MGR_FREE_COMBAT / 4]};
    if (!memcmp(now, last, sizeof now)) return;
    memcpy(last, now, sizeof now);
    // The warp's state 2 (main+0x48b59c) moves the player only when no entry of the table at main's GOT slot
    // main+0xdb8970 (count at +0x8b8, entries of 0xd0 from +0x8b8 + 0xc) has the value 3 at its +0xc.
    const uint8_t *tab = (const uint8_t *)(uintptr_t)*(volatile uint64_t *)(uintptr_t)(MOD_MAIN_BASE + 0xdb8970ull);
    char states[16 * 11 + 1];   // 16 values of at most 10 digits and a space
    size_t n = 0;
    int count = tab ? *(const volatile int32_t *)(tab + 0x8b8) : 0, blocking = 0;
    for (int i = 0; i < count && i < 16; i++) {
        uint32_t v = *(const volatile uint32_t *)(tab + 0x8c4 + 0xd0 * i);
        blocking += v == 3;
        n += (size_t)snprintf(states + n, sizeof states - n, " %u", v);
    }
    states[n] = 0;
    rt_log("quest: warp watch: zone warp asked %u, warp state %u, fade %d, tutorial mode %u, free combat mode %u; "
           "%d entries of the move's table, %d in state 3 (no move):%s",
           now[0], now[1], (int)now[2], now[3], now[4], count, blocking, n ? states : " none");
}

void rt_quest_if(Cpu *c) {
    static uint32_t last_type, last_answer;
    warp_watch();
    uint32_t type = *(const uint32_t *)(c->x[19] + 4);
    uint32_t kind = type >> 16, arg = type & 0xffff;
    if (kind < QUEST_IF_PAD || kind > QUEST_IF_KEYS_NOT) return;
    int held = kind <= QUEST_IF_PAD_NOT ? ((uint32_t)ov_pad_buttons() & arg) == arg : keys_held(arg);
    uint32_t answer = (uint32_t)(kind == QUEST_IF_PAD || kind == QUEST_IF_KEYS ? held : !held);
    if (type != last_type || answer != last_answer)
        rt_log("quest: host condition %#x (%s %#x %s) is %s", type, kind <= QUEST_IF_PAD_NOT ? "buttons" : "keys", arg,
               kind == QUEST_IF_PAD || kind == QUEST_IF_KEYS ? "held" : "not held", answer ? "true" : "false");
    last_type = type, last_answer = answer;
    c->x[20] = answer;
}

// ---- training menu rows (set questmenu)
enum { MENU_ROWS = 29, MENU_PAGE = 0x400 };
#define MENU_TABLE (MOD_MAIN_BASE + 0xd71630ull)
#define MENU_GOT (MOD_MAIN_BASE + 0xdbbdc0ull)
typedef struct { uint32_t value, text, group; } MenuRow;
// The row added after the game's, with its quest: the first <training> of the DLCs (dlc.c; quest_init). One row:
// recomp/patches-v101.txt counts 29 + 1 rows in the sites' instruction words.
enum { ADDED_MAX = 1 };
static struct { MenuRow row; uint32_t quest; const char *text_id; } added[ADDED_MAX];
static uint32_t nadded;
static int menu_on;   // the copy is in place and the GOT slot points at it

static const MenuRow *menu_rows(void) { return (const MenuRow *)(uintptr_t)*(volatile uint64_t *)(uintptr_t)MENU_GOT; }

// main+0x4bc5c0, after its 28 unrolled tests: w0 = the row of value w0 among row 28 and the added rows, else 0 (the
// csel it replaces: row 28 or 0).
void rt_quest_menu_index_w0(Cpu *c) {
    uint32_t v = (uint32_t)c->x[0];
    const MenuRow *t = menu_rows();
    uint32_t row = t[28].value == v ? 28 : 0;
    if (menu_on && !row)
        for (uint32_t i = 0; i < nadded; i++)
            if (added[i].row.value == v) row = MENU_ROWS + i;
    c->x[0] = row;
}

// main+0x4bcff0, its count of the rows shown: w19 is 0 for the main list, 1 for the sparring list; the added rows
// are in the main list.
void rt_quest_menu_count_w0(Cpu *c) {
    if (menu_on && (uint32_t)c->x[19] == 0) c->x[0] = (uint32_t)c->x[0] + nadded;
}

// The start of the training: w8 = the chosen value - 1, the quest loaded from the game's table into w19 / w20 (0
// for none); an added row's quest instead, with the warp to the training zone asked for.
static uint32_t menu_quest(uint32_t index, uint32_t game_quest, const char *from) {
    uint32_t quest = game_quest;
    int ours = 0;
    if (menu_on)
        for (uint32_t i = 0; i < nadded; i++)
            if (added[i].row.value - 1 == index) quest = added[i].quest, ours = 1;
    volatile uint32_t *mgr = quest_mgr();
    char loaded[MGR_SLOT_COUNT * 5 + 1];
    size_t n = 0;
    for (int i = 0; i < MGR_SLOT_COUNT; i++) {
        uint32_t id = mgr[MGR_SLOTS / 4 + i];
        if (id) n += (size_t)snprintf(loaded + n, sizeof loaded - n, " %04x", id & 0xffff);
    }
    loaded[n] = 0;
    rt_log("quest: training value %u starts quest %04x (%s, %s): before it tutorial mode %u, free combat mode %u, "
           "zone warp asked %u, warp state %u, fade %d; quests loaded:%s", index + 1, quest, from,
           ours ? "a row of the port, zone warp asked" : "a row of the game", mgr[MGR_TUTORIAL / 4],
           mgr[MGR_FREE_COMBAT / 4], mgr[MGR_ZONE_WARP / 4], mgr[MGR_WARP / 4], (int)mgr[MGR_WARP_FADE / 4],
           n ? loaded : " none");
    if (ours) mgr[MGR_ZONE_WARP / 4] = 1;
    return quest;
}
void rt_quest_menu_quest_w19(Cpu *c) {
    c->x[19] = menu_quest((uint32_t)c->x[8], (uint32_t)c->x[19], "main+0x4bc8c0");
}
void rt_quest_menu_quest_w20(Cpu *c) {
    c->x[20] = menu_quest((uint32_t)c->x[8], (uint32_t)c->x[20], "main+0x4bd3e0");
}

// ---- tutorial windows (set quest)
// The window numbers of the DLCs (dlc.c <window>; quest_init).
enum { WINDOWS_MAX = 16 };
static uint32_t windows[WINDOWS_MAX];
static uint32_t nwindows;

// The game's lookup of the window number w20 found nothing. For a window of the port: w21 = -1 (in no chapter of the
// archive) and w11 = 1, which the site's cbnz takes back to the found path; else w20 = 0xffff as the instruction it
// replaces and w11 = 0 (x11 is dead at both sites: every path from them calls a function or returns before reading
// it).
void rt_quest_window(Cpu *c) {
    uint32_t number = (uint32_t)c->x[20];
    for (uint32_t i = 0; i < nwindows; i++)
        if (windows[i] == number) {
            c->x[21] = 0xffffffffu;
            c->x[11] = 1;
            return;
        }
    c->x[20] = 0xffff;
    c->x[11] = 0;
}

// ---- texts the port fills in (set quest)
// The text ids whose {ACTION} fields take the bound inputs: the DLCs' texts with live="1" (dlc.c; quest_init).
enum { LIVE_MAX = 4, TEXT_PAGE = 0x600, TEXT_SLOT = 0x200, TEXT_CHARS = TEXT_SLOT / 2 };
_Static_assert(LIVE_MAX * TEXT_SLOT <= 0x800, "the patch page holds 4 slots (recomp/patches.py)");
static const char *live_ids[LIVE_MAX];
static uint32_t live_hash[LIVE_MAX];
static int nlive;

// The action named by name[0..n) (input_map.h's names, any case), -1 for none.
static int action_named(const uint16_t *name, size_t n) {
    for (int a = 0; a < ACTION_COUNT; a++) {
        const char *want = input_action_name(a);
        size_t i = 0;
        for (; want && i < n && want[i]; i++) {
            uint16_t ch = name[i] >= 'a' && name[i] <= 'z' ? (uint16_t)(name[i] - 'a' + 'A') : name[i];
            if (ch != (uint8_t)want[i]) break;
        }
        if (want && i == n && !want[i]) return a;
    }
    return -1;
}

// Appends UTF-8 s to out[*len..cap) as UTF-16 (outside the BMP: U+FFFD).
static void put_utf8(uint16_t *out, size_t *len, size_t cap, const char *s) {
    const uint8_t *p = (const uint8_t *)s;
    while (*p && *len < cap) {
        uint32_t ch = *p++;
        int more = ch >= 0xF0 ? 3 : ch >= 0xE0 ? 2 : ch >= 0xC0 ? 1 : 0;
        if (more) ch &= 0x3Fu >> more;
        for (; more && (*p & 0xC0) == 0x80; more--) ch = ch << 6 | (*p++ & 0x3F);
        out[(*len)++] = (uint16_t)(ch > 0xFFFF || more ? 0xFFFD : ch);
    }
}

// main+0x8fc090, after the lookup's store of the string: for a text of live_ids[], the text with its fields filled
// (the texts in the header). x1 the result, w3 the id's hash.
void rt_quest_text(Cpu *c) {
    uint32_t hash = (uint32_t)c->x[3];
    int slot = -1;
    for (int i = 0; i < nlive; i++)
        if (live_hash[i] == hash) slot = i;
    uint8_t *res = (uint8_t *)(uintptr_t)c->x[1];
    const uint16_t *src = (const uint16_t *)(uintptr_t)*(const uint64_t *)(res + 8);
    uint32_t chars = *(const uint32_t *)(res + 4);
    if (slot < 0) {
        // A text that opens with a field but is not one of live_ids[] (a DLC's text without live="1"): logged once
        // per hash, the last 8 kept.
        static uint32_t seen[8];
        static unsigned next;
        if (chars && src[0] == '{') {
            for (int i = 0; i < 8; i++)
                if (seen[i] == hash) return;
            seen[next++ % 8] = hash;
            rt_log("quest: text %#x opens with a field but no DLC names it with live=\"1\"; shown as written", hash);
        }
        return;
    }
    static uint32_t logged;
    if (!(logged & 1u << slot)) {
        logged |= 1u << slot;
        rt_log("quest: text %s (%#x) filled in with the inputs bound in the Controls tab", live_ids[slot], hash);
    }
    uint16_t text[TEXT_CHARS];
    size_t len = 0, cap = TEXT_CHARS - 1;
    for (uint32_t i = 0; i < chars && len < cap; i++) {
        uint16_t ch = src[i];
        if (!ch) break;
        if (ch == '{') {
            uint16_t name[16];
            uint32_t j = i + 1, n = 0;
            for (; j < chars && n < 16; j++, n++) {
                name[n] = src[j];
                if (name[n] == '}') break;
            }
            int action = j < chars && n < 16 ? action_named(name, n) : -1;
            if (action >= 0) {
                char label[96];
                if (!ui_bind_label(action, label, sizeof label)) snprintf(label, sizeof label, "%s",
                                                                          input_action_name(action));
                put_utf8(text, &len, cap, label);
                i = j;
                continue;
            }
        }
        text[len++] = ch;
    }
    text[len++] = 0;
    uint8_t *dst = (uint8_t *)rt_patch_page() + TEXT_PAGE + (size_t)slot * TEXT_SLOT;
    memcpy(dst, text, len * sizeof *text);
    *(uint32_t *)res = (uint32_t)(2 * len);
    *(uint32_t *)(res + 4) = (uint32_t)len;
    *(uint64_t *)(res + 8) = (uint64_t)(uintptr_t)dst;
}

void quest_init(void) {
    dlc_scan();
    const char *id;
    for (int i = 0; nlive < LIVE_MAX && dlc_live_text(i, &id); i++) live_ids[nlive] = id, live_hash[nlive++] = dlc_hash(id);
    for (int i = 0; nwindows < WINDOWS_MAX && dlc_window(i, &windows[nwindows]); i++) nwindows++;
#ifdef RT_PATCH_COUNT
    int set = rt_patch_set("questmenu");
    if (!rt_patch_enabled(set)) return;
    const MenuRow *game = (const MenuRow *)(uintptr_t)MENU_TABLE;
    uint64_t got = *(volatile uint64_t *)(uintptr_t)MENU_GOT;
    if (got != MENU_TABLE || game[28].value != 0x3f || game[28].text != 0x7ecd6873 || game[0].value != 1) {
        rt_log("quest: the training menu's table is not the one of 1.0.1 (GOT %#llx, row 28 {%#x, %#x}); "
               "no rows added", (unsigned long long)got, game[28].value, game[28].text);
        rt_patch_disable(set);
        return;
    }
    // A row's value picks its quest from the game's table of 17 (main+0xbe30e0, value - 1): only a value whose slot
    // is empty there is free, 2 and 7 in 1.0.1.
    const uint32_t *quest_table = (const uint32_t *)(uintptr_t)(MOD_MAIN_BASE + 0xbe30e0ull);
    uint32_t value, quest;
    for (int i = 0; dlc_training(i, &value, &id, &quest); i++) {
        if (nadded == ADDED_MAX) {
            rt_log("quest: training row %s (value %u) left out: the menu takes one added row", id, value);
        } else if (value < 1 || value > 17 || quest_table[value - 1]) {
            rt_log("quest: training row %s left out: its value %u is not free in the game's menu (2 and 7 are)", id,
                   value);
        } else {
            added[nadded].row = (MenuRow){value, dlc_hash(id), 0};
            added[nadded].quest = quest, added[nadded++].text_id = id;
        }
    }
    if (!nadded) {   // the sites count 30 rows
        rt_patch_disable(set);
        return;
    }
    MenuRow *copy = (MenuRow *)((uint8_t *)rt_patch_page() + MENU_PAGE);
    memcpy(copy, game, MENU_ROWS * sizeof *copy);
    for (uint32_t i = 0; i < nadded; i++) copy[MENU_ROWS + i] = added[i].row;
    *(volatile uint64_t *)(uintptr_t)MENU_GOT = (uint64_t)(uintptr_t)copy;
    menu_on = 1;
    for (uint32_t i = 0; i < nadded; i++)
        rt_log("quest: training menu row %u added: %s (value %u) starts quest %04x", MENU_ROWS + i, added[i].text_id,
               added[i].row.value, added[i].quest);
#endif
}
