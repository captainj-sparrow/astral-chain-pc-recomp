// --probe-dlc: the DLC builder (dlc.c) on archives and DLC folders of the probe's own. Needs no game files.
//
// It writes a DLC folder next to astralchain.exe (dlc-probe/, removed at the end) and two small archives in the
// game's formats, has dlc_build add the DLCs to them, and reads the result back with readers of this file, written
// apart from dlc.c's writers: the archive's header, strings and alignment, a DAT's files through its hash table as the
// game finds them, a BXM's tree, a text file's rows. What it checks:
//   - the setting dlc is off unless asked, and nothing is read then;
//   - a quest with a base (the base's files under the new id, the folder's over them), one without, a file taken as it
//     is, a text per language with its default, a line break, an entity, a character outside the BMP;
//   - every entry of the game the DLCs do not touch keeps its compressed bytes;
//   - a DLC the port cannot use is left out and the others load: no manifest, XML that does not parse, elements and
//     attributes that mean nothing, a base the archive lacks, a text file that is not one;
//   - the same sources give the same bytes and the same key, and another source another key;
//   - archives and manifests cut short or with bytes changed never stop the process (2 000 archives, 300 manifests,
//     from a fixed seed): dlc_build answers NULL or an archive.
// With AC_DLC_PROBE_DIR=<dir> it builds instead <dir>/quest.pkz and <dir>/Text.pkz (the game's own archives, which
// tests/dlc_check.py puts there) with the DLC folders of <dir>/dlc into <dir>/out/, for that test to compare with its
// own writers.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <zstd.h>

#include "cfg.h"
#include "dlc.h"
#include "host.h"
#include "platform/platform.h"

static int fails, checks;
#define CHECK(c, what) do { checks++; if (!(c)) { fails++; fprintf(stderr, "FAIL dlc: %s\n", what); } } while (0)

static uint32_t r32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t r64(const uint8_t *p) { return r32(p) | (uint64_t)r32(p + 4) << 32; }
static uint32_t b16(const uint8_t *p) { return (uint32_t)p[0] << 8 | p[1]; }

static void put(const char *root, const char *rel, const char *text) {
    char p[2048];
    snprintf(p, sizeof p, "%s/%s", root, rel);
    char *slash = strrchr(p, '/');
    *slash = 0;
    plat_file_mkdir_p(p);
    *slash = '/';
    if (slash[1]) plat_file_write(p, text, strlen(text));
}

static int remove_cb(const char *name, int kind, uint64_t size, void *ctx);
static void remove_tree(const char *dir) {
    plat_fs_list(dir, remove_cb, (void *)dir);
    plat_fs_rmdir(dir);
}
static int remove_cb(const char *name, int kind, uint64_t size, void *ctx) {
    (void)size;
    char sub[2048];
    snprintf(sub, sizeof sub, "%s/%s", (const char *)ctx, name);
    if (kind == PLAT_FS_DIR) remove_tree(sub);
    else plat_fs_unlink(sub);
    return 1;
}

// ---- readers of this file
typedef struct { const char *name; uint64_t size, off, comp; const uint8_t *frame; } Entry;

// The entries of an archive into e (at most max), with the layout the game's archives have; -1 when one rule fails.
static int pkz_entries(const uint8_t *p, size_t n, Entry *e, int max) {
    if (n < 0x20 || memcmp(p, "pkzl", 4) || r64(p + 8) != n || r32(p + 20) != 0x20 || n % 64) return -1;
    uint32_t count = r32(p + 16);
    uint64_t strings = r64(p + 24), at = 0x20 + 32ull * count;
    if (count > (uint32_t)max || at + strings > n || strings < 16 || memcmp(p + at, "ZStandard\0\0\0\0\0\0", 16)) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *d = p + 0x20 + 32 * (size_t)i;
        e[i] = (Entry){(const char *)p + at + r32(d), r64(d + 8), r64(d + 16), r64(d + 24), p + r64(d + 16)};
        if (r32(d) >= strings || r32(d + 4) || e[i].off % 64 || e[i].off + e[i].comp > n) return -1;
        if (i && strcasecmp(e[i - 1].name, e[i].name) >= 0) return -1;   // sorted without case, no name twice
    }
    return (int)count;
}

static const Entry *entry_named(const Entry *e, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (!strcmp(e[i].name, name)) return &e[i];
    return NULL;
}

// An entry's bytes, malloc'd.
static uint8_t *unpack(const Entry *e) {
    uint8_t *raw = malloc(e->size ? (size_t)e->size : 1);
    if (raw && ZSTD_decompress(raw, (size_t)e->size, e->frame, (size_t)e->comp) != e->size) {
        free(raw);
        raw = NULL;
    }
    return raw;
}

// The file `name` of a DAT, found through its hash table as the game finds it; NULL for none. *count: its files.
static const uint8_t *dat_file(const uint8_t *d, size_t n, const char *name, size_t *size, uint32_t *count) {
    if (n < 0x20 || memcmp(d, "DAT", 4) || n % 16) return NULL;
    uint32_t files = r32(d + 4), offs = r32(d + 8), names = r32(d + 16), sizes = r32(d + 20), table = r32(d + 24);
    uint32_t width = r32(d + names), shift = r32(d + table), buckets = r32(d + table + 4), hashes = r32(d + table + 8),
             indexes = r32(d + table + 12), hash = dlc_hash(name);
    if (count) *count = files;
    uint32_t k = d[table + buckets + 2 * (hash >> shift)] | d[table + buckets + 2 * (hash >> shift) + 1] << 8;
    for (; k < files && r32(d + table + hashes + 4 * k) >> shift == hash >> shift; k++) {
        if (r32(d + table + hashes + 4 * k) != hash) continue;
        uint32_t i = d[table + indexes + 2 * k] | d[table + indexes + 2 * k + 1] << 8;
        if (i >= files || strcmp((const char *)d + names + 4 + i * width, name)) return NULL;
        const char *dot = strrchr(name, '.');
        char ext[4] = {0};
        strncpy(ext, dot ? dot + 1 : name, 4);
        if (memcmp(d + r32(d + 12) + 4 * i, ext, 4) || r32(d + offs + 4 * i) % 16) return NULL;
        *size = r32(d + sizes + 4 * i);
        return d + r32(d + offs + 4 * i);
    }
    return NULL;
}

// A BXM's tree as text into out: name=value[attribute=value...]{children}.
static void bxm_node(const uint8_t *b, uint32_t i, char *out, size_t cap, int depth) {
    uint32_t nodes = b16(b + 8), rows = b16(b + 10);
    const uint8_t *node = b + 16 + 8 * i, *row = b + 16 + 8 * nodes, *str = row + 4 * rows;
    uint32_t kids = b16(node), first = b16(node + 2), attrs = b16(node + 4), at = b16(node + 6);
    size_t len = strlen(out);
#define STR(off) ((off) == 0xffff ? "" : (const char *)str + (off))
    len += (size_t)snprintf(out + len, cap - len, "%s=%s", STR(b16(row + 4 * at)), STR(b16(row + 4 * at + 2)));
    for (uint32_t a = 1; a <= attrs && len < cap; a++)
        len += (size_t)snprintf(out + len, cap - len, "[%s=%s]", STR(b16(row + 4 * (at + a))), STR(b16(row + 4 * (at + a) + 2)));
#undef STR
    if (!kids || depth > 16 || len >= cap) return;
    snprintf(out + len, cap - len, "{");
    for (uint32_t k = 0; k < kids; k++) {
        if (k) strncat(out, ",", cap - strlen(out) - 1);
        bxm_node(b, first + k, out, cap, depth + 1);
    }
    strncat(out, "}", cap - strlen(out) - 1);
}
static const char *bxm_text(const uint8_t *b, size_t n, const char *magic) {
    static char out[2048];
    out[0] = 0;
    if (!b || n < 16 || memcmp(b, magic, 4) || r32(b + 4)) return "not a BXM";
    uint32_t nodes = b16(b + 8), rows = b16(b + 10);
    if (16 + 8ull * nodes + 4ull * rows + (b[12] << 24 | b[13] << 16 | b[14] << 8 | b[15]) != n) return "sizes do not add up";
    bxm_node(b, 0, out, sizeof out, 0);
    return out;
}

// The text `id` of a text file as UTF-16 code units (0 terminated) into out; -1 for none or a broken file, else the
// units. The id row, the first column's row and the first list must all hold it.
static int ptd_text(const uint8_t *d, size_t n, const char *id, uint16_t *out, int cap) {
    if (n < 0x1c || memcmp(d, "PTD", 4) || r32(d + 4) != 2 || r32(d + 8) != 0x26 || r32(d + 16) != 0x1c) return -1;
    uint32_t hash = dlc_hash(id), ids = r32(d + 12), sec = r32(d + 24), found = 0, last = 0;
    for (uint32_t i = 0; i < ids; i++) {
        const uint8_t *r = d + 0x1c + 16 * (size_t)i;
        if (i && r32(r) <= last) return -1;   // sorted by hash
        last = r32(r);
        if (last != hash) continue;
        const uint8_t *s = r + r32(r + 4);
        for (size_t k = 0; id[k]; k++)
            if ((uint8_t)(s[2 * k] - 0x26) != (uint8_t)id[k] || (uint8_t)(s[2 * k + 1] - 0x26)) return -1;
        found = 1;
    }
    if (!found || !r32(d + 20)) return -1;
    const uint8_t *s = d + sec, *list = s + r32(s + 8), *col = s + r32(s + 16);
    int listed = 0;
    for (uint32_t v = 0; v < r32(list + 4); v++) listed += r32(list + r32(list + 8) + 4 * (size_t)v) == hash;
    if (listed != 1) return -1;
    const uint8_t *rows = col + r32(col + 8);
    for (uint32_t i = 0; i < r32(col + 4); i++) {
        const uint8_t *r = rows + 16 * (size_t)i;
        if (r32(r) != hash) continue;
        uint32_t chars = r32(r + 8);
        if (r32(r + 12) != 2 * chars || (int)chars > cap || !chars) return -1;
        for (uint32_t k = 0; k < chars; k++)
            out[k] = (uint16_t)((uint8_t)(r[r32(r + 4) + 2 * k] - 0x26) | (uint8_t)(r[r32(r + 4) + 2 * k + 1] - 0x26) << 8);
        return out[chars - 1] ? -1 : (int)chars - 1;
    }
    return -1;
}

static int text_is(const uint8_t *d, size_t n, const char *id, const uint16_t *want) {
    uint16_t got[128];
    int len = ptd_text(d, n, id, got, 128), k = 0;
    while (want[k]) k++;
    return len == k && !memcmp(got, want, (size_t)k * 2);
}

// ---- the probe's data
static const char QUEST_XML[] =
    "\xEF\xBB\xBF<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<!-- a comment before the root -->\n"
    "<QuestRoot>\n"
    "  <Version>16</Version>\n"
    "  <QuestData>\n"
    "    <TaskList TaskName=\"start &amp; go\" TemplateName=''>\n"
    "      <TaskEnable>1</TaskEnable> <!-- a comment between elements -->\n"
    "      <CommandList><typeIF>131840</typeIF><typeEXEC>1</typeEXEC></CommandList>\n"
    "      <CommandList><typeIF>131840</typeIF><typeEXEC>1</typeEXEC></CommandList>\n"
    "    </TaskList>\n"
    "  </QuestData>\n"
    "  <Empty/>\n"
    "</QuestRoot>\n";
static const char QUEST_TREE[] =
    "QuestRoot={Version=16,QuestData={TaskList=[TaskName=start & go][TemplateName=]{TaskEnable=1,"
    "CommandList={typeIF=131840,typeEXEC=1},CommandList={typeIF=131840,typeEXEC=1}}},Empty=}";

static const char MANIFEST_A[] =
    "<dlc>\n"
    "  <quest id=\"c0f1\" base=\"c00b\"/>\n"
    "  <quest id=\"c0f2\"/>\n"
    "  <quest id=\"c0f3\" base=\"ffff\"/>\n"
    "  <training value=\"2\" text=\"PROBE_MENU\" quest=\"c0f1\"/>\n"
    "  <window number=\"af01\"/>\n"
    "  <text file=\"Core_text\" id=\"PROBE_ONE\" live=\"1\">\n"
    "    <t>one {ZL} &lt;\xC3\xA9&gt; \xF0\x9F\x98\x80</t>\n"
    "    <t lang=\"EUfr\">un\\nligne</t>\n"
    "  </text>\n"
    "  <text file=\"Core_text\" id=\"OLD_TEXT\"><t>replaced</t></text>\n"
    "  <text file=\"menu_text\" id=\"PROBE_MENU\"><t>Probe</t></text>\n"
    "</dlc>\n";
static const char MANIFEST_ODD[] =
    "<dlc>\n"
    "  <quest id=\"zzzz\"/> <quest/> <quest id=\"c0f4\"/>\n"
    "  <training/> <training value=\"x\" text=\"T\" quest=\"nope\"/>\n"
    "  <window number=\"12345\"/> <window/>\n"
    "  <text file=\"Core_text\"/> <text file=\"Core_text\" id=\"NO_T\"/> <text id=\"NO_FILE\"><t>x</t></text>\n"
    "  <unknown attr=\"1\">text</unknown>\n"
    "  <window number=\"af02\"/>\n"
    "</dlc>\n";

static uint32_t seed = 0x2545f491u;
static uint32_t rnd(void) {
    seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5;
    return seed;
}

static uint8_t *file_at(const char *dir, const char *name, size_t *n) {
    char p[2048];
    uint64_t size = 0;
    snprintf(p, sizeof p, "%s/%s", dir, name);
    uint8_t *d = plat_file_read(p, &size);
    *n = (size_t)size;
    return d;
}

// AC_DLC_PROBE_DIR: the game's archives of that folder with its DLC folders, into its out/.
static int build_folder(const char *dir) {
    char sub[2048];
    snprintf(sub, sizeof sub, "%s/dlc", dir);
    dlc_probe_scan(sub);
    snprintf(sub, sizeof sub, "%s/out", dir);
    plat_file_mkdir_p(sub);
    static const char *const rels[] = {"quest/quest.pkz", "Text/Text.pkz"};
    for (int i = 0; i < 2; i++) {
        const char *name = strchr(rels[i], '/') + 1;
        size_t n, out_n = 0;
        uint8_t *in = file_at(dir, name, &n);
        void *out = in && dlc_changes(rels[i]) ? dlc_build(rels[i], in, n, &out_n) : NULL;
        snprintf(sub, sizeof sub, "%s/out/%s", dir, name);
        if (!out || !plat_file_write(sub, out, out_n)) {
            fprintf(stderr, "probe-dlc: %s of %s was not built\n", name, dir);
            return 1;
        }
        printf("probe-dlc: %s: %zu -> %zu bytes\n", sub, n, out_n);
        free(out);
        free(in);
    }
    return 0;
}

int probe_dlc(void) {
    const char *given = cfg_get("AC_DLC_PROBE_DIR");
    if (given && *given) return build_folder(given);

    // off unless asked: the setting's default, and a scan that reads nothing
    const CfgMeta *meta = cfg_meta("AC_DLC");
    CHECK(meta && !strcmp(meta->def, "0") && !meta->live, "the setting dlc is off by default and read at start");
    if (!cfg_bool("AC_DLC")) {
        dlc_scan();
        CHECK(!dlc_started_on() && !dlc_name(0), "dlc = 0: no DLC folder is read");
        CHECK(!dlc_changes("quest/quest.pkz") && !dlc_changes("Text/Text.pkz"), "dlc = 0: no archive is changed");
        uint32_t v, q;
        const char *id;
        CHECK(!dlc_training(0, &v, &id, &q) && !dlc_window(0, &v) && !dlc_live_text(0, &id),
              "dlc = 0: no training row, window or text");
        printf("probe-dlc: dlc = 0 here: checked that nothing is read\n");
    } else {
        printf("probe-dlc: dlc = 1 here (astralchain.ini or AC_DLC): the three checks of dlc = 0 did not run\n");
    }

    char root[1024], dir[1100];
    if (!plat_exe_path(root, sizeof root)) return 1;
    char *cut = strrchr(root, '/'), *back = strrchr(root, '\\');
    if (back > cut) cut = back;
    if (!cut) return 1;
    strcpy(cut + 1, "dlc-probe");
    remove_tree(root);
    snprintf(dir, sizeof dir, "%s/dlc", root);
    put(dir, "a/dlc.xml", MANIFEST_A);
    put(dir, "a/quest/c0f1/QuestData.xml", QUEST_XML);
    put(dir, "a/quest/c0f1/extra.csv", "as it is\r\n");
    put(dir, "a/quest/c0f2/QuestData.xml", "<QuestRoot><Version>16</Version></QuestRoot>");
    put(dir, "b-broken/dlc.xml", "<dlc><quest id=\"c0f5\"></dlc>");
    put(dir, "c-nomanifest/quest/c0f6/QuestData.xml", "<QuestRoot/>");
    put(dir, "d-odd/dlc.xml", MANIFEST_ODD);
    put(dir, "e-notdlc/dlc.xml", "<mod/>");
    put(dir, "README.md", "a file next to the DLC folders");

    // the game's side: two quests and five text files
    // one node "Q": 1 node, 1 row, 2 string bytes; the node (no child, first child = the node count, no attribute,
    // row 0); the row (name at 0, no value); "Q" and its 0
    static const uint8_t base_bxm[] = "BXM\0" "\0\0\0\0" "\0\1\0\1" "\0\0\0\2" "\0\0\0\1\0\0\0\0" "\0\0\xff\xff" "Q";
    static const char talk[] = "XML\0the base's talk file", ex[] = "Training\r\n";
    const char *dat_names[] = {"TalkScript_c00b.bxm", "QuestData.bxm", "ExData.csv", "Empty.csv"};
    const void *dat_data[] = {talk, base_bxm, ex, ""};
    size_t dat_sizes[] = {sizeof talk - 1, sizeof base_bxm, sizeof ex - 1, 0}, base_n, other_n, quest_n;
    CHECK(!strcmp(bxm_text(base_bxm, sizeof base_bxm, "BXM"), "Q="), "the probe's BXM reader reads a file made by hand");
    void *base = dlc_probe_dat(dat_names, dat_data, dat_sizes, 4, &base_n);
    void *other = dlc_probe_dat(dat_names + 1, dat_data + 1, dat_sizes + 1, 1, &other_n);
    const char *q_names[] = {"questc00b.dat", "quest0100.dat"};
    const void *q_data[] = {base, other};
    size_t q_sizes[] = {base_n, other_n};
    uint8_t *quest_pkz = dlc_probe_pkz(q_names, q_data, q_sizes, 2, &quest_n);

    const char *ids[] = {"OLD_TEXT", "KEPT_TEXT"}, *texts[] = {"old", "kept"};
    size_t ptd_n, text_n;
    void *ptd = dlc_probe_ptd(ids, texts, 2, &ptd_n);
    const char *t_names[] = {"Core_text_USen.bin", "Core_text_EUfr.bin", "menu_text_USen.bin", "Other_USen.bin",
                             "Core_text_JPja.bin"};
    const void *t_data[] = {ptd, ptd, ptd, ptd, "not a text file"};
    size_t t_sizes[] = {ptd_n, ptd_n, ptd_n, ptd_n, 15};
    uint8_t *text_pkz = dlc_probe_pkz(t_names, t_data, t_sizes, 5, &text_n);

    Entry e[16], in[16];
    CHECK(pkz_entries(quest_pkz, quest_n, in, 16) == 2 && !strcmp(in[0].name, "quest0100.dat"),
          "the probe's quest archive reads back, sorted");
    size_t size;
    uint32_t count;
    const uint8_t *f = dat_file(base, base_n, "TalkScript_c00b.bxm", &size, &count);
    CHECK(f && count == 4 && size == sizeof talk - 1 && !memcmp(f, talk, size), "the probe's DAT reads back");

    // the scan: a and d-odd load, the three others are left out
    dlc_probe_scan(dir);
    CHECK(dlc_name(0) && !strcmp(dlc_name(0), "a") && dlc_name(1) && !strcmp(dlc_name(1), "d-odd") && !dlc_name(2),
          "the DLCs with a manifest that parses load; a broken one, one without and one of another root do not");
    uint32_t value, quest, number;
    const char *id;
    CHECK(dlc_training(0, &value, &id, &quest) && value == 2 && quest == 0xc0f1 && !strcmp(id, "PROBE_MENU") &&
          !dlc_training(1, &value, &id, &quest), "one training row: the rows without a quest or a text are left out");
    CHECK(dlc_window(0, &number) && number == 0xaf01 && dlc_window(1, &number) && number == 0xaf02 &&
          !dlc_window(2, &number), "two windows: a number above four hex digits is left out");
    CHECK(dlc_live_text(0, &id) && !strcmp(id, "PROBE_ONE") && !dlc_live_text(1, &id), "one text with input fields");
    CHECK(dlc_changes("quest/quest.pkz") && dlc_changes("Text/Text.pkz") && !dlc_changes("ui/ui.pkz"),
          "the DLCs change the quest and text archives only");
    uint8_t key[32];
    memcpy(key, dlc_key(), 32);

    // the quests
    size_t out_n = 0, again_n = 0;
    uint8_t *out = dlc_build("quest/quest.pkz", quest_pkz, quest_n, &out_n);
    int n = out ? pkz_entries(out, out_n, e, 16) : -1;
    CHECK(n == 4, "the quest archive holds the game's two quests and the two a DLC can build (not c0f3, whose base is "
                  "missing, nor c0f4, which has no file)");
    if (n == 4) {
        CHECK(!strcmp(e[0].name, "quest0100.dat") && !strcmp(e[1].name, "questc00b.dat") &&
              !strcmp(e[2].name, "questc0f1.dat") && !strcmp(e[3].name, "questc0f2.dat"), "the entries are sorted");
        for (int i = 0; i < 2; i++) {
            const Entry *was = entry_named(in, 2, e[i].name);
            CHECK(was && was->size == e[i].size && was->comp == e[i].comp && !memcmp(was->frame, e[i].frame, (size_t)was->comp),
                  "a quest of the game keeps its compressed bytes");
        }
        uint8_t *dat = unpack(&e[2]);
        size_t dn = (size_t)e[2].size;
        CHECK(dat != NULL, "the added quest unpacks");
        if (dat) {
            f = dat_file(dat, dn, "QuestData.bxm", &size, &count);
            CHECK(count == 5, "the quest with a base holds the base's four files and the folder's new one");
            CHECK(f && !strcmp(bxm_text(f, size, "BXM"), QUEST_TREE),
                  "QuestData.xml became the BXM of its tree, with the magic of the base's file");
            f = dat_file(dat, dn, "TalkScript_c0f1.bxm", &size, NULL);
            CHECK(f && size == sizeof talk - 1 && !memcmp(f, talk, size) && !dat_file(dat, dn, "TalkScript_c00b.bxm", &size, NULL),
                  "the base's file is there under the new quest's id");
            f = dat_file(dat, dn, "ExData.csv", &size, NULL);
            CHECK(f && size == sizeof ex - 1 && !memcmp(f, ex, size), "a file of the base the folder does not give is kept");
            f = dat_file(dat, dn, "extra.csv", &size, NULL);
            CHECK(f && size == 10 && !memcmp(f, "as it is\r\n", 10), "a file that is not XML is taken as it is");
            f = dat_file(dat, dn, "Empty.csv", &size, NULL);
            CHECK(f && !size, "an empty file of the base stays empty");
            free(dat);
        }
        dat = unpack(&e[3]);
        if (dat) {
            f = dat_file(dat, (size_t)e[3].size, "QuestData.bxm", &size, &count);
            CHECK(f && count == 1 && !strcmp(bxm_text(f, size, "XML"), "QuestRoot={Version=16}"),
                  "a quest without a base holds its folder's files");
            free(dat);
        }
    }
    uint8_t *again = dlc_build("quest/quest.pkz", quest_pkz, quest_n, &again_n);
    CHECK(out && again && again_n == out_n && !memcmp(out, again, out_n), "the same sources build the same bytes");
    free(again);
    free(out);

    // the texts
    out = dlc_build("Text/Text.pkz", text_pkz, text_n, &out_n);
    int tn = pkz_entries(text_pkz, text_n, in, 16);
    n = out ? pkz_entries(out, out_n, e, 16) : -1;
    CHECK(n == 5 && tn == 5, "the text archive keeps its five files");
    if (n == 5 && tn == 5) {
        static const uint16_t one[] = {'o', 'n', 'e', ' ', '{', 'Z', 'L', '}', ' ', '<', 0xe9, '>', ' ', 0xd83d, 0xde00, 0},
                              fr[] = {'u', 'n', '\r', '\n', 'l', 'i', 'g', 'n', 'e', 0}, kept[] = {'k', 'e', 'p', 't', 0},
                              replaced[] = {'r', 'e', 'p', 'l', 'a', 'c', 'e', 'd', 0}, menu[] = {'P', 'r', 'o', 'b', 'e', 0},
                              old[] = {'o', 'l', 'd', 0};
        const Entry *x = entry_named(e, n, "Core_text_USen.bin");
        uint8_t *t = x ? unpack(x) : NULL;
        CHECK(t && text_is(t, (size_t)x->size, "PROBE_ONE", one), "a text without lang is the text of a language that has none");
        CHECK(t && text_is(t, (size_t)x->size, "KEPT_TEXT", kept), "a text of the game stays");
        CHECK(t && text_is(t, (size_t)x->size, "OLD_TEXT", replaced), "a text of the game's id is replaced, once");
        free(t);
        x = entry_named(e, n, "Core_text_EUfr.bin");
        t = x ? unpack(x) : NULL;
        CHECK(t && text_is(t, (size_t)x->size, "PROBE_ONE", fr), "a language takes its own text, \\n a line break");
        free(t);
        x = entry_named(e, n, "menu_text_USen.bin");
        t = x ? unpack(x) : NULL;
        uint16_t none[8];
        CHECK(t && text_is(t, (size_t)x->size, "PROBE_MENU", menu) && text_is(t, (size_t)x->size, "OLD_TEXT", old) &&
              ptd_text(t, (size_t)x->size, "PROBE_ONE", none, 8) < 0, "a text goes to its own file only");
        free(t);
        for (int i = 0; i < 2; i++) {
            const char *name = i ? "Core_text_JPja.bin" : "Other_USen.bin";
            const Entry *was = entry_named(in, tn, name), *now = entry_named(e, n, name);
            CHECK(was && now && was->comp == now->comp && !memcmp(was->frame, now->frame, (size_t)was->comp),
                  i ? "a file that is not a text file keeps its bytes" : "a text file no DLC names keeps its bytes");
        }
    }
    free(out);

    // the key: the same for the same sources, another for another source
    dlc_probe_scan(dir);
    CHECK(!memcmp(key, dlc_key(), 32), "the same sources give the same key");
    put(dir, "a/quest/c0f1/extra.csv", "as it was\r\n");
    dlc_probe_scan(dir);
    CHECK(memcmp(key, dlc_key(), 32) != 0, "a changed quest file changes the key");
    put(dir, "a/quest/c0f1/extra.csv", "as it is\r\n");
    dlc_probe_scan(dir);
    CHECK(!memcmp(key, dlc_key(), 32), "the file put back gives the first key again");

    // archives cut short or changed: an answer, never a stop
    uint8_t *bad = malloc(quest_n > text_n ? quest_n : text_n);
    int answered = 0;
    for (int i = 0; bad && i < 2000; i++) {
        int is_text = i & 1;
        const uint8_t *src = is_text ? text_pkz : quest_pkz;
        size_t len = is_text ? text_n : quest_n, head = 0x20 + 32 * (is_text ? 5 : 2) + 96;
        memcpy(bad, src, len);
        if (i % 5 == 0) len = rnd() % (len + 1);
        for (uint32_t k = rnd() % 4 + (i % 5 != 0); k; k--) bad[rnd() % (i % 3 ? head : len ? len : 1)] ^= (uint8_t)(1u << rnd() % 8);
        size_t bn;
        free(dlc_build(is_text ? "Text/Text.pkz" : "quest/quest.pkz", bad, len, &bn));
        answered++;
    }
    free(bad);
    CHECK(answered == 2000, "2 000 archives cut short or with bits changed are answered");

    // manifests cut short or changed
    char xml[sizeof MANIFEST_A];
    int read = 0, loaded = 0, odd_missing = 0;
    for (int i = 0; i < 300; i++) {
        memcpy(xml, MANIFEST_A, sizeof xml);
        size_t len = sizeof xml - 1;
        if (i % 4 == 0) xml[rnd() % len] = 0;
        for (uint32_t k = rnd() % 3 + 1; k; k--) xml[rnd() % len] = "<>/\"'=&; \n\\{x"[rnd() % 14];
        put(dir, "a/dlc.xml", xml);
        dlc_probe_scan(dir);
        size_t bn;
        if (dlc_changes("quest/quest.pkz")) free(dlc_build("quest/quest.pkz", quest_pkz, quest_n, &bn));
        if (dlc_changes("Text/Text.pkz")) free(dlc_build("Text/Text.pkz", text_pkz, text_n, &bn));
        read++;
        int a_loaded = dlc_name(0) && !strcmp(dlc_name(0), "a");
        loaded += a_loaded;
        const char *odd = dlc_name(a_loaded);   // d-odd, which the loop does not change, comes after a when a loads
        odd_missing += !odd || strcmp(odd, "d-odd");
    }
    CHECK(read == 300, "300 manifests cut short or with characters changed are read");
    CHECK(loaded > 0 && loaded < 300, "some of the changed manifests still parse and some do not");
    CHECK(!odd_missing, "the DLC next to a changed one loads whatever the changed one holds");
    printf("probe-dlc: %d of 300 changed manifests still parse\n", loaded);

    free(quest_pkz), free(text_pkz), free(ptd), free(base), free(other);
    remove_tree(root);
    printf("probe-dlc: %d checks, %d failed\n", checks, fails);
    printf(fails ? "probe-dlc: FAILED\n" : "probe-dlc: ok\n");
    return fails != 0;
}
