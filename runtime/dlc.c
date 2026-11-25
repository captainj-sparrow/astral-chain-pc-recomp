// DLCs of the port: additions to the game written as plain XML, built into the game's archives at start (dlc.h).
//
// A DLC is a folder <dir of astralchain.exe>/dlc/<name>/ with a manifest dlc.xml and its quests' files. It holds no
// file of the game: the port builds the archives the game reads from the player's own RomFS and these sources, and
// romfs.c serves them in place of the game's (dlc_overlay there keeps the result in <profile>/dlc/ and builds again
// only when a source, the builder or the archive under it changed). Nothing of it runs unless the player switched the
// setting dlc on (the Mods tab's "Load DLC", off by default): with dlc = 0 no folder is read, no archive is built or
// served, and the patch sets quest and questmenu (recomp/patches-v101.txt), which follow the same setting, stay off.
//
// dlc.xml:
//   <dlc>
//     <quest id="c0f0" base="c00b"/>
//     <training value="2" text="MENU_TUTO_NAME_DLC_00" quest="c0f0"/>
//     <window number="af00"/>
//     <text file="Core_text" id="CORE_TUTO_W_NAME_af00_01" live="1">
//       <t>{ZL} + {ZR}</t>  <t lang="EUfr">...</t>
//     </text>
//   </dlc>
// quest: the quest <id> (four hex digits) is added to quest/quest.pkz as quest<id>.dat. Its files are those of the
// folder quest/<id>/ next to dlc.xml: <name>.xml becomes <name>.bxm (the game's binary XML, below), any other file is
// taken as it is. With base, the files the folder does not give are the base quest's, read from the player's archive,
// with "_<base>." in their names changed to "_<id>." (TalkScript_c00b.bxm).
// training, window: a row of the training menu and a tutorial window's number (quest.c applies them).
// text: the text <id> is added to Text/Text.pkz, to <file>_<language>.bin of each of the game's languages: the <t>
// without lang, or the one of the language (the game's names: USen, EUfr, JPja...). "\n" in a text is a line break.
// live="1": the text's {ACTION} fields take the bound inputs when the game reads it (quest.c rt_quest_text).
//
// The formats, as the game's files have them (every multi-byte value little endian but BXM's):
//   XML -> BXM. An element is a node: its name, its text (trimmed; the text before its first child) as the value, its
//   attributes, its children. BXM, big endian: "XML\0" or "BXM\0", u32 0, u16 node count, u16 data row count, u32
//   string bytes; per node u16 child count, u16 index of its first child (the node count for none), u16 attribute
//   count, u16 data row; per data row u16 name offset, u16 value offset into the strings (0xffff for ""); the strings
//   (UTF-8, each with its 0). The nodes are breadth first, a node's children together; a data row per distinct (name,
//   value, attributes) with its attributes' rows after it; a string per distinct text in the order first met.
//   DAT: "DAT\0", count, offsets of the file offsets, the extensions (4 bytes each), the names (u32 width, then names
//   of that width), the sizes, the hash table, 0; then those tables and the files' data, each at a multiple of 16.
//   The hash table: the shift, 0x10, the hashes' offset, the indexes' offset; u16 per bucket (the first slot of a
//   hash whose top bits are the bucket, 0xffff for none), the hashes (dlc_hash of the names) and u16 file indexes,
//   both sorted by bucket. The shift is 32 - log2 of the next power of two of the count, at most 31.
//   PKZ: "pkzl", version, u64 size, count, 0x20, u64 string bytes; per entry u32 name offset, u32 compression name
//   offset, u64 size, u64 data offset, u64 compressed size; the strings ("ZStandard", then the names in the entries'
//   order, each padded to 8 bytes with at least one 0); each entry's zstd frame at a multiple of 64. The game skips an
//   entry whose compression name is neither "None" nor "ZStandard" (main+0x9a6400). The entries are sorted by name
//   without case.
//   PTD (a text file): "PTD\0", 2, the key 0x26, id count, 0x1c, section count, the sections' offset; the id rows
//   (16 bytes sorted by hash: hash, string offset from the row, characters, bytes) and their strings; per section
//   (20 bytes) hash, list count, lists' offset, column count, columns' offset (from the section); a list is hash,
//   count, offset to that many u32 (the ids' hashes); a column is hash, row count, offset to its rows (as the id
//   rows) and strings. Strings are UTF-16 with 0x26 added to every byte, 0 included; a column starts at a multiple
//   of 2. A text is its id in the id rows, a row of each column of the first section (the first holds the text,
//   the others "") and its hash at the end of each of that section's lists.
// Checked against the game's files: every archive, DAT and text file of 1.0.1 written back gives the same bytes with
// the tools these writers were ported from; the 1 669 BXM files give the same trees (507 the same bytes: the game's
// writer shares strings in another order in the others, which its reader does not depend on).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include <zstd.h>

#include "cfg.h"
#include "dlc.h"
#include "host.h"
#include "platform/platform.h"
#include "sha256.h"

#define DLC_BUILDER "dlc builder 1"   // part of dlc_key: changed when the builder writes other bytes for a source
enum { MAX_DLC = 32, MAX_QUESTS = 32, MAX_FILES = 64, MAX_TEXTS = 512, MAX_LANGS = 16, MAX_TRAININGS = 4,
       MAX_WINDOWS = 16, MAX_LIVE = 4, PTD_KEY = 0x26, ZSTD_LEVEL = 6 };

static void need(int c, const char *what) {
    if (!c) rt_fatal("dlc: out of memory (%s)", what);
}

// ---- memory: strings and nodes that live as long as the manifests (never freed) or one build (arena_free)
typedef struct Block { struct Block *next; size_t used, cap; } Block;
typedef struct { Block *head; } Arena;
static void *arena_get(Arena *a, size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (!a->head || a->head->used + n > a->head->cap) {
        size_t cap = n > 65536 ? n : 65536;
        Block *b = malloc(sizeof *b + cap);
        need(b != NULL, "arena");
        *b = (Block){a->head, 0, cap};
        a->head = b;
    }
    void *p = (uint8_t *)(a->head + 1) + a->head->used;
    a->head->used += n;
    return p;
}
static char *arena_str(Arena *a, const char *s, size_t n) {
    char *p = arena_get(a, n + 1);
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}
static void arena_free(Arena *a) {
    for (Block *b = a->head, *next; b; b = next) {
        next = b->next;
        free(b);
    }
    a->head = NULL;
}

typedef struct { uint8_t *p; size_t n, cap; } Buf;
static void buf_room(Buf *b, size_t n) {
    if (b->n + n <= b->cap) return;
    while (b->cap < b->n + n) b->cap = b->cap ? b->cap * 2 : 4096;
    b->p = realloc(b->p, b->cap);
    need(b->p != NULL, "buffer");
}
static void buf_put(Buf *b, const void *d, size_t n) {   // d NULL: zeros
    buf_room(b, n);
    if (d) memcpy(b->p + b->n, d, n);
    else memset(b->p + b->n, 0, n);
    b->n += n;
}
static void buf_align(Buf *b, size_t a) { buf_put(b, NULL, (a - b->n % a) % a); }
static void le16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v, p[1] = (uint8_t)(v >> 8); }
static void le32(uint8_t *p, uint32_t v) { le16(p, v), le16(p + 2, v >> 16); }
static void le64(uint8_t *p, uint64_t v) { le32(p, (uint32_t)v), le32(p + 4, (uint32_t)(v >> 32)); }
static void be16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 8), p[1] = (uint8_t)v; }
static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t *p) { return rd32(p) | (uint64_t)rd32(p + 4) << 32; }
static void put32(Buf *b, uint32_t v) {
    uint8_t t[4];
    le32(t, v);
    buf_put(b, t, 4);
}

uint32_t dlc_hash(const char *name) {
    uint32_t crc = ~0u;
    for (; *name; name++) {
        crc ^= (uint8_t)(*name >= 'A' && *name <= 'Z' ? *name - 'A' + 'a' : *name);
        for (int k = 0; k < 8; k++) crc = crc >> 1 ^ (0xEDB88320u & -(crc & 1));
    }
    return ~crc & 0x7fffffffu;
}

// ---- XML
typedef struct XNode {
    const char *name, *value;
    const char **attr;   // nattr pairs: name, value
    struct XNode **kid;
    int nattr, nkid;
} XNode;
typedef struct { const char *s, *file; Arena *a; char err[160]; } XParse;

static int x_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
static int x_name(char c) { return c && !x_space(c) && c != '/' && c != '>' && c != '=' && c != '<'; }

// s[0..max) as printable ASCII into out (max + 1 bytes): what the log shows of a name or text read from a file that
// may be damaged or in another encoding. Any other byte is shown as '?', so the log stays valid UTF-8 whatever the
// input held.
static const char *printable(const char *s, size_t max, char *out) {
    size_t i = 0;
    for (; s[i] && i < max; i++) out[i] = (uint8_t)s[i] >= 0x20 && (uint8_t)s[i] < 0x7f ? s[i] : '?';
    out[i] = 0;
    return out;
}

static XNode *x_fail(XParse *p, const char *what) {
    char near[25];
    if (!p->err[0]) snprintf(p->err, sizeof p->err, "%s near \"%s\"", what, printable(p->s, 24, near));
    return NULL;
}

// s[0..n) with the character references and the five named entities resolved, in the arena.
static const char *x_text(XParse *p, const char *s, size_t n) {
    char *out = arena_get(p->a, n + 1), *o = out;
    for (size_t i = 0; i < n;) {
        const char *semi = s[i] == '&' ? memchr(s + i, ';', n - i) : NULL;
        size_t len = semi ? (size_t)(semi - (s + i)) + 1 : 0;
        uint32_t ch = 0;
        if (len >= 4 && s[i + 1] == '#') {
            ch = (uint32_t)strtoul(s + i + (s[i + 2] == 'x' ? 3 : 2), NULL, s[i + 2] == 'x' ? 16 : 10);
        } else if (len) {
            static const struct { const char *name; char ch; } ents[] = {
                {"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'}, {"&quot;", '"'}, {"&apos;", '\''}};
            for (size_t k = 0; k < sizeof ents / sizeof *ents; k++)
                if (len == strlen(ents[k].name) && !memcmp(s + i, ents[k].name, len)) ch = (uint8_t)ents[k].ch;
        }
        if (!ch) {
            *o++ = s[i++];
            continue;
        }
        if (ch < 0x80) *o++ = (char)ch;
        else if (ch < 0x800) *o++ = (char)(0xC0 | ch >> 6), *o++ = (char)(0x80 | (ch & 0x3F));
        else if (ch < 0x10000)
            *o++ = (char)(0xE0 | ch >> 12), *o++ = (char)(0x80 | (ch >> 6 & 0x3F)), *o++ = (char)(0x80 | (ch & 0x3F));
        else   // a reference is at least 8 characters here, the 4 bytes fit
            *o++ = (char)(0xF0 | ch >> 18), *o++ = (char)(0x80 | (ch >> 12 & 0x3F)),
            *o++ = (char)(0x80 | (ch >> 6 & 0x3F)), *o++ = (char)(0x80 | (ch & 0x3F));
        i += len;
    }
    *o = 0;
    return out;
}

// Past white space, comments, processing instructions and a DOCTYPE.
static void x_skip(XParse *p) {
    for (;;) {
        while (x_space(*p->s)) p->s++;
        const char *end = !strncmp(p->s, "<!--", 4) ? strstr(p->s, "-->") : !strncmp(p->s, "<?", 2) ? strstr(p->s, "?>")
                        : !strncmp(p->s, "<!", 2) ? strchr(p->s, '>') : NULL;
        if (!end) return;
        p->s = end + (*end == '>' ? 1 : end[1] == '>' ? 2 : 3);
    }
}

static void x_push(XParse *p, void ***list, int *n, void *item) {
    if (!(*n & (*n + 1))) {   // 0, 1, 3, 7...: the list is full
        void **grown = arena_get(p->a, (size_t)(*n * 2 + 2) * sizeof *grown);
        if (*n) memcpy(grown, *list, (size_t)*n * sizeof *grown);
        *list = grown;
    }
    (*list)[(*n)++] = item;
}

static XNode *x_elem(XParse *p, int depth) {
    if (*p->s != '<' || depth > 64) return x_fail(p, depth > 64 ? "elements nested too deep" : "'<' expected");
    const char *s = ++p->s;
    while (x_name(*p->s)) p->s++;
    if (p->s == s) return x_fail(p, "an element name expected");
    XNode *n = arena_get(p->a, sizeof *n);
    *n = (XNode){.name = arena_str(p->a, s, (size_t)(p->s - s)), .value = ""};
    int nwords = 0;
    for (;;) {
        while (x_space(*p->s)) p->s++;
        if (*p->s == '/' || *p->s == '>') break;
        s = p->s;
        while (x_name(*p->s)) p->s++;
        const char *key = arena_str(p->a, s, (size_t)(p->s - s));
        while (x_space(*p->s)) p->s++;
        if (p->s == s || *p->s != '=') return x_fail(p, "an attribute and '=' expected");
        do p->s++; while (x_space(*p->s));
        char quote = *p->s;
        const char *end = quote == '"' || quote == '\'' ? strchr(p->s + 1, quote) : NULL;
        if (!end) return x_fail(p, "a quoted attribute value expected");
        x_push(p, (void ***)&n->attr, &nwords, (void *)key);
        x_push(p, (void ***)&n->attr, &nwords, (void *)x_text(p, p->s + 1, (size_t)(end - p->s - 1)));
        p->s = end + 1;
    }
    n->nattr = nwords / 2;
    if (*p->s == '/') {
        if (p->s[1] != '>') return x_fail(p, "'/>' expected");
        p->s += 2;
        return n;
    }
    p->s++;
    for (;;) {
        s = p->s;
        while (*p->s && *p->s != '<') p->s++;
        if (!n->nkid && !n->value[0]) {   // the text before the first child, trimmed
            const char *e = p->s;
            while (s < e && x_space(*s)) s++;
            while (e > s && x_space(e[-1])) e--;
            if (e > s) n->value = x_text(p, s, (size_t)(e - s));
        }
        x_skip(p);
        if (!*p->s) return x_fail(p, "the file ends inside an element");
        if (*p->s != '<') continue;   // text after a comment
        if (p->s[1] == '/') {
            size_t len = strlen(n->name);
            if (strncmp(p->s + 2, n->name, len)) return x_fail(p, "another element's closing tag");
            p->s += 2 + len;
            while (x_space(*p->s)) p->s++;
            if (*p->s != '>') return x_fail(p, "'>' expected");
            p->s++;
            return n;
        }
        XNode *kid = x_elem(p, depth + 1);
        if (!kid) return NULL;
        x_push(p, (void ***)&n->kid, &n->nkid, kid);
    }
}

// The root element of the XML text (0 terminated), NULL with p->err set.
static XNode *x_parse(XParse *p) {
    if (!strncmp(p->s, "\xEF\xBB\xBF", 3)) p->s += 3;
    x_skip(p);
    return x_elem(p, 0);
}

static const char *x_attr(const XNode *n, const char *name) {
    for (int i = 0; i < n->nattr; i++)
        if (!strcmp(n->attr[2 * i], name)) return n->attr[2 * i + 1];
    return NULL;
}

// ---- BXM
static int bxm_same(const XNode *a, const XNode *b) {
    if (strcmp(a->name, b->name) || strcmp(a->value, b->value) || a->nattr != b->nattr) return 0;
    for (int i = 0; i < 2 * a->nattr; i++)
        if (strcmp(a->attr[i], b->attr[i])) return 0;
    return 1;
}

// The BXM of a tree; 0 when it has more nodes, rows or string bytes than the format's 16-bit fields hold.
static int bxm_write(const XNode *root, const char magic[4], Buf *out, Arena *a) {
    enum { LIMIT = 0xfffe };
    const XNode **order = arena_get(a, (LIMIT + 1) * sizeof *order);
    uint16_t *first = arena_get(a, (LIMIT + 1) * sizeof *first), *row = arena_get(a, (LIMIT + 1) * sizeof *row);
    size_t count = 1;
    order[0] = root;
    for (size_t i = 0; i < count; i++) {
        first[i] = (uint16_t)count;
        if (count + (size_t)order[i]->nkid > LIMIT) return 0;
        for (int k = 0; k < order[i]->nkid; k++) order[count++] = order[i]->kid[k];
    }
    // data rows: per distinct node its (name, value), then its attributes'
    const char **cell = arena_get(a, 2 * (LIMIT + 1) * sizeof *cell);
    const XNode **keyed = arena_get(a, (LIMIT + 1) * sizeof *keyed);
    uint16_t *keyed_row = arena_get(a, (LIMIT + 1) * sizeof *keyed_row);
    size_t rows = 0, nkeyed = 0;
    for (size_t i = 0; i < count; i++) {
        size_t k = 0;
        while (k < nkeyed && !bxm_same(keyed[k], order[i])) k++;
        if (k == nkeyed) {
            if (rows + 1 + (size_t)order[i]->nattr > LIMIT) return 0;
            keyed[nkeyed] = order[i], keyed_row[nkeyed++] = (uint16_t)rows;
            cell[2 * rows] = order[i]->name, cell[2 * rows + 1] = order[i]->value, rows++;
            for (int t = 0; t < order[i]->nattr; t++, rows++)
                cell[2 * rows] = order[i]->attr[2 * t], cell[2 * rows + 1] = order[i]->attr[2 * t + 1];
        }
        row[i] = keyed_row[k];
    }
    // strings: one per distinct text, in the order of the rows' cells
    Buf strings = {0};
    uint16_t *ref = arena_get(a, 2 * (LIMIT + 1) * sizeof *ref);
    for (size_t c = 0; c < 2 * rows; c++) {
        if (!cell[c][0]) {
            ref[c] = 0xffff;
            continue;
        }
        size_t prev = 0;
        while (prev < c && strcmp(cell[prev], cell[c])) prev++;
        if (prev < c) {
            ref[c] = ref[prev];
            continue;
        }
        if (strings.n + strlen(cell[c]) + 1 > LIMIT) {
            free(strings.p);
            return 0;
        }
        ref[c] = (uint16_t)strings.n;
        buf_put(&strings, cell[c], strlen(cell[c]) + 1);
    }
    uint8_t head[16] = {0}, w[8];
    memcpy(head, magic, 4);
    be16(head + 8, (uint32_t)count), be16(head + 10, (uint32_t)rows);
    be16(head + 12, (uint32_t)(strings.n >> 16)), be16(head + 14, (uint32_t)strings.n);
    buf_put(out, head, sizeof head);
    for (size_t i = 0; i < count; i++) {
        be16(w, (uint32_t)order[i]->nkid), be16(w + 2, order[i]->nkid ? first[i] : (uint32_t)count);
        be16(w + 4, (uint32_t)order[i]->nattr), be16(w + 6, row[i]);
        buf_put(out, w, 8);
    }
    for (size_t r = 0; r < rows; r++) {
        be16(w, ref[2 * r]), be16(w + 2, ref[2 * r + 1]);
        buf_put(out, w, 4);
    }
    buf_put(out, strings.p, strings.n);
    free(strings.p);
    return 1;
}

// ---- DAT
typedef struct { const char *name; const uint8_t *data; size_t size; } DatFile;

// The files of a DAT into list (at most max); -1 when it is not one.
static int dat_list(const uint8_t *d, size_t n, DatFile *list, int max, Arena *a) {
    if (n < 0x20 || memcmp(d, "DAT", 4)) return -1;
    uint32_t count = rd32(d + 4), offs = rd32(d + 8), names = rd32(d + 16), sizes = rd32(d + 20);
    if (count > (uint32_t)max || (uint64_t)offs + 4ull * count > n || (uint64_t)sizes + 4ull * count > n ||
        (uint64_t)names + 4 > n)
        return -1;
    uint32_t width = rd32(d + names);
    if (!width || (uint64_t)names + 4 + (uint64_t)width * count > n) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const char *s = (const char *)d + names + 4 + (size_t)i * width;
        uint32_t off = rd32(d + offs + 4 * i), size = rd32(d + sizes + 4 * i);
        if ((uint64_t)off + size > n) return -1;
        list[i] = (DatFile){arena_str(a, s, strnlen(s, width)), d + off, size};
    }
    return (int)count;
}

static void dat_build(const DatFile *f, int n, Buf *out) {
    size_t width = 0, start = out->n;
    for (int i = 0; i < n; i++)
        if (strlen(f[i].name) + 1 > width) width = strlen(f[i].name) + 1;
    buf_put(out, NULL, 0x20);
    size_t offs = out->n - start;
    buf_put(out, NULL, 4 * (size_t)n);
    size_t exts = out->n - start;
    for (int i = 0; i < n; i++) {
        const char *dot = strrchr(f[i].name, '.');
        char ext[4] = {0};
        strncpy(ext, dot ? dot + 1 : f[i].name, 4);
        buf_put(out, ext, 4);
    }
    size_t names = out->n - start;
    put32(out, (uint32_t)width);
    for (int i = 0; i < n; i++) {
        buf_put(out, f[i].name, strlen(f[i].name));
        buf_put(out, NULL, width - strlen(f[i].name));
    }
    buf_align(out, 4);
    size_t sizes = out->n - start;
    buf_put(out, NULL, 4 * (size_t)n);
    size_t table = out->n - start;
    // the hash table: slots sorted by bucket, then by file index (an insertion sort: a DAT holds a few files)
    uint32_t p2 = 1, bits = 0, hash[MAX_FILES];
    uint16_t slot[MAX_FILES];
    while (p2 < (uint32_t)n) p2 <<= 1, bits++;
    uint32_t shift = 32 - bits > 31 ? 31 : 32 - bits, buckets = 1u << (31 - shift);
    for (int i = 0; i < n; i++) {
        hash[i] = dlc_hash(f[i].name);
        int k = i;
        for (; k > 0 && hash[slot[k - 1]] >> shift > hash[i] >> shift; k--) slot[k] = slot[k - 1];
        slot[k] = (uint16_t)i;
    }
    put32(out, shift), put32(out, 0x10), put32(out, 0x10 + 2 * buckets), put32(out, 0x10 + 2 * buckets + 4 * (uint32_t)n);
    size_t bucket_at = out->n;
    buf_put(out, NULL, 2 * (size_t)buckets);
    memset(out->p + bucket_at, 0xff, 2 * (size_t)buckets);
    for (int k = n - 1; k >= 0; k--) le16(out->p + bucket_at + 2 * (hash[slot[k]] >> shift), (uint32_t)k);
    for (int k = 0; k < n; k++) put32(out, hash[slot[k]]);
    for (int k = 0; k < n; k++) {
        uint8_t t[2];
        le16(t, slot[k]);
        buf_put(out, t, 2);
    }
    for (int i = 0; i < n; i++) {
        if (!f[i].size) continue;
        buf_align(out, 16);   // the DAT starts the buffer or lies at a multiple of 16 in it (dlc_build: at 0)
        le32(out->p + start + offs + 4 * (size_t)i, (uint32_t)(out->n - start));
        le32(out->p + start + sizes + 4 * (size_t)i, (uint32_t)f[i].size);
        buf_put(out, f[i].data, f[i].size);
    }
    buf_align(out, 16);
    uint8_t *h = out->p + start;
    memcpy(h, "DAT", 4);
    le32(h + 4, (uint32_t)n), le32(h + 8, (uint32_t)offs), le32(h + 12, (uint32_t)exts), le32(h + 16, (uint32_t)names);
    le32(h + 20, (uint32_t)sizes), le32(h + 24, (uint32_t)table);
}

// ---- PKZ
typedef struct { const char *name; uint64_t size; const uint8_t *data; uint64_t comp; } PkzEntry;
typedef struct { uint32_t version; PkzEntry *e; int n, cap; } Pkz;

static int pkz_read(const uint8_t *p, size_t n, Pkz *z) {
    if (n < 0x20 || memcmp(p, "pkzl", 4)) return 0;
    uint32_t count = rd32(p + 16), desc = rd32(p + 20);
    uint64_t strings = rd64(p + 24), names = (uint64_t)desc + 32ull * count;
    if (names + strings > n || !strings) return 0;
    const char *s = (const char *)p + names;
    if (s[strings - 1]) return 0;   // every string below ends inside the table
    *z = (Pkz){rd32(p + 4), malloc(((size_t)count + MAX_QUESTS) * sizeof(PkzEntry)), (int)count, (int)count + MAX_QUESTS};
    need(z->e != NULL, "archive entries");
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *d = p + desc + 32 * (size_t)i;
        uint32_t name = rd32(d), comp_name = rd32(d + 4);
        uint64_t off = rd64(d + 16), comp = rd64(d + 24);
        if (name >= strings || comp_name >= strings || strcmp(s + comp_name, "ZStandard") || off > n || comp > n - off) {
            free(z->e);
            return 0;
        }
        z->e[i] = (PkzEntry){s + name, rd64(d + 8), p + off, comp};
    }
    return 1;
}

// The entry `name` (without case), -1 for none; *at: its place among the sorted entries.
static int pkz_find(const Pkz *z, const char *name, int *at) {
    *at = z->n;
    for (int i = 0; i < z->n; i++) {
        int c = strcasecmp(z->e[i].name, name);
        if (!c) return i;
        if (c > 0 && *at == z->n) *at = i;
    }
    return -1;
}

// Adds or replaces the entry `name`: raw[0..size) compressed here; the frame lives in the arena.
static void pkz_set(Pkz *z, const char *name, const uint8_t *raw, size_t size, Arena *a) {
    size_t cap = ZSTD_compressBound(size);
    uint8_t *frame = arena_get(a, cap);
    size_t comp = ZSTD_compress(frame, cap, raw, size, ZSTD_LEVEL);
    need(!ZSTD_isError(comp), "zstd");
    int at, i = pkz_find(z, name, &at);
    if (i < 0) {
        if (z->n == z->cap) return;   // dlc_build adds at most MAX_QUESTS entries
        memmove(z->e + at + 1, z->e + at, (size_t)(z->n - at) * sizeof *z->e);
        z->n++, i = at;
        name = arena_str(a, name, strlen(name));
    } else {
        name = z->e[i].name;
    }
    z->e[i] = (PkzEntry){name, size, frame, comp};
}

static void pkz_write(const Pkz *z, Buf *out) {
    static const char zstd_name[16] = "ZStandard";
    buf_put(out, NULL, 0x20 + 32 * (size_t)z->n);
    size_t strings = out->n;
    buf_put(out, zstd_name, sizeof zstd_name);
    for (int i = 0; i < z->n; i++) {
        le32(out->p + 0x20 + 32 * (size_t)i, (uint32_t)(out->n - strings));
        size_t len = strlen(z->e[i].name) + 1;
        buf_put(out, z->e[i].name, len);
        buf_put(out, NULL, (8 - len % 8) % 8);
    }
    size_t strings_n = out->n - strings;
    for (int i = 0; i < z->n; i++) {
        buf_align(out, 64);
        uint8_t *d = out->p + 0x20 + 32 * (size_t)i;
        le64(d + 8, z->e[i].size), le64(d + 16, out->n), le64(d + 24, z->e[i].comp);
        buf_put(out, z->e[i].data, (size_t)z->e[i].comp);
    }
    buf_align(out, 64);
    memcpy(out->p, "pkzl", 4);
    le32(out->p + 4, z->version), le64(out->p + 8, out->n), le32(out->p + 16, (uint32_t)z->n);
    le32(out->p + 20, 0x20), le64(out->p + 24, strings_n);
}

// An entry's bytes, malloc'd; NULL when its frame does not give its size.
static uint8_t *pkz_unpack(const PkzEntry *e) {
    if (e->size > (256u << 20)) return NULL;
    uint8_t *raw = malloc(e->size ? (size_t)e->size : 1);
    need(raw != NULL, "archive entry");
    if (ZSTD_decompress(raw, (size_t)e->size, e->data, (size_t)e->comp) != e->size) {
        free(raw);
        return NULL;
    }
    return raw;
}

// ---- PTD
typedef struct { uint32_t hash, chars, bytes; const uint8_t *s; } PtdRow;
typedef struct { uint32_t hash; int n; PtdRow *r; } PtdRows;
typedef struct { uint32_t hash; int n; uint32_t *v; } PtdList;
typedef struct { uint32_t hash; int nlists, ncols; PtdList *list; PtdRows *col; } PtdSection;
typedef struct { PtdRows ids; int nsections; PtdSection *section; } Ptd;

// Rows at d[at..) with `extra` free rows after them for ptd_set.
static int ptd_rows(const uint8_t *d, size_t n, size_t at, uint32_t count, PtdRows *out, int extra, Arena *a) {
    if (at + 16ull * count > n) return 0;
    out->n = (int)count;
    out->r = arena_get(a, ((size_t)count + (size_t)extra) * sizeof *out->r);
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *r = d + at + 16 * (size_t)i;
        uint32_t off = rd32(r + 4), chars = rd32(r + 8), bytes = rd32(r + 12);
        if (at + 16ull * i + off + bytes > n || bytes != 2 * chars) return 0;
        out->r[i] = (PtdRow){rd32(r), chars, bytes, r + off};
    }
    return 1;
}

static int ptd_read(const uint8_t *d, size_t n, Ptd *t, int extra, Arena *a) {
    if (n < 0x1c || memcmp(d, "PTD", 4) || rd32(d + 4) != 2 || rd32(d + 8) != PTD_KEY || rd32(d + 16) != 0x1c) return 0;
    uint32_t nsec = rd32(d + 20), sec_at = rd32(d + 24);
    if (!ptd_rows(d, n, 0x1c, rd32(d + 12), &t->ids, extra, a) || sec_at + 20ull * nsec > n || nsec > 4096) return 0;
    t->nsections = (int)nsec;
    t->section = arena_get(a, (size_t)nsec * sizeof *t->section + 1);
    for (uint32_t k = 0; k < nsec; k++) {
        size_t s = sec_at + 20 * (size_t)k;
        PtdSection *sec = &t->section[k];
        uint32_t nlists = rd32(d + s + 4), lists = rd32(d + s + 8), ncols = rd32(d + s + 12), cols = rd32(d + s + 16);
        if (s + lists + 12ull * nlists > n || s + cols + 12ull * ncols > n || nlists > 4096 || ncols > 4096) return 0;
        *sec = (PtdSection){rd32(d + s), (int)nlists, (int)ncols, arena_get(a, (size_t)nlists * sizeof(PtdList) + 1),
                            arena_get(a, (size_t)ncols * sizeof(PtdRows) + 1)};
        for (uint32_t i = 0; i < nlists; i++) {
            size_t r = s + lists + 12 * (size_t)i;
            uint32_t count = rd32(d + r + 4), off = rd32(d + r + 8);
            if (r + off + 4ull * count > n) return 0;
            sec->list[i] = (PtdList){rd32(d + r), (int)count, arena_get(a, ((size_t)count + (size_t)extra) * 4)};
            for (uint32_t v = 0; v < count; v++) sec->list[i].v[v] = rd32(d + r + off + 4 * (size_t)v);
        }
        for (uint32_t i = 0; i < ncols; i++) {
            size_t r = s + cols + 12 * (size_t)i;
            sec->col[i].hash = rd32(d + r);
            if (!ptd_rows(d, n, r + rd32(d + r + 8), rd32(d + r + 4), &sec->col[i], extra, a)) return 0;
        }
    }
    return 1;
}

// UTF-8 as a PTD string (UTF-16 with its 0, the key added to every byte); "\n" (the two characters) is a line break.
static PtdRow ptd_string(uint32_t hash, const char *s, Arena *a) {
    uint8_t *out = arena_get(a, 4 * strlen(s) + 2), *o = out;
    const uint8_t *p = (const uint8_t *)s;
    while (*p) {
        uint32_t ch = *p++;
        if (ch == '\\' && *p == 'n') {
            p++;
            *o++ = '\r' + PTD_KEY, *o++ = PTD_KEY, *o++ = '\n' + PTD_KEY, *o++ = PTD_KEY;
            continue;
        }
        int more = ch >= 0xF0 ? 3 : ch >= 0xE0 ? 2 : ch >= 0xC0 ? 1 : 0;
        if (more) ch &= 0x3Fu >> more;
        for (; more && (*p & 0xC0) == 0x80; more--) ch = ch << 6 | (*p++ & 0x3F);
        if (more || ch > 0x10FFFF) ch = 0xFFFD;
        uint32_t unit[2] = {ch, 0};
        if (ch > 0xFFFF) unit[0] = 0xD800 + ((ch - 0x10000) >> 10), unit[1] = 0xDC00 + ((ch - 0x10000) & 0x3FF);
        for (int k = 0; k < 2 && (k == 0 || unit[1]); k++)
            *o++ = (uint8_t)(unit[k] + PTD_KEY), *o++ = (uint8_t)((unit[k] >> 8) + PTD_KEY);
    }
    *o++ = PTD_KEY, *o++ = PTD_KEY;
    return (PtdRow){hash, (uint32_t)(o - out) / 2, (uint32_t)(o - out), out};
}

// Row `row` into rows kept sorted by hash; replaces the row of its hash.
static void ptd_row_set(PtdRows *rows, PtdRow row) {
    int at = 0;
    while (at < rows->n && rows->r[at].hash < row.hash) at++;
    if (at == rows->n || rows->r[at].hash != row.hash) {
        memmove(rows->r + at + 1, rows->r + at, (size_t)(rows->n - at) * sizeof *rows->r);
        rows->n++;
    }
    rows->r[at] = row;
}

static void ptd_set(Ptd *t, const char *id, const char *text, Arena *a) {
    uint32_t hash = dlc_hash(id);
    ptd_row_set(&t->ids, ptd_string(hash, id, a));
    if (!t->nsections) return;
    PtdSection *sec = &t->section[0];
    for (int i = 0; i < sec->ncols; i++) ptd_row_set(&sec->col[i], ptd_string(hash, i ? "" : text, a));
    for (int i = 0; i < sec->nlists; i++) {
        PtdList *l = &sec->list[i];
        int k = 0;
        for (int v = 0; v < l->n; v++)
            if (l->v[v] != hash) l->v[k++] = l->v[v];
        l->v[k++] = hash;
        l->n = k;
    }
}

static void ptd_put_rows(Buf *out, const PtdRows *rows) {
    size_t at = out->n;
    buf_put(out, NULL, 16 * (size_t)rows->n);
    for (int i = 0; i < rows->n; i++) {
        size_t r = at + 16 * (size_t)i;
        le32(out->p + r, rows->r[i].hash), le32(out->p + r + 4, (uint32_t)(out->n - r));
        le32(out->p + r + 8, rows->r[i].chars), le32(out->p + r + 12, rows->r[i].bytes);
        buf_put(out, rows->r[i].s, rows->r[i].bytes);
    }
}

static void ptd_write(const Ptd *t, Buf *out) {
    buf_put(out, NULL, 0x1c);
    ptd_put_rows(out, &t->ids);
    buf_align(out, 2);
    size_t sec_at = out->n;
    buf_put(out, NULL, 20 * (size_t)t->nsections);
    for (int k = 0; k < t->nsections; k++) {
        const PtdSection *sec = &t->section[k];
        size_t s = sec_at + 20 * (size_t)k, lists = out->n;
        buf_put(out, NULL, 12 * (size_t)sec->nlists);
        for (int i = 0; i < sec->nlists; i++) {
            size_t r = lists + 12 * (size_t)i;
            le32(out->p + r, sec->list[i].hash), le32(out->p + r + 4, (uint32_t)sec->list[i].n);
            le32(out->p + r + 8, (uint32_t)(out->n - r));
            for (int v = 0; v < sec->list[i].n; v++) put32(out, sec->list[i].v[v]);
        }
        size_t cols = out->n;
        buf_put(out, NULL, 12 * (size_t)sec->ncols);
        for (int i = 0; i < sec->ncols; i++) {
            buf_align(out, 2);
            size_t r = cols + 12 * (size_t)i;
            le32(out->p + r, sec->col[i].hash), le32(out->p + r + 4, (uint32_t)sec->col[i].n);
            le32(out->p + r + 8, (uint32_t)(out->n - r));
            ptd_put_rows(out, &sec->col[i]);
        }
        le32(out->p + s, sec->hash), le32(out->p + s + 4, (uint32_t)sec->nlists);
        le32(out->p + s + 8, (uint32_t)(lists - s)), le32(out->p + s + 12, (uint32_t)sec->ncols);
        le32(out->p + s + 16, (uint32_t)(cols - s));
    }
    memcpy(out->p, "PTD", 4);
    le32(out->p + 4, 2), le32(out->p + 8, PTD_KEY), le32(out->p + 12, (uint32_t)t->ids.n), le32(out->p + 16, 0x1c);
    le32(out->p + 20, (uint32_t)t->nsections), le32(out->p + 24, (uint32_t)sec_at);
}

// ---- the manifests
typedef struct { const char *dlc, *dir; uint32_t id, base; int has_base; } Quest;
typedef struct { const char *lang, *text; } Lang;
typedef struct { const char *file, *id; Lang t[MAX_LANGS]; int nt; } Text;
typedef struct { uint32_t value, quest; const char *text; } Training;

static Arena kept;   // the manifests' strings
static Quest quests[MAX_QUESTS];
static Text *texts;
static Training trainings[MAX_TRAININGS];
static uint32_t windows[MAX_WINDOWS];
static const char *live[MAX_LIVE];
static const char *loaded[MAX_DLC];
static int nquests, ntexts, ntrainings, nwindows, nlive, nloaded, started_on;
static uint8_t key[32];
static Sha256 key_hash;

// A file as a 0-terminated string, malloc'd; its bytes go into the key.
static char *source(const char *path, const char *name, uint64_t *size) {
    uint8_t *d = plat_file_read(path, size);
    if (!d) return NULL;
    char *s = realloc(d, (size_t)*size + 1);
    need(s != NULL, "source file");
    s[*size] = 0;
    sha256_update(&key_hash, name, strlen(name) + 1);
    sha256_update(&key_hash, size, sizeof *size);
    sha256_update(&key_hash, s, (size_t)*size);
    return s;
}

static int hex4(const char *s, uint32_t *out) {
    char *end;
    unsigned long v = s && *s ? strtoul(s, &end, 16) : 0x10000;
    if (v > 0xffff || *end) return 0;
    *out = (uint32_t)v;
    return 1;
}

typedef struct { char name[MAX_FILES][96]; int n; } Names;
static int names_cb(const char *name, int kind, uint64_t size, void *ctx) {
    (void)size;
    Names *l = ctx;
    if (kind != PLAT_FS_FILE && kind != PLAT_FS_DIR) return 1;
    if (l->n == MAX_FILES || strlen(name) >= sizeof l->name[0]) return 1;
    int at = l->n++;
    for (; at > 0 && strcmp(l->name[at - 1], name) > 0; at--) memcpy(l->name[at], l->name[at - 1], sizeof l->name[0]);
    snprintf(l->name[at], sizeof l->name[0], "%s", name);
    return 1;
}
// The entries of a folder, sorted (the key must not depend on the file system's order).
static void names_of(const char *dir, Names *l) {
    l->n = 0;
    plat_fs_list(dir, names_cb, l);
}

static void manifest(const char *dlc, const char *dir) {
    char path[2300];
    uint64_t size;
    snprintf(path, sizeof path, "%s/dlc.xml", dir);
    char *xml = source(path, dlc, &size);
    if (!xml) {
        rt_log("dlc: %s: no dlc.xml; left out", dlc);
        return;
    }
    XParse p = {.s = xml, .a = &kept};
    XNode *root = x_parse(&p);
    if (!root || strcmp(root->name, "dlc")) {
        rt_log("dlc: %s: dlc.xml: %s; left out", dlc, root ? "its root element is not <dlc>" : p.err);
        free(xml);
        return;
    }
    int q0 = nquests, t0 = ntexts, r0 = ntrainings, w0 = nwindows, l0 = nlive;
    for (int i = 0; i < root->nkid; i++) {
        const XNode *n = root->kid[i];
        uint32_t v;
        if (!strcmp(n->name, "quest") && nquests < MAX_QUESTS && hex4(x_attr(n, "id"), &v)) {
            Quest *q = &quests[nquests];
            *q = (Quest){.dlc = arena_str(&kept, dlc, strlen(dlc)), .id = v};
            q->has_base = hex4(x_attr(n, "base"), &q->base);
            snprintf(path, sizeof path, "%s/quest/%04x", dir, v);
            q->dir = arena_str(&kept, path, strlen(path));
            Names files;
            names_of(q->dir, &files);
            for (int k = 0; k < files.n; k++) {   // the quest's files into the key
                char file[2500];
                snprintf(file, sizeof file, "%s/%s", q->dir, files.name[k]);
                free(source(file, files.name[k], &size));
            }
            if (files.n || q->has_base) nquests++;
            else rt_log("dlc: %s: quest %04x has neither files in %s nor a base; left out", dlc, v, q->dir);
        } else if (!strcmp(n->name, "training") && ntrainings < MAX_TRAININGS && x_attr(n, "text") &&
                   hex4(x_attr(n, "quest"), &v)) {
            trainings[ntrainings++] = (Training){(uint32_t)strtoul(x_attr(n, "value") ? x_attr(n, "value") : "0", NULL, 10),
                                                 v, x_attr(n, "text")};
        } else if (!strcmp(n->name, "window") && nwindows < MAX_WINDOWS && hex4(x_attr(n, "number"), &v)) {
            windows[nwindows++] = v;
        } else if (!strcmp(n->name, "text") && ntexts < MAX_TEXTS && x_attr(n, "file") && x_attr(n, "id")) {
            Text *t = &texts[ntexts];
            *t = (Text){.file = x_attr(n, "file"), .id = x_attr(n, "id")};
            for (int k = 0; k < n->nkid && t->nt < MAX_LANGS; k++)
                if (!strcmp(n->kid[k]->name, "t")) t->t[t->nt++] = (Lang){x_attr(n->kid[k], "lang"), n->kid[k]->value};
            if (t->nt) ntexts++;
            const char *is_live = x_attr(n, "live");
            if (t->nt && is_live && !strcmp(is_live, "1") && nlive < MAX_LIVE) live[nlive++] = t->id;
        } else {
            char shown[49];
            rt_log("dlc: %s: dlc.xml: <%s> is not understood, lacks an attribute or is one too many; left out", dlc,
                   printable(n->name, 48, shown));
        }
    }
    rt_log("dlc: %s: %d quests, %d texts (%d with input fields), %d training rows, %d tutorial windows", dlc,
           nquests - q0, ntexts - t0, nlive - l0, ntrainings - r0, nwindows - w0);
    loaded[nloaded++] = arena_str(&kept, dlc, strlen(dlc));
    free(xml);
}

int dlc_dir(char *out, size_t n) {
    char exe[2048];
    if (!plat_exe_path(exe, sizeof exe)) return 0;
    char *cut = strrchr(exe, '/'), *back = strrchr(exe, '\\');
    if (back > cut) cut = back;
    if (!cut) return 0;
    *cut = 0;
    return snprintf(out, n, "%s/dlc", exe) < (int)n;
}

// The DLC folders of dir (NULL: none) in place of what was read before, and the key of what they hold.
static void scan_dir(const char *dir) {
    char sub[2200];
    sha256_init(&key_hash);
    sha256_update(&key_hash, DLC_BUILDER, sizeof DLC_BUILDER);
    if (!texts) texts = calloc(MAX_TEXTS, sizeof *texts);
    need(texts != NULL, "texts");
    nquests = ntexts = ntrainings = nwindows = nlive = nloaded = 0;
    if (dir) {
        Names folders;
        names_of(dir, &folders);
        for (int i = 0; i < folders.n && nloaded < MAX_DLC; i++) {
            snprintf(sub, sizeof sub, "%s/%s", dir, folders.name[i]);
            if (plat_file_exists(sub)) continue;   // a file next to the DLC folders
            manifest(folders.name[i], sub);
        }
    }
    sha256_final(&key_hash, key);
}

static void scan_once(void *arg) {
    (void)arg;
    char dir[2100];
    started_on = cfg_bool("AC_DLC");
    int have = started_on && dlc_dir(dir, sizeof dir);
    if (!started_on) rt_log("dlc: off (dlc = 0, the Mods tab's Load DLC): the game's archives as they are");
    scan_dir(have ? dir : NULL);
    if (have && !nloaded) rt_log("dlc: on, but %s holds no DLC folder", dir);
}

void dlc_scan(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, scan_once, NULL);
}

int dlc_started_on(void) { return started_on; }
const char *dlc_name(int i) { return i >= 0 && i < nloaded ? loaded[i] : NULL; }
const uint8_t *dlc_key(void) { return key; }
int dlc_changes(const char *rel) {
    return !strcmp(rel, "quest/quest.pkz") ? nquests > 0 : !strcmp(rel, "Text/Text.pkz") ? ntexts > 0 : 0;
}
int dlc_training(int i, uint32_t *value, const char **text_id, uint32_t *quest) {
    if (i < 0 || i >= ntrainings) return 0;
    *value = trainings[i].value, *text_id = trainings[i].text, *quest = trainings[i].quest;
    return 1;
}
int dlc_window(int i, uint32_t *number) {
    if (i < 0 || i >= nwindows) return 0;
    *number = windows[i];
    return 1;
}
int dlc_live_text(int i, const char **id) {
    if (i < 0 || i >= nlive) return 0;
    *id = live[i];
    return 1;
}

// ---- the builds
// The quest's DAT into out: the base's files, then the folder's over them. 0 when it cannot be built (logged).
static int quest_dat(const Quest *q, const Pkz *z, Buf *out, Arena *a) {
    DatFile f[MAX_FILES];
    int n = 0, at;
    char name[128], from[16], to[16];
    if (q->has_base) {
        snprintf(name, sizeof name, "quest%04x.dat", q->base);
        int i = pkz_find(z, name, &at);
        uint8_t *raw = i >= 0 ? pkz_unpack(&z->e[i]) : NULL;
        n = raw ? dat_list(raw, (size_t)z->e[i].size, f, MAX_FILES, a) : -1;
        if (n < 0) {
            rt_log("dlc: %s: quest %04x: its base %s is not in the game's archive or is not a DAT; left out", q->dlc,
                   q->id, name);
            free(raw);
            return 0;
        }
        snprintf(from, sizeof from, "_%04x.", q->base);
        snprintf(to, sizeof to, "_%04x.", q->id);
        for (int k = 0; k < n; k++) {   // the files' bytes into the arena, the names to this quest's
            uint8_t *copy = arena_get(a, f[k].size + 1);
            memcpy(copy, f[k].data, f[k].size);
            f[k].data = copy;
            char *tag = strstr(f[k].name, from);
            if (tag) memcpy(tag, to, strlen(to));
        }
        free(raw);
    }
    Names files;
    names_of(q->dir, &files);
    for (int k = 0; k < files.n; k++) {
        char path[2500];
        snprintf(path, sizeof path, "%s/%s", q->dir, files.name[k]);
        uint64_t size;
        uint8_t *d = plat_file_read(path, &size);
        if (!d) continue;   // a folder
        size_t len = strlen(files.name[k]);
        int is_xml = len > 4 && !strcasecmp(files.name[k] + len - 4, ".xml");
        snprintf(name, sizeof name, "%.*s%s", (int)(is_xml ? len - 4 : len), files.name[k], is_xml ? ".bxm" : "");
        int i = 0;
        while (i < n && strcasecmp(f[i].name, name)) i++;
        if (i == MAX_FILES) {
            rt_log("dlc: %s: quest %04x holds more than %d files; %s left out", q->dlc, q->id, MAX_FILES, name);
            free(d);
            continue;
        }
        DatFile file = {i < n ? f[i].name : arena_str(a, name, strlen(name)), NULL, 0};
        if (is_xml) {
            char *text = arena_str(a, (const char *)d, (size_t)size);
            XParse p = {.s = text, .a = a};
            XNode *root = x_parse(&p);
            Buf bxm = {0};
            // the base's file names which of the two magics the game's own file has
            const char *magic = i < n && f[i].size >= 4 && !memcmp(f[i].data, "BXM", 4) ? "BXM" : "XML";
            if (!root || !bxm_write(root, magic, &bxm, a)) {
                rt_log("dlc: %s: quest %04x: %s: %s; the quest is left out", q->dlc, q->id, files.name[k],
                       root ? "too large for a BXM file (65 534 nodes, rows or string bytes)" : p.err);
                free(bxm.p);
                free(d);
                return 0;
            }
            uint8_t *copy = arena_get(a, bxm.n + 1);
            memcpy(copy, bxm.p, bxm.n);
            file.data = copy, file.size = bxm.n;
            free(bxm.p);
        } else {
            uint8_t *copy = arena_get(a, (size_t)size + 1);
            memcpy(copy, d, (size_t)size);
            file.data = copy, file.size = (size_t)size;
        }
        free(d);
        f[i] = file;
        if (i == n) n++;
    }
    if (!n) return 0;
    dat_build(f, n, out);
    return 1;
}

static int build_quests(Pkz *z, Arena *a) {
    int added = 0;
    for (int i = 0; i < nquests; i++) {
        Buf dat = {0};
        char name[32];
        snprintf(name, sizeof name, "quest%04x.dat", quests[i].id);
        if (quest_dat(&quests[i], z, &dat, a)) {
            int at, old = pkz_find(z, name, &at);
            pkz_set(z, name, dat.p, dat.n, a);
            rt_log("dlc: %s: %s (%zu bytes) %s quest/quest.pkz", quests[i].dlc, name, dat.n,
                   old >= 0 ? "replaces the one of" : "added to");
            added++;
        }
        free(dat.p);
    }
    return added;
}

static int build_texts(Pkz *z, Arena *a) {
    int changed = 0, count = z->n;
    for (int e = 0; e < count; e++) {
        const char *name = z->e[e].name, *dot = strrchr(name, '.'), *bar = dot ? dot : name;
        while (bar > name && *bar != '_') bar--;
        if (!dot || bar == name || strcmp(dot, ".bin")) continue;
        size_t stem = (size_t)(bar - name), lang = (size_t)(dot - bar - 1);
        int wanted = 0;
        for (int i = 0; i < ntexts; i++) wanted += strlen(texts[i].file) == stem && !memcmp(texts[i].file, name, stem);
        if (!wanted) continue;
        uint8_t *raw = pkz_unpack(&z->e[e]);
        Ptd t;
        if (!raw || !ptd_read(raw, (size_t)z->e[e].size, &t, wanted, a)) {
            char shown[97];
            rt_log("dlc: %s of Text/Text.pkz is not a text file the builder reads; it keeps the game's texts",
                   printable(name, 96, shown));
            free(raw);
            continue;
        }
        for (int i = 0; i < ntexts; i++) {
            const Text *x = &texts[i];
            if (strlen(x->file) != stem || memcmp(x->file, name, stem)) continue;
            const char *text = NULL;
            for (int k = 0; k < x->nt; k++) {
                if (!x->t[k].lang && !text) text = x->t[k].text;
                if (x->t[k].lang && strlen(x->t[k].lang) == lang && !memcmp(x->t[k].lang, bar + 1, lang)) {
                    text = x->t[k].text;
                    break;
                }
            }
            if (text) ptd_set(&t, x->id, text, a);
        }
        Buf out = {0};
        ptd_write(&t, &out);
        pkz_set(z, name, out.p, out.n, a);   // the rows point into raw until here
        free(out.p);
        free(raw);
        changed++;
    }
    if (changed) rt_log("dlc: %d texts added to %d files of Text/Text.pkz", ntexts, changed);
    return changed;
}

void *dlc_build(const char *rel, const void *in, size_t n, size_t *out_n) {
    Pkz z;
    if (!pkz_read(in, n, &z)) {
        rt_log("dlc: %s is not an archive the builder reads; the DLCs' part of it is left out", rel);
        return NULL;
    }
    Arena a = {0};
    int changed = !strcmp(rel, "quest/quest.pkz") ? build_quests(&z, &a) : build_texts(&z, &a);
    Buf out = {0};
    if (changed) pkz_write(&z, &out);
    free(z.e);
    arena_free(&a);
    *out_n = out.n;
    return out.p;
}

// ---- for --probe-dlc (dlc_probe.c): the folder it made in place of the one next to astralchain.exe, whatever the
// setting, and the game's containers written from its own data (the archives it hands to dlc_build).
void dlc_probe_scan(const char *dir) {
    dlc_scan();   // this run's one scan first: what follows replaces its result
    scan_dir(dir);
}

void *dlc_probe_pkz(const char *const *names, const void *const *data, const size_t *sizes, int n, size_t *out_n) {
    Arena a = {0};
    Pkz z = {0x10000, malloc(((size_t)n + 1) * sizeof(PkzEntry)), 0, n};
    need(z.e != NULL, "probe archive");
    for (int i = 0; i < n; i++) pkz_set(&z, names[i], data[i], sizes[i], &a);
    Buf out = {0};
    pkz_write(&z, &out);
    free(z.e);
    arena_free(&a);
    *out_n = out.n;
    return out.p;
}

void *dlc_probe_dat(const char *const *names, const void *const *data, const size_t *sizes, int n, size_t *out_n) {
    DatFile f[MAX_FILES];
    if (n > MAX_FILES) n = MAX_FILES;
    for (int i = 0; i < n; i++) f[i] = (DatFile){names[i], data[i], sizes[i]};
    Buf out = {0};
    dat_build(f, n, &out);
    *out_n = out.n;
    return out.p;
}

// A text file of one section with one list and two columns, as the game's: the texts, and "" for each.
void *dlc_probe_ptd(const char *const *ids, const char *const *text, int n, size_t *out_n) {
    Arena a = {0};
    size_t rows = ((size_t)n + 1) * sizeof(PtdRow);
    PtdList list = {.hash = 2, .v = arena_get(&a, 4 * (size_t)n + 4)};
    PtdRows cols[2] = {{.hash = 3, .r = arena_get(&a, rows)}, {.hash = 4, .r = arena_get(&a, rows)}};
    PtdSection sec = {.hash = 1, .nlists = 1, .ncols = 2, .list = &list, .col = cols};
    Ptd t = {.ids = {.r = arena_get(&a, rows)}, .nsections = 1, .section = &sec};
    for (int i = 0; i < n; i++) ptd_set(&t, ids[i], text[i], &a);
    Buf out = {0};
    ptd_write(&t, &out);
    arena_free(&a);
    *out_n = out.n;
    return out.p;
}
