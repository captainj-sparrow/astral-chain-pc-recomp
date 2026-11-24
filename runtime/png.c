// PNG writer (png.h): 8-bit RGB. Moved out of flight.c, which wrote its 320x180 thumbnails with a fixed-Huffman
// stream, so the photo mode's screenshots (photo.c) share it; it now takes any size and RGB or RGBA input. No
// compression library is linked (the release builds link none, on either OS), so the deflate is our own:
//   - each row takes the filter (PNG 9.2: none, sub, up, average, Paeth) whose residuals have the smallest sum of
//     absolute values, the heuristic of the PNG specification's 12.8 and of libpng;
//   - LZ77 over a 32 KiB window through hash chains of 3-byte keys, at most CHAIN candidates a position, with one step
//     of lazy matching (a match is put off when the next position has a longer one), as zlib's deflate_slow;
//   - dynamic Huffman blocks (RFC 1951 3.2.7) of at most BLOCK tokens, code lengths limited to 15 (7 for the code
//     length code) by halving the counts until the tree fits.
// Measured on a 1920x1080 photo of Harmony Square (a photo.c screenshot), clang -O2, one thread of a Core Ultra 9
// 285H: the former fixed-Huffman stream with Paeth rows gave 4,744,385 bytes (zlib level 1 through Pillow: 4,728,994),
// this writer 3,395,680 bytes in 234 ms (zlib level 6: 3,300,515); the same picture scaled to 3840x2160 9,225,641
// bytes in 897 ms (level 6: 8,901,958). Every test picture (random, flat, 1x1, one row, one column) decoded back
// byte for byte with Pillow. NICE is zlib level 6's nice length; CHAIN is a quarter of its chain (128).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "png.h"

#define WBITS 15
#define WSIZE (1u << WBITS)   // the window: the largest distance a match may have
#define HBITS 15
#define CHAIN 32              // candidates tried per position
#define NICE 128              // a match this long ends the search
#define BLOCK 32768           // tokens per Huffman block

typedef struct { uint8_t *p; size_t n; uint64_t bits; int nb; } Bits;

static void put(Bits *b, uint32_t v, int n) {
    b->bits |= (uint64_t)v << b->nb;
    b->nb += n;
    while (b->nb >= 8) b->p[b->n++] = (uint8_t)b->bits, b->bits >>= 8, b->nb -= 8;
}

static uint32_t rev(uint32_t v, int n) {
    uint32_t r = 0;
    while (n--) r = r << 1 | (v & 1), v >>= 1;
    return r;
}

static const uint16_t lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
                                   131, 163, 195, 227, 258};
static const uint8_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
                                   1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
static const uint8_t cl_order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

static uint8_t len_code[259];   // match length -> length code index (0..28)
static uint8_t dist_lo[256];    // distance - 1 < 256 -> distance code
static uint8_t dist_hi[256];    // (distance - 1) >> 7 for larger distances -> distance code

static void tables(void) {
    static int done;
    if (__atomic_load_n(&done, __ATOMIC_ACQUIRE)) return;
    for (int k = 0; k < 29; k++)
        for (int l = lbase[k]; l < (k == 28 ? 259 : lbase[k + 1]); l++) len_code[l] = (uint8_t)k;
    len_code[258] = 28;
    for (int k = 0; k < 30; k++)
        for (int d = dbase[k]; d < (k == 29 ? 32769 : dbase[k + 1]); d++) {
            if (d - 1 < 256) dist_lo[d - 1] = (uint8_t)k;
            else dist_hi[(d - 1) >> 7] = (uint8_t)k;
        }
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);   // the same tables whichever thread builds them first
}

static int dist_code(uint32_t d) { return d - 1 < 256 ? dist_lo[d - 1] : dist_hi[(d - 1) >> 7]; }

// ---- Huffman code lengths, limited to `limit` bits

typedef struct { uint32_t f; int s; } Node;

static int node_cmp(const void *a, const void *b) {
    const Node *x = a, *y = b;
    return x->f != y->f ? (x->f < y->f ? -1 : 1) : x->s - y->s;
}

// Lengths of the Huffman code for the n counts in f (0: unused). Two queues over the sorted leaves (no heap); when the
// longest code passes limit, the counts are halved (a used one stays at least 1) and the tree is built again.
static void huff_lengths(const uint32_t *f_in, int n, int limit, uint8_t *len) {
    uint32_t f[288];
    Node leaf[288];
    uint32_t wt[576];
    int parent[576], depth[576];
    memcpy(f, f_in, sizeof(uint32_t) * (size_t)n);
    for (;;) {
        int m = 0;
        for (int i = 0; i < n; i++)
            if (f[i]) leaf[m++] = (Node){f[i], i};
        memset(len, 0, (size_t)n);
        if (m == 0) return;
        if (m == 1) {   // one symbol: one code of one bit (RFC 1951 lets a single code be incomplete)
            len[leaf[0].s] = 1;
            return;
        }
        qsort(leaf, (size_t)m, sizeof *leaf, node_cmp);
        // nodes 0..m-1 the leaves in order, m..2m-2 the internal nodes in the order they are made
        for (int i = 0; i < m; i++) wt[i] = leaf[i].f;
        int li = 0, ii = m, made = m;
        for (int k = 0; k < m - 1; k++) {
            int pick[2];
            for (int t = 0; t < 2; t++)
                pick[t] = li < m && (ii >= made || wt[li] <= wt[ii]) ? li++ : ii++;
            wt[made] = wt[pick[0]] + wt[pick[1]];
            parent[pick[0]] = parent[pick[1]] = made;
            made++;
        }
        depth[made - 1] = 0;
        int longest = 0;
        for (int i = made - 2; i >= 0; i--) depth[i] = depth[parent[i]] + 1;
        for (int i = 0; i < m; i++) {
            len[leaf[i].s] = (uint8_t)depth[i];
            if (depth[i] > longest) longest = depth[i];
        }
        if (longest <= limit) return;
        for (int i = 0; i < n; i++)
            if (f[i]) f[i] = f[i] >> 1 | 1;
    }
}

// Canonical codes (RFC 1951 3.2.2), stored bit-reversed for the LSB-first writer.
static void huff_codes(const uint8_t *len, int n, uint16_t *code) {
    uint16_t count[16] = {0}, next[16];
    for (int i = 0; i < n; i++) count[len[i]]++;
    count[0] = 0;
    uint16_t c = 0;
    for (int b = 1; b < 16; b++) c = (uint16_t)((c + count[b - 1]) << 1), next[b] = c;
    for (int i = 0; i < n; i++)
        if (len[i]) code[i] = (uint16_t)rev(next[len[i]]++, len[i]);
}

// ---- blocks

typedef struct { uint16_t a, d; } Tok;   // d = 0: literal a; else a match of length a at distance d

static void block(Bits *b, const Tok *t, size_t nt, int last) {
    uint32_t lf[286] = {0}, df[30] = {0};
    for (size_t i = 0; i < nt; i++) {
        if (!t[i].d) lf[t[i].a]++;
        else lf[257 + len_code[t[i].a]]++, df[dist_code(t[i].d)]++;
    }
    lf[256] = 1;
    uint8_t ll[286], dl[30];
    huff_lengths(lf, 286, 15, ll);
    huff_lengths(df, 30, 15, dl);
    int nd = 0;
    for (int i = 0; i < 30; i++)
        if (dl[i]) nd = i + 1;
    if (!nd) dl[0] = 1, nd = 1;   // a block without matches still sends one distance code
    int nl = 257;
    for (int i = 257; i < 286; i++)
        if (ll[i]) nl = i + 1;
    // the code lengths of both codes as one run-length coded sequence (16: repeat the previous 3-6 times, 17: 3-10
    // zeros, 18: 11-138 zeros)
    uint8_t seq[316], rl[316], ex[316];
    int ns = 0, nr = 0;
    memcpy(seq, ll, (size_t)nl);
    memcpy(seq + nl, dl, (size_t)nd);
    ns = nl + nd;
    uint32_t cf[19] = {0};
    for (int i = 0; i < ns;) {
        int v = seq[i], run = 1;
        while (i + run < ns && seq[i + run] == v) run++;
        if (v == 0 && run >= 3) {
            int k = run > 138 ? 138 : run;
            rl[nr] = k >= 11 ? 18 : 17, ex[nr++] = (uint8_t)(k >= 11 ? k - 11 : k - 3);
            cf[k >= 11 ? 18 : 17]++;
            i += k;
        } else if (v != 0 && run >= 4) {
            rl[nr] = (uint8_t)v, ex[nr++] = 0, cf[v]++;
            int k = run - 1 > 6 ? 6 : run - 1;
            rl[nr] = 16, ex[nr++] = (uint8_t)(k - 3), cf[16]++;
            i += 1 + k;
        } else {
            rl[nr] = (uint8_t)v, ex[nr++] = 0, cf[v]++;
            i++;
        }
    }
    uint8_t cl[19];
    uint16_t cc[19] = {0}, lc[286] = {0}, dc[30] = {0};
    huff_lengths(cf, 19, 7, cl);
    int ncl = 4;
    for (int i = 0; i < 19; i++)
        if (cl[cl_order[i]]) ncl = i + 1 > ncl ? i + 1 : ncl;
    huff_codes(cl, 19, cc);
    huff_codes(ll, 286, lc);
    huff_codes(dl, 30, dc);
    put(b, (uint32_t)last, 1), put(b, 2, 2);
    put(b, (uint32_t)(nl - 257), 5), put(b, (uint32_t)(nd - 1), 5), put(b, (uint32_t)(ncl - 4), 4);
    for (int i = 0; i < ncl; i++) put(b, cl[cl_order[i]], 3);
    for (int i = 0; i < nr; i++) {
        put(b, cc[rl[i]], cl[rl[i]]);
        if (rl[i] == 16) put(b, ex[i], 2);
        else if (rl[i] == 17) put(b, ex[i], 3);
        else if (rl[i] == 18) put(b, ex[i], 7);
    }
    for (size_t i = 0; i < nt; i++) {
        if (!t[i].d) {
            put(b, lc[t[i].a], ll[t[i].a]);
            continue;
        }
        int k = len_code[t[i].a], j = dist_code(t[i].d);
        put(b, lc[257 + k], ll[257 + k]);
        if (lext[k]) put(b, (uint32_t)(t[i].a - lbase[k]), lext[k]);
        put(b, dc[j], dl[j]);
        if (dext[j]) put(b, (uint32_t)(t[i].d - dbase[j]), dext[j]);
    }
    put(b, lc[256], ll[256]);
}

// ---- LZ77

static inline uint32_t hash3(const uint8_t *p) {
    return ((uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2]) * 2654435761u >> (32 - HBITS);
}

// The longest match for position i among the chain's candidates; its length (0 below 3) and distance.
static uint32_t longest(const uint8_t *src, size_t n, size_t i, int32_t cand, const int32_t *prev, uint32_t *dist) {
    uint32_t best = 0, lim = n - i < 258 ? (uint32_t)(n - i) : 258;
    for (int k = 0; k < CHAIN && cand >= 0 && i - (size_t)cand <= WSIZE; k++) {
        const uint8_t *a = src + cand, *s = src + i;
        if (a[best] == s[best]) {   // only a candidate that can beat the best is compared whole
            uint32_t m = 0;
            while (m < lim && a[m] == s[m]) m++;
            if (m > best) {
                best = m, *dist = (uint32_t)(i - (size_t)cand);
                if (m >= NICE || m == lim) break;
            }
        }
        int32_t nx = prev[cand & (WSIZE - 1)];
        if (nx >= cand) break;   // an older position's slot, already overwritten
        cand = nx;
    }
    return best >= 3 ? best : 0;
}

static size_t deflate_dyn(const uint8_t *src, size_t n, uint8_t *out, int32_t *head, int32_t *prev, Tok *tok) {
    tables();
    Bits b = {out, 0, 0, 0};
    b.p[b.n++] = 0x78, b.p[b.n++] = 0x9c;
    for (int i = 0; i < 1 << HBITS; i++) head[i] = -1;
    size_t nt = 0, i = 0;
    uint32_t plen = 0, pdist = 0;   // a match found at i - 1, put off to look at i (lazy matching)
#define INSERT(pos) do { if ((pos) + 3 <= n) { uint32_t h_ = hash3(src + (pos)); prev[(pos) & (WSIZE - 1)] = head[h_]; head[h_] = (int32_t)(pos); } } while (0)
#define EMIT(A, D) do { tok[nt++] = (Tok){(uint16_t)(A), (uint16_t)(D)}; if (nt == BLOCK) block(&b, tok, nt, 0), nt = 0; } while (0)
    while (i < n) {
        uint32_t len = 0, dist = 0;
        if (i + 3 <= n) {
            uint32_t h = hash3(src + i);
            int32_t cand = head[h];
            prev[i & (WSIZE - 1)] = cand;
            head[h] = (int32_t)i;
            if (cand >= 0 && (!plen || plen < NICE)) len = longest(src, n, i, cand, prev, &dist);
        }
        if (plen) {
            if (len > plen) {   // the later match is longer: the earlier position goes out as a literal
                EMIT(src[i - 1], 0);
                plen = len, pdist = dist;
                i++;
                continue;
            }
            EMIT(plen, pdist);   // the earlier match: it covers i - 1 .. i - 1 + plen, i already inserted
            for (size_t s = i + 1; s < i - 1 + plen; s++) INSERT(s);
            i = i - 1 + plen;
            plen = 0;
            continue;
        }
        if (len && len < NICE) {   // look at the next position before taking it
            plen = len, pdist = dist;
            i++;
            continue;
        }
        if (len) {
            EMIT(len, dist);
            for (size_t s = i + 1; s < i + len; s++) INSERT(s);
            i += len;
            continue;
        }
        EMIT(src[i], 0);
        i++;
    }
    if (plen) EMIT(plen, pdist);
#undef INSERT
#undef EMIT
    block(&b, tok, nt, 1);
    if (b.nb) put(&b, 0, 8 - b.nb);
    uint32_t a = 1, s2 = 0;
    for (size_t k = 0; k < n;) {   // Adler-32 in runs short enough for 32-bit sums
        size_t e = n - k < 5552 ? n : k + 5552;
        for (; k < e; k++) a += src[k], s2 += a;
        a %= 65521, s2 %= 65521;
    }
    uint32_t ad = s2 << 16 | a;
    b.p[b.n++] = (uint8_t)(ad >> 24), b.p[b.n++] = (uint8_t)(ad >> 16), b.p[b.n++] = (uint8_t)(ad >> 8), b.p[b.n++] = (uint8_t)ad;
    return b.n;
}

// ---- PNG

static uint32_t crc_table[256];

static uint32_t crc(uint32_t c, const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}

static void chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t h[8] = {(uint8_t)(n >> 24), (uint8_t)(n >> 16), (uint8_t)(n >> 8), (uint8_t)n};
    memcpy(h + 4, type, 4);
    uint32_t c = crc(crc(~0u, h + 4, 4), data, n) ^ ~0u;
    uint8_t t[4] = {(uint8_t)(c >> 24), (uint8_t)(c >> 16), (uint8_t)(c >> 8), (uint8_t)c};
    fwrite(h, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    fwrite(t, 1, 4, f);
}

static inline uint8_t paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return (uint8_t)(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

void png_bufs_free(PngBufs *b) {
    free(b->raw), free(b->z), free(b->head);
    memset(b, 0, sizeof *b);
}

// Row cur (w RGB pixels; up: the row above, NULL for the first) filtered into o, filter byte first, with the filter
// whose residuals have the smallest sum of absolute values. cand: room for the five candidates.
static void filter_row(const uint8_t *cur, const uint8_t *up, uint32_t w, uint8_t *o, uint8_t *cand) {
    size_t m = (size_t)w * 3, k = m < 3 ? m : 3;
    uint8_t *r[5];
    for (int f = 0; f < 5; f++) r[f] = cand + (size_t)f * m;
    memcpy(r[0], cur, m);
    for (size_t i = 0; i < k; i++) r[1][i] = cur[i];
    for (size_t i = k; i < m; i++) r[1][i] = (uint8_t)(cur[i] - cur[i - 3]);
    if (up) {
        for (size_t i = 0; i < m; i++) r[2][i] = (uint8_t)(cur[i] - up[i]);
        for (size_t i = 0; i < k; i++) r[3][i] = (uint8_t)(cur[i] - (up[i] >> 1)), r[4][i] = (uint8_t)(cur[i] - up[i]);
        for (size_t i = k; i < m; i++) {
            r[3][i] = (uint8_t)(cur[i] - ((cur[i - 3] + up[i]) >> 1));
            r[4][i] = (uint8_t)(cur[i] - paeth(cur[i - 3], up[i], up[i - 3]));
        }
    } else {   // the first row: average is the left neighbour's half; up and Paeth would equal none and sub
        for (size_t i = 0; i < k; i++) r[3][i] = cur[i];
        for (size_t i = k; i < m; i++) r[3][i] = (uint8_t)(cur[i] - (cur[i - 3] >> 1));
    }
    static const int all[5] = {0, 1, 2, 3, 4}, first[3] = {0, 1, 3};
    const int *fs = up ? all : first;
    uint64_t best = ~0ull;
    int pick = 0;
    for (int n = 0; n < (up ? 5 : 3); n++) {
        const uint8_t *q = r[fs[n]];
        uint64_t sum = 0;
        for (size_t i = 0; i < m; i++) sum += q[i] < 128 ? q[i] : 256 - q[i];
        if (sum < best) best = sum, pick = fs[n];
    }
    o[0] = (uint8_t)pick;
    memcpy(o + 1, r[pick], m);
}

int png_write_rgb(const char *path, const uint8_t *px, uint32_t w, uint32_t h, uint32_t step, size_t stride,
                  PngBufs *b) {
    if (!crc_table[1]) {   // the same table whichever thread builds it first
        uint32_t t[256];
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        memcpy(crc_table, t, sizeof t);
    }
    size_t row = 1 + 3 * (size_t)w, raw_n = row * h;
    // raw: the filtered rows; z: the stream (stored-size bound plus the block headers); head: the hash heads, then
    // the chain (WSIZE), the tokens (BLOCK) and the work rows (two RGB rows and five candidates)
    // A Huffman code averages under 9 bits a symbol here (at most one bit over the entropy of 286 symbols, and the
    // length limit flattens it), and each block adds at most 320 bytes of code lengths.
    size_t z_n = raw_n + raw_n / 4 + (raw_n / BLOCK + 1) * 320 + 64;
    size_t aux = (sizeof(int32_t) << HBITS) + sizeof(int32_t) * WSIZE + sizeof(Tok) * BLOCK + 7 * 3 * (size_t)w;
    if (raw_n > b->cap || !b->raw || !b->z || !b->head) {
        png_bufs_free(b);
        b->raw = malloc(raw_n);
        b->z = malloc(z_n);
        b->head = malloc(aux);
        if (!b->raw || !b->z || !b->head) {
            png_bufs_free(b);
            return 0;
        }
        b->cap = raw_n;
    }
    int32_t *prev = b->head + (1 << HBITS);
    Tok *tok = (Tok *)(prev + WSIZE);
    uint8_t *rows = (uint8_t *)(tok + BLOCK), *rgb[2] = {rows, rows + 3 * (size_t)w}, *cand = rows + 6 * (size_t)w;
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t *s = px + (size_t)y * stride;
        uint8_t *c = rgb[y & 1];
        if (step == 3) memcpy(c, s, 3 * (size_t)w);
        else
            for (uint32_t x = 0; x < w; x++) c[3 * x] = s[step * x], c[3 * x + 1] = s[step * x + 1], c[3 * x + 2] = s[step * x + 2];
        filter_row(c, y ? rgb[(y + 1) & 1] : NULL, w, b->raw + (size_t)y * row, cand);
    }
    size_t zn = deflate_dyn(b->raw, raw_n, b->z, b->head, prev, tok);
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                        (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
    fwrite(sig, 1, 8, f);
    chunk(f, "IHDR", ihdr, 13);
    chunk(f, "IDAT", b->z, (uint32_t)zn);
    chunk(f, "IEND", NULL, 0);
    return fclose(f) == 0;
}
