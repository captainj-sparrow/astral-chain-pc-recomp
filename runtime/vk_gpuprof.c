// AC_GPU_PASSES=<frames>: GPU time of every render pass, dispatch and copy the replay records, summed over that many
// presented frames and logged as a table, most expensive first. Each marker is a reset and a BOTTOM_OF_PIPE timestamp
// written outside render passes: at the start of a command buffer (vk_exec.c vk_exec_record and maybe_split), before
// vkCmdBeginRendering (the barriers, contact shadows and DLAA work recorded since the previous marker: "pre"), after
// vkCmdEndRendering ("pass"), after a dispatch or copy op outside a pass, after the ambient occlusion ("ao", vk_ao.c),
// and at the end of the command buffer
// ("tail": the final barrier and the VRAM mirror copies). A marker's time is the span since the previous marker of its
// command buffer; the first marker of a command buffer only starts a span, so the idle time between submissions counts
// nowhere. The replay barriers before every pass, so passes do not overlap and the spans add up to the replay's GPU
// time. Rows are keyed by the marker's place in its frame (ordinal) and what it rendered, so a stable scene sums the
// same pass of every frame into one row. The present path (FSR, downscale, overlay) records outside the replay and is
// not in the table.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "vk_exec.h"

#define GP_QUERIES 32768
#define GP_ROWS 1024

typedef struct {
    uint8_t kind;          // VK_GP_*
    uint16_t ordinal;      // marker number in its frame (shown, not keyed: it shifts when a pass comes and goes)
    uint16_t occ;          // how many markers with the same signature came before it in its frame (keyed)
    uint32_t w, h, layers;
    int32_t fmt0, fmtd;    // NVNformat of colour target 0 and of the depth target, -1 for none
    uint32_t ncolor, draws;
    uint64_t prog;         // dispatch: the program serial; copy: the op
    uint64_t frame;
} GpLabel;

static const char *const kind_name[] = {"start", "pre", "pass", "dispatch", "copy", "tail", "ao"};

static struct {
    int on;                // -1 unread, 0 off, else frames per report
    VkQueryPool pool;
    double period;
    uint64_t mask;
    uint32_t next, first;  // next query; first query since the last flush
    GpLabel lab[GP_QUERIES];
    volatile uint8_t busy[GP_QUERIES];
    uint64_t frame;        // render thread: frame of the ordinals
    uint16_t ordinal;
    GpLabel sig[256];      // render thread: signatures seen in this frame, occ holding their count
    int nsig;
    uint64_t pass_draws0;  // vk_exec_frame.draws at the open pass's begin
} G = {.on = -1};

// Completion thread: the table.
typedef struct {
    GpLabel k;
    uint64_t ns, n, draws;
} GpRow;
static GpRow rows[GP_ROWS];
static int nrows, overflow;
static uint64_t total_ns, frames_seen, last_frame, dropped;

int vk_gpuprof_on(void) {
    if (G.on < 0) {
        const char *e = cfg_get("AC_GPU_PASSES");   // diagnostic: no settings row, unset is off
        int n = e && *e ? atoi(e) : 0;
        G.on = n > 0 ? n : 0;
        if (G.on) {
            VkCtx *c = vk_ctx();
            VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                        .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = GP_QUERIES};
            if (c->props.limits.timestampPeriod <= 0 || vkCreateQueryPool(c->device, &qi, NULL, &G.pool) != VK_SUCCESS) {
                rt_log("gpu passes: no timestamp query pool: off");
                G.on = 0;
            } else {
                G.period = c->props.limits.timestampPeriod;
                G.mask = ~0ull;
                rt_log("gpu passes: GPU time per pass, logged every %d frames (AC_GPU_PASSES)", G.on);
            }
        }
    }
    return G.on;
}

static int same_sig(const GpLabel *a, const GpLabel *b) {
    return a->kind == b->kind && a->w == b->w && a->h == b->h && a->layers == b->layers && a->fmt0 == b->fmt0 &&
           a->fmtd == b->fmtd && a->ncolor == b->ncolor && a->prog == b->prog;
}

static int same_key(const GpLabel *a, const GpLabel *b) {
    return a->kind == b->kind && a->occ == b->occ && a->w == b->w && a->h == b->h && a->layers == b->layers &&
           a->fmt0 == b->fmt0 && a->fmtd == b->fmtd && a->ncolor == b->ncolor && a->prog == b->prog;
}

static int cmp_rows(const void *x, const void *y) {
    const GpRow *a = x, *b = y;
    return a->ns < b->ns ? 1 : a->ns > b->ns ? -1 : 0;
}

static void report(void) {
    qsort(rows, (size_t)nrows, sizeof rows[0], cmp_rows);
    double f = (double)frames_seen;
    rt_log("gpu passes: %llu frames, replay %.2f ms/frame over %d rows%s%s", (unsigned long long)frames_seen,
           (double)total_ns / f / 1e6, nrows, overflow ? " (table full: rows lost)" : "",
           dropped ? " (markers dropped: query ring busy)" : "");
    double cum = 0;
    for (int i = 0; i < nrows && i < 40; i++) {
        const GpRow *r = &rows[i];
        double ms = (double)r->ns / f / 1e6;
        cum += (double)r->ns;
        char what[96];
        if (r->k.kind == 3) snprintf(what, sizeof what, "prog %llu", (unsigned long long)r->k.prog);
        else if (r->k.kind == 4) snprintf(what, sizeof what, "op %llu", (unsigned long long)r->k.prog);
        else if (r->k.kind == 6) snprintf(what, sizeof what, "step %llu", (unsigned long long)r->k.prog);
        else snprintf(what, sizeof what, "%ux%u layers %u colour %u fmt %#x depth %#x", r->k.w, r->k.h, r->k.layers,
                      r->k.ncolor, (unsigned)r->k.fmt0, (unsigned)r->k.fmtd);
        rt_log("gpu passes: %2d #%-3u.%u %-8s %6.3f ms %5.1f%% (cum %5.1f%%) x%.2f/frame draws %.0f  %s", i, r->k.ordinal, r->k.occ,
               kind_name[r->k.kind], ms, 100.0 * (double)r->ns / (double)total_ns, 100.0 * cum / (double)total_ns,
               (double)r->n / f, r->n ? (double)r->draws / (double)r->n : 0.0, what);
    }
    // Totals by kind and, for passes, by render area: full, half (a quarter of the pixels or less), other.
    double by_kind[7] = {0}, by_area[3] = {0};
    uint32_t big = 0;
    for (int i = 0; i < nrows; i++)
        if (rows[i].k.kind == 2 && rows[i].k.w * rows[i].k.h > big) big = rows[i].k.w * rows[i].k.h;
    for (int i = 0; i < nrows; i++) {
        const GpRow *r = &rows[i];
        by_kind[r->k.kind] += (double)r->ns;
        if (r->k.kind != 2) continue;
        uint32_t px = r->k.w * r->k.h;
        by_area[px == big ? 0 : px * 4 <= big ? 1 : 2] += (double)r->ns;
    }
    rt_log("gpu passes: by kind (ms/frame): pass %.2f, pre (barriers, contact shadows, DLAA) %.2f, dispatch %.2f, copy %.2f, "
           "tail %.2f, ambient occlusion %.2f; passes by area: largest (%u px) %.2f, a quarter or less %.2f, between %.2f",
           by_kind[2] / f / 1e6, by_kind[1] / f / 1e6, by_kind[3] / f / 1e6, by_kind[4] / f / 1e6, by_kind[5] / f / 1e6,
           by_kind[6] / f / 1e6, big,
           by_area[0] / f / 1e6, by_area[1] / f / 1e6, by_area[2] / f / 1e6);
    nrows = overflow = 0;
    total_ns = frames_seen = 0;
    dropped = 0;
}

typedef struct { uint32_t first, n; } GpBatch;

static void read_batch(void *arg) {   // completion thread, in submission order
    GpBatch *b = arg;
    VkCtx *c = vk_ctx();
    uint64_t prev = 0;
    int have_prev = 0;
    for (uint32_t i = 0; i < b->n; i++) {
        uint32_t q = (b->first + i) % GP_QUERIES;
        uint64_t t;
        const GpLabel *l = &G.lab[q];
        int ok = vkGetQueryPoolResults(c->device, G.pool, q, 1, sizeof t, &t, sizeof t, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        __atomic_store_n(&G.busy[q], 0, __ATOMIC_RELEASE);
        if (!ok) { have_prev = 0; continue; }
        if (l->kind != 0 && have_prev && t >= prev) {
            uint64_t ns = (uint64_t)((double)(t - prev) * G.period);
            if (l->frame != last_frame) last_frame = l->frame, frames_seen++;
            total_ns += ns;
            int r = 0;
            while (r < nrows && !same_key(&rows[r].k, l)) r++;
            if (r == nrows) {
                if (nrows < GP_ROWS) rows[nrows++] = (GpRow){.k = *l};
                else overflow = 1, r = -1;
            }
            if (r >= 0) rows[r].ns += ns, rows[r].n++, rows[r].draws += l->draws;
        }
        prev = t, have_prev = 1;
    }
    free(b);
    if (frames_seen > (uint64_t)G.on) report();
}

// Render thread, outside a render pass.
void vk_gpuprof_mark(VkExec *ex, int kind, uint64_t prog) {
    if (!vk_gpuprof_on() || !ex->cb) return;
    uint32_t q = G.next % GP_QUERIES;
    if (__atomic_load_n(&G.busy[q], __ATOMIC_ACQUIRE)) { dropped++; return; }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (frame != G.frame) G.frame = frame, G.ordinal = 0, G.nsig = 0;
    GpLabel *l = &G.lab[q];
    memset(l, 0, sizeof *l);
    l->kind = (uint8_t)kind, l->frame = frame, l->fmt0 = l->fmtd = -1, l->prog = prog;
    if (kind != 0) l->ordinal = G.ordinal++;
    if (kind == 1 || kind == 2) {   // pre: the pass about to begin; pass: the one just ended
        l->w = ex->pass_w, l->h = ex->pass_h, l->layers = ex->pass_layers, l->ncolor = ex->pass_ncolor;
        if (ex->pass_ncolor && ex->color[0].tex) l->fmt0 = ex->color[0].nvn_format;
        if (ex->depth.tex) l->fmtd = ex->depth.nvn_format;
        if (kind == 1) G.pass_draws0 = vk_exec_frame.draws;
        else l->draws = (uint32_t)(vk_exec_frame.draws - G.pass_draws0);
    }
    if (kind != 0) {   // occurrence of this signature in the frame
        int i = 0;
        while (i < G.nsig && !same_sig(&G.sig[i], l)) i++;
        if (i == G.nsig && G.nsig < 256) G.sig[G.nsig] = *l, G.sig[G.nsig++].occ = 0;
        if (i < 256) l->occ = G.sig[i].occ++;
        else l->occ = 0xffff;
    }
    G.busy[q] = 1;
    G.next++;
    vkCmdResetQueryPool(ex->cb, G.pool, q, 1);
    vkCmdWriteTimestamp(ex->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.pool, q);
}

// End of vk_exec_record: the markers since the last flush are read once fence is done.
void vk_gpuprof_flush(uint64_t fence) {
    if (!G.on || G.next == G.first) return;
    GpBatch *b = malloc(sizeof *b);
    if (!b) rt_fatal("vk: out of memory");
    b->first = G.first % GP_QUERIES, b->n = G.next - G.first;
    G.first = G.next;
    vk_completion_add(fence, read_batch, b);
}
