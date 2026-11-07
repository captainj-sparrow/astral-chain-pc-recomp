// Last frame's uniform blocks of the scene's objects, for per-object motion vectors (AC_DLAA_OBJMV, on with DLAA;
// vk_exec_draw.c "per-object motion" draws them).
//
// The TotK Recomp finds last frame's block of an object in guest memory: TotK writes each object's blocks into two
// copies that take turns, so last frame's sits next to this frame's. Astral Chain does not: measured (AC_DLAA_SURVEY on
// frames 4800-4802 outside the police HQ), its per-object blocks (c14 the object's place, c5-c13 a skinned model's
// bones and such) come from a linear allocator that hands out new addresses every frame, 15 of 144 c14 blocks keep
// theirs, none two frames apart, and the old bytes are written over. So the blocks each motion candidate reads are
// copied here, from the replay's uniform snapshot (vk_usnap.c, as of the submit call), into a host-visible arena with
// one slot per frame, and found again next frame by the draw's identity:
//   key     the vertex program, index buffer, first vertex, index count and instance count: one part of one model
//   anchor  the first matrix's translation of the first object bank the position reads: which of several copies of
//           that part (the same crate twice) a draw is; last frame's entry with the nearest anchor, at most MAX_JUMP
//           away, is the same object
// A block bound by several draws in a frame (a model's parts share their place and bones) is copied once. The slots
// rotate with the DLAA evaluations (vk_mvprev_swap), so a frame's copies stay untouched while the GPU may still read
// them. Summary every SUMMARY frames:
//   mv_prev: <n> objects a frame, matched a %, new b %, of them still c %; <k> KiB copied a frame (<m> blocks)
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_dlaa.h"

#define SLOTS 4u               // frames the GPU may still be reading last frame's copies of, and this one
#define SLOT_BYTES (8u << 20)  // copies a frame
#define OBJS 8192u             // objects a frame
#define TABLE (1u << 14)       // hash table entries (keys and block addresses), power of two
#define BANKS 14               // c3..c16
#define ALIGN 256u
#define MAX_JUMP 4.0f          // farther than this from every candidate: a new object
#define SUMMARY 600

typedef struct {
    uint64_t key;
    float anchor[3];
    int has_anchor;
    uint32_t off[BANKS], size[BANKS];   // copy of each bank in its frame's slot; size 0: none
    int32_t next;                       // next object of the same key, -1 none
} Obj;

typedef struct { uint64_t k; int32_t v; } Ent;   // k 0: empty

typedef struct {
    Obj obj[OBJS];
    int n;
    Ent keys[TABLE];     // key -> first object
    Ent blocks[TABLE];   // block address ^ size << 48 -> offset in the slot
    uint32_t used;       // bytes of the slot used
    uint32_t slot;
} Frame;

static struct {
    int knob;            // -1 not read
    int timed;           // AC_DLAA_TIMES: the summary's times
    uint64_t ns[2];      // since the last summary: copying the blocks, matching and comparing
    VkBuffer buf;
    VmaAllocation mem;
    uint8_t *map;
    Frame *f[2];         // this frame's, last frame's
    uint32_t slot;       // this frame's arena slot
    uint64_t frames, objs, matched, still, bytes, nblocks, full;
} M = {.knob = -1};

static uint32_t hash64(uint64_t k) { return (uint32_t)((k * 0x9E3779B97F4A7C15ull) >> 40) & (TABLE - 1); }

static Ent *ent(Ent *t, uint64_t k, int add) {
    for (uint32_t i = hash64(k), n = 0; n < TABLE; i = (i + 1) & (TABLE - 1), n++) {
        if (t[i].k == k) return &t[i];
        if (!t[i].k) {
            if (!add) return NULL;
            t[i].k = k, t[i].v = -1;
            return &t[i];
        }
    }
    return NULL;
}

static void frame_reset(Frame *f, uint32_t slot) {
    f->n = 0, f->used = 0, f->slot = slot;
    memset(f->keys, 0, sizeof f->keys), memset(f->blocks, 0, sizeof f->blocks);
}

static int init(void) {
    if (M.knob >= 0) return M.knob;
    M.knob = rt_knob("DLAA_OBJMV", "1");
    if (!M.knob) return 0;
    M.timed = rt_knob("DLAA_TIMES", "0");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)SLOTS * SLOT_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    // random access: the copies are compared on the CPU next frame, which uncached memory would make slow
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    M.f[0] = calloc(1, sizeof(Frame)), M.f[1] = calloc(1, sizeof(Frame));
    if (!M.f[0] || !M.f[1] || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &M.buf, &M.mem, &info) != VK_SUCCESS) {
        rt_log("mv_prev: the history arena could not be made: per-object motion off");
        return M.knob = 0;
    }
    M.map = info.pMappedData;
    frame_reset(M.f[0], 0), frame_reset(M.f[1], SLOTS - 1);
    M.f[1]->n = 0;
    rt_log("mv_prev: per-object motion on (%u MiB history, AC_DLAA_OBJMV=0 turns it off)", SLOTS * SLOT_BYTES >> 20);
    return 1;
}

int vk_mvprev_on(void) { return vk_dlaa_available() && init(); }

// This frame's copy of a block: an earlier draw's copy of the same block, else a new one. -1: no room or not in the
// snapshot.
static int64_t copy_block(uint64_t addr, uint32_t size) {
    Frame *f = M.f[0];
    Ent *e = ent(f->blocks, addr ^ (uint64_t)size << 48, 1);
    if (!e) return -1;
    if (e->v >= 0) return e->v;
    // The snapshot's bytes: with the replay on the submitting thread (replay_async = 0) guest memory still holds them
    // and is cached memory; the snapshot arena is write-combined video memory, slow to read (1.7 ms a frame here).
    VkDescriptorBufferInfo bi;
    if (!vk_usnap_find(addr, size, 1, &bi)) return e->k = 0, -1;
    const uint8_t *src = vk_replay_async() ? vk_usnap_bytes(addr, size) : (const uint8_t *)(uintptr_t)addr;
    uint32_t at = (f->used + ALIGN - 1) & ~(ALIGN - 1);
    if (!src || at + size > SLOT_BYTES) {
        if (src) M.full++;
        e->k = 0;   // not kept: the next draw tries again
        return -1;
    }
    memcpy(M.map + (size_t)f->slot * SLOT_BYTES + at, src, size);
    f->used = at + size;
    M.bytes += size, M.nblocks++;
    return e->v = (int32_t)at;
}

int vk_mvprev_blocks(uint64_t key, uint32_t banks, const p_bind_buf ubo[], uint64_t max, VkDescriptorBufferInfo prev[17],
                     uint32_t *have, uint32_t *changed) {
    *have = *changed = 0;
    if (!vk_mvprev_on()) return 0;
    Frame *f = M.f[0], *l = M.f[1];
    if (f->n >= (int)OBJS) return 0;
    uint64_t t0 = M.timed ? kern_now_ns() : 0;
    Obj *o = &f->obj[f->n];
    memset(o, 0, sizeof *o);
    o->key = key, o->next = -1;
    for (int b = 3; b <= 16; b++) {
        if (!(banks >> b & 1)) continue;
        const p_bind_buf *u = &ubo[b - 3];
        uint32_t n = (uint32_t)(u->size < max ? u->size : max);
        if (!u->addr || !n) continue;
        int64_t at = copy_block(u->addr, n);
        if (at < 0) return 0;   // a block the draw needs is missing: it keeps the camera's motion
        o->off[b - 3] = (uint32_t)at, o->size[b - 3] = n;
        if (!o->has_anchor && b != 3 && n >= 48) {   // the first matrix's translation: rows' fourth floats
            const float *m = (const float *)(M.map + (size_t)f->slot * SLOT_BYTES + at);
            o->anchor[0] = m[3], o->anchor[1] = m[7], o->anchor[2] = m[11], o->has_anchor = 1;
        }
    }
    uint64_t t1 = M.timed ? kern_now_ns() : 0;
    if (M.timed) M.ns[0] += t1 - t0;
    Ent *k = ent(f->keys, key, 1);
    if (!k) return 0;
    o->next = k->v, k->v = f->n++;
    M.objs++;
    // last frame's object: the same key, the nearest anchor
    Ent *lk = ent(l->keys, key, 0);
    const Obj *best = NULL;
    float bd = MAX_JUMP * MAX_JUMP;
    for (int32_t i = lk ? lk->v : -1; i >= 0; i = l->obj[i].next) {
        const Obj *c = &l->obj[i];
        float d = 0;
        if (o->has_anchor && c->has_anchor)
            for (int j = 0; j < 3; j++) d += (o->anchor[j] - c->anchor[j]) * (o->anchor[j] - c->anchor[j]);
        if (d <= bd) bd = d, best = c;
    }
    if (!best) return 0;
    M.matched++;
    const uint8_t *now = M.map + (size_t)f->slot * SLOT_BYTES, *was = M.map + (size_t)l->slot * SLOT_BYTES;
    for (int b = 3; b <= 16; b++) {
        int i = b - 3;
        if (!(banks >> b & 1) || !o->size[i] || best->size[i] != o->size[i]) continue;
        prev[b] = (VkDescriptorBufferInfo){M.buf, (VkDeviceSize)l->slot * SLOT_BYTES + best->off[i], best->size[i]};
        *have |= 1u << b;
        if (memcmp(now + o->off[i], was + best->off[i], o->size[i])) *changed |= 1u << b;
    }
    if (!(*changed & ~(1u << 3))) M.still++;
    if (M.timed) M.ns[1] += kern_now_ns() - t1;
    return 1;
}

void vk_mvprev_swap(void) {
    if (M.knob <= 0) return;
    Frame *t = M.f[1];
    M.f[1] = M.f[0], M.f[0] = t;
    // for the GPU's reads next frame; a no-op on coherent memory, but cached memory (this arena's) need not be coherent
    vmaFlushAllocation(vk_ctx()->vma, M.mem, (VkDeviceSize)M.f[1]->slot * SLOT_BYTES, M.f[1]->used);
    M.slot = (M.f[1]->slot + 1) % SLOTS;
    frame_reset(M.f[0], M.slot);
    if (++M.frames % SUMMARY) return;
    double fr = SUMMARY, ob = M.objs ? (double)M.objs : 1;
    rt_log("mv_prev: %.0f objects a frame, matched %.1f %%, new %.1f %%, still %.1f %%; %.0f KiB copied a frame (%.0f "
           "blocks)%s", M.objs / fr, 100 * M.matched / ob, 100 * (M.objs - M.matched) / ob, 100 * M.still / ob,
           M.bytes / fr / 1024, M.nblocks / fr, M.full ? "; history full in some frames" : "");
    if (M.timed) rt_log("mv_prev: %.3f ms a frame copying blocks, %.3f ms matching and comparing", M.ns[0] / 1e6 / fr,
                        M.ns[1] / 1e6 / fr);
    M.objs = M.matched = M.still = M.bytes = M.nblocks = M.full = 0, M.ns[0] = M.ns[1] = 0;
}
