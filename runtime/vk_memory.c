// Vulkan memory of the graphics layer: NVN memory pools as Vulkan
// buffers, the virtual-pool page table, GPU address resolution, page write generations for re-uploads, and the
// upload batch (command buffer, staging buffers, deferred destruction) that vk_texture.c records into.
//
// Behaviour ported from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/totk_vpool.c (request checks, unmap
// semantics, 64 KiB pages) and nvn_dirty.c (generation stamps: a texture is stale when a covered page carries a
// generation newer than its upload), @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile); and from LAS
// Recomp runtime/nvn/vulkan/exec.c:2094-2131 (vk_pages_changed: every write-watch query resets the OS bits, so each
// query stamps the pages it saw with a new generation shared by all users) and textures.c:240-270 (deferred
// destruction) @ da35b83a (MIT, Copyright (c) 2026 Peiikai). MNG aliases host VM (mach_vm_remap) and protects pages
// (fault handler); here virtual pools are a GPU-side page table over device-local memory and dirty pages come from
// GetWriteWatch (mem_written_pages), so neither is needed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "vk.h"
#include "platform/platform.h"

// Timestamps for the tracking cost, in ns; qpc_ms converts.
static uint64_t qpc(void) { return plat_now_ns(); }

static double qpc_ms(uint64_t ticks) { return (double)ticks / 1e6; }

// ---------------------------------------------------------------- pools
enum { POOL_IMPORT = 1, POOL_PHYS, POOL_VIRT };
enum { MS_NONE, MS_ON, MS_REFUSED };

typedef struct { struct PoolRec *phys; uint64_t off; } VPage;   // virtual page -> physical pool and offset

// Write-back span of a mirrored pool (gen_lock): the pool bytes [lo, hi), exact, that one copy region of a
// submission's write-back (vk_mem_mirror_end) moves from the mirror into the imported guest pages. fence is that
// submission's; purge_gen is 0 until its completion (wb_purge) stamped the span's written pages, then that stamp.
// lazy: whole pages a shader stored into that no copy moves: they stay in the mirror, held from the CPU at the fence
// (mirror_hold).
typedef struct { uint64_t lo, hi; uint32_t fence; uint32_t purge_gen; uint8_t lazy; } WbSpan;

typedef struct PoolRec {
    const pool_t *key;
    int kind;
    uint32_t flags;
    uint64_t lo, size;                 // GPU address range of the pool
    uint64_t import_lo, import_hi;     // POOL_IMPORT: the imported host range (aligned outwards)
    VkBuffer buf;
    VkDeviceMemory mem;                // POOL_IMPORT
    uint64_t local_bytes;              // POOL_IMPORT: the import's bytes when its memory type is on a DEVICE_LOCAL heap
    VmaAllocation alloc;               // POOL_PHYS
    VPage *pt;                         // POOL_VIRT: size / 64 KiB entries
    uint64_t npages;
    volatile int32_t used;                // resolved at least once: the GPU may still read it at finalize
    // AC_TRACE=dirty census (vk_mem_dirty_submit), replay thread under the submit lock only:
    uint32_t bound_sub;                // last submit that bound it as a buffer (dirty_sub), 0 never
    uint32_t synced_gen;               // page generation its lazy-mirror count covers; DIRTY_UNSYNCED until scanned
    uint8_t bufbound;                  // ever bound as a vertex, index, uniform, storage or indirect buffer
    uint8_t untracked;                 // the last dirty scan found no write watch over it
    // VRAM mirror (POOL_IMPORT, "VRAM mirror" below). mstate and mbuf change on the replay thread only.
    uint8_t mstate;                    // MS_NONE until the first buffer use, then MS_ON or MS_REFUSED
    VkBuffer mbuf;                     // MS_ON: device-local copy of [import_lo, import_hi), same offsets as buf
    VmaAllocation malloc_;
    uint32_t *mlive;                   // MS_ON, replay: per MCHUNK of the mirror, the last submission that bound it; 0:
                                       // the chunk is not in the mirror (mirror_cover)
    uint64_t mlive_bytes;              // replay: bytes of the chunks in the mirror
    uint32_t acc_sub;                  // replay: the last submission that synced the mirror (mirror_sync)
    uint32_t wb_sub;                   // replay: the last submission that wrote mirror bytes back to guest memory
    uint64_t wb_lo, wb_hi;             // pool offsets of that write-back's span union (ping-pong diagnostic: the next
                                       // sync's re-upload inside it is bytes the GPU just wrote, copied back uselessly)
    uint32_t wb_span_sub;              // replay: the submission [wb_lo, wb_hi) belongs to (copy_ranges)
    // Write-back spans not yet seen by a sync after their purge (gen_lock; mirror_sync, "write-back spans" there).
    WbSpan *wb;
    int nwb, cwb;
    uint32_t mgen;                     // gen_lock: every page write stamped up to this generation is in the mirror
                                       // (or in a copy queued before its next use)
    uint64_t fl_lo, fl_hi;             // flush_lock: nvnMemoryPoolFlushMappedRange bytes not copied yet (fl_hi 0: none)
    // Shadow storage (POOL_IMPORT in shadow mode, "shadow storage" below): buf is a host-visible copy of the guest pages.
    VmaAllocation salloc;
    uint8_t *smap;                     // its mapping; offsets as in buf (from import_lo)
    uint32_t sgen;                     // gen_lock: every page write stamped up to this generation is in the shadow
    volatile uint8_t stouched;         // on the touched list: the GPU may use it in the submission being recorded
} PoolRec;

static void mirror_forget(PoolRec *r);   // pool_lock exclusive: off the mirrored list
static int shadow_on;                    // shadow mode: imported pools get shadow storage instead (vk_mem_class_log)
static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size);
static void shadow_touch(PoolRec *r);
static void shadow_forget(PoolRec *r);
static void mirror_retiring(PoolRec *r);  // vk_pool_fini: kept until mirror_retire (or vk_mem_shutdown)
static void mirror_retire(void *arg);    // completion thread: destroys a finalized mirrored pool's record
static void grow(void **arr, int *cap, size_t elem);

static PlatMutex pool_lock = PLAT_MUTEX_INIT;
// Bumped under pool_lock (exclusive) by every change to the index or a page table: vk_mem_resolve's per-thread hits
// stay valid while it is unchanged.
static uint64_t pool_gen = 1;
// Bumped when a buffer use may get another buffer for the same span (a mirror created, refused or forgotten) and at
// every submission's start (mirror syncs are per submission). vk_mem_gen: both together.
static uint64_t mirror_gen;
static pidx pool_index;                // POOL_IMPORT, POOL_PHYS and POOL_VIRT records by GPU address
static PoolRec **phys_list;            // physical pools by key, for MapVirtual requests
static int nphys, cap_phys;
static int32_t n_imports, n_import_failures;
// Imported guest memory the driver counts on a DEVICE_LOCAL heap: on an integrated GPU, whose one heap is the system
// memory, the guest pools (about 2.5 GiB in play) sit inside the device-local usage the budget query reports; on a
// discrete GPU they import into the host heap and this stays 0. vk_mem_vram takes it out under AC_VRAM_LIMIT_MIB.
static int64_t import_local_bytes;
static PFN_vkGetMemoryHostPointerPropertiesEXT p_host_props;

static VkBufferUsageFlags pool_usage(const VkCtx *c) {
    VkBufferUsageFlags u = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (c->has_transform_feedback)
        u |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    if (c->has_conditional_rendering) u |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    return u;
}

static int memory_type(const VkCtx *c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if ((bits >> i & 1) && (c->mem_props.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if (bits >> i & 1) return (int)i;
    return -1;
}

// CPU-visible storage: the guest pages themselves become the buffer's memory (VK_EXT_external_memory_host).
static int import_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    if (shadow_on) return shadow_pool(c, r, addr, size);
    if (!p_host_props)
        p_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(c->device, "vkGetMemoryHostPointerPropertiesEXT");
    uint64_t al = c->min_imported_host_alignment ? c->min_imported_host_alignment : 0x1000;
    r->import_lo = addr & ~(al - 1);
    r->import_hi = (addr + size + al - 1) & ~(al - 1);
    uint64_t n = r->import_hi - r->import_lo;
    VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult res = p_host_props ? p_host_props(c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                               (void *)(uintptr_t)r->import_lo, &hp)
                                : VK_ERROR_EXTENSION_NOT_PRESENT;
    if (res != VK_SUCCESS || !hp.memoryTypeBits) {
        rt_log("vk: pool %#llx+%#llx: host pointer not importable (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        return 0;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = n,
                             .usage = pool_usage(c), .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if ((res = vkCreateBuffer(c->device, &bi, NULL, &r->buf)) != VK_SUCCESS) {
        rt_log("vk: pool %#llx: vkCreateBuffer for the import failed (%d)", (unsigned long long)addr, (int)res);
        return 0;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, r->buf, &req);
    int type = memory_type(c, req.memoryTypeBits & hp.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type < 0 || req.size > n) {
        rt_log("vk: pool %#llx: no memory type for the import (buffer %#x, host %#x, size %#llx of %#llx)",
               (unsigned long long)addr, req.memoryTypeBits, hp.memoryTypeBits, (unsigned long long)req.size,
               (unsigned long long)n);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        return 0;
    }
    VkImportMemoryHostPointerInfoEXT im = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                           .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                           .pHostPointer = (void *)(uintptr_t)r->import_lo};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &im, .allocationSize = n,
                               .memoryTypeIndex = (uint32_t)type};
    if ((res = vkAllocateMemory(c->device, &ai, NULL, &r->mem)) != VK_SUCCESS ||
        (res = vkBindBufferMemory(c->device, r->buf, r->mem, 0)) != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: host memory import failed (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->mem = VK_NULL_HANDLE;
        return 0;
    }
    uint32_t heap = c->mem_props.memoryTypes[type].heapIndex;
    if (c->mem_props.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        r->local_bytes = n;
        __atomic_add_fetch(&import_local_bytes, (int64_t)n, __ATOMIC_RELAXED);
    }
    plat_inc32(&n_imports);
    return 1;
}

// Physical pool: device-local memory of the pool size that both buffers and optimal-tiling images can be bound to.
static int alloc_physical(VkCtx *c, PoolRec *r) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = r->size, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult res = vkCreateBuffer(c->device, &bi, NULL, &r->buf);
    if (res != VK_SUCCESS) { rt_log("vk: physical pool: vkCreateBuffer failed (%d)", (int)res); return 0; }
    VkMemoryRequirements rb;
    vkGetBufferMemoryRequirements(c->device, r->buf, &rb);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {256, 256, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VkDeviceImageMemoryRequirements di = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &ii};
    VkMemoryRequirements2 ri = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(c->device, &di, &ri);
    uint32_t bits = rb.memoryTypeBits & ri.memoryRequirements.memoryTypeBits;
    if (!bits) bits = rb.memoryTypeBits;
    VkDeviceSize align = NVN_VPOOL_PAGE;
    if (rb.alignment > align) align = rb.alignment;
    if (ri.memoryRequirements.alignment > align) align = ri.memoryRequirements.alignment;
    VkMemoryRequirements req = {.size = (rb.size + align - 1) & ~(align - 1), .alignment = align, .memoryTypeBits = bits};
    VmaAllocationCreateInfo aci = {.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                   .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if ((res = vmaAllocateMemory(c->vma, &req, &aci, &r->alloc, NULL)) != VK_SUCCESS ||
        (res = vmaBindBufferMemory(c->vma, r->alloc, r->buf)) != VK_SUCCESS) {
        rt_log("vk: physical pool of %#llx bytes: device-local allocation failed (%d)", (unsigned long long)r->size, (int)res);
        if (r->alloc) vmaFreeMemory(c->vma, r->alloc);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->alloc = NULL;
        return 0;
    }
    if (vk_fix_zero()) {   // Switch pool memory starts zeroed; once per pool, not per image placed in it
        VkCommandBuffer cb = vk_upload_begin();
        vkCmdFillBuffer(cb, r->buf, 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(cb, &di);
        vk_upload_end();
    }
    return 1;
}

int vk_fix_zero(void) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("AC_FIX_ZERO");
    }
    return on;
}

static void index_put(PoolRec *r) {
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    pidx_ent e = {.lo = r->lo, .hi = r->lo + r->size, .base = r->lo, .buf = r, .gpu = r->lo, .key = r->key};
    pidx_remove(&pool_index, r->key, NULL);   // an object initialized again without Finalize
    if (!pidx_insert(&pool_index, &e)) rt_fatal("vk: out of memory for the pool index");
}

static PoolRec *phys_find(const pool_t *key) {
    for (int i = 0; i < nphys; i++) if (phys_list[i]->key == key) return phys_list[i];
    return NULL;
}

static void destroy_rec(VkCtx *c, PoolRec *r) {
    if (r->mbuf) vmaDestroyBuffer(c->vma, r->mbuf, r->malloc_);
    if (r->salloc) vmaDestroyBuffer(c->vma, r->buf, r->salloc), r->buf = VK_NULL_HANDLE;
    if (r->buf) vkDestroyBuffer(c->device, r->buf, NULL);
    if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
    if (r->local_bytes) __atomic_sub_fetch(&import_local_bytes, (int64_t)r->local_bytes, __ATOMIC_RELAXED);
    if (r->alloc) vmaFreeMemory(c->vma, r->alloc);
    free(r->pt);
    free(r->wb);
    free(r->mlive);
    free(r);
}

int vk_pool_init(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) { rt_log("vk: MemoryPoolInitialize before vk_init"); return 0; }
    // The GPU never sees a GPU_NO_ACCESS pool, except a physical one (0x189; 1.4.3 also 0x18c, CPU-visible) through the
    // virtual pools that map it. Either gets device-local memory: 1.4.3's CPU never writes its 0x18c pool's storage
    // (unchanged over 3600 frames of the intro), so the guest storage and the GPU copy need no sync.
    if ((p->flags & NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS) && !(p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL)) return 1;
    PoolRec *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("vk: out of memory for a pool record");
    r->key = p;
    r->flags = p->flags;
    r->lo = (uintptr_t)p->mem;
    r->size = p->size;
    int ok;
    if (p->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) {
        r->kind = POOL_VIRT;
        r->npages = p->size / NVN_VPOOL_PAGE;
        r->pt = calloc(r->npages ? r->npages : 1, sizeof *r->pt);
        ok = r->pt != NULL;
    } else if (p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) {
        r->kind = POOL_PHYS;
        ok = alloc_physical(c, r);
    } else {
        r->kind = POOL_IMPORT;
        mem_thaw((uintptr_t)p->mem, p->size);   // the import may need write access to the pages
        ok = import_pool(c, r, (uintptr_t)p->mem, p->size);
        if (!ok) {
            // Not fatal for the game: the pool works for the CPU, and vk_mem_resolve reports its addresses as unbacked.
            long n = plat_inc32(&n_import_failures);
            if (n <= 8) rt_log("vk: pool %p (flags %#x, %#llx bytes) has no GPU buffer (%ld so far)", (void *)p->mem,
                               p->flags, (unsigned long long)p->size, n);
            r->buf = VK_NULL_HANDLE;
            ok = 1;
        }
    }
    if (!ok) { free(r->pt); free(r); return 0; }
    plat_mutex_lock(&pool_lock);
    index_put(r);
    if (r->kind == POOL_PHYS) {
        if (nphys == cap_phys) {
            cap_phys = cap_phys ? cap_phys * 2 : 16;
            phys_list = realloc(phys_list, (size_t)cap_phys * sizeof *phys_list);
            if (!phys_list) rt_fatal("vk: out of memory for the physical pool list");
        }
        phys_list[nphys++] = r;
    }
    plat_mutex_unlock(&pool_lock);
    return 1;
}

static void need_pool_gone(uint64_t addr, uint64_t size);

void vk_pool_fini(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    pidx_ent e;
    plat_mutex_lock(&pool_lock);
    int had = pidx_remove(&pool_index, p, &e);
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    PoolRec *r = had ? e.buf : NULL;
    if (r && r->mstate == MS_ON) mirror_forget(r);
    if (r && r->kind == POOL_PHYS) {
        for (int i = 0; i < nphys; i++)
            if (phys_list[i] == r) { phys_list[i] = phys_list[--nphys]; break; }
        for (int i = 0; i < pool_index.n; i++) {   // virtual pages that still map it become unmapped
            PoolRec *v = pool_index.e[i].buf;
            if (v->kind != POOL_VIRT) continue;
            for (uint64_t k = 0; k < v->npages; k++)
                if (v->pt[k].phys == r) v->pt[k].phys = NULL;
        }
    }
    plat_mutex_unlock(&pool_lock);
    if (!r) return;
    if (r->smap) shadow_forget(r);
    if (r->kind == POOL_IMPORT) mem_thaw(r->lo, r->size);   // the storage goes back to the guest writable
    if (vk_need_on && r->kind == POOL_IMPORT) need_pool_gone(r->lo, r->size);
    // Images cut from this range go first; a used buffer or memory waits until the GPU is done with it, since the
    // guest gets the storage back when this returns.
    vk_texture_range_gone(r->lo, r->size, r->alloc);
    if (r->used && r->kind != POOL_VIRT) {
        vk_upload_flush(1);
        vk_queue_lock();
        vkQueueWaitIdle(c->gfx_queue);
        vk_queue_unlock();
    }
    vk_mem_collect();
    // A mirrored pool's record and buffers outlive the submission being recorded, whose sync copies and write-backs may
    // name them (the game frees a pool only once the GPU is done with it, but its CPU writes may still be queued).
    if (r->mbuf) mirror_retiring(r), vk_completion_add(vk_sync_issued() + 1, mirror_retire, r);
    else destroy_rec(c, r);
}

// Requests were checked by the front end (nvn_api_core.c, MNG totk_vpool.c); apply them in order.
int vk_pool_map_virtual(pool_t *v, int n, const void *requests) {
    const nvn_mapping_t *rq = requests;
    plat_mutex_lock(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)v->mem, 0);
    PoolRec *r = e && e->key == v ? e->buf : NULL;
    int ok = r && r->kind == POOL_VIRT;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; ok && i < n; i++) {
        PoolRec *ph = rq[i].physical_pool ? phys_find((const pool_t *)(const void *)rq[i].physical_pool) : NULL;
        if (rq[i].physical_pool && !ph) {
            rt_log("vk: MapVirtual request %d names physical pool %p that has no Vulkan memory", i,
                   (const void *)rq[i].physical_pool);
            ok = 0;
            break;
        }
        uint64_t first = (uint64_t)rq[i].virtual_offset / NVN_VPOOL_PAGE, count = rq[i].size / NVN_VPOOL_PAGE;
        if (first + count > r->npages) { ok = 0; break; }
        for (uint64_t k = 0; k < count; k++)
            r->pt[first + k] = (VPage){ph, ph ? (uint64_t)rq[i].physical_offset + k * NVN_VPOOL_PAGE : 0};
    }
    plat_mutex_unlock(&pool_lock);
    if (!ok) rt_log("vk: MapVirtual on pool %p failed", (void *)v->mem);
    return ok;
}

// Per-thread copies of resolved import and physical pools that overlap no other index entry (so pidx_find picks them for
// any range inside), valid while pool_gen is unchanged: the replay resolves a few dozen pools thousands of times per
// frame (16 vertex streams and every buffer binding of each draw). Direct-mapped by the 4 KiB page of the address, so
// a hit is one compare however many pools a frame touches. AC_RESOLVE_CHECK=1 compares every hit with the locked
// lookup and stops on a difference.
static int rec_class(const PoolRec *r) {
    if (r->kind != POOL_IMPORT) return VK_MCLS_DEVICE;
    return r->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? VK_MCLS_HOST_NOACC
         : r->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED ? VK_MCLS_HOST_CACHED : VK_MCLS_HOST_UNC;
}

typedef struct { uint64_t gen, lo, hi, base, end; VkBuffer buf; VmaAllocation alloc; int cls; void *pool; } ResolveHit;
#define RHIT_SLOTS 512
static _Thread_local ResolveHit rhit[RHIT_SLOTS];

static ResolveHit *rhit_slot(uint64_t addr) { return &rhit[(addr >> 12 ^ addr >> 21) & (RHIT_SLOTS - 1)]; }

static int resolve_check_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("AC_RESOLVE_CHECK");
        on = e && *e && strcmp(e, "0");
    }
    return on;
}

static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out);

int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out) {
    memset(out, 0, sizeof *out);
    uint64_t g = __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE), end = addr + size;
    const ResolveHit *h = rhit_slot(addr);
    if (end >= addr && h->gen == g && addr >= h->lo && end <= h->hi) {
        out->buffer = h->buf;
        out->offset = addr - h->base;
        out->avail = h->end - addr;
        out->alloc = h->alloc;
        out->cls = h->cls;
        out->pool = h->pool;
        if (h->alloc) out->alloc_offset = out->offset;
        if (shadow_on && h->pool && !((PoolRec *)h->pool)->stouched) shadow_touch(h->pool);
        if (resolve_check_on()) {
            VkMemSpan s;
            if (!resolve_locked(addr, size, &s) || memcmp(&s, out, sizeof s))
                rt_fatal("vk: resolve cache of 0x%llx differs from the pool index", (unsigned long long)addr);
        }
        return 1;
    }
    return resolve_locked(addr, size, out);
}

// The pool index lookup under the lock; a non-overlapped import or physical pool goes into the address's cache slot.
static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out) {
    memset(out, 0, sizeof *out);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)addr, size);
    PoolRec *r = e ? e->buf : NULL;
    int ok = 0;
    if (r && (r->kind == POOL_IMPORT ? r->buf != VK_NULL_HANDLE : r->kind == POOL_PHYS)) {
        int i = (int)(e - pool_index.e);
        if ((i == 0 || pool_index.maxhi[i - 1] <= e->lo) && (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= e->hi)) {
            ResolveHit *h = rhit_slot(addr);
            h->gen = __atomic_load_n(&pool_gen, __ATOMIC_RELAXED);   // stable: changed only under the exclusive lock
            h->lo = e->lo, h->hi = e->hi, h->buf = r->buf, h->cls = rec_class(r);
            if (r->kind == POOL_IMPORT) h->base = r->import_lo, h->end = r->import_hi, h->alloc = NULL, h->pool = r;
            else h->base = r->lo, h->end = r->lo + r->size, h->alloc = r->alloc, h->pool = NULL;
        }
    }
    if (r && r->kind == POOL_IMPORT && r->buf) {
        out->cls = rec_class(r);
        out->pool = r;
        out->buffer = r->buf;
        out->offset = addr - r->import_lo;
        out->avail = r->import_hi - addr;
        if (r->smap && !r->stouched) shadow_touch(r);
        ok = 1;
    } else if (r && r->kind == POOL_PHYS) {
        out->buffer = r->buf;
        out->offset = out->alloc_offset = addr - r->lo;
        out->avail = r->size - out->offset;
        out->alloc = r->alloc;
        out->cls = VK_MCLS_DEVICE;
        ok = 1;
    } else if (r && r->kind == POOL_VIRT) {
        uint64_t rel = addr - r->lo, k = rel / NVN_VPOOL_PAGE;
        if (k < r->npages && r->pt[k].phys) {
            PoolRec *ph = r->pt[k].phys;
            uint64_t off = r->pt[k].off + rel % NVN_VPOOL_PAGE, end = r->pt[k].off + NVN_VPOOL_PAGE;
            for (uint64_t j = k + 1; j < r->npages && r->pt[j].phys == ph && r->pt[j].off == end; j++) end += NVN_VPOOL_PAGE;
            out->buffer = ph->buf;
            out->offset = out->alloc_offset = off;
            out->avail = end - off;
            out->alloc = ph->alloc;
            out->cls = VK_MCLS_DEVICE;
            ok = out->avail >= size;
            if (ok && !ph->used) plat_xchg32(&ph->used, 1);   // read first: the flag is only ever set
        }
    }
    if (ok && !r->used) plat_xchg32(&r->used, 1);
    plat_mutex_unlock_shared(&pool_lock);
    if (!ok) memset(out, 0, sizeof *out);
    return ok;
}

uint64_t vk_mem_gen(void) {
    return __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE) + __atomic_load_n(&mirror_gen, __ATOMIC_ACQUIRE);
}

int vk_mem_class_of(uint64_t addr) {
    VkMemSpan m;
    return vk_mem_resolve(addr, 1, &m) ? m.cls : VK_MCLS_NONE;
}

// ---------------------------------------------------------------- census
static volatile int64_t traffic[VK_MCLS_N][VK_MUSE_N];

// Every caller is the replay (vk_exec*, under vk_sync_submit's lock), about ten adds per draw: one writer at a time, so
// a plain add, not a locked one. An add racing the census's reset may be lost.
void vk_mem_count(int cls, int use, uint64_t bytes) {
    if ((unsigned)cls < VK_MCLS_N && (unsigned)use < VK_MUSE_N)
        __atomic_store_n(&traffic[cls][use], __atomic_load_n(&traffic[cls][use], __ATOMIC_RELAXED) + (int64_t)bytes,
                         __ATOMIC_RELAXED);
}

static void flag_names(uint32_t f, char *buf, size_t n) {
    static const char *const names[] = {"CPU_NO_ACCESS", "CPU_UNCACHED", "CPU_CACHED", "GPU_NO_ACCESS", "GPU_UNCACHED",
                                        "GPU_CACHED", "SHADER_CODE", "COMPRESSIBLE", "PHYSICAL", "VIRTUAL"};
    size_t k = 0;
    buf[0] = 0;
    for (int i = 0; i < 10; i++)
        if (f >> i & 1) k += (size_t)snprintf(buf + k, k < n ? n - k : 0, "%s%s", k ? "|" : "", names[i]);
    if (!k) snprintf(buf, n, "none");
}

// One line per (kind, flags) group of the live pools, then the bytes the replay bound or copied per frame by pool
// class since the previous census (the first one: since frame 1200). Logged at frame 1800 (in the world from the
// run-game saves), at frame 600, and with AC_TRACE=perf every 1800 frames.
static void dirty_frame(uint64_t frame);

static void mirror_log(uint64_t frame);
static void shadow_log(uint64_t frame);
static int dead_pending(void);

static void vram_log(uint64_t frame);
static void vma_detail_log(void);

void vk_mem_census(uint64_t frame) {
    static uint64_t window_start;
    if ((rt_trace & RT_TRACE_PERF) && frame % 600 == 0) vram_log(frame);
    if ((rt_trace & RT_TRACE_DIRTY) && !vk_mirror_on) dirty_frame(frame);
    if ((rt_trace & RT_TRACE_PERF) ? frame % 60 == 0 : frame == 600 || frame == 1800)
        mirror_log(frame), shadow_log(frame);   // mirror_log also logs the write-watch line
    if (frame == 1200) {   // loading is over: the first window starts here
        for (int c = 0; c < VK_MCLS_N; c++)
            for (int u = 0; u < VK_MUSE_N; u++) plat_xchg64(&traffic[c][u], 0);
        window_start = frame;
        return;
    }
    // Frame 600 as well: runs on slow machines end before 1800.
    if (frame != 600 && (frame % 1800 || !(frame == 1800 || (rt_trace & RT_TRACE_PERF)))) return;
    typedef struct { int kind; uint32_t flags; uint64_t n, bytes; } Group;
    Group g[64];
    int ng = 0;
    uint64_t total[4] = {0}, count[4] = {0};
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        int j = 0;
        while (j < ng && (g[j].kind != r->kind || g[j].flags != r->flags)) j++;
        if (j == ng && ng < 64) g[ng++] = (Group){r->kind, r->flags, 0, 0};
        if (j < ng) g[j].n++, g[j].bytes += r->size;
        total[r->kind] += r->size, count[r->kind]++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    rt_log("mem: census at frame %llu: import %llu pools %.1f MiB (%d without a GPU buffer), physical %llu pools %.1f MiB, "
           "virtual %llu pools %.1f MiB of address space",
           (unsigned long long)frame, (unsigned long long)count[POOL_IMPORT], total[POOL_IMPORT] / 1048576.0,
           n_import_failures, (unsigned long long)count[POOL_PHYS], total[POOL_PHYS] / 1048576.0,
           (unsigned long long)count[POOL_VIRT], total[POOL_VIRT] / 1048576.0);
    // The commit a crash report prints, as a baseline: a crash far above it points at growth during play. A driver short
    // of memory faults or throws instead of failing an allocation, and the commit is what tells that apart.
    double proc, sys_used, sys_limit;
    plat_commit_info(&proc, &sys_used, &sys_limit);
    rt_log("mem: commit at frame %llu: this process %.0f MB, the system %.0f of %.0f MB", (unsigned long long)frame, proc,
           sys_used, sys_limit);
    // Per heap, what VMA holds (its blocks, and the allocations inside them) beside what the driver counts for the
    // process: usage above VMA's blocks is the imported guest pools (on an integrated GPU's device-local heap) and the
    // driver's own memory (pipelines, command buffers, descriptor pools).
    VkCtx *vc = vk_ctx();
    if (vc && vc->vma) {
        VmaBudget vb[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(vc->vma, vb);
        for (uint32_t h = 0; h < vc->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            rt_log("mem: heap %u (flags %#x) at frame %llu: vma %u blocks %.0f MiB, %u allocations %.0f MiB; driver usage "
                   "%.0f of %.0f MiB", h, vc->mem_props.memoryHeaps[h].flags, (unsigned long long)frame,
                   vb[h].statistics.blockCount, vb[h].statistics.blockBytes / 1048576.0, vb[h].statistics.allocationCount,
                   vb[h].statistics.allocationBytes / 1048576.0, vb[h].usage / 1048576.0, vb[h].budget / 1048576.0);
        uint64_t used, budget;
        int exact;
        int64_t imp = __atomic_load_n(&import_local_bytes, __ATOMIC_RELAXED);
        if (vk_mem_vram_limit() && vk_mem_vram(&used, &budget, &exact))
            rt_log("mem: video memory at frame %llu as AC_VRAM_LIMIT_MIB sees it: %.0f of %.0f MiB (%.0f MiB of imported "
                   "guest pools left out)", (unsigned long long)frame, used / 1048576.0, budget / 1048576.0,
                   imp / 1048576.0);
    }
    static const char *const kinds[] = {"", "import", "physical", "virtual"};
    for (int j = 0; j < ng; j++) {
        char f[160];
        flag_names(g[j].flags, f, sizeof f);
        rt_log("mem:   %s %#x %s: %llu pools, %.1f MiB", kinds[g[j].kind], g[j].flags, f, (unsigned long long)g[j].n,
               g[j].bytes / 1048576.0);
    }
    uint64_t frames = window_start && frame > window_start ? frame - window_start : frame;
    static const char *const cls[] = {"", "host CPU_NO_ACCESS", "host CPU_UNCACHED", "host CPU_CACHED", "device", "scratch"};
    rt_log("mem: traffic per frame over %llu frames, MiB (bound ranges, not bytes read): class: vertex index uniform "
           "storage indirect copy-src gpu-write cpu-read", (unsigned long long)frames);
    for (int c = 1; c < VK_MCLS_N; c++) {
        double v[VK_MUSE_N];
        for (int u = 0; u < VK_MUSE_N; u++) v[u] = plat_xchg64(&traffic[c][u], 0) / 1048576.0 / (double)(frames ? frames : 1);
        rt_log("mem:   %s: %.2f %.2f %.2f %.2f %.3f %.2f %.2f %.3f", cls[c], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
    window_start = frame;
}

// AC_VRAM_LIMIT_MIB: the port acts as if the card had at most this much video memory, so the paths written for small
// cards run on a large one or on an integrated GPU. Read once; the driver knows nothing of it, so nothing fails to
// allocate past it: what changes is every decision the port takes from the device's memory. vk_mem_vram's budget is at
// most the limit (the VRAM pressure sweep and over_budget in vk_texture.c, the texture cache's default bound, the
// mirror refusal in mirror_create), and the choices made from the card's size at start take the limit as the size
// (vk_astc.c auto_bc, boot.c settle_resolution). Empty or 0: no limit.
uint64_t vk_mem_vram_limit(void) {
    static int64_t limit = -1;   // bytes; -1 until read
    int64_t l = __atomic_load_n(&limit, __ATOMIC_RELAXED);
    if (l >= 0) return (uint64_t)l;
    const char *s = cfg_get("AC_VRAM_LIMIT_MIB");
    long long mb = s && *s ? strtoll(s, NULL, 10) : 0;
    l = mb > 0 ? (int64_t)mb << 20 : 0;
    int64_t unread = -1;
    if (__atomic_compare_exchange_n(&limit, &unread, l, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED) && l)
        rt_log("mem: video memory limited to %lld MiB (AC_VRAM_LIMIT_MIB): the budget and the size the port works from; "
               "the guest pools imported into a device-local heap (an integrated GPU's) count outside it, as they would "
               "on a discrete card", mb);
    return (uint64_t)l;
}

int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact) {
    VkCtx *c = vk_ctx();
    *used = *budget = 0, *exact = 0;
    if (!c) return 0;
    uint64_t limit = vk_mem_vram_limit();
    const VkPhysicalDeviceMemoryProperties *mp = &c->mem_props;
    if (c->has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
        for (uint32_t h = 0; h < p2.memoryProperties.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            if (p2.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                *used += b.heapUsage[h], *budget += b.heapBudget[h];
        *exact = 1;
        if (limit) {
            // The driver's usage holds the imported guest pools on an integrated GPU; a discrete card of the size
            // asked for keeps them in system memory, so they leave the usage the limit is held against.
            uint64_t imp = (uint64_t)__atomic_load_n(&import_local_bytes, __ATOMIC_RELAXED);
            *used = *used > imp ? *used - imp : 0;
            if (*budget > limit) *budget = limit;
        }
        return *budget != 0;
    }
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    for (uint32_t h = 0; h < mp->memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (mp->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            *used += vb[h].statistics.blockBytes, *budget += mp->memoryHeaps[h].size;
    if (limit && *budget > limit) *budget = limit;   // VMA's blocks hold no imports: the usage stays as it is
    return *budget != 0;
}

// ---------------------------------------------------------------- page write generations
// Two-level table over the guest address space (below 2^39): 256 MiB chunks of 65,536 page generations, allocated
// when a page of the chunk is first stamped. 0 = never stamped.
#define GEN_CHUNK_SHIFT 28
#define GEN_PAGE_SHIFT 12
#define GEN_CHUNKS (RT_ADDRESS_LIMIT >> GEN_CHUNK_SHIFT)
#define GEN_PER_CHUNK (1u << (GEN_CHUNK_SHIFT - GEN_PAGE_SHIFT))
static uint32_t *gen_chunk[GEN_CHUNKS];
static uint32_t gen_now;                 // last generation handed out
static PlatMutex gen_lock = PLAT_MUTEX_INIT;
static uint64_t gen_pages[1024];
// The newest generation stamped on any page of each 2 MiB region (the write-watch snapshot regions below): the mirror
// sync skips regions with nothing newer than its pool's mirror.
#define REG_SHIFT 21
static uint32_t reg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];
// GPU writes reported at the fence (vk_gpu_written) stamp a separate table on the same clock: the texture checks read
// both (a texture overlapping either is stale), while the mirror sync reads only the CPU table above. Stamping them
// together made every submission re-upload what the previous one wrote back: the write-back's own stamp looked like a
// CPU write the mirror was missing (about 49 MiB re-uploaded per frame for 37 MiB written back at the title).
static uint32_t *gpu_chunk[GEN_CHUNKS];
static uint32_t gpu_reg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];

// Every write-watch query of the generation table (gen_lock held exclusively): calls, bytes and time since the last
// mirror summary, all consumers together; the queries that found written pages (they also flush TLBs) apart.
static uint64_t ww_calls, ww_bytes, ww_ticks, ww_dirty_calls, ww_dirty_ticks, ww_pages;

static uint64_t ww_query(uint64_t addr, uint64_t size) {
    uint64_t t0 = qpc();
    uint64_t n = mem_written_pages(addr, size, gen_pages, sizeof gen_pages / sizeof *gen_pages);
    uint64_t dt = qpc() - t0;
    ww_ticks += dt, ww_calls++, ww_bytes += size;
    if (n && n != UINT64_MAX) ww_dirty_calls++, ww_dirty_ticks += dt, ww_pages += n;
    return n;
}

// Frozen regions: a 2 MiB region whose pages nothing wrote for freeze_idle submissions is made read-only
// (memory.c mem_freeze), and write-watch queries skip it until a write or a file read thaws it. Most of the pool memory
// the mirror and the texture checks cover is static (meshes, textures), and a query costs about 1 us per MiB (4-5 on an
// 8th-gen i7), so this keeps the queries to the regions the CPU writes. The region need not lie wholly in pools: NVN
// pools are small and not 2 MiB aligned, so most regions also hold other guest memory, and a rule that refused those
// left 340 of 436 queried MiB a frame unfrozen in town although nothing had written them for 300 submissions. The
// other memory is safe to freeze for the same reason pools are: the only kernel writes into guest memory are file
// reads, which thaw their buffer first (fs_host.c, romfs.c), and every user-mode write faults and thaws.
#define FREEZE_PER_SUB 32
static int freeze_idle;                       // 0: off
static uint32_t ww_ep = 1;                    // gen_lock: the submission being recorded (vk_mem_mirror_begin)
static uint32_t reg_last[RT_ADDRESS_LIMIT >> REG_SHIFT];    // gen_lock: ww_ep of the region's last stamp
static uint32_t reg_tried[RT_ADDRESS_LIMIT >> REG_SHIFT];   // gen_lock: ww_ep of its last freeze candidacy
static uint32_t *fcand;                       // gen_lock: regions to try at the next freeze pass
static int nfcand, cfcand;
static uint64_t n_frz_refused;                // gen_lock: candidates mem_freeze refused

static uint32_t *gen_slot(uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!gen_chunk[ci]) {
        if (!create) return NULL;
        gen_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!gen_chunk[ci]) rt_fatal("vk: out of memory for page generations");
    }
    return &gen_chunk[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

// gen_lock held exclusively; g is the newest generation handed out.
static void stamp_page(uint64_t a, uint32_t g) {
    uint32_t *s = gen_slot(a, 1);
    if (!s) return;
    *s = g;
    reg_max[a >> REG_SHIFT] = g;
    reg_last[a >> REG_SHIFT] = ww_ep;
}

static void stamp_range(uint64_t lo, uint64_t hi, uint32_t g) {
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_page(a, g);
}

static uint32_t *gpu_slot(uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!gpu_chunk[ci]) {
        if (!create) return NULL;
        gpu_chunk[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!gpu_chunk[ci]) rt_fatal("vk: out of memory for GPU page generations");
    }
    return &gpu_chunk[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

// gen_lock held exclusively; GPU writes never drive the region maxima the mirror sync and the freeze pass read.
static void stamp_gpu_page(uint64_t a, uint32_t g) {
    uint32_t *s = gpu_slot(a, 1);
    if (!s) return;
    *s = g;
    gpu_reg_max[a >> REG_SHIFT] = g;
}

static void stamp_gpu_range(uint64_t lo, uint64_t hi, uint32_t g) {
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_gpu_page(a, g);
}

// ---------------------------------------------------------------- needed pages (AC_WATCH_USED)
// With shadow storage on an integrated GPU, only the pages something reads through the generation table are watched:
// the ranges the replay binds or copies from as buffers (vk_mem_use, vk_mem_pick) and the texture storage the checks
// ask about (vk_mem_changed). Astral Chain keeps CPU-only data in its graphics pools, and its uniform blocks reach the
// GPU through the snapshots (vk_usnap.c), not the shadows: in the headquarters nearly all of the 2,400 pages it writes
// a frame are pages no buffer binding reads, and each took a write-protection fault (about 1.2 us on a Z1 Extreme at
// 15 W, 8.7% of the render thread) and a copy into the shadow for nothing. Watching the needed pages only: 2,360 to
// 145 faults a frame, 8 to 0.4 MiB copied, the shadow sync 2.7 to 0.6 ms and the replay 8.2 to 6.3 ms a frame,
// 56.9 to 59.8 fps (15 W, 4 cores / 8 threads, Steam Deck profile, camera turning, two runs each).
// A page is needed from its first use, for as long as its pool lives: need_mark stamps it then, so the next sync copies
// it and a texture check sees it changed. The kernel protects a page again only when a query with reset covers it, so a
// query peeks first (no reset: the pages nobody needs stay written and take no fault again) and resets only the runs
// of needed pages the peek found written. Writes are never lost: the reset is atomic with its report, and a page that
// becomes needed between two queries is reported written by the next one.
// AC_WATCH_USED=0 watches every page of the queried ranges, as before.
int vk_need_on;
static uint64_t *need_chunk[GEN_CHUNKS];                   // a bit per page; set and cleared under gen_lock, read without
static uint16_t reg_need[RT_ADDRESS_LIMIT >> REG_SHIFT];   // gen_lock: needed pages of the 2 MiB region
static uint32_t reg_arm[RT_ADDRESS_LIMIT >> REG_SHIFT];    // gen_lock: ww_ep of the region's last full reset (freeze_pass)
static uint64_t need_count;                                // gen_lock: needed pages
static uint64_t peek_pages[1024];                          // gen_lock

static int need_has(uint64_t a) {
    if (a >= RT_ADDRESS_LIMIT) return 0;
    const uint64_t *c = __atomic_load_n(&need_chunk[a >> GEN_CHUNK_SHIFT], __ATOMIC_ACQUIRE);
    uint64_t p = (a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1);
    return c && (__atomic_load_n(&c[p >> 6], __ATOMIC_RELAXED) >> (p & 63) & 1);
}

// 1 when every page of [lo, hi) (page aligned, below RT_ADDRESS_LIMIT) is needed; a word of the bitmap at a time.
static int need_all(uint64_t lo, uint64_t hi) {
    for (uint64_t a = lo; a < hi;) {
        const uint64_t *c = __atomic_load_n(&need_chunk[a >> GEN_CHUNK_SHIFT], __ATOMIC_ACQUIRE);
        if (!c) return 0;
        uint64_t p = (a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1), bit = p & 63, left = (hi - a) >> GEN_PAGE_SHIFT;
        uint64_t n = 64 - bit < left ? 64 - bit : left;
        uint64_t mask = (n == 64 ? ~0ull : (1ull << n) - 1) << bit;
        if ((__atomic_load_n(&c[p >> 6], __ATOMIC_RELAXED) & mask) != mask) return 0;
        a += n << GEN_PAGE_SHIFT;
    }
    return 1;
}

// gen_lock held exclusively: the pages of [lo, hi) (page aligned) become needed; the new ones are stamped as written.
static void need_mark(uint64_t lo, uint64_t hi) {
    uint32_t g = 0;
    if (hi > RT_ADDRESS_LIMIT) hi = RT_ADDRESS_LIMIT;
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) {
        if (need_has(a)) continue;
        uint64_t *c = need_chunk[a >> GEN_CHUNK_SHIFT];
        if (!c) {
            c = calloc(GEN_PER_CHUNK / 64, sizeof *c);
            if (!c) rt_fatal("vk: out of memory for the needed pages");
            __atomic_store_n(&need_chunk[a >> GEN_CHUNK_SHIFT], c, __ATOMIC_RELEASE);
        }
        uint64_t p = (a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1);
        __atomic_or_fetch(&c[p >> 6], 1ull << (p & 63), __ATOMIC_RELAXED);
        reg_need[a >> REG_SHIFT]++, need_count++;
        if (!g) g = ++gen_now;
        stamp_page(a, g);
    }
}

// The pages of [addr, addr+size) become needed. The test takes no lock: a bind of a range needed already, every bind
// but the first, costs a few loads.
static void need_range(uint64_t addr, uint64_t size) {
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    if (!size || lo >= RT_ADDRESS_LIMIT) return;
    if (hi > RT_ADDRESS_LIMIT) hi = RT_ADDRESS_LIMIT;
    if (need_all(lo, hi)) return;
    plat_mutex_lock(&gen_lock);
    need_mark(lo, hi);
    plat_mutex_unlock(&gen_lock);
}

// gen_lock held exclusively: the whole pages inside [addr, addr+size) are needed no more (their pool is gone).
static void need_clear(uint64_t addr, uint64_t size) {
    uint64_t lo = (addr + 0xfff) & ~0xfffull, hi = (addr + size) & ~0xfffull;
    if (hi > RT_ADDRESS_LIMIT) hi = RT_ADDRESS_LIMIT;
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) {
        if (!need_has(a)) continue;
        uint64_t p = (a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1);
        __atomic_and_fetch(&need_chunk[a >> GEN_CHUNK_SHIFT][p >> 6], ~(1ull << (p & 63)), __ATOMIC_RELAXED);
        reg_need[a >> REG_SHIFT]--, need_count--;
    }
}

static void need_pool_gone(uint64_t addr, uint64_t size) {
    plat_mutex_lock(&gen_lock);
    need_clear(addr, size);
    plat_mutex_unlock(&gen_lock);
}

// gen_lock held exclusively. Stamps the pages of [lo, hi) written since their last reset with g and resets them; with
// vk_need_on, the needed ones only. 0: the range is not write-watched.
static int ww_scan(uint64_t lo, uint64_t hi, uint32_t g) {
    const uint64_t max = sizeof gen_pages / sizeof *gen_pages;
    if (!vk_need_on) {
        for (;;) {
            uint64_t n = ww_query(lo, hi - lo);
            if (n == UINT64_MAX) return 0;
            for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g);
            if (n < max) return 1;   // a full answer may have left pages for the next call
        }
    }
    for (uint64_t at = lo; at < hi;) {
        uint64_t t0 = qpc();
        uint64_t n = mem_written_peek(at, hi - at, peek_pages, max);
        ww_ticks += qpc() - t0, ww_calls++, ww_bytes += hi - at;
        if (n == UINT64_MAX) return 0;
        for (uint64_t i = 0; i < n;) {
            uint64_t a = peek_pages[i++];
            if (!need_has(a)) continue;
            uint64_t b = a + (1ull << GEN_PAGE_SHIFT);
            while (i < n && peek_pages[i] == b && need_has(b)) b += 1ull << GEN_PAGE_SHIFT, i++;
            uint64_t m = ww_query(a, b - a);
            if (m == UINT64_MAX) return 0;
            for (uint64_t k = 0; k < m; k++) stamp_page(gen_pages[k], g);
        }
        if (n < max) break;
        at = peek_pages[n - 1] + (1ull << GEN_PAGE_SHIFT);
    }
    return 1;
}

void vk_guest_written(uint64_t addr, uint64_t size) {
    if (!size || nvn_gpu_only(addr)) return;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    stamp_range(lo, hi, ++gen_now);
    plat_mutex_unlock(&gen_lock);
}

// nvnBufferFlushMappedRange. Without shadow storage the range is stamped as a write, as before. With it, the write
// watch already reports every page the game wrote, at the snapshot of the submission that uses the range: the stamp
// only made the sync copy the pages a second time, a submission later. Astral Chain flushes whole buffers it rewrote
// (about 500 calls and 7.4 MiB a frame in the headquarters), and the copies into the shadows fell from about 17 to
// 8 MiB a frame without the stamp, the shadow sync from 3.7 to 2.6 ms a frame and the replay from 9.5 to 8.2 ms
// (Z1 Extreme at 15 W, 4 cores, camera turning; 56.4 to 57.5 fps). AC_FLUSH_STAMP=1 stamps as before; ab
// alternates with the AC_AB blocks.
void vk_guest_flushed(uint64_t addr, uint64_t size) {
    static int mode = -1;
    if (mode < 0) {
        mode = rt_knob("FLUSH_STAMP", shadow_on ? "0" : "1");
        if (shadow_on) rt_log("mem: buffer flushes %s (AC_FLUSH_STAMP)", mode == 2 ? "stamp their pages in the B blocks (AC_AB)"
                              : mode ? "stamp their pages as writes" : "stamp nothing: the write watch reports the writes");
    }
    if (RT_KNOB_ON(mode)) vk_guest_written(addr, size);
}

int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    if (vk_need_on) need_mark(lo, hi);   // before g: the caller keeps g, and a page needed from now on is older than it
    uint32_t g = ++gen_now;
    int tracked = 1, frozen = 1;
    for (uint64_t a = lo & ~((1ull << REG_SHIFT) - 1); a < hi && frozen; a += 1ull << REG_SHIFT) frozen = mem_frozen(a);
    if (!frozen) tracked = ww_scan(lo, hi, g);
    int changed = 0;
    for (uint64_t a = lo; tracked && a < hi && !changed; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 0);
        if (s && *s > since) changed = 1;
        else {
            s = gpu_slot(a, 0);
            if (s && *s > since) changed = 1;
        }
    }
    plat_mutex_unlock(&gen_lock);
    *now = g;
    return tracked ? changed : -1;
}

// Write-watch snapshots per submission: the guest address space in 2 MiB regions, each queried at most once per epoch
// (a guest submission), at the first check that covers it. A query costs about 1.3 us plus 0.55 us per MiB, so one per
// region instead of one per texture. The snapshot only adds observations to the generation table: a page written
// before it carries a generation newer than any check before it, and a page written after it keeps its OS bit until
// the next query over it (at the latest the next epoch's snapshot).
#define SNAP_SHIFT 21
#define SNAP_REGIONS (RT_ADDRESS_LIMIT >> SNAP_SHIFT)
static uint32_t snap_at[SNAP_REGIONS];      // epoch of the region's last snapshot; 0 never
static uint8_t snap_untracked[SNAP_REGIONS];
static uint64_t n_snap_checks, n_snap_calls, n_snap_untracked;

// gen_lock held exclusively. Queries the regions r0..r1 that were not snapped in epoch, one query per run of them;
// returns the number of queries.
static uint64_t snap_regions(uint64_t r0, uint64_t r1, uint32_t epoch) {
    uint32_t g = 0;
    uint64_t calls = 0;
    for (uint64_t r = r0; r <= r1;) {
        if (snap_at[r] == epoch) { r++; continue; }
        // Nothing in it changed since its freeze, or (vk_need_on) nothing in it is needed: a page that becomes
        // needed later in the epoch is stamped then (need_mark).
        if (mem_frozen(r << SNAP_SHIFT) || (vk_need_on && !reg_need[r])) {
            snap_at[r] = epoch, snap_untracked[r] = 0;
            r++;
            continue;
        }
        uint64_t k = r;   // one query over a run of regions not seen in this epoch
        while (k + 1 <= r1 && snap_at[k + 1] != epoch && !mem_frozen((k + 1) << SNAP_SHIFT) &&
               (!vk_need_on || reg_need[k + 1])) k++;
        uint64_t a = r << SNAP_SHIFT, b = (k + 1) << SNAP_SHIFT;
        if (!g) g = ++gen_now;
        int tracked = ww_scan(a, b, g);
        calls++;
        for (uint64_t j = r; j <= k; j++) {
            snap_at[j] = epoch, snap_untracked[j] = (uint8_t)!tracked;
            if (tracked && freeze_idle && ww_ep - reg_last[j] >= (uint32_t)freeze_idle &&
                ww_ep - reg_tried[j] >= (uint32_t)freeze_idle) {
                reg_tried[j] = ww_ep;
                if (nfcand == cfcand) grow((void **)&fcand, &cfcand, sizeof *fcand);
                fcand[nfcand++] = (uint32_t)j;
            }
        }
        r = k + 1;
    }
    return calls;
}

int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (!epoch || r1 >= SNAP_REGIONS) return vk_mem_changed(addr, size, since, now);
    if (vk_need_on) need_range(lo, hi - lo);
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t r = r0; r <= r1; r++)
        if (snap_at[r] != epoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        n_snap_calls += snap_regions(r0, r1, epoch);
    }
    int untracked = 0, changed = 0;
    for (uint64_t r = r0; r <= r1; r++) untracked |= snap_untracked[r];
    // Pages only in regions stamped after since (reg_max; REG_SHIFT is SNAP_SHIFT): an unchanged texture of 4 MiB was
    // 1,024 page reads, at about 1,100 texture checks a frame.
    for (uint64_t r = r0; !untracked && r <= r1 && !changed; r++) {
        if (reg_max[r] <= since && gpu_reg_max[r] <= since) continue;
        uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
        for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
            uint32_t *s = gen_slot(a, 0);
            if (s && *s > since) changed = 1;
            else {
                s = gpu_slot(a, 0);
                if (s && *s > since) changed = 1;
            }
        }
    }
    *now = gen_now;   // every observation so far is in the table
    __atomic_add_fetch(&n_snap_checks, 1, __ATOMIC_RELAXED);
    if (untracked) __atomic_add_fetch(&n_snap_untracked, 1, __ATOMIC_RELAXED);
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    return untracked ? -1 : changed;
}

void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked) {
    *checks = __atomic_exchange_n(&n_snap_checks, 0, __ATOMIC_RELAXED);
    *calls = __atomic_exchange_n(&n_snap_calls, 0, __ATOMIC_RELAXED);
    *untracked = __atomic_exchange_n(&n_snap_untracked, 0, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------- dirty census (AC_TRACE=dirty)
// A trace item of its own, not part of AC_TRACE=perf: its queries walk every buffer-bound pool at each submit, which
// under Linux's write watch reads all of their page table entries (9.8 ms a frame on an Arrow Lake GPU, 15.5 ms on a
// Steam Deck), so a perf log taken with it measured the census as much as the game.
// What a device-local mirror of the buffer-bound pools would upload. The replay notes every
// imported pool it binds as a vertex, index, uniform, storage or indirect buffer (vk_mem_bound_note). At the start of
// each submit one write-watch query per run of such pools stamps the pages written since with a new generation, in the
// shared generation table (no second consumer of the OS bits: texture checks see the stamps), and the pages of those
// pools are counted:
// - eager: pages stamped after the previous submit's scan, in every buffer-bound pool (upload at every submit);
// - lazy: pages stamped after the pool's previous count, in pools the previous submit bound (upload when bound);
// - first use: the size of a pool the first time it is bound (a mirror copies the whole pool then).
// Classes: CPU_CACHED, CPU_UNCACHED, other (CPU_NO_ACCESS). Logged as `mem: dirty` every 60 frames.
#define DIRTY_UNSYNCED UINT32_MAX
#define DIRTY_GAP 0x100000ull   // pools closer than this share one query (the gap's pages are stamped too)
#define DIRTY_WIN 60
enum { DC_CACHED, DC_UNC, DC_OTHER, DC_N };
typedef struct {
    uint64_t eager[DC_N], lazy[DC_N], first[DC_N];
    uint64_t submits, queries, scan_ns, sub_max, frames;
} DirtyAcc;
static uint32_t dirty_sub = 1, dirty_gen;   // submit lock: the current submit number, generation of the last scan
static uint64_t first_pending[DC_N], ever_n[DC_N], ever_bytes[DC_N];   // submit lock
static PoolRec **dlist;                     // submit lock: the buffer-bound pools of a scan
static int dcap;
static PlatMutex dirty_lock = PLAT_MUTEX_INIT;   // everything below
static DirtyAcc dfr, dwin;                  // this frame, this window
static uint32_t dwin_kib[DIRTY_WIN][DC_N];  // eager KiB of each frame of the window
static uint64_t live_n[DC_N], live_bytes[DC_N], live_untracked, ever_snap_n[DC_N], ever_snap_bytes[DC_N];

static int dc_class(const PoolRec *r) {
    int c = rec_class(r);
    return c == VK_MCLS_HOST_CACHED ? DC_CACHED : c == VK_MCLS_HOST_UNC ? DC_UNC : DC_OTHER;
}

void vk_mem_bound_note(void *pool) {
    PoolRec *r = pool;
    if (r->bound_sub == dirty_sub) return;
    r->bound_sub = dirty_sub;
    if (r->bufbound) return;
    r->bufbound = 1, r->synced_gen = DIRTY_UNSYNCED;
    int k = dc_class(r);
    ever_n[k]++, ever_bytes[k] += r->size, first_pending[k] += r->size;
}

static uint64_t page_lo(const PoolRec *r) { return r->lo & ~0xfffull; }
static uint64_t page_hi(const PoolRec *r) { return (r->lo + r->size + 0xfff) & ~0xfffull; }

// gen_lock held exclusively. 0: the range is not write-watched.
static int query_stamp(uint64_t lo, uint64_t hi, uint32_t g) { return ww_scan(lo, hi, g); }

// ---------------------------------------------------------------- shadow storage
// Where importing guest memory costs too much (Linux: re-protecting pages that amdgpu imported as a userptr
// invalidates the buffer, about 1.3 ms per 16 MiB at the next submit, and the import's write pinning clears the
// write watch), a CPU-visible pool gets a host-visible buffer of its own instead, at the offsets the import would have:
// - CPU -> shadow: right before each submit (vk_mem_shadow_sync), the pools the replay resolved since the last one are
//   queried for written pages; pages stamped after a pool's sgen are copied in by the CPU.
// - GPU -> guest: the replay notes each GPU write into CPU-visible memory at record time (vk_mem_gpu_pending) and
//   reports it at the fence (vk_gpu_written), which copies those bytes back into guest memory. Until then the CPU
//   copies skip them: a page the CPU and the GPU both write keeps the GPU's bytes.
// AC_SHADOW=1/0 forces the mode on or off; the default is on for Linux and Wine.
typedef struct { uint64_t lo, hi; } SRange;
static PlatMutex shadow_lock = PLAT_MUTEX_INIT;   // after pool_lock; before gen_lock
static PoolRec **stouch;                          // shadow_lock: pools resolved since the last sync
static int nstouch, cstouch;
static SRange *spend;                             // shadow_lock: GPU writes recorded and not copied back yet
static int nspend, cspend;
static int64_t s_live_n, s_live_bytes;            // shadow_lock
static uint64_t s_syncs, s_copied, s_queries, s_untracked, s_wb_bytes, s_held_bytes, s_ticks;   // shadow_lock: since the last log
static uint64_t s_qticks, s_seen;   // shadow_lock: time in the write-watch queries; regions already snapped in the epoch

static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    r->import_lo = addr & ~0xfffull;
    r->import_hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t n = r->import_hi - r->import_lo;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    // Cached: the fence copies (vk_gpu_written) read it back on the CPU, and reading uncached (write-combined) memory
    // runs at a few hundred MiB/s. Coherent where a cached type is (AMD, NVIDIA). Intel's integrated GPUs without a
    // shared LLC (Meteor Lake, Arrow Lake, Lunar Lake) offer cached memory only non-coherent, next to coherent
    // write-combined memory: requiring coherence there put the shadows in write-combined memory and the write-back
    // copies held shadow_lock for most of each frame (22-26 fps instead of 60 on Arrow Lake). Non-coherent memory is
    // flushed after the CPU copies in and invalidated before the CPU reads the GPU's writes; VMA skips both on
    // coherent memory. Without any cached type: coherent, as before.
    VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                  .preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    VkResult res = vmaCreateBuffer(c->vma, &bi, &ai, &r->buf, &r->salloc, &info);
    if (res != VK_SUCCESS) {
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ai.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        res = vmaCreateBuffer(c->vma, &bi, &ai, &r->buf, &r->salloc, &info);
    }
    if (res != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: shadow allocation failed (%d)", (unsigned long long)addr, (unsigned long long)size,
               (int)res);
        r->buf = VK_NULL_HANDLE, r->salloc = NULL;
        return 0;
    }
    r->smap = info.pMappedData;
    plat_mutex_lock(&gen_lock);
    r->sgen = gen_now;   // the copy below holds every write stamped so far; later stamps are copied again
    plat_mutex_unlock(&gen_lock);
    memcpy(r->smap, (const void *)(uintptr_t)r->import_lo, n);
    vmaFlushAllocation(c->vma, r->salloc, 0, n);
    plat_mutex_lock(&shadow_lock);
    s_live_n++, s_live_bytes += (int64_t)n;
    plat_mutex_unlock(&shadow_lock);
    plat_inc32(&n_imports);
    return 1;
}

static void shadow_touch(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    if (!r->stouched) {
        if (nstouch == cstouch) grow((void **)&stouch, &cstouch, sizeof *stouch);
        stouch[nstouch++] = r;
        r->stouched = 1;
    }
    plat_mutex_unlock(&shadow_lock);
}

// vk_pool_fini, after the pool left the index: off the touched list before its record goes.
static void shadow_forget(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    for (int i = 0; i < nstouch; i++)
        if (stouch[i] == r) { stouch[i] = stouch[--nstouch]; break; }
    s_live_n--, s_live_bytes -= (int64_t)(r->import_hi - r->import_lo);
    plat_mutex_unlock(&shadow_lock);
}

// shadow_lock held: copies guest [a, b) into the shadow, leaving out the bytes of GPU writes not copied back yet
// (spend from index i on) and the ranges this pool holds (gpu_written): the GPU stamps their pages, and reading them
// would fetch them back.
static void shadow_copy(PoolRec *r, uint64_t a, uint64_t b, int i) {
    for (; i < nspend; i++)
        if (spend[i].lo < b && spend[i].hi > a) {
            if (spend[i].lo > a) shadow_copy(r, a, spend[i].lo, i + 1);
            if (spend[i].hi < b) shadow_copy(r, spend[i].hi, b, i + 1);
            return;
        }
    uint64_t hlo, hhi;
    if (mem_held_by(a, b, r, &hlo, &hhi)) {   // the shadow has the newest bytes there, guest memory older ones
        if (hlo > a) shadow_copy(r, a, hlo, nspend);
        if (hhi < b) shadow_copy(r, hhi, b, nspend);
        return;
    }
    memcpy(r->smap + (a - r->import_lo), (const void *)(uintptr_t)a, b - a);
    vmaFlushAllocation(vk_ctx()->vma, r->salloc, a - r->import_lo, b - a);   // non-coherent memory only (shadow_pool)
    s_copied += b - a;
}

// AC_SHADOW_SNAP (default 1; ab alternates with the AC_AB blocks): a pool's sync takes its written pages from the
// per-submission write-watch snapshots (snap_regions, shared with the texture checks and the VRAM mirror): each 2 MiB
// region is queried at most once per guest submission (vk_texture_new_submit), however many pools share it and however
// many parts of the submission sync them, and a shadow region left unwritten for AC_FREEZE_IDLE submissions becomes a
// freeze candidate, so later queries skip it. The GPU then reads a pool as of the first part that syncs it, nearer the
// submit call, the moment the game means (vk_usnap.c); a guest write after that snapshot keeps its OS bit and is copied
// at the next submission's. 0 queries each pool at every sync, as before. After TotK Recomp's vk_memory.c
// shadow_sync_pool (MIT; THIRD_PARTY_NOTICES.md).
static int shadow_snap = -1;
static void shadow_copy_newer(PoolRec *r, uint32_t since);

// shadow_lock and gen_lock held: stamps the pool's written pages and copies those newer than its shadow.
static void shadow_sync_pool(PoolRec *r) {
    uint64_t lo = r->import_lo, hi = r->import_hi;
    if (shadow_snap < 0) {
        shadow_snap = rt_knob("SHADOW_SNAP", "1");
        rt_log("mem: shadow sync: %s (AC_SHADOW_SNAP)", shadow_snap == 2 ? "alternating with the AC_AB blocks"
               : shadow_snap ? "per-submission region snapshots" : "a query per pool at every part");
    }
    uint32_t epoch = RT_KNOB_ON(shadow_snap) ? vk_texture_epoch() : 0;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (epoch && r1 < SNAP_REGIONS) {
        uint64_t q0 = qpc();
        for (uint64_t k = r0; k <= r1; k++) s_seen += snap_at[k] == epoch;
        s_queries += snap_regions(r0, r1, epoch);   // stamps what the snapshot found, if the region was not seen yet
        s_qticks += qpc() - q0;
        int untracked = 0;
        for (uint64_t k = r0; k <= r1; k++) untracked |= snap_untracked[k];
        uint32_t since = r->sgen;
        r->sgen = gen_now;
        if (untracked) {   // nothing tells which pages changed: all of them
            s_untracked++;
            shadow_copy(r, lo, hi, 0);
            return;
        }
        shadow_copy_newer(r, since);
        return;
    }
    uint64_t q0 = qpc();
    uint32_t g = ++gen_now;
    int tracked = 1;
    for (uint64_t a = lo; a < hi;) {   // one query per run of regions that are not frozen
        uint64_t b = (a | ((1ull << REG_SHIFT) - 1)) + 1;
        if (b > hi) b = hi;
        if (mem_frozen(a)) { a = b; continue; }
        while (b < hi && !mem_frozen(b)) b = b + (1ull << REG_SHIFT) < hi ? b + (1ull << REG_SHIFT) : hi;
        s_queries++;
        if (!query_stamp(a, b, g)) tracked = 0;
        a = b;
    }
    s_qticks += qpc() - q0;
    uint32_t since = r->sgen;
    r->sgen = gen_now;
    if (!tracked) {   // nothing tells which pages changed: all of them
        s_untracked++;
        shadow_copy(r, lo, hi, 0);
        return;
    }
    shadow_copy_newer(r, since);
}

// shadow_lock and gen_lock held: copies the pool's pages stamped after since, by the CPU or the GPU domain, into its
// shadow, in runs.
static void shadow_copy_newer(PoolRec *r, uint32_t since) {
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t run = 0;
    for (uint64_t k = lo >> REG_SHIFT; k <= (hi - 1) >> REG_SHIFT; k++) {
        uint64_t a = k << REG_SHIFT, b = a + (1ull << REG_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        const uint32_t *gh = gpu_chunk[a >> GEN_CHUNK_SHIFT];
        if ((reg_max[k] <= since || !ch) && (gpu_reg_max[k] <= since || !gh)) {
            if (run) shadow_copy(r, run, a, 0), run = 0;
            continue;
        }
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            int dirty = ch && ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since;
            if (!dirty && gh) dirty = gh[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since;
            if (dirty && !run) run = x;
            else if (!dirty && run) shadow_copy(r, run, x, 0), run = 0;
        }
    }
    if (run) shadow_copy(r, run, hi, 0);
}

void vk_mem_shadow_sync(void) {
    if (!shadow_on) return;
    uint64_t t0 = qpc();
    plat_mutex_lock(&shadow_lock);
    if (nstouch) {
        plat_mutex_lock(&gen_lock);
        for (int i = 0; i < nstouch; i++) shadow_sync_pool(stouch[i]), stouch[i]->stouched = 0;
        plat_mutex_unlock(&gen_lock);
        s_syncs += (uint64_t)nstouch;
        nstouch = 0;
    }
    s_ticks += qpc() - t0;
    plat_mutex_unlock(&shadow_lock);
}

void vk_mem_gpu_pending(uint64_t addr, uint64_t size) {
    if (!shadow_on || !size) return;
    plat_mutex_lock(&shadow_lock);
    if (nspend == cspend) grow((void **)&spend, &cspend, sizeof *spend);
    spend[nspend++] = (SRange){addr, addr + size};
    plat_mutex_unlock(&shadow_lock);
}

// memory.c fetch for a held range (gpu_written): the shadow's bytes, which the GPU wrote. No lock: the pool's shadow
// outlives its holds (vk_pool_fini's mem_thaw fetches them back first).
static void shadow_fetch(void *ctx, uint64_t a, uint64_t n, void *dst) {
    PoolRec *r = ctx;
    vmaInvalidateAllocation(vk_ctx()->vma, r->salloc, a - r->import_lo, n);   // non-coherent memory only
    memcpy(dst, r->smap + (a - r->import_lo), n);
}

// shadow_lock held: copies the GPU's bytes of [a, b) from the shadow into guest memory.
static void write_back(PoolRec *r, uint64_t a, uint64_t b) {
    if (a >= b) return;
    mem_thaw(a, b - a);
    plat_mem_will_write(a, b - a);   // one call instead of a write-watch fault per page
    // Non-coherent memory only (shadow_pool): drop the CPU's cached lines so the GPU's bytes are read.
    vmaInvalidateAllocation(vk_ctx()->vma, r->salloc, a - r->import_lo, b - a);
    memcpy((void *)(uintptr_t)a, r->smap + (a - r->import_lo), b - a);
    s_wb_bytes += b - a;
}

// hold: the whole pages of a range of HOLD_MIN and up stay in the shadow (mem_hold) until the CPU touches them, and the
// partial pages at its ends are copied back. The GPU's storage buffers of the title menu (light lists and the like,
// 36 MiB a frame) are never read by the CPU: copying them back at each fence cost about 6.5 ms a frame on the
// completion thread, which every fence wait and every submission (shadow_lock) waited for. A range still held from
// an earlier write costs nothing.
#define HOLD_MIN (1ull << 20)
static void gpu_written(uint64_t addr, uint64_t size, int hold) {
    if (shadow_on && size) {
        plat_mutex_lock_shared(&pool_lock);
        const pidx_ent *e = pidx_find(&pool_index, addr, size);
        PoolRec *r = e ? e->buf : NULL;
        plat_mutex_lock(&shadow_lock);
        if (r && r->smap && addr >= r->import_lo && addr + size <= r->import_hi) {
            uint64_t g = plat_mem_grain(), lo = (addr + g - 1) & ~(g - 1), hi = (addr + size) & ~(g - 1);   // whole host pages
            if (hold && hi > lo && hi - lo >= HOLD_MIN && mem_hold(lo, hi - lo, shadow_fetch, r)) {
                write_back(r, addr, lo);
                write_back(r, hi, addr + size);
                s_held_bytes += hi - lo;
            } else {
                write_back(r, addr, addr + size);
            }
        }
    for (int i = 0; i < nspend; i++)
            if (spend[i].lo == addr && spend[i].hi == addr + size) { spend[i] = spend[--nspend]; break; }
        plat_mutex_unlock(&shadow_lock);
        plat_mutex_unlock_shared(&pool_lock);
    }
    if (size && !nvn_gpu_only(addr)) {   // the GPU domain: textures overlapping it re-upload, the mirror does not
        uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
        plat_mutex_lock(&gen_lock);
        stamp_gpu_range(lo, hi, ++gen_now);
        plat_mutex_unlock(&gen_lock);
    }
}

void vk_gpu_written(uint64_t addr, uint64_t size) { gpu_written(addr, size, 0); }
void vk_gpu_stored(uint64_t addr, uint64_t size) { gpu_written(addr, size, 1); }

static void shadow_log(uint64_t frame) {
    if (!shadow_on) return;
    plat_mutex_lock(&shadow_lock);
    int held_now;
    uint64_t holds, fetched;
    mem_hold_stats(&held_now, &holds, &fetched);
    rt_log("mem: shadow at frame %llu: %lld pools %.1f MiB; since the last: %llu pool syncs, %llu queries, %.1f MiB "
           "copied in (%llu untracked), %.1f MiB written back, %.1f MiB left in the shadow (%d ranges held, %llu new, "
           "%llu fetched back), %.2f ms (%.2f ms in queries; %llu regions already snapped), %d GPU writes pending, "
           "%.1f MiB of pages watched%s",
           (unsigned long long)frame, (long long)s_live_n, s_live_bytes / 1048576.0, (unsigned long long)s_syncs,
           (unsigned long long)s_queries, s_copied / 1048576.0, (unsigned long long)s_untracked, s_wb_bytes / 1048576.0,
           s_held_bytes / 1048576.0, held_now, (unsigned long long)holds, (unsigned long long)fetched, qpc_ms(s_ticks),
           qpc_ms(s_qticks), (unsigned long long)s_seen, nspend,
           vk_need_on ? __atomic_load_n(&need_count, __ATOMIC_RELAXED) * 4096 / 1048576.0 : s_live_bytes / 1048576.0,
           vk_need_on ? " (AC_WATCH_USED)" : "");
    s_syncs = s_queries = s_copied = s_untracked = s_wb_bytes = s_held_bytes = s_ticks = s_qticks = s_seen = 0;
    plat_mutex_unlock(&shadow_lock);
}

void vk_mem_dirty_submit(void) {
    if (!(rt_trace & RT_TRACE_DIRTY) || vk_mirror_on) return;   // the mirror's own line replaces the census
    uint64_t t0 = kern_now_ns();
    uint32_t prev = dirty_sub;   // the submit whose replay noted the bindings
    dirty_sub = prev + 1 ? prev + 1 : 1;
    uint64_t eager[DC_N] = {0}, lazy[DC_N] = {0}, ln[DC_N] = {0}, lb[DC_N] = {0}, untracked = 0, queries = 0;
    plat_mutex_lock_shared(&pool_lock);
    if (dcap < pool_index.n) {
        dcap = pool_index.n * 2;
        dlist = realloc(dlist, (size_t)dcap * sizeof *dlist);
        if (!dlist) rt_fatal("vk: out of memory for the dirty census");
    }
    int n = 0;
    for (int i = 0; i < pool_index.n; i++) {
        PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT && r->bufbound) dlist[n++] = r;
    }
    plat_mutex_lock(&gen_lock);
    uint32_t g = ++gen_now, since = dirty_gen;
    dirty_gen = g;
    for (int i = 0; i < n;) {   // one query per run of pools (sorted by address)
        uint64_t lo = page_lo(dlist[i]), hi = page_hi(dlist[i]);
        int j = i + 1;
        for (; j < n && page_lo(dlist[j]) <= hi + DIRTY_GAP; j++)
            if (page_hi(dlist[j]) > hi) hi = page_hi(dlist[j]);
        queries++;
        int ok = query_stamp(lo, hi, g);
        for (int k = i; k < j; k++) {
            dlist[k]->untracked = 0;
            if (!ok) queries++, dlist[k]->untracked = (uint8_t)!query_stamp(page_lo(dlist[k]), page_hi(dlist[k]), g);
        }
        i = j;
    }
    uint64_t covered = 0;
    for (int i = 0; i < n; i++) {
        PoolRec *r = dlist[i];
        int k = dc_class(r);
        ln[k]++, lb[k] += r->size;
        int fresh = r->synced_gen == DIRTY_UNSYNCED, lz = !fresh && r->bound_sub == prev;
        uint32_t ls = r->synced_gen;
        if (fresh || r->bound_sub == prev) r->synced_gen = g;
        if (r->untracked) { untracked += r->size; continue; }
        if (fresh) continue;   // counted as first use
        uint64_t lo = page_lo(r), hi = page_hi(r), ne = 0, nl = 0;
        if (lo < covered) lo = covered;   // overlapping pools: each page once
        if (lo >= hi) continue;
        covered = hi;
        for (uint64_t a = lo; a < hi;) {
            uint64_t ci = a >> GEN_CHUNK_SHIFT, ce = (ci + 1) << GEN_CHUNK_SHIFT;
            if (ce > hi) ce = hi;
            const uint32_t *ch = ci < GEN_CHUNKS ? gen_chunk[ci] : NULL;
            const uint32_t *gh = ci < GEN_CHUNKS ? gpu_chunk[ci] : NULL;
            if (ch || gh)
                for (uint64_t x = a; x < ce; x += 1ull << GEN_PAGE_SHIFT) {
                    uint32_t v = ch ? ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] : 0;
                    uint32_t w = gh ? gh[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] : 0;
                    if (w > v) v = w;
                    ne += v > since;
                    nl += lz && v > ls;
                }
            a = ce;
        }
        eager[k] += ne << GEN_PAGE_SHIFT, lazy[k] += nl << GEN_PAGE_SHIFT;
    }
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    uint64_t tot = 0;
    for (int k = 0; k < DC_N; k++) tot += eager[k];
    plat_mutex_lock(&dirty_lock);
    for (int k = 0; k < DC_N; k++) {
        dfr.eager[k] += eager[k], dfr.lazy[k] += lazy[k], dfr.first[k] += first_pending[k];
        first_pending[k] = 0;
        live_n[k] = ln[k], live_bytes[k] = lb[k], ever_snap_n[k] = ever_n[k], ever_snap_bytes[k] = ever_bytes[k];
    }
    live_untracked = untracked;
    dfr.submits++, dfr.queries += queries;
    if (tot > dfr.sub_max) dfr.sub_max = tot;
    dfr.scan_ns += kern_now_ns() - t0;
    plat_mutex_unlock(&dirty_lock);
}

static void dirty_frame(uint64_t frame) {
    plat_mutex_lock(&dirty_lock);
    if (dwin.frames < DIRTY_WIN)
        for (int k = 0; k < DC_N; k++) dwin_kib[dwin.frames][k] = (uint32_t)(dfr.eager[k] >> 10);
    for (int k = 0; k < DC_N; k++) dwin.eager[k] += dfr.eager[k], dwin.lazy[k] += dfr.lazy[k], dwin.first[k] += dfr.first[k];
    dwin.submits += dfr.submits, dwin.queries += dfr.queries, dwin.scan_ns += dfr.scan_ns, dwin.frames++;
    if (dfr.sub_max > dwin.sub_max) dwin.sub_max = dfr.sub_max;
    memset(&dfr, 0, sizeof dfr);
    if (frame % DIRTY_WIN) { plat_mutex_unlock(&dirty_lock); return; }
    DirtyAcc w = dwin;
    uint32_t kib[DIRTY_WIN][DC_N];
    memcpy(kib, dwin_kib, sizeof kib);
    uint64_t lv[DC_N], lvb[DC_N], ev[DC_N], evb[DC_N], unt = live_untracked;
    for (int k = 0; k < DC_N; k++) lv[k] = live_n[k], lvb[k] = live_bytes[k], ev[k] = ever_snap_n[k], evb[k] = ever_snap_bytes[k];
    memset(&dwin, 0, sizeof dwin);
    plat_mutex_unlock(&dirty_lock);
    if (!w.frames) return;
    uint64_t fl[2][3][2];
    nvn_flush_stats(fl);
    double f = (double)w.frames, M = 1048576.0;
    uint64_t sub_tot = 0;
    for (int k = 0; k < DC_N; k++) sub_tot += w.eager[k];
#define C3(a, d) (a)[0] / (d), (a)[1] / (d), (a)[2] / (d)
    rt_log("mem: dirty frame %llu: frames %llu submits_per_frame %.2f; per frame, cached/uncached/other: eager_mib "
           "%.3f/%.3f/%.3f lazy_mib %.3f/%.3f/%.3f first_mib %.3f/%.3f/%.3f flush_pool_calls %.1f/%.1f/%.1f "
           "flush_pool_mib %.3f/%.3f/%.3f flush_buf_calls %.1f/%.1f/%.1f flush_buf_mib %.3f/%.3f/%.3f; per submit: "
           "avg_mib %.3f max_mib %.3f; mirror: live_pools %llu/%llu/%llu live_mib %.1f/%.1f/%.1f ever_pools %llu/%llu/%llu "
           "ever_mib %.1f/%.1f/%.1f untracked_mib %.1f; cost: queries_per_frame %.1f scan_ms_per_frame %.3f",
           (unsigned long long)frame, (unsigned long long)w.frames, w.submits / f,
           C3(w.eager, f * M), C3(w.lazy, f * M), C3(w.first, f * M),
           fl[0][0][0] / f, fl[0][1][0] / f, fl[0][2][0] / f, fl[0][0][1] / f / M, fl[0][1][1] / f / M, fl[0][2][1] / f / M,
           fl[1][0][0] / f, fl[1][1][0] / f, fl[1][2][0] / f, fl[1][0][1] / f / M, fl[1][1][1] / f / M, fl[1][2][1] / f / M,
           w.submits ? sub_tot / (double)w.submits / M : 0.0, w.sub_max / M,
           (unsigned long long)lv[0], (unsigned long long)lv[1], (unsigned long long)lv[2], C3(lvb, M),
           (unsigned long long)ev[0], (unsigned long long)ev[1], (unsigned long long)ev[2], C3(evb, M), unt / M,
           w.queries / f, w.scan_ns / 1e6 / f);
#undef C3
    char line[DIRTY_WIN * 36 + 64];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: dirty frames KiB cached/uncached/other:");
    for (uint64_t i = 0; i < w.frames && i < DIRTY_WIN; i++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, " %u/%u/%u", kib[i][0], kib[i][1], kib[i][2]);
    rt_log("%s", line);
}

// ---------------------------------------------------------------- VRAM mirror of buffer-bound pools
// An imported pool keeps vertices, indices, constants
// and storage buffers in system memory, so a discrete GPU reads them over PCIe on every draw (GTX 1050 Ti and RTX
// 3090 Ti: about 110 ms of GPU time per frame in the field). In mirror mode (device class discrete, AC_MEMORY) an
// imported pool gets a device-local buffer over its imported range the first time the replay uses it as a vertex,
// index, uniform, storage or indirect buffer (vk_mem_use). From then on those bindings, and the buffer copies, fills
// and uniform updates that touch the pool (vk_mem_pick), use the mirror at the same offsets. Guest memory stays what
// the CPU reads and writes; textures and every host path that reads or writes pool bytes keep using it.
// - Chunks: the mirror holds the pool's bytes only where the replay bound it. Each MCHUNK (64 KiB) of the mirror is
//   in it or not (PoolRec.mlive); a bind, or a copy, fill or update that writes, brings the chunks of its range in with
//   one copy each (mirror_cover), and a copy that only reads a range with no chunk in the mirror reads the imported
//   buffer, which holds those bytes. A game pool is mostly textures and CPU data beside its buffers: over a save load
//   and a fight at 1920x1080 the five mirrors spanned 1980 MiB of which 90 MiB was ever bound, and of the 17 MiB a
//   frame of written pages only 0.8 MiB lay in bound chunks. A chunk no submission bound for MIRROR_AGE submissions
//   leaves the mirror when the CPU writes it again. AC_MIRROR_CHUNKS=0 puts the whole pool in at the first use and
//   keeps it, as before.
// - CPU -> mirror: the first access of a submission to a mirror syncs it (mirror_sync): the 2 MiB write-watch
//   snapshots of its regions for this epoch (snap_regions, shared with the texture checks: the OS bits keep one
//   consumer, the shared page generations) and every page stamped after the mirror's mgen that lies in a chunk of
//   the mirror goes into a copy; so do the nvnMemoryPoolFlushMappedRange ranges. The copies run in a prologue command
//   buffer that vk_sync.c submits right before the part that accessed the mirror first, between full barriers.
//   vk_mem_mirror_begin snaps the regions of the pools the previous submission accessed in a few merged queries.
// - GPU -> guest: what the replay writes into a mirror (storage buffers its shaders write, buffer copies, fills,
//   uniform updates, texture-to-buffer copies: vk_mem_gpu_wrote) goes back into the imported buffer at the end of the
//   submission, exact byte ranges, before the fence signals (vk_mem_mirror_end).
// - Lazy write-back: the whole pages of a range of HOLD_MIN and up that a shader stores into are not copied back.
//   The game's large storage buffers (a per-pixel light list, two simulation buffers: 36.7 MiB a frame at 1920x1080
//   with the simulation running) are never read or written by the CPU, and copying them from video memory to system
//   memory every frame is the mirror's largest traffic on the bus. At the submission's fence the pages become
//   inaccessible to the CPU (memory.c held ranges, mirror_hold); the first access, by the guest or the host, reads
//   the range back from the mirror once (mirror_fetch) and the range is copied back at each submission again for
//   LAZY_BACKOFF submissions, doubling with each further access. The partial pages at the range's ends, and every
//   copy, fill and update, keep the copy at the end of the submission. AC_MIRROR_LAZY=0 copies everything back.
// - AC_MEMORY=verify: every submission also copies up to VERIFY_BYTES of the mirrors' chunks, round robin, into host
//   memory; at its fence the pages the CPU has not written since their sync are compared with guest memory.
// A pool stays imported when its mirror would not fit the VRAM budget, its range is not write-watched, it overlaps
// another pool or the allocation fails.
#define VERIFY_BYTES (4u << 20)
#define MCHUNK_SHIFT 16
#define MCHUNK (1ull << MCHUNK_SHIFT)
#define MIRROR_AGE 600u               // submissions without a bind before a chunk the CPU writes leaves the mirror
#define VERIFY_SAMPLES 64
#define VERIFY_LOG 16
enum { MR_BUDGET, MR_UNTRACKED, MR_OVERLAP, MR_ALLOC, MR_N };

int vk_mirror_on;
static int mirror_verify;
static int mirror_gap = 2;             // pre-snap: hot pools at most this many 2 MiB regions apart share one query
static int mirror_chunks = 1;          // AC_MIRROR_CHUNKS: 0 keeps every chunk of a mirror in it from its first use
static int mirror_lazy = 1;            // AC_MIRROR_LAZY: 0 copies every GPU write back at the end of its submission
static int lazy_all;                   // probes (vk_mem_lazy_all): fills and copies are left in the mirror too
static uint64_t verify_bytes = VERIFY_BYTES;   // AC_VERIFY_MIB: what a submission copies out in verify mode
static PoolRec **mlist;                // pool_lock: mirrored pools by import_lo
static int nm, mcap;
static uint32_t msub;                  // replay: number of the submission being recorded
static uint32_t mepoch;                // replay: its write-watch snapshot epoch (vk_texture_epoch)
typedef struct { PoolRec *r; uint64_t off, size; } MRange;
static MRange *pend, *wbl;             // replay: copies for the next prologue; GPU writes into mirrors this submission
static int npend, cpend, nwbl, cwbl;
static MRange *lzl;                    // replay: whole pages shaders stored into this submission, left in the mirror
static int nlzl, clzl;
static VkBufferCopy *mregions;         // replay: vkCmdCopyBuffer regions
static int cregions;
static PlatMutex flush_lock = PLAT_MUTEX_INIT;

typedef struct {
    uint64_t subs, syncs, upload_bytes, upload_copies, first_n, first_bytes, wb_bytes, wb_copies, wb_ranges;
    uint64_t presnap_queries, sync_queries, presnap_t, sync_t, prologues, presnap_bytes;
    uint64_t reup_in, reup_out;        // sync uploads inside / outside the previous submission's write-back span
    uint64_t reup_kept;                // written pages' bytes left out of the sync: write-back spans (sync_page)
    uint64_t chunk_in, chunk_out;      // bytes of the chunks brought into the mirrors (mirror_cover) and aged out
    uint64_t skipped;                  // written pages' bytes outside the mirrors' chunks: not copied
    uint64_t lazy_bytes;               // GPU-written bytes left in the mirrors instead of copied back
} MirrorAcc;
static MirrorAcc mrep;                 // replay: this submission
static MirrorAcc macc;                 // mirror_stat_lock: since the last summary
static PlatMutex mirror_stat_lock = PLAT_MUTEX_INIT;
static volatile int64_t m_live_n, m_live_bytes, m_refused_n[MR_N], m_refused_bytes[MR_N];
static volatile int64_t m_chunk_bytes;   // bytes of the chunks in the mirrors, all pools
// Lazy write-back: ranges read back at a CPU access (and their bytes), ranges written back at the fence because their
// pages could not be held.
static volatile int64_t m_lazy_fetches, m_lazy_fetch_bytes, m_lazy_unheld, m_lazy_total_fetches;
static volatile int64_t m_flush_calls, m_flush_bytes, v_pages, v_skipped, v_mismatch;

static void mirror_mode(int discrete) {
    const char *e = cfg_value("AC_MEMORY");
    int mode = discrete ? 1 : 0;   // 0 import, 1 mirror, 2 verify
    const char *why = discrete ? "auto, discrete gpu" : "auto, uma";
    if (strcmp(e, "auto")) {
        if (!strcmp(e, "import")) mode = 0, why = "AC_MEMORY=import";
        else if (!strcmp(e, "mirror")) mode = 1, why = "AC_MEMORY=mirror";
        else if (!strcmp(e, "verify")) mode = 2, why = "AC_MEMORY=verify";
        else rt_log("mem: AC_MEMORY=%s is not import, mirror, verify or auto: auto", e);
    }
    vk_mirror_on = mode > 0;
    mirror_verify = mode == 2;
    const char *g = cfg_get("AC_MIRROR_GAP");
    mirror_gap = g && *g ? atoi(g) : 2;
    if (mirror_gap < 0) mirror_gap = 0;
    const char *mc = cfg_get("AC_MIRROR_CHUNKS");
    mirror_chunks = !(mc && !strcmp(mc, "0"));
    const char *ml = cfg_get("AC_MIRROR_LAZY");
    mirror_lazy = !(ml && !strcmp(ml, "0"));
    const char *vm = cfg_get("AC_VERIFY_MIB");
    if (vm && atoi(vm) > 0 && atoi(vm) <= 1024) verify_bytes = (uint64_t)atoi(vm) << 20;
    // Submissions a region stays unwritten before it is frozen. 60 rather than 300: in a fight 100 of the 257 MiB
    // queried a frame lay in regions idle for 30-299 submissions, and a thaw by a write costs about 60 us.
    const char *fi = cfg_get("AC_FREEZE_IDLE");
    freeze_idle = fi && *fi ? atoi(fi) : 60;
    if (freeze_idle < 0) freeze_idle = 0;
    const char *sh = cfg_get("AC_SHADOW");
#ifdef _WIN32
    shadow_on = sh && *sh ? strcmp(sh, "0") != 0 : plat_is_wine();
#else
    shadow_on = !sh || !*sh || strcmp(sh, "0");
#endif
    if (shadow_on) rt_log("mem: shadow storage: CPU-visible pools are host-visible copies of guest memory, not imports%s",
                          sh && *sh ? " (AC_SHADOW)" : "");
    if (shadow_on) mirror_lazy = 0;   // the shadow storage holds the same pages for itself (gpu_written)
    const char *wu = cfg_get("AC_WATCH_USED");
    vk_need_on = shadow_on && !vk_mirror_on && !(wu && !strcmp(wu, "0"));
    if (shadow_on && !vk_mirror_on)
        rt_log("mem: write watch: %s (AC_WATCH_USED)", vk_need_on ? "only the pages the GPU used as buffers or textures"
                                                                 : "every page of the pools");
    rt_log("mem: mode %s (%s): %s%s%s", mode == 2 ? "verify" : mode ? "mirror" : "import", why,
           mode ? "pools used as buffers get a device-local mirror" : "every CPU-visible pool is imported host memory",
           !mode ? "" : mirror_chunks ? ", filled by the 64 KiB chunks the replay binds"
                                      : ", copied whole at the first use (AC_MIRROR_CHUNKS=0)",
           !mode ? "" : mirror_lazy ? "; large storage buffers the shaders write stay in it until the CPU touches them"
                                    : "; every GPU write is copied back (AC_MIRROR_LAZY=0)");
}

static void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size) {
    if (*n && (*a)[*n - 1].r == r && (*a)[*n - 1].off + (*a)[*n - 1].size == off) {
        (*a)[*n - 1].size += size;
        return;
    }
    if (*n == *cap) grow((void **)a, cap, sizeof **a);
    (*a)[(*n)++] = (MRange){r, off, size};
}

static int range_cmp(const void *x, const void *y) {
    const MRange *a = x, *b = y;
    if (a->r != b->r) return (uintptr_t)a->r < (uintptr_t)b->r ? -1 : 1;
    return a->off < b->off ? -1 : a->off > b->off;
}

typedef struct { PoolRec *r; uint32_t fence; } WbPurge;

// gen_lock exclusive: registers the write-back of pool bytes [lo, hi) by the submission of fence (mirror_sync's
// write-back span rule).
static void wb_span_add(PoolRec *r, uint64_t lo, uint64_t hi, uint64_t fence, int lazy) {
    if (r->nwb == r->cwb) grow((void **)&r->wb, &r->cwb, sizeof *r->wb);
    r->wb[r->nwb++] = (WbSpan){lo, hi, (uint32_t)fence, 0, (uint8_t)lazy};
}

static void mirror_hold(PoolRec *r, uint64_t lo, uint64_t hi);

// Completion thread, once the submission's fence is done: its write-back copies into the imported guest pages are
// over. Their write-watch bits, and those of CPU writes made into the same pages while the submission was in flight
// (the two cannot be told apart), are consumed into the generation table like any snapshot (query_stamp, region
// maxima included), so the texture checks and every other consumer still see those pages as written. The spans of
// this pool and fence get the stamp as purge_gen: mirror_sync then re-uploads those pages without the bytes the GPU
// wrote, and a page the CPU writes after the fence carries a newer generation and is copied whole. Spans are found
// by fence under gen_lock, not by a pointer: mirror_sync compacts the array in between.
static void wb_purge(void *arg) {
    WbPurge *p = arg;
    PoolRec *r = p->r;
    plat_mutex_lock(&gen_lock);
    uint32_t g = ++gen_now;
    int nlz = 0;
    for (int i = 0; i < r->nwb; i++)
        if (!r->wb[i].purge_gen && r->wb[i].fence == p->fence && r->wb[i].lazy) nlz++;
    uint64_t (*lz)[2] = nlz ? malloc((size_t)nlz * sizeof *lz) : NULL;
    if (nlz && !lz) rt_fatal("vk: out of memory (lazy write-back)");
    nlz = 0;
    for (int i = 0; i < r->nwb; i++) {
        WbSpan *sp = &r->wb[i];
        if (sp->purge_gen || sp->fence != p->fence) continue;
        query_stamp(r->import_lo + (sp->lo & ~0xfffull), r->import_lo + ((sp->hi + 0xfff) & ~0xfffull), g);
        sp->purge_gen = g;
        if (sp->lazy) lz[nlz][0] = sp->lo, lz[nlz][1] = sp->hi, nlz++;
    }
    plat_mutex_unlock(&gen_lock);
    // Outside gen_lock: holding a range may read another one back (memory.c mem_hold), which waits for the GPU. The
    // fence's waiters wake after this callback, so the pages are held before the game can know the GPU is done.
    for (int i = 0; i < nlz; i++) mirror_hold(r, lz[i][0], lz[i][1]);
    free(lz);
    free(p);
}

// Records the ranges as one vkCmdCopyBuffer per pool, overlapping and adjacent ranges merged; to_mirror: buf -> mbuf,
// else mbuf -> buf. Returns the commands recorded; adds what they move to *bytes and *copies.
// Write-backs (to_mirror 0) register each merged region as a span of the pool with the submission's fence, and one
// wb_purge per pool runs at that fence.
static int copy_ranges(VkCommandBuffer cb, MRange *a, int n, int to_mirror, uint64_t fence, uint64_t *bytes,
                       uint64_t *copies) {
    qsort(a, (size_t)n, sizeof *a, range_cmp);
    int rec = 0;
    for (int i = 0; i < n;) {
        PoolRec *r = a[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        int k = 0;
        for (; i < n && a[i].r == r; i++) {
            uint64_t lo = a[i].off, hi = a[i].off + a[i].size;
            if (hi > lim) hi = lim;
            if (lo >= hi) continue;
            if (!to_mirror) {   // mirror_end's write-backs: the span the next sync measures its re-upload against
                // wb_span_sub, not wb_sub: vk_mem_mirror_end stamps wb_sub before it calls here, so wb_sub cannot
                // tell this submission's first write-back range from a later one
                if (r->wb_span_sub != msub) r->wb_lo = lo, r->wb_hi = hi, r->wb_span_sub = msub;
                else {
                    if (lo < r->wb_lo) r->wb_lo = lo;
                    if (hi > r->wb_hi) r->wb_hi = hi;
                }
            }
            if (k && lo <= mregions[k - 1].srcOffset + mregions[k - 1].size) {
                uint64_t end = mregions[k - 1].srcOffset + mregions[k - 1].size;
                if (hi > end) mregions[k - 1].size += hi - end;
                continue;
            }
            if (k == cregions) grow((void **)&mregions, &cregions, sizeof *mregions);
            mregions[k++] = (VkBufferCopy){lo, lo, hi - lo};
        }
        if (!k) continue;
        if (to_mirror) vkCmdCopyBuffer(cb, r->buf, r->mbuf, (uint32_t)k, mregions);
        else vkCmdCopyBuffer(cb, r->mbuf, r->buf, (uint32_t)k, mregions);
        rec++;
        for (int j = 0; j < k; j++) *bytes += mregions[j].size;
        *copies += (uint64_t)k;
        if (!to_mirror) {
            WbPurge *p = malloc(sizeof *p);
            if (!p) rt_fatal("vk: out of memory (write-back purge)");
            *p = (WbPurge){r, (uint32_t)fence};
            plat_mutex_lock(&gen_lock);
            for (int j = 0; j < k; j++) wb_span_add(r, mregions[j].srcOffset, mregions[j].srcOffset + mregions[j].size, fence, 0);
            plat_mutex_unlock(&gen_lock);
            vk_completion_add(fence, wb_purge, p);
        }
    }
    return rec;
}

static void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                     VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                           .dstStageMask = ds, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// ---- lazy write-back
// The mirror's bytes [off, off+n) of the pool, copied out by the GPU after everything submitted so far. Any thread:
// a held range's fetch runs in the fault handler of whichever thread touched it, under memory.c's lock. It takes
// rb_lock, the queue lock and the allocator's locks, which no thread holds while it touches guest memory.
static PlatMutex rb_lock = PLAT_MUTEX_INIT;
static VkCommandPool rb_pool;
static VkCommandBuffer rb_cb;
static VkFence rb_fence;

static void mirror_read(PoolRec *r, uint64_t off, uint64_t n, void *dst) {
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VkBuffer buf;
    VmaAllocation alloc;
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &buf, &alloc, &info) != VK_SUCCESS || !info.pMappedData)
        rt_fatal("vk: no memory to read %.1f MiB of a VRAM mirror back", n / 1048576.0);
    plat_mutex_lock(&rb_lock);
    if (!rb_pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateCommandPool(c->device, &pi, NULL, &rb_pool) != VK_SUCCESS) rt_fatal("vk: mirror read-back command pool");
        VkCommandBufferAllocateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = rb_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        if (vkAllocateCommandBuffers(c->device, &ci, &rb_cb) != VK_SUCCESS || vkCreateFence(c->device, &fi, NULL, &rb_fence) != VK_SUCCESS)
            rt_fatal("vk: mirror read-back command buffer");
    }
    vkResetCommandBuffer(rb_cb, 0);
    VkCommandBufferBeginInfo bg = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(rb_cb, &bg);
    mbarrier(rb_cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy cp = {off, 0, n};
    vkCmdCopyBuffer(rb_cb, r->mbuf, buf, 1, &cp);
    mbarrier(rb_cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
             VK_ACCESS_2_HOST_READ_BIT);
    vkEndCommandBuffer(rb_cb);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &rb_cb};
    vk_queue_lock();
    VkResult q = vkQueueSubmit(c->gfx_queue, 1, &si, rb_fence);
    vk_queue_unlock();
    if (q == VK_SUCCESS) q = vkWaitForFences(c->device, 1, &rb_fence, VK_TRUE, 20000000000ull);
    if (q != VK_SUCCESS) {
        vk_device_fault_report(q);
        rt_fatal("vk: reading %.1f MiB of a VRAM mirror back failed (%d; device lost?)", n / 1048576.0, (int)q);
    }
    vkResetFences(c->device, 1, &rb_fence);
    plat_mutex_unlock(&rb_lock);
    vmaInvalidateAllocation(c->vma, alloc, 0, VK_WHOLE_SIZE);   // non-coherent memory only
    memcpy(dst, info.pMappedData, n);
    vmaDestroyBuffer(c->vma, buf, alloc);
}

// Ranges the CPU accessed while they were held: copied back at each submission until `until` (set by the replay
// thread at its next write of the range, from the submission count), LAZY_BACKOFF submissions the first time and
// twice as long with each further access. A buffer the game reads every frame costs one read-back and then
// behaves as before.
#define LAZY_BACKOFF 600u
#define LAZY_OFF_N 16
typedef struct { PoolRec *r; uint64_t lo, hi; uint32_t until; uint8_t n, armed; } LazyOff;
static LazyOff lzoff[LAZY_OFF_N];
static volatile int32_t nlzoff;
static PlatMutex lazy_lock = PLAT_MUTEX_INIT;   // leaf

static void lazy_off_note(PoolRec *r, uint64_t lo, uint64_t hi) {
    plat_mutex_lock(&lazy_lock);
    int i = 0, n = nlzoff;
    while (i < n && !(lzoff[i].r == r && lzoff[i].lo == lo && lzoff[i].hi == hi)) i++;
    if (i == n) {
        if (n == LAZY_OFF_N) memmove(lzoff, lzoff + 1, (LAZY_OFF_N - 1) * sizeof *lzoff), i = LAZY_OFF_N - 1;   // the oldest goes
        else __atomic_store_n(&nlzoff, n + 1, __ATOMIC_RELEASE);
        lzoff[i] = (LazyOff){r, lo, hi, 0, 0, 0};
    }
    if (lzoff[i].n < 6) lzoff[i].n++;
    lzoff[i].armed = 0;
    plat_mutex_unlock(&lazy_lock);
}

// Replay thread: 1 while pool bytes [lo, hi) are to be copied back at each submission.
static int lazy_off(PoolRec *r, uint64_t lo, uint64_t hi) {
    if (!__atomic_load_n(&nlzoff, __ATOMIC_ACQUIRE)) return 0;
    int off = 0;
    plat_mutex_lock(&lazy_lock);
    for (int i = 0; i < nlzoff; i++) {
        LazyOff *e = &lzoff[i];
        if (e->r != r || e->lo >= hi || e->hi <= lo) continue;
        if (!e->armed) e->until = msub + (LAZY_BACKOFF << (e->n - 1)), e->armed = 1;
        if ((int32_t)(e->until - msub) > 0) off = 1;
    }
    plat_mutex_unlock(&lazy_lock);
    return off;
}

static void lazy_off_forget(const PoolRec *r) {
    plat_mutex_lock(&lazy_lock);
    int k = 0;
    for (int i = 0; i < nlzoff; i++)
        if (lzoff[i].r != r) lzoff[k++] = lzoff[i];
    __atomic_store_n(&nlzoff, k, __ATOMIC_RELEASE);
    plat_mutex_unlock(&lazy_lock);
}

// memory.c's fetch of a held range (MemFetch): the guest pages [a, a+n) of pool ctx get the mirror's bytes.
static void mirror_fetch(void *ctx, uint64_t a, uint64_t n, void *dst) {
    PoolRec *r = ctx;
    mirror_read(r, a - r->import_lo, n, dst);
    lazy_off_note(r, a - r->import_lo, a - r->import_lo + n);
    plat_inc64(&m_lazy_fetches);
    plat_inc64(&m_lazy_total_fetches);
    plat_add64(&m_lazy_fetch_bytes, (int64_t)n);
}

// Pool bytes [lo, hi) from the mirror into the guest pages now, and copied back from then on.
static void mirror_unheld(PoolRec *r, uint64_t lo, uint64_t hi) {
    uint64_t a = r->import_lo + lo, n = hi - lo;
    void *tmp = malloc(n);
    if (!tmp) rt_fatal("vk: out of memory (lazy write-back)");
    mirror_read(r, lo, n, tmp);
    memcpy((void *)(uintptr_t)a, tmp, n);
    free(tmp);
    lazy_off_note(r, lo, hi);
    plat_inc64(&m_lazy_unheld);
}

// Completion thread, at the fence of a submission that left pool bytes [lo, hi) (whole pages) in the mirror: the CPU
// loses access to them until it asks. A hold covers whole host pages (platform.h plat_mem_grain: 16 KiB on Apple
// Silicon), so the partial host pages at the ends, and pages that cannot be held, get their bytes now, as the copy at
// the end of the submission would have put them.
static void mirror_hold(PoolRec *r, uint64_t lo, uint64_t hi) {
    uint64_t g = plat_mem_grain(), a = r->import_lo + lo, b = r->import_lo + hi;
    uint64_t ha = (a + g - 1) & ~(g - 1), hb = b & ~(g - 1);
    if (hb <= ha || !mem_hold(ha, hb - ha, mirror_fetch, r)) {
        mirror_unheld(r, lo, hi);
        return;
    }
    if (ha > a) mirror_unheld(r, lo, ha - r->import_lo);
    if (b > hb) mirror_unheld(r, hb - r->import_lo, hi);
}

void vk_mem_lazy_all(int on) { lazy_all = on; }

int vk_mem_lazy_on(void) { return vk_mirror_on && mirror_lazy; }

const void *vk_mem_lazy_owner(uint64_t addr) {
    VkMemSpan sp;
    return vk_mem_resolve(addr, 1, &sp) ? sp.pool : NULL;
}

void vk_mem_lazy_stats(uint64_t *fetches, uint64_t *unheld) {
    *fetches = (uint64_t)__atomic_load_n(&m_lazy_total_fetches, __ATOMIC_RELAXED);
    *unheld = (uint64_t)__atomic_load_n(&m_lazy_unheld, __ATOMIC_RELAXED);
}

// Replay thread: the chunks of the mirror over pool bytes [off, off+n) (n 0: to the end of the pool) are in it from
// this submission on. A chunk that was not is queued for the prologue, whole: the copy reads the imported buffer
// before the part being recorded runs, and from then on mirror_sync keeps the chunk up to date.
static void mirror_cover(PoolRec *r, uint64_t off, uint64_t n) {
    uint64_t lim = r->import_hi - r->import_lo;
    if (off >= lim) return;
    if (!n || n > lim - off) n = lim - off;
    for (uint64_t c = off >> MCHUNK_SHIFT, c1 = (off + n - 1) >> MCHUNK_SHIFT; c <= c1; c++) {
        if (r->mlive[c] == msub) continue;
        if (!r->mlive[c]) {
            uint64_t a = c << MCHUNK_SHIFT, b = a + MCHUNK < lim ? a + MCHUNK : lim;
            range_add(&pend, &npend, &cpend, r, a, b - a);
            r->mlive_bytes += b - a, mrep.chunk_in += b - a;
            plat_add64(&m_chunk_bytes, (int64_t)(b - a));
        }
        r->mlive[c] = msub;
    }
}

// 1 when a chunk over pool bytes [off, off+n) is in the mirror.
static int mirror_has(const PoolRec *r, uint64_t off, uint64_t n) {
    uint64_t lim = r->import_hi - r->import_lo;
    if (off >= lim) return 0;
    if (!n || n > lim - off) n = lim - off;
    for (uint64_t c = off >> MCHUNK_SHIFT, c1 = (off + n - 1) >> MCHUNK_SHIFT; c <= c1; c++)
        if (r->mlive[c]) return 1;
    return 0;
}

// Queues pool bytes [a, b) for the prologue where their chunks are in the mirror.
static void queue_live(PoolRec *r, uint64_t a, uint64_t b) {
    while (a < b) {
        uint64_t c = a >> MCHUNK_SHIFT, e = (c + 1) << MCHUNK_SHIFT < b ? (c + 1) << MCHUNK_SHIFT : b;
        if (r->mlive[c]) range_add(&pend, &npend, &cpend, r, a, e - a);
        a = e;
    }
}

static void refuse(PoolRec *r, int why, const char *what) {
    r->mstate = MS_REFUSED;
    uint64_t n = r->import_hi - r->import_lo;
    if (plat_inc64(&m_refused_n[why]) <= 4) {
        char f[160];
        flag_names(r->flags, f, sizeof f);
        rt_log("mem: pool %#llx+%#llx (%s) stays imported: %s", (unsigned long long)r->lo, (unsigned long long)r->size,
               f, what);
    }
    plat_add64(&m_refused_bytes[why], (int64_t)n);
}

// Replay thread: the pool's mirror on its first buffer use.
static void mirror_create(PoolRec *r) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    r->mstate = MS_REFUSED;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    if (r->kind != POOL_IMPORT || !r->buf || !n) return;
    plat_mutex_lock_shared(&pool_lock);
    int i = pidx_upper(&pool_index, r->lo) - 1;
    while (i >= 0 && pool_index.e[i].buf != r) i--;
    int alone = i >= 0 && (i == 0 || pool_index.maxhi[i - 1] <= pool_index.e[i].lo) &&
                (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= pool_index.e[i].hi);
    plat_mutex_unlock_shared(&pool_lock);
    if (i < 0) return;   // finalized meanwhile
    if (!alone) return refuse(r, MR_OVERLAP, "it overlaps another pool");
    uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return refuse(r, MR_UNTRACKED, "outside the write-watched address space");
    uint64_t used, budget;
    int exact;
    if (vk_mem_vram(&used, &budget, &exact) && used + n + budget / 8 > budget) {
        char why[128];
        snprintf(why, sizeof why, "its %.1f MiB mirror does not fit the VRAM budget (%.0f of %.0f MiB in use)",
                 n / 1048576.0, used / 1048576.0, budget / 1048576.0);
        return refuse(r, MR_BUDGET, why);
    }
    plat_mutex_lock(&gen_lock);
    mrep.sync_queries += snap_regions(r0, r1, mepoch);
    int untracked = 0;
    for (uint64_t k = r0; k <= r1; k++) untracked |= snap_untracked[k];
    r->mgen = gen_now;   // the whole-range copy below holds every write observed so far
    plat_mutex_unlock(&gen_lock);
    if (untracked) return refuse(r, MR_UNTRACKED, "its range is not write-watched");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    r->mlive = calloc((size_t)((n + MCHUNK - 1) >> MCHUNK_SHIFT), sizeof *r->mlive);
    VkResult res = r->mlive ? vmaCreateBuffer(c->vma, &bi, &ai, &r->mbuf, &r->malloc_, NULL) : VK_ERROR_OUT_OF_HOST_MEMORY;
    if (res != VK_SUCCESS) {
        r->mbuf = VK_NULL_HANDLE, r->malloc_ = NULL;
        free(r->mlive), r->mlive = NULL;
        char why[96];
        snprintf(why, sizeof why, "device-local allocation of %.1f MiB failed (%d)", n / 1048576.0, (int)res);
        return refuse(r, MR_ALLOC, why);
    }
    plat_mutex_lock(&pool_lock);
    if (nm == mcap) grow((void **)&mlist, &mcap, sizeof *mlist);
    int at = nm;
    while (at > 0 && mlist[at - 1]->import_lo > r->import_lo) at--;
    memmove(mlist + at + 1, mlist + at, (size_t)(nm - at) * sizeof *mlist);
    mlist[at] = r;
    nm++;
    plat_mutex_unlock(&pool_lock);
    r->acc_sub = msub;
    r->mstate = MS_ON;
    mrep.first_n++;
    plat_inc64(&m_live_n);
    plat_add64(&m_live_bytes, (int64_t)n);
    if (!mirror_chunks) mirror_cover(r, 0, n), mrep.first_bytes += n;   // the whole pool, in the mirror for good
}

static void mirror_forget(PoolRec *r) {
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < nm; i++)
        if (mlist[i] == r) {
            memmove(mlist + i, mlist + i + 1, (size_t)(nm - i - 1) * sizeof *mlist);
            nm--;
            plat_dec64(&m_live_n);
            plat_add64(&m_live_bytes, -(int64_t)(r->import_hi - r->import_lo));
            plat_add64(&m_chunk_bytes, -(int64_t)r->mlive_bytes);
            lazy_off_forget(r);
            break;
        }
}

// Finalized mirrored pools waiting for mirror_retire; vk_mem_shutdown destroys those whose callback never ran.
static PoolRec **retiring;
static int nretiring, cretiring;
static PlatMutex retire_lock = PLAT_MUTEX_INIT;

static void mirror_retiring(PoolRec *r) {
    plat_mutex_lock(&retire_lock);
    if (nretiring == cretiring) grow((void **)&retiring, &cretiring, sizeof *retiring);
    retiring[nretiring++] = r;
    plat_mutex_unlock(&retire_lock);
}

static void mirror_retire(void *arg) {
    plat_mutex_lock(&retire_lock);
    for (int i = 0; i < nretiring; i++)
        if (retiring[i] == arg) { retiring[i] = retiring[--nretiring]; break; }
    plat_mutex_unlock(&retire_lock);
    destroy_rec(vk_ctx(), arg);
}

// Write-back spans in mirror_sync. A submission's write-back copies its GPU writes from the mirror into the imported
// guest pages; the DMA sets their write-watch bits like a CPU write would, and syncing those pages whole re-uploaded
// what the mirror already held (8-15 MiB a frame at the title). A page written since the last sync that a span covers
// (still in flight, or purged with the page stamped at or before its purge) is therefore queued without the span's
// bytes: the mirror holds the GPU's values there, while CPU bytes beside them in the page, written during the flight,
// still reach the mirror. Only a CPU write into the very bytes a shader wrote in that submission is lost, a race on
// hardware as well. A page stamped after the purge was written by the CPU after the fence and is copied whole.
// gen_lock held: queues [off, off+page) of the pool minus the covering spans' bytes; returns the bytes left out.
static uint64_t sync_page(PoolRec *r, uint64_t off, uint64_t page, uint32_t gen) {
    uint64_t cur = off, end = off + page, out = 0;
    while (cur < end) {
        uint64_t reach = cur, next = end;   // the furthest covering span end from cur; the nearest span start after it
        for (int i = 0; i < r->nwb; i++) {
            const WbSpan *sp = &r->wb[i];
            if (sp->hi <= cur || sp->lo >= end || (sp->purge_gen && gen > sp->purge_gen)) continue;
            if (sp->lo <= cur) {
                if (sp->hi > reach) reach = sp->hi;
            } else if (sp->lo < next) next = sp->lo;
        }
        if (reach > cur) {
            if (reach > end) reach = end;
            out += reach - cur, cur = reach;
            continue;
        }
        range_add(&pend, &npend, &cpend, r, cur, next - cur);
        cur = next;
    }
    return out;
}

// Replay thread, the first access of this submission: queues the copies of the pages written since the last sync.
static void mirror_sync(PoolRec *r) {
    uint64_t t0 = qpc();
    r->acc_sub = msub;
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t k = r0; k <= r1; k++)
        if (snap_at[k] != mepoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        mrep.sync_queries += snap_regions(r0, r1, mepoch);
    }
    uint32_t since = r->mgen;
    for (uint64_t k = r0; k <= r1; k++) {
        if (reg_max[k] <= since) continue;
        uint64_t a = k << SNAP_SHIFT, b = a + (1ull << SNAP_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (!ch) continue;
        uint64_t page = 1ull << GEN_PAGE_SHIFT;
        int in_wb = r->wb_span_sub && r->wb_span_sub == msub - 1;   // the previous submission wrote [wb_lo, wb_hi) back
        for (uint64_t x = a; x < b; x += page)
            if (ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since) {
                uint64_t off = x - lo, kept = 0;
                uint32_t *lv = &r->mlive[off >> MCHUNK_SHIFT];
                // A chunk no submission bound for a while leaves the mirror here, where keeping it would cost a copy;
                // its next bind brings it back whole (mirror_cover).
                if (*lv && mirror_chunks && msub - *lv > MIRROR_AGE) {
                    uint64_t ca = (off >> MCHUNK_SHIFT) << MCHUNK_SHIFT, cb = ca + MCHUNK < hi - lo ? ca + MCHUNK : hi - lo;
                    uint64_t hl, hh;
                    if (mem_held_by(lo + ca, lo + cb, r, &hl, &hh)) {
                        *lv = msub;   // a held range's pages are in the mirror only: the chunk stays
                    } else {
                        *lv = 0, r->mlive_bytes -= cb - ca, mrep.chunk_out += cb - ca;
                        plat_add64(&m_chunk_bytes, -(int64_t)(cb - ca));
                    }
                }
                if (!*lv) { mrep.skipped += page; continue; }
                if (r->nwb) kept = sync_page(r, off, page, ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)]);
                else range_add(&pend, &npend, &cpend, r, off, page);
                mrep.reup_kept += kept;
                if (in_wb && off >= r->wb_lo && off < r->wb_hi) mrep.reup_in += page - kept;
                else mrep.reup_out += page - kept;
            }
    }
    r->mgen = gen_now;
    // A span purged before this pass is done: its stamps are at or below the new mgen. One purged since (between the
    // pass and the exclusive lock) stays for the next pass. Exclusive: wb_span_add and wb_purge change the array.
    if (!excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
    }
    int j = 0;
    for (int i = 0; i < r->nwb; i++)
        if (!r->wb[i].purge_gen || r->wb[i].purge_gen > r->mgen) r->wb[j++] = r->wb[i];
    r->nwb = j;
    plat_mutex_unlock(&gen_lock);
    if (__atomic_load_n(&r->fl_hi, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&flush_lock);
        uint64_t a = r->fl_lo, b = r->fl_hi;
        r->fl_hi = 0;
        plat_mutex_unlock(&flush_lock);
        a = a > lo ? a & ~0xfffull : lo;
        b = b < hi ? (b + 0xfff) & ~0xfffull : hi;
        if (a < b) queue_live(r, a - lo, b - lo);
    }
    mrep.syncs++;
    mrep.sync_t += qpc() - t0;
}

// vk_need_on: the GPU reads n bytes of the pool's shadow from sp on (0: to its end).
static void shadow_need(const PoolRec *r, uint64_t off, uint64_t n) {
    uint64_t lim = r->import_hi - r->import_lo;
    if (!r->smap || off >= lim) return;
    if (!n || n > lim - off) n = lim - off;
    need_range(r->import_lo + off, n);
}

void vk_mem_use(VkMemSpan *sp, uint64_t n) {
    PoolRec *r = sp->pool;
    if (vk_need_on) { shadow_need(r, sp->offset, n); return; }
    if (r->mstate == MS_NONE) mirror_create(r);
    if (r->mstate != MS_ON) return;
    if (r->acc_sub != msub) mirror_sync(r);
    mirror_cover(r, sp->offset, n);
    sp->buffer = r->mbuf;
}

void vk_mem_pick(VkMemSpan *sp, uint64_t n, int write) {
    PoolRec *r = sp->pool;
    if (vk_need_on) {   // a GPU write needs nothing of the CPU's bytes: the pages around it are copied when read
        if (r && !write) shadow_need(r, sp->offset, n);
        return;
    }
    if (!r || r->mstate != MS_ON) return;
    if (r->acc_sub != msub) mirror_sync(r);
    // A read of bytes the mirror never held: the GPU wrote none of them, so the imported buffer has them.
    if (!write && !mirror_has(r, sp->offset, n)) return;
    mirror_cover(r, sp->offset, n);
    sp->buffer = r->mbuf;
}

int vk_mem_mirror_probe(uint64_t addr) {
    VkMemSpan sp;
    if (!vk_mirror_on || !vk_mem_resolve(addr, 1, &sp) || !sp.pool) return 0;
    mepoch = vk_texture_epoch();
    vk_mem_use(&sp, 1);
    return ((PoolRec *)sp.pool)->mstate == MS_ON;
}

void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size, int stored) {
    PoolRec *r = sp->pool;
    if (!r || r->mstate != MS_ON || sp->buffer != r->mbuf || !size) return;
    if (mirror_lazy && (stored || lazy_all)) {
        // The pool's offsets are page-aligned with guest memory (import_lo is).
        uint64_t lo = (sp->offset + 0xfff) & ~0xfffull, hi = (sp->offset + size) & ~0xfffull;
        if (hi > lo && hi - lo >= HOLD_MIN && !lazy_off(r, lo, hi)) {
            int i = 0;   // a range several shaders of the submission store into counts once; no merge with a neighbour
            while (i < nlzl && !(lzl[i].r == r && lzl[i].off == lo && lzl[i].size == hi - lo)) i++;
            if (i == nlzl) {
                if (nlzl == clzl) grow((void **)&lzl, &clzl, sizeof *lzl);
                lzl[nlzl++] = (MRange){r, lo, hi - lo};
            }
            if (lo > sp->offset) range_add(&wbl, &nwbl, &cwbl, r, sp->offset, lo - sp->offset);
            if (sp->offset + size > hi) range_add(&wbl, &nwbl, &cwbl, r, hi, sp->offset + size - hi);
            return;
        }
    }
    range_add(&wbl, &nwbl, &cwbl, r, sp->offset, size);
}

// Replay thread, before the write-backs are recorded: the submission's lazy ranges become spans of their pools, as
// the copied ranges do (mirror_sync's rule for a page written since the last sync is the same for both), and the
// completion callback that holds their pages is queued. A lazy range that overlaps another one is merged with it; one
// that a copy, fill or update of this submission also wrote is copied back whole.
static void lazy_end(uint64_t fence) {
    qsort(lzl, (size_t)nlzl, sizeof *lzl, range_cmp);
    int k = 0;
    for (int i = 0; i < nlzl; i++) {
        if (k && lzl[k - 1].r == lzl[i].r && lzl[i].off < lzl[k - 1].off + lzl[k - 1].size) {
            uint64_t e = lzl[i].off + lzl[i].size;
            if (e > lzl[k - 1].off + lzl[k - 1].size) lzl[k - 1].size = e - lzl[k - 1].off;
            continue;
        }
        lzl[k++] = lzl[i];
    }
    PoolRec *last = NULL;
    for (int i = 0; i < k; i++) {
        PoolRec *r = lzl[i].r;
        uint64_t lo = lzl[i].off, hi = lo + lzl[i].size;
        int copied = 0;
        for (int j = 0; j < nwbl && !copied; j++) copied = wbl[j].r == r && wbl[j].off < hi && wbl[j].off + wbl[j].size > lo;
        if (copied) {
            range_add(&wbl, &nwbl, &cwbl, r, lo, hi - lo);
            continue;
        }
        plat_mutex_lock(&gen_lock);
        wb_span_add(r, lo, hi, fence, 1);
        plat_mutex_unlock(&gen_lock);
        r->wb_sub = msub;
        mrep.lazy_bytes += hi - lo;
        if (r == last) continue;
        last = r;
        WbPurge *p = malloc(sizeof *p);
        if (!p) rt_fatal("vk: out of memory (write-back purge)");
        *p = (WbPurge){r, (uint32_t)fence};
        vk_completion_add(fence, wb_purge, p);
    }
    nlzl = 0;
}

void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size) {
    if (!vk_mirror_on || !size || !p->mem) return;
    plat_inc64(&m_flush_calls);
    plat_add64(&m_flush_bytes, (int64_t)size);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)p->mem, 0);
    PoolRec *r = e && e->key == p ? e->buf : NULL;
    if (r && r->mstate == MS_ON) {
        uint64_t a = (uintptr_t)p->mem + off, b = a + size;
        plat_mutex_lock(&flush_lock);
        if (!r->fl_hi) r->fl_lo = a, r->fl_hi = b;
        else {
            if (a < r->fl_lo) r->fl_lo = a;
            if (b > r->fl_hi) r->fl_hi = b;
        }
        plat_mutex_unlock(&flush_lock);
    }
    plat_mutex_unlock_shared(&pool_lock);
}

// gen_lock exclusive (and in mirror mode pool_lock shared, as mirror_begin holds): 1 when any mirrored pool has a
// write-back span in flight inside [a, b): its DMA is still writing guest pages there, so the region cannot be made
// read-only (the DMA would fault or stall the engine silently, no CPU fault thaw covers it).
static int wb_flight_overlap(uint64_t a, uint64_t b) {
    if (!vk_mirror_on) return 0;
    for (int i = 0; i < nm; i++) {
        const PoolRec *r = mlist[i];
        for (int s = 0; s < r->nwb; s++) {
            const WbSpan *sp = &r->wb[s];
            if (sp->purge_gen) continue;   // done (purged): its DMA is over
            if (r->import_lo + sp->lo < b && r->import_lo + sp->hi > a) return 1;
        }
    }
    return 0;
}

// gen_lock exclusive, vk_need_on. The queries leave the pages nobody needs written, so they cannot tell whether the
// CPU still writes them, and freezing a region the game writes every frame costs a fault and a thaw each time (about
// 60 us). A candidate is therefore reset whole once (every page protected again, the needed ones found written
// stamped), and frozen at its next candidacy, freeze_idle submissions later, only if no page was written in between:
// 1 then. A region the CPU still writes is reset again instead, one fault per written page per freeze_idle submissions.
static int freeze_armed(uint64_t j) {
    const uint64_t max = sizeof gen_pages / sizeof *gen_pages;
    uint64_t a = j << REG_SHIFT, b = a + (1ull << REG_SHIFT);
    if (reg_arm[j] > reg_last[j]) {   // reset since the last stamp: any written page is a write since then
        uint64_t n = mem_written_peek(a, b - a, peek_pages, 1);
        ww_calls++, ww_bytes += b - a;
        if (!n) return 1;
        if (n == UINT64_MAX) return 0;
    }
    uint32_t g = 0;
    for (;;) {
        uint64_t n = ww_query(a, b - a);
        if (n == UINT64_MAX) return 0;
        for (uint64_t k = 0; k < n; k++)
            if (need_has(gen_pages[k])) {
                if (!g) g = ++gen_now;
                stamp_page(gen_pages[k], g);
            }
        if (n < max) break;
    }
    reg_arm[j] = ww_ep;   // a stamp above made reg_last the same: not armed, the region is not idle
    return 0;
}

// gen_lock exclusive: freezes up to FREEZE_PER_SUB candidate regions still idle. The query after the
// protection collects the writes made before it took effect.
static void freeze_pass(void) {
    for (int done = 0; nfcand && done < FREEZE_PER_SUB;) {
        uint64_t j = fcand[--nfcand], a = j << REG_SHIFT;
        if (ww_ep - reg_last[j] < (uint32_t)freeze_idle || mem_frozen(a)) continue;
        if (wb_flight_overlap(a, a + (1ull << REG_SHIFT))) continue;
        done++;
        if (vk_need_on && !freeze_armed(j)) continue;
        if (!mem_freeze(a)) { n_frz_refused++; continue; }
        query_stamp(a, a + (1ull << REG_SHIFT), ++gen_now);
    }
}

void vk_mem_mirror_begin(void) {
    if (!vk_mirror_on) {   // import mode: the texture checks' snapshots still skip frozen regions
        if (!freeze_idle) return;
        plat_mutex_lock(&gen_lock);
        ww_ep++;
        freeze_pass();
        plat_mutex_unlock(&gen_lock);
        return;
    }
    uint64_t t0 = qpc();
    uint32_t prev = msub;
    msub = msub + 1 ? msub + 1 : 1;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    mepoch = vk_texture_epoch();
    uint64_t q = 0, run0 = 0, run1 = 0;
    int have = 0;
    plat_mutex_lock_shared(&pool_lock);
    plat_mutex_lock(&gen_lock);
    ww_ep++;
    uint64_t b0 = ww_bytes;
    for (int i = 0; i < nm; i++) {
        const PoolRec *r = mlist[i];
        if (r->acc_sub != prev) continue;
        uint64_t a = r->import_lo >> SNAP_SHIFT, b = (r->import_hi - 1) >> SNAP_SHIFT;
        if (have && a <= run1 + 1 + (uint64_t)mirror_gap) {
            if (b > run1) run1 = b;
            continue;
        }
        if (have) q += snap_regions(run0, run1, mepoch);
        run0 = a, run1 = b, have = 1;
    }
    if (have) q += snap_regions(run0, run1, mepoch);
    mrep.presnap_bytes += ww_bytes - b0;
    if (freeze_idle) freeze_pass();
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    mrep.subs++;
    mrep.presnap_queries += q;
    mrep.presnap_t += qpc() - t0;
}

int vk_mem_mirror_pending(void) { return npend; }

void vk_mem_mirror_prologue(VkCommandBuffer cb) {
    if (!npend) return;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    copy_ranges(cb, pend, npend, 1, 0, &mrep.upload_bytes, &mrep.upload_copies);
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    npend = 0;
    mrep.prologues++;
}

// ---- verify mode
typedef struct { PoolRec *r; uint64_t off, len, dst; uint32_t mgen; } VSample;
typedef struct { uint32_t sub; int n; VkBuffer buf; VmaAllocation alloc; const uint8_t *map; VSample s[VERIFY_SAMPLES]; } VJob;
static int ver_i;
static uint64_t ver_off;

// The CPU wrote the page since the mirror's sync that mgen records: stamped later, or its write-watch bit is set.
static int page_dirty(uint64_t a, uint32_t mgen) {
    uint32_t *s = gen_slot(a, 0);
    if (s && __atomic_load_n(s, __ATOMIC_RELAXED) > mgen) return 1;
    s = gpu_slot(a, 0);
    if (s && __atomic_load_n(s, __ATOMIC_RELAXED) > mgen) return 1;
    uint64_t p[1];
    return mem_written_peek(a, 1ull << GEN_PAGE_SHIFT, p, 1) != 0;   // UINT64_MAX (not watched) counts as written
}

static void verify_cb(void *arg) {   // completion thread
    VJob *j = arg;
    vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);   // non-coherent memory only
    static volatile int32_t logged;
    uint64_t pages = 0, skipped = 0, bad = 0;
    for (int i = 0; i < j->n; i++) {
        const VSample *s = &j->s[i];
        PoolRec *r = s->r;
        uint32_t wb = __atomic_load_n(&r->wb_sub, __ATOMIC_RELAXED);
        if (wb != j->sub && wb - j->sub < 0x80000000u) {   // a later submission wrote back into it
            skipped += s->len >> GEN_PAGE_SHIFT;
            continue;
        }
        for (uint64_t o = 0; o < s->len; o += 1ull << GEN_PAGE_SHIFT) {
            uint64_t a = r->import_lo + s->off + o;
            const uint8_t *m = j->map + s->dst + o, *g = (const uint8_t *)(uintptr_t)a;
            uint64_t hl, hh;
            if (mem_held_by(a, a + (1ull << GEN_PAGE_SHIFT), r, &hl, &hh)) { skipped++; continue; }   // only the mirror has it
            pages++;
            if (!memcmp(m, g, 1ull << GEN_PAGE_SHIFT)) continue;
            if (page_dirty(a, s->mgen)) { skipped++; continue; }
            bad++;
            if (plat_inc32(&logged) <= VERIFY_LOG) {
                int k = 0, e = (1 << GEN_PAGE_SHIFT) - 1;
                while (k < e && m[k] == g[k]) k++;
                while (e > k && m[e] == g[e]) e--;
                char f[160];
                flag_names(r->flags, f, sizeof f);
                rt_log("mem: verify mismatch: pool %#llx+%#llx (%s) offset %#llx: bytes %#x..%#x of the page differ, "
                       "mirror %02x guest %02x at the first (submission %u)", (unsigned long long)r->lo,
                       (unsigned long long)r->size, f, (unsigned long long)(a - r->lo), k, e, m[k], g[k], j->sub);
            }
        }
    }
    plat_add64(&v_pages, (int64_t)pages);
    plat_add64(&v_skipped, (int64_t)skipped);
    plat_add64(&v_mismatch, (int64_t)bad);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static int verify_record(VkCommandBuffer cb, uint64_t fence) {
    VJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk: out of memory");
    uint64_t left = verify_bytes, dst = 0;
    plat_mutex_lock_shared(&pool_lock);
    // Runs of chunks that are in a mirror, from where the last submission stopped; at most one turn over the pools.
    for (int pools = 0; nm && left && j->n < VERIFY_SAMPLES && pools <= nm;) {
        if (ver_i >= nm) ver_i = 0, ver_off = 0;
        PoolRec *r = mlist[ver_i];
        uint64_t size = r->import_hi - r->import_lo;
        while (ver_off < size && !r->mlive[ver_off >> MCHUNK_SHIFT]) ver_off = ((ver_off >> MCHUNK_SHIFT) + 1) << MCHUNK_SHIFT;
        if (ver_off >= size) { ver_i++, ver_off = 0, pools++; continue; }
        uint64_t end = ver_off;
        while (end < size && end - ver_off < left && r->mlive[end >> MCHUNK_SHIFT]) end = ((end >> MCHUNK_SHIFT) + 1) << MCHUNK_SHIFT;
        if (end > size) end = size;
        uint64_t len = end - ver_off < left ? end - ver_off : left;
        j->s[j->n++] = (VSample){r, ver_off, len, dst, r->mgen};
        dst += len, left -= len, ver_off += len;
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (!j->n) { free(j); return 0; }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = dst,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS) { free(j); return 0; }
    j->map = info.pMappedData;
    j->sub = msub;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < j->n; i++) {
        VkBufferCopy c = {j->s[i].off, j->s[i].dst, j->s[i].len};
        vkCmdCopyBuffer(cb, j->s[i].r->mbuf, j->buf, 1, &c);
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
             VK_ACCESS_2_HOST_READ_BIT);
    vk_completion_add(fence, verify_cb, j);
    return j->n + 2;
}

int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence) {
    if (!vk_mirror_on) return 0;
    int rec = 0;
    if (nlzl) lazy_end(fence);
    if (nwbl) {
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
        for (int i = 0; i < nwbl; i++) wbl[i].r->wb_sub = msub;
        mrep.wb_ranges += (uint64_t)nwbl;
        rec += copy_ranges(cb, wbl, nwbl, 0, fence, &mrep.wb_bytes, &mrep.wb_copies) + 2;
        mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        nwbl = 0;
    }
    if (mirror_verify) rec += verify_record(cb, fence);
    plat_mutex_lock(&mirror_stat_lock);
    uint64_t *d = (uint64_t *)&macc, *s = (uint64_t *)&mrep;
    for (size_t i = 0; i < sizeof macc / sizeof(uint64_t); i++) d[i] += s[i];
    plat_mutex_unlock(&mirror_stat_lock);
    memset(&mrep, 0, sizeof mrep);
    return rec;
}

// One summary line: with AC_TRACE=perf every 60 frames, else at frames 600 and 1800.
static void mirror_log(uint64_t frame) {
    static uint64_t last, ww_last;
    plat_mutex_lock(&gen_lock);
    uint64_t wc = ww_calls, wb = ww_bytes, wt = ww_ticks, dc = ww_dirty_calls, dt = ww_dirty_ticks, wp = ww_pages;
    ww_pages = 0;
    uint64_t nr = n_frz_refused;
    ww_calls = ww_bytes = ww_ticks = ww_dirty_calls = ww_dirty_ticks = n_frz_refused = 0;
    plat_mutex_unlock(&gen_lock);
    double wf = frame > ww_last ? (double)(frame - ww_last) : 1.0;
    ww_last = frame;
    uint64_t fz, ff, ft, fo;
    mem_freeze_stats(&fz, &ff, &ft, &fo);
    if (rt_trace & RT_TRACE_PERF)
        rt_log("mem: write-watch frame %llu: per frame, every consumer: %.1f queries over %.1f MiB, %.3f ms (%.1f found "
               "%.0f written pages, %.3f ms); frozen %.0f MiB; since the last line: freezes %llu, thaws by a write %llu, by a file read "
               "or mapping change %llu, refused %llu",
               (unsigned long long)frame, wc / wf, wb / wf / 1048576.0, qpc_ms(wt) / wf, dc / wf, wp / wf, qpc_ms(dt) / wf,
               fz * 2.0, (unsigned long long)ff, (unsigned long long)ft, (unsigned long long)fo, (unsigned long long)nr);
    if (!vk_mirror_on) return;
    plat_mutex_lock(&mirror_stat_lock);
    MirrorAcc a = macc;
    memset(&macc, 0, sizeof macc);
    plat_mutex_unlock(&mirror_stat_lock);
    double f = frame > last ? (double)(frame - last) : 1.0, M = 1048576.0;
    last = frame;
    uint64_t fc = (uint64_t)plat_xchg64(&m_flush_calls, 0), fb = (uint64_t)plat_xchg64(&m_flush_bytes, 0);
    char ver[160] = "";
    if (mirror_verify)
        snprintf(ver, sizeof ver, "; verify: pages compared %lld, skipped %lld (written since), mismatches %lld",
                 (long long)plat_xchg64(&v_pages, 0), (long long)plat_xchg64(&v_skipped, 0),
                 (long long)plat_xchg64(&v_mismatch, 0));
    rt_log("mem: mirror chunks frame %llu: %.1f MiB in the mirrors; per frame: %.3f MiB brought in, %.3f MiB aged out, "
           "%.3f MiB of written pages outside them not copied", (unsigned long long)frame, m_chunk_bytes / M, a.chunk_in / f / M,
           a.chunk_out / f / M, a.skipped / f / M);
    rt_log("mem: mirror lazy frame %llu: per frame %.3f MiB of GPU writes left in the mirrors; since the last line: %lld "
           "ranges read back for the CPU (%.1f MiB), %lld written back at the fence (pages not holdable, in total)",
           (unsigned long long)frame, a.lazy_bytes / f / M, (long long)plat_xchg64(&m_lazy_fetches, 0),
           (double)plat_xchg64(&m_lazy_fetch_bytes, 0) / M, (long long)m_lazy_unheld);
    rt_log("mem: mirror frame %llu: pools %lld, %.1f MiB device-local; stay imported: budget %lld (%.1f MiB), untracked "
           "%lld, overlapping %lld, failed %lld; per frame: submits %.2f, syncs %.1f, upload %.3f MiB in %.1f ranges, "
           "first use %.2f pools %.3f MiB, write-back %.3f MiB in %.1f ranges, flush calls %.1f (%.3f MiB), queries "
           "%.1f pre-snap (%.1f MiB) + %.1f sync, prologues %.2f, tracking cpu %.3f ms (pre-snap %.3f, syncs %.3f), re-up %.1f + %.1f MiB (write-back kept %.1f MiB)%s",
           (unsigned long long)frame, (long long)m_live_n, m_live_bytes / M, (long long)m_refused_n[MR_BUDGET],
           m_refused_bytes[MR_BUDGET] / M, (long long)m_refused_n[MR_UNTRACKED], (long long)m_refused_n[MR_OVERLAP],
           (long long)m_refused_n[MR_ALLOC], a.subs / f, a.syncs / f, a.upload_bytes / f / M, a.upload_copies / f,
           a.first_n / f, a.first_bytes / f / M, a.wb_bytes / f / M, a.wb_copies / f, fc / f, fb / f / M,
           a.presnap_queries / f, a.presnap_bytes / f / M, a.sync_queries / f, a.prologues / f, qpc_ms(a.presnap_t + a.sync_t) / f,
           qpc_ms(a.presnap_t) / f, qpc_ms(a.sync_t) / f, a.reup_in / f / M, a.reup_out / f / M, a.reup_kept / f / M, ver);
    vk_wrote_attrib_log();
}

// With AC_TRACE=perf every 600 frames: what holds device-local memory, to see what grows over a long session.
// VMA's blocks and allocations on the device-local heaps, texture entries with their own memory, the VRAM mirrors and
// the objects waiting for their last submission to retire.
static void vram_log(uint64_t frame) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t blocks = 0, block_bytes = 0, allocs = 0, alloc_bytes = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            blocks += vb[h].statistics.blockCount, block_bytes += vb[h].statistics.blockBytes,
                allocs += vb[h].statistics.allocationCount, alloc_bytes += vb[h].statistics.allocationBytes;
    VkTexCensus t;
    vk_texture_census(&t);
    const double M = 1048576.0;
    rt_log("mem: vram frame %llu: vma %llu blocks %.0f MiB, %llu allocations %.0f MiB; textures live %llu (own memory %llu, "
           "%.0f MiB), released %llu (own memory %llu, %.0f MiB); mirrors %lld %.0f MiB; releases pending %d; cache %.0f of "
           "%.0f MiB, %llu dropped (%.0f MiB)",
           (unsigned long long)frame, (unsigned long long)blocks, block_bytes / M, (unsigned long long)allocs,
           alloc_bytes / M, (unsigned long long)t.live, (unsigned long long)t.own_live, t.bytes_live / M,
           (unsigned long long)t.released, (unsigned long long)t.own_released, t.bytes_released / M, (long long)m_live_n,
           m_live_bytes / M, dead_pending(), (t.bytes_live + t.bytes_released) / M, t.cache_budget / M,
           (unsigned long long)t.cache_drops, t.cache_dropped / M);
    rt_log("mem:   astc 2D with mips, larger side >= 2048 / 1024 / 512: %u %.0f MiB / %u %.0f MiB / %u %.0f MiB",
           t.astc_n[0], t.astc_bytes[0] / M, t.astc_n[1], t.astc_bytes[1] / M, t.astc_n[2], t.astc_bytes[2] / M);
    vma_detail_log();
}

// Device class: INTEGRATED -> uma; a discrete GPU with a DEVICE_LOCAL heap of 1 GiB or more beside a
// heap that is not device-local (host memory) -> discrete; otherwise uma (import everywhere). Sets the mirror mode.
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m) {
    int big_local = -1, host_heap = -1;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        if (m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (m->memoryHeaps[h].size >= (1ull << 30) && big_local < 0) big_local = (int)h;
        } else if (host_heap < 0) host_heap = (int)h;
    }
    static const char *const types[] = {"other", "integrated", "discrete", "virtual", "cpu"};
    int t = (int)p->deviceType;
    const char *tn = t >= 0 && t <= 4 ? types[t] : "unknown";
    const char *cls, *why;
    if (p->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) cls = "uma", why = "integrated gpu";
    else if (p->deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) cls = "uma", why = "not a discrete gpu";
    else if (big_local < 0) cls = "uma", why = "no device-local heap of 1 GiB or more";
    else if (host_heap < 0) cls = "uma", why = "every heap is device-local";
    else cls = "discrete", why = "device-local heap separate from host memory";
    int discrete = cls[0] == 'd';
    char line[2048];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: class %s (%s; device type %d %s, %s)", cls, why, t, tn, p->deviceName);
    for (uint32_t h = 0; h < m->memoryHeapCount; h++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; heap %u %.0f MiB flags %#x%s%s", h,
                              m->memoryHeaps[h].size / 1048576.0, m->memoryHeaps[h].flags,
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT ? " MULTI_INSTANCE" : "");
    rt_log("%s", line);
    k = (size_t)snprintf(line, sizeof line, "mem: types");
    for (uint32_t i = 0; i < m->memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m->memoryTypes[i].propertyFlags;
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; %u heap %u flags %#x%s%s%s%s%s", i,
                              m->memoryTypes[i].heapIndex, f, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HOST_VISIBLE" : "",
                              f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " HOST_COHERENT" : "",
                              f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " HOST_CACHED" : "",
                              f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? " LAZY" : "");
    }
    rt_log("%s", line);
    mirror_mode(discrete);
}

// ---------------------------------------------------------------- upload batches and deferred destruction
typedef struct { VkBuffer buf; VmaAllocation alloc; } Staging;
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    uint64_t seq;
    Staging *stg;
    int nstg, cap;
} Batch;

enum { DEAD_IMAGE = 1, DEAD_IMAGE_VMA, DEAD_VIEW, DEAD_BUFFER_VMA };
typedef struct { int kind; uint64_t handle; VmaAllocation alloc; uint64_t seq, tl; } Dead;

static PlatRMutex cmd_lock;
static PlatOnce cmd_once = PLAT_ONCE_INIT;
static VkCommandPool cmd_pool;
static Batch open_b;
static int open_active;
static Batch *inflight;
static int ninflight, cap_inflight;
static Batch *spare;                  // completed batches whose command buffer and fence are reused
static int nspare, cap_spare;
static uint64_t next_seq, done_seq;   // batches handed out / completed (all batches up to done_seq)
static Dead *dead;
static int ndead, cap_dead;
static uint64_t tl_noted;             // newest timeline value vk_sync.c reported submitted

static void cmd_init(void *p) {
    (void)p;
    return;
}

static void grow(void **arr, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 16;
    void *n = realloc(*arr, (size_t)*cap * elem);
    if (!n) rt_fatal("vk: out of memory (upload bookkeeping)");
    *arr = n;
}

static void free_staging(VkCtx *c, Batch *b) {
    for (int i = 0; i < b->nstg; i++) vmaDestroyBuffer(c->vma, b->stg[i].buf, b->stg[i].alloc);
    b->nstg = 0;
}

// Batches complete in submission order (one queue): retire from the front while their fences are signalled.
static void retire(VkCtx *c, int wait_all) {
    int k = 0;
    for (; k < ninflight; k++) {
        Batch *b = &inflight[k];
        if (wait_all) vkWaitForFences(c->device, 1, &b->fence, VK_TRUE, UINT64_MAX);
        else if (vkGetFenceStatus(c->device, b->fence) != VK_SUCCESS) break;
        free_staging(c, b);
        done_seq = b->seq;
        if (nspare == cap_spare) grow((void **)&spare, &cap_spare, sizeof *spare);
        spare[nspare++] = *b;
    }
    memmove(inflight, inflight + k, (size_t)(ninflight - k) * sizeof *inflight);
    ninflight -= k;
}

static void collect_locked(VkCtx *c) {
    retire(c, 0);
    uint64_t tl = 0;
    if (c->timeline) vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
    int k = 0;
    for (int i = 0; i < ndead; i++) {
        Dead *d = &dead[i];
        if (d->seq > done_seq || d->tl > tl) { dead[k++] = *d; continue; }
        switch (d->kind) {
        case DEAD_IMAGE: vkDestroyImage(c->device, (VkImage)(uintptr_t)d->handle, NULL); break;
        case DEAD_IMAGE_VMA: vmaDestroyImage(c->vma, (VkImage)(uintptr_t)d->handle, d->alloc); break;
        case DEAD_VIEW: vkDestroyImageView(c->device, (VkImageView)(uintptr_t)d->handle, NULL); break;
        case DEAD_BUFFER_VMA: vmaDestroyBuffer(c->vma, (VkBuffer)(uintptr_t)d->handle, d->alloc); break;
        }
    }
    ndead = k;
}

VkCommandBuffer vk_upload_begin(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    if (!c) rt_fatal("vk: upload before vk_init");
    if (open_active) return open_b.cb;
    if (!cmd_pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: upload command pool");
    }
    retire(c, 0);
    memset(&open_b, 0, sizeof open_b);
    if (nspare) {
        Batch s = spare[--nspare];
        open_b.cb = s.cb, open_b.fence = s.fence, open_b.stg = s.stg, open_b.cap = s.cap;
        vkResetCommandBuffer(open_b.cb, 0);
        vkResetFences(c->device, 1, &open_b.fence);
    } else {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(c->device, &ai, &open_b.cb) != VK_SUCCESS ||
            vkCreateFence(c->device, &fi, NULL, &open_b.fence) != VK_SUCCESS)
            rt_fatal("vk: upload command buffer");
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(open_b.cb, &bi);
    open_b.seq = ++next_seq;
    vk_crumb(open_b.cb, VK_CRUMB_UPLOAD, open_b.seq);
    open_active = 1;
    return open_b.cb;
}

void vk_upload_end(void) { plat_rmutex_unlock(&cmd_lock); }

// Staging memory comes from a pool of its own that keeps STG_MIN_BLOCKS blocks from its first use (at boot). In the
// default pools VMA creates a block under the lock of its memory type (VmaBlockVector::Allocate), so a block made by
// the headroom thread (vk_headroom.c) or a burst of uploads held every staging allocation of the render thread: one
// took 47-50 ms, for a few bytes, in the first frame after a save load. Larger buffers than a block go to the default
// pool.
#define STG_BLOCK (64ull << 20)
#define STG_MIN_BLOCKS 2
static VmaPool stg_pool;
static int stg_pool_tried;

// The ASTC decoder's scratch buffers (vk_astc.c vk_astc_record: the raw blocks, the decoded texels and the BC blocks of
// one texture, up to 32 MiB of texels a chunk) live until their upload batch completes. In the default device-local
// pool they took blocks between the texture images created in the same load and left them full of holes when they
// went: in the deck profile, in play on a save, 110 MiB of free ranges in the texture images' blocks, 286 MiB once
// AC_TEX_MIP_SKIP made those images smaller. Their own pool keeps that churn out of the images' blocks. VMA keeps one
// empty block of a pool for the next use (VmaBlockVector::Free), which is the cost: one SCR_BLOCK. A larger scratch
// goes to the default pool.
#define SCR_BLOCK (32ull << 20)
static VmaPool scr_pool;
static int scr_pool_tried;

VmaPool vk_scratch_pool(const VkBufferCreateInfo *bi) {
    VkCtx *c = vk_ctx();
    if (!__atomic_load_n(&scr_pool_tried, __ATOMIC_ACQUIRE)) {
        plat_once(&cmd_once, cmd_init, NULL);
        plat_rmutex_lock(&cmd_lock);
        if (!scr_pool_tried) {
            uint32_t type;
            VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
            VmaPoolCreateInfo pi = {.blockSize = SCR_BLOCK};
            if (vmaFindMemoryTypeIndexForBufferInfo(c->vma, bi, &aci, &type) == VK_SUCCESS) {
                pi.memoryTypeIndex = type;
                if (vmaCreatePool(c->vma, &pi, &scr_pool) != VK_SUCCESS) scr_pool = NULL;
            }
            rt_log("vk: ASTC scratch pool %s (blocks of %llu MiB)", scr_pool ? "created" : "unavailable: default pool",
                   (unsigned long long)(SCR_BLOCK >> 20));
            __atomic_store_n(&scr_pool_tried, 1, __ATOMIC_RELEASE);
        }
        plat_rmutex_unlock(&cmd_lock);
    }
    return bi->size <= SCR_BLOCK ? scr_pool : NULL;
}

// AC_TRACE=perf, with the vram line (vram_log): where VMA's blocks hold memory no allocation uses. One line per memory
// type with blocks: its blocks and allocations, the free ranges between them and the largest one; then the staging
// pool, whose blocks are kept even when empty (STG_MIN_BLOCKS). A dedicated allocation counts as a block of its own
// size, so it adds no slack.
static void vma_detail_log(void) {
    VkCtx *c = vk_ctx();
    static VmaTotalStatistics s;   // large (one entry per memory type and heap); only the census thread calls this
    vmaCalculateStatistics(c->vma, &s);
    const double M = 1048576.0;
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; i++) {
        const VmaDetailedStatistics *d = &s.memoryType[i];
        if (!d->statistics.blockCount) continue;
        rt_log("mem:   vma type %u (heap %u, flags %#x): %u blocks %.1f MiB, %u allocations %.1f MiB, slack %.1f MiB in %u "
               "free ranges (largest %.1f MiB)",
               i, c->mem_props.memoryTypes[i].heapIndex, c->mem_props.memoryTypes[i].propertyFlags, d->statistics.blockCount,
               d->statistics.blockBytes / M, d->statistics.allocationCount, d->statistics.allocationBytes / M,
               (d->statistics.blockBytes - d->statistics.allocationBytes) / M, d->unusedRangeCount,
               d->unusedRangeCount ? d->unusedRangeSizeMax / M : 0.0);
    }
    if (stg_pool) {
        VmaDetailedStatistics p;
        vmaCalculatePoolStatistics(c->vma, stg_pool, &p);
        rt_log("mem:   vma staging pool: %u blocks %.1f MiB, %u allocations %.1f MiB", p.statistics.blockCount,
               p.statistics.blockBytes / M, p.statistics.allocationCount, p.statistics.allocationBytes / M);
    }
    if (scr_pool) {
        VmaDetailedStatistics p;
        vmaCalculatePoolStatistics(c->vma, scr_pool, &p);
        rt_log("mem:   vma ASTC scratch pool: %u blocks %.1f MiB, %u allocations %.1f MiB", p.statistics.blockCount,
               p.statistics.blockBytes / M, p.statistics.allocationCount, p.statistics.allocationBytes / M);
    }
}

void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc) {
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    if (!__atomic_load_n(&stg_pool_tried, __ATOMIC_ACQUIRE)) {
        plat_once(&cmd_once, cmd_init, NULL);
        plat_rmutex_lock(&cmd_lock);
        if (!stg_pool_tried) {
            uint32_t type;
            VmaPoolCreateInfo pi = {.blockSize = STG_BLOCK, .minBlockCount = STG_MIN_BLOCKS};
            if (vmaFindMemoryTypeIndexForBufferInfo(c->vma, &bi, &aci, &type) == VK_SUCCESS) {
                pi.memoryTypeIndex = type;
                if (vmaCreatePool(c->vma, &pi, &stg_pool) != VK_SUCCESS) stg_pool = NULL;
            }
            rt_log("vk: staging pool %s (%d blocks of %llu MiB kept)", stg_pool ? "created" : "unavailable: default pool",
                   STG_MIN_BLOCKS, (unsigned long long)(STG_BLOCK >> 20));
            __atomic_store_n(&stg_pool_tried, 1, __ATOMIC_RELEASE);
        }
        plat_rmutex_unlock(&cmd_lock);
    }
    if (stg_pool && bi.size <= STG_BLOCK / 2) aci.pool = stg_pool;
    VmaAllocationInfo info;
    VkBuffer b;
    VmaAllocation a;
    VkResult r = vmaCreateBuffer(c->vma, &bi, &aci, &b, &a, &info);
    if (r != VK_SUCCESS) rt_fatal("vk: staging buffer of %llu bytes failed (%d)", (unsigned long long)size, (int)r);
    *buf = b;
    if (alloc) *alloc = a;   // the caller destroys it (vmaDestroyBuffer) after vk_upload_flush(1)
    else {
        if (open_b.nstg == open_b.cap) grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
        open_b.stg[open_b.nstg++] = (Staging){b, a};
    }
    return info.pMappedData;
}

uint64_t vk_upload_flush(int wait) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    uint64_t seq = next_seq;
    if (c && open_active) {
        for (int i = 0; i < open_b.nstg; i++) vmaFlushAllocation(c->vma, open_b.stg[i].alloc, 0, VK_WHOLE_SIZE);
        vk_crumb(open_b.cb, VK_CRUMB_UPLOAD_END, open_b.seq);
        vkEndCommandBuffer(open_b.cb);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &open_b.cb};
        vk_queue_lock();
        VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, open_b.fence);
        vk_queue_unlock();
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: upload submit failed (%d)", (int)r);
        }
        if (ninflight == cap_inflight) grow((void **)&inflight, &cap_inflight, sizeof *inflight);
        inflight[ninflight++] = open_b;
        open_active = 0;
    }
    if (c && wait) {
        while (ninflight && done_seq < seq) {
            vkWaitForFences(c->device, 1, &inflight[0].fence, VK_TRUE, UINT64_MAX);
            retire(c, 0);
        }
    }
    if (c) collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
    return seq;
}

// tl: the timeline value the object waits for besides the upload batches; never later than the newest submitted one,
// which TL_NOTED asks for. 0: the upload batches only.
#define TL_NOTED UINT64_MAX

static void defer_tl(int kind, uint64_t handle, VmaAllocation alloc, uint64_t tl) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (ndead == cap_dead) grow((void **)&dead, &cap_dead, sizeof *dead);
    if (tl > tl_noted) tl = tl_noted;
    // The open batch may record into the object: it counts as the newest user.
    dead[ndead++] = (Dead){kind, handle, alloc, open_active ? open_b.seq : next_seq, tl};
    plat_rmutex_unlock(&cmd_lock);
}

static void defer(int kind, uint64_t handle, VmaAllocation alloc) { defer_tl(kind, handle, alloc, TL_NOTED); }

void vk_defer_image_after(VkImage img, VmaAllocation alloc, uint64_t tl) {
    if (img) defer_tl(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc, tl);
}

void vk_defer_buffer_upload(VkBuffer buf, VmaAllocation alloc) {
    if (buf) defer_tl(DEAD_BUFFER_VMA, (uint64_t)(uintptr_t)buf, alloc, 0);
}

static int dead_pending(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    int n = ndead;
    plat_rmutex_unlock(&cmd_lock);
    return n;
}

void vk_defer_image(VkImage img, VmaAllocation alloc) {
    if (img) defer(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc);
}
void vk_defer_view(VkImageView view) { if (view) defer(DEAD_VIEW, (uint64_t)(uintptr_t)view, NULL); }
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc) {
    if (buf) defer(DEAD_BUFFER_VMA, (uint64_t)(uintptr_t)buf, alloc);
}

void vk_mem_note_submit(uint64_t timeline_value) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (timeline_value > tl_noted) tl_noted = timeline_value;
    plat_rmutex_unlock(&cmd_lock);
}

void vk_mem_collect(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
}

// Video memory full (vk_texture.c tex_reclaim): the deferred objects are freed once the upload batches and the frames
// in flight are done with them, which the next collect would see a frame or two later; the allocation that failed needs
// their memory now. That is the ASTC decoder's scratch buffers of the textures this submit already uploaded
// (vk_defer_buffer_upload: a load decodes tens of them into one batch, each with a scratch of up to 32 MiB and more),
// and the images of a texture cache purge (vk_defer_image_after). The open batch goes to the queue (a deferred object
// counts it as a user), then the CPU waits for it and for timeline value tl. tl must be a submitted value
// (vk_sync_issued): the value of a submission still being recorded is only signalled once its recording, maybe the
// caller's own thread, is done. The wait is bounded: the frames in flight end within a few frame times, and a GPU that
// does not (a hang) leaves the objects to a later collect instead of holding the caller.
#define RECLAIM_WAIT_NS 500000000ull

int vk_mem_reclaim(uint64_t tl) {
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    plat_rmutex_lock(&cmd_lock);
    int before = ndead;   // before the flush, whose collect may free some of them already
    plat_rmutex_unlock(&cmd_lock);
    vk_upload_flush(1);
    if (c->timeline && tl) {
        VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                                  .pSemaphores = &c->timeline, .pValues = &tl};
        vkWaitSemaphores(c->device, &wi, RECLAIM_WAIT_NS);
    }
    plat_rmutex_lock(&cmd_lock);
    collect_locked(c);
    int freed = before - ndead;
    plat_rmutex_unlock(&cmd_lock);
    return freed > 0 ? freed : 0;
}

void vk_mem_stats(VkMemStats *s) {
    memset(s, 0, sizeof *s);
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT) s->imported_pools++;
        else if (r->kind == POOL_PHYS) s->physical_pools++;
        else s->virtual_pools++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    s->import_failures = (uint64_t)n_import_failures;
    s->upload_batches = next_seq;
}

void vk_mem_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    vk_upload_flush(1);
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    // Ranges still held get their bytes while their mirrors exist; then the read-back objects go.
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < nm; i++) mem_thaw(mlist[i]->import_lo, mlist[i]->import_hi - mlist[i]->import_lo);
    plat_mutex_unlock_shared(&pool_lock);
    plat_mutex_lock(&rb_lock);
    if (rb_pool) {
        vkDestroyFence(c->device, rb_fence, NULL);
        vkDestroyCommandPool(c->device, rb_pool, NULL);
        rb_pool = VK_NULL_HANDLE, rb_cb = VK_NULL_HANDLE, rb_fence = VK_NULL_HANDLE;
    }
    plat_mutex_unlock(&rb_lock);
    vk_texture_shutdown();
    plat_rmutex_lock(&cmd_lock);
    retire(c, 1);
    uint64_t keep = tl_noted;
    tl_noted = 0;
    for (int i = 0; i < ndead; i++) dead[i].seq = 0, dead[i].tl = 0;   // the device is idle
    collect_locked(c);
    tl_noted = keep;
    for (int i = 0; i < nspare; i++) {
        vkDestroyFence(c->device, spare[i].fence, NULL);
        free(spare[i].stg);
    }
    nspare = 0;
    if (cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
    cmd_pool = VK_NULL_HANDLE;
    if (stg_pool) vmaDestroyPool(c->vma, stg_pool);   // after retire: no staging buffer is left in it
    stg_pool = NULL;
    stg_pool_tried = 0;
    if (scr_pool) vmaDestroyPool(c->vma, scr_pool);   // its scratch buffers went with the dead list above
    scr_pool = NULL;
    scr_pool_tried = 0;
    plat_rmutex_unlock(&cmd_lock);
    plat_mutex_lock(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) destroy_rec(c, pool_index.e[i].buf);
    pool_index.n = 0;
    nm = npend = nwbl = 0;   // mirrors went with their pools
    plat_mutex_lock(&shadow_lock);
    nstouch = nspend = 0;   // shadows too
    s_live_n = s_live_bytes = 0;
    plat_mutex_unlock(&shadow_lock);
    for (int i = 0; i < nretiring; i++) destroy_rec(c, retiring[i]);   // finalized ones whose callback never ran
    nretiring = 0;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    nphys = 0;
    plat_mutex_unlock(&pool_lock);
    plat_mutex_lock(&gen_lock);
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) free(gen_chunk[i]), gen_chunk[i] = NULL;
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) free(gpu_chunk[i]), gpu_chunk[i] = NULL;
    plat_mutex_unlock(&gen_lock);
}
