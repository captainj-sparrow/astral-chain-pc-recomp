// macOS (Apple Silicon) implementation of platform.h, memory half: reservations, 4 KiB guest pages on 16 KiB host
// pages, the write watch and the fault handler.
//
// Pages. The guest's pages are 4 KiB and the host's 16 KiB, and protection applies to whole host pages only. A byte
// per 4 KiB page (pstate, as in linux.c) keeps the state the runtime asked for, and plat_mem_state answers from it.
// A host page gets the most permissive state of its 4 KiB pages (platform.h, plat_mem_grain): never stricter than the
// guest asked, so nothing that should run faults, at the cost of a fault that should happen not happening when a host
// page mixes states. The runtime's uses that need the fault work in whole host pages (memory.c holds, romfs.c's
// read-fault paging).
//
// Write watch. Darwin has nothing like GetWriteWatch or userfaultfd, so it is done with protection: a committed
// read-write host page of a watched reservation that nothing wrote since the last reset is kept read-only (armed). The
// first write faults; the handler copies the host page into a snapshot slot (it is still read-only, so the copy is
// what was there), opens it (makes it writable), and the write runs again. A reset query arms the page again first,
// then compares each 4 KiB page with the snapshot, so it reports the 4 KiB pages whose bytes changed, as exactly as
// linux.c and win32.c report written pages; the changes go into a byte per 4 KiB page (wdirty), which keeps those
// outside the query's range for later queries. Without a snapshot (the pool of SNAP_MAX slots ran out, or the host
// page could not be read) the whole host page counts as written. A write that leaves the bytes as they were is not
// reported: nothing changed for the back end to upload.
//
// Every change of a host page's protection happens under its lock bit in hstate: the fault handler takes one host
// page's lock and nothing else (no lock of the runtime's, no allocation), mapping changes and queries take them in
// address order. A query's re-arm and a fault's disarm of the same host page can then not interleave, which would
// leave a written page writable and unmarked.
#include <errno.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include "platform.h"

#define PG 4096ull
#define PS_LIMIT (1ull << 39)
enum { PS_FREE, PS_RESERVED, PS_NONE, PS_READ, PS_RW };   // PS_NONE.. are committed: PLAT_PROT_* + PS_NONE
enum { HS_LOCK = 1, HS_MAPPED = 2, HS_WATCH = 4, HS_OPEN = 8 };   // HS_OPEN: written since it was last armed
#define RUN_MAX 4096u   // host pages one mapping change locks at a time
#define SNAP_MAX 32768u // snapshot slots: 512 MiB reserved, backed as used

static uint8_t *pstate;            // per 4 KiB page below PS_LIMIT
static uint8_t *hstate;            // per host page below PS_LIMIT: HS_*
static uint64_t hpg, hsh, nsub;    // host page size, its log2, 4 KiB pages per host page
static uint8_t *wdirty;            // per 4 KiB page: its bytes changed since the last reset (settled, see settle)
static uint32_t *hsnap;            // per host page: its snapshot slot + 1, 0 for none
static uint8_t *snaps;             // the snapshot slots
static uint32_t snap_free[SNAP_MAX], snap_nfree, snap_next;
static volatile uint32_t snap_lock;   // a leaf: taken with a host page's lock held, around the slot lists only
static PlatMutex vm_lock = PLAT_MUTEX_INIT;   // mapping changes, one at a time
static pthread_once_t tables_once = PTHREAD_ONCE_INIT;

static void tables_init(void) {
    hpg = (uint64_t)vm_page_size;
    for (hsh = 12; (1ull << hsh) < hpg; hsh++) {}
    nsub = hpg / PG;
    void *p = mmap(NULL, PS_LIMIT / PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    void *h = mmap(NULL, PS_LIMIT >> hsh, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    void *w = mmap(NULL, PS_LIMIT / PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    void *hs = mmap(NULL, (PS_LIMIT >> hsh) * 4, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    void *sn = mmap(NULL, SNAP_MAX * hpg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED || h == MAP_FAILED || w == MAP_FAILED || hs == MAP_FAILED || sn == MAP_FAILED) {
        fprintf(stderr, "FATAL: platform: no memory for the page tables\n");
        _exit(70);
    }
    pstate = p, hstate = h, wdirty = w, hsnap = hs, snaps = sn;
}

static void tables(void) { pthread_once(&tables_once, tables_init); }

uint64_t plat_mem_grain(void) {
    tables();
    return hpg;
}

static void hs_lock(uint64_t h) {
    for (;;) {
        uint8_t s = __atomic_load_n(&hstate[h], __ATOMIC_RELAXED);
        if (!(s & HS_LOCK) &&
            __atomic_compare_exchange_n(&hstate[h], &s, s | HS_LOCK, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        plat_pause();
    }
}

static void hs_unlock(uint64_t h) { __atomic_fetch_and(&hstate[h], (uint8_t)~HS_LOCK, __ATOMIC_RELEASE); }
static void hs_set(uint64_t h, uint8_t bits) { __atomic_fetch_or(&hstate[h], bits, __ATOMIC_RELAXED); }
static void hs_clear(uint64_t h, uint8_t bits) { __atomic_fetch_and(&hstate[h], (uint8_t)~bits, __ATOMIC_RELAXED); }

static void ps_set(uint64_t addr, uint64_t n, uint8_t v) { memset(pstate + addr / PG, v, (size_t)(n / PG)); }

// The protection host page h gets (its lock held): the most permissive of its 4 KiB pages, read-only while armed.
static int host_prot(uint64_t h) {
    int best = PS_NONE;
    for (uint64_t i = h * nsub, e = i + nsub; i < e; i++)
        if (pstate[i] > best) best = pstate[i];
    uint8_t s = __atomic_load_n(&hstate[h], __ATOMIC_RELAXED);
    if (best == PS_RW && (s & HS_WATCH) && !(s & HS_OPEN)) best = PS_READ;
    return best == PS_RW ? PROT_READ | PROT_WRITE : best == PS_READ ? PROT_READ : PROT_NONE;
}

// Applies host_prot to the host pages [h0, h1), whose locks the caller holds, in one mprotect per run of equal
// protection. 1 when every call succeeded.
static int hosts_apply(uint64_t h0, uint64_t h1) {
    int ok = 1;
    for (uint64_t a = h0, b; a < h1; a = b) {
        int prot = host_prot(a);
        for (b = a + 1; b < h1 && host_prot(b) == prot; b++) {}
        ok &= !mprotect((void *)(uintptr_t)(a << hsh), (size_t)((b - a) << hsh), prot);
    }
    return ok;
}

// ---- snapshots (the write watch). All with host page h's lock held.

static void sl_lock(void) {
    while (__atomic_exchange_n(&snap_lock, 1, __ATOMIC_ACQUIRE)) plat_pause();
}
static void sl_unlock(void) { __atomic_store_n(&snap_lock, 0, __ATOMIC_RELEASE); }
static uint8_t *snap_of(uint64_t h) { return hsnap[h] ? snaps + (uint64_t)(hsnap[h] - 1) * hpg : NULL; }

// Host page h's bytes into a snapshot slot (h must be readable), unless it has one; 0 when the pool is empty.
static int snap_take(uint64_t h) {
    if (hsnap[h]) return 1;
    sl_lock();
    uint32_t slot = snap_nfree ? snap_free[--snap_nfree] : snap_next < SNAP_MAX ? ++snap_next : 0;
    sl_unlock();
    if (!slot) return 0;
    hsnap[h] = slot;
    memcpy(snap_of(h), (const void *)(uintptr_t)(h << hsh), hpg);
    return 1;
}

static void snap_drop(uint64_t h) {
    uint32_t slot = hsnap[h];
    if (!slot) return;
    hsnap[h] = 0;
    sl_lock();
    snap_free[snap_nfree++] = slot;
    sl_unlock();
}

// Opens host page h for writes (the caller applies its protection), with a snapshot to tell later which 4 KiB pages
// changed.
static void open_page(uint64_t h) {
    if (hstate[h] & HS_OPEN) return;
    snap_take(h);   // still armed, so readable; without a slot the whole host page will count as written
    hs_set(h, HS_OPEN);
}

// Host page h written since it was armed: arms it again, then records which of its 4 KiB pages changed in wdirty and
// drops the snapshot. A host page that cannot be read (inaccessible meanwhile) counts as written whole: reading it
// here would fault into the handler, which waits for this lock.
static void settle(uint64_t h) {
    if (!(hstate[h] & HS_OPEN)) return;
    hs_clear(h, HS_OPEN);
    hosts_apply(h, h + 1);
    const uint8_t *snap = host_prot(h) != PROT_NONE ? snap_of(h) : NULL;
    for (uint64_t i = 0; i < nsub; i++)
        if (!snap || memcmp((const void *)(uintptr_t)((h << hsh) + i * PG), snap + i * PG, PG)) wdirty[h * nsub + i] = 1;
    snap_drop(h);
}

// Locks the host pages of [addr, addr + n) a run at a time and applies their protection. 1 on success.
static int hosts_update(uint64_t addr, uint64_t n) {
    uint64_t h0 = addr >> hsh, h1 = (addr + n + hpg - 1) >> hsh;
    int ok = 1;
    for (uint64_t a = h0; a < h1;) {
        uint64_t b = a + RUN_MAX < h1 ? a + RUN_MAX : h1;
        for (uint64_t h = a; h < b; h++) hs_lock(h);
        ok &= hosts_apply(a, b);
        for (uint64_t h = a; h < b; h++) hs_unlock(h);
        a = b;
    }
    return ok;
}

// [addr, addr + n) is 4 KiB-aligned, below PS_LIMIT and wholly in our reservations (committed with need_commit).
static int ours(uint64_t addr, uint64_t n, int need_commit) {
    if ((addr | n) & (PG - 1) || !n || addr >= PS_LIMIT || n > PS_LIMIT - addr) return 0;
    for (uint64_t i = addr / PG, e = (addr + n) / PG; i < e; i++)
        if (pstate[i] < (need_commit ? PS_NONE : PS_RESERVED)) return 0;
    return 1;
}

// ---- reservations. mach_vm_allocate with VM_FLAGS_FIXED fails where anything is mapped already, as linux.c's
// MAP_FIXED_NOREPLACE does, so the OS checks the range. Only its first and last host page can be ours already: a
// reservation that does not start or end on a host page shares that host page with its neighbour, mapped once, and
// its 4 KiB pages in the range must be free. (memory.c's search tries thousands of ranges of tens of GiB.)

// 1 when host page h is mapped by us and its 4 KiB pages inside [addr, addr + n) are all free.
static int shared_edge(uint64_t h, uint64_t addr, uint64_t n) {
    if (!(hstate[h] & HS_MAPPED)) return 0;
    uint64_t lo = h << hsh > addr ? h << hsh : addr, hi = (h + 1) << hsh < addr + n ? (h + 1) << hsh : addr + n;
    for (uint64_t i = lo / PG; i < hi / PG; i++)
        if (pstate[i] != PS_FREE) return -1;
    return 1;
}

int plat_mem_reserve(uint64_t addr, uint64_t n, int watch) {
    tables();
    if ((addr | n) & (PG - 1) || !n || addr >= PS_LIMIT || n > PS_LIMIT - addr) return 0;
    plat_mutex_lock(&vm_lock);
    uint64_t h0 = addr >> hsh, h1 = (addr + n + hpg - 1) >> hsh, m0 = h0, m1 = h1;   // [m0, m1): to be mapped
    int e0 = shared_edge(h0, addr, n), e1 = h1 - 1 > h0 ? shared_edge(h1 - 1, addr, n) : e0, ok = e0 >= 0 && e1 >= 0;
    if (e0 > 0) m0++;
    if (e1 > 0 && m1 > m0) m1--;
    if (ok && m1 > m0) {
        mach_vm_address_t at = m0 << hsh;
        ok = mach_vm_allocate(mach_task_self(), &at, (m1 - m0) << hsh, VM_FLAGS_FIXED) == KERN_SUCCESS;
        if (ok && mprotect((void *)(uintptr_t)at, (size_t)((m1 - m0) << hsh), PROT_NONE)) {
            mach_vm_deallocate(mach_task_self(), at, (m1 - m0) << hsh);
            ok = 0;
        }
        for (uint64_t h = m0; ok && h < m1; h++) hs_set(h, HS_MAPPED);
    }
    if (ok) {
        ps_set(addr, n, PS_RESERVED);
        if (watch)
            for (uint64_t h = h0; h < h1; h++) hs_set(h, HS_WATCH);
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

// Below PS_LIMIT the pages count as committed read-write, so plat_mem_protect seals them per host page like guest
// memory (dispatch.c's tables). Above it, plat_mem_protect protects the host pages they wholly cover.
void *plat_mem_alloc(size_t n) {
    tables();
    size_t len = (n + hpg - 1) & ~(size_t)(hpg - 1);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    uint64_t a = (uint64_t)(uintptr_t)p;
    if (a < PS_LIMIT && len <= PS_LIMIT - a) {
        plat_mutex_lock(&vm_lock);
        ps_set(a, len, PS_RW);
        for (uint64_t h = a >> hsh; h < (a + len) >> hsh; h++) hs_set(h, HS_MAPPED);
        plat_mutex_unlock(&vm_lock);
    }
    return p;
}

int plat_mem_commit(uint64_t addr, size_t n) {
    tables();
    plat_mutex_lock(&vm_lock);
    int ok = ours(addr, n, 0);
    if (ok) {
        ps_set(addr, n, PS_RW);
        ok = hosts_update(addr, n);
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_protect(uint64_t addr, uint64_t n, int prot) {
    tables();
    int p = prot == PLAT_PROT_RW ? PROT_READ | PROT_WRITE : prot == PLAT_PROT_READ ? PROT_READ : PROT_NONE;
    plat_mutex_lock(&vm_lock);
    int ok = ours(addr, n, 1);
    if (ok) {
        ps_set(addr, n, (uint8_t)(PS_NONE + prot));
        ok = hosts_update(addr, n);
    }
    plat_mutex_unlock(&vm_lock);
    if (ok || ((addr | n) & (PG - 1)) || !n) return ok;
    // Not ours (memory above PS_LIMIT from plat_mem_alloc): widening takes the host pages it touches, narrowing only
    // those it covers, so no byte outside the range loses access.
    uint64_t lo = prot == PLAT_PROT_RW ? addr & ~(hpg - 1) : (addr + hpg - 1) & ~(hpg - 1);
    uint64_t hi = prot == PLAT_PROT_RW ? (addr + n + hpg - 1) & ~(hpg - 1) : (addr + n) & ~(hpg - 1);
    return hi <= lo || !mprotect((void *)(uintptr_t)lo, (size_t)(hi - lo), p);
}

// Decommitted pages read zero when committed again. A host page left with no committed 4 KiB page is replaced by a
// fresh inaccessible mapping in one mmap per run, which drops its memory; one that keeps a committed neighbour has the
// decommitted pages zeroed in place (and counts as written).
// Host page h (locked) after [lo, hi) inside it was marked reserved: zero the pages of that span that were committed
// (was: their states before), in place, as it keeps a committed page; 0 when it keeps none (the caller remaps it).
static int zero_kept(uint64_t h, uint64_t lo, uint64_t hi, const uint8_t *was, int *ok) {
    int kept = 0;
    for (uint64_t i = h * nsub; i < (h + 1) * nsub; i++) kept |= pstate[i] >= PS_NONE;
    if (!kept) return 0;
    if (mprotect((void *)(uintptr_t)(h << hsh), hpg, PROT_READ | PROT_WRITE)) {
        *ok = 0;
        return 1;
    }
    for (uint64_t x = lo; x < hi; x += PG)
        if (was[(x - lo) / PG] >= PS_NONE) memset((void *)(uintptr_t)x, 0, PG);
    if (hsnap[h]) memcpy(snap_of(h) + (lo - (h << hsh)), (const void *)(uintptr_t)lo, (size_t)(hi - lo));   // not a write
    memset(wdirty + lo / PG, 0, (size_t)((hi - lo) / PG));   // decommitted pages are not written ones, as on Linux
    return 1;
}

int plat_mem_decommit(uint64_t addr, uint64_t n) {
    tables();
    plat_mutex_lock(&vm_lock);
    int ok = ours(addr, n, 0);
    for (uint64_t a = addr >> hsh, h1 = (addr + n + hpg - 1) >> hsh; ok && a < h1;) {
        uint64_t b = a + RUN_MAX < h1 ? a + RUN_MAX : h1;
        for (uint64_t h = a; h < b; h++) hs_lock(h);
        for (uint64_t h = a; h < b;) {
            uint64_t lo = h << hsh > addr ? h << hsh : addr, hi = (h + 1) << hsh < addr + n ? (h + 1) << hsh : addr + n;
            uint8_t was[64];   // nsub <= 64
            memcpy(was, pstate + lo / PG, (size_t)((hi - lo) / PG));
            ps_set(lo, hi - lo, PS_RESERVED);
            if (zero_kept(h, lo, hi, was, &ok)) {
                h++;
                continue;
            }
            // Keeps nothing: one fresh mapping for it and the following host pages the range wholly covers.
            uint64_t e = h + 1;
            for (; e < b && (e + 1) << hsh <= addr + n; e++) ps_set(e << hsh, hpg, PS_RESERVED);
            if (mmap((void *)(uintptr_t)(h << hsh), (size_t)((e - h) << hsh), PROT_NONE,
                     MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == MAP_FAILED)
                ok = 0;
            for (uint64_t x = h; x < e; x++) hs_clear(x, HS_OPEN), snap_drop(x);
            memset(wdirty + h * nsub, 0, (size_t)((e - h) * nsub));
            h = e;
        }
        ok &= hosts_apply(a, b);
        for (uint64_t h = a; h < b; h++) hs_unlock(h);
        a = b;
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_state(uint64_t addr, uint64_t limit, uint64_t *end) {
    tables();
    if (addr >= PS_LIMIT) {
        *end = limit;
        return PLAT_MEM_FREE;
    }
    uint64_t lim = limit < PS_LIMIT ? limit : PS_LIMIT, i = addr / PG;
    uint8_t v = pstate[i];
    while ((i + 1) * PG < lim && pstate[i + 1] == v) i++;
    *end = (i + 1) * PG < lim ? (i + 1) * PG : lim;
    return v == PS_FREE ? PLAT_MEM_FREE : v == PS_RESERVED ? PLAT_MEM_RESERVED : v - PS_NONE;
}

// ---- write watch

// 1 when 4 KiB page i of host page h changed since the last reset: settled into wdirty, or (h open, not settled: a
// query without reset) different from the snapshot, or written at all without one.
static int sub_written(uint64_t h, uint64_t i) {
    if (wdirty[h * nsub + i]) return 1;
    if (!(hstate[h] & HS_OPEN)) return 0;
    const uint8_t *snap = host_prot(h) != PROT_NONE ? snap_of(h) : NULL;
    return !snap || memcmp((const void *)(uintptr_t)((h << hsh) + i * PG), snap + i * PG, PG) != 0;
}

uint64_t plat_mem_written(uint64_t addr, uint64_t n, uint64_t *pages, uint64_t max, int reset) {
    tables();
    if (!max || !n) return 0;
    if (addr >= PS_LIMIT || n > PS_LIMIT - addr) return UINT64_MAX;
    uint64_t lo = addr & ~(PG - 1), hi = (addr + n + PG - 1) & ~(PG - 1);
    uint64_t h0 = lo >> hsh, h1 = (hi + hpg - 1) >> hsh, got = 0;
    if (!(hstate[h0] & HS_WATCH) || !(hstate[h1 - 1] & HS_WATCH)) return UINT64_MAX;   // memory.c: one reservation
    int full = 1;
    for (uint64_t h = h0; h < h1 && full; h++) {
        int any = (__atomic_load_n(&hstate[h], __ATOMIC_RELAXED) & HS_OPEN) != 0;
        for (uint64_t i = 0; i < nsub && !any; i++) any = wdirty[h * nsub + i];
        if (!any) continue;
        // Under the lock: a reset arms the page again before it is compared, so a write meanwhile faults and opens it
        // again; a query without reset compares an open page as it is.
        hs_lock(h);
        if (reset) settle(h);
        uint64_t a = h << hsh > lo ? h << hsh : lo, b = (h + 1) << hsh < hi ? (h + 1) << hsh : hi;
        for (uint64_t x = a; x < b; x += PG) {
            uint64_t i = x / PG - h * nsub;
            if (pstate[x / PG] < PS_NONE) {   // decommitted pages are not written ones
                if (reset) wdirty[h * nsub + i] = 0;
                continue;
            }
            if (!sub_written(h, i)) continue;
            if (got == max) {
                full = 0;   // listed by the next query
                break;
            }
            pages[got++] = x;
            if (reset) wdirty[h * nsub + i] = 0;
        }
        hs_unlock(h);
    }
    return got;
}

// The caller writes the range next: its pages count as written, and their host pages open without a fault (keeping a
// snapshot for the 4 KiB pages outside the range).
void plat_mem_will_write(uint64_t addr, uint64_t n) {
    tables();
    if (!n || addr >= PS_LIMIT || n > PS_LIMIT - addr) return;
    uint64_t lo = addr & ~(PG - 1), hi = (addr + n + PG - 1) & ~(PG - 1);
    if (!(hstate[lo >> hsh] & HS_WATCH) || !(hstate[(hi - 1) >> hsh] & HS_WATCH))
        return;   // outside the watch: its writes take no fault for it anyway
    plat_mutex_lock(&vm_lock);
    for (uint64_t a = lo >> hsh, h1 = (hi + hpg - 1) >> hsh; a < h1;) {
        uint64_t b = a + RUN_MAX < h1 ? a + RUN_MAX : h1;
        for (uint64_t h = a; h < b; h++) {
            hs_lock(h);
            int whole = (h << hsh) >= lo && ((h + 1) << hsh) <= hi;
            if (!(hstate[h] & HS_OPEN)) {
                if (!whole && host_prot(h) != PROT_NONE) snap_take(h);
                hs_set(h, HS_OPEN);
            }
        }
        memset(wdirty + (a << hsh > lo ? a << hsh : lo) / PG, 1,
               (size_t)(((b << hsh < hi ? b << hsh : hi) - (a << hsh > lo ? a << hsh : lo)) / PG));
        hosts_apply(a, b);
        for (uint64_t h = a; h < b; h++) hs_unlock(h);
        a = b;
    }
    plat_mutex_unlock(&vm_lock);
}

// ---- replacing pages. mach_vm_remap with VM_FLAGS_OVERWRITE moves the staging pages in, as linux.c's mremap does:
// other threads see the old pages (and fault on them) or the new ones. A fault meanwhile waits on the host page's lock
// in the handler and then retries. Host pages the range only partly covers (the runtime holds whole host pages, so
// none in practice) are copied in instead.

void *plat_mem_stage(uint64_t n) {
    tables();
    void *p = mmap(NULL, (size_t)((n + hpg - 1) & ~(hpg - 1)), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

int plat_mem_replace(uint64_t addr, void *stage, uint64_t n) {
    tables();
    uint64_t len = (n + hpg - 1) & ~(hpg - 1);
    plat_mutex_lock(&vm_lock);
    int ok = ours(addr, n, 1);
    uint64_t h0 = addr >> hsh, h1 = (addr + n + hpg - 1) >> hsh;
    if (ok) {
        for (uint64_t h = h0; h < h1; h++) hs_lock(h);
        // The stage is host-page aligned, so the whole host pages move when addr starts one; the rest is copied.
        uint64_t moved = addr & (hpg - 1) ? 0 : n & ~(hpg - 1);
        if (moved) {
            mach_vm_address_t to = addr;
            vm_prot_t cur, maxp;
            ok = mach_vm_remap(mach_task_self(), &to, moved, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(),
                               (mach_vm_address_t)(uintptr_t)stage, FALSE, &cur, &maxp, VM_INHERIT_COPY) ==
                     KERN_SUCCESS && to == addr;
        }
        if (ok && moved < n) {
            uint64_t a = (addr + moved) & ~(hpg - 1), b = (addr + n + hpg - 1) & ~(hpg - 1);
            ok = !mprotect((void *)(uintptr_t)a, (size_t)(b - a), PROT_READ | PROT_WRITE);
            if (ok) memcpy((void *)(uintptr_t)(addr + moved), (char *)stage + moved, (size_t)(n - moved));
        }
        if (ok) ps_set(addr, n, PS_RW);
        for (uint64_t h = h0; h < h1; h++)
            if (hstate[h] & HS_WATCH) {
                // the pages written whole are written; a host page the range covers only partly keeps its snapshot
                if ((h << hsh) >= addr && ((h + 1) << hsh) <= addr + n) snap_drop(h);
                hs_set(h, HS_OPEN);
            }
        if (ok) memset(wdirty + addr / PG, 1, (size_t)(n / PG));
        ok &= hosts_apply(h0, h1);
        for (uint64_t h = h0; h < h1; h++) hs_unlock(h);
    }
    plat_mutex_unlock(&vm_lock);
    munmap(stage, (size_t)len);   // the moved pages stay where they were moved to
    return ok;
}

// ---- faults. Darwin reports a protection fault as SIGBUS and an access to unmapped memory as SIGSEGV; whether the
// access was a read, a write or an instruction fetch is in the exception syndrome (ESR) of the thread state. One
// handler, on the thread's signal stack: our own host page states first (an armed watch, a race with a protection
// change), then the write-fault hook (memory.c thaws a frozen region), the read-fault hook (memory.c fetches a held
// range, romfs.c fills a page), then the crash hook (main.c), then the default action.

static int (*fault_hook)(uint64_t addr);
static int (*read_fault_hook)(uint64_t addr);
static void (*crash_fn)(const PlatCrash *c);

// 0 read, 1 write, 8 instruction fetch, -1 not an access.
static int access_of(const ucontext_t *u) {
    uint64_t esr = u && u->uc_mcontext ? u->uc_mcontext->__es.__esr : 0;
    uint32_t ec = (uint32_t)(esr >> 26) & 0x3f;
    if (ec == 0x20 || ec == 0x21) return 8;            // instruction abort
    if (ec == 0x24 || ec == 0x25)                      // data abort: WnR, except for cache maintenance (CM)
        return (esr & (1u << 6)) && !(esr & (1u << 8)) ? 1 : 0;
    return -1;
}

// 1 when the fault at a is ours to resolve and was resolved: the 4 KiB page allows the access and its host page now
// does too.
static int own_fault(uint64_t a, int access) {
    if (!pstate || a >= PS_LIMIT || (access != 0 && access != 1)) return 0;
    uint64_t h = a >> hsh;
    if (!(__atomic_load_n(&hstate[h], __ATOMIC_RELAXED) & HS_MAPPED)) return 0;
    hs_lock(h);
    uint8_t s = __atomic_load_n(&pstate[a / PG], __ATOMIC_RELAXED);
    int ok = access ? s == PS_RW : s >= PS_READ;
    if (ok) {
        if (access && (hstate[h] & HS_WATCH)) open_page(h);
        ok = hosts_apply(h, h + 1);   // a failed mprotect goes on to the crash handler instead of faulting forever
    }
    hs_unlock(h);
    return ok;
}

static void fault_handler(int sig, siginfo_t *si, void *ucv) {
    ucontext_t *u = ucv;
    // plat_intr_continue (platform.h): sigreturn restores every register, so only pc changes.
    if (sig == SIGTRAP && __darwin_arm_thread_state64_get_pc(u->uc_mcontext->__ss) == (uintptr_t)plat_intr_continue) {
        uint64_t to = *(const uint64_t *)(uintptr_t)(__darwin_arm_thread_state64_get_sp(u->uc_mcontext->__ss) -
                                                     PLAT_INTR_SLOT);
        __darwin_arm_thread_state64_set_pc_fptr(u->uc_mcontext->__ss, (void *)(uintptr_t)to);
        return;
    }
    uint64_t a = (uint64_t)(uintptr_t)si->si_addr;
    int access = sig == SIGBUS || sig == SIGSEGV ? access_of(u) : -1;
    if (access >= 0 && own_fault(a, access)) return;
    if (access == 1 && fault_hook && fault_hook(a)) return;
    if (access == 0 && read_fault_hook && read_fault_hook(a)) return;
    uintptr_t pc = (uintptr_t)__darwin_arm_thread_state64_get_pc(u->uc_mcontext->__ss);
    Dl_info at = {0}, self;
    if (crash_fn && dladdr((void *)plat_crash_hook, &self)) {
        int mine = dladdr((void *)pc, &at) && at.dli_fbase == self.dli_fbase;
        PlatCrash c = {.what = sig == SIGSEGV || sig == SIGBUS ? "access violation"
                               : sig == SIGILL ? "illegal instruction" : sig == SIGTRAP ? "breakpoint"
                               : "arithmetic exception",
                       .code = (uint32_t)sig, .rip = pc, .offset = pc - (uintptr_t)self.dli_fbase, .access = -1,
                       .base = (uintptr_t)self.dli_fbase};
        if (access >= 0) c.access = access, c.addr = a;
        pthread_getname_np(pthread_self(), c.thread, sizeof c.thread);
        if (!mine) {   // a fault in a library (MoltenVK, Metal): the library, then the frames of ours the thread was in
            const char *name = at.dli_fname && *at.dli_fname ? at.dli_fname : "no module";
            const char *slash = strrchr(name, '/');
            c.module = slash ? slash + 1 : name;
            if (at.dli_fbase) c.offset = pc - (uintptr_t)at.dli_fbase;
            void *f[64];
            int n = backtrace(f, 64), past_handler = 0;
            for (int i = 0; i < n && c.nframes < (int)(sizeof c.frames / sizeof *c.frames); i++) {
                Dl_info fi;
                int in_self = dladdr(f[i], &fi) && fi.dli_fbase == self.dli_fbase;
                if (!in_self) past_handler = 1;
                else if (past_handler) c.frames[c.nframes++] = (uintptr_t)f[i];
            }
        }
        crash_fn(&c);
    }
    signal(sig, SIG_DFL);   // returns into the faulting instruction, which faults again with the default action
}

// A signal stack for the calling thread (macos.c starts every thread with one), so the handler still runs when the
// thread's own stack is exhausted. Never freed, as in linux.c.
void plat_alt_stack(void) {
    stack_t cur;
    if (!sigaltstack(NULL, &cur) && !(cur.ss_flags & SS_DISABLE)) return;
    size_t size = 64 << 10;
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return;
    stack_t ss = {.ss_sp = p, .ss_size = size};
    sigaltstack(&ss, NULL);
}

static void fault_install(void) {
    static int done;
    if (__atomic_load_n(&done, __ATOMIC_ACQUIRE)) return;
    tables();
    void *warm[1];
    backtrace(warm, 1);   // loads what backtrace needs now, not in the handler
    plat_alt_stack();
    struct sigaction sa = {.sa_sigaction = fault_handler, .sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGTRAP, &sa, NULL);   // brk: fiber_entry's end, __builtin_trap
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
}

void plat_mem_fault_hook(int (*fn)(uint64_t addr)) {
    fault_hook = fn;
    fault_install();
}

void plat_mem_read_fault_hook(int (*fn)(uint64_t addr)) {
    read_fault_hook = fn;
    fault_install();
}

void plat_crash_hook(void (*fn)(const PlatCrash *c)) {
    crash_fn = fn;
    fault_install();
}

// ---- reads that do not fault

int plat_mem_read(uint64_t addr, void *out, size_t n) {
    mach_vm_size_t got = 0;
    return mach_vm_read_overwrite(mach_task_self(), (mach_vm_address_t)addr, (mach_vm_size_t)n,
                                  (mach_vm_address_t)(uintptr_t)out, &got) == KERN_SUCCESS && got == n;
}

int plat_mem_readable(uint64_t addr, uint64_t n) {
    uint8_t b;
    uint64_t end = addr + (n ? n : 1);
    for (uint64_t a = addr; a < end; a = (a | (PG - 1)) + 1)
        if (!plat_mem_read(a, &b, 1)) return 0;
    return 1;
}

void *plat_aligned_alloc(size_t size, size_t align) {
    void *p = NULL;
    return posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, size) ? NULL : p;
}

void plat_aligned_free(void *p) { free(p); }
