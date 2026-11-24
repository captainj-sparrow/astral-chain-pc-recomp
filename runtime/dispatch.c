// Indirect-branch dispatch: guest address -> lifted function, O(1).
//
// Layout: one table per module, one uint32_t slot per 4-byte guest byte over the module's [base, base + size)
// range. A slot is 0 when no function starts there, else the host function's address minus `anchor` (an address just
// below the lowest host function), so 4 bytes are enough as long as all lifted code lies within 4 GiB, which a single
// PE image guarantees. Memory: 4 bytes per 4 guest bytes of each module's size. The gaps between modules (the
// fixed-address RomFS image sits between main and sdk) hold no table: a lookup first finds the module by base (two
// modules, two compares), then indexes that module's table, so the flat-table span limit no longer applies. The gaps
// inside a module (rodata, data, bss, alignment) are reserved but never written, so the OS never backs them with RAM.
//
// The tables are filled once by rt_dispatch_init and then made read-only, so any number of threads read them without
// locks. Host (HLE) functions at addresses outside the modules live in a small hash table that is also fixed at init;
// they are looked up only after the module tables miss.
//
// Each thread keeps the targets it calls in a small cache of its own in front of the tables ("target cache" below):
// a slot of a table shares its cache line and its page with no other target a thread calls.
#include "dispatch.h"
#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TABLE_SPAN_LIMIT (1ull << 30)   // guest bytes covered by ONE module table, and its size in bytes
#define HOST_CAP 8192                   // host function registry, power of two
#define MAX_MODULES 16

// Hot data read by every rt_dispatch: one cache line.
static struct __attribute__((aligned(64))) {
    uintptr_t anchor;
} D;

static const RtModule *g_mods[MAX_MODULES];   // sorted by base
static struct { const uint32_t *slot; uint64_t base, slots, bytes; } g_t[MAX_MODULES];   // per-module tables, g_mods order
static int g_nmods;
static bool g_ready;
static bool dc_off;   // AC_DISPATCH_CACHE=0, read by rt_dispatch_init: the threads' target caches stay empty
static uint64_t g_table_bytes, g_functions, g_lo, g_hi, g_slots;

static struct { uint64_t addr; GuestFn fn; } g_host[HOST_CAP];
static uint64_t g_nhost;

void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    rt_log_raw("FATAL: ", 7);
    rt_vlogf(fmt, ap);
    rt_log_raw("\n", 1);
    va_end(ap);
    // A failed allocation (malloc, realloc, VirtualAlloc) means the commit limit was reached: this line tells a leak of
    // ours (process private bytes near the limit) from a machine short of page file.
    double proc, used, limit;
    plat_commit_info(&proc, &used, &limit);
    rt_log("fatal: memory commit: this process %.0f MB, the system %.0f of %.0f MB", proc, used, limit);
    kern_free_cores_note();
    plat_exit(70);   // flushes stdout and stderr; exits like _Exit, or without the DLLs' unload handlers when asked
}

static const RtModule *module_at(uint64_t pc) {
    for (int i = 0; i < g_nmods; i++)
        if (pc - g_mods[i]->base < g_mods[i]->size) return g_mods[i];
    return NULL;
}

const char *rt_guest_name(uint64_t pc, char *buf, int size) {
    const RtModule *m = module_at(pc);
    if (m) snprintf(buf, size, "%#llx (%s+%#llx)", (unsigned long long)pc, m->name, (unsigned long long)(pc - m->base));
    else snprintf(buf, size, "%#llx", (unsigned long long)pc);
    return buf;
}

static uint64_t host_hash(uint64_t addr) { return (addr >> 2) * 0x9E3779B97F4A7C15ull >> 51; }   // 13 bits

static GuestFn host_lookup(uint64_t addr) {
    for (uint64_t h = host_hash(addr), n = 0; n < HOST_CAP; h = (h + 1) & (HOST_CAP - 1), n++) {
        if (g_host[h].addr == addr) return g_host[h].fn;
        if (!g_host[h].addr) return NULL;
    }
    return NULL;
}

void rt_register_host(uint64_t addr, GuestFn fn) {
    if (g_ready) rt_fatal("rt_register_host(%#llx) after rt_dispatch_init", (unsigned long long)addr);
    if (!addr || (addr & 3) || !fn) rt_fatal("rt_register_host: bad entry %#llx", (unsigned long long)addr);
    if (g_nhost >= HOST_CAP / 2) rt_fatal("rt_register_host: more than %d host functions", HOST_CAP / 2);
    uint64_t h = host_hash(addr);
    while (g_host[h].addr && g_host[h].addr != addr) h = (h + 1) & (HOST_CAP - 1);
    if (g_host[h].addr) rt_fatal("rt_register_host: %#llx registered twice", (unsigned long long)addr);
    g_host[h].addr = addr;
    g_host[h].fn = fn;
    g_nhost++;
}

static void *table_alloc(uint64_t bytes) {
    return plat_mem_alloc(bytes);   // zero pages, backed on first write
}

static void table_seal(void *p, uint64_t bytes) {
    if (!plat_mem_protect((uint64_t)(uintptr_t)p, bytes, PLAT_PROT_READ)) rt_fatal("dispatch: sealing the table failed");
}

void rt_dispatch_init(const RtModule *mods, int count) {
    if (g_ready) rt_fatal("rt_dispatch_init called twice");
    if (count <= 0 || count > MAX_MODULES) rt_fatal("rt_dispatch_init: %d modules", count);

    // Modules by base, checked for overlap.
    for (int i = 0; i < count; i++) {
        int j = g_nmods++;
        while (j > 0 && g_mods[j - 1]->base > mods[i].base) { g_mods[j] = g_mods[j - 1]; j--; }
        g_mods[j] = &mods[i];
    }
    for (int i = 1; i < g_nmods; i++)
        if (g_mods[i - 1]->base + g_mods[i - 1]->size > g_mods[i]->base)
            rt_fatal("dispatch: modules %s and %s overlap", g_mods[i - 1]->name, g_mods[i]->name);

    // Function tables: sorted, aligned, inside their module. Range of guest starts and of host functions.
    uint64_t lo = UINT64_MAX, hi = 0;
    uintptr_t host_lo = UINTPTR_MAX, host_hi = 0;
    for (int i = 0; i < g_nmods; i++) {
        const RtModule *m = g_mods[i];
        for (uint64_t k = 0; k < m->count; k++) {
            uint64_t a = m->addrs[k];
            uintptr_t f = (uintptr_t)m->fns[k];
            if ((a & 3) || a - m->base >= m->size)
                rt_fatal("dispatch: %s_fn_addrs[%llu] = %#llx is misaligned or outside the module", m->name,
                         (unsigned long long)k, (unsigned long long)a);
            if (k && a <= m->addrs[k - 1])
                rt_fatal("dispatch: %s_fn_addrs is not strictly ascending at [%llu] = %#llx", m->name,
                         (unsigned long long)k, (unsigned long long)a);
            if (!f) rt_fatal("dispatch: %s_fn_ptrs[%llu] is NULL", m->name, (unsigned long long)k);
            if (f < host_lo) host_lo = f;
            if (f > host_hi) host_hi = f;
        }
        if (m->count) {
            if (m->addrs[0] < lo) lo = m->addrs[0];
            if (m->addrs[m->count - 1] + 4 > hi) hi = m->addrs[m->count - 1] + 4;
        }
    }
    if (hi <= lo) rt_fatal("dispatch: no functions in any module");
    uintptr_t anchor = (host_lo & ~(uintptr_t)15) - 16;
    if (host_hi - anchor > UINT32_MAX)
        rt_fatal("dispatch: lifted code spans %#llx host bytes; slots hold 32-bit offsets (4 GiB)",
                 (unsigned long long)(host_hi - anchor));

    for (uint64_t h = 0; h < HOST_CAP; h++)
        if (g_host[h].addr && module_at(g_host[h].addr))
            rt_fatal("dispatch: host function at %#llx lies inside module %s", (unsigned long long)g_host[h].addr,
                     module_at(g_host[h].addr)->name);

    g_lo = lo;
    g_hi = hi;
    D.anchor = anchor;
    for (int i = 0; i < g_nmods; i++) {
        const RtModule *m = g_mods[i];
        if (m->size > TABLE_SPAN_LIMIT)
            rt_fatal("dispatch: module %s spans %#llx guest bytes, over the %#llx one table allows", m->name,
                     (unsigned long long)m->size, TABLE_SPAN_LIMIT);
        uint64_t slots = m->size >> 2, bytes = (slots * 4 + 0xFFFF) & ~0xFFFFull;
        uint32_t *slot = table_alloc(bytes);
        if (!slot) rt_fatal("dispatch: cannot allocate the %llu-byte table of %s", (unsigned long long)bytes, m->name);
        for (uint64_t k = 0; k < m->count; k++)
            slot[(m->addrs[k] - m->base) >> 2] = (uint32_t)((uintptr_t)m->fns[k] - anchor);
        table_seal(slot, bytes);
        g_t[i] = (typeof(g_t[0])){slot, m->base, slots, bytes};
        g_table_bytes += bytes;
        g_slots += slots;
        g_functions += m->count;
    }
    const char *dce = cfg_get("AC_DISPATCH_CACHE");
    dc_off = dce && !strcmp(dce, "0");
    __atomic_store_n(&g_ready, true, __ATOMIC_RELEASE);
}

GuestFn rt_dispatch_lookup(uint64_t pc) {
    for (int i = 0; i < g_nmods; i++) {
        uint64_t j = __builtin_rotateright64(pc - g_t[i].base, 2);   // a pc outside the module lands far above its slots
        if (j < g_t[i].slots && g_t[i].slot[j]) return (GuestFn)(D.anchor + g_t[i].slot[j]);
    }
    return g_ready ? host_lookup(pc) : NULL;
}

// ---------------------------------------------------------------- target cache
// The module tables hold one slot per 4 guest bytes, so the slots of two functions share a cache line only when the
// functions start within 64 bytes of each other, and the slots a thread reads are spread over hundreds of pages of
// a table of 200 MiB. In the town a thread calls 130 to 520 different targets in a million indirect calls, 57 000 to
// 78 000 calls a frame on each of the three busiest threads: the slots of the targets called many times a frame stay
// in the L1, those of the others come from memory again every frame, with a page walk (on a Z1 Extreme at 13 W, 29%
// of the sampled loads of main's table by the game thread came from DRAM after a miss of both TLBs, and rt_dispatch
// took 2.1% of that thread's cycles).
// So each thread keeps the targets it called in DC_SIZE entries of its own, 16 KiB: an entry shares its line with
// three others the same thread calls, and the whole cache takes four pages. Direct mapped on the guest address:
// simulated on the same scene, 1024 entries answer 99.7% or more of the calls of each of the five busiest threads
// (512: 98.3 to 99.95%, 4096: 99.9%), with the address folded once (the entries of functions DC_SIZE * 4 bytes apart
// would collide otherwise: 99.0 to 99.8% at 1024).
// A key is the guest address rotated right by 2, as the tables index it, so a misaligned address matches no entry. The
// tables and the host registry never change once sealed, so an entry never goes stale and nothing invalidates the
// caches. An empty entry must match no address: the key 0 belongs to entry 0 (the address 0, a call through a null
// pointer, which must reach dispatch_fail), so entry 0 starts with a key of another entry and the others with 0.
// AC_DISPATCH_CACHE=0 fills no entry: every call takes the tables, as before.
// On Windows the cache is in the thread's native TLS block (the .tls section, no emulated TLS): rt_dispatch loads
// _tls_index and the block's address before the entry, three loads more than on Linux, and still keeps no stack frame
// (clang 23.1.4 of MSYS2 UCRT64). Measured there with the sampling profiler (AC_PROFILE, 45 s of the world clock
// standing in the first save's scene at 60 fps, a Core Ultra 9 285H, two runs each way, interleaved): rt_dispatch and
// dispatch_miss together take 2.1 and 2.2% of the samples inside the executable with the cache (361 and 375 of
// 17 000), 3.5% both times without it (592 and 599 of 17 200). The frame time does not show it in that scene: the game
// holds 60 fps either way, the render thread 69% of its time in the pacing wait.
#define DC_BITS 10
#define DC_SIZE (1u << DC_BITS)
typedef struct { uint64_t key; GuestFn fn; } DcEntry;
static __thread DcEntry dc[DC_SIZE] = {[0] = {.key = 1}};

static inline uint64_t dc_key(uint64_t pc) { return __builtin_rotateright64(pc, 2); }
static inline DcEntry *dc_entry(uint64_t key) { return &dc[(key ^ (key >> DC_BITS)) & (DC_SIZE - 1)]; }

__attribute__((noinline, noreturn, cold)) static void dispatch_fail(Cpu *c) {
    char at[64], lr[64];
    uint64_t pc = c->pc;
    if (!g_ready) rt_fatal("rt_dispatch to %#llx before rt_dispatch_init", (unsigned long long)pc);
    const char *why = (pc & 3) ? "misaligned address"
                    : module_at(pc) ? "not a function start"
                    : "outside every module, no host function registered";
    rt_fatal("rt_dispatch: no function at guest pc %s: %s (lr %s)", rt_guest_name(pc, at, sizeof at), why,
             rt_guest_name(c->x[30], lr, sizeof lr));
}

// A target this thread's cache does not hold: the tables, then the host functions, then the fatal error. A separate
// tail call keeps the fast path below free of a stack frame.
__attribute__((noinline)) static void dispatch_miss(Cpu *c) {
    uint64_t pc = c->pc;
    uint32_t off = 0;
    for (int i = 0; i < g_nmods && !off; i++) {
        uint64_t j = __builtin_rotateright64(pc - g_t[i].base, 2);   // a pc outside the module lands far above its slots
        off = j < g_t[i].slots ? g_t[i].slot[j] : 0;
    }
    GuestFn f = off ? (GuestFn)(D.anchor + off) : g_ready ? host_lookup(pc) : NULL;
    if (__builtin_expect(!f, 0)) dispatch_fail(c);
    if (!off) {   // a host function: where its calls need nothing of hostfn_entry, the cache holds the function itself
        GuestFn direct = hostfn_direct(pc);
        if (direct) f = direct;
    }
    if (g_ready && !dc_off) {   // before the seal a host function may still be registered at an address that has none
        uint64_t key = dc_key(pc);
        DcEntry *e = dc_entry(key);
        // No key with the old function or the new one half written, should anything ever run guest code on this
        // thread from inside this function: the entry matches nothing meanwhile.
        e->key = e == dc;
        e->fn = f;
        e->key = key;
    }
    __attribute__((musttail)) return f(c);
}

// A pending preempt request at an indirect call (a safe point: the guest state is all in Cpu). Out of line, so
// rt_dispatch keeps no stack frame.
static __attribute__((noinline)) void dispatch_preempt(Cpu *c) {
    kern_checkpoint();
    __attribute__((musttail)) return rt_dispatch(c);
}

// Jump to the guest address in c->pc. Always a tail call, so chains of indirect jumps keep the host stack flat. A
// pending preempt request costs two loads and a branch to check.
void rt_dispatch(Cpu *c) {
    if (__builtin_expect(c->thread && kern_preempt_pending(c->thread), 0))
        __attribute__((musttail)) return dispatch_preempt(c);
    uint64_t key = dc_key(c->pc);
    const DcEntry *e = dc_entry(key);
    GuestFn f = e->fn;
    __asm__("" : "+r"(f));   // the load stays here, before the test: the compiler otherwise rebuilds the entry's
                             // address after it (the thread pointer loaded and added) to jump through memory
    if (__builtin_expect(e->key != key, 0)) __attribute__((musttail)) return dispatch_miss(c);
    __attribute__((musttail)) return f(c);
}

RtDispatchStats rt_dispatch_stats(void) {
    return (RtDispatchStats){g_lo, g_hi, g_slots, g_table_bytes, g_functions, g_nhost};
}
