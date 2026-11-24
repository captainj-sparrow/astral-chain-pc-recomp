// Adapted from LAS Recomp runtime/hle.h and runtime/dispatch.c @ da35b83 (MIT, Copyright (c) 2026 Peiikai).
// Host functions at synthetic guest addresses: the HOST_OVERRIDE / HOST_OBSERVER
// linker tables, the registry behind them, and the calls between host and guest (rt_call, rt_original).
//
// Every registered function is registered as hostfn_entry. In play a thread's target cache holds the function itself
// instead (hostfn_direct, from dispatch_miss; AC_HOSTFN_DIRECT=0 keeps hostfn_entry, which then jumps to it); with
// AC_TRACE=host or AC_CORE_DIAG, and for nn::fs, hostfn_entry stays the target and goes through hostfn_full, which
// counts the call (hostfn_count_calls), traces the first one (AC_TRACE=host) and raises the host depth (AC_CORE_DIAG).
// The registry is fixed by hostfn_seal, which builds the dispatch table; guest threads start after that, so lookups
// need no lock.
//
// Measured for hostfn_direct on 2026-11-21 (ROG Ally at 15 W, 4 cores / 8 threads, Steam Deck profile, plain build,
// town with the camera turning):
// - Before: hostfn_entry was 1.75% of the draw command thread's cycles, 0.40% of the render thread's.
// - Four in-launch A/Bs (throwaway builds switching the cache's targets, 8 s a side, 22-23 blocks each): the draw
//   command thread's CPU time per frame -1.9, -2.4, -1.8 and -2.5% (+-0.8 to 1.1), its instructions -6.1 to -6.6%;
//   the render thread's CPU time unchanged; the frame interval -0.3, -0.4, -0.3 and -0.6% (+-0.2 to 0.3).
// - Twelve launches, two alternated series of six, do not resolve it. The first (with a shared store on every cache
//   miss, since removed) gave the frame +0.7% and the draw thread's CPU time -0.3%, the second +0.1% and -3.2%; the
//   replay thread, which the change does not touch, moved up to 2% in cycles between their sides.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"

#define MAX_HOSTFN 4096   // dispatch.c holds HOST_CAP / 2 host functions
#define MAX_ADDED 64

#ifdef _WIN32
// PE/COFF: the linker sorts .ovr$a < .ovr$m (entries, host.h) < .ovr$z by the text after '$'.
__attribute__((section(".ovr$a"), used)) static const RtHostEntry ovr_start[1];
__attribute__((section(".ovr$z"), used)) static const RtHostEntry ovr_stop[1];
__attribute__((section(".obs$a"), used)) static const RtHostEntry obs_start[1];
__attribute__((section(".obs$z"), used)) static const RtHostEntry obs_stop[1];
#else
// ELF: the linker defines __start_<section> and __stop_<section> around a section named like a C identifier. A
// build without observers has no such section, and both bounds are then NULL.
extern const RtHostEntry __start_ac_ovr[] __attribute__((weak)), __stop_ac_ovr[] __attribute__((weak));
extern const RtHostEntry __start_ac_obs[] __attribute__((weak)), __stop_ac_obs[] __attribute__((weak));
#define ovr_start __start_ac_ovr
#define ovr_stop __stop_ac_ovr
#define obs_start __start_ac_obs
#define obs_stop __stop_ac_obs
#endif

typedef struct {
    const char *name;
    GuestFn fn;
    uint64_t original;   // observers: the guest definition rt_original calls
    uint64_t calls;
    int blocking;        // runs with the virtual core given up (hostfn_blocking)
} Slot;
static Slot slots[MAX_HOSTFN];
static int nslots;
static RtHostEntry added[MAX_ADDED];
static int nadded;
static int sealed;

static GuestFn table_find(const RtHostEntry *start, const RtHostEntry *stop, const char *name) {
    __asm__("" : "+r"(start), "+r"(stop));   // separate objects: keep the compiler from reasoning about them
    for (const RtHostEntry *p = start; p < stop; p++)
        if (p->name && !strcmp(p->name, name)) return p->fn;
    return NULL;
}

GuestFn hostfn_override(const char *name) {
    for (int i = 0; i < nadded; i++)
        if (!strcmp(added[i].name, name)) return added[i].fn;
    return table_find(ovr_start, ovr_stop, name);
}

GuestFn hostfn_observer(const char *name) { return table_find(obs_start, obs_stop, name); }

// Linked HOST_OVERRIDE / HOST_OBSERVER entries (the loader reports how many it installed).
int hostfn_table_size(int observers) {
    const RtHostEntry *p = observers ? obs_start : ovr_start, *e = observers ? obs_stop : ovr_stop;
    __asm__("" : "+r"(p), "+r"(e));
    int n = 0;
    for (; p < e; p++) n += p->name != NULL;
    return n;
}

const char *hostfn_table_name(int observers, int i) {
    const RtHostEntry *p = observers ? obs_start : ovr_start, *e = observers ? obs_stop : ovr_stop;
    __asm__("" : "+r"(p), "+r"(e));
    for (; p < e; p++)
        if (p->name && i-- == 0) return p->name;
    return NULL;
}

void hostfn_add_override(const char *name, GuestFn fn) {
    if (ldr_count()) rt_fatal("hostfn_add_override(%s) after ldr_load", name);
    if (nadded == MAX_ADDED) rt_fatal("hostfn_add_override: more than %d", MAX_ADDED);
    added[nadded++] = (RtHostEntry){name, fn};
}

static void hostfn_entry(Cpu *c);

static uint64_t slot_addr(int i) { return RT_HOSTFN_BASE + 16ull * (uint64_t)i; }

uint64_t hostfn_register(const char *name, GuestFn fn) {
    if (sealed) rt_fatal("hostfn_register(%s) after hostfn_seal", name);
    if (!name || !fn) rt_fatal("hostfn_register: empty entry");
    for (int i = 0; i < nslots; i++)
        if (slots[i].fn == fn && !strcmp(slots[i].name, name)) return slot_addr(i);
    if (nslots == MAX_HOSTFN) rt_fatal("hostfn_register: more than %d host functions", MAX_HOSTFN);
    uint64_t a = slot_addr(nslots);
    if (a >= RT_RET_HOST) rt_fatal("hostfn_register: host-function window full");
    slots[nslots] = (Slot){.name = name, .fn = fn, .blocking = !strncmp(name, "_ZN2nn2fs", 9)};
    rt_register_host(a, hostfn_entry);
    nslots++;
    return a;
}

// Loader: an observer slot and the guest definition it wraps.
uint64_t hostfn_register_observer(const char *name, GuestFn fn, uint64_t original) {
    uint64_t a = hostfn_register(name, fn);
    slots[(a - RT_HOSTFN_BASE) >> 4].original = original;
    return a;
}

const char *hostfn_name(uint64_t addr) {
    if (addr == RT_RET_HOST) return "<return to host>";
    uint64_t i = (addr - RT_HOSTFN_BASE) >> 4;
    if ((addr & 15) || i >= (uint64_t)nslots) return NULL;
    return slots[i].name;
}

__attribute__((noinline, noreturn, cold)) static void bad_entry(Cpu *c) {
    char lr[96];
    rt_fatal("host function call at %#llx: no host function registered there (lr %s)", (unsigned long long)c->pc,
             ldr_describe(c->x[30], lr, sizeof lr));
}

// The host depth. Asynchronous preemption (kernel.c ap_try) may move a core owner to kern_ap_stub only while it runs
// game code with no host frame under it: a host function may hold a host lock or keep state on the host stack while
// guest code it called runs. A host function's own code is never moved: its host rip is outside the lifted code, which
// ap_try checks first. So the depth is raised around the guest code that host code runs, in rt_call and rt_original,
// the only ways host code enters guest code (keep it so: a host function that called a lifted function directly could
// be preempted inside it), and not around every host call. Raising it on every call (a frame in hostfn_entry, a second
// one here and the depth's stores) took 6.4% of the town's draw command thread (0x1007, pinned to virtual core 1 with
// the work that bounds the frame), about 60 000 calls a frame of NVN getters and setters of ten instructions each.
// Jumping straight to the function took that thread's CPU time down by 4.9% and 4.2% (0.37 and 0.39 ms a frame) and
// the frame interval by 1.7% and 0.9% in two in-run A/Bs of the camera turn on the ROG Ally at 14 W (2026-11-20).
// AC_CORE_DIAG still raises it on every call (hostfn_track_calls), for its "host" class of core waits.
// Not a tail call, so the host stack holds this frame while fn runs; the jump out of fn (ExitThread) leaves the depth
// raised on a thread that ends.
__attribute__((noinline)) static void hostfn_tracked(Cpu *c, GuestFn fn) {
    void *t = c->thread;
    int *d = kern_host_depth(t);
    __atomic_store_n(d, *d + 1, __ATOMIC_RELAXED);
    fn(c);
    int n = *d - 1;
    __atomic_store_n(d, n, __ATOMIC_RELAXED);
    if (!n && kern_preempt_pending(t)) kern_host_left(t);
}

// The count per function feeds the probes' statistics and the AC_TRACE=host report; a game run turns it off (main.c)
// and then counts nothing, unless AC_CORE_DIAG sends every call through hostfn_full. That counting is a locked
// increment of a word every guest thread shares: in play the game's threads call the same NVN functions thousands of
// times a frame, the word's cache line moved between their cores on every call, and 4.3% of the process's cycles went
// into this one instruction. Without it (AC_CORE_DIAG) each thread counts its own calls and adds them to the total
// CALL_BATCH at a time, so the total then lags by less than CALL_BATCH per thread.
#define CALL_BATCH 4096
static int counting = 1;
static int track_calls;                   // AC_CORE_DIAG: every call in the host depth
static int full_path = 1;                 // counting || track_calls: hostfn_entry goes through hostfn_full
static int direct_on;                     // AC_HOSTFN_DIRECT, read by hostfn_seal
static int direct_given;                  // hostfn_direct gave a target, which some thread's cache keeps
static uint64_t batched;                  // calls counted per thread and handed over, CALL_BATCH at a time
static _Thread_local uint32_t own_calls;

// A thread's cached target is never invalidated (dispatch.c), so a call that has to go through hostfn_full cannot be
// asked for once one thread may hold a function's own address.
static void need_entry(const char *who) {
    if (__atomic_load_n(&direct_given, __ATOMIC_RELAXED))
        rt_fatal("%s after host calls were dispatched straight to their functions (AC_HOSTFN_DIRECT=0 keeps them in "
                 "hostfn_entry)", who);
}

void hostfn_count_calls(int on) {
    if (on) need_entry("hostfn_count_calls");
    counting = on;
    full_path = counting || track_calls;
}

void hostfn_track_calls(void) {
    need_entry("hostfn_track_calls");
    track_calls = 1;
    full_path = 1;
}

// nn::fs: the call waits for the disk and the OS, and on the Switch it is an IPC request that blocks its thread, so
// the thread gives its virtual core up for it. AC_CORE_DIAG found the game's loader threads (priority 58) holding
// a core through CloseFile and CreateCacheStorage while job workers and better threads waited 2-6 ms for it.
__attribute__((noinline)) static void hostfn_blocking(Cpu *c, GuestFn fn) {
    int away = kern_core_away();
    if (track_calls && c->thread) hostfn_tracked(c, fn);
    else fn(c);
    kern_core_back(&away);
}

__attribute__((noinline)) static void hostfn_full(Cpu *c) {
    Slot *s = &slots[(c->pc - RT_HOSTFN_BASE) >> 4];   // hostfn_entry checked the address
    if (__builtin_expect(counting, 0)) {
        uint64_t n = __atomic_fetch_add(&s->calls, 1, __ATOMIC_RELAXED);
        if (__builtin_expect(!n && (rt_trace & RT_TRACE_HOSTCALL), 0)) {
            char lr[96];
            rt_log("host call %s from %s", s->name, ldr_describe(c->x[30] - 4, lr, sizeof lr));
        }
    } else if (__builtin_expect(++own_calls == CALL_BATCH, 0)) {
        own_calls = 0;
        __atomic_fetch_add(&batched, CALL_BATCH, __ATOMIC_RELAXED);
    }
    if (s->blocking) {
        hostfn_blocking(c, s->fn);
        return;
    }
    if (track_calls && c->thread) {
        hostfn_tracked(c, s->fn);
        return;
    }
    __attribute__((musttail)) return s->fn(c);
}

// No stack frame in play: the slot, then a jump to the function. The rest is in hostfn_full.
static void hostfn_entry(Cpu *c) {
    uint64_t i = (c->pc - RT_HOSTFN_BASE) >> 4;
    if (__builtin_expect((c->pc & 15) || i >= (uint64_t)nslots, 0)) bad_entry(c);
    const Slot *s = &slots[i];
    if (__builtin_expect(full_path | s->blocking, 0)) __attribute__((musttail)) return hostfn_full(c);
    __attribute__((musttail)) return s->fn(c);
}

GuestFn hostfn_direct(uint64_t addr) {
    uint64_t i = (addr - RT_HOSTFN_BASE) >> 4;
    if (!direct_on || full_path || (addr & 15) || i >= (uint64_t)nslots || slots[i].blocking) return NULL;
    if (!__atomic_load_n(&direct_given, __ATOMIC_RELAXED))   // written once: every thread's cache misses come here
        __atomic_store_n(&direct_given, 1, __ATOMIC_RELAXED);
    return slots[i].fn;
}

// A guest jump to the rt_call return sentinel: guest code used x30 as data, which a lifted RET never does.
static void return_sentinel(Cpu *c) {
    rt_fatal("guest code jumped to the host return address %#llx (x0 %#llx)", (unsigned long long)RT_RET_HOST,
             (unsigned long long)c->x[0]);
}

void hostfn_seal(void) {
    static RtModule mods[16];   // kept by the dispatcher
    if (sealed) return;
    rt_register_host(RT_RET_HOST, return_sentinel);
    direct_on = cfg_bool("AC_HOSTFN_DIRECT");
    sealed = 1;
    int n = rt_linked_modules(mods, 16);
    rt_dispatch_init(mods, n);
}

uint64_t hostfn_calls(void) {
    uint64_t n = __atomic_load_n(&batched, __ATOMIC_RELAXED);
    for (int i = 0; i < nslots; i++) n += __atomic_load_n(&slots[i].calls, __ATOMIC_RELAXED);
    return n;
}

static int by_calls(const void *a, const void *b) {
    uint64_t x = (*(const Slot *const *)a)->calls, y = (*(const Slot *const *)b)->calls;
    return x < y ? 1 : x > y ? -1 : 0;
}

void hostfn_report(void) {
    static const Slot *order[MAX_HOSTFN];
    int n = 0;
    for (int i = 0; i < nslots; i++)
        if (__atomic_load_n(&slots[i].calls, __ATOMIC_RELAXED)) order[n++] = &slots[i];
    qsort(order, n, sizeof *order, by_calls);
    rt_log("host calls: %llu in %d of %d host functions", (unsigned long long)hostfn_calls(), n, nslots);
    for (int i = 0; i < n; i++) rt_log("  %10llu  %s", (unsigned long long)order[i]->calls, order[i]->name);
}

// Guest code run from host code, inside the thread's host depth (above, hostfn_tracked), except at the start of a
// thread, a fiber or the game (rt_call_top), where no host function sits under it.
static uint64_t call_guest(Cpu *c, uint64_t fn, int nested) {
    uint64_t lr = c->x[30], pc = c->pc;
    c->x[30] = RT_RET_HOST;
    c->pc = fn;
    rt_fpcr_sync(c);   // guest code may start here on a host thread or fiber that has not run it
    void *t = nested && kern_host_track ? c->thread : NULL;
    int *d = t ? kern_host_depth(t) : NULL;
    if (d) __atomic_store_n(d, *d + 1, __ATOMIC_RELAXED);
    rt_dispatch(c);
    if (d) {
        int n = *d - 1;
        __atomic_store_n(d, n, __ATOMIC_RELAXED);
        if (!n && kern_preempt_pending(t)) kern_host_left(t);
    }
    c->x[30] = lr;
    c->pc = pc;
    return c->x[0];
}

static uint64_t call_args(Cpu *c, uint64_t fn, int n, va_list ap, int nested) {
    if (n < 0 || n > 8) rt_fatal("rt_call: %d arguments", n);
    if (!sealed) hostfn_seal();
    for (int i = 0; i < n; i++) c->x[i] = va_arg(ap, uint64_t);
    return call_guest(c, fn, nested);
}

uint64_t rt_call(Cpu *c, uint64_t fn, int n, ...) {
    va_list ap;
    va_start(ap, n);
    uint64_t r = call_args(c, fn, n, ap, 1);
    va_end(ap);
    return r;
}

uint64_t rt_call_top(Cpu *c, uint64_t fn, int n, ...) {
    va_list ap;
    va_start(ap, n);
    uint64_t r = call_args(c, fn, n, ap, 0);
    va_end(ap);
    return r;
}

uint64_t rt_original(Cpu *c) {
    uint64_t i = (c->pc - RT_HOSTFN_BASE) >> 4;
    if ((c->pc & 15) || i >= (uint64_t)nslots || !slots[i].original)
        rt_fatal("rt_original at %#llx: not inside an observer with a guest definition", (unsigned long long)c->pc);
    return call_guest(c, slots[i].original, 1);
}
