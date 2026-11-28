// Standalone validation of the macOS platform backend (runtime/platform/macos*.c): scheduling, thread interruption,
// fibers (which move between host threads and carry their own FP state), and the memory semantics the runtime relies
// on with 4 KiB guest pages on 16 KiB host pages (memory.c's freezes and holds, the Vulkan back end's write watch).
// It drives platform.h directly and checks observable behaviour. Build:
//
//   SDL="$(brew --prefix sdl3)"
//   clang -std=c11 -O1 -Wall -Wextra -Werror tests/macos_platform_test.c runtime/platform/macos.c \
//       runtime/platform/macos_mem.c runtime/platform/macos_posix.c runtime/platform/common.c \
//       -framework CoreFoundation -framework CoreText -framework IOKit -I"$SDL/include" -L"$SDL/lib" -lSDL3 \
//       -Wl,-rpath,"$SDL/lib" -o build/macos-platform-test && build/macos-platform-test
//
// No game files and no renderer.
#include "../runtime/platform/platform.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
static int checks;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        checks++;                                                                                                      \
        if (!(cond)) {                                                                                                 \
            failures++;                                                                                                \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                                       \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fputc('\n', stderr);                                                                                       \
        }                                                                                                              \
    } while (0)

// ---- scheduling

static PlatMutex g_mtx = PLAT_MUTEX_INIT;
static int g_counter;

typedef struct { atomic_int ran; atomic_int arg_seen; uint32_t tid; } Sched;

// Worker: records its thread id, bumps a shared counter under a lock.
static void worker(void *arg) {
    Sched *s = arg;
    s->tid = plat_thread_id();
    atomic_store(&s->arg_seen, 1);
    for (int i = 0; i < 1000; i++) {
        plat_mutex_lock(&g_mtx);
        g_counter++;   // not atomic: correctness relies on the mutex
        plat_mutex_unlock(&g_mtx);
    }
    atomic_fetch_add(&s->ran, 1);
}

// -- condition variable handoff
typedef struct { PlatMutex m; PlatCond c; int ready; int value; } Cv;

static void cv_producer(void *arg) {
    Cv *cv = arg;
    plat_mutex_lock(&cv->m);
    cv->value = 1234;
    cv->ready = 1;
    plat_cond_signal(&cv->c);
    plat_mutex_unlock(&cv->m);
}
static void cv_producer_broadcast(void *arg) {
    Cv *cv = arg;
    plat_mutex_lock(&cv->m);
    cv->value = 99;
    cv->ready = 1;
    plat_cond_broadcast(&cv->c);
    plat_mutex_unlock(&cv->m);
}
static void cv_consumer(void *arg) {
    Cv *cv = arg;
    plat_mutex_lock(&cv->m);
    while (!cv->ready) plat_cond_wait(&cv->c, &cv->m);
    cv->value += 1;
    plat_mutex_unlock(&cv->m);
}

// -- a worker that blocks until a gate event is set
typedef struct { PlatEvent *gate; } GateArg;
static void gate_worker(void *arg) {
    GateArg *g = arg;
    plat_event_wait(g->gate);
}

// -- a detached worker that signals an event
typedef struct { PlatEvent *e; } SpawnArg;
static void spawn_worker(void *arg) {
    SpawnArg *sp = arg;
    plat_event_set(sp->e);
}

static void once_fn(void *arg) { (*(int *)arg)++; }

static void sched_tests(void) {
    plat_mutex_init(&g_mtx);

    // basic thread create + join
    g_counter = 0;
    Sched s = {0};
    PlatThread *t = plat_thread_new(worker, &s, "sched-test");
    CHECK(t != NULL, "plat_thread_new");
    plat_thread_join(t);
    CHECK(atomic_load(&s.ran) == 1 && atomic_load(&s.arg_seen) == 1, "thread ran once with its arg");
    CHECK(g_counter == 1000, "mutex serialized 1000 increments (got %d)", g_counter);
    CHECK(s.tid != 0 && s.tid != plat_thread_id(), "worker has its own nonzero thread id");

    // N threads * 1000 increments under one lock
    g_counter = 0;
    enum { N = 8 };
    Sched sn[N];
    PlatThread *ts[N];
    for (int i = 0; i < N; i++) { sn[i] = (Sched){0}; ts[i] = plat_thread_new(worker, &sn[i], NULL); }
    for (int i = 0; i < N; i++) plat_thread_join(ts[i]);
    CHECK(g_counter == N * 1000, "%d threads * 1000 serialized increments (got %d)", N, g_counter);

    // join_ns: a fast thread is joined...
    PlatThread *fast = plat_thread_new(worker, &(Sched){0}, NULL);
    CHECK(plat_thread_join_ns(fast, 2000000000ull), "join_ns returns 1 for a finished thread");

    // ...a slow one is not, and t stays valid; then we let it finish and join it
    PlatEvent gate;
    plat_event_init(&gate, 1, 0);
    GateArg ga = { &gate };
    PlatThread *slowt = plat_thread_new(gate_worker, &ga, NULL);
    CHECK(!plat_thread_join_ns(slowt, 1000000ull), "join_ns returns 0 for a still-running thread");
    plat_event_set(&gate);
    plat_thread_join(slowt);
    CHECK(1, "slow thread joined after being let go");

    // detached spawn
    PlatEvent spawned;
    plat_event_init(&spawned, 1, 0);
    SpawnArg sp = { &spawned };
    CHECK(plat_thread_spawn(spawn_worker, &sp, "spawned", 0), "plat_thread_spawn");
    plat_event_wait(&spawned);
    CHECK(1, "detached spawned thread ran");

    // condition variable: signal
    Cv cv;
    plat_mutex_init(&cv.m); plat_cond_init(&cv.c);
    cv.ready = 0; cv.value = 0;
    PlatThread *prod = plat_thread_new(cv_producer, &cv, NULL);
    plat_mutex_lock(&cv.m);
    while (!cv.ready) plat_cond_wait(&cv.c, &cv.m);
    int got = cv.value;
    plat_mutex_unlock(&cv.m);
    plat_thread_join(prod);
    CHECK(got == 1234, "cond signal handoff (got %d)", got);

    // condition variable: broadcast to several waiters
    Cv bcv;
    plat_mutex_init(&bcv.m); plat_cond_init(&bcv.c);
    bcv.ready = 0; bcv.value = 0;
    PlatThread *ws[4];
    for (int i = 0; i < 4; i++) ws[i] = plat_thread_new(cv_consumer, &bcv, NULL);
    PlatThread *bprod = plat_thread_new(cv_producer_broadcast, &bcv, NULL);
    for (int i = 0; i < 4; i++) plat_thread_join(ws[i]);
    plat_thread_join(bprod);
    plat_mutex_lock(&bcv.m);
    CHECK(bcv.value == 99 + 4, "broadcast woke all 4 waiters (value %d)", bcv.value);
    plat_mutex_unlock(&bcv.m);

    // cond_wait_ns timeout
    Cv tcv;
    plat_mutex_init(&tcv.m); plat_cond_init(&tcv.c);
    tcv.ready = 0; tcv.value = 0;
    plat_mutex_lock(&tcv.m);
    int ok = plat_cond_wait_ns(&tcv.c, &tcv.m, 1000000ull);   // 1 ms, nobody signals
    plat_mutex_unlock(&tcv.m);
    CHECK(!ok, "cond_wait_ns times out (returns 0)");

    // reader-writer lock
    PlatMutex rw = PLAT_MUTEX_INIT;
    plat_mutex_init(&rw);
    plat_mutex_lock_shared(&rw);
    CHECK(plat_mutex_trylock_shared(&rw), "second shared lock ok");
    plat_mutex_unlock_shared(&rw);
    CHECK(!plat_mutex_trylock(&rw), "exclusive fails while shared held");
    plat_mutex_unlock_shared(&rw);
    CHECK(plat_mutex_trylock(&rw), "exclusive ok once shared released");
    CHECK(!plat_mutex_trylock_shared(&rw), "shared fails while exclusive held");
    plat_mutex_unlock(&rw);

    // recursive mutex
    PlatRMutex rm = PLAT_RMUTEX_INIT;
    plat_rmutex_lock(&rm); plat_rmutex_lock(&rm);
    plat_rmutex_unlock(&rm); plat_rmutex_unlock(&rm);
    CHECK(1, "recursive lock twice + unlock twice");

    // events (auto-reset + manual-reset)
    PlatEvent ae; plat_event_init(&ae, 0, 0);
    plat_event_set(&ae);
    CHECK(plat_event_wait_ns(&ae, 0), "auto-reset event set");
    CHECK(!plat_event_wait_ns(&ae, 1000000ull), "auto-reset event cleared after one waiter");
    PlatEvent me; plat_event_init(&me, 1, 0);
    plat_event_set(&me);
    CHECK(plat_event_wait_ns(&me, 0) && plat_event_wait_ns(&me, 0), "manual-reset event stays set");
    plat_event_reset(&me);
    CHECK(!plat_event_wait_ns(&me, 1000000ull), "manual-reset event reset");

    // once
    PlatOnce once = PLAT_ONCE_INIT;
    int once_count = 0;
    plat_once(&once, once_fn, &once_count);
    plat_once(&once, once_fn, &once_count);
    CHECK(once_count == 1, "plat_once runs fn once");

    // scheduling knobs
    CHECK(plat_cpu_count() >= 1, "cpu_count >= 1");
    CHECK(plat_cpu_usable() >= 1, "cpu_usable >= 1");
    uint64_t all = 0, first = 0;
    int cores = plat_cpu_cores(&all, &first);
    CHECK(cores >= 1, "cpu_cores >= 1");
    int prio_before = plat_thread_priority_get();
    plat_thread_priority(PLAT_PRIO_LOWEST);
    int prio_low = plat_thread_priority_get();
    plat_thread_priority(PLAT_PRIO_NORMAL);
    int prio_norm = plat_thread_priority_get();
    CHECK(prio_low <= prio_before, "lowered priority reflected (%d -> %d)", prio_before, prio_low);
    CHECK(prio_norm >= prio_low, "restored normal priority");
    plat_thread_set_priority(NULL, PLAT_PRIO_NORMAL);
    plat_yield();
    CHECK(1, "yield");

    // time is monotonic
    uint64_t a = plat_now_ns();
    plat_sleep_ns(1000000);   // 1 ms
    uint64_t b = plat_now_ns();
    CHECK(b >= a, "monotonic clock");
    CHECK(b - a >= 1000000, "sleep_ns slept at least 1ms (%llu ns)", (unsigned long long)(b - a));
    CHECK(plat_counter_hz() > 0, "counter_hz > 0");
}

// ---- thread interruption (kernel async preemption uses this)

typedef struct { PlatIntr *intr; atomic_int in_loop; atomic_int steps; volatile int stopped; } IntrTarget;

// A busy guest-thread stand-in: spins. Another thread suspends it, reads pc/sp, resumes it.
static void intr_spin(void *arg) {
    IntrTarget *t = arg;
    t->intr = plat_intr_self();
    atomic_store(&t->in_loop, 1);
    volatile uint64_t sink = 0;
    while (!t->stopped) { sink++; atomic_fetch_add(&t->steps, 1); }
    (void)sink;
    if (t->intr) plat_intr_free(t->intr);
}

static void intr_tests(void) {
    IntrTarget t = {0};
    PlatThread *th = plat_thread_new(intr_spin, &t, "intr-target");
    while (!atomic_load(&t.in_loop)) plat_yield();
    while (!t.intr) plat_yield();   // wait until it owns its handle
    int before = atomic_load(&t.steps);
    uint64_t rip = 0, rsp = 0;
    int stopped = plat_intr_stop(t.intr, &rip, &rsp);
    CHECK(stopped, "plat_intr_stop suspended a running thread");
    int during = atomic_load(&t.steps);
    CHECK(rip != 0, "interrupted pc read (%#llx)", (unsigned long long)rip);
    CHECK(rsp != 0, "interrupted sp read (%#llx)", (unsigned long long)rsp);
    int frozen = atomic_load(&t.steps);
    CHECK(atomic_load(&t.steps) == frozen, "no progress while stopped");
    CHECK(during >= before, "no backwards progress");
    int resumed = plat_intr_resume(t.intr, 0);
    CHECK(resumed, "plat_intr_resume(to=0)");
    while (atomic_load(&t.steps) < frozen + 100) plat_yield();
    t.stopped = 1;
    plat_thread_join(th);
    CHECK(1, "thread ran on after resume");
}

// ---- fibers

typedef struct { PlatFiber *back; uint64_t seq; uint32_t fpcr, fpsr; } FiberCtx;

static _Thread_local uint64_t canary[8];

// A fiber body: sets callee-saved state, switches out, verifies it + FP state survived, switches back. Never returns.
__attribute__((noinline)) static void fiber_pingpong(void *arg) {
    FiberCtx *c = arg;
    static const uint64_t saved[8] = {11, 22, 33, 44, 55, 66, 77, 88};
    memcpy(canary, saved, sizeof saved);
    uint64_t fpcr0, fpsr0;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr0));
    __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr0));
    c->seq = 1;
    plat_fiber_switch(c->back);          // out; caller may resume us on another thread
    uint64_t fpcr1, fpsr1;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr1));
    __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr1));
    c->fpcr = (uint32_t)fpcr1; c->fpsr = (uint32_t)fpsr1;
    int ok = memcmp(canary, saved, sizeof saved) == 0 && fpcr0 == fpcr1 && fpsr0 == fpsr1;
    c->seq = ok ? 2 : 0xBAD;
    plat_fiber_switch(c->back);          // second out
    for (;;) plat_fiber_switch(c->back); // never return
}

static volatile uint64_t g_fiber_fpcr_after;

// separate body to store result in a global
__attribute__((noinline)) static void fp_fiber2(void *arg) {
    PlatFiber *back = arg;
    uint64_t orig;
    __asm__ volatile("mrs %0, fpcr" : "=r"(orig));
    __asm__ volatile("msr fpcr, %0" :: "r"(orig | (1ull << 24)));   // FZ in this fiber
    plat_fiber_switch(back);
    uint64_t now;
    __asm__ volatile("mrs %0, fpcr" : "=r"(now));
    g_fiber_fpcr_after = now;
    plat_fiber_switch(back);
    for (;;) plat_fiber_switch(back);
}

// ---- fiber moving between host threads
typedef struct {
    PlatFiber *fiber, *a_base, *b_base;
    PlatEvent a_partway, b_done;
    uint32_t a_tid, b_tid;
    int phase_a, phase_b, resumed_on_b;
    uint64_t acc;
} Migrate;

// Runs phase A on thread A, switches back; later (on thread B) continues with state intact, switches back to B.
__attribute__((noinline)) static void migrating_fiber(void *arg) {
    Migrate *m = arg;
    m->acc = 0xA;
    m->phase_a = 1;
    plat_fiber_switch(m->a_base);        // back to thread A
    m->resumed_on_b = (plat_thread_id() != m->a_tid);
    m->b_tid = plat_thread_id();
    m->acc += 0xB;
    m->phase_b = 1;
    plat_fiber_switch(m->b_base);        // back to thread B; never returns
}

static void thread_a(void *arg) {
    Migrate *m = arg;
    m->a_tid = plat_thread_id();
    m->a_base = plat_fiber_self();
    plat_fiber_switch(m->fiber);         // run migrating_fiber on A until it switches back
    plat_event_set(&m->a_partway);       // fiber switched back to us on thread A
    plat_event_wait(&m->b_done);         // wait for thread B to finish with the fiber
    plat_fiber_unself();
}

static void thread_b(void *arg) {
    Migrate *m = arg;
    m->b_base = plat_fiber_self();
    plat_event_wait(&m->a_partway);
    plat_fiber_switch(m->fiber);         // resume the SAME fiber on thread B
    plat_fiber_unself();                 // fiber switched back to us on thread B
    plat_event_set(&m->b_done);
}

static void fiber_tests(void) {
    // basic round trip on one thread
    PlatFiber *self = plat_fiber_self();
    CHECK(self != NULL, "plat_fiber_self");
    FiberCtx c = {0};
    c.back = self;
    PlatFiber *f = plat_fiber_new(64 * 1024, fiber_pingpong, &c);
    CHECK(f != NULL, "plat_fiber_new");
    plat_fiber_switch(f);
    CHECK(c.seq == 1, "fiber ran partway (seq=%llu)", (unsigned long long)c.seq);
    plat_fiber_switch(f);
    CHECK(c.seq == 2, "callee-saved + FP state preserved across switch (seq=%llu)", (unsigned long long)c.seq);
    plat_fiber_switch(f);   // let it run its infinite tail once; it switches back immediately
    plat_fiber_free(f);
    CHECK(1, "fiber_free");

    // FP control state is per-fiber
    uint64_t host_fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(host_fpcr));
    PlatFiber *fp = plat_fiber_new(32 * 1024, fp_fiber2, self);
    CHECK(fp != NULL, "fp fiber created");
    plat_fiber_switch(fp);   // fiber sets FZ, switches out
    // host's FPCR must be restored for us after the fiber ran
    uint64_t host_after;
    __asm__ volatile("mrs %0, fpcr" : "=r"(host_after));
    CHECK(host_after == host_fpcr, "host FPCR restored after fiber ran (%#llx vs %#llx)",
          (unsigned long long)host_after, (unsigned long long)host_fpcr);
    __asm__ volatile("msr fpcr, %0" :: "r"(host_fpcr));   // clear FZ on host
    plat_fiber_switch(fp);   // re-enter; fiber's own FPCR (FZ) must be restored for it
    CHECK((g_fiber_fpcr_after & (1ull << 24)) != 0, "fiber FPCR (FZ) restored on re-entry (%#llx)",
          (unsigned long long)g_fiber_fpcr_after);
    // host FPCR restored again
    uint64_t host_after2;
    __asm__ volatile("mrs %0, fpcr" : "=r"(host_after2));
    CHECK(host_after2 == host_fpcr, "host FPCR restored again");
    plat_fiber_free(fp);

    // many fibers ping-pong on one thread
    plat_fiber_unself();
    PlatFiber *cur = plat_fiber_self();
    enum { K = 16 };
    FiberCtx cs[K];
    PlatFiber *fs[K];
    for (int i = 0; i < K; i++) { cs[i] = (FiberCtx){0}; cs[i].back = cur; fs[i] = plat_fiber_new(32 * 1024, fiber_pingpong, &cs[i]); }
    for (int round = 0; round < 2; round++)
        for (int i = 0; i < K; i++) plat_fiber_switch(fs[i]);
    int all = 1;
    for (int i = 0; i < K; i++) all &= (cs[i].seq >= 1);
    CHECK(all, "%d fibers ping-pong on one thread", K);
    for (int i = 0; i < K; i++) plat_fiber_free(fs[i]);
    plat_fiber_unself();

    // FIBER MOVING BETWEEN HOST THREADS (the key platform requirement)
    Migrate m = {0};
    plat_event_init(&m.a_partway, 1, 0);
    plat_event_init(&m.b_done, 1, 0);
    m.fiber = plat_fiber_new(256 * 1024, migrating_fiber, &m);
    CHECK(m.fiber != NULL, "migrating fiber created");
    PlatThread *ta = plat_thread_new(thread_a, &m, "thread-a");
    PlatThread *tb = plat_thread_new(thread_b, &m, "thread-b");
    plat_thread_join(ta);
    plat_thread_join(tb);
    CHECK(m.phase_a == 1, "fiber ran partway on thread A");
    CHECK(m.phase_b == 1, "fiber resumed + finished on thread B");
    CHECK(m.resumed_on_b, "fiber moved between host threads");
    CHECK(m.a_tid != m.b_tid, "two distinct host threads (A=%u B=%u)", m.a_tid, m.b_tid);
    CHECK(m.acc == 0xA + 0xB, "fiber state preserved across host-thread move (%#llx)", (unsigned long long)m.acc);
    plat_fiber_free(m.fiber);
}

// ---- virtual memory

static volatile uint64_t hook_write_at, hook_read_at;
static volatile int hook_writes, hook_reads;

// memory.c's thaw: a write to a read-only page makes it writable and retries.
static int write_hook(uint64_t a) {
    hook_write_at = a, hook_writes++;
    uint64_t end;
    if (plat_mem_state(a & ~0xfffull, (a & ~0xfffull) + 0x1000, &end) != PLAT_PROT_READ) return 0;
    return plat_mem_protect(a & ~(plat_mem_grain() - 1), plat_mem_grain(), PLAT_PROT_RW);
}

// memory.c's hold / romfs.c's fill: a read of an inaccessible page commits it with a pattern and retries.
static int read_hook(uint64_t a) {
    hook_read_at = a, hook_reads++;
    uint64_t g = plat_mem_grain(), lo = a & ~(g - 1);
    void *stage = plat_mem_stage(g);
    if (!stage) return 0;
    memset(stage, 0x5a, g);
    return plat_mem_replace(lo, stage, g);
}

// Below 2^39 Apple Silicon leaves only [448 GiB, 512 GiB) to a process (the GPU carve-out takes 64-448 GiB), and
// libmalloc reserves about 24 GiB of it at a random place: search it in 1 GiB steps, as memory.c does.
static uint64_t reserve_somewhere(uint64_t n, int watch) {
    for (uint64_t at = 0x7000000000ull; at + n <= 0x8000000000ull; at += 0x40000000ull)
        if (plat_mem_reserve(at, n, watch)) return at;
    return 0;
}

static int contains(const uint64_t *p, uint64_t n, uint64_t v) {
    for (uint64_t i = 0; i < n; i++)
        if (p[i] == v) return 1;
    return 0;
}

// The write-watch stress: writers write pages of a watched range while the main thread queries with reset. A write
// that completed after one query ended and before the next one began must be in the next one's list.
enum { WW_PAGES = 256, WW_WRITERS = 4 };
typedef struct { uint64_t base; volatile int stop; volatile uint64_t done[WW_PAGES]; uint32_t seed; } WwStress;

static void ww_writer(void *arg) {
    WwStress *s = arg;
    uint32_t x = __atomic_add_fetch(&s->seed, 0x9e3779b9u, __ATOMIC_RELAXED);
    while (!s->stop) {
        x = x * 1664525u + 1013904223u;
        uint64_t p = (x >> 8) % WW_PAGES;
        volatile uint32_t *w = (volatile uint32_t *)(uintptr_t)(s->base + p * 0x1000 + ((x >> 4) & 0xff) * 4);
        *w = x;
        __atomic_add_fetch(&s->done[p], 1, __ATOMIC_RELEASE);
    }
}

static void mem_tests(void) {
    uint64_t g = plat_mem_grain();
    CHECK(g == (uint64_t)sysconf(_SC_PAGESIZE), "grain is the host page size (%llu)", (unsigned long long)g);
    plat_mem_fault_hook(write_hook);
    plat_mem_read_fault_hook(read_hook);

    // reservations: exact, and a second one over it fails
    uint64_t base = reserve_somewhere(64 * g, 0), end;
    CHECK(base, "reserve 64 host pages");
    if (!base) return;
    CHECK(!plat_mem_reserve(base + 4 * g, g, 0), "a reservation over a reserved range fails");
    CHECK(plat_mem_state(base, base + 64 * g, &end) == PLAT_MEM_RESERVED && end == base + 64 * g, "reserved range");
    CHECK(plat_mem_reserve(base + 64 * g, 0x1000, 0), "a 4 KiB reservation sharing the next host page");
    CHECK(!plat_mem_reserve(base + 64 * g, 0x1000, 0), "and not twice");
    CHECK(plat_mem_reserve(base + 64 * g + 0x1000, 0x1000, 0), "its neighbour in the same host page");
    CHECK(!plat_mem_commit(base + 80 * g, 0x1000), "commit outside our reservations fails");

    // 4 KiB states inside one host page
    CHECK(plat_mem_commit(base + 0x1000, 0x1000), "commit one 4 KiB page");
    CHECK(plat_mem_state(base, base + g, &end) == PLAT_MEM_RESERVED && end == base + 0x1000, "page 0 reserved");
    CHECK(plat_mem_state(base + 0x1000, base + g, &end) == PLAT_PROT_RW && end == base + 0x2000, "page 1 committed");
    CHECK(plat_mem_state(base + 0x2000, base + g, &end) == PLAT_MEM_RESERVED && end == base + g, "pages 2-3 reserved");
    memset((void *)(uintptr_t)(base + 0x1000), 0xab, 0x1000);
    CHECK(*(volatile uint8_t *)(uintptr_t)(base + 0x1fff) == 0xab, "committed page holds data");

    // a host page all read-only: a write faults into the write hook at its address, which thaws and retries
    CHECK(plat_mem_commit(base + 4 * g, 4 * g), "commit 4 host pages");
    memset((void *)(uintptr_t)(base + 4 * g), 7, 4 * g);
    CHECK(plat_mem_protect(base + 5 * g, g, PLAT_PROT_READ), "freeze host page 5");
    CHECK(plat_mem_state(base + 5 * g, base + 8 * g, &end) == PLAT_PROT_READ && end == base + 6 * g, "read-only");
    hook_writes = 0;
    CHECK(*(volatile uint8_t *)(uintptr_t)(base + 5 * g + 8) == 7, "a read of a read-only page does not fault");
    CHECK(hook_writes == 0, "no write fault on a read");
    *(volatile uint8_t *)(uintptr_t)(base + 5 * g + 0x2345) = 9;
    CHECK(hook_writes == 1 && hook_write_at == base + 5 * g + 0x2345, "write fault at the written byte (%d at %#llx)",
          hook_writes, (unsigned long long)hook_write_at);
    CHECK(*(volatile uint8_t *)(uintptr_t)(base + 5 * g + 0x2345) == 9, "the write landed after the thaw");

    // a 4 KiB read-only page beside read-write ones: its host page stays writable (most permissive), no fault
    CHECK(plat_mem_protect(base + 6 * g, 0x1000, PLAT_PROT_READ), "read-only 4 KiB page in a read-write host page");
    hook_writes = 0;
    *(volatile uint8_t *)(uintptr_t)(base + 6 * g + 0x1800) = 1;
    CHECK(hook_writes == 0, "a mixed host page takes the most permissive state");
    CHECK(plat_mem_state(base + 6 * g, base + 7 * g, &end) == PLAT_PROT_READ && end == base + 6 * g + 0x1000,
          "but plat_mem_state keeps the 4 KiB state");

    // an inaccessible host page: a read faults into the read hook, which replaces the pages and retries
    CHECK(plat_mem_protect(base + 7 * g, g, PLAT_PROT_NONE), "hold host page 7");
    hook_reads = 0;
    uint8_t v = *(volatile uint8_t *)(uintptr_t)(base + 7 * g + 0x3001);
    CHECK(hook_reads == 1 && hook_read_at == base + 7 * g + 0x3001, "read fault at the read byte");
    CHECK(v == 0x5a, "the read sees the replaced bytes (%#x)", v);
    CHECK(plat_mem_state(base + 7 * g, base + 8 * g, &end) == PLAT_PROT_RW && end == base + 8 * g, "replaced: RW");
    uint8_t probe;
    CHECK(plat_mem_protect(base + 7 * g, g, PLAT_PROT_NONE) && !plat_mem_read(base + 7 * g, &probe, 1),
          "plat_mem_read fails instead of faulting");
    CHECK(plat_mem_read(base + 4 * g, &probe, 1) && probe == 7, "plat_mem_read reads accessible memory");
    CHECK(plat_mem_readable(base + 4 * g, g) && !plat_mem_readable(base + 6 * g, 2 * g), "plat_mem_readable");
    CHECK(plat_mem_protect(base + 7 * g, g, PLAT_PROT_RW), "release host page 7");

    // decommit: whole host pages and single 4 KiB pages come back zero; neighbours keep their bytes
    memset((void *)(uintptr_t)(base + 4 * g), 0xee, 2 * g);
    CHECK(plat_mem_decommit(base + 4 * g + 0x1000, 0x1000), "decommit one 4 KiB page");
    CHECK(plat_mem_state(base + 4 * g + 0x1000, base + 5 * g, &end) == PLAT_MEM_RESERVED, "decommitted: reserved");
    CHECK(*(volatile uint8_t *)(uintptr_t)(base + 4 * g) == 0xee && *(volatile uint8_t *)(uintptr_t)(base + 4 * g + 0x2000) == 0xee,
          "its neighbours keep their bytes");
    CHECK(plat_mem_commit(base + 4 * g + 0x1000, 0x1000) && *(volatile uint32_t *)(uintptr_t)(base + 4 * g + 0x1ffc) == 0,
          "recommitted 4 KiB page reads zero");
    CHECK(plat_mem_decommit(base + 4 * g, 2 * g) && plat_mem_commit(base + 4 * g, 2 * g), "decommit + commit 2 host pages");
    int zero = 1;
    for (uint64_t a = base + 4 * g; a < base + 6 * g; a += 8) zero &= *(volatile uint64_t *)(uintptr_t)a == 0;
    CHECK(zero, "recommitted host pages read zero");

    // write watch: per 4 KiB page; a write marks its host page, reset arms it again
    uint64_t wb = reserve_somewhere(128 * g, 1), pages[64];
    CHECK(wb, "reserve a watched range");
    if (!wb) return;
    CHECK(plat_mem_commit(wb, 16 * g), "commit 16 watched host pages");
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == 0, "nothing written after the commit");
    CHECK(plat_mem_written(base, g, pages, 64, 1) == UINT64_MAX, "an unwatched range is untracked");
    *(volatile uint8_t *)(uintptr_t)(wb + 3 * g + 0x1234) = 1;
    uint64_t n = plat_mem_written(wb, 16 * g, pages, 64, 0);
    CHECK(n == g / 0x1000 && pages[0] == wb + 3 * g && contains(pages, n, wb + 3 * g + 0x1000), "peek: the written host "
          "page's 4 KiB pages (%llu)", (unsigned long long)n);
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == g / 0x1000, "query with reset lists them");
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == 0, "and the next one does not");
    *(volatile uint8_t *)(uintptr_t)(wb + 3 * g + 8) = 2;
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == g / 0x1000, "a write after the reset is seen again");
    CHECK(plat_mem_written(wb, 16 * g, pages, 2, 1) == 0, "nothing pending");
    *(volatile uint8_t *)(uintptr_t)(wb + 5 * g) = 3;
    CHECK(plat_mem_written(wb, 16 * g, pages, 2, 1) == 2, "a full list stops at max");
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == g / 0x1000, "and keeps the host page for the next query");
    plat_mem_will_write(wb + 8 * g, 2 * g);
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == 2 * g / 0x1000, "plat_mem_will_write marks without a write");
    *(volatile uint8_t *)(uintptr_t)(wb + 8 * g) = 4;
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == g / 0x1000, "re-armed after a will_write query");
    CHECK(plat_mem_written(wb + 12 * g, 4 * g, pages, 64, 1) == 0, "an unwritten part");
    // a frozen watched host page: the write goes through the write hook, then the watch, and is reported
    CHECK(plat_mem_protect(wb + 9 * g, g, PLAT_PROT_READ), "freeze a watched host page");
    hook_writes = 0;
    *(volatile uint8_t *)(uintptr_t)(wb + 9 * g + 16) = 5;
    CHECK(hook_writes == 1 && plat_mem_written(wb, 16 * g, pages, 64, 1) == g / 0x1000, "frozen write thawed + seen");
    // a replaced range counts as written
    void *stage = plat_mem_stage(2 * g);
    memset(stage, 0x11, 2 * g);
    CHECK(plat_mem_protect(wb + 10 * g, 2 * g, PLAT_PROT_NONE) && plat_mem_replace(wb + 10 * g, stage, 2 * g),
          "replace 2 watched host pages");
    CHECK(*(volatile uint8_t *)(uintptr_t)(wb + 11 * g + 5) == 0x11, "replaced bytes");
    CHECK(plat_mem_written(wb, 16 * g, pages, 64, 1) == 2 * g / 0x1000, "replaced pages count as written");

    // concurrent writes against queries with reset: none missed
    static WwStress s;
    s.base = wb + 16 * g;
    CHECK(plat_mem_commit(s.base, WW_PAGES * 0x1000), "commit the stress range");
    PlatThread *w[WW_WRITERS];
    for (int i = 0; i < WW_WRITERS; i++) w[i] = plat_thread_new(ww_writer, &s, "ww writer");
    static uint64_t after_prev[WW_PAGES], before[WW_PAGES], got[WW_PAGES];
    int missed = 0, queries = 0;
    uint64_t reported = 0;
    for (uint64_t t0 = plat_now_ns(); plat_now_ns() - t0 < 300000000ull; queries++) {
        for (int p = 0; p < WW_PAGES; p++) before[p] = __atomic_load_n(&s.done[p], __ATOMIC_ACQUIRE);
        uint64_t k = plat_mem_written(s.base, WW_PAGES * 0x1000, got, WW_PAGES, 1);
        reported += k;
        for (int p = 0; p < WW_PAGES; p++)
            if (before[p] > after_prev[p] && !contains(got, k, s.base + (uint64_t)p * 0x1000)) missed++;
        for (int p = 0; p < WW_PAGES; p++) after_prev[p] = __atomic_load_n(&s.done[p], __ATOMIC_ACQUIRE);
    }
    s.stop = 1;
    for (int i = 0; i < WW_WRITERS; i++) plat_thread_join(w[i]);
    CHECK(!missed && reported, "no write missed in %d queries against %d writers (%d missed, %llu pages reported)",
          queries, WW_WRITERS, missed, (unsigned long long)reported);
    uint64_t writes = 0;
    for (int p = 0; p < WW_PAGES; p++) writes += s.done[p];
    printf("write watch stress: %llu writes, %d queries, %llu pages reported, %d missed\n", (unsigned long long)writes,
           queries, (unsigned long long)reported, missed);

    // plat_mem_alloc memory seals per host page, as dispatch.c's tables
    uint8_t *t = plat_mem_alloc(3 * g);
    CHECK(t && t[0] == 0 && t[3 * g - 1] == 0, "plat_mem_alloc gives zero memory");
    t[0] = 1;
    CHECK(plat_mem_protect((uint64_t)(uintptr_t)t, g, PLAT_PROT_READ), "seal its first host page");
    hook_writes = 0;
    t[g] = 2;
    CHECK(hook_writes == 0 && t[g] == 2, "the next host page stays writable");
}

int main(void) {
    printf("host page size: %ld\n", sysconf(_SC_PAGESIZE));
    sched_tests();
    intr_tests();
    fiber_tests();
    mem_tests();
    printf("macos platform test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
