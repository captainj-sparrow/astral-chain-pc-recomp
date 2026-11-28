// macOS (Apple Silicon) implementation of platform.h, scheduling half: Mach time, pthreads with QoS classes, locks
// and condition variables on os_sync_wait_on_address (macOS 14.4, Darwin's futex), thread interruption through Mach
// thread suspension, and fibers. Virtual memory and the fault handler are in macos_mem.c, files and processes in
// macos_posix.c.
#include <errno.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <os/clock.h>
#include <os/os_sync_wait_on_address.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include "platform.h"

// ---- time: mach_absolute_time (24 MHz on Apple Silicon), scaled to ns without a division per call.

static pthread_once_t time_once = PTHREAD_ONCE_INIT;
static mach_timebase_info_data_t timebase;
static PlatScale ns_scale;

static void time_init(void) {
    mach_timebase_info(&timebase);
    ns_scale = plat_scale(timebase.numer, timebase.denom);
}

uint64_t plat_counter(void) { return mach_absolute_time(); }

uint64_t plat_counter_hz(void) {
    pthread_once(&time_once, time_init);
    return 1000000000ull * timebase.denom / timebase.numer;
}

uint64_t plat_now_ns(void) {
    pthread_once(&time_once, time_init);
    return plat_scaled(ns_scale, mach_absolute_time());
}

void plat_timer_1ms(void) { pthread_once(&time_once, time_init); }   // Darwin's timers are precise already

int plat_power_full(void) { return -1; }

// mach_wait_until sleeps on the high-resolution timer to an absolute deadline in mach ticks.
void plat_wait_ns(uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    for (uint64_t now; (now = plat_now_ns()) < end;)
        mach_wait_until(mach_absolute_time() + ((__uint128_t)(end - now) * timebase.denom + timebase.numer - 1) /
                                                   timebase.numer);
}

void plat_sleep_ns(uint64_t ns) { plat_wait_ns(ns); }

void plat_sleep_ms(uint32_t ms) { plat_wait_ns((uint64_t)ms * 1000000u); }

void plat_yield(void) { sched_yield(); }

// ---- processors. Apple Silicon has no SMT: every logical processor is a core. Darwin offers no CPU binding.

uint32_t plat_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1;
}

uint32_t plat_cpu_usable(void) { return plat_cpu_count(); }

int plat_cpu_cores(uint64_t *all, uint64_t *first) {
    uint32_t n = plat_cpu_count();
    *all = *first = n >= 64 ? ~0ull : (1ull << n) - 1;
    return (int)n;
}

int plat_process_affinity(uint64_t mask) { (void)mask; return 0; }

uint64_t plat_thread_affinity(PlatThread *t, uint64_t mask) { (void)t, (void)mask; return 0; }

uint32_t plat_cpu_mhz(void) { return 0; }   // Darwin does not report the current frequency without privileges

uint32_t plat_thread_id(void) { return pthread_mach_thread_np(pthread_self()); }

uint64_t plat_thread_cpu_ns(uint32_t tid) {
    thread_basic_info_data_t info;
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    if (thread_info((thread_act_t)tid, THREAD_BASIC_INFO, (thread_info_t)&info, &count) != KERN_SUCCESS) return 0;
    return ((uint64_t)info.user_time.seconds + (uint64_t)info.system_time.seconds) * 1000000000u +
           ((uint64_t)info.user_time.microseconds + (uint64_t)info.system_time.microseconds) * 1000u;
}

// ---- threads. Each starts at the QoS class of PLAT_PRIO_NORMAL with a signal stack of its own (macos_mem.c's fault
// handler runs there, also when the thread's stack is exhausted). The handle is shared by the thread and its owner
// until both let go of it.

struct PlatThread {
    pthread_t thread;
    PlatThreadFn fn;
    void *arg;
    pthread_override_t ovr;   // a raise by plat_thread_set_priority, until the thread sets its own priority
    int32_t refs;
    char name[64];
};

static __thread PlatThread *thread_me;

void plat_alt_stack(void);   // macos_mem.c

static void thread_release(PlatThread *t) {
    if (__atomic_sub_fetch(&t->refs, 1, __ATOMIC_ACQ_REL) == 0) free(t);
}

static void *thread_main(void *p) {
    PlatThread *t = p;
    thread_me = t;
    plat_alt_stack();
    if (t->name[0]) pthread_setname_np(t->name);
    t->fn(t->arg);
    pthread_override_t o = __atomic_exchange_n(&t->ovr, NULL, __ATOMIC_ACQ_REL);
    if (o) pthread_override_qos_class_end_np(o);
    thread_me = NULL;
    thread_release(t);
    return NULL;
}

static qos_class_t qos_of(int prio) {
    return prio <= PLAT_PRIO_IDLE ? QOS_CLASS_BACKGROUND : prio < PLAT_PRIO_NORMAL ? QOS_CLASS_UTILITY
         : prio == PLAT_PRIO_NORMAL ? QOS_CLASS_USER_INITIATED : QOS_CLASS_USER_INTERACTIVE;
}

PlatThread *plat_thread_new_ex(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->fn = fn, t->arg = arg, t->refs = 2;
    if (name) snprintf(t->name, sizeof t->name, "%s", name);
    size_t page = (size_t)getpagesize();
    stack = stack ? stack : PLAT_STACK_DEFAULT;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_set_qos_class_np(&a, qos_of(PLAT_PRIO_NORMAL), 0);
    int e = stack > SIZE_MAX - page ? EINVAL : pthread_attr_setstacksize(&a, (stack + page - 1) & ~(page - 1));
    if (!e) e = pthread_create(&t->thread, &a, thread_main, t);
    pthread_attr_destroy(&a);
    if (e) { free(t); return NULL; }
    return t;
}

PlatThread *plat_thread_new(PlatThreadFn fn, void *arg, const char *name) { return plat_thread_new_ex(fn, arg, name, 0); }

void plat_thread_join(PlatThread *t) {
    pthread_join(t->thread, NULL);
    thread_release(t);
}

// pthread_join has no timeout: a thread that is not done by the deadline keeps running and its handle stays valid.
// pthread_kill with signal 0 tells whether it still runs; a thread that has returned is joined at once.
int plat_thread_join_ns(PlatThread *t, uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    for (;;) {
        if (__atomic_load_n(&t->refs, __ATOMIC_ACQUIRE) == 1) {   // thread_main let go: fn returned
            plat_thread_join(t);
            return 1;
        }
        uint64_t now = plat_now_ns();
        if (now >= end) return 0;
        plat_wait_ns(end - now < 200000u ? end - now : 200000u);
    }
}

void plat_thread_detach(PlatThread *t) {
    pthread_detach(t->thread);
    thread_release(t);
}

int plat_thread_spawn(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = plat_thread_new_ex(fn, arg, name, stack);
    if (!t) return 0;
    plat_thread_detach(t);
    return 1;
}

// ---- priorities as QoS classes, which also decide where Apple Silicon runs a thread: background work goes to the
// efficiency cores. A thread sets its own class; another thread can only raise it, with a QoS override that lasts
// until the thread next sets its own priority (vk_pipeline.c raises a pipeline worker a waiter depends on). Threads
// below normal priority never poll in their waits, as on Linux.

static __thread int8_t spin_off;

void plat_thread_priority(int prio) {
    spin_off = prio < PLAT_PRIO_NORMAL;
    pthread_set_qos_class_self_np(qos_of(prio), 0);
    PlatThread *t = thread_me;
    pthread_override_t o = t ? __atomic_exchange_n(&t->ovr, NULL, __ATOMIC_ACQ_REL) : NULL;
    if (o) pthread_override_qos_class_end_np(o);
}

void plat_thread_set_priority(PlatThread *t, int prio) {
    if (!t) {
        plat_thread_priority(prio);
        return;
    }
    pthread_override_t o = pthread_override_qos_class_start_np(t->thread, qos_of(prio), 0);
    o = __atomic_exchange_n(&t->ovr, o, __ATOMIC_ACQ_REL);
    if (o) pthread_override_qos_class_end_np(o);
}

int plat_thread_priority_get(void) {
    qos_class_t q = qos_class_self();
    return q == QOS_CLASS_BACKGROUND ? PLAT_PRIO_IDLE : q == QOS_CLASS_UTILITY ? PLAT_PRIO_BELOW_NORMAL
         : q == QOS_CLASS_USER_INTERACTIVE ? PLAT_PRIO_HIGHEST : PLAT_PRIO_NORMAL;
}

// ---- CPU performance. Darwin's scheduler chooses cores and frequencies from the QoS classes; nothing here needs or
// can take a privilege. What remains of linux.c's measures is the wait polling (cond_spin), which keeps a core awake
// across the short gaps between bursts of work instead of a trip through the scheduler.

static uint64_t spin_cap_ns;   // the longest poll of a wait; 0: every wait sleeps at once

void plat_cpu_boost(int mode, uint64_t spin_ns, char *out, size_t cap) {
    __atomic_store_n(&spin_cap_ns, mode ? spin_ns : 0, __ATOMIC_RELAXED);
    if (!mode) snprintf(out, cap, "performance hints off");
    else snprintf(out, cap, "Darwin QoS classes for the game's threads; waits %s", spin_ns ? "poll before they sleep"
                                                                                           : "sleep at once");
}

int plat_cpu_boost_note(char *out, size_t cap) {
    (void)out, (void)cap;
    return 0;
}

// ---- locks: linux.c's reader-writer lock in one 32-bit word, zero when free, with os_sync_wait_on_address and
// os_sync_wake_by_address_all in place of the futex. bit 31: a writer holds it; bit 30: some thread sleeps on the word;
// bits 0-29: readers holding it. Readers do not enter while the sleeper bit is set, so a waiting writer is not starved.

#define LK_W 0x80000000u
#define LK_S 0x40000000u
#define LK_R 0x3fffffffu
#define LK_SPINS 64

static inline volatile uint32_t *lk(PlatMutex *m) { return (volatile uint32_t *)&m->p; }

static void word_wait(volatile uint32_t *w, uint32_t v) {
    os_sync_wait_on_address((void *)w, v, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
}

static void word_wake(volatile uint32_t *w, int all) {
    if (all) os_sync_wake_by_address_all((void *)w, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
    else os_sync_wake_by_address_any((void *)w, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

static void lk_sleep(volatile uint32_t *w, uint32_t s) {
    if (!(s & LK_S) && !__atomic_compare_exchange_n(w, &s, s | LK_S, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    word_wait(w, s | LK_S);
}

static void lk_wake(volatile uint32_t *w) {
    __atomic_fetch_and(w, ~LK_S, __ATOMIC_RELAXED);
    word_wake(w, 1);
}

void plat_mutex_init(PlatMutex *m) { m->p = 0; }

int plat_mutex_trylock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_R)))
        if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

void plat_mutex_lock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_R))) {
            if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) plat_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_fetch_and(w, ~LK_W, __ATOMIC_RELEASE);
    if (s & LK_S) lk_wake(w);
}

int plat_mutex_trylock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_S)))
        if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

void plat_mutex_lock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_S))) {
            if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) plat_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_sub_fetch(w, 1, __ATOMIC_RELEASE);
    if (!(s & LK_R) && (s & LK_S)) lk_wake(w);
}

// ---- condition variables: a sequence number in the first 32 bits, as on Linux. A waiter reads it before releasing
// the lock and sleeps only while it is unchanged, so a signal between the release and the sleep is not lost. Before it
// sleeps a waiter polls the word for SPIN_PER_WORK times the CPU time it used since its previous wait, up to the cap
// plat_cpu_boost set (linux.c cond_spin has the measurements).

#define SPIN_PER_WORK 16
#define SPIN_MIN_NS 2000u
#define SPIN_YIELD 32
static __thread uint64_t spin_cpu, spin_spent;

static inline volatile uint32_t *cv(PlatCond *c) { return (volatile uint32_t *)&c->p; }

void plat_cond_init(PlatCond *c) { c->p = 0; }

static uint64_t cond_spin(volatile uint32_t *w, uint32_t seq, uint64_t limit_ns, int *changed) {
    *changed = 0;
    uint64_t cap = __atomic_load_n(&spin_cap_ns, __ATOMIC_RELAXED);
    if (!cap || spin_off) return 0;
    uint64_t cpu = clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID), used = cpu - spin_cpu;
    uint64_t work = used > spin_spent ? used - spin_spent : 0;
    spin_cpu = cpu;
    spin_spent = 0;
    uint64_t budget = work < cap / SPIN_PER_WORK ? work * SPIN_PER_WORK : cap;
    if (budget > limit_ns) budget = limit_ns;
    if (budget < SPIN_MIN_NS) return 0;
    uint64_t t0 = plat_now_ns(), now = t0;
    for (unsigned round = 1; now - t0 < budget && !*changed; round++) {
        for (int i = 0; i < 16 && !*changed; i++) {
            if (__atomic_load_n(w, __ATOMIC_ACQUIRE) != seq) *changed = 1;
            else plat_pause();
        }
        if (!*changed && round % SPIN_YIELD == 0) sched_yield();
        now = plat_now_ns();
    }
    spin_spent = now - t0;
    return spin_spent;
}

void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    int changed;
    plat_mutex_unlock(m);
    cond_spin(cv(c), seq, UINT64_MAX, &changed);
    if (!changed) word_wait(cv(c), seq);
    plat_mutex_lock(m);
}

int plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    int changed;
    plat_mutex_unlock(m);
    uint64_t spun = cond_spin(cv(c), seq, ns, &changed);
    if (changed || spun >= ns) {
        plat_mutex_lock(m);
        return changed;
    }
    ns -= spun;
    pthread_once(&time_once, time_init);
    // A timeout of 0 is an error to the call, and so is an early return with nobody waking us; only ETIMEDOUT is one.
    int r = ns ? os_sync_wait_on_address_with_timeout((void *)cv(c), seq, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                      OS_CLOCK_MACH_ABSOLUTE_TIME,
                                                      (uint64_t)((__uint128_t)ns * timebase.denom / timebase.numer) + 1)
               : -1;
    int timed_out = !ns || (r == -1 && errno == ETIMEDOUT);
    plat_mutex_lock(m);
    return !timed_out;
}

void plat_cond_signal(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    word_wake(cv(c), 0);
}

void plat_cond_broadcast(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    word_wake(cv(c), 1);
}

// ---- thread interruption: Mach thread suspension in place of Windows' SuspendThread / Get/SetThreadContext. No
// handler runs on the stopped thread, so the game's fiber stack is never written; a thread blocked in the kernel stops
// too (its pc is then in libsystem_kernel, which kernel.c does not redirect).
struct PlatIntr {
    mach_port_t thread;
    arm_thread_state64_t saved;
    int stopped;
};

PlatIntr *plat_intr_self(void) {
    PlatIntr *t = calloc(1, sizeof *t);
    if (t) t->thread = mach_thread_self();   // a send right of our own, released in plat_intr_free
    return t;
}

void plat_intr_free(PlatIntr *t) {
    if (!t) return;
    if (t->stopped) thread_resume(t->thread);
    mach_port_deallocate(mach_task_self(), t->thread);
    free(t);
}

int plat_intr_stop(PlatIntr *t, uint64_t *rip, uint64_t *rsp) {
    if (!t || t->stopped || t->thread == pthread_mach_thread_np(pthread_self())) return 0;
    if (thread_suspend(t->thread) != KERN_SUCCESS) return 0;   // returns once the thread no longer runs
    mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
    if (thread_get_state(t->thread, ARM_THREAD_STATE64, (thread_state_t)&t->saved, &n) != KERN_SUCCESS) {
        thread_resume(t->thread);
        return 0;
    }
    t->stopped = 1;
    *rip = (uint64_t)__darwin_arm_thread_state64_get_pc(t->saved);
    *rsp = (uint64_t)__darwin_arm_thread_state64_get_sp(t->saved);
    return 1;
}

int plat_intr_resume(PlatIntr *t, uint64_t to) {
    if (!t || !t->stopped) return 0;
    int ok = 1;
    if (to) {
        __darwin_arm_thread_state64_set_pc_fptr(t->saved, (void *)(uintptr_t)to);
        ok = thread_set_state(t->thread, ARM_THREAD_STATE64, (thread_state_t)&t->saved, ARM_THREAD_STATE64_COUNT) ==
             KERN_SUCCESS;
    }
    if (thread_resume(t->thread) != KERN_SUCCESS) return 0;
    t->stopped = 0;
    return ok;
}

// ---- fibers: a stack switch of our own. fiber_swap saves what AAPCS64 makes callee-saved (x19-x30, the low halves of
// v8-v15) and the FP control and status registers in a 176-byte frame on the current stack, stores the stack pointer
// in *save and loads the other one. x18 is Darwin's and never touched. A new fiber's stack is laid out as fiber_swap
// leaves one, returning into fiber_entry, which calls fn(arg) (x19 and x20).
struct PlatFiber {
    void *sp;
    void *stack;   // mapped stack (NULL for a thread's own context)
    size_t size;
};

void plat_fiber_swap(void **save, void *load);
void plat_fiber_entry(void);
__asm__(".text\n"
        ".globl _plat_fiber_swap\n"
        ".p2align 2\n"
        "_plat_fiber_swap:\n"
        "    sub sp, sp, #176\n"
        "    stp x19, x20, [sp, #0]\n"
        "    stp x21, x22, [sp, #16]\n"
        "    stp x23, x24, [sp, #32]\n"
        "    stp x25, x26, [sp, #48]\n"
        "    stp x27, x28, [sp, #64]\n"
        "    stp x29, x30, [sp, #80]\n"
        "    stp d8, d9, [sp, #96]\n"
        "    stp d10, d11, [sp, #112]\n"
        "    stp d12, d13, [sp, #128]\n"
        "    stp d14, d15, [sp, #144]\n"
        "    mrs x9, fpcr\n"
        "    mrs x10, fpsr\n"
        "    stp x9, x10, [sp, #160]\n"
        "    mov x9, sp\n"
        "    str x9, [x0]\n"
        "    mov sp, x1\n"
        "    ldp x19, x20, [sp, #0]\n"
        "    ldp x21, x22, [sp, #16]\n"
        "    ldp x23, x24, [sp, #32]\n"
        "    ldp x25, x26, [sp, #48]\n"
        "    ldp x27, x28, [sp, #64]\n"
        "    ldp x29, x30, [sp, #80]\n"
        "    ldp d8, d9, [sp, #96]\n"
        "    ldp d10, d11, [sp, #112]\n"
        "    ldp d12, d13, [sp, #128]\n"
        "    ldp d14, d15, [sp, #144]\n"
        "    ldp x9, x10, [sp, #160]\n"
        "    msr fpcr, x9\n"
        "    msr fpsr, x10\n"
        "    add sp, sp, #176\n"
        "    ret\n"
        ".globl _plat_fiber_entry\n"
        ".p2align 2\n"
        "_plat_fiber_entry:\n"
        "    mov x0, x20\n"
        "    blr x19\n"
        "    brk #0\n");

static __thread PlatFiber *fiber_cur;
static __thread PlatFiber fiber_own;

PlatFiber *plat_fiber_self(void) {
    fiber_own = (PlatFiber){0};
    fiber_cur = &fiber_own;
    return &fiber_own;
}

void plat_fiber_unself(void) { fiber_cur = NULL; }

PlatFiber *plat_fiber_new(size_t stack, void (*fn)(void *arg), void *arg) {
    size_t page = (size_t)getpagesize();
    stack = stack ? stack : PLAT_STACK_DEFAULT;
    if (stack > SIZE_MAX - 2 * page) return NULL;
    size_t n = ((stack + page - 1) & ~(page - 1)) + page;   // and a guard page below it
    PlatFiber *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    void *m = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);   // backed as it is touched
    if (m == MAP_FAILED) {
        free(f);
        return NULL;
    }
    if (mprotect(m, page, PROT_NONE)) {
        munmap(m, n);
        free(f);
        return NULL;
    }
    f->stack = m, f->size = n;
    // The frame fiber_swap loads: x19 = fn, x20 = arg, x30 = plat_fiber_entry, FPCR and FPSR zero (round to nearest,
    // no flush to zero, no flags). After its add, sp is the top of the stack, 16-byte aligned.
    uint64_t *sp = (uint64_t *)((char *)m + n - 176);
    memset(sp, 0, 176);
    sp[0] = (uint64_t)(uintptr_t)fn;
    sp[1] = (uint64_t)(uintptr_t)arg;
    sp[11] = (uint64_t)(uintptr_t)plat_fiber_entry;
    f->sp = sp;
    return f;
}

void plat_fiber_switch(PlatFiber *to) {
    PlatFiber *from = fiber_cur;
    if (!from || !to) abort();
    fiber_cur = to;   // on this thread; the switch back into from sets it on whichever thread that is
    plat_fiber_swap(&from->sp, to->sp);
}

void plat_fiber_free(PlatFiber *f) {
    if (!f) return;
    if (f->stack) munmap(f->stack, f->size);
    free(f);
}
