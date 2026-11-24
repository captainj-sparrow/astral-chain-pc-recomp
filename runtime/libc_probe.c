// --probe-libc: the host versions of the sdk's memcpy, memset, memmove, memcmp, strlen, strcmp and strncmp (ov_libc.c)
// against the sdk's own lifted functions, on the same inputs. Needs the game's ExeFS (the sdk module).
//
// Each case runs the lifted function (rt_call on a guest thread) and the host function (called directly on a copy of
// the Cpu) on two copies of the same buffers, then compares the whole touched range plus a 64-byte guard zone on each
// side, x0 (all 64 bits) and, for the host function, every other field of the Cpu: the host version must write x0 and
// nothing else (no other X or V register, no flags), so that a direct call with a narrower register contract stays
// possible. Inputs: sizes 0-256 densely, then powers of two and their neighbours up to 8 MiB; misalignments 0-63 on
// each side; memmove with every overlap (destination inside the source, source inside the destination, equal,
// adjacent) in both directions; memcmp, strcmp and strncmp with the first difference at every position and bytes above
// 0x7f on either side; strings that end on the last byte of a page whose next page is inaccessible (an overread would
// fault and stop the probe); memset values with the upper bits of w1 set.
//
// It then times both per size class (ns per call, the median of 9 rounds), as the game reaches them: the lifted
// function called directly (an import stub whose slot holds it) against the host function reached through rt_dispatch
// at its synthetic address (the stub's slot holds that address when host_libc is on), and the host function called
// directly. The timing table is what the size handling in ov_libc.c rests on.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "platform/platform.h"

static int failures, checks;
#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        checks++;                                                            \
        if (!(cond) && ++failures <= 20) {                                   \
            printf("  FAIL: " __VA_ARGS__);                                  \
            putchar('\n');                                                   \
        }                                                                    \
    } while (0)

enum { F_MEMCPY, F_MEMSET, F_MEMMOVE, F_MEMCMP, F_STRLEN, F_STRCMP, F_STRNCMP, F_N };
static const char *const names[F_N] = {"memcpy", "memset", "memmove", "memcmp", "strlen", "strcmp", "strncmp"};
static uint64_t guest[F_N];    // the sdk's definition
static GuestFn lifted[F_N];    // its lifted function
static GuestFn hostf[F_N];     // ov_libc.c
static uint64_t hostaddr[F_N]; // the host function's synthetic address (the GOT slot's value)

#define BUF (16ull << 20)       // each of the two working copies: room for 8 MiB at either end plus misalignment
#define GUARD 64
static uint8_t *ga, *ha, *pristine;   // guest copy, host copy, the content both start from
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return rng;
}

static void fill_random(uint8_t *p, uint64_t n) {
    for (uint64_t i = 0; i + 8 <= n; i += 8) {
        uint64_t v = rnd();
        memcpy(p + i, &v, 8);
    }
}

// Restores [lo, hi) of both copies from the pristine content.
static void restore(uint64_t lo, uint64_t hi) {
    memcpy(ga + lo, pristine + lo, hi - lo);
    memcpy(ha + lo, pristine + lo, hi - lo);
}

static Cpu *gc;
static uint64_t mismatches[F_N], cases[F_N];

// The Cpu the host function starts from: random registers, so a write to any of them shows.
static void random_cpu(Cpu *h) {
    *h = *gc;
    for (int i = 0; i < 31; i++) h->x[i] = rnd();
    for (int i = 0; i < 32; i++) h->v[i] = ((__uint128_t)rnd() << 64) | rnd();
    h->nzcv = rnd() & 0xf0000000u;
}

// Runs function f on (a, b, n) in both copies (a and b are offsets into the copies; b is a value for memset). Returns
// 1 when x0, the Cpu and the bytes in [lo, hi) agree.
static uint8_t *gbase, *hbase;   // what a and b are offsets of: ga and ha, or the page-end blocks
static int run_case(int f, uint64_t a, uint64_t b, uint64_t n, uint64_t lo, uint64_t hi) {
    uint64_t ga0 = (uint64_t)(uintptr_t)gbase, ha0 = (uint64_t)(uintptr_t)hbase;
    int ptr_b = f != F_MEMSET;
    uint64_t gx0 = rt_call(gc, guest[f], 3, ga0 + a, ptr_b ? ga0 + b : b, n);
    Cpu h, before;
    random_cpu(&h);
    h.x[0] = ha0 + a, h.x[1] = ptr_b ? ha0 + b : b, h.x[2] = n;
    before = h;
    hostf[f](&h);
    uint64_t hx0 = h.x[0];
    h.x[0] = before.x[0];
    int cpu_ok = !memcmp(&h, &before, sizeof h);
    // x0 compared as the guest's result relative to its own buffer for the pointer results
    int ptr_ret = f == F_MEMCPY || f == F_MEMSET || f == F_MEMMOVE;
    int x0_ok = ptr_ret ? gx0 - ga0 == hx0 - ha0 : gx0 == hx0;
    int mem_ok = !memcmp(gbase + lo, hbase + lo, hi - lo);
    cases[f]++;
    if (!(cpu_ok && x0_ok && mem_ok)) {
        mismatches[f]++;
        CHECK(0, "%s(a+%llu, %s%llu, %llu): x0 guest %#llx host %#llx%s%s", names[f], (unsigned long long)a,
              ptr_b ? "a+" : "", (unsigned long long)b, (unsigned long long)n,
              (unsigned long long)(ptr_ret ? gx0 - ga0 : gx0), (unsigned long long)(ptr_ret ? hx0 - ha0 : hx0),
              cpu_ok ? "" : ", the host function wrote another register", mem_ok ? "" : ", memory differs");
        return 0;
    }
    return 1;
}

static uint64_t pick_size(void) {
    static const uint64_t big[] = {256, 512, 1024, 4096, 65536, 262144, 1048576, 1835008, 2097152, 4194304, 8388608};
    uint64_t r = rnd() % 100;
    if (r < 60) return rnd() % 257;
    if (r < 90) return 257 + rnd() % 4096;
    uint64_t s = big[rnd() % (sizeof big / sizeof *big)];
    return s + (rnd() % 3) - 1;   // the size and its neighbours
}

static void test_copies(void) {
    for (int f = F_MEMCPY; f <= F_MEMMOVE; f++) {
        int big_done = 0;
        for (int k = 0; k < 40000; k++) {
            uint64_t n = pick_size();
            if (n > (1u << 20) && big_done++ > 60) continue;   // a few dozen of the slow sizes
            uint64_t base = (BUF / 2) & ~63ull, a, b;
            if (f == F_MEMSET) {
                a = base - (n / 2 & ~63ull) + rnd() % 64;
                b = rnd();   // upper bits of w1 set: the low byte only counts
                uint64_t lo = a - GUARD, hi = a + n + GUARD;
                run_case(f, a, b, n, lo, hi);
                restore(lo, hi);
                continue;
            }
            // memmove: every overlap; memcpy: apart or adjacent only (overlap is undefined for it)
            uint64_t kind = rnd() % 6;
            if (f == F_MEMCPY && kind >= 1 && kind <= 3) kind = 0;
            a = base + rnd() % 64;
            int64_t shift;
            switch (kind) {
            case 0: shift = (int64_t)(n + 64 + rnd() % 4096); break;                 // apart
            case 1: shift = n ? (int64_t)(rnd() % n) : 0; break;                     // dst inside src, after it
            case 2: shift = n ? -(int64_t)(rnd() % n) : 0; break;                    // src inside dst
            case 3: shift = 0; break;                                                 // equal
            case 4: shift = (int64_t)n; break;                                        // adjacent
            default: shift = -(int64_t)n - (int64_t)(rnd() % 64); break;
            }
            b = a - (uint64_t)shift;   // source
            if (b < 2 * GUARD || b > BUF || b + n + GUARD > BUF || a + n + GUARD > BUF) continue;
            uint64_t lo = (a < b ? a : b) - GUARD, hi = (a + n > b + n ? a + n : b + n) + GUARD;
            run_case(f, a, b, n, lo, hi);
            restore(lo, hi);
        }
    }
}

// Two equal blocks, then one difference (bytes above 0x7f either side) at a random or every position.
static void test_memcmp(void) {
    for (int k = 0; k < 40000; k++) {
        uint64_t n = pick_size();
        if (n > 65536) n = 65536 + rnd() % 3;
        uint64_t a = (BUF / 4) + rnd() % 64, b = (BUF / 2) + rnd() % 64;
        memcpy(ga + b, ga + a, n), memcpy(ha + b, ha + a, n);
        if (n && rnd() % 8) {
            uint64_t p = k < 2000 ? (uint64_t)k % n : rnd() % n;
            uint8_t v = (uint8_t)rnd();
            if (v == ga[a + p]) v ^= 0x80;
            ga[b + p] = ha[b + p] = v;
        }
        uint64_t len = n ? n - rnd() % (n < 4 ? n : 4) : 0;   // sometimes shorter than the equal run
        run_case(F_MEMCMP, a, b, len, a - GUARD, a + n + GUARD);
        CHECK(!memcmp(ga + b, ha + b, n), "memcmp changed memory");
        restore(a - GUARD, a + n + GUARD), restore(b - GUARD, b + n + GUARD);
    }
}

// Strings of 0-300 characters (non-zero bytes, some above 0x7f), equal or with one difference, and strncmp with n
// below, at and above the length.
static void test_strings(uint8_t *page_end_g, uint8_t *page_end_h) {
    for (int k = 0; k < 60000; k++) {
        uint64_t len = rnd() % 8 ? rnd() % 64 : rnd() % 301;
        uint64_t a = (BUF / 4) + rnd() % 64, b = (BUF / 2) + rnd() % 64;
        for (uint64_t i = 0; i < len; i++) {
            uint8_t v = (uint8_t)(rnd() | 1);
            ga[a + i] = ha[a + i] = v, ga[b + i] = ha[b + i] = v;
        }
        ga[a + len] = ha[a + len] = 0;
        ga[b + len] = ha[b + len] = 0;
        uint64_t diff = len + 1;
        if (rnd() % 4) {
            diff = rnd() % (len + 1);
            uint8_t v = (uint8_t)rnd();
            if (v == ga[a + diff]) v ^= 0x80;
            ga[b + diff] = ha[b + diff] = v;   // may end b early (v 0) or lengthen it past a's end
        }
        // nothing is written: the bytes around a stay equal in both copies (both get the same stores)
        uint64_t lo = a - GUARD, hi = a + len + GUARD;
        run_case(F_STRLEN, a, 0, 0, lo, hi);
        run_case(F_STRCMP, a, b, 0, lo, hi);
        run_case(F_STRCMP, b, a, 0, lo, hi);
        run_case(F_STRNCMP, a, b, rnd() % (len + 20), lo, hi);
        run_case(F_STRNCMP, a, b, diff + rnd() % 3, lo, hi);
    }
    // Strings ending on the last byte of a page whose next page is inaccessible (a block of its own per copy, outside
    // the watched heap, so an overread faults and stops the probe), against strings at the start of the same page.
    gbase = page_end_g, hbase = page_end_h;
    for (uint64_t len = 0; len < 80; len++)
        for (uint64_t mis = 0; mis < 16; mis++) {
            uint64_t a = 4096 - len - 1, b = 64 + mis;   // a's NUL is the page's last byte
            for (int k = 0; k < 2; k++) {
                uint8_t *p = k ? hbase : gbase;
                for (uint64_t i = 0; i < len; i++) p[a + i] = p[b + i] = (uint8_t)('a' + i % 26);
                p[a + len] = p[b + len] = 0;
            }
            run_case(F_STRLEN, a, 0, 0, a, a + len + 1);
            run_case(F_STRCMP, a, b, 0, a, a + len + 1);
            run_case(F_STRCMP, b, a, 0, a, a + len + 1);
            run_case(F_STRNCMP, a, b, len + 8, a, a + len + 1);
            run_case(F_STRNCMP, b, a, len + 8, a, a + len + 1);
            if (len) {   // one difference at the end
                gbase[b + len - 1] = hbase[b + len - 1] = 0x80;
                run_case(F_STRCMP, a, b, 0, a, a + len + 1);
                run_case(F_STRNCMP, b, a, len + 8, a, a + len + 1);
            }
        }
    gbase = ga, hbase = ha;
}

// ---- timing
static uint64_t now(void) { return kern_now_ns(); }
static int cmp_u64(const void *x, const void *y) {
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return (a > b) - (a < b);
}

// ns per call of mode 0 (the lifted function directly), 1 (the host function through rt_dispatch), 2 (the host
// function directly), the median of 9 rounds.
static double time_one(int f, int mode, uint64_t a, uint64_t b, uint64_t n) {
    uint64_t base = (uint64_t)(uintptr_t)ga;
    int reps = n > 65536 ? 20 : n > 4096 ? 200 : 2000;
    uint64_t r[9];
    Cpu *c = gc;
    for (int k = 0; k < 9; k++) {
        uint64_t t0 = now();
        for (int i = 0; i < reps; i++) {
            c->x[0] = base + a, c->x[1] = f == F_MEMSET ? b : base + b, c->x[2] = n;
            if (mode == 0) {
                c->x[30] = RT_RET_HOST;
                lifted[f](c);
            } else if (mode == 1) {
                c->pc = hostaddr[f];
                rt_dispatch(c);
            } else {
                hostf[f](c);
            }
        }
        r[k] = now() - t0;
    }
    qsort(r, 9, sizeof *r, cmp_u64);
    return (double)r[4] / reps;
}

// Each function twice: both pointers 64-byte aligned, then the first one byte and the second four bytes past that
// (their difference not a multiple of 8, where the sdk's memcmp, strcmp and strncmp go a byte at a time). memcpy's
// source lies a further half page on: a multiple of 4 KiB from the destination, each load had the low 12 address bits
// of the previous call's store and waited for it (4 KiB aliasing). The other functions do not need it: the compares
// only load, and memmove is timed overlapping.
static void timing(void) {
    static const uint64_t sizes[] = {0, 1, 4, 8, 16, 24, 32, 64, 112, 128, 256, 512, 1024, 4096, 65536, 1u << 20, 4u << 20};
    printf("\ntiming, ns per call (median of 9 rounds): lifted direct / host via rt_dispatch / host direct\n");
    for (int f = 0; f < F_N; f++) {
        if (!lifted[f] || !hostaddr[f]) continue;
        for (int mis = 0; mis < 2; mis++) {
            printf("  %-8s %s", names[f], mis ? "+1/+4  " : "aligned");
            for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
                uint64_t n = sizes[i];
                if (f >= F_STRLEN && n > 4096) continue;
                uint64_t a = BUF / 4 + (mis ? 1 : 0), b = BUF / 2 + (mis ? 4 : 0);
                if (f >= F_STRLEN) {   // equal strings of n characters
                    memset(ga + a, 'x', n), memset(ga + b, 'x', n);
                    ga[a + n] = ga[b + n] = 0;
                } else if (f == F_MEMCMP) {
                    memcpy(ga + b, ga + a, n);   // equal blocks: the whole length is compared
                } else if (f == F_MEMCPY) {
                    b += 2048;                   // half a page, see above timing()
                } else if (f == F_MEMMOVE && n) {
                    b = a + n / 2 + 1;           // overlapping, the destination below the source
                }
                uint64_t arg = f == F_MEMSET ? 0x5a : b;
                double t0 = time_one(f, 0, a, arg, n), t1 = time_one(f, 1, a, arg, n), t2 = time_one(f, 2, a, arg, n);
                printf(" | %llu: %.1f/%.1f/%.1f", (unsigned long long)n, t0, t1, t2);
            }
            printf("\n");
        }
    }
}

static uint8_t *page_g, *page_h;

static void probe_main(Cpu *c) {
    gc = c;
    uint64_t heap = 0;
    CHECK(mem_set_heap_size(3 * BUF + (64u << 20), &heap) == 0 && heap, "SetHeapSize");
    if (!heap) return;
    ga = (uint8_t *)(uintptr_t)heap;
    ha = ga + BUF + (8u << 20);
    pristine = ha + BUF + (8u << 20);
    gbase = ga, hbase = ha;
    fill_random(pristine, BUF);
    memcpy(ga, pristine, BUF), memcpy(ha, pristine, BUF);
    // page-end strings: two pages per copy, the second made inaccessible
    page_g = plat_mem_alloc(8192), page_h = plat_mem_alloc(8192);
    CHECK(page_g && page_h, "page-end blocks");
    if (!page_g || !page_h) return;
    memset(page_g, 1, 8192), memset(page_h, 1, 8192);
    CHECK(plat_mem_protect((uint64_t)(uintptr_t)page_g + 4096, 4096, PLAT_PROT_NONE) &&
              plat_mem_protect((uint64_t)(uintptr_t)page_h + 4096, 4096, PLAT_PROT_NONE),
          "inaccessible page after the page-end strings");
    for (int f = 0; f < F_N; f++) {
        guest[f] = ldr_symbol(names[f]);
        lifted[f] = guest[f] ? rt_dispatch_lookup(guest[f]) : NULL;
        hostf[f] = libc_host_fn(names[f]);
        uint64_t slot[4];
        hostaddr[f] = ldr_jump_slots(names[f], slot, 4) ? LD64(slot[0]) : 0;
        CHECK(guest[f] && lifted[f] && hostf[f], "%s: guest %#llx, lifted %p, host %p", names[f],
              (unsigned long long)guest[f], (void *)lifted[f], (void *)hostf[f]);
        CHECK(hostaddr[f] - RT_HOSTFN_BASE < RT_HOSTFN_SIZE, "%s: the GOT slot holds %#llx, not a host function",
              names[f], (unsigned long long)hostaddr[f]);
    }
    if (failures) return;
    test_copies();
    test_memcmp();
    test_strings(page_g, page_h);
    printf("cases (mismatches):");
    for (int f = 0; f < F_N; f++) printf(" %s %llu (%llu)", names[f], (unsigned long long)cases[f], (unsigned long long)mismatches[f]);
    printf("\n");
    timing();
}

int probe_libc(void) {
    libc_bind_all();   // the host functions in every GOT slot, whatever host_libc says
    ldr_load(rt_config()->exefs);
    hostfn_seal();           // the dispatch table, for rt_dispatch_lookup
    hostfn_count_calls(0);   // as in a game run: hostfn_entry jumps straight to the function
    kern_run_on_guest_main(probe_main);
    if (failures) {
        printf("probe-libc: FAILED (%d of %d checks)\n", failures, checks);
        return 1;
    }
    printf("probe-libc: passed (%d checks)\n", checks);
    return 0;
}
