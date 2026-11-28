// runtime/memory.c on the macOS platform layer, without the game: the guest layout mem_init finds on Apple Silicon
// (where the GPU carve-out leaves only 448-512 GiB below 2^39), QueryMemory around it, the heap, physical memory in
// the alias region, a freeze and a hold. Needs generated/<version>/layout.h (tools/build_game.py's generator output)
// for the module window. Build and run:
//
//   SDL="$(brew --prefix sdl3)"
//   clang -std=c11 -O1 -Wall -Wextra -Werror -Iruntime -Igenerated/v101 tests/macos_memory_test.c runtime/memory.c \
//       runtime/platform/macos.c runtime/platform/macos_mem.c runtime/platform/macos_posix.c \
//       runtime/platform/common.c -framework CoreFoundation -framework CoreText -framework IOKit \
//       -I"$SDL/include" -L"$SDL/lib" -lSDL3 -Wl,-rpath,"$SDL/lib" -o build/macos-memory-test && build/macos-memory-test
//
// SHOW_WINDOW=1 also prints what occupies [448 GiB, 512 GiB) after mem_init.
#include <execinfo.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"

// ---- what memory.c needs from the rest of the runtime
const RtLayoutEntry rt_layout[] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE}, {"sdk", MOD_SDK_BASE, MOD_SDK_SIZE}};
const int rt_layout_count = 2;
const char *cfg_get(const char *name) { return getenv(name); }
void cfg_init(void) {}
int ldr_count(void) { return 0; }
const RtLoaded *ldr_module(int i) { (void)i; return NULL; }
const char *ldr_describe(uint64_t a, char *buf, int n) { snprintf(buf, (size_t)n, "%#llx", (unsigned long long)a); return buf; }
uint64_t res_heap_extra_bytes(void) { return 0; }
void rt_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
}
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "FATAL: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(2);
}

static int checks, failures;
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

static void fetch_pattern(void *ctx, uint64_t a, uint64_t n, void *dst) {
    (void)a;
    memset(dst, *(int *)ctx, n);
}

// main.c's crash report, short: where, what, and the frames.
static void crash(const PlatCrash *c) {
    fprintf(stderr, "CRASH %s (signal %u) at pc exe+%#llx, %s %#llx, module %s\n", c->what, c->code,
            (unsigned long long)c->offset, c->access == 1 ? "write" : c->access == 0 ? "read" : "access",
            (unsigned long long)c->addr, c->module ? c->module : "-");
    void *f[32];
    backtrace_symbols_fd(f, backtrace(f, 32), 2);
}

// What else is in [448 GiB, 512 GiB) once the guest is in: the window the layout had to fit into, as runs of guest
// reservations, other mappings and free space.
static void show_run(uint64_t lo, uint64_t hi, int kind) {
    static const char *const what[] = {"free", "guest", "other"};
    if (hi > lo) printf("  %#llx-%#llx %6.2f GiB %s\n", (unsigned long long)lo, (unsigned long long)hi,
                        (hi - lo) / 1073741824.0, what[kind]);
}

static void show_window(void) {
    uint64_t at = 0x7000000000ull, run = at, end;
    int kind = 0;
    while (at < RT_ADDRESS_LIMIT) {
        mach_vm_address_t a = at;
        mach_vm_size_t sz = 0;
        natural_t depth = 0;
        vm_region_submap_info_data_64_t i;
        mach_msg_type_number_t c = VM_REGION_SUBMAP_INFO_COUNT_64;
        if (mach_vm_region_recurse(mach_task_self(), &a, &sz, &depth, (vm_region_recurse_info_t)&i, &c) ||
            a >= RT_ADDRESS_LIMIT)
            break;
        if (a > at && kind != 0) show_run(run, at, kind), run = at, kind = 0;   // a gap before the entry
        int now = plat_mem_state(a, a + 1, &end) != PLAT_MEM_FREE ? 1 : 2;
        if (now != kind) show_run(run, a, kind), run = a, kind = now;
        at = a + sz < RT_ADDRESS_LIMIT ? a + sz : RT_ADDRESS_LIMIT;
    }
    if (kind != 0 && at < RT_ADDRESS_LIMIT) show_run(run, at, kind), run = at, kind = 0;
    show_run(run, RT_ADDRESS_LIMIT, kind);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);   // a fault leaves what was printed before it
    plat_crash_hook(crash);
    mem_init();
    const RtGuestLayout *g = mem_layout();
    uint64_t low = g->stack.base - 0x10000000ull;
    printf("guest layout: aslr %#llx+%#llx stack %#llx heap %#llx alias %#llx+%#llx (%llu GiB) kernel %#llx\n",
           (unsigned long long)g->aslr.base, (unsigned long long)g->aslr.size, (unsigned long long)g->stack.base,
           (unsigned long long)g->heap.base, (unsigned long long)g->alias.base, (unsigned long long)g->alias.size,
           (unsigned long long)(g->alias.size >> 30), (unsigned long long)g->kernel.base);
    if (getenv("SHOW_WINDOW")) show_window();
    CHECK(g->alias.base + g->alias.size <= RT_ADDRESS_LIMIT && g->alias.size >= (8ull << 30), "alias region");
    CHECK(g->heap.base + g->heap.size <= g->alias.base, "heap below the alias region");
    CHECK((g->alias.base + g->alias.size <= MOD_MAIN_BASE || g->stack.base >= g->hostfn.base + g->hostfn.size) &&
              g->aslr.base <= MOD_MAIN_BASE && g->aslr.base <= g->stack.base &&
              g->aslr.base + g->aslr.size == RT_ADDRESS_LIMIT,
          "guest regions overlap or leave the ASLR region");

    RtMemoryInfo q;
    mem_query(MOD_MAIN_BASE, &q);
    CHECK(q.state != 0, "the module window is not free to the guest (state %#x)", q.state);
    mem_query(g->kernel.base, &q);
    CHECK(q.state == 0xC && q.base == g->kernel.base, "kernel region is thread-local");
    mem_query(g->hostfn.base, &q);
    CHECK(q.state == 0x10, "host-function window is inaccessible");
    mem_query(g->alias.base, &q);
    CHECK(q.state == 0 && q.base == g->alias.base && q.size == g->alias.size, "free alias region, whole (%#llx+%#llx)",
          (unsigned long long)q.base, (unsigned long long)q.size);
    mem_query(g->stack.base, &q);
    CHECK(q.state == 0 && q.base == low, "free start of the stack and heap block");
    if (g->aslr.base > 0x100000000ull) {
        mem_query(g->aslr.base - 0x1000, &q);
        CHECK(q.state == 0 && q.base + q.size == g->aslr.base, "below the ASLR region is free");
    }

    uint64_t heap = 0;
    CHECK(mem_set_heap_size(64ull << 20, &heap) == 0 && heap == g->heap.base, "SetHeapSize 64 MiB");
    memset((void *)(uintptr_t)heap, 0x3c, 64ull << 20);
    mem_query(heap + 0x123, &q);
    CHECK(q.state == 5 && q.base == heap && q.size == 64ull << 20, "QueryMemory on the heap");

    uint64_t a = g->alias.base + (1ull << 30);
    CHECK(mem_map_physical(a, 0x200000) == 0, "MapPhysicalMemory 2 MiB");
    CHECK(*(volatile uint64_t *)(uintptr_t)(a + 0x1ff8) == 0, "physical memory reads zero");
    memset((void *)(uintptr_t)a, 0x77, 0x200000);
    mem_query(a + 0x10, &q);
    CHECK(q.state == 5 && q.base == a && q.size == 0x200000, "QueryMemory on mapped physical memory");
    mem_query(a + 0x200000, &q);
    CHECK(q.state == 0 && q.base == a + 0x200000 && q.base + q.size == g->alias.base + g->alias.size,
          "free alias memory after it ends at the region's end");

    // a freeze makes the region read-only; the first write thaws it (memory.c thaw_fault through the platform)
    CHECK(mem_freeze(a), "freeze the 2 MiB region");
    CHECK(mem_frozen(a), "frozen");
    *(volatile uint8_t *)(uintptr_t)(a + 0x4321) = 1;
    CHECK(!mem_frozen(a) && *(volatile uint8_t *)(uintptr_t)(a + 0x4321) == 1, "a write thawed it");

    // a hold: whole host pages only, and the first read fetches the bytes back
    uint64_t gr = plat_mem_grain();
    static int pat = 0x42;
    CHECK(!mem_hold(a + 0x1000, gr, fetch_pattern, &pat) || gr == 0x1000, "a hold off the host page grain is refused");
    CHECK(mem_hold(a + 4 * gr, 4 * gr, fetch_pattern, &pat), "hold 4 host pages");
    CHECK(*(volatile uint8_t *)(uintptr_t)(a + 5 * gr + 3) == 0x42, "the first read fetched the held bytes");
    CHECK(*(volatile uint8_t *)(uintptr_t)(a + 4 * gr - 1) == 0x77, "the page before the hold kept its bytes");
    uint64_t lo, hi;
    CHECK(!mem_held_by(a, a + 0x200000, &pat, &lo, &hi), "and the hold is gone");

    CHECK(mem_unmap_physical(a, 0x200000) == 0, "UnmapPhysicalMemory");
    mem_query(a, &q);
    CHECK(q.state == 0 && q.base == g->alias.base && q.size == g->alias.size, "the alias region is free again");
    CHECK(mem_map_physical(a, 0x1000) == 0 && *(volatile uint64_t *)(uintptr_t)(a + 8) == 0,
          "remapped physical memory reads zero");
    printf("macos memory test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
