// Host versions of the C library functions the game spends measurable time in. The sdk's own are portable C compiled
// for ARM64 (musl's): memcmp compares a byte at a time unless both pointers share their alignment, 8 bytes at a time
// otherwise. The game calls it after every uniform block it fills, to compare the block with the previous one of the
// same ring (main+0x9d6510, sizes rounded up to 256 bytes): in Harmony Square the lifted memcmp took 7.5% of the main
// thread's running time. memcpy is replaced in nvn_api_core.c, where it also reads back what the GPU wrote.
#include <string.h>

#include "host.h"

// musl's result is the difference of the first two bytes that differ; the host's memcmp only promises its sign. Equal
// blocks, the common case, take the host's fast path; otherwise the first differing byte is found in 32-byte steps.
static void ov_memcmp(Cpu *c) {
    const unsigned char *a = arg_p(c, 0), *b = arg_p(c, 1);
    uint64_t n = arg_x(c, 2);
    int r = 0;
    if (n && memcmp(a, b, (size_t)n)) {
        uint64_t i = 0;
        while (n - i > 32 && !memcmp(a + i, b + i, 32)) i += 32;
        while (a[i] == b[i]) i++;
        r = a[i] - b[i];
    }
    ret_x(c, (uint32_t)r);
}
HOST_OVERRIDE("memcmp", ov_memcmp);

static void ov_memset(Cpu *c) {
    void *d = arg_p(c, 0);
    memset(d, (int)arg_x(c, 1), (size_t)arg_x(c, 2));
    ret_x(c, (uint64_t)(uintptr_t)d);
}
HOST_OVERRIDE("memset", ov_memset);
