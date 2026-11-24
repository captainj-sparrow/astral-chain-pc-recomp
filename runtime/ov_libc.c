// Host versions of the sdk's C library functions. Behaviour described from Longshot f3aef2a6 (GPL v3; no code taken).
//
// memcpy always runs here (a linked HOST_OVERRIDE, moved from nvn_api_core.c): a copy of pool storage the GPU rendered
// into must copy what the GPU wrote (Astral Chain moves its generated grass maps between pools with memcpy), so copies
// of 1 KiB or more first ask the back end for those bytes (gfx guest_read). memset, memmove, memcmp, strlen, strcmp
// and strncmp run here when host_libc is on (cfg.c, default 1): libc_init binds them with hostfn_add_override before
// ldr_load, so the lift stays as it is (each import stub's rt_stub_hit fails and rt_dispatch reaches the host function)
// and host_libc = 0 gives back the direct stub calls of the lifted functions. The sdk's memmove hands its
// non-overlapping copies to memcpy through its PLT, so memmove is the same function as memcpy, guest_read included.
//
// Results, as the sdk's (--probe-libc compares every byte and register with the lifted functions): memcpy, memmove
// and memset return the destination in x0, untouched; memcmp, strcmp and strncmp return the difference of the first
// unequal bytes as unsigned bytes, a 32-bit value zero-extended in x0 (the C runtime promises only the sign, so its
// memcmp and strcmp are not used); strlen counts to the NUL. The host functions write x0 and nothing else: no other X
// register, no V register, no flags. strlen, strcmp and strncmp never load past the end of a page the bytes they need
// lie in (an aligned 16-byte load cannot; an unaligned one is used only 16 bytes or more before a page end).
//
// Sizes. Up to 128 bytes, copies and fills are overlapping 16-, 8-, 4-byte or byte stores, every load of a copy done
// before its first store, so an overlapping copy needs no direction; above, the C runtime's memmove and memset. memcmp
// compares 64 bytes per step, then 16, then the last 16 overlapping; below 16 bytes two overlapping 8- or 4-byte loads.
// There is no size gate: a host function cannot hand a size class back to the lifted code for less than the dispatch
// it already paid (rt_call costs tens of ns). --probe-libc (2026-11-22, Core Ultra 9 285H, Windows, unpinned, with
// hostfn_direct, median of seven runs) times both, the host function reached through rt_dispatch as the stubs reach it
// (ns, lifted / host, 64-byte aligned pointers): memcpy 1 B 2.2 / 2.0, 64 B 2.4 / 1.7; memset 112 B 6.1 / 1.8, 1 KiB
// 17.3 / 4.2; overlapping memmove 64 B 13.2 / 4.8; memcmp of equal blocks 0 B 1.3 / 1.2, 4 B 2.1 / 1.5, 64 B 3.5 / 1.8,
// 512 B 26.4 / 5.5; strlen 16 1.6 / 1.4, 32 2.8 / 1.5; strcmp of equal strings 1 1.6 / 1.9, 8 1.8 / 1.8, 64 3.9 / 3.3.
// The host side loses on strcmp of 1 to 4 characters only, by up to 0.4 ns. With the two pointers 3 bytes out of step,
// where the sdk's memcmp, strcmp and strncmp go a byte at a time: memcmp 64 B 20.1 / 1.8, 512 B 109.0 / 7.0; strcmp 16
// 5.9 / 1.9, 64 19.4 / 3.2; strncmp 64 21.7 / 3.5.
//
// Large blocks. ucrtbase's memmove switches to non-temporal stores between 1 and 1.75 MiB here: a 1.75 MiB copy took
// 32 us instead of 66, and reading the destination afterwards 65 us instead of 20 (Core Ultra 9 285H, P-core, median of
// 21; copybench, copy of a just-written source then a read of the destination). On Windows copies from 1 to 8 MiB go in
// pieces of 256 KiB, each a cached copy: copy then read 4-14% faster from 1.75 to 8 MiB (P-core 2 MiB 110.7 -> 104.5 us,
// 4 MiB 278.8 -> 263.3; E-core 4 MiB 376.9 -> 324.0, 8 MiB 794 -> 754). Above 8 MiB the pieces lost (16 MiB 1183 ->
// 1471 us on the P-core: source and destination no longer fit the 24 MiB L3), so those stay one call. 512 KiB pieces
// measured the same as 256, 64 KiB pieces slower. memset showed no such switch up to 8 MiB and stays one call. A
// preemption point (kern_checkpoint, valid on a guest thread as at an SVC) follows each piece: host code is never
// moved by the asynchronous preemption. The game copies 1.75 MiB or more only while it loads (three memcpy and five to
// seven memset of that size per run, none in play), so in play the pieces change nothing. Elsewhere a copy is one
// memmove of any size, with no preemption point inside (the Linux figures are below).
//
// Measured in the game (2026-11-22, Core Ultra 9 285H with the Arc 140T, a plain build, the game pinned to four E-cores
// as a CPU-bound stand-in, the motorway chase of the opening at 60 fps; the profile's save holds only the start of the
// game, so no fight on foot, Neuron headquarters or Harmony Square was reachable by script):
// - Before (three 1 ms profiles, 60 s of the chase each): the six lifted functions, their stubs and memcpy's path took
//   2.3 to 2.9% of the render thread's running samples (memcmp alone 1.5 to 1.9%), 1.1 to 1.6% of all guest threads'.
//   Calls per frame (a counting build): render thread memcpy 5,500, memcmp 680, memset 275, strlen 176, strcmp 120;
//   the busiest job thread memcpy 4,000, memcmp 640; another up to 12,700 memcpy; memmove and strncmp under one. memcmp
//   compares 4-63 bytes (strstr) and 256-1023 (the uniform block compare at main+0x9d6510); memset fills 64 B-1 KiB.
//   From the title to the motorway (the load, unpinned) they took 2.4% of the running samples, 4% of the guest ones.
// - In-launch A/B (AC_HOST_LIBC_AB=1, AC_AB=16, two runs of three minutes, 85-88 windows of 2 s a side): the render
//   thread's guest time per frame -2.5% +-1.7 and -3.0% +-1.9, its CPU time -3.9% +-2.9 and -8.1% +-3.4; the four
//   other busy guest threads' guest time -1 to -15%. Across launches (off, on, on, off, off, on, 48 s each): the
//   render thread's CPU per frame 5.17, 5.25, 5.20 ms off against 4.39, 4.58, 3.66 on; all guest threads 11.30, 11.57,
//   11.49 against 11.22, 10.59, 10.56. Both exceed the functions' own share: on four cores the time they free also
//   shortens other threads' waits (the replay thread's time fell 16% in the same A/B blocks). In a profile after the
//   change the host memcmp took 1.6% of the render thread, about what the lifted one took: its time is the cache
//   misses on the blocks it compares, as the CPU study of 2026-11-17 found (tools/cpu_study/ov_libc.c, its own host
//   memcmp and memset, about -0.6% then).
// - Direct calls from the lifter (helpers with a register contract in place of the stub call) were not made: the stub
//   and dispatch they would save were put at about 1 ns a call (since hostfn_direct the probe's dispatch alone is
//   mostly 0.1-0.5 ns), the hottest memcpy callers reach the stub by tail calls from small wrappers (main+0x9b1700 copies 64
//   bytes), and the register stores the call site would skip come to a few ns, so about 0.5% of the render thread's
//   time at most, the level a cross-launch comparison cannot resolve here, for a re-lift and a new PGO profile.
//
// Measured on Linux (2026-11-21, ROG Ally, Z1 Extreme at 15 W, glibc 2.44, Steam Deck profile, 4 cores / 8 threads,
// the town and headquarters saves with the camera turning):
// - No plat_mem_will_write here. Until this date copies and fills of 64 KiB or more called it, to save a write-watch
//   fault per page; on pages that are not protected the ioctl costs about twice the copy (--probe-libc, ns: memcpy
//   64 KiB 2,794 -> 925 without it, the lifted one 873; memset 64 KiB 2,210 -> 450, the lifted one 964). Three town
//   loads with every other call timed without it: 94% of the 1.3 GiB such calls move per load had no protected page
//   and took 54 ns/KiB longer with the ioctl; blocks mostly protected, under 3% of the bytes, copied up to twice as
//   fast with it. All copies together: 816 ns/KiB without, 860 with, about 55 ms of CPU time per load, summed over
//   threads (the load itself was not timed). For memset the game's 200 or so such calls, of very different sizes, did
//   not settle it: that half rests on the probe. None of that size seen in play. (The probe's memcpy source then lay
//   4 MiB from its destination, so with 4 KiB aliasing; not timed again on the Ally since the probe moved it.)
// - No lower threshold either. Of 285 write faults a frame in the town, 30 were copies of 128 bytes or less (a page
//   each) and 14 came from the 75 or so memsets a frame of 4 to 32 KiB: an ioctl for each would cost more.
// - guest_read left as it is: 62 to 68 copies a frame of 1 KiB or more, none of 220,000 checks found a stale texture,
//   vk_texture_cpu_stale 0.01% of the one thread it showed on.
// - No pieces. glibc switches to non-temporal stores and rep movsb at 12 MiB here (3/4 of the L3), above every size
//   the pieces cover, so one call already is a cached copy and the cuts only cost: 1-4% of copy + read up to 4 MiB,
//   up to 7% at 6-8 MiB (200-400 cycles a cut up to 4 MiB, not the calls themselves: a loop of one piece costs
//   nothing). With the threshold lowered to 1 or 3 MiB by tunables the pieces win again, but that says nothing sure
//   of a Deck: it leaves the L3 at 16 MiB, and the largest gains were misaligned rep movsb copies, slow on Zen 4.
// Not measured: a real Steam Deck (Zen 2, 4 MiB of L3), a desktop, and a --release-opt build, whose PGO and ThinLTO
// make the lifted functions (generated code) faster and the gain smaller.
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"

static inline uint64_t ld64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline void st64(void *p, uint64_t v) { memcpy(p, &v, 8); }
static inline void st32(void *p, uint32_t v) { memcpy(p, &v, 4); }

// The difference of the first unequal bytes, unsigned, as the sdk returns it: a 32-bit value zero-extended in x0.
static inline uint64_t byte_diff(const uint8_t *a, const uint8_t *b, size_t i) { return (uint32_t)((uint32_t)a[i] - (uint32_t)b[i]); }

// ---- memcmp
static uint64_t libc_memcmp(const uint8_t *a, const uint8_t *b, size_t n) {
    if (n >= 16) {
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            __m128i e0 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(a + i)), _mm_loadu_si128((const __m128i *)(b + i)));
            __m128i e1 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(a + i + 16)), _mm_loadu_si128((const __m128i *)(b + i + 16)));
            __m128i e2 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(a + i + 32)), _mm_loadu_si128((const __m128i *)(b + i + 32)));
            __m128i e3 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(a + i + 48)), _mm_loadu_si128((const __m128i *)(b + i + 48)));
            __m128i all = _mm_and_si128(_mm_and_si128(e0, e1), _mm_and_si128(e2, e3));
            if (_mm_movemask_epi8(all) != 0xffff) {
                uint64_t m = (uint64_t)(uint16_t)~_mm_movemask_epi8(e0) | (uint64_t)(uint16_t)~_mm_movemask_epi8(e1) << 16 |
                             (uint64_t)(uint16_t)~_mm_movemask_epi8(e2) << 32 | (uint64_t)(uint16_t)~_mm_movemask_epi8(e3) << 48;
                return byte_diff(a, b, i + (size_t)__builtin_ctzll(m));
            }
        }
        for (; i + 16 <= n; i += 16) {
            unsigned m = (uint16_t)~_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(a + i)),
                                                                   _mm_loadu_si128((const __m128i *)(b + i))));
            if (m) return byte_diff(a, b, i + (size_t)__builtin_ctz(m));
        }
        if (i < n) {   // the last 16 bytes, overlapping what was compared
            i = n - 16;
            unsigned m = (uint16_t)~_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(a + i)),
                                                                   _mm_loadu_si128((const __m128i *)(b + i))));
            if (m) return byte_diff(a, b, i + (size_t)__builtin_ctz(m));
        }
        return 0;
    }
    if (n >= 8) {
        uint64_t x = ld64(a) ^ ld64(b);
        if (x) return byte_diff(a, b, (size_t)__builtin_ctzll(x) >> 3);
        x = ld64(a + n - 8) ^ ld64(b + n - 8);
        return x ? byte_diff(a, b, n - 8 + ((size_t)__builtin_ctzll(x) >> 3)) : 0;
    }
    if (n >= 4) {
        uint32_t x = ld32(a) ^ ld32(b);
        if (x) return byte_diff(a, b, (size_t)__builtin_ctz(x) >> 3);
        x = ld32(a + n - 4) ^ ld32(b + n - 4);
        return x ? byte_diff(a, b, n - 4 + ((size_t)__builtin_ctz(x) >> 3)) : 0;
    }
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i]) return byte_diff(a, b, i);
    return 0;
}

// ---- strcmp / strncmp / strlen
// 16 bytes at a time while neither string is within 16 bytes of the end of its page, so a load never touches the next
// page, which may be unmapped; a byte at a time across the boundary.
#define NEAR_PAGE_END(p) (((uintptr_t)(p) & 4095) > 4096 - 16)
static uint64_t libc_strcmp(const uint8_t *a, const uint8_t *b) {
    if (a[0] != b[0] || !a[0]) return byte_diff(a, b, 0);
    size_t i = 1;
    for (;;) {
        if (NEAR_PAGE_END(a + i) || NEAR_PAGE_END(b + i)) {
            uint8_t x = a[i], y = b[i];
            if (x != y || !x) return byte_diff(a, b, i);
            i++;
            continue;
        }
        __m128i x = _mm_loadu_si128((const __m128i *)(a + i)), y = _mm_loadu_si128((const __m128i *)(b + i));
        // a byte that differs or is the end of a
        unsigned m = (uint16_t)~_mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) | (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(x, _mm_setzero_si128()));
        if (m) return byte_diff(a, b, i + (size_t)__builtin_ctz(m));
        i += 16;
    }
}

// Below 16 bytes to go the 16-byte load still runs (the page test allows it) and the bytes past n are masked off.
static uint64_t libc_strncmp(const uint8_t *a, const uint8_t *b, size_t n) {
    size_t i = 0;
    while (i < n) {
        if (NEAR_PAGE_END(a + i) || NEAR_PAGE_END(b + i)) {
            uint8_t x = a[i], y = b[i];
            if (x != y || !x) return byte_diff(a, b, i);
            i++;
            continue;
        }
        __m128i x = _mm_loadu_si128((const __m128i *)(a + i)), y = _mm_loadu_si128((const __m128i *)(b + i));
        unsigned m = (uint16_t)~_mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) | (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(x, _mm_setzero_si128()));
        if (n - i < 16) m &= (1u << (n - i)) - 1;
        if (m) return byte_diff(a, b, i + (size_t)__builtin_ctz(m));
        i += 16;
    }
    return 0;
}

// Aligned 16-byte loads never cross a page.
static uint64_t libc_strlen(const uint8_t *s) {
    const uint8_t *p = (const uint8_t *)((uintptr_t)s & ~(uintptr_t)15);
    unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)p), _mm_setzero_si128()));
    m >>= (uintptr_t)s & 15;
    if (m) return (uint64_t)__builtin_ctz(m);
    for (;;) {
        p += 16;
        m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_load_si128((const __m128i *)p), _mm_setzero_si128()));
        if (m) return (uint64_t)(p - s) + (uint64_t)__builtin_ctz(m);
    }
}

// ---- memset / memmove
// Up to 128 bytes: overlapping stores (and, for a copy, every load before the first store, so overlap is safe).
static inline void set_small(uint8_t *d, uint8_t v, size_t n) {
    if (n >= 16) {
        __m128i x = _mm_set1_epi8((char)v);
        if (n > 64) {
            _mm_storeu_si128((__m128i *)d, x), _mm_storeu_si128((__m128i *)(d + 16), x);
            _mm_storeu_si128((__m128i *)(d + 32), x), _mm_storeu_si128((__m128i *)(d + 48), x);
            d += n - 64, n = 64;
        }
        if (n > 32) {
            _mm_storeu_si128((__m128i *)d, x), _mm_storeu_si128((__m128i *)(d + 16), x);
            _mm_storeu_si128((__m128i *)(d + n - 32), x), _mm_storeu_si128((__m128i *)(d + n - 16), x);
        } else {
            _mm_storeu_si128((__m128i *)d, x), _mm_storeu_si128((__m128i *)(d + n - 16), x);
        }
    } else if (n >= 8) {
        uint64_t x = 0x0101010101010101ull * v;
        st64(d, x), st64(d + n - 8, x);
    } else if (n >= 4) {
        uint32_t x = 0x01010101u * v;
        st32(d, x), st32(d + n - 4, x);
    } else if (n) {
        d[0] = v, d[n - 1] = v, d[n >> 1] = v;
    }
}

static inline void move_small(uint8_t *d, const uint8_t *s, size_t n) {
    if (n > 64) {   // 65..128
        __m128i a = _mm_loadu_si128((const __m128i *)s), b = _mm_loadu_si128((const __m128i *)(s + 16));
        __m128i e = _mm_loadu_si128((const __m128i *)(s + 32)), f = _mm_loadu_si128((const __m128i *)(s + 48));
        __m128i g = _mm_loadu_si128((const __m128i *)(s + n - 64)), h = _mm_loadu_si128((const __m128i *)(s + n - 48));
        __m128i i = _mm_loadu_si128((const __m128i *)(s + n - 32)), j = _mm_loadu_si128((const __m128i *)(s + n - 16));
        _mm_storeu_si128((__m128i *)d, a), _mm_storeu_si128((__m128i *)(d + 16), b);
        _mm_storeu_si128((__m128i *)(d + 32), e), _mm_storeu_si128((__m128i *)(d + 48), f);
        _mm_storeu_si128((__m128i *)(d + n - 64), g), _mm_storeu_si128((__m128i *)(d + n - 48), h);
        _mm_storeu_si128((__m128i *)(d + n - 32), i), _mm_storeu_si128((__m128i *)(d + n - 16), j);
    } else if (n > 32) {
        __m128i a = _mm_loadu_si128((const __m128i *)s), b = _mm_loadu_si128((const __m128i *)(s + 16));
        __m128i i = _mm_loadu_si128((const __m128i *)(s + n - 32)), j = _mm_loadu_si128((const __m128i *)(s + n - 16));
        _mm_storeu_si128((__m128i *)d, a), _mm_storeu_si128((__m128i *)(d + 16), b);
        _mm_storeu_si128((__m128i *)(d + n - 32), i), _mm_storeu_si128((__m128i *)(d + n - 16), j);
    } else if (n >= 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)s), j = _mm_loadu_si128((const __m128i *)(s + n - 16));
        _mm_storeu_si128((__m128i *)d, a), _mm_storeu_si128((__m128i *)(d + n - 16), j);
    } else if (n >= 8) {
        uint64_t a = ld64(s), b = ld64(s + n - 8);
        st64(d, a), st64(d + n - 8, b);
    } else if (n >= 4) {
        uint32_t a = ld32(s), b = ld32(s + n - 4);
        st32(d, a), st32(d + n - 4, b);
    } else if (n) {
        uint8_t a = s[0], b = s[n >> 1], c = s[n - 1];
        d[0] = a, d[n >> 1] = b, d[n - 1] = c;
    }
}

// ---- large blocks
#ifdef _WIN32   // pieces, against ucrtbase's early switch to non-temporal stores ("Large blocks" above)
#define PIECE (256u << 10)
#define PIECES_FROM (1u << 20)
#define PIECES_TO (8u << 20)

static inline void preempt_point(Cpu *c) {
    if (c->thread && kern_preempt_pending(c->thread)) kern_checkpoint();
}

static void move_large(Cpu *c, uint8_t *d, const uint8_t *s, size_t n) {
    if (n <= PIECES_FROM || n > PIECES_TO) {
        memmove(d, s, n);
        return;
    }
    if (d <= s || d >= s + n) {   // forward: a piece never overwrites source bytes a later piece reads
        for (size_t o = 0; o < n; o += PIECE) {
            memmove(d + o, s + o, n - o < PIECE ? n - o : PIECE);
            preempt_point(c);
        }
    } else {                      // the destination overlaps the end of the source: from the end
        for (size_t e = n; e;) {
            size_t k = e < PIECE ? e : PIECE;
            e -= k;
            memmove(d + e, s + e, k);
            preempt_point(c);
        }
    }
}
#else
static inline void move_large(Cpu *c, uint8_t *d, const uint8_t *s, size_t n) {
    (void)c;
    memmove(d, s, n);
}
#endif

// ---- the host functions (AAPCS64: x0-x2 in, x0 out; nothing else written: x0 already holds the destination for the
// copies and the fill)
// memcpy and memmove: the same function. The sdk's memmove hands its non-overlapping copies to memcpy, through its PLT
// and so to this function with its guest_read, so memmove keeps the check too.
static void ov_memcpy(Cpu *c) {
    uint64_t dst = c->x[0], src = c->x[1], n = c->x[2];
    uint8_t *d = (uint8_t *)(uintptr_t)dst;
    const uint8_t *s = (const uint8_t *)(uintptr_t)src;
    if (n <= 128) {
        move_small(d, s, n);
    } else {
        if (n >= 1024) gfx_backend()->guest_read(src, n);
        move_large(c, d, s, n);
    }
}
HOST_OVERRIDE("memcpy", ov_memcpy);

static void ov_memset(Cpu *c) {
    uint8_t *d = (uint8_t *)(uintptr_t)c->x[0];
    uint8_t v = (uint8_t)c->x[1];
    size_t n = c->x[2];
    if (n <= 128) {
        set_small(d, v, n);
    } else {
        memset(d, v, n);
    }
}

static void ov_memcmp(Cpu *c) { c->x[0] = libc_memcmp((const uint8_t *)(uintptr_t)c->x[0], (const uint8_t *)(uintptr_t)c->x[1], c->x[2]); }
static void ov_strlen(Cpu *c) { c->x[0] = libc_strlen((const uint8_t *)(uintptr_t)c->x[0]); }
static void ov_strcmp(Cpu *c) { c->x[0] = libc_strcmp((const uint8_t *)(uintptr_t)c->x[0], (const uint8_t *)(uintptr_t)c->x[1]); }
static void ov_strncmp(Cpu *c) { c->x[0] = libc_strncmp((const uint8_t *)(uintptr_t)c->x[0], (const uint8_t *)(uintptr_t)c->x[1], c->x[2]); }

static const struct { const char *name; GuestFn fn; } bound[] = {
    {"memset", ov_memset}, {"memmove", ov_memcpy}, {"memcmp", ov_memcmp},
    {"strlen", ov_strlen}, {"strcmp", ov_strcmp}, {"strncmp", ov_strncmp},
};
#define NBOUND (sizeof bound / sizeof *bound)

static int ab_mode;   // AC_HOST_LIBC_AB

void libc_bind_all(void) {
    for (size_t i = 0; i < NBOUND; i++) hostfn_add_override(bound[i].name, bound[i].fn);
}

void libc_init(void) {
    if (!cfg_bool("AC_HOST_LIBC")) {
        rt_log("libc: memset, memmove, memcmp, strlen, strcmp and strncmp run as the sdk's lifted code (host_libc = 0)");
        return;
    }
    libc_bind_all();
    const char *ab = cfg_get("AC_HOST_LIBC_AB");
    ab_mode = ab && *ab && strcmp(ab, "0");
}

GuestFn libc_host_fn(const char *name) {
    if (!strcmp(name, "memcpy")) return ov_memcpy;
    for (size_t i = 0; i < NBOUND; i++)
        if (!strcmp(bound[i].name, name)) return bound[i].fn;
    return NULL;
}

// ---- in-launch A/B: AC_HOST_LIBC_AB=1 with AC_AB=<seconds> (perf.c rt_ab). The GOT slots hold the host functions in
// the B blocks and the sdk's lifted functions in the A blocks, where every import stub calls them directly again, as
// with host_libc = 0. A slot is one aligned 64-bit word: a thread loading it during the switch sees either value.
#define MAX_SLOTS 8
static int ab_side = -1, ab_n;
static struct { uint64_t slot, host, guest; } ab_slot[NBOUND * MAX_SLOTS];

void libc_ab_frame(void) {
    if (!ab_mode) return;
    if (ab_side < 0) {   // the first present: the slots and both of their values
        for (size_t i = 0; i < NBOUND; i++) {
            uint64_t s[MAX_SLOTS], guest = ldr_symbol(bound[i].name);
            int n = ldr_jump_slots(bound[i].name, s, MAX_SLOTS);
            for (int k = 0; k < n && guest; k++)
                ab_slot[ab_n++] = (typeof(ab_slot[0])){s[k], __atomic_load_n((uint64_t *)(uintptr_t)s[k], __ATOMIC_RELAXED), guest};
        }
        rt_log("libc: in-launch A/B over %d GOT slots: the sdk's functions in A blocks, the host's in B blocks", ab_n);
    }
    int side = rt_ab();
    if (side == ab_side) return;
    ab_side = side;
    for (int i = 0; i < ab_n; i++)
        __atomic_store_n((uint64_t *)(uintptr_t)ab_slot[i].slot, side ? ab_slot[i].host : ab_slot[i].guest, __ATOMIC_RELAXED);
}
