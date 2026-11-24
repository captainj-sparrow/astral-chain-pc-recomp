// Low latency (AC_LOW_LATENCY = off|on|boost): the time from the game reading the pad to its frame reaching the
// display gets shorter when the frame starts as late as it can and still makes its display slot. Three back ends, one
// at most, picked at device creation:
//   - NVIDIA Reflex, VK_NV_low_latency2 with VK_KHR_present_id: the driver sleeps the game before its frame and is told
//     where each frame's stages are (latency markers) and which submissions and present belong to it (present IDs).
//   - AMD Anti-Lag 2, VK_AMD_anti_lag: one vkAntiLagUpdateAMD call per frame before the game reads its input; the
//     driver sleeps in it. Without presentation info: the driver ends a frame at vkQueuePresentKHR, which stays right
//     with the asynchronous replay, where the next frame's input comes before this frame's present (vkd3d-proton made
//     the same choice for the same reason).
//   - built-in, for every other GPU (Intel: XeLL has no Vulkan API; the Steam Deck until its Mesa has the AMD
//     extension): a sleep before the frame of the slot's distance minus the work the last frames took (builtin_sleep).
// Off does nothing: no extension is enabled and every call below returns at once.
//
// The frames (all back ends, and AC_TRACE=latency): a frame starts when the game acquires its window texture
// (nvn_api_obj.c nvnWindowAcquireTexture, after the pacer's wait), which hands it the next ID. The ID follows the
// frame's window image to its present (lowlat_present), its submissions (the game's current frame, or the frame the
// replay or present thread was given with its work: vk_lowlat_submit_frame) and the host present of its blit
// (gfx_present.c flip_locked). Reflex's present IDs must grow on a swapchain: a present without a frame of its own
// takes the next value.
//
// AC_LOW_LATENCY_BACKEND=auto|reflex|antilag|builtin (not in the menu) narrows the choice, for tests.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"

enum { LL_NONE, LL_REFLEX, LL_ANTILAG, LL_BUILTIN };
static const char *const ll_name[] = {"off", "NVIDIA Reflex", "AMD Anti-Lag 2", "built-in"};

static int g_backend;                 // LL_*, set by vk_lowlat_init
static int g_on;                      // a back end runs, or AC_TRACE=latency: the frames are followed

int vk_lowlat_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *v = cfg_value("AC_LOW_LATENCY");
        mode = !strcmp(v, "boost") ? 2 : !strcmp(v, "on") ? 1 : 0;
    }
    return mode;
}

// AC_LOW_LATENCY_BACKEND: may back end b (LL_*) run?
static int backend_allowed(int b) {
    const char *e = cfg_get("AC_LOW_LATENCY_BACKEND");
    if (!e || !*e || !strcmp(e, "auto")) return 1;
    return (b == LL_REFLEX && !strcmp(e, "reflex")) || (b == LL_ANTILAG && !strcmp(e, "antilag")) ||
           (b == LL_BUILTIN && !strcmp(e, "builtin"));
}
int vk_lowlat_wants_reflex(void) { return vk_lowlat_mode() > 0 && backend_allowed(LL_REFLEX); }
int vk_lowlat_wants_antilag(void) { return vk_lowlat_mode() > 0 && backend_allowed(LL_ANTILAG); }

const char *vk_lowlat_backend_name(void) { return ll_name[g_backend]; }
int vk_lowlat_backend(void) { return g_backend; }

// ---- the frames
#define FR_N 128
typedef struct {
    uint64_t id;
    uint64_t start;      // the frame's start: its window texture acquired, after any sleep
    uint64_t slept;      // the low-latency sleep before it
    uint64_t input;      // the first pad read of the frame
    uint64_t sim_end;    // the game's present call
    uint64_t slot;       // the display time the pacer gave it
    uint64_t present;    // the host present of its blit returned
    uint64_t gpu_done;   // its blit completed on the GPU
    uint32_t inputs;     // pad reads during the frame
    uint32_t input_tid;
} FrameRec;
static FrameRec fr[FR_N];
static uint64_t g_next;               // the last ID handed out
static uint64_t g_frame;              // the frame the game builds now
static uint64_t g_img_id[4];          // per window image: the frame that acquired it
static uint64_t g_slot_id, g_slot;    // the last frame given a display slot, and its slot
static uint64_t g_period;             // the pacer's period at that present
static uint32_t g_acquire_tid;
static _Thread_local uint64_t t_present_id;   // lowlat_present -> the back end's present on the same thread
static _Thread_local uint64_t t_submit;       // the frame whose work this thread submits; 0: the game's current frame

static FrameRec *rec_of(uint64_t id) {
    FrameRec *r = &fr[id % FR_N];
    return r->id == id ? r : NULL;
}

uint64_t lowlat_frame(void) { return __atomic_load_n(&g_frame, __ATOMIC_RELAXED); }
uint64_t vk_lowlat_present_id(void) { return g_on ? t_present_id : 0; }
void vk_lowlat_submit_frame(uint64_t id) { t_submit = id; }
static uint64_t submit_id(void) { return t_submit ? t_submit : lowlat_frame(); }

static void sleep_until(uint64_t at) {
    uint64_t now = kern_now_ns();
    if (at > now + 1500000) plat_wait_ns(at - now - 1000000);   // the timer alone wakes up to ~0.5 ms late
    while (kern_now_ns() < at) plat_yield();
}

// ---- Vulkan state (vk_lowlat_init, vk_lowlat_release)
static PFN_vkSetLatencySleepModeNV p_set_mode;
static PFN_vkLatencySleepNV p_sleep;
static PFN_vkSetLatencyMarkerNV p_marker;
static PFN_vkGetLatencyTimingsNV p_timings;
static PFN_vkAntiLagUpdateAMD p_antilag;
static VkSemaphore g_sem;             // Reflex: the timeline semaphore its sleep signals
static uint64_t g_sem_val;

// The swapchain the markers and the sleep mode name. They need no external synchronisation except against the
// swapchain's destruction (VK_NV_low_latency2), so they take sc_lock shared rather than the present lock, which the
// present thread holds through its fence wait and image acquire.
static PlatMutex sc_lock = PLAT_MUTEX_INIT;
static VkSwapchainKHR g_sc;
static int g_mode_set = -1;           // the sleep mode the swapchain has: -1 none yet
static uint32_t g_min_us;
static uint64_t g_last_pid;           // the last present ID presented (present lock): present IDs only go up

static void marker(VkLatencyMarkerNV m, uint64_t id) {
    if (g_backend != LL_REFLEX || !id) return;
    plat_mutex_lock_shared(&sc_lock);
    if (g_sc) {
        VkSetLatencyMarkerInfoNV mi = {.sType = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV, .presentID = id, .marker = m};
        p_marker(vk_ctx()->device, g_sc, &mi);
    }
    plat_mutex_unlock_shared(&sc_lock);
}

// Reflex's sleep mode, set again on every new swapchain (it starts with low latency off) and when the pacer's period
// changes: the driver's frame limit is the pacer's (AC_FPS, a held cutscene at 30), so its sleep and the pacer agree.
static void reflex_mode_apply(void) {
    if (g_backend != LL_REFLEX) return;
    uint64_t period = __atomic_load_n(&g_period, __ATOMIC_RELAXED);
    uint32_t min_us = period ? (uint32_t)(period / 1000) : 0;
    plat_mutex_lock_shared(&sc_lock);
    if (g_sc && (g_mode_set != vk_lowlat_mode() || min_us != g_min_us)) {
        VkLatencySleepModeInfoNV mi = {.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV, .lowLatencyMode = VK_TRUE,
                                       .lowLatencyBoost = vk_lowlat_mode() == 2, .minimumIntervalUs = min_us};
        VkResult r = p_set_mode(vk_ctx()->device, g_sc, &mi);
        if (g_mode_set < 0 || r != VK_SUCCESS)
            rt_log("lowlat: Reflex sleep mode %s, minimum interval %u us: %s", vk_lowlat_mode() == 2 ? "on + boost" : "on",
                   min_us, r == VK_SUCCESS ? "set" : "refused");
        g_mode_set = vk_lowlat_mode();
        g_min_us = min_us;
    }
    plat_mutex_unlock_shared(&sc_lock);
}

// Before the frame (the game's thread, its core given up): the driver's sleep. vkLatencySleepNV needs the swapchain
// synchronised with the acquire and the present, which run under the present lock; the wait stays outside it.
static void reflex_sleep(void) {
    VkCtx *c = vk_ctx();
    VkSwapchainKHR sc = vk_present_lock();
    VkResult r = VK_NOT_READY;
    uint64_t val = 0;
    if (sc) {
        val = ++g_sem_val;
        VkLatencySleepInfoNV si = {.sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV, .signalSemaphore = g_sem, .value = val};
        r = p_sleep(c->device, sc, &si);
    }
    vk_present_unlock();
    if (r != VK_SUCCESS) return;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_sem,
                              .pValues = &val};
    vkWaitSemaphores(c->device, &wi, 100000000ull);   // a driver that never signals costs 100 ms, not the game
}

static void antilag_update(void) {
    uint64_t period = __atomic_load_n(&g_period, __ATOMIC_RELAXED);
    VkAntiLagDataAMD d = {.sType = VK_STRUCTURE_TYPE_ANTI_LAG_DATA_AMD, .mode = VK_ANTI_LAG_MODE_ON_AMD,
                          .maxFPS = period ? (uint32_t)(1e9 / (double)period + 0.5) : 0, .pPresentationInfo = NULL};
    p_antilag(vk_ctx()->device, &d);
}

// ---- built-in: sleep until the frame's slot minus its predicted work minus a margin
// The work of a frame is its start to its blit's completion on the GPU (vk_lowlat_frame_gpu), so it covers the game's
// threads, the replay and the GPU. The prediction follows DXVK's (v3.1.1, src/dxvk/dxvk_latency_builtin.cpp,
// estimateTime): each frame's median with its two neighbours, then the largest of those medians, so a single slow frame
// is ignored and a run of them is not; over the last EST_N frames rather than DXVK's 8. In Harmony Square on the Arc
// 140T at 60 fps, 8 frames and a fixed 1 ms margin cut the time from the pad read to the present from 33.3 to 22.3 ms
// but made 14 frames over 25 ms in 20 s against 1 without the sleep.
//
// The margin adapts to the machine: it starts at 2 ms (1 ms with boost), grows by 2 ms each time a frame that slept
// misses its slot (up to half a period) and shrinks by 0.5 ms after every 120 frames in a row that made theirs. At
// 30 fps a fixed 2 ms margin cut the latency from 66.7 to about 25 ms but turned the replay's spikes, which the 66 ms of
// slack used to absorb, into 6 frames over 50 ms in 30 s against none. No sleep before EST_MIN frames were measured;
// the sleep never exceeds one period.
#define EST_N 32
#define EST_MIN 8
#define MARGIN_STEP 2000000ull
#define MARGIN_EASE 500000ull
#define MARGIN_CLEAN 120
static PlatMutex est_lock = PLAT_MUTEX_INIT;
static uint64_t est_w[EST_N];
static int est_n, est_head;
static uint64_t g_margin;   // est_lock: the built-in sleep's margin now, 0 before the first frame
static int g_clean;         // est_lock: frames in a row that made their slot
static uint64_t g_missed;   // frames whose blit completed after their slot (AC_TRACE=latency)

static uint64_t margin_base(void) { return vk_lowlat_mode() == 2 ? 1000000 : 2000000; }

static uint64_t med3(uint64_t a, uint64_t b, uint64_t c) {
    if (a > b) { uint64_t t = a; a = b; b = t; }
    if (b > c) b = c;
    return a > b ? a : b;
}

static uint64_t est_work(void) {   // est_lock held; est_n >= EST_MIN
    uint64_t best = 0;
    int first = (est_head + EST_N - est_n) % EST_N;   // the oldest sample
    for (int i = 0; i < est_n; i++) {
        uint64_t b = est_w[(first + i) % EST_N];
        uint64_t a = i ? est_w[(first + i - 1) % EST_N] : b, c = i + 1 < est_n ? est_w[(first + i + 1) % EST_N] : b;
        uint64_t m = med3(a, b, c);
        if (m > best) best = m;
    }
    return best;
}

static uint64_t builtin_sleep(uint64_t id) {
    uint64_t slot_id = __atomic_load_n(&g_slot_id, __ATOMIC_ACQUIRE), slot0 = __atomic_load_n(&g_slot, __ATOMIC_RELAXED);
    uint64_t period = __atomic_load_n(&g_period, __ATOMIC_RELAXED);
    if (!slot_id || !period || id <= slot_id) return 0;
    plat_mutex_lock(&est_lock);
    uint64_t work = est_n >= EST_MIN ? est_work() : 0;
    if (!g_margin) g_margin = margin_base();
    uint64_t margin = g_margin;
    plat_mutex_unlock(&est_lock);
    if (!work) return 0;
    uint64_t slot = slot0 + (id - slot_id) * period;   // the display slot this frame should get
    uint64_t now = kern_now_ns();
    if (slot <= now + work + margin) return 0;
    uint64_t until = slot - work - margin;
    if (until > now + period) until = now + period;
    sleep_until(until);
    return kern_now_ns() - now;
}

// The completion thread: frame id's blit is done on the GPU.
static void gpu_done(void *arg) {
    uint64_t id = (uint64_t)(uintptr_t)arg, now = kern_now_ns();
    FrameRec *r = rec_of(id);
    if (!r || !r->start) return;
    r->gpu_done = now;
    int missed = r->slot && now > r->slot + 500000;
    if (missed) __atomic_add_fetch(&g_missed, 1, __ATOMIC_RELAXED);
    if (g_backend != LL_BUILTIN) return;
    uint64_t period = __atomic_load_n(&g_period, __ATOMIC_RELAXED);
    plat_mutex_lock(&est_lock);
    est_w[est_head] = now - r->start;
    est_head = (est_head + 1) % EST_N;
    if (est_n < EST_N) est_n++;
    if (missed && r->slept) {   // the sleep cost this frame its slot: a wider margin
        g_clean = 0;
        g_margin += MARGIN_STEP;
        if (period && g_margin > period / 2) g_margin = period / 2;
    } else if (!missed && ++g_clean >= MARGIN_CLEAN) {
        g_clean = 0;
        g_margin = g_margin > margin_base() + MARGIN_EASE ? g_margin - MARGIN_EASE : margin_base();
    }
    plat_mutex_unlock(&est_lock);
}

void vk_lowlat_frame_gpu(uint64_t id, uint64_t value) {
    if (g_on && id && value) vk_completion_add(value, gpu_done, (void *)(uintptr_t)id);
}

// ---- AC_TRACE=latency: every LOG_EVERY frames, where a frame's time went
#define LOG_EVERY 600
static struct {
    uint64_t n, n_input, inputs, input_ns, work_ns, gpu_ns, wait_ns, slept_ns, ahead;
    float lat[LOG_EVERY];
    int said_threads;
} T;

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static void reflex_report(char *out, size_t n) {
    out[0] = 0;
    if (g_backend != LL_REFLEX || !p_timings) return;
    static VkLatencyTimingsFrameReportNV rep[64];
    VkSwapchainKHR sc = vk_present_lock();
    VkGetLatencyMarkerInfoNV gi = {.sType = VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV};
    if (sc) {
        p_timings(vk_ctx()->device, sc, &gi);   // the count first
        if (gi.timingCount > 64) gi.timingCount = 64;
        for (uint32_t i = 0; i < gi.timingCount; i++)
            rep[i] = (VkLatencyTimingsFrameReportNV){.sType = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV};
        gi.pTimings = rep;
        if (gi.timingCount) p_timings(vk_ctx()->device, sc, &gi);
    }
    vk_present_unlock();
    uint32_t full = 0;
    double sum = 0;
    for (uint32_t i = 0; i < gi.timingCount; i++) {
        const VkLatencyTimingsFrameReportNV *r = &rep[i];
        if (r->inputSampleTimeUs && r->simStartTimeUs && r->presentStartTimeUs && r->gpuRenderEndTimeUs > r->inputSampleTimeUs)
            full++, sum += (double)(r->gpuRenderEndTimeUs - r->inputSampleTimeUs);
    }
    snprintf(out, n, "; driver: %u of %u frames with every marker, input to GPU end %.2f ms", full, gi.timingCount,
             full ? sum / full / 1000.0 : 0.0);
}

// The game's thread at a frame's start: frame id, which ended long enough ago that every stamp is in.
static void trace_frame(uint64_t id) {
    FrameRec *r = rec_of(id);
    if (!r || !r->start || !r->sim_end || !r->present || !r->gpu_done) return;
    if (!T.said_threads && r->input_tid) {
        T.said_threads = 1;
        rt_log("lowlat: the pad is read on thread %u, the window texture acquired on thread %u", r->input_tid,
               g_acquire_tid);
    }
    if (r->input) {
        T.lat[T.n_input++] = (float)((double)(r->present - r->input) / 1e6);
        T.input_ns += r->input - r->start;
    }
    T.inputs += r->inputs;
    T.work_ns += r->sim_end - r->start;
    T.gpu_ns += r->gpu_done > r->sim_end ? r->gpu_done - r->sim_end : 0;
    T.wait_ns += r->present > r->gpu_done ? r->present - r->gpu_done : 0;
    T.slept_ns += r->slept;
    T.ahead += r->slot > r->start ? r->slot - r->start : 0;
    if (++T.n < LOG_EVERY) return;
    double n = (double)T.n, avg = 0, p99 = 0;
    if (T.n_input) {
        for (uint64_t i = 0; i < T.n_input; i++) avg += T.lat[i];
        avg /= (double)T.n_input;
        qsort(T.lat, T.n_input, sizeof *T.lat, cmp_float);
        p99 = T.lat[(T.n_input * 99) / 100 < T.n_input ? (T.n_input * 99) / 100 : T.n_input - 1];
    }
    char drv[160];
    reflex_report(drv, sizeof drv);
    rt_log("lowlat: %s, %llu frames: first pad read to host present %.2f ms (p99 %.2f) over %llu frames, %.2f reads a "
           "frame, first read %.2f ms after the start; start to game present %.2f ms, to GPU done %.2f ms more, then "
           "%.2f ms to the host present; start to slot %.2f ms; slept %.2f ms a frame (margin %.1f ms); %llu frames done "
           "on the GPU after their slot%s",
           vk_lowlat_mode() ? ll_name[g_backend] : "off", (unsigned long long)T.n, avg, p99,
           (unsigned long long)T.n_input, T.inputs / n, T.n_input ? T.input_ns / 1e6 / (double)T.n_input : 0.0,
           T.work_ns / 1e6 / n, T.gpu_ns / 1e6 / n, T.wait_ns / 1e6 / n, T.ahead / 1e6 / n, T.slept_ns / 1e6 / n,
           __atomic_load_n(&g_margin, __ATOMIC_RELAXED) / 1e6, (unsigned long long)__atomic_exchange_n(&g_missed, 0, __ATOMIC_RELAXED), drv);
    int said = T.said_threads;
    memset(&T, 0, sizeof T);
    T.said_threads = said;
}

// ---- the front end's calls (gfx.h)
void lowlat_acquire(int image) {
    if (!g_on) return;
    uint64_t slept = 0, t0 = perf_begin(), id = __atomic_load_n(&g_next, __ATOMIC_RELAXED) + 1;
    switch (g_backend) {
    case LL_REFLEX: reflex_sleep(); break;
    case LL_ANTILAG: antilag_update(); break;
    case LL_BUILTIN: builtin_sleep(id); break;
    }
    if (g_backend != LL_NONE) perf_end(PERF_PACE, t0);   // a pacing wait in the perf lines, as the pacer's own
    uint64_t now = kern_now_ns();
    if (g_backend != LL_NONE) slept = now - t0;
    __atomic_store_n(&g_next, id, __ATOMIC_RELAXED);
    __atomic_store_n(&g_frame, id, __ATOMIC_RELAXED);
    g_img_id[image & 3] = id;
    g_acquire_tid = plat_thread_id();
    FrameRec *r = &fr[id % FR_N];
    *r = (FrameRec){.id = id, .start = now, .slept = slept};
    marker(VK_LATENCY_MARKER_SIMULATION_START_NV, id);
    if (rt_trace & RT_TRACE_LATENCY && id > 16) trace_frame(id - 16);
}

void lowlat_input(void) {
    if (!g_on) return;
    uint64_t id = lowlat_frame();
    FrameRec *r = id ? rec_of(id) : NULL;
    if (!r) return;
    if (!r->inputs++) {
        marker(VK_LATENCY_MARKER_INPUT_SAMPLE_NV, id);
        r->input = kern_now_ns();
        r->input_tid = plat_thread_id();
    }
}

// A pause's repeats took present IDs of their own (vk_lowlat_flip_begin). The frames' IDs go on above them, or every
// later frame's present would carry another ID than its markers. The margin covers the repeats still on their way to
// the present.
#define RESUME_ID_MARGIN 8
void lowlat_resume(void) {
    if (g_backend != LL_REFLEX) return;
    uint64_t id = __atomic_load_n(&g_last_pid, __ATOMIC_RELAXED) + RESUME_ID_MARGIN;
    if (id > __atomic_load_n(&g_next, __ATOMIC_RELAXED)) __atomic_store_n(&g_next, id, __ATOMIC_RELAXED);
}

void lowlat_present(int image, uint64_t slot, uint64_t period) {
    if (!g_on) return;
    uint64_t id = g_img_id[image & 3];
    t_present_id = id;
    __atomic_store_n(&g_period, period, __ATOMIC_RELAXED);
    FrameRec *r = id ? rec_of(id) : NULL;
    if (!r) return;
    r->sim_end = kern_now_ns();
    r->slot = slot;
    __atomic_store_n(&g_slot, slot, __ATOMIC_RELAXED);
    __atomic_store_n(&g_slot_id, id, __ATOMIC_RELEASE);
    marker(VK_LATENCY_MARKER_SIMULATION_END_NV, id);
}

// ---- the back end's calls (vk.h)
// vk_sync.c, before a submission: Reflex's frame ID for it, and RENDERSUBMIT_START at the frame's first recorded
// submission (a later one would move the start of its render submission past the work before it).
const void *vk_lowlat_submit_info(VkLatencySubmissionPresentIdNV *s, int recorded) {
    if (g_backend != LL_REFLEX) return NULL;
    static uint64_t started;   // the last frame whose RENDERSUBMIT_START was sent
    *s = (VkLatencySubmissionPresentIdNV){.sType = VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV,
                                          .presentID = submit_id()};
    if (recorded && s->presentID > __atomic_load_n(&started, __ATOMIC_RELAXED)) {
        __atomic_store_n(&started, s->presentID, __ATOMIC_RELAXED);
        marker(VK_LATENCY_MARKER_RENDERSUBMIT_START_NV, s->presentID);
    }
    return s->presentID ? s : NULL;
}

// After it: RENDERSUBMIT_END, at every recorded submission; the frame's last one counts.
void vk_lowlat_submit_done(const VkLatencySubmissionPresentIdNV *s, int recorded) {
    if (g_backend == LL_REFLEX && recorded && s->presentID) marker(VK_LATENCY_MARKER_RENDERSUBMIT_END_NV, s->presentID);
}

// gfx_present.c create_swapchain (present lock held): the new swapchain, before the old one goes away.
void vk_lowlat_swapchain_info(VkSwapchainLatencyCreateInfoNV *li, VkSwapchainCreateInfoKHR *ci) {
    if (g_backend != LL_REFLEX) return;
    *li = (VkSwapchainLatencyCreateInfoNV){.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV,
                                           .pNext = ci->pNext, .latencyModeEnable = VK_TRUE};
    ci->pNext = li;
}

void vk_lowlat_swapchain(VkSwapchainKHR sc) {
    plat_mutex_lock(&sc_lock);
    g_sc = sc;
    g_mode_set = -1;
    plat_mutex_unlock(&sc_lock);
    reflex_mode_apply();
}

// flip_locked (present lock held), around vkQueuePresentKHR. id: the frame of the blit presented, 0 for none.
const void *vk_lowlat_flip_begin(uint64_t id, VkPresentIdKHR *pi, uint64_t *pid) {
    *pid = 0;
    if (g_backend != LL_REFLEX) return NULL;
    reflex_mode_apply();   // the pacer's period may have changed (a held cutscene)
    *pid = id > g_last_pid ? id : g_last_pid + 1;
    __atomic_store_n(&g_last_pid, *pid, __ATOMIC_RELAXED);
    *pi = (VkPresentIdKHR){.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR, .swapchainCount = 1, .pPresentIds = pid};
    marker(VK_LATENCY_MARKER_PRESENT_START_NV, *pid);
    return pi;
}

void vk_lowlat_flip_end(uint64_t id, uint64_t pid) {
    if (!g_on) return;
    if (pid) marker(VK_LATENCY_MARKER_PRESENT_END_NV, pid);
    FrameRec *r = id ? rec_of(id) : NULL;
    if (r) r->present = kern_now_ns();
}

// vk_device.c, after the device was made: the back end, its entry points and Reflex's semaphore.
void vk_lowlat_init(VkCtx *c) {
    int mode = vk_lowlat_mode();
    g_backend = LL_NONE;
    if (mode && c->has_nv_low_latency2) {
        p_set_mode = (PFN_vkSetLatencySleepModeNV)vkGetDeviceProcAddr(c->device, "vkSetLatencySleepModeNV");
        p_sleep = (PFN_vkLatencySleepNV)vkGetDeviceProcAddr(c->device, "vkLatencySleepNV");
        p_marker = (PFN_vkSetLatencyMarkerNV)vkGetDeviceProcAddr(c->device, "vkSetLatencyMarkerNV");
        p_timings = (PFN_vkGetLatencyTimingsNV)vkGetDeviceProcAddr(c->device, "vkGetLatencyTimingsNV");
        VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
        if (p_set_mode && p_sleep && p_marker && vkCreateSemaphore(c->device, &si, NULL, &g_sem) == VK_SUCCESS)
            g_backend = LL_REFLEX;
    }
    if (mode && g_backend == LL_NONE && c->has_amd_anti_lag) {
        p_antilag = (PFN_vkAntiLagUpdateAMD)vkGetDeviceProcAddr(c->device, "vkAntiLagUpdateAMD");
        if (p_antilag) g_backend = LL_ANTILAG;
    }
    if (mode && g_backend == LL_NONE && backend_allowed(LL_BUILTIN)) g_backend = LL_BUILTIN;
    g_on = g_backend != LL_NONE || (rt_trace & RT_TRACE_LATENCY);
    const char *forced = cfg_get("AC_LOW_LATENCY_BACKEND");
    rt_log("lowlat: %s (AC_LOW_LATENCY=%s%s%s; VK_NV_low_latency2 %s, VK_AMD_anti_lag %s)", ll_name[g_backend],
           cfg_value("AC_LOW_LATENCY"), forced ? ", AC_LOW_LATENCY_BACKEND=" : "", forced ? forced : "",
           c->has_nv_low_latency2 ? "on" : "off", c->has_amd_anti_lag ? "on" : "off");
}

void vk_lowlat_release(VkCtx *c) {
    if (g_sem) vkDestroySemaphore(c->device, g_sem, NULL);
    g_sem = VK_NULL_HANDLE;
    g_backend = LL_NONE;
    g_on = 0;
    p_set_mode = NULL, p_sleep = NULL, p_marker = NULL, p_timings = NULL, p_antilag = NULL;
    plat_mutex_lock(&sc_lock);
    g_sc = VK_NULL_HANDLE;
    plat_mutex_unlock(&sc_lock);
}

// --probe-latency (host.h): the device with AC_LOW_LATENCY=on, then 120 frames of a synthetic game at 60 Hz through
// the front end's calls: the pacer's wait (the window image is free one period after its display), 4 ms of work, the
// pacer's slot (nvn_api_obj.c pace) and an empty submission as the frame's GPU work. Without a window, so Reflex's
// sleep and markers return at once (they need a swapchain) and the probe checks that its calls are harmless; the
// built-in sleep is checked for what it is for: the frames start less than a period before their slot (without it,
// two periods), and none misses it. AC_VK_VALIDATION=1 counts validation errors.
int probe_latency(void) {
    cfg_set("AC_LOW_LATENCY", "on");
    if (!vk_init()) {
        printf("probe-latency: FAILED (no Vulkan device)\n");
        return 1;
    }
    VkCtx *c = vk_ctx();
    printf("probe-latency: back end %s (VK_NV_low_latency2 %d, VK_AMD_anti_lag %d)\n", ll_name[g_backend],
           c->has_nv_low_latency2, c->has_amd_anti_lag);
    enum { N = 120, FROM = 60 };
    const uint64_t period = 16666667, work = 4000000;
    static uint64_t slots[N], ids[N];
    uint64_t target = 0;
    for (int i = 0; i < N; i++) {
        if (i >= 2) sleep_until(slots[i - 2] + period);
        lowlat_acquire(i % 3);
        ids[i] = lowlat_frame();
        lowlat_input();
        sleep_until(kern_now_ns() + work);
        uint64_t now = kern_now_ns(), next = target + period;
        target = !target ? now : now <= next ? next : now - next < period ? next : now;
        slots[i] = target;
        lowlat_present(i % 3, slots[i], period);
        vk_lowlat_frame_gpu(ids[i], vk_sync_submit(NULL, 0));
    }
    vk_sync_finish();
    double ahead = 0;
    int missed = 0, n = 0;
    for (int i = FROM; i < N; i++) {
        FrameRec *r = rec_of(ids[i]);
        if (!r || !r->start) continue;
        n++;
        ahead += (double)(slots[i] - r->start) / 1e6;
        if (r->gpu_done > slots[i] + 500000) missed++;
    }
    ahead = n ? ahead / n : 0;
    int builtin = g_backend == LL_BUILTIN;
    const char *name = ll_name[g_backend];   // vk_shutdown resets it
    printf("probe-latency: frames %d-%d start %.2f ms before their slot (period %.2f ms), %d missed it\n", FROM, N - 1,
           ahead, period / 1e6, missed);
    vk_shutdown();
    long errors = vk_validation_errors_total();
    const char *why = n < N - FROM - 2                     ? "frames were not followed"
                    : missed > 2                           ? "frames missed their slot"
                    : builtin && ahead >= period / 1e6     ? "the built-in sleep did not shorten the wait"
                    : errors                               ? "validation errors"
                                                           : NULL;
    if (why) {
        printf("probe-latency: FAILED (%s; %ld validation errors)\n", why, errors);
        return 1;
    }
    printf("probe-latency: passed (%s, %ld validation errors)\n", name, errors);
    return 0;
}
