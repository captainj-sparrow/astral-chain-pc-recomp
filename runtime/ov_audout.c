// The audio output service ("audout:u"): the legacy nn::audio output path Astral Chain plays all of its sound through.
//
// nn::audio keeps a small set of AudioOut sessions, each fed with AudioOutBuffers of signed 16-bit little-endian
// interleaved PCM at 48 kHz. The game appends a buffer, waits on the buffer event the SDK registered from
// RegisterBufferEvent, takes the tags of the buffers the device has finished with from GetReleasedAudioOutBuffers, and
// refills and appends them again. Nothing here blocks: an append converts the guest PCM into the host ring and returns,
// and the SDL audio thread drains that ring without taking a guest lock. A buffer counts as played once its samples have
// left the ring, which is what paces the game's audio thread.
//
// audren:u (runtime/audren.c) is the other renderer path and has its own mixer; this game never opens a renderer, only
// the IAudioDevice service for the master volume (nn::audio::SetAudioDeviceOutputVolume). Both open SDL's audio
// subsystem, which SDL reference-counts.
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "audren.h"
#include "cfg.h"
#include "host.h"
#include "ipc.h"
#include "ov.h"
#include "platform/platform.h"

#define AUDOUT_RATE 48000
#define AUDOUT_CHANNELS 2
#define AUDOUT_MAX_SESSIONS 8
#define AUDOUT_MAX_BUFFERS 32                 // the SDK's own queue limit (AudioCore::BufferCount)
#define AUDOUT_RING_FRAMES 32768              // stereo frames of output queued ahead, about 680 ms
#define AUDOUT_MAX_FRAME 1024                 // frames converted per callback step
#define AUDOUT_DRY_MAX (AUDOUT_RATE / 2)      // frames of one dry stretch that count as sound coming late (0.5 s)
#define AUDOUT_SAMPLE_FORMAT_PCM16 1u
#define AUDOUT_STATE_STARTED 0u
#define AUDOUT_STATE_STOPPED 1u
#define AUDOUT_NAME_BYTES 0x100
#define AUDOUT_R_INVALID_SAMPLE_RATE 0x7C1u   // nn::audio::ResultInvalidSampleRate
#define AUDOUT_R_INVALID_CHANNEL_COUNT 0x7C2u // nn::audio::ResultInvalidChannelCount
#define AUDOUT_R_OUT_OF_RESOURCE 0xC401u

// ---- output device: one ring and one SDL stream for every AudioOut session
//
// played and queued are cumulative frame counters; the ring holds the frames in [played, queued), indexed modulo
// AUDOUT_RING_FRAMES. Both start at 1 so a queued buffer's end marker is never 0 (0 means "no buffer"). Guest threads
// append under push (one producer at a time); the SDL thread only moves played.
static struct {
    SDL_AudioStream *stream;
    int device_ok, opened;
    float ring[AUDOUT_RING_FRAMES * 2];
    _Atomic uint64_t played, queued;
    PlatMutex push;
    volatile int64_t overruns;
    volatile int64_t appends, releases, zero_releases;
    // The ring ran dry while the game was feeding it: the frames of silence the device got instead, and how many
    // times it happened (audout_pull). fed: the game has queued sound since the last stop or flush.
    _Atomic uint64_t dry_frames, dry_gaps;
    _Atomic int fed;
    uint64_t dry_logged, dry_gaps_logged, dry_log_ns;   // what the last log line reported, and when (out_dry_report)
    double level_sum, level_peak;
    uint64_t level_count;
} out = {.push = PLAT_MUTEX_INIT, .played = 1, .queued = 1};

typedef struct {
    uint64_t tag, end;   // end: the cumulative frame counter at which this buffer has been played out
} AudoutBuffer;

typedef struct AudoutOut {
    volatile int32_t refs;         // the IPC session plus any SDL-thread use
    uint32_t state;
    float volume;
    int rate, channels;
    uint32_t event;                // writable kernel handle behind the readable one the client waits on
    PlatMutex lock;                // the buffer queue, taken by guest threads only
    AudoutBuffer q[AUDOUT_MAX_BUFFERS];
    int head, count;
    _Atomic uint64_t oldest_end;   // q[head].end, 0 when the queue is empty (read by the SDL thread)
    _Atomic uint64_t signalled;    // the last oldest_end the event was signalled for
    uint64_t played_base;          // out.played when the session started, for GetAudioOutPlayedSampleCount
    int logged;
} AudoutOut;

// Sessions the SDL thread may look at. It only ever try-locks the list, so it can never wait for a guest thread (which
// may hold the kernel lock, and kern_event_signal takes that).
static AudoutOut *g_sessions[AUDOUT_MAX_SESSIONS];
static PlatMutex g_sessions_lock = PLAT_MUTEX_INIT;
static volatile int32_t g_sessions_live;

static void session_unref(AudoutOut *o) {
    if (plat_dec32(&o->refs)) return;
    kern_handle_close(o->event);
    free(o);
}

// AC_VOLUME: the player's volume in percent (0..100) on top of the game's own. Re-read when the configuration changes,
// so the menu's slider applies while the game runs. Guest threads only.
static float host_volume(void) {
    static float v = 1.0f;
    static unsigned seen = ~0u;
    unsigned ver = cfg_version();
    if (ver == seen) return v;
    seen = ver;
    double pct = cfg_double("AC_VOLUME");
    v = (float)((pct < 0 ? 0 : pct > 100 ? 100 : pct) / 100);
    return v;
}

// Signals the buffer event of every session whose oldest queued buffer has been played out. Runs on the SDL audio
// thread: it collects the sessions under a try-lock and signals only after dropping it, so it never holds a lock a guest
// thread waits for while taking the kernel lock.
static void notify_released(void) {
    AudoutOut *use[AUDOUT_MAX_SESSIONS];
    int n = 0;
    uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
    if (plat_mutex_trylock_shared(&g_sessions_lock)) {
        for (int i = 0; i < AUDOUT_MAX_SESSIONS; i++) {
            AudoutOut *o = g_sessions[i];
            if (!o) continue;
            uint64_t end = atomic_load_explicit(&o->oldest_end, memory_order_acquire);
            if (!end || played < end) continue;
            if (atomic_load_explicit(&o->signalled, memory_order_relaxed) == end) continue;
            atomic_store_explicit(&o->signalled, end, memory_order_relaxed);
            plat_inc32(&o->refs);
            use[n++] = o;
        }
        plat_mutex_unlock_shared(&g_sessions_lock);
    }
    for (int i = 0; i < n; i++) {
        kern_event_signal(use[i]->event);
        session_unref(use[i]);
    }
}

// SDL's audio thread: hands stereo float to the device, silence when the ring runs dry. A dry ring while the game feeds
// it means the game's sound thread did not refill in time: the device plays silence, and everything the game streams
// (a voice, a movie's sound) comes that much later, for good, since the game queues a buffer only when one comes back.
// Those frames are counted for the log (out_dry_report) and move the dry clock on (audio_mixer.c
// audio_device_dry_ns), by which a cutscene's picture waits for its sound (frame_rate.c cutscene_lost); every request
// moves the device clock on. Only the first AUDOUT_DRY_MAX frames of one dry stretch move the dry clock: a longer one
// is not a late sound thread but a game that stopped feeding the output, and no picture has to wait for that.
// AC_TEST_FREEZE (frame_rate.c frame_test_frozen): while a test freeze lasts the device gets silence, the ring is
// left alone and neither clock moves, as if the process stood still.
static void SDLCALL audout_pull(void *user, SDL_AudioStream *stream, int additional, int total) {
    (void)user;
    (void)total;
    float tmp[2 * AUDOUT_MAX_FRAME];
    static uint64_t dry;   // frames of silence in the dry stretch the last request was in, 0 none (SDL thread only)
    uint64_t played = atomic_load_explicit(&out.played, memory_order_relaxed);
    while (additional > 0) {
        int frames = additional / (int)(2 * sizeof(float));
        if (frames > AUDOUT_MAX_FRAME) frames = AUDOUT_MAX_FRAME;
        if (frames <= 0) break;
        uint64_t queued = atomic_load_explicit(&out.queued, memory_order_acquire);
        uint64_t have = queued > played ? queued - played : 0;
        int frozen = frame_test_frozen();
        if (frozen) have = 0;
        int n = have < (uint64_t)frames ? (int)have : frames;
        uint64_t late = 0;   // frames of this request that count as the game's sound coming late
        if (n < frames && !frozen && atomic_load_explicit(&out.fed, memory_order_relaxed)) {
            uint64_t silent = (uint64_t)(frames - n);
            atomic_fetch_add_explicit(&out.dry_frames, silent, memory_order_relaxed);
            if (!dry) atomic_fetch_add_explicit(&out.dry_gaps, 1, memory_order_relaxed);
            late = dry >= AUDOUT_DRY_MAX ? 0 : silent < AUDOUT_DRY_MAX - dry ? silent : AUDOUT_DRY_MAX - dry;
            dry += silent;
        } else if (n == frames) {
            dry = 0;
        }
        if (!frozen) audio_device_clock_add((uint64_t)frames, late, AUDOUT_RATE);
        for (int i = 0; i < n; i++) {
            uint64_t k = (played + (uint64_t)i) % AUDOUT_RING_FRAMES;
            tmp[2 * i] = out.ring[2 * k];
            tmp[2 * i + 1] = out.ring[2 * k + 1];
        }
        if (n < frames) memset(tmp + 2 * n, 0, (size_t)(frames - n) * 2 * sizeof(float));
        played += (uint64_t)n;
        atomic_store_explicit(&out.played, played, memory_order_release);
        if (!SDL_PutAudioStreamData(stream, tmp, (int)((size_t)frames * 2 * sizeof(float)))) break;
        additional -= frames * (int)(2 * sizeof(float));
        notify_released();
    }
}

// Opens the output once. Without a device the service still answers and every buffer is reported played at once, so the
// game's audio thread keeps recycling instead of stalling (the same choice audio_mixer.c makes).
static void out_open(void) {
    if (out.opened) return;
    out.opened = 1;
    const char *mode = cfg_get("AC_AUDIO");
    if ((mode && !strcmp(mode, "none")) || cfg_get("AC_AUDIO_OFF")) {
        rt_log("audout: AC_AUDIO=none: buffers are recycled without a device");
        return;
    }
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        rt_log("audout: SDL audio init failed (%s): buffers are recycled without sound", SDL_GetError());
        return;
    }
    SDL_AudioSpec spec = {SDL_AUDIO_F32, AUDOUT_CHANNELS, AUDOUT_RATE};
    out.stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, audout_pull, NULL);
    if (!out.stream) {
        rt_log("audout: no output device (%s): buffers are recycled without sound", SDL_GetError());
        return;
    }
    if (!SDL_ResumeAudioStreamDevice(out.stream)) {
        rt_log("audout: starting the output failed (%s): buffers are recycled without sound", SDL_GetError());
        SDL_DestroyAudioStream(out.stream);
        out.stream = NULL;
        return;
    }
    out.device_ok = 1;
    SDL_AudioSpec dev;
    int period = 0;
    SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(out.stream), &dev, &period);
    if (period > 0 && dev.freq > 0) audio_device_set_period((uint64_t)period, (uint32_t)dev.freq);
    const char *name = SDL_GetAudioDeviceName(SDL_GetAudioStreamDevice(out.stream));
    rt_log("audout: SDL device '%s' opened, %d Hz stereo signed 16-bit in, float out, device period %d frames",
           name ? name : "?", AUDOUT_RATE, period);
}

// Converts and queues guest PCM (signed 16-bit interleaved) and returns the frame counter at which it has been played.
// Frames past the ring's capacity are dropped; the buffer still comes back to the game.
static uint64_t out_push(const int16_t *pcm, uint64_t frames, float volume) {
    plat_mutex_lock(&out.push);
    uint64_t queued = atomic_load_explicit(&out.queued, memory_order_relaxed);
    uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
    uint64_t have = queued > played ? queued - played : 0;
    uint64_t room = have < AUDOUT_RING_FRAMES ? AUDOUT_RING_FRAMES - have : 0;
    uint64_t n = frames < room ? frames : room;
    if (!out.device_ok) n = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t k = (queued + i) % AUDOUT_RING_FRAMES;
        float l = (float)pcm[2 * i] * (1.0f / 32768.0f) * volume;
        float r = (float)pcm[2 * i + 1] * (1.0f / 32768.0f) * volume;
        out.ring[2 * k] = l;
        out.ring[2 * k + 1] = r;
        out.level_sum += (double)l * l + (double)r * r;
        if (fabsf(l) > out.level_peak) out.level_peak = fabsf(l);
        if (fabsf(r) > out.level_peak) out.level_peak = fabsf(r);
    }
    out.level_count += 2 * n;
    queued += n;
    atomic_store_explicit(&out.queued, queued, memory_order_release);
    if (n) atomic_store_explicit(&out.fed, 1, memory_order_relaxed);
    if (n < frames && out.device_ok && plat_xadd64(&out.overruns, 1) == 0)
        rt_log("audout: the output ring is full (%d frames queued): dropping frames of a buffer", AUDOUT_RING_FRAMES);
    if (!out.device_ok)   // no device: everything is already played, so the game recycles at its own pace
        atomic_store_explicit(&out.played, queued, memory_order_release);
    plat_mutex_unlock(&out.push);
    return queued;
}

// ---- one AudioOut session
static void queue_refresh(AudoutOut *o) {
    atomic_store_explicit(&o->oldest_end, o->count > 0 ? o->q[o->head].end : 0, memory_order_release);
}

// Marks every queued buffer as played without playing it: the tags still go back to the game through
// GetReleasedAudioOutBuffers, as the hardware's flush does.
static int queue_release_all(AudoutOut *o) {
    if (!o->count) return 0;
    uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
    for (int i = 0; i < o->count; i++) {
        int at = (o->head + i) % AUDOUT_MAX_BUFFERS;
        if (o->q[at].end > played) o->q[at].end = played;
    }
    queue_refresh(o);
    return 1;
}

static void out_start(AudoutOut *o) {
    o->state = AUDOUT_STATE_STARTED;
    o->played_base = atomic_load_explicit(&out.played, memory_order_acquire);
    rt_log("audout: session started (%d Hz, %d channels, %d buffers queued)", o->rate, o->channels, o->count);
}

static void out_stop(AudoutOut *o) {
    o->state = AUDOUT_STATE_STOPPED;
    plat_mutex_lock(&o->lock);
    int dropped = queue_release_all(o);
    plat_mutex_unlock(&o->lock);
    atomic_store_explicit(&out.fed, 0, memory_order_relaxed);   // an empty ring is no gap from here on
    if (dropped) {
        atomic_store_explicit(&out.played, atomic_load_explicit(&out.queued, memory_order_acquire), memory_order_release);
        kern_event_signal(o->event);
    }
    rt_log("audout: session stopped");
}

static int out_contains(AudoutOut *o, uint64_t tag) {
    plat_mutex_lock(&o->lock);
    int found = 0;
    for (int i = 0; i < o->count && !found; i++)
        if (o->q[(o->head + i) % AUDOUT_MAX_BUFFERS].tag == tag) found = 1;
    plat_mutex_unlock(&o->lock);
    return found;
}

static uint32_t out_flush(AudoutOut *o) {
    if (o->state != AUDOUT_STATE_STARTED) return 0;
    plat_mutex_lock(&o->lock);
    int dropped = queue_release_all(o);
    plat_mutex_unlock(&o->lock);
    atomic_store_explicit(&out.fed, 0, memory_order_relaxed);
    if (dropped) {
        atomic_store_explicit(&out.played, atomic_load_explicit(&out.queued, memory_order_acquire), memory_order_release);
        kern_event_signal(o->event);
    }
    return dropped ? 1 : 0;
}

static const IpcBuffer *first_buffer(const IpcRequest *rq, int *is_static) {
    if (is_static) *is_static = 0;
    if (rq->na && rq->a[0].size) return &rq->a[0];
    if (rq->nb && rq->b[0].size) return &rq->b[0];
    if (rq->nw && rq->w[0].size) return &rq->w[0];
    if (rq->nx && rq->x[0].size) {   // nn::sf's AutoSelect sends a small read-only array as a static buffer
        if (is_static) *is_static = 1;
        return &rq->x[0];
    }
    if (rq->nc && rq->c[0].size) return &rq->c[0];
    return NULL;
}

// One log line when the ring ran dry (audout_pull counts, on SDL's thread; this runs on the game's sound thread at its
// next append), at most one per 5 s: a log with these lines explains sound that lags the picture.
static void out_dry_report(void) {
    uint64_t dry = atomic_load_explicit(&out.dry_frames, memory_order_relaxed);
    if (dry == out.dry_logged) return;
    uint64_t now = kern_now_ns();
    if (out.dry_log_ns && now - out.dry_log_ns < 5000000000ull) return;
    uint64_t gaps = atomic_load_explicit(&out.dry_gaps, memory_order_relaxed);
    rt_log("audout: the output ran dry %llu times since the last report (%.0f ms of silence; %.0f ms in %llu gaps "
           "since the start): the game's sound thread refilled late, and streamed sound is that much later",
           (unsigned long long)(gaps - out.dry_gaps_logged), (double)(dry - out.dry_logged) * 1000.0 / AUDOUT_RATE,
           (double)dry * 1000.0 / AUDOUT_RATE, (unsigned long long)gaps);
    out.dry_logged = dry, out.dry_gaps_logged = gaps, out.dry_log_ns = now;
}

// AC_TEST_AUDIO_STALL=<ms>[:<every ms>] (testing): the game's sound thread sleeps <ms> before an append, once per
// <every ms> (5000 without it), so the output runs dry as it does when that thread gets no processor time.
static void test_audio_stall(void) {
    static int parsed;
    static unsigned ms, every = 5000;
    static uint64_t last_ns;
    if (!parsed) {
        parsed = 1;
        const char *e = cfg_get("AC_TEST_AUDIO_STALL");
        if (e && sscanf(e, "%u:%u", &ms, &every) >= 1 && ms)
            rt_log("audout: test stall of the sound thread, %u ms every %u ms", ms, every);
    }
    if (!ms) return;
    uint64_t now = kern_now_ns();
    if (!last_ns) last_ns = now;
    if (now - last_ns < every * 1000000ull) return;
    last_ns = now;
    plat_sleep_ns(ms * 1000000ull);
}

// AppendAudioOutBuffer: the AudioOutBuffer record (0x28 bytes) in a buffer, its tag in the raw data.
static void out_append(AudoutOut *o, const IpcRequest *rq, IpcResponse *rs) {
    test_audio_stall();
    out_dry_report();
    const IpcBuffer *b = first_buffer(rq, NULL);
    if (!b || b->size < 0x28) {
        rt_log("audout: AppendAudioOutBuffer cmd %u with %llu bytes, expected 0x28 (x%u a%u b%u w%u c%u data %u)",
               rq->cmd, (unsigned long long)(b ? b->size : 0), rq->nx, rq->na, rq->nb, rq->nw, rq->nc, rq->data_size);
        rs->result = IPC_R_BAD_HANDLE;
        return;
    }
    const uint8_t *p = ipc_guest(b->addr, b->size);
    uint64_t samples, capacity, size, offset;
    memcpy(&samples, p + 0x08, 8);
    memcpy(&capacity, p + 0x10, 8);
    memcpy(&size, p + 0x18, 8);
    memcpy(&offset, p + 0x20, 8);
    uint64_t tag = rq->data_size >= 8 ? IPC_ARG(rq, uint64_t, 0) : b->addr;

    // data_offset is where the payload starts inside the buffer; the SDK leaves it zeroed, and a value that does not fit
    // the capacity is treated as zero rather than read out of bounds.
    if (offset >= capacity || offset + size > capacity) offset = 0;
    uint64_t frames = 0;
    const int16_t *pcm = NULL;
    if (size) {
        if (offset + size > (1u << 30) || !plat_mem_readable(samples + offset, size)) {
            rt_log("audout: AppendAudioOutBuffer tag %#llx: %llu bytes at %#llx are not mapped", (unsigned long long)tag,
                   (unsigned long long)size, (unsigned long long)(samples + offset));
            rs->result = IPC_R_BAD_HANDLE;
            return;
        }
        frames = size / (uint64_t)(o->channels * 2);
        pcm = (const int16_t *)(uintptr_t)(samples + offset);
    }

    plat_mutex_lock(&o->lock);
    if (o->count >= AUDOUT_MAX_BUFFERS) {
        plat_mutex_unlock(&o->lock);
        rt_log("audout: AppendAudioOutBuffer: %d buffers are already queued", AUDOUT_MAX_BUFFERS);
        rs->result = AUDOUT_R_OUT_OF_RESOURCE;
        return;
    }
    uint64_t end = frames ? out_push(pcm, frames, o->volume * host_volume() * audio_device_volume())
                          : atomic_load_explicit(&out.queued, memory_order_acquire);
    int at = (o->head + o->count) % AUDOUT_MAX_BUFFERS;
    o->q[at] = (AudoutBuffer){tag, end};
    o->count++;
    queue_refresh(o);
    int queued = o->count;
    plat_mutex_unlock(&o->lock);

    if (!o->logged) {
        o->logged = 1;
        rt_log("audout: first buffer appended: tag %#llx, %llu bytes at %#llx, offset %llu, %llu frames, %d queued "
               "(volume %.2f x %.2f)",
               (unsigned long long)tag, (unsigned long long)size, (unsigned long long)samples,
               (unsigned long long)offset, (unsigned long long)frames, queued, o->volume, audio_device_volume() * host_volume());
    }
    plat_inc64(&out.appends);
    if ((rt_trace & RT_TRACE_AUDIO) && out.level_count >= (uint64_t)AUDOUT_RATE * 5) {   // one line per 5 s queued
        uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
        uint64_t queued_now = atomic_load_explicit(&out.queued, memory_order_acquire);
        rt_log("audout: level rms %.4f peak %.4f over %.1f s; %lld buffers appended, %lld released (%lld empty), "
               "%llu frames queued, %.1f ms in the ring, %lld overruns",
               sqrt(out.level_sum / (double)out.level_count), out.level_peak, out.level_count / 2.0 / AUDOUT_RATE,
               (long long)out.appends, (long long)out.releases, (long long)out.zero_releases,
               (unsigned long long)(queued_now - 1), (queued_now - played) * 1000.0 / AUDOUT_RATE,
               (long long)out.overruns);
        out.level_sum = out.level_peak = 0;
        out.level_count = 0;
    }
    notify_released();   // the device may already be past it (no device, or an empty buffer)
}

// GetReleasedAudioOutBuffers: the tags of the buffers the device has finished, at most as many as the client's array
// holds (the SDK passes room for one).
static void out_released(AudoutOut *o, const IpcRequest *rq, IpcResponse *rs) {
    int is_static = 0;
    const IpcBuffer *b = first_buffer(rq, &is_static);
    uint32_t slots = b && b->size >= 8 ? (uint32_t)(b->size / 8) : 1;
    if (slots > AUDOUT_MAX_BUFFERS) slots = AUDOUT_MAX_BUFFERS;
    uint64_t tags[AUDOUT_MAX_BUFFERS];
    uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
    uint32_t n = 0;
    plat_mutex_lock(&o->lock);
    while (n < slots && o->count > 0 && played >= o->q[o->head].end) {
        tags[n++] = o->q[o->head].tag;
        o->head = (o->head + 1) % AUDOUT_MAX_BUFFERS;
        o->count--;
    }
    int more = o->count > 0 && played >= o->q[o->head].end;
    queue_refresh(o);
    plat_mutex_unlock(&o->lock);
    if (b && b->size) {
        uint64_t *dst = ipc_guest(b->addr, b->size);
        memset(dst, 0, b->size);
        memcpy(dst, tags, (size_t)n * 8);
        if (is_static) rs->xout[rs->nxout++] = (IpcBuffer){b->addr, b->size, b->index};
    }
    uint32_t count = n;
    IPC_OUT(rs, count);
    if (n) plat_add64(&out.releases, (int64_t)n);
    else plat_inc64(&out.zero_releases);
    if (more) kern_event_signal(o->event);   // the client takes one tag per call; keep it coming back
}

static uint64_t out_played_samples(AudoutOut *o) {
    uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
    uint64_t frames = played > o->played_base ? played - o->played_base : 0;
    return frames * (uint64_t)o->channels;
}

static void out_session_close(void *state) {
    AudoutOut *o = state;
    plat_mutex_lock(&g_sessions_lock);
    for (int i = 0; i < AUDOUT_MAX_SESSIONS; i++)
        if (g_sessions[i] == o) g_sessions[i] = NULL;
    plat_mutex_unlock(&g_sessions_lock);
    plat_dec32(&g_sessions_live);
    rt_log("audout: session closed (%d buffers still queued)", o->count);
    session_unref(o);
}

static void audout_out_handler(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    AudoutOut *o = s->state;
    switch (rq->cmd) {
    case 0: {   // GetAudioOutState
        uint32_t v = o->state;
        IPC_OUT(rs, v);
        return;
    }
    case 1:   // Start
        out_start(o);
        return;
    case 2:   // Stop
        out_stop(o);
        return;
    case 3:   // AppendAudioOutBuffer
    case 7:   // AppendAudioOutBufferAuto (what nn::audio::AppendAudioOutBuffer sends)
        out_append(o, rq, rs);
        return;
    case 4:   // RegisterBufferEvent -> handle<copy>
        rs->copy[rs->ncopy++] = kern_event_readable(o->event);
        return;
    case 5:   // GetReleasedAudioOutBuffers
    case 8:   // GetReleasedAudioOutBuffersAuto (what nn::audio::GetReleasedAudioOutBuffer sends)
        out_released(o, rq, rs);
        return;
    case 6: {   // ContainsAudioOutBuffer(tag) -> bool
        uint64_t tag = IPC_ARG(rq, uint64_t, 0);
        uint8_t contains = (uint8_t)out_contains(o, tag);
        IPC_OUT(rs, contains);
        return;
    }
    case 9: {   // GetAudioOutBufferCount
        plat_mutex_lock(&o->lock);
        uint32_t n = (uint32_t)o->count;
        plat_mutex_unlock(&o->lock);
        IPC_OUT(rs, n);
        return;
    }
    case 10: {   // GetAudioOutPlayedSampleCount
        uint64_t n = out_played_samples(o);
        IPC_OUT(rs, n);
        return;
    }
    case 11: {   // FlushAudioOutBuffers -> bool
        uint8_t flushed = (uint8_t)out_flush(o);
        IPC_OUT(rs, flushed);
        return;
    }
    case 12: {   // SetAudioOutVolume
        float v = IPC_ARG(rq, float, 0);
        o->volume = isfinite(v) ? (v < 0 ? 0 : v > 1 ? 1 : v) : 1.0f;
        return;
    }
    case 13: {   // GetAudioOutVolume
        float v = o->volume;
        IPC_OUT(rs, v);
        return;
    }
    default:
        ipc_unimplemented(s, rq);
    }
}

// ---- IAudioOutManager
static const char audout_name[] = "DeviceOut";

// The name the service picked is the out buffer; the SDK sends the name it asked for in an A buffer and the answer in a
// B buffer, so the B buffer wins when both are present.
static const IpcBuffer *name_buffer(const IpcRequest *rq) {
    if (rq->nb && rq->b[0].size) return &rq->b[0];
    if (rq->nw && rq->w[0].size) return &rq->w[0];
    if (rq->nc && rq->c[0].size) return &rq->c[0];
    if (rq->na && rq->a[0].size) return &rq->a[0];
    return NULL;
}

static void write_name(const IpcBuffer *b) {
    if (!b || !b->size) return;
    char *dst = ipc_guest(b->addr, b->size);
    memset(dst, 0, b->size);
    size_t n = sizeof audout_name < b->size ? sizeof audout_name : b->size;
    memcpy(dst, audout_name, n);
}

static void audout_list(const IpcRequest *rq, IpcResponse *rs) {
    const IpcBuffer *b = name_buffer(rq);
    uint32_t count = 0;
    if (b && b->size >= AUDOUT_NAME_BYTES) {
        write_name(b);
        count = 1;
    }
    IPC_OUT(rs, count);
}

static void audout_open(const IpcRequest *rq, IpcResponse *rs) {
    uint32_t rate = 0;
    uint16_t channels = 0;
    if (rq->data_size >= 8) {
        memcpy(&rate, rq->data, 4);
        memcpy(&channels, rq->data + 4, 2);
    }
    if (rate && rate != AUDOUT_RATE) {
        rt_log("audout: OpenAudioOut asked for %u Hz, only %d Hz is offered", rate, AUDOUT_RATE);
        rs->result = AUDOUT_R_INVALID_SAMPLE_RATE;
        return;
    }
    if (channels != 0 && channels != 2) {
        rt_log("audout: OpenAudioOut asked for %u channels, only stereo is offered", channels);
        rs->result = AUDOUT_R_INVALID_CHANNEL_COUNT;
        return;
    }
    if (plat_cas32(&g_sessions_live, 0, 0) >= AUDOUT_MAX_SESSIONS) {
        rt_log("audout: OpenAudioOut with %d sessions already open", AUDOUT_MAX_SESSIONS);
        rs->result = AUDOUT_R_OUT_OF_RESOURCE;
        return;
    }
    out_open();
    AudoutOut *o = calloc(1, sizeof *o);
    if (!o) rt_fatal("audout: session allocation");
    plat_mutex_init(&o->lock);
    o->refs = 1;
    o->state = AUDOUT_STATE_STOPPED;
    o->volume = 1.0f;
    o->rate = AUDOUT_RATE;
    o->channels = AUDOUT_CHANNELS;
    o->event = kern_event_new();
    int slot = -1;
    plat_mutex_lock(&g_sessions_lock);
    for (int i = 0; i < AUDOUT_MAX_SESSIONS && slot < 0; i++)
        if (!g_sessions[i]) slot = i;
    if (slot >= 0) g_sessions[slot] = o;
    plat_mutex_unlock(&g_sessions_lock);
    if (slot < 0) {
        session_unref(o);
        rs->result = AUDOUT_R_OUT_OF_RESOURCE;
        return;
    }
    plat_inc32(&g_sessions_live);

    uint32_t internal[4] = {AUDOUT_RATE, AUDOUT_CHANNELS, AUDOUT_SAMPLE_FORMAT_PCM16, AUDOUT_STATE_STOPPED};
    IPC_OUT(rs, internal);
    write_name(name_buffer(rq));
    rs->move[rs->nmove++] = ipc_session_new("IAudioOut", audout_out_handler, o, out_session_close);
    rt_log("audout: OpenAudioOut -> session %d, %d Hz, %d channels, PCM16", plat_cas32(&g_sessions_live, 0, 0),
           AUDOUT_RATE, AUDOUT_CHANNELS);
}

void audout_manager_handler(IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    (void)s;
    switch (rq->cmd) {
    case 0:   // ListAudioOuts
    case 2:   // ListAudioOutsAuto
        audout_list(rq, rs);
        return;
    case 1:   // OpenAudioOut
    case 3:   // OpenAudioOutAuto
        audout_open(rq, rs);
        return;
    default:
        ipc_unimplemented(s, rq);
    }
}
