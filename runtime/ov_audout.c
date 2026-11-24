// The audio output service ("audout:u"): the legacy nn::audio output path Astral Chain plays all of its sound through.
//
// nn::audio keeps a small set of AudioOut sessions, each fed with AudioOutBuffers of signed 16-bit little-endian
// interleaved PCM at 48 kHz. The game appends a buffer, waits on the buffer event the SDK registered from
// RegisterBufferEvent, takes the tags of the buffers the device has finished with from GetReleasedAudioOutBuffers, and
// refills and appends them again. Nothing here blocks: an append converts the guest PCM into the host ring and returns,
// and the output thread (audio_out.h: WASAPI on Windows, SDL3 elsewhere) drains that ring without taking a guest lock. A
// buffer counts as played once its samples have left the ring, which is what paces the game's audio thread.
//
// 5.1 (AC_AUDIO_OUTPUT=5.1, or auto on an output that takes it: audio_out.h). The game's sound engine is Wwise 2019.1; its Switch sink (main+0x83a110) opens the output
// for stereo, then reads nn::audio::GetAudioOutChannelCount, which returns the count this service answered in
// OpenAudioOut, and sets its master mix to it (a table of channel masks for 1 to 8 channels; 6 gives 0x60f, FL FR FC
// LFE SL SR). So the service answers 6 and the game mixes 5.1 itself: the sound effects, voices and reverbs (the
// se_master bus tree) follow the output, the music (mixer_bgm) is authored stereo and stays in front, and the LFE gets
// nothing. The sources are mono (7117 streamed files, the voices) or stereo (1053). The count is fixed when the game
// opens its output, at start, so the setting applies after a restart.
// Measured on the new game's opening (title to the motorway, radio dialogue over traffic, 2 s RMS, 2026-11-21): in
// stereo the fronts carry 0.069-0.082; in 5.1 the fronts 0.049-0.059, the centre 0.054-0.079 (the mono 2D sounds, the
// voices and interface beeps, go there at full level instead of -3 dB per side), the surrounds 0.005-0.026, the LFE 0.
// The voices came out about 3.4 dB louder than in stereo and a phone beep too loud on headphones, hence the centre
// default of -2 dB (AC_AUDIO_CENTER). The player sets a gain per speaker, sends the bass to the subwoofer and tests the
// speakers from the menu (mix_frame). 8 channels (7.1) also ran without a fault in one test and are left out for now.
// A device without a 5.1 path gets the mix folded down here (ITU-R BS.775: centre and surround at -3 dB, the fronts
// at unity, the LFE dropped), so the setting never silences anything.
//
// audren:u (runtime/audren.c) is the other renderer path and has its own mixer over SDL; this game never opens a
// renderer, only the IAudioDevice service for the master volume (nn::audio::SetAudioDeviceOutputVolume).
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_out.h"
#include "audren.h"
#include "cfg.h"
#include "host.h"
#include "ipc.h"
#include "ov.h"
#include "platform/platform.h"

#define AUDOUT_RATE 48000
#define AUDOUT_CHANNELS 2                     // what the game mixes unless the output is 5.1 (out_open)
#define AUDOUT_MAX_CHANNELS 6                 // 5.1: FL FR FC LFE SL SR
#define AUDOUT_MAX_SESSIONS 8
#define AUDOUT_MAX_BUFFERS 32                 // the SDK's own queue limit (AudioCore::BufferCount)
#define AUDOUT_RING_FRAMES 32768              // frames of output queued ahead, about 680 ms
#define AUDOUT_DRY_MAX (AUDOUT_RATE / 2)      // frames of one dry stretch that count as sound coming late (0.5 s)
#define PAUSE_RAMP (AUDOUT_RATE / 200)        // 5 ms: the fade at each end of a pause (out_fill)
#define AUDOUT_FOLD 0.70710678f               // -3 dB: centre and surrounds in a fold-down to stereo
#define AUDOUT_SAMPLE_FORMAT_PCM16 1u
#define AUDOUT_STATE_STARTED 0u
#define AUDOUT_STATE_STOPPED 1u
#define AUDOUT_NAME_BYTES 0x100
#define AUDOUT_R_INVALID_SAMPLE_RATE 0x7C1u   // nn::audio::ResultInvalidSampleRate
#define AUDOUT_R_INVALID_CHANNEL_COUNT 0x7C2u // nn::audio::ResultInvalidChannelCount
#define AUDOUT_R_OUT_OF_RESOURCE 0xC401u

// ---- output device: one ring and one output stream for every AudioOut session
//
// played and queued are cumulative frame counters; the ring holds the frames in [played, queued), indexed modulo
// AUDOUT_RING_FRAMES, each AUDOUT_MAX_CHANNELS floats wide whatever the game mixes. Both start at 1 so a queued buffer's
// end marker is never 0 (0 means "no buffer"). Guest threads append under push (one producer at a time); the output
// thread only moves played.
static struct {
    int device_ok, opened;
    int channels;   // what the game mixes, chosen once at the first OpenAudioOut (out_open): 2 or 6
    float ring[AUDOUT_RING_FRAMES * AUDOUT_MAX_CHANNELS];
    _Atomic uint64_t played, queued;
    PlatMutex push;
    volatile int64_t overruns;
    volatile int64_t appends, releases, zero_releases;
    // The ring ran dry while the game was feeding it: the frames of silence the device got instead, and how many
    // times it happened (audout_pull). fed: the game has queued sound since the last stop or flush.
    _Atomic uint64_t dry_frames, dry_gaps;
    _Atomic int fed;
    uint64_t dry_logged, dry_gaps_logged, dry_log_ns;   // what the last log line reported, and when (out_dry_report)
    double level_sum[AUDOUT_MAX_CHANNELS], level_peak;   // per channel, the game's mix after the volume (AC_TRACE=audio)
    uint64_t level_count;                                // frames in level_sum
} out = {.push = PLAT_MUTEX_INIT, .played = 1, .queued = 1, .channels = AUDOUT_CHANNELS};

typedef struct {
    uint64_t tag, end;   // end: the cumulative frame counter at which this buffer has been played out
} AudoutBuffer;

typedef struct AudoutOut {
    volatile int32_t refs;         // the IPC session plus any output-thread use
    uint32_t state;
    float volume;
    int rate, channels;
    uint32_t event;                // writable kernel handle behind the readable one the client waits on
    PlatMutex lock;                // the buffer queue, taken by guest threads only
    AudoutBuffer q[AUDOUT_MAX_BUFFERS];
    int head, count;
    _Atomic uint64_t oldest_end;   // q[head].end, 0 when the queue is empty (read by the output thread)
    _Atomic uint64_t signalled;    // the last oldest_end the event was signalled for
    uint64_t played_base;          // out.played when the session started, for GetAudioOutPlayedSampleCount
    int logged;
} AudoutOut;

// Sessions the output thread may look at. It only ever try-locks the list, so it can never wait for a guest thread (which
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

// Signals the buffer event of every session whose oldest queued buffer has been played out. Runs on the output
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

// ---- the player's mix, on the output thread: a gain per speaker, the bass to the subwoofer, the speaker test, meters
//
// Applied as the output takes frames rather than as the game queues them, so a gain moved in the menu is heard at once
// and not ~80 ms later (the ring), and the test and the meters work while the game queues nothing. Channel order FL FR
// FC LFE SL SR; a stereo mix uses the first two and takes no gains (they are a 5.1 mix's), but the test and the meters.
// The configuration is read on this thread through cfg_version, whose values are safe to read from any thread (cfg.c
// keeps a replaced value's string).
#define MIX_SUB_SUM 0.316f        // -10 dB: a receiver plays the LFE channel 10 dB hot, so the summed bass goes in 10 dB down
#define MIX_TEST_ON (AUDOUT_RATE * 6 / 5)   // the speaker test: 1.2 s of noise per speaker...
#define MIX_TEST_STEP (AUDOUT_RATE * 8 / 5)  // ...every 1.6 s
#define MIX_TEST_FADE (AUDOUT_RATE / 50)     // 20 ms fades, no clicks
#define MIX_TEST_PINK 0.058f      // pink noise at -20 dBFS RMS (measured: the generator's RMS is 1.72)
#define MIX_TEST_LOW 0.092f       // the same noise low-passed at 120 Hz, -20 dBFS RMS (its RMS is 1.09)

typedef struct { float b0, b1, b2, a1, a2, z1, z2; } MixBiquad;

static float biquad_run(MixBiquad *q, float x) {   // transposed direct form II
    float y = q->b0 * x + q->z1;
    q->z1 = q->b1 * x - q->a1 * y + q->z2;
    q->z2 = q->b2 * x - q->a2 * y;
    return y;
}

// The test order of an A/V receiver, clockwise from the front left; a stereo mix tests its two.
static const int mix_test_51[] = {0, 2, 1, 5, 4, 3}, mix_test_20[] = {0, 1};

static struct {
    unsigned seen;                              // cfg_version the gains were read at
    float gain[AUDOUT_MAX_CHANNELS];            // linear, per speaker
    float volume;                               // AC_VOLUME, for the test noise (the game's sound has it already)
    int bass;                                   // AC_AUDIO_BASS
    MixBiquad bass_lp, test_lp;                 // 120 Hz low-pass (Butterworth, Q 0.707) of the bass and the LFE test
    _Atomic int test_req;                       // the menu: 1 start, -1 stop, 0 nothing new
    _Atomic int test_ch;                        // the speaker the test plays on now, -1 none (read by the menu)
    _Atomic int test_on;                        // the test runs (the output thread starts and ends it)
    uint64_t test_pos;                          // frames since the test started
    uint32_t rng;
    float pink[3];
    _Atomic uint32_t peak[AUDOUT_MAX_CHANNELS]; // float bits: the highest sample per speaker since the menu last read
} mix = {.seen = ~0u, .test_ch = -1, .rng = 0x12345678u,
         .bass_lp = {6.1006172e-05f, 1.2201234e-04f, 6.1006172e-05f, -1.9777863f, 0.9780303f, 0, 0},
         .test_lp = {6.1006172e-05f, 1.2201234e-04f, 6.1006172e-05f, -1.9777863f, 0.9780303f, 0, 0}};

static float db_gain(const char *key) {
    double d = cfg_double(key);
    return (float)pow(10.0, (d < -12 ? -12 : d > 6 ? 6 : d) / 20.0);
}

static void mix_update(void) {
    unsigned ver = cfg_version();
    if (ver == mix.seen) return;
    mix.seen = ver;
    mix.gain[0] = db_gain("AC_AUDIO_FRONT_LEFT");
    mix.gain[1] = db_gain("AC_AUDIO_FRONT_RIGHT");
    mix.gain[2] = db_gain("AC_AUDIO_CENTER");
    mix.gain[3] = db_gain("AC_AUDIO_SUBWOOFER");
    mix.gain[4] = db_gain("AC_AUDIO_SURROUND_LEFT");
    mix.gain[5] = db_gain("AC_AUDIO_SURROUND_RIGHT");
    mix.bass = cfg_bool("AC_AUDIO_BASS");
    double pct = cfg_double("AC_VOLUME");
    mix.volume = (float)((pct < 0 ? 0 : pct > 100 ? 100 : pct) / 100);
}

// Paul Kellet's economy pink noise filter over a xorshift white noise in [-1, 1).
static float pink_noise(void) {
    uint32_t s = mix.rng;
    s ^= s << 13, s ^= s >> 17, s ^= s << 5;
    mix.rng = s;
    float w = (float)s * (1.0f / 2147483648.0f) - 1.0f;
    mix.pink[0] = 0.99765f * mix.pink[0] + w * 0.0990460f;
    mix.pink[1] = 0.96300f * mix.pink[1] + w * 0.2965164f;
    mix.pink[2] = 0.57000f * mix.pink[2] + w * 1.0526913f;
    return mix.pink[0] + mix.pink[1] + mix.pink[2] + w * 0.1848f;
}

// The speaker test's sample for this frame: *ch the speaker it plays on, -1 between two (or once it is over).
static float test_sample(int nmix, int *ch) {
    const int *order = nmix == 6 ? mix_test_51 : mix_test_20;
    const int count = nmix == 6 ? 6 : 2;
    uint64_t step = mix.test_pos / MIX_TEST_STEP, at = mix.test_pos % MIX_TEST_STEP;
    mix.test_pos++;
    if (step >= (uint64_t)count) {
        mix.test_on = 0;
        *ch = -1;
        return 0;
    }
    float p = pink_noise();
    *ch = order[step];
    if (at >= MIX_TEST_ON) {
        *ch = -1;
        return 0;
    }
    float env = at < MIX_TEST_FADE ? (float)at / MIX_TEST_FADE
              : at > MIX_TEST_ON - MIX_TEST_FADE ? (float)(MIX_TEST_ON - at) / MIX_TEST_FADE : 1.0f;
    if (order[step] == 3) return biquad_run(&mix.test_lp, p) * MIX_TEST_LOW * env;   // the subwoofer: its band only
    return p * MIX_TEST_PINK * env;
}

void audout_test(int on) { atomic_store_explicit(&mix.test_req, on ? 1 : -1, memory_order_release); }
int audout_test_channel(void) { return atomic_load_explicit(&mix.test_ch, memory_order_acquire); }
int audout_test_running(void) { return mix.test_on || atomic_load_explicit(&mix.test_req, memory_order_acquire) == 1; }

// The highest sample per speaker (FL FR FC LFE SL SR) since the last call, after the gains; 0 for a speaker the mix has
// not. Any thread (the menu's meters).
void audout_meters(float peak[AUDOUT_MAX_CHANNELS]) {
    for (int c = 0; c < AUDOUT_MAX_CHANNELS; c++) {
        uint32_t bits = atomic_exchange_explicit(&mix.peak[c], 0, memory_order_acq_rel);
        memcpy(&peak[c], &bits, 4);
    }
}

// One frame of the mix: x (nmix wide, the ring's frame or silence) through the gains, the bass and the test, into d
// (channels wide: nmix, or 2 for a fold-down of a 5.1 mix). peak: this call's highest sample per speaker.
static void mix_frame(const float *x, int nmix, float *d, int channels, float *peak) {
    float f[AUDOUT_MAX_CHANNELS];
    if (nmix == 6) {
        for (int c = 0; c < 6; c++) f[c] = x[c] * mix.gain[c];
        float low = biquad_run(&mix.bass_lp, f[0] + f[1] + f[2] + f[4] + f[5]);
        if (mix.bass) f[3] += low * MIX_SUB_SUM * mix.gain[3];
    } else {
        f[0] = x[0], f[1] = x[1];
    }
    if (mix.test_on) {   // the game's sound is muted while the test runs, as on a receiver
        int ch;
        float t = test_sample(nmix, &ch) * mix.volume;
        for (int c = 0; c < nmix; c++) f[c] = 0;
        if (ch >= 0) f[ch] = t * (nmix == 6 ? mix.gain[ch] : 1.0f);
        atomic_store_explicit(&mix.test_ch, ch, memory_order_relaxed);
        if (!mix.test_on) atomic_store_explicit(&mix.test_ch, -1, memory_order_relaxed);
    }
    for (int c = 0; c < nmix; c++)
        if (fabsf(f[c]) > peak[c]) peak[c] = fabsf(f[c]);
    if (channels == nmix) {
        for (int c = 0; c < channels; c++) d[c] = f[c];
    } else {   // 5.1 to stereo (ITU-R BS.775): the fronts at unity, centre and surrounds at -3 dB, the LFE dropped
        d[0] = f[0] + AUDOUT_FOLD * (f[2] + f[4]);
        d[1] = f[1] + AUDOUT_FOLD * (f[2] + f[5]);
    }
}

// The output thread (audio_out.h): frames channels wide (mixed, the game's count, or 2 for a fold-down of a 5.1 mix), silence
// when the ring runs dry. A dry ring while the game feeds it means the game's sound thread did not refill in time: the
// device plays silence, and everything the game streams (a voice, a movie's sound) comes that much later, for good, since
// the game queues a buffer only when one comes back. Those frames are counted for the log (out_dry_report) and move the
// dry clock on (audio_mixer.c audio_device_dry_ns), by which a cutscene's picture waits for its sound (frame_rate.c
// cutscene_lost); every request moves the device clock on. Only the first AUDOUT_DRY_MAX frames of one dry stretch move
// the dry clock: a longer one is not a late sound thread but a game that stopped feeding the output, and no picture has
// to wait for that. AC_TEST_FREEZE (frame_rate.c frame_test_frozen): while a test freeze lasts the device gets silence,
// the ring is left alone and neither clock moves, as if the process stood still. A paused game (kernel.c kern_paused)
// is the same, except that the speaker test still plays (the menu is what paused the game) and that both ends are
// ramped against clicks: the first request of the pause plays the next PAUSE_RAMP frames fading out without taking
// them, and the first one after it plays them again fading in, so pauses add no offset between sound and picture.
static void out_fill(float *dst, int frames, int mixed, int channels) {
    static uint64_t dry;   // frames of silence in the dry stretch the last request was in, 0 none (output thread only)
    static const float silence[AUDOUT_MAX_CHANNELS];
    static int was_paused;   // the last request found the game paused (output thread only)
    const int nmix = mixed;   // out.channels once out_open has it; under auto the output knows it first
    uint64_t played = atomic_load_explicit(&out.played, memory_order_relaxed);
    uint64_t queued = atomic_load_explicit(&out.queued, memory_order_acquire);
    uint64_t have = queued > played ? queued - played : 0;
    int frozen = frame_test_frozen();
    int paused = kern_paused(), turn = paused != was_paused;
    was_paused = paused;
    if (paused && (!turn || have > PAUSE_RAMP)) have = turn ? PAUSE_RAMP : 0;
    if (frozen) have = 0;
    int n = have < (uint64_t)frames ? (int)have : frames;
    int ramp = !turn ? 0 : n < PAUSE_RAMP ? n : PAUSE_RAMP;
    uint64_t late = 0;   // frames of this request that count as the game's sound coming late
    if (n < frames && !frozen && !paused && atomic_load_explicit(&out.fed, memory_order_relaxed)) {
        uint64_t silent = (uint64_t)(frames - n);
        atomic_fetch_add_explicit(&out.dry_frames, silent, memory_order_relaxed);
        if (!dry) atomic_fetch_add_explicit(&out.dry_gaps, 1, memory_order_relaxed);
        late = dry >= AUDOUT_DRY_MAX ? 0 : silent < AUDOUT_DRY_MAX - dry ? silent : AUDOUT_DRY_MAX - dry;
        dry += silent;
    } else if (n == frames && !paused) {
        dry = 0;
    }
    if (!frozen && !paused) audio_device_clock_add((uint64_t)frames, late, AUDOUT_RATE);
    mix_update();
    int req = atomic_exchange_explicit(&mix.test_req, 0, memory_order_acq_rel);
    if (req == 1) mix.test_on = 1, mix.test_pos = 0;
    if (req == -1 || frozen) {
        mix.test_on = 0;
        atomic_store_explicit(&mix.test_ch, -1, memory_order_relaxed);
    }
    float peak[AUDOUT_MAX_CHANNELS] = {0};
    for (int i = 0; i < frames; i++) {
        const float *x = i < n ? &out.ring[AUDOUT_MAX_CHANNELS * ((played + (uint64_t)i) % AUDOUT_RING_FRAMES)] : silence;
        float ramped[AUDOUT_MAX_CHANNELS];
        if (i < ramp) {
            float g = (float)(paused ? ramp - i : i + 1) / (float)(ramp + 1);
            for (int c = 0; c < nmix; c++) ramped[c] = x[c] * g;
            x = ramped;
        }
        mix_frame(x, nmix, dst + (size_t)channels * (size_t)i, channels, peak);
    }
    for (int c = 0; c < nmix; c++) {   // keep the higher of this call's peak and what the menu has not read yet
        uint32_t bits, old = atomic_load_explicit(&mix.peak[c], memory_order_relaxed);
        float was;
        memcpy(&was, &old, 4);
        if (peak[c] <= was) continue;
        memcpy(&bits, &peak[c], 4);
        atomic_compare_exchange_strong_explicit(&mix.peak[c], &old, bits, memory_order_acq_rel, memory_order_relaxed);
    }
    if (!paused) atomic_store_explicit(&out.played, played + (uint64_t)n, memory_order_release);
    notify_released();
}

// Opens the output once and chooses what the game mixes: 6 channels with AC_AUDIO_OUTPUT=5.1, 2 with stereo, and under
// auto (the default) what the output's first device takes (audio_out.h audio_out_auto_channels). Without an output
// (AC_AUDIO=none, or one that cannot start) the service still answers and every buffer is reported played at once, so
// the game's audio thread keeps recycling instead of stalling (the same choice audio_mixer.c makes). Runs on a guest
// thread: opening a device takes tens of ms, so the virtual core is given back meanwhile.
static void out_open(void) {
    if (out.opened) return;
    out.opened = 1;
    const char *set = cfg_value("AC_AUDIO_OUTPUT");
    int want = !strcmp(set, "5.1") ? 6 : !strcmp(set, "stereo") ? AUDOUT_CHANNELS : AUDIO_OUT_AUTO;
    out.channels = want ? want : AUDOUT_CHANNELS;
    const char *mode = cfg_get("AC_AUDIO");
    if ((mode && !strcmp(mode, "none")) || cfg_get("AC_AUDIO_OFF")) {
        rt_log("audout: AC_AUDIO=none: buffers are recycled without a device");
        return;
    }
    int away = kern_core_away();
    int got = audio_out_start(want, out_fill);
    kern_core_back(&away);
    out.device_ok = got != 0;
    if (got) out.channels = got;
}

int audout_mix_channels(void) { return out.opened ? out.channels : 0; }

// Converts and queues guest PCM (signed 16-bit interleaved, out.channels wide) and returns the frame counter at which it
// has been played. The player's mix applies later, on the output thread (mix_frame). Frames past the ring's capacity are
// dropped; the buffer still comes back to the game.
static uint64_t out_push(const int16_t *pcm, uint64_t frames, float volume) {
    const int ch = out.channels;
    const float gain = volume * (1.0f / 32768.0f);
    plat_mutex_lock(&out.push);
    uint64_t queued = atomic_load_explicit(&out.queued, memory_order_relaxed);
    uint64_t played = atomic_load_explicit(&out.played, memory_order_acquire);
    uint64_t have = queued > played ? queued - played : 0;
    uint64_t room = have < AUDOUT_RING_FRAMES ? AUDOUT_RING_FRAMES - have : 0;
    uint64_t n = frames < room ? frames : room;
    if (!out.device_ok) n = 0;
    for (uint64_t i = 0; i < n; i++) {
        float *dst = &out.ring[AUDOUT_MAX_CHANNELS * ((queued + i) % AUDOUT_RING_FRAMES)];
        const int16_t *src = pcm + (size_t)ch * i;
        for (int c = 0; c < ch; c++) {
            float v = (float)src[c] * gain;
            dst[c] = v;
            out.level_sum[c] += (double)v * v;
            if (fabsf(v) > out.level_peak) out.level_peak = fabsf(v);
        }
    }
    out.level_count += n;
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

// One log line when the ring ran dry (out_fill counts, on the output thread; this runs on the game's sound thread at its
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
        char rms[160];   // per channel: L R, or L R C LFE SL SR
        int at = 0;
        for (int c = 0; c < out.channels; c++)
            at += snprintf(rms + at, sizeof rms - (size_t)at, " %.4f", sqrt(out.level_sum[c] / (double)out.level_count));
        rt_log("audout: level rms%s peak %.4f over %.1f s; %lld buffers appended, %lld released (%lld empty), "
               "%llu frames queued, %.1f ms in the ring, %lld overruns",
               rms, out.level_peak, (double)out.level_count / AUDOUT_RATE, (long long)out.appends,
               (long long)out.releases, (long long)out.zero_releases, (unsigned long long)(queued_now - 1),
               (queued_now - played) * 1000.0 / AUDOUT_RATE, (long long)out.overruns);
        memset(out.level_sum, 0, sizeof out.level_sum);
        out.level_peak = 0;
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
    // The game asks for stereo and then mixes for the count answered (the header's 5.1 paragraph); 6 is taken too.
    if (channels != 0 && channels != 2 && channels != 6) {
        rt_log("audout: OpenAudioOut asked for %u channels, only stereo and 5.1 are offered", channels);
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
    o->channels = out.channels;
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

    uint32_t internal[4] = {AUDOUT_RATE, (uint32_t)out.channels, AUDOUT_SAMPLE_FORMAT_PCM16, AUDOUT_STATE_STOPPED};
    IPC_OUT(rs, internal);
    write_name(name_buffer(rq));
    rs->move[rs->nmove++] = ipc_session_new("IAudioOut", audout_out_handler, o, out_session_close);
    rt_log("audout: OpenAudioOut (%u channels asked) -> session %d, %d Hz, %d channels, PCM16", channels,
           plat_cas32(&g_sessions_live, 0, 0), AUDOUT_RATE, out.channels);
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
