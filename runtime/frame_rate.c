// Frame rate: the game's 60 fps patch (recomp/patches-v101.txt, set fps60) is on by default, and the menu's frame
// rate row switches it off for the game's shipping 30 fps. The set replaces the game's frame timer with a real-time step
// (rt_fps_timer) and asks the frame call for present interval 1; this file switches the set while the game runs, keeps
// the step and the whole 30 fps ticks it adds up to, caps the host's present rate at 60 or 30, and keeps the world clock
// for scripted input. The whole ticks and the step make the game's per-frame counters and blends frame-rate independent
// (the input/action timers, hitstop, rumble, the camera smoothing, the dialogue text reveal, the cue scheduler).
//
// The game steps a cutscene by a fixed frame, so at 60 presents per second it would play twice as fast as its
// animation data. Under the 60 fps patch an in-engine cutscene steps by the real time instead and presents at 60
// (cutscene_step below); a cutscene with a full-screen movie is held to 30 presents per second (cutscene_rate below)
// and steps by the whole 30 fps frames of the real time (held_step below), as every cutscene does at 30 fps
// (rt_fixed_step): the game's sound plays on the audio device's clock, and the picture has to keep to it.
//
// What remains:
//   - the frame limit (AC_FPS): 60 fps by default, 30 fps when the patch is off. The pacer (nvn_api_obj.c) holds
//     frames to that rate where the game presents faster than it.
//   - the world clock (frame_step_world_us): wall time from the first present. Recorded and scripted input
//     (input.c) runs on it, so a replay presses at the same game time at any frame rate.
//
// AC_TRACE=perf adds a line per 120 presents: fps and the game seconds stepped against wall seconds.
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audren.h"
#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"

#define STEP_MAX_S (1.0 / 15.0)    // longest step; a longer frame slows the game down
#define STEP_RESET_S 0.25          // a gap this long (loading, a debugger) restarts the average

static double step_now;   // the step the game used this frame (whole vsyncs: interval / 60)
static uint64_t last_ns;

// ---- dynamic frame rate: the game's frame step (30 fps units) and the whole 30 fps ticks it adds up to.
// Both come from the game's own frame timer (main+0x91DBB0, rt_fps_timer below), once per game frame on the thread that
// steps the game, so every counter site of a frame sees the same ticks and the ticks add up to exactly the step the game
// moved by. The guest counter sites read them through the helpers below.
static _Atomic uint32_t g_fps_ticks;      // whole ticks of this game frame
static _Atomic uint32_t g_fps_step_bits;  // the game's step this frame (30 fps units, a float)
static double g_fps_rem;                  // what the whole ticks left of the steps (game thread only)
static double g_fps_debt_ms;              // frame time not yet handed out as whole present periods (game thread only)
static volatile int in_cutscene;          // the game's fixed-step flag at the last timer call

static int fps60_on = -1, fps_hz;
static unsigned fps_seen;
static int fps_read;

// ---- 60 fps cutscenes: part of the 60 fps patch (AC_FPS=60), without a setting of their own
// The game steps a cutscene by a fixed 1.0 (1/30 s) a frame and plays it frame-locked, so at 60 presents per second it
// would run at twice its speed. Under the 60 fps patch a cutscene frame steps by the real time instead, in
// whole 60 Hz vsyncs (0.5 each): the cutscene presents at 60 and keeps to the wall clock, which the voices and music
// play on. A frame that took longer than its vsyncs (a hitch, a shader compile) leaves its time as debt, which the next
// frames pay back at up to 4 vsyncs (step 2.0, the original timer's high clamp) each, so the picture catches up with
// the sound instead of staying behind. The debt is capped at CUT_DEBT_MAX_MS: what a longer stall leaves beyond that
// is dropped (and counted in the log). The cap was 1 s up to 1.12.2, and hitches that come close together (pipelines
// built for a new scene, a slow disk) went past it: the picture then stayed behind its sound for the rest of the
// cutscene. A stall the sound stood still in as well (the process suspended, a debugger break) is not caught up at
// all: the debt is there to bring the picture back to the sound, and that did not move (cutscene_lost), which is what
// keeps such a pause from ending in seconds of fast-forward. The cut clock's end-of-cut test is made
// exact for steps other than 1.0 (rt_cut_* below). A full-screen movie (a Bink file with its own clock) keeps the
// cutscene it plays in at the 30 fps hold (rt_movie_* below).
//
// A cutscene held to 30 keeps to the wall clock the same way, in whole 30 fps frames (held_step): it steps 1.0 a
// frame, and 2.0 once it is a frame or more behind. The game presents a held cutscene at most 30 times a second,
// so without this every frame that came late stayed lost: on a machine that shows the opening movie at 26 frames per
// second the scene ran at 87% of its speed while its sound played on at full speed. The movie in it
// catches up on its own clock, by skipping frames (rt_movie_skip_w0). At 30 fps (the patch off) every cutscene is
// such a held one, stepped by rt_fixed_step.
#define CUT_VSYNC_MS (1000.0 / 60.0)
#define CUT_VSYNCS_MAX 4             // a frame steps 2.0 at most, the original timer's high clamp
#define HELD_FRAME_MS (1000.0 / 30.0)
#define HELD_RUSH_ON_MS (4 * HELD_FRAME_MS)   // the debt from which a held cutscene presents at 60 to catch up
#define HELD_FRAMES_MAX 2            // the same clamp, in the 30 fps frames a held cutscene steps by
#define CUT_DEBT_MAX_MS 5000.0       // the frame time a cutscene, and the movie in it, catch up after stalls
#define CUT_STALL_MS 100.0           // a frame this long is checked against the audio device's clock
#define CUT_STALL_SLACK_MS 50.0      // what the device's clock may lag a frame's time by, on top of two of its periods
#define MOVIE_HOLD_FRAMES 8          // a movie counts as playing for this many game frames after its last BinkWait

static volatile int cut60_now;            // this game frame steps a cutscene at 60 (rt_fps_timer; game thread only writes)
static volatile int cut_exact_now;        // this game frame's cut clocks end their cuts exactly (rt_cut_*): a cutscene
                                          // at 60, or a held one on a frame that steps other than 1.0
static volatile int held_rush;            // a held cutscene is far behind and presents at 60 until it is back (held_step)
static _Atomic unsigned fps_switches;     // times frame_limit_live switched the 60 fps patch (rt_fps_timer restarts on it)
static _Atomic uint64_t g_timer_frames;   // rt_fps_timer calls so far: the game frame the movie hold counts in
static double g_cut_debt_ms;              // cutscene time not yet stepped (game thread only)
// one run of cutscene frames, for the log line at its end (cutscene_report; game thread only)
static unsigned cut_catchup;
static double cut_drop_ms, cut_debt_peak;

static int movie_playing(void);
static void frame_limit_live(void);

void frame_step_init(void) {
    frame_limit_live();
}

int frame_step_realtime(void) { return 0; }

// The game's frame timer, main+0x91DBB0 (one call per game frame, from the main loop at main+0x3E39CC; x19 = the timer
// object, x21 = 0 while the flag of main+0x42FE60 asks for a fixed step, which the game sets while a cutscene plays).
// Fields: +0xC8 the previous tick, +0xD0 the step in 30 fps frames (what about 120 sites read), +0xD4 the carry,
// +0xD8 / +0xE4 the frame-time sum and frame count it averages, +0xDC 30/1000, +0xE0 this frame's time in ms.
// The original clamps the step to [1, 2] and turns a shorter frame into carry, so at 60 fps every frame steps a full
// 1/30 s; it also takes the audio-sync override (main+0x8A9570, -1 when unset; written by the cutscene player at
// main+0x8B9F0C and main+0x8C2F5C as a 1 +- correction, clamped to [0.5, 2]) as the whole step.
//
// rt_fps_timer replaces everything after the frame time store (site main+0x91DC10; the site branches to the tick store
// at main+0x91DCB4 next):
//   - fixed step (a cutscene): the whole vsyncs of the real time (cutscene_step), presented at 60. While a full-screen
//     movie plays in it the present pacer holds the cutscene to 30 presents per second (nvn_api_obj.c present_period,
//     frame_step_cutscene) and the step is the whole 30 fps frames of the real time (held_step): 1.0, as the
//     original, while the frames come on time, 2.0 on a frame that makes up for late ones.
//   - otherwise the frame time, on the present grid (frame_paced_ms), times 30/1000, times the override when it is set:
//     the override is a speed relative to one 30 fps frame, so it scales the measured step instead of replacing it.
//   - clamped to [1/16, 2]: 2 is the original's high clamp (a slower frame slows the game); 1/16 keeps the step away
//     from 0 for the few readers that divide by it. No carry: the step is the frame's real time.
// The averaging fields and the carry are cleared, so the original takes over cleanly when the set is switched off.
static double frame_paced_ms(double dt_ms);

static void fps_publish(float step) {
    uint32_t bits; memcpy(&bits, &step, 4);
    atomic_store_explicit(&g_fps_step_bits, bits, memory_order_relaxed);
    g_fps_rem += step;
    uint32_t ticks = (uint32_t)g_fps_rem;
    g_fps_rem -= (double)ticks;
    atomic_store_explicit(&g_fps_ticks, ticks, memory_order_relaxed);
}

// The cutscene step of one frame at 60 (30 fps units): the whole vsyncs of the time the cutscene owes, 0 to 4
// (cutscene_units). The first frame of a run starts without debt and counts one to two vsyncs, whatever came before it
// (a load, a movie), and restarts the whole-tick remainder at 1/2, so the first frame and every other one after it
// have a tick. A 30 fps frame runs its logic at the start of its 1/30 s; of the two 60 fps frames in that 1/30 s, the
// first starts at the same time, so the tick-gated talk logic (rt_fps_talk_*, rt_fps_voiced_*) runs when it runs at
// 30 rather than on whatever phase the gameplay before the cutscene left.
//
// sound_lost_ms: the time since a mark in which the sound did not play, which the picture that goes with it gives up
// (the cutscene's step here, a movie's clock in rt_movie_late_w9), or it would end up ahead of its sound.
//   - The sound stood still: a stretch over CUT_STALL_MS counts only as long as the audio device asked for sound in
//     it (audio_mixer.c audio_device_clock_ns) plus the slack and two device periods, the steps that clock moves by,
//     so a device with long periods does not read as one that stood still. When the game thread alone stalled (a
//     shader compile, streaming) the device kept asking and the whole stretch counts; when the device stood still too
//     (a suspended process, a debugger break), the sound is where it was and the rest of the stretch is left out.
//   - The sound came late: when the game's sound thread does not refill the audio output in time (a processor with
//     too few cores for the moment's work), the output plays silence, and what the game streams is that much later
//     from then on (ov_audout.c audout_pull, audio_mixer.c audio_device_dry_ns). Each such gap would leave the
//     picture further ahead of its sound.
// Without a device the clocks are 0 and nothing is lost. The mark is taken again at each reading; first: only that.
#define CUT_WAIT_STEP (1.0f / 16.0f)   // the step of a frame that waits: the timer's low clamp, not 0

static double wall_s(void);

typedef struct { uint64_t wall_ns, clock_ns, dry_ns; } SoundMark;   // one reading of the wall clock and the device's

typedef struct { double dry_ms, still_ms, dt_ms; } SoundLoss;   // what sound_lost_ms found, by kind, and the stretch

static SoundMark cut_mark;   // the last frame of the cutscene run (game thread only)

static double sound_lost_ms(SoundMark *mark, int first, SoundLoss *out) {
    SoundMark was = *mark;
    *mark = (SoundMark){kern_now_ns(), audio_device_clock_ns(), audio_device_dry_ns()};
    if (out) *out = (SoundLoss){0, 0, 0};
    if (first || !was.clock_ns) return 0;
    double dry = (double)(mark->dry_ns - was.dry_ns) / 1e6, still = 0;
    double dt_ms = (double)(mark->wall_ns - was.wall_ns) / 1e6;
    if (dt_ms > CUT_STALL_MS) {
        // The slack only tells a device that stood still from one whose clock is a period or two behind; what is
        // lost is the stretch less what the device asked for, or every such stall would leave the slack behind.
        double asked_ms = (double)(mark->clock_ns - was.clock_ns) / 1e6;
        double slack_ms = CUT_STALL_SLACK_MS + 2e-6 * (double)audio_device_period_ns();
        if (asked_ms + slack_ms < dt_ms) still = dt_ms - asked_ms;
    }
    if (out) *out = (SoundLoss){dry, still, dt_ms};
    return dry + still;
}

// ---- the audio desync log, for reports. The picture is measured against the sound as the cutscene and movie log lines
// say (picture minus sound); beyond DESYNC_MS for DESYNC_SUSTAIN_NS (a hitch is a dip of the picture, over in under
// that) it is a desync the player saw and heard, and one line says so, with the case of tools/av_sync_test.py it
// looks like, and one line says when it ended. Cheap by construction: per frame a few additions into the cause
// windows and a comparison; the text is built only for a line; at most DESYNC_LINES_MAX lines a launch, so a machine
// that cannot keep up writes a few hundred bytes of log, not a line per scene.
#define DESYNC_MS 250.0
#define DESYNC_OK_MS 100.0              // back within this, the episode is over
#define DESYNC_SUSTAIN_NS 2000000000ull
#define DESYNC_LINES_MAX 12
#define CAUSE_WINDOW_NS 5000000000ull   // the cause is read over this window and the one before it

typedef struct { uint64_t start_ns; unsigned frames; double dt_sum, dt_max, dry_ms, still_ms; } CauseWin;
static CauseWin cause_cur, cause_prev;   // cutscene frames, game thread only
static unsigned desync_lines;

static void cause_note(const SoundLoss *l) {
    uint64_t now = kern_now_ns();
    if (!cause_cur.start_ns) cause_cur.start_ns = now;
    else if (now - cause_cur.start_ns > CAUSE_WINDOW_NS) cause_prev = cause_cur, cause_cur = (CauseWin){now, 0, 0, 0, 0, 0};
    cause_cur.frames++;
    cause_cur.dt_sum += l->dt_ms;
    if (l->dt_ms > cause_cur.dt_max) cause_cur.dt_max = l->dt_ms;
    cause_cur.dry_ms += l->dry_ms;
    cause_cur.still_ms += l->still_ms;
}

// The scenario of tools/av_sync_test.py the last seconds of cutscene frames look like, and their figures.
static const char *cause_case(char *fig, size_t cap) {
    unsigned n = cause_cur.frames + cause_prev.frames;
    double sum = cause_cur.dt_sum + cause_prev.dt_sum, mean = n ? sum / n : 0;
    double max = cause_cur.dt_max > cause_prev.dt_max ? cause_cur.dt_max : cause_prev.dt_max;
    double dry = cause_cur.dry_ms + cause_prev.dry_ms, still = cause_cur.still_ms + cause_prev.still_ms;
    snprintf(fig, cap, "%u cutscene frames in the last seconds, mean %.1f ms, longest %.0f ms, sound ran dry %.0f ms, "
             "sound stood still %.0f ms", n, mean, max, dry, still);
    if (!n) return "no cutscene frames to tell by (a movie outside a cutscene)";
    if (mean > 1000.0 / 15.0) return "very-slow";
    if (still >= 250.0) return "freeze";
    if (dry >= 100.0) return "sound-starved";
    if (max >= 1800.0) return "stutter-2500";
    if (max >= 600.0) return "stutter-900";
    if (max >= 250.0) return "stutter-300";
    if (max >= 60.0 && mean < 36.0) return "stutter-80";
    if (mean > 36.0) return "slow";
    return "none of the matrix's cases: frames on time, sound not starved";
}

typedef struct { uint64_t since_ns, began_ns; int active; } DesyncWatch;
enum { DESYNC_NONE, DESYNC_BEGIN, DESYNC_END };

// One reading of picture minus sound for a watch; returns DESYNC_BEGIN once an episode has lasted long enough to
// report (and the lines are not used up), DESYNC_END when a reported one is over. *secs: how long it lasted.
static int desync_step(DesyncWatch *w, double av_ms, double *secs) {
    double a = av_ms < 0 ? -av_ms : av_ms;
    if (a > DESYNC_MS) {
        if (w->active) return DESYNC_NONE;
        uint64_t now = kern_now_ns();
        if (!w->since_ns) {
            w->since_ns = now;
        } else if (now - w->since_ns >= DESYNC_SUSTAIN_NS && desync_lines < DESYNC_LINES_MAX) {
            w->active = 1, w->began_ns = w->since_ns, desync_lines++;
            *secs = (double)(now - w->since_ns) / 1e9;
            return DESYNC_BEGIN;
        }
    } else if (!w->active) {
        w->since_ns = 0;
    } else if (a < DESYNC_OK_MS) {
        *secs = (double)(kern_now_ns() - w->began_ns) / 1e9;
        w->active = 0, w->since_ns = 0;
        return DESYNC_END;
    }
    return DESYNC_NONE;
}

static void desync_report_begin(const char *where, double av_ms, double secs) {
    char fig[200];
    const char *kase = cause_case(fig, sizeof fig);
    rt_log("frame rate: audio desync: %s is %+.0f ms off its sound for %.1f s (wall %.3f s, %d fps mode); matrix case: "
           "%s (tools/av_sync_test.py); %s", where, av_ms, secs, wall_s(), fps_hz ? fps_hz : 60, kase, fig);
}

// over: 1 when the picture came back to its sound, 0 when it ended (the cutscene) still off it.
static void desync_report_end(const char *where, double av_ms, double secs, int back) {
    if (back)
        rt_log("frame rate: audio desync over: %s is back within %+.0f ms of its sound after %.1f s (wall %.3f s)",
               where, av_ms, secs, wall_s());
    else
        rt_log("frame rate: audio desync over: %s ended still %+.0f ms off its sound, %.1f s after it began (wall %.3f "
               "s)", where, av_ms, secs, wall_s());
}

// What the sound lost in the cutscene frame that ends here, counted in the run's log line.
static double cutscene_lost(int first) {
    SoundLoss l;
    double lost = sound_lost_ms(&cut_mark, first, &l);
    if (!first) cause_note(&l);
    cut_drop_ms += lost;
    return lost;
}

// The whole units (vsyncs at 60, 30 fps frames when held) a cutscene frame steps by, 0 to most, given the time it
// adds to what the cutscene owes (the frame's time less what the sound lost in it); what the units leave is the new
// debt.
//   - One unit while the cutscene owes less than two. Frame times that jitter around the unit, a late frame and the
//     early one the present pacer follows it with, then step evenly, as they are shown.
//   - The whole units owed from two up: a slow machine's extra unit every few frames, a hitch's catching up.
//   - None while the debt is negative: the picture is ahead, which only sound that came late brings about, and the
//     frame waits (CUT_WAIT_STEP).
// The debt stays within CUT_DEBT_MAX_MS either way; what a stall leaves above it is dropped and counted.
// Up to 1.12.2 the units were the owed time rounded, one at least, and a debt below minus one unit was cut off
// there. Every early frame after a late one could then put the picture ahead for good: with frame times that jitter,
// on a machine that does not hold the frame rate, a cutscene ran ahead of its sound, the more the longer it lasted
// (0.2 s in 40 s of frames 5 to 50 ms late).
static int cutscene_units(double owed_ms, double unit_ms, int most) {
    double d = g_cut_debt_ms + owed_ms;
    int n = d < 0 ? 0 : d < 2 * unit_ms ? 1 : (int)floor(d / unit_ms);
    if (n > most) n = most;
    d -= n ? n * unit_ms : CUT_WAIT_STEP * (1000.0 / 30.0);   // a frame that waits steps that much all the same
    if (d > CUT_DEBT_MAX_MS) {
        cut_drop_ms += d - CUT_DEBT_MAX_MS;
        d = CUT_DEBT_MAX_MS;
    }
    if (d < -CUT_DEBT_MAX_MS) d = -CUT_DEBT_MAX_MS;
    g_cut_debt_ms = d;
    if (n > most / 2 && owed_ms < (n - 0.5) * unit_ms) cut_catchup++;   // more units than the frame itself took
    if (d > cut_debt_peak) cut_debt_peak = d;
    return n;
}

static float cutscene_step(double dt_ms, int first) {
    if (first) {
        g_cut_debt_ms = 0;
        g_fps_rem = 0.5;   // a whole tick on the first frame and every other one after it (see above)
        dt_ms = dt_ms < CUT_VSYNC_MS ? CUT_VSYNC_MS : dt_ms > 2 * CUT_VSYNC_MS ? 2 * CUT_VSYNC_MS : dt_ms;
    }
    if (!(dt_ms >= 0.0)) dt_ms = CUT_VSYNC_MS;
    double lost = cutscene_lost(first);
    int n = cutscene_units(dt_ms - lost, CUT_VSYNC_MS, CUT_VSYNCS_MAX);
    return n ? (float)n * 0.5f : CUT_WAIT_STEP;
}

// The step of one frame of a cutscene held to 30 (30 fps units): 1.0, 2.0 once the cutscene is a whole 30 fps frame
// behind the frames' time, and a frame that waits while its sound came late (cutscene_units). The first frame of a
// run steps 1.0 whatever came before it. The movie in it gives up the same time the sound lost, on its own clock
// (rt_movie_late_w9), so it does not catch up what the cutscene and the sound did not play.
static float held_step(double dt_ms, int first) {
    if (first) {
        g_cut_debt_ms = 0;
        g_fps_rem = 0;
        dt_ms = HELD_FRAME_MS;
    }
    if (!(dt_ms >= 0.0)) dt_ms = HELD_FRAME_MS;
    double lost = cutscene_lost(first);
    int n = cutscene_units(dt_ms - lost, HELD_FRAME_MS, HELD_FRAMES_MAX);
    // Far behind (a long stall), the hold itself is what keeps the cutscene from catching up faster than one frame
    // per frame: the present pacer lets it present at 60 until it is back (frame_step_rush), each frame still
    // stepping 2.0 and the movie skipping a frame, so 2.5 s are made up in under a second instead of 2.5 s.
    if (first) held_rush = 0;
    else if (g_cut_debt_ms > HELD_RUSH_ON_MS) held_rush = 1;
    else if (g_cut_debt_ms < HELD_FRAME_MS) held_rush = 0;
    return n ? (float)n : CUT_WAIT_STEP;
}

int frame_step_rush(void) { return held_rush; }

// Seconds since the guest started, for the log lines that mark where things happen.
static double wall_s(void) { return rt_guest_start_ns ? (kern_now_ns() - rt_guest_start_ns) / 1e9 : 0.0; }

// ---- the cutscene log line: one per run of cutscene frames in one mode (at 60, or held to 30), at its end. Measured
// is the frame time the game's timer took, stepped what the frames moved the cutscene by (1/30 s per 1.0 of step); the
// two differ by the debt still open, the first frame's clamp and the time left out: what the sound stood still or
// came late in (cutscene_lost) and what a stall left above the debt's cap.
//
// Picture minus sound is what the run stepped against what the audio device played of the game's sound since the
// run's first frame (sound_played_ns: audio_mixer.c's device clock less its dry clock): how far the cutscene's picture is ahead of (+) or behind
// (-) anything the game streams through it, a voice track or music. It moves by a device period (about 10 ms) and a
// frame from one reading to the next; a run that keeps its sound stays within a few tens of ms of 0. Left out
// without an audio device. AC_TRACE=perf logs the same every 300 frames of a run, for a run the test ends in.
static unsigned run_frames;
static double run_wall_ms, run_game_ms;
static int run_at60;                              // the run steps at 60 (1) or is held to 30 (0)
static uint64_t run_sound0_ns;                    // the device clock at the run's first frame, 0 without a device
static double run_av_ms, run_av_lo, run_av_hi;    // picture minus sound at the last frame, and its range over the run
static DesyncWatch run_watch;                     // the run's desync episode, for the audio desync log

static void run_line(const char *what) {
    char av[112] = "";
    if (run_sound0_ns)
        snprintf(av, sizeof av, "; picture minus sound %+.0f ms, %+.0f..%+.0f ms over the run", run_av_ms, run_av_lo,
                 run_av_hi);
    rt_log("frame rate: cutscene run %s%s: %u frames, %.3f s measured, %.3f s stepped, %u catch-up frames, peak debt "
           "%.1f ms, %.1f ms left out%s",
           run_at60 ? "at 60" : "held to 30", what, run_frames, run_wall_ms / 1000.0, run_game_ms / 1000.0,
           cut_catchup, cut_debt_peak, cut_drop_ms, av);
}

static void cutscene_report(void) {
    if (!run_frames) return;
    if (run_watch.active) {   // the run ended out of sync: the episode ends with it
        double secs = (double)(kern_now_ns() - run_watch.began_ns) / 1e9;
        desync_report_end(run_at60 ? "the cutscene at 60 fps" : "the cutscene held to 30", run_av_ms, secs, 0);
    }
    run_watch = (DesyncWatch){0, 0, 0};
    run_line("");
    run_frames = cut_catchup = 0;
    run_wall_ms = run_game_ms = cut_drop_ms = cut_debt_peak = 0;
}

static uint64_t sound_played_ns(void) { return audio_device_clock_ns() - audio_device_dry_ns(); }

// One cutscene frame for the run line (first: the run starts with it, at 60 or held): the frame's reading of picture
// minus sound, taken before its own step is added (what the frames before it stepped, against the sound played since
// the run's first frame started), then the frame's step and time.
static void run_frame(int first, int at60, float step, double dt_ms) {
    uint64_t sound = sound_played_ns();
    if (first) {
        run_at60 = at60;
        run_sound0_ns = sound;
        run_av_ms = run_av_lo = run_av_hi = 0;
    } else if (run_sound0_ns) {
        run_av_ms = run_game_ms - (double)(sound - run_sound0_ns) / 1e6;
        if (run_av_ms < run_av_lo) run_av_lo = run_av_ms;
        if (run_av_ms > run_av_hi) run_av_hi = run_av_ms;
        double secs;
        int d = desync_step(&run_watch, run_av_ms, &secs);
        if (d == DESYNC_BEGIN) desync_report_begin(run_at60 ? "the cutscene at 60 fps" : "the cutscene held to 30",
                                                   run_av_ms, secs);
        else if (d == DESYNC_END) desync_report_end(run_at60 ? "the cutscene at 60 fps" : "the cutscene held to 30",
                                                    run_av_ms, secs, 1);
        if ((rt_trace & RT_TRACE_PERF) && run_frames % 300 == 0) run_line(" so far");
    }
    run_frames++;
    run_game_ms += step * (1000.0 / 30.0);
    run_wall_ms += first ? step * (1000.0 / 30.0) : dt_ms;
}

// The game's audio sync of an event with a voice track (main+0x8B9DC0 and its twin main+0x8C2E10): s0 = the track's
// position minus the picture's, in seconds (the cut frames are 1/59.94 s). The game turns more than 30 ms of it into
// the timer override, which only the step outside the cutscene flag takes, so the check may run with the flag off.
// Recorded either way (site: after the fadd that forms it) and logged once the checks stop for a second.
static unsigned sync_n, sync_flag_n;
static double sync_sum, sync_lo, sync_hi;
static uint64_t sync_last_ns;

void rt_cut_drift_s0(Cpu *c) {
    double ms = VSF32(0) * 1000.0;
    if (!sync_n || ms < sync_lo) sync_lo = ms;
    if (!sync_n || ms > sync_hi) sync_hi = ms;
    sync_sum += ms;
    sync_n++;
    sync_flag_n += in_cutscene != 0;
    sync_last_ns = kern_now_ns();
}

static void sync_report(void) {
    if (!sync_n || kern_now_ns() - sync_last_ns < 1000000000ull) return;
    rt_log("frame rate: voice sync (wall %.3f s): audio minus picture %+.1f ms mean, %+.1f..%+.1f ms over %u checks, "
           "%u of them with the cutscene flag set",
           wall_s(), sync_sum / sync_n, sync_lo, sync_hi, sync_n, sync_flag_n);
    sync_n = sync_flag_n = 0;
    sync_sum = 0;
}

// Testing: what a cutscene meets on the machines the port runs on, brought about on purpose so that the picture's
// hold on the sound can be measured (the cutscene run and movie log lines). All are read with the frame rate.
// AC_TEST_HITCH=<ms>[:<every ms>[:<random ms>]]: the game thread sleeps <ms>, plus up to <random ms> more, in a
// cutscene frame, once per <every ms> of wall time (every frame with 0 or without it): a machine too slow for the
// frame rate (every frame), or one that hitches.
// AC_TEST_FREEZE=<ms>[:<every ms>]: the game thread sleeps <ms> in a cutscene frame once per <every ms> (5000 without
// it) and the audio output stands still meanwhile (ov_audout.c audout_pull): a suspended process, a debugger break.
// ov_audout.c has AC_TEST_AUDIO_STALL, which holds the game's sound thread up alone.
static volatile uint32_t hitch_ms, hitch_every_ms, hitch_random_ms;   // frame_limit_live writes, the game thread reads
static volatile uint32_t freeze_ms, freeze_every_ms;
static uint64_t hitch_last_ns, freeze_last_ns;
static _Atomic uint64_t g_freeze_until_ns;   // the audio output stands still until then; 0 none

int frame_test_frozen(void) {
    uint64_t until = atomic_load_explicit(&g_freeze_until_ns, memory_order_relaxed);
    return until && kern_now_ns() < until;
}

static void test_hitch(void) {
    uint64_t now = kern_now_ns();
    if (freeze_ms) {
        if (!freeze_last_ns) freeze_last_ns = now;
        if (now - freeze_last_ns >= freeze_every_ms * 1000000ull) {
            freeze_last_ns = now;
            atomic_store_explicit(&g_freeze_until_ns, now + freeze_ms * 1000000ull, memory_order_relaxed);
            plat_sleep_ns(freeze_ms * 1000000ull);
            now = kern_now_ns();
        }
    }
    if (!hitch_ms) return;
    if (hitch_every_ms && hitch_last_ns && now - hitch_last_ns < hitch_every_ms * 1000000ull) return;
    hitch_last_ns = now;
    static uint32_t seed = 0x2545F491u;
    seed = seed * 1664525u + 1013904223u;
    uint32_t extra = hitch_random_ms ? (seed >> 8) % (hitch_random_ms + 1) : 0;
    plat_sleep_ns((hitch_ms + extra) * 1000000ull);
}

static int movie_now;   // a full-screen movie held the last cutscene frame (game thread, for the log)
static unsigned timer_switches;   // fps_switches at the last timer call (game thread only)
static int fixed30;   // at 30 fps, the game's own timer's last call was a cutscene frame (rt_fixed_step; game thread only)

void rt_fps_timer(Cpu *c) {
    uint8_t *t = (uint8_t *)(uintptr_t)c->x[19];
    int fixed = (uint32_t)c->x[21] == 0;
    double dt_ms = *(float volatile *)(t + 0xE0);
    atomic_fetch_add_explicit(&g_timer_frames, 1, memory_order_relaxed);
    // The 60 fps patch was switched since the last call (frame_limit_live, on the present path): the timer did not run
    // at 30, so what it knew of a cutscene is stale. The run it was in ends here, under the mode it ran in, and this
    // frame starts from gameplay. Done here so that only the game thread writes this state.
    unsigned switches = atomic_load_explicit(&fps_switches, memory_order_relaxed);
    if (switches != timer_switches) {
        timer_switches = switches;
        cutscene_report();
        in_cutscene = cut60_now = cut_exact_now = movie_now = fixed30 = held_rush = 0;
    }
    int movie = fixed && movie_playing();       // a full-screen movie keeps its cutscene at the 30 fps hold
    int cut60 = fixed && !movie;
    int ended = in_cutscene && (!fixed || cut60 != cut60_now);   // the run of cutscene frames before this one
    int first = fixed && (!in_cutscene || cut60 != cut60_now);   // a new run: its frame time is what came before it
    if (ended) cutscene_report();               // before this frame's step, which counts for the run it starts
    float step;
    if (fixed) {
        step = cut60 ? cutscene_step(dt_ms, first) : held_step(dt_ms, first);
        g_fps_debt_ms = 0;
    } else {
        double s = frame_paced_ms(dt_ms) * *(float volatile *)(t + 0xDC);
        float *ov = *(float *volatile *)((uintptr_t)MOD_MAIN_BASE + 0xDC0F10ull);   // main+0x8A9570's float
        if (ov && *ov > 0.0f) s *= *ov;
        step = s > 2.0 ? 2.0f : s < 1.0 / 16.0 ? 1.0f / 16.0f : (float)s;
    }
    *(float volatile *)(t + 0xD0) = step;
    *(uint32_t volatile *)(t + 0xD4) = 0;
    *(uint32_t volatile *)(t + 0xD8) = 0;
    *(uint32_t volatile *)(t + 0xE4) = 0;
    if (in_cutscene != fixed || cut60 != cut60_now || movie != movie_now)
        rt_log("frame rate: %s (world %.2f s, wall %.3f s): presents held to %d per second",
               !fixed ? "gameplay" : cut60 ? "cutscene at 60 fps" : movie ? "cutscene with a movie" : "cutscene",
               (double)frame_step_world_us() / 1e6, wall_s(), fixed && !cut60 ? 30 : 60);
    // frame_step_cutscene reads both on the present path: written in an order that never shows a cutscene at 60 as held
    if (fixed) cut60_now = cut60, in_cutscene = 1;
    else in_cutscene = 0, cut60_now = 0;
    if (!fixed || cut60) held_rush = 0;
    cut_exact_now = fixed && (cut60 || step != 1.0f);
    movie_now = movie;
    if (fixed) {
        run_frame(first, cut60, step, dt_ms);
        test_hitch();
    }
    sync_report();
    fps_publish(step);
}

// The timer's first call (no previous tick, site main+0x91DBDC): the game keeps the step it has, which is one whole
// frame for the counters.
void rt_fps_timer_first(Cpu *c) {
    if (c->x[8]) return;
    g_fps_rem = 0;
    g_fps_debt_ms = 0;
    uint8_t *t = (uint8_t *)(uintptr_t)c->x[19];
    float step = *(float volatile *)(t + 0xD0);
    fps_publish(step > 0.0f ? step : 1.0f);
}

// The frame time on the present grid. The game loop waits for its swapchain image (nvn_api_obj.c), so on average a
// frame lasts whole present periods, but the time the timer measures jitters by a millisecond or two around them. A
// step that follows the jitter moves objects by uneven amounts between evenly shown frames (judder). So the time is
// handed out in whole periods, and what rounding leaves is carried to the next frame: the sum stays the wall time.
// A frame outside a quarter to four and a half periods (a load, a debugger, a menu that does not present) is used as
// measured and drops the carry.
static double frame_paced_ms(double dt_ms) {
    double period = 1000.0 / (fps_hz ? fps_hz : 60);
    if (!(dt_ms >= 0.25 * period && dt_ms <= 4.5 * period)) {
        g_fps_debt_ms = 0;
        return dt_ms;
    }
    double d = g_fps_debt_ms + dt_ms;
    int n = (int)floor(d / period + 0.5);
    if (n < 1) n = 1;
    g_fps_debt_ms = d - n * period;
    if (g_fps_debt_ms < -period || g_fps_debt_ms > period) g_fps_debt_ms = 0;
    return n * period;
}

// ---- dynamic frame-rate guest helpers (called by @fn/@fnz patch sites in the lifted game code)

uint32_t rt_fps_ticks(void) { return atomic_load_explicit(&g_fps_ticks, memory_order_relaxed); }

float rt_fps_step(void) {
    uint32_t b = atomic_load_explicit(&g_fps_step_bits, memory_order_relaxed);
    float f; memcpy(&f, &b, 4); return f;
}

// The fcvtzs/fcvtzu step->int reader sites: x8 = the whole-tick count.
void rt_fps_ticks_w8(Cpu *c) { c->x[8] = rt_fps_ticks(); }

// The scaled decrement that replaces `subs wN, wN, #1` (and `subs w9, w10, #1`): decrement by the whole-tick count,
// clamped at 0, with NZCV matching what the 30 fps original produced. Flags: N = the counter was already 0 or below, Z =
// it reaches 0 this frame (counter <= ticks), C = it was not 0, V = 0. The stored value underflows when the counter is
// already 0 or below, as `subs wN,wN,#1` does, so a following `b.lt`/`b.ge` sees the same "expired" condition.
static uint64_t fps_hs(uint32_t w, uint32_t ticks, uint32_t *out) {
    if ((int32_t)w <= 0) {
        *out = (uint32_t)(w - ticks);
        return MKFLAGS(1, 0, w != 0, 0);
    }
    if (ticks == 0) {
        *out = w;
        return MKFLAGS(0, 0, 1, 0);
    }
    uint32_t m = w < ticks ? w : ticks;
    *out = w - m;
    return MKFLAGS(0, w <= ticks, 1, 0);
}

uint64_t rt_fps_hs_w8(Cpu *c) {
    uint32_t out; uint64_t f = fps_hs((uint32_t)c->x[8], rt_fps_ticks(), &out);
    c->x[8] = out; return f;
}
uint64_t rt_fps_hs_w9(Cpu *c) {
    uint32_t out; uint64_t f = fps_hs((uint32_t)c->x[9], rt_fps_ticks(), &out);
    c->x[9] = out; return f;
}
uint64_t rt_fps_hs_w10(Cpu *c) {
    uint32_t out; uint64_t f = fps_hs((uint32_t)c->x[10], rt_fps_ticks(), &out);
    c->x[10] = out; return f;
}
uint64_t rt_fps_hs_w10w9(Cpu *c) {
    uint32_t out; uint64_t f = fps_hs((uint32_t)c->x[10], rt_fps_ticks(), &out);
    c->x[9] = out; return f;
}

// `subs w9, w8, #1`: the scaled decrement of w8, written to w9 (the stick gesture parser's press windows).
uint64_t rt_fps_hs_w8w9(Cpu *c) {
    uint32_t out; uint64_t f = fps_hs((uint32_t)c->x[8], rt_fps_ticks(), &out);
    c->x[9] = out; return f;
}

// The stick gesture parser's sector stamp (`str w9, [x8, #0x1508]`, w9 = 12): the sector the stick points at stays live
// for 12 frames, and 6 live sectors make a rotation, which clears the stick reversal and turns the attack into the
// rotation attack. At 60 fps the stick is sampled twice as often, so a back-to-forward sweep along the rim stamped more
// sectors than at 30 and read as a rotation more often. Sectors are stamped only on frames with a whole tick: 30
// samples a second, as the game was tuned for.
void rt_fps_sector_str(Cpu *c) {
    if (rt_fps_ticks()) *(uint32_t volatile *)(uintptr_t)(c->x[8] + 0x1508) = (uint32_t)c->x[9];
}

// A float countdown of frames (`fmov s1, #-1.0` before `fadd s0, s0, s1`): s1 = minus the game's step, so it counts
// 30 fps frames of real time.
void rt_fps_neg_step_s1(Cpu *c) { FP_SETS(1, -rt_fps_step()); }

// `sub w8, w8, #1` (no flags), the r2 bank of tap/buffer counters: max(w8 - ticks, 0).
void rt_fps_sub_w8(Cpu *c) {
    uint32_t w = (uint32_t)c->x[8], t = rt_fps_ticks(), r = w - t;
    c->x[8] = (w > t) ? r : 0;
}

// `add w9, w9, #1` (no flags), the pad auto-repeat count: w9 + ticks. A repeat takes the interval off the count, which
// leaves it at or above the delay, and nothing fires at or below the delay. So a frame without a tick cannot fire twice,
// unless the interval is 0.
void rt_fps_up_w9(Cpu *c) { c->x[9] = (uint32_t)((uint32_t)c->x[9] + rt_fps_ticks()); }

// The hitstop manager's freeze phase: `sub w9,w9,#1` runs only at 2 or more and stays at 1 or above, and `add w8,w8,#1`
// counts its length. Neither writes flags, as the instructions they replace do not. Its ramp adds a constant a frame:
// `ldr s1,<constant>` is followed by the same constant times the ticks.
void rt_fps_floor1_w9(Cpu *c) {
    int32_t r = (int32_t)(uint32_t)c->x[9] - (int32_t)rt_fps_ticks();
    c->x[9] = (uint32_t)(r < 1 ? 1 : r);
}
void rt_fps_up_w8(Cpu *c) { c->x[8] = (uint32_t)((uint32_t)c->x[8] + rt_fps_ticks()); }
void rt_fps_ramp_s1(Cpu *c) { FP_SETS(1, VSF32(1) * (float)rt_fps_ticks()); }

// The tap/hold count, `add w9,w9,#1` while a button is held: on release 1 to 9 is a tap and 10 or more a hold, and 0 is
// neither, so the first held frame always counts even without a tick. The lockout count, `add w10,w9,#1`, only tests its
// old value against 15. Neither writes flags, as the instructions they replace do not.
void rt_fps_up_min1_w9(Cpu *c) {
    uint32_t w = (uint32_t)c->x[9], t = rt_fps_ticks();
    c->x[9] = (uint32_t)(w + (w == 0 && t == 0 ? 1 : t));
}
void rt_fps_up_w9w10(Cpu *c) { c->x[10] = (uint32_t)((uint32_t)c->x[9] + rt_fps_ticks()); }

// The rumble counters (`subs w9,w8,#1` and `subs w10,w9,#1`): as fps_hs, but a counter at 1 always reaches 0, so a tick
// of 0 cannot keep a channel at its last count on the frame that would stop it. A frame without a tick repeats the same
// strength, which the game blends from the old count.
static uint64_t fps_rumble(uint32_t w, uint32_t *out) { return fps_hs(w, w == 1 ? 1 : rt_fps_ticks(), out); }
uint64_t rt_fps_rumble_w8w9(Cpu *c) {
    uint32_t out; uint64_t f = fps_rumble((uint32_t)c->x[8], &out);
    c->x[9] = out; return f;
}
uint64_t rt_fps_rumble_w9w10(Cpu *c) {
    uint32_t out; uint64_t f = fps_rumble((uint32_t)c->x[9], &out);
    c->x[10] = out; return f;
}

// The per-object clock scale's counters in x16 (the player's deferred hitstop arms them through main+0x91D6A0). The delay
// is the usual scaled decrement. The duration (`subs w16,w16,#1; str w16; b.gt`) is a plain subtraction of the ticks with
// its real flags: the game ends it at or below 0.
uint64_t rt_fps_hs_w16(Cpu *c) {
    uint32_t out; uint64_t f = fps_hs((uint32_t)c->x[16], rt_fps_ticks(), &out);
    c->x[16] = out; return f;
}
uint64_t rt_fps_duration_w16(Cpu *c) {
    uint32_t w = (uint32_t)c->x[16], t = rt_fps_ticks(), r = w - t;
    c->x[16] = r;
    return MKFLAGS(r >> 31, r == 0, w >= t, ((w ^ t) & (w ^ r)) >> 31);
}

// Camera smoothing: s8 = 1 - powf(1 - a, clamp(step, 0, 4)), the per-frame blend a rescaled to per-unit-time. a <= 0,
// a >= 1, NaN and step == 1.0 all pass through unchanged.
static float fps_blend(float a) {
    float step = rt_fps_step();
    if (step == 1.0f) return a;
    if (!(a > 0.0f) || !(a < 1.0f)) return a;
    float s = step < 0.0f ? 0.0f : (step > 4.0f ? 4.0f : step);
    return 1.0f - powf(1.0f - a, s);
}

void rt_fps_cam_s0(Cpu *c) {
    float m = VSF32(0), s10 = VSF32(10);
    FP_SETS(8, fps_blend(m < s10 ? m : s10));
}
void rt_fps_cam_s1(Cpu *c) {
    float m = VSF32(1), s10 = VSF32(10);
    FP_SETS(8, fps_blend(m < s10 ? m : s10));
}
void rt_fps_cam_fcsel(Cpu *c) {
    float s10 = VSF32(10);
    FP_SETS(9, fps_blend(VSF32(9)));   // the smoothed s9, for the +0x100/+0x108 blend
    float t = *(float volatile *)(c->x[19] + 0x214ull);   // re-test [cam+0x214] > 0
    FP_SETS(8, t > 0.0f ? s10 : VSF32(9));
}

// The dialogue text reveal (main+0x615FFC): [x19+0xA8] += [x19+0xAC] a frame, and a character shows each time it
// passes 1. A talk line ends 0.5 s after both its text and its voice are done, so at 60 fps the text ran twice as fast
// and lines whose text outlasts the voice ended early, in gameplay as well as cutscenes. s2 = the rate times the game's
// step (1.0 in a cutscene held to 30 presents per second, the real time in one at 60).
// The talk line's state machine (main+0x6161D0, [x19+0xB0] = 0..3, anything above returns at once), run every frame by
// the line's update. It times nothing itself: it polls what is timed (the voice, the text box's layout animations,
// which follow the step) and moves on a frame after it sees a change: 33 ms at 30, 17 ms at 60. With the talk commands
// below it made each line of a cutscene at 60 come about 50 ms sooner, 1.3 s over a 124 s scene. In a cutscene at 60
// the machine now runs only on frames with a whole 30 fps tick: w8 = 4 (nothing to do) on the others, so it sees the
// changes 30 times a second, as the game was built for.
void rt_fps_talk_state_w8(Cpu *c) {
    c->x[8] = cut60_now && !rt_fps_ticks() ? 4u : *(uint32_t volatile *)(uintptr_t)(c->x[19] + 0xB0);
}

// The event script's talk command (main+0x443E50, an entry of the event command table at main+0xAB2720): the script
// calls it every frame until it returns 1. It polls the line (voice done, text box state) and moves on a frame after
// it sees a change, and its waits are countdowns: [x19+0x84], 0.5 s after a line, adds step * -1/30 a call (at 30 fps
// -1/30 rounds to -0.033333335 in single precision, so 0.5 is still 2e-8 after 15 frames and the wait lasts 16 frames,
// 0.533 s), and [x19+0x7C] counts down the same way (main+0x443F90). At 60 the command ran twice as often: each
// line's waits and reactions came sooner, and a 124 s scene ended 1.3 s early. In a cutscene at 60 it runs only on
// frames with a whole 30 fps tick, and its step-based countdowns take the whole ticks: the same calls, decrements and
// rounding as at 30. On the other frames it returns 0 (still running) at once and leaves the scene's per-frame talking
// flag ([[x19+8]+0x9F0], which it clears on entry and sets while a line plays) as the last run left it.
static int talk_skip(void) { return cut60_now && !rt_fps_ticks(); }

void rt_fps_talk_clear(Cpu *c) {   // str wzr, [x8, #0x9f0]: the talking flag, cleared on a run
    if (!talk_skip()) *(uint32_t volatile *)(uintptr_t)(c->x[8] + 0x9F0) = 0;
}
void rt_fps_talk_busy_w8(Cpu *c) {   // ldr w8, [x24, #0x28]: nonzero returns at once; 1 on a skipped frame
    c->x[8] = talk_skip() ? 1u : *(uint32_t volatile *)(uintptr_t)(c->x[24] + 0x28);
}
// mov w20, #1: that return's value, 0 (still running) on a skipped frame. A helper may only change caller-saved
// registers, so it sets w8 (dead on the way out) and the site's `mov w20, w8` copies it.
void rt_fps_talk_ret_w8(Cpu *c) { c->x[8] = talk_skip() ? 0u : 1u; }

// The voiced talk command (main+0x446C40, with its variant main+0x447600 behind [x19+0x4C]): while its voice
// ([x19+0x60]) is still in the sound engine it waits, and then counts [x19+0x68] / [x19+0x70] down from 0.5 s by
// step * -1/30 (main+0x4473DC, main+0x447414, main+0x4477FC), the same countdown as the talk command's. It is gated
// the same way: in a cutscene at 60, on frames without a whole tick, it reads its done flag ([x19+0x44], which it sets
// itself at main+0x4472BC and main+0x447578 and returns 1 on at its next call) and its variant switch as 0 and its
// state as 4, which together return 0 (still running) at once; its countdowns take the whole ticks. The scene measured
// does not use it, so this follows from the code, not from a measurement.
void rt_fps_voiced_done_w8(Cpu *c) {   // ldr w8, [x19, #0x44]
    c->x[8] = talk_skip() ? 0u : *(uint32_t volatile *)(uintptr_t)(c->x[19] + 0x44);
}
void rt_fps_voiced_alt_w8(Cpu *c) {   // ldr w8, [x19, #0x4c]
    c->x[8] = talk_skip() ? 0u : *(uint32_t volatile *)(uintptr_t)(c->x[19] + 0x4C);
}
void rt_fps_voiced_state_w8(Cpu *c) {   // ldr w8, [x19, #0x24]
    c->x[8] = talk_skip() ? 4u : *(uint32_t volatile *)(uintptr_t)(c->x[19] + 0x24);
}

// The other talk command (main+0x446300): [x19+0x24] = 0 starts the line (main+0x615000), 1 polls the text box and
// the voice until the line is over and returns 1; any other state returns 0 (still running) at once. In a cutscene at
// 60 it reads 2 on frames without a whole 30 fps tick, so it starts and ends lines on the frames it would at 30.
void rt_fps_msg_state_w8(Cpu *c) {
    c->x[8] = talk_skip() ? 2u : *(uint32_t volatile *)(uintptr_t)(c->x[19] + 0x24);
}

void rt_fps_talk_wait_s1(Cpu *c) {   // ldr s1, [x8|x9, #0xd0] (the step) at a countdown: the whole ticks at 60
    FP_SETS(1, cut60_now ? (float)rt_fps_ticks() : rt_fps_step());
}

// In a cutscene at 60 the rate is times the whole 30 fps ticks instead: the same additions as at 30, so a line's text
// finishes on the same 30 fps frame (rate * 0.5 rounds differently, and some lines finished a frame early).
void rt_fps_text_rate_s2(Cpu *c) {
    FP_SETS(2, VSF32(2) * (cut60_now ? (float)rt_fps_ticks() : rt_fps_step()));
    // AC_TRACE=perf: the start of each line's text in a cutscene ([x19+0xA0] the characters shown, +0xA4 its length)
    if (!(rt_trace & RT_TRACE_PERF)) return;
    static uintptr_t line_obj;
    static uint32_t line_shown, line_n;
    uintptr_t obj = (uintptr_t)c->x[19];
    uint32_t shown = *(uint32_t volatile *)(obj + 0xA0), len = *(uint32_t volatile *)(obj + 0xA4);
    if (in_cutscene && (obj != line_obj || shown < line_shown))
        rt_log("frame rate: line %u (%u characters) starts at %.3f s stepped, wall %.3f s", ++line_n, len,
               run_game_ms / 1000.0, wall_s());
    line_obj = obj, line_shown = shown;
}

// The frame-counted cue scheduler (main+0x6A3EC0, run while the mode word at GOT main+0xDBF4A0 is 1; mode 2 schedules
// by the music time instead). Each call adds 1 to a frame counter and fires every table entry whose frame equals it,
// so at 60 calls per second the cues come twice as fast. The counter now adds the whole ticks, and an entry fires when
// its frame is in (old counter, new counter]: a frame without a tick fires nothing (an equal compare would fire the
// same cues again) and a frame of 2 ticks fires the frame it skipped.
static int32_t g_cue_prev;   // the counter before this call's ticks (game thread only)

void rt_fps_cue_up_w8(Cpu *c) {
    g_cue_prev = (int32_t)(uint32_t)c->x[8];
    c->x[8] = (uint32_t)((uint32_t)c->x[8] + rt_fps_ticks());
}
// `cmp w9, w8` (w9 = the entry's frame, w8 = the counter), read by b.ne: Z when the entry is due this call.
uint64_t rt_fps_cue_cmp(Cpu *c) {
    int32_t f = (int32_t)(uint32_t)c->x[9], now = (int32_t)(uint32_t)c->x[8];
    return MKFLAGS(0, f > g_cue_prev && f <= now, 1, 0);
}

// The cut clock (main+0x8C2410, one call per clock and frame): time in the current cut += rate * step * speed, where
// rate (s8) is 1 or 2 by the data's authored frame rate and the cut lasts len = w24 units (x19 = the clock it
// writes, [x19] the cut, [x19+4] the time; x20 = the clock it read). Past len - rate it shows len - rate once (the
// hold) and then starts the next cut at 0. At step 1.0 the time lands on len - rate exactly, so every frame of the cut
// shows once. At step 0.5 the hold would end each cut one 60 fps frame early (1/60 s per cut, the picture pulling ahead
// of the voices), and after a hitch's larger step the next cut would start at 0 and lose what the step went past the
// cut's end. On a frame of a cutscene at 60, and on a frame of a held cutscene that steps 2.0 to make up for late ones
// (cut_exact_now), the end test becomes: the cut goes on while its time is below len, and the next cut starts at what
// the time went past len. At step 1.0 that is the original's sequence, so a held cutscene's frames of 1.0 keep the
// game's own test, and the two tests take over from each other at any frame: after the original's hold the time is
// len - rate, from where the exact test goes past len; after an exact frame it is on the original's grid.
void rt_cut_hold_s1(Cpu *c) {   // fsub s1, s1, s8 (s1 = len): the time the cut may reach and still go on
    float len = VSF32(1);
    FP_SETS(1, cut_exact_now ? nextafterf(len, -INFINITY) : len - VSF32(8));
}
void rt_cut_prev_s0(Cpu *c) {   // ldr s0, [x20, #4]: the time before this frame; the hold runs while it is below s1
    FP_SETS(0, cut_exact_now ? VSF32(1) : *(float volatile *)(uintptr_t)(c->x[20] + 4));
}
void rt_cut_next_s1(Cpu *c) {   // fmov s1, wzr: where the next cut starts
    float start = 0.0f;
    if (cut_exact_now) {
        float over = *(float volatile *)(uintptr_t)(c->x[19] + 4) - (float)(int32_t)(uint32_t)c->x[24];
        float most = 2.0f * VSF32(8);   // the step's high clamp, 2.0, times the rate
        start = over > 0.0f ? (over < most ? over : most) : 0.0f;
    }
    FP_SETS(1, start);
}

// AC_TRACE=perf, diagnostics: the scene clock after its update and the event's commit (main+0x8B7CB0, x19 = the event
// object, +0x20 the cut and time it kept, +0x28 the cut before). Logs each committed cut change and each stretch the
// clock stood still (a stop point waiting for a line), with where it stood. The event's commit (main+0x8C2590) applies
// the stop points, not the cut clock's end test: in the scene measured both modes stood at the same cut times.
void rt_cut_state(Cpu *c) {
    if (!(rt_trace & RT_TRACE_PERF) || !in_cutscene) return;
    static struct { uintptr_t obj; int32_t idx; float t; unsigned still; double since; } k[4];
    uintptr_t obj = (uintptr_t)c->x[19];
    int32_t idx = *(int32_t volatile *)(obj + 0x20);
    float t = *(float volatile *)(obj + 0x24);
    int i = 0;
    while (i < 4 && k[i].obj && k[i].obj != obj) i++;
    if (i == 4) return;
    if (!k[i].obj) {
        k[i].obj = obj, k[i].idx = idx, k[i].t = t;
        return;
    }
    if (idx == k[i].idx && t == k[i].t) {
        if (!k[i].still++) k[i].since = run_game_ms / 1000.0;
        return;
    }
    if (k[i].still)
        rt_log("frame rate: scene %06llx stood at cut %d, %.2f for %u frames (%.3f s) from %.3f s stepped",
               (unsigned long long)(obj & 0xFFFFFF), k[i].idx, k[i].t, k[i].still, run_game_ms / 1000.0 - k[i].since,
               k[i].since);
    if (idx != k[i].idx)
        rt_log("frame rate: scene %06llx cut %d (%.2f) -> cut %d (%.2f) at %.3f s stepped, wall %.3f s",
               (unsigned long long)(obj & 0xFFFFFF), k[i].idx, k[i].t, idx, t, run_game_ms / 1000.0, wall_s());
    k[i].idx = idx, k[i].t = t, k[i].still = 0;
}

// ---- full-screen movies: the game's Bink player (the class whose open is main+0x9BEFD0) plays both the event movies
// (movie/ev%04x.bk2: the opening, the chapter movies, the title screen's background) and the effect movies
// (mv/mv%04x.bk2, textures in the scene). The event movies are 1600x900 or 1280x720 Bink files at 30 fps on Bink's own
// clock (BinkWait, main+0x7003E0). The game opens most of them from a file handle, so there is no name to tell them
// by, but their size does: of the effect movies only four short ones (mv0065, mv0504, mv0510, mv0511) are as large,
// and the rest are 1280x460 or smaller. While one plays in a cutscene, the cutscene keeps the 30 fps hold
// (rt_fps_timer): Bink shows a new frame 30 times a second, and decoding 1600x900 on the game thread made a third of
// 60 fps frames late in the opening movie. Outside cutscenes movies are not held (frame_step_cutscene), so the title
// screen's menus stay at 60. The player's BinkOpen call returns the handle (main+0x9BF058, x0), BinkClose
// (main+0x9BF26C) and BinkWait (three sites) take it in x0. The sites are a set of their own (patches-v101.txt,
// movies), on at 30 fps too, so a movie opened before a switch to 60 is known.
//
// Those four effect movies play once on screens in the scene, and the game keeps them open afterwards for as long as
// the scene lasts, calling BinkWait every frame on a movie that stays at its last frame (three of them are open at
// once on the motorway the game starts on). They hold a cutscene like any movie of their size, but they are not the
// picture, and rt_movie_showing leaves them out. With no name to go by, their frame counts tell them from the
// 1280x720 event movies (movie_on_screen).
//
// Whatever the movie, one that has stayed at its last frame for MOVIE_ENDED_PRESENTS presents is over (rt_movie_wait):
// it no longer holds a cutscene nor sets the picture's shape, although it is still open and waited on. A looping movie
// leaves its last frame at once, and a full-screen one is closed when it ends.
//
// Keeping a movie with its sound. An event movie has no sound track of its own: the game's sound engine
// plays its sound, on the audio device's clock, and Bink shows a frame once its own timer says the frame is due
// (BinkWait: the timer since the movie's sync point against the frames played since then). Bink never skips a frame
// of a movie without sound tracks (BinkShouldSkip answers 0 for one), and a cutscene with a movie presents at most 30
// times a second, which is the movie's own rate: a frame shown late stayed late for the rest of the movie, the delays
// added up, and a movie more than four frames late took a new sync point where it was (BinkWait's test against twice
// BINK_TWO_FRAMES), which made the delay final. Now:
//   - rt_movie_late_w9 (BinkWait's subtraction, main+0x7004D4) records how late the frame BinkWait asks about is, and
//     keeps BinkWait from taking a new sync point while that is at most MOVIE_LATE_MAX_MS;
//   - rt_movie_skip_w0 (the player's read of BinkShouldSkip's answer, main+0x9BF760) has the player decode one frame
//     more and show that one, by the game's own code for a skipped frame, once the movie is a frame and a half late:
//     the movie gains a frame per game frame until it is back with its clock.
// A movie later than MOVIE_LATE_MAX_MS gives up what is beyond that, as the cutscene around it does with its debt:
// its sync point moves later by as much, and the movie stays behind its sound by it; the log names it. The time the
// sound itself lost (sound_lost_ms: it stood still, or came late) moves the sync point later too, in a cutscene or
// out of one, so the movie waits for its sound as the cutscene around it does (held_step).
#define MOVIE_SLOTS 4
#define MOVIE_MIN_PIXELS (1280u * 720u)
#define MOVIE_ENDED_PRESENTS 60   // 1 s at 60 fps, 2 s in a cutscene held to 30
static _Atomic uintptr_t g_movies[MOVIE_SLOTS];   // the open movies' Bink handles, those of MOVIE_MIN_PIXELS or more
static _Atomic int g_movie_screen[MOVIE_SLOTS];   // each one plays on a screen in the scene, not as the picture
// Each one's present count plus one (g_presents) at the first BinkWait that found it at its last frame, 0 while it is
// before that frame; and whether it is over.
static _Atomic uint64_t g_movie_last[MOVIE_SLOTS];
static _Atomic int g_movie_over[MOVIE_SLOTS];
static _Atomic uint32_t g_movie_loop[MOVIE_SLOTS];   // each one's player's loop word when it opened (for the log)
static _Atomic uint64_t g_movie_wait_frame;      // the game frame (g_timer_frames) of the last BinkWait on one; 0 none
// The same in presents, which count at 30 fps too (rt_fps_timer runs with the 60 fps patch only): the present count
// plus one at the last open or BinkWait of a movie that is the picture, 0 none (rt_movie_showing).
static _Atomic uint64_t g_presents, g_movie_wait_present;

// Bink's own fields in HBINK (the Bink 2 linked into main), as BinkWait (main+0x7003E0) reads them.
#define BINK_PLAYED 0x630       // frames played since the movie opened; counts on through a loop
#define BINK_SYNC_TIME 0x644    // Bink's timer (ms) at the sync point; 0: the next BinkWait takes one
#define BINK_TWO_FRAMES 0x64C   // two frames in ms, rounded down
#define MOVIE_LATE_MAX_MS ((int32_t)CUT_DEBT_MAX_MS)   // what a late movie catches up, as the cutscene around it

static _Atomic int32_t g_movie_late[MOVIE_SLOTS];       // how late the frame the last BinkWait asked about was, ms

// Each movie's hold on its sound, for the log. Picture minus sound is the frames played since a reference against
// what the audio device played of the game's sound since then (audio_mixer.c audio_device_clock_ns): how far the
// movie is ahead of (+) or behind (-) its sound, as in the cutscene run line. The reference is taken again, with the
// measure kept, at each of Bink's sync points (the start, the end of a pause), except one Bink took because the movie
// was too late, whose delay belongs in the measure. Written on the thread that plays the movie; rt_movie_close reads.
static struct MovieStat {
    SoundMark mark;               // the last BinkWait that looked at the movie's clock, for what the sound lost since
    double lost_sound_ms;         // of that, the part of a ms not yet taken off the movie's clock
    uint32_t sync;                // BINK_SYNC_TIME the reference was taken at, 0 none
    DesyncWatch watch;            // the movie's desync episode, for the audio desync log
    uint32_t played0;             // BINK_PLAYED at the reference
    uint64_t sound0_ns;           // the device clock at the reference, 0 without a device
    double av0_ms;                // picture minus sound at the reference
    double av_ms, av_lo, av_hi;   // picture minus sound at the last BinkWait, and its range since the movie opened
    uint32_t skipped;             // frames skipped to catch up
    uint32_t resyncs;             // times the movie was too late to catch all of it up
    double lost_ms;               // the delay given up then
    uint32_t traced;              // AC_TRACE=perf: BINK_PLAYED at the last line
} g_movie_stat[MOVIE_SLOTS];

static int movie_slot(uintptr_t h) {
    if (h)
        for (int i = 0; i < MOVIE_SLOTS; i++)
            if (atomic_load_explicit(&g_movies[i], memory_order_relaxed) == h) return i;
    return -1;
}

static uint32_t bink32(uintptr_t h, unsigned off) { return *(uint32_t volatile *)(h + off); }

// One frame's time in ms, from the header's frame rate (+0x14) and its divisor (+0x18).
static double movie_frame_ms(const uint32_t *hb) { return hb[5] ? 1000.0 * hb[6] / hb[5] : 1000.0 / 30.0; }

// The effect movies of MOVIE_MIN_PIXELS or more, by their Bink header: mv0065 (379 frames), mv0504 (150), mv0510
// (137) and mv0511 (157), all 1280x720. The 1280x720 event movies have 301, 1476, 1900 and 2573 frames.
static int movie_on_screen(const uint32_t *hb) {
    return hb[0] == 1280 && hb[1] == 720 && (hb[2] == 379 || hb[2] == 150 || hb[2] == 137 || hb[2] == 157);
}

// A movie was opened or waited for its next frame: the hold counts from here, and the picture's shape too unless the
// movie plays on a screen in the scene.
static void movie_seen(int screen) {
    atomic_store_explicit(&g_movie_wait_frame, atomic_load_explicit(&g_timer_frames, memory_order_relaxed),
                          memory_order_relaxed);
    if (!screen)
        atomic_store_explicit(&g_movie_wait_present, atomic_load_explicit(&g_presents, memory_order_relaxed) + 1,
                              memory_order_relaxed);
}

// x0 = the handle BinkOpen returned, x20 = the player: its word at 0x118 is non-zero for a movie that loops
// (main+0x9BF6F0 then skips its end-of-movie test), set before the call. HBINK starts with the width, the height, the
// frame count and the current frame's number, which counts from 1.
void rt_movie_opened(Cpu *c) {
    uintptr_t h = (uintptr_t)c->x[0];
    if (!h) return;
    const uint32_t *hb = (const uint32_t *)h;
    if ((uint64_t)hb[0] * hb[1] < MOVIE_MIN_PIXELS) return;
    int screen = movie_on_screen(hb);
    uint32_t loop;
    memcpy(&loop, (const uint8_t *)(uintptr_t)c->x[20] + 0x118, sizeof loop);
    for (int i = 0; i < MOVIE_SLOTS; i++) {
        uintptr_t none = 0;
        if (atomic_compare_exchange_strong(&g_movies[i], &none, h)) {
            atomic_store(&g_movie_screen[i], screen);
            atomic_store(&g_movie_last[i], 0);
            atomic_store(&g_movie_over[i], 0);
            atomic_store(&g_movie_loop[i], loop);
            atomic_store(&g_movie_late[i], 0);
            memset(&g_movie_stat[i], 0, sizeof g_movie_stat[i]);
            // It counts as playing from here on: the timer ran before the player in this frame, so without this the
            // frames up to the first BinkWait would step at 60.
            movie_seen(screen);
            rt_log("frame rate: movie opened, %ux%u, %u frames, %s%s (slot %d, wall %.3f s); a cutscene holds 30 fps "
                   "while it plays", hb[0], hb[1], hb[2], loop ? "looping" : "played once",
                   screen ? ", on a screen in the scene" : "", i, wall_s());
            return;
        }
    }
    // BinkClose has one caller in main (the site at main+0x9BF26C), so a slot is never left behind; this is more
    // movies of that size open at once than the table holds, and this one plays without the hold.
    rt_log("frame rate: movie opened, %ux%u, %u frames, but %d movies of that size are open already: not tracked",
           hb[0], hb[1], hb[2], MOVIE_SLOTS);
}

// The last movie closing ends the hold at once; closing one of two (the opening's logo, before the opening) keeps it.
// The picture's shape likewise, with the last movie that is the picture.
void rt_movie_close(Cpu *c) {
    uintptr_t h = (uintptr_t)c->x[0];
    int closed = -1, open = 0, picture = 0;
    for (int i = 0; i < MOVIE_SLOTS; i++) {
        uintptr_t want = h;
        if (h && atomic_compare_exchange_strong(&g_movies[i], &want, 0)) closed = i;
        else if (atomic_load(&g_movies[i])) open++, picture |= !atomic_load(&g_movie_screen[i]);
    }
    if (closed < 0) return;
    if (!open) atomic_store(&g_movie_wait_frame, 0);
    if (!picture) atomic_store(&g_movie_wait_present, 0);
    const uint32_t *hb = (const uint32_t *)h;   // BinkClose runs after this
    const struct MovieStat *s = &g_movie_stat[closed];
    char kept[200] = "";
    if (s->sound0_ns)
        snprintf(kept, sizeof kept, "; %u frames skipped to keep up, %u delays too long gave up %.0f ms; picture "
                 "minus sound %+.0f ms, %+.0f..%+.0f ms while it played", s->skipped, s->resyncs, s->lost_ms,
                 s->av_ms, s->av_lo, s->av_hi);
    rt_log("frame rate: movie closed, %ux%u at frame %u of %u%s (slot %d, %d still open, wall %.3f s)%s", hb[0], hb[1],
           hb[3], hb[2], atomic_load(&g_movie_over[closed]) ? ", over" : "", closed, open, wall_s(), kept);
}

// The open movies for a log line, each as "1280x720 frame 137/137 once screen over": its size, where it is, whether
// it loops or is played once, whether it is the picture or a screen in the scene, and "over" once it ended. Any
// thread; the handles are read as they are, for the log only.
void rt_movie_list(char *buf, size_t cap) {
    size_t n = 0;
    if (cap) buf[0] = 0;
    for (int i = 0; i < MOVIE_SLOTS; i++) {
        const uint32_t *hb = (const uint32_t *)atomic_load(&g_movies[i]);
        if (!hb || n >= cap) continue;
        int k = snprintf(buf + n, cap - n, "%s%ux%u frame %u/%u %s %s%s", n ? "; " : "", hb[0], hb[1], hb[3], hb[2],
                         atomic_load(&g_movie_loop[i]) ? "loop" : "once",
                         atomic_load(&g_movie_screen[i]) ? "screen" : "picture",
                         atomic_load(&g_movie_over[i]) ? " over" : "");
        if (k > 0) n += (size_t)k;
    }
    if (!n && cap) snprintf(buf, cap, "none open");
}

// The player is about to call BinkWait (x0 = the movie).
void rt_movie_wait(Cpu *c) {
    uintptr_t h = (uintptr_t)c->x[0];
    for (int i = 0; i < MOVIE_SLOTS; i++)
        if (h && atomic_load_explicit(&g_movies[i], memory_order_relaxed) == h) {
            const uint32_t *hb = (const uint32_t *)h;
            // not late until this BinkWait says so (rt_movie_late_w9): it has paths that do not ask
            atomic_store_explicit(&g_movie_late[i], 0, memory_order_relaxed);
            if (hb[3] < hb[2]) {
                atomic_store_explicit(&g_movie_last[i], 0, memory_order_relaxed);
                if (atomic_exchange_explicit(&g_movie_over[i], 0, memory_order_relaxed))
                    rt_log("frame rate: the %ux%u movie of %u frames plays again, at frame %u (slot %d, wall %.3f s)",
                           hb[0], hb[1], hb[2], hb[3], i, wall_s());
            } else {
                uint64_t now = atomic_load_explicit(&g_presents, memory_order_relaxed) + 1;
                uint64_t since = atomic_load_explicit(&g_movie_last[i], memory_order_relaxed);
                if (!since) atomic_store_explicit(&g_movie_last[i], since = now, memory_order_relaxed);
                if (now - since > MOVIE_ENDED_PRESENTS) {
                    if (!atomic_exchange_explicit(&g_movie_over[i], 1, memory_order_relaxed))
                        rt_log("frame rate: the %ux%u movie of %u frames is over, at its last frame for %d presents "
                               "(slot %d, wall %.3f s)", hb[0], hb[1], hb[2], MOVIE_ENDED_PRESENTS, i, wall_s());
                    return;
                }
            }
            movie_seen(atomic_load_explicit(&g_movie_screen[i], memory_order_relaxed));
            return;
        }
}

// One reading of a movie's hold on its sound, at a BinkWait that looked at the movie's clock.
static void movie_sample(int i, uintptr_t h) {
    struct MovieStat *s = &g_movie_stat[i];
    const uint32_t *hb = (const uint32_t *)h;
    uint32_t sync = bink32(h, BINK_SYNC_TIME), played = bink32(h, BINK_PLAYED);
    uint64_t sound = sound_played_ns();
    if (sync != s->sync) {   // a new sync point (the start, the end of a pause): the measure starts again from here
        s->sync = sync;
        s->played0 = played, s->sound0_ns = sound, s->av0_ms = s->av_ms;
        s->traced = played;
        return;
    }
    if (!s->sound0_ns) return;
    s->av_ms = s->av0_ms + (double)(played - s->played0) * movie_frame_ms(hb) - (double)(sound - s->sound0_ns) / 1e6;
    if (s->av_ms < s->av_lo) s->av_lo = s->av_ms;
    if (s->av_ms > s->av_hi) s->av_hi = s->av_ms;
    double secs;
    int d = desync_step(&s->watch, s->av_ms, &secs);
    if (d) {
        char where[64];
        snprintf(where, sizeof where, "the movie in slot %d (frame %u of %u)", i, hb[3], hb[2]);
        if (d == DESYNC_BEGIN) desync_report_begin(where, s->av_ms, secs);
        else desync_report_end(where, s->av_ms, secs, 1);
    }
    if ((rt_trace & RT_TRACE_PERF) && played - s->traced >= 150) {
        s->traced = played;
        rt_log("frame rate: movie slot %d at frame %u of %u: picture minus sound %+.0f ms, %+.0f..%+.0f ms so far; %u "
               "frames skipped to keep up, %u delays too long gave up %.0f ms (wall %.3f s)", i, hb[3], hb[2],
               s->av_ms, s->av_lo, s->av_hi, s->skipped, s->resyncs, s->lost_ms, wall_s());
    }
}

// BinkWait's `subs w9, w10, w9` (main+0x7004D4): w10 = Bink's timer since the movie's sync point, w9 = when the next
// frame is due since then, both in ms; x19 = the movie. The difference is how late that frame is. Negative, the frame
// is not due and BinkWait answers "wait"; above twice BINK_TWO_FRAMES, BinkWait takes a new sync point at the next
// call, which gives the delay up. For a tracked movie the delay is kept for rt_movie_skip_w0 and BinkWait is handed
// twice BINK_TWO_FRAMES at most, so the sync point stays and the skipped frames bring the movie back; of a delay above
// MOVIE_LATE_MAX_MS the sync point moves later by what is beyond it, which the movie gives up. The flags are the
// subtraction's own. What the sound lost since the last such BinkWait (sound_lost_ms) moves the sync point later
// first, in whole ms.
uint64_t rt_movie_late_w9(Cpu *c) {
    uint32_t since = (uint32_t)c->x[10], due = (uint32_t)c->x[9];
    uintptr_t h = (uintptr_t)c->x[19];
    int i = movie_slot(h);
    if (i >= 0) {
        struct MovieStat *s = &g_movie_stat[i];
        s->lost_sound_ms += sound_lost_ms(&s->mark, 0, NULL);
        uint32_t forgive = (uint32_t)s->lost_sound_ms;
        if (forgive) {
            s->lost_sound_ms -= forgive;
            *(uint32_t volatile *)(h + BINK_SYNC_TIME) += forgive;
            since -= forgive;
            if (s->sync) s->sync += forgive;
        }
    }
    uint32_t r = since - due;
    uint64_t flags = MKFLAGS(r >> 31, r == 0, since >= due, ((since ^ due) & (since ^ r)) >> 31);
    if (i >= 0) {
        struct MovieStat *s = &g_movie_stat[i];
        int32_t late = (int32_t)r, most = 2 * (int32_t)bink32(h, BINK_TWO_FRAMES);
        movie_sample(i, h);
        // No frame has been played since the sync point (the movie starts, or goes on after a pause): the time it
        // took to get going is not caught up, the sync point moves to now and the first frame is due. The game
        // starts the movie's sound along with it, and skipping frames here put the movie ahead of that sound, by
        // 160 ms when the start was slow.
        if (late > 0 && bink32(h, BINK_PLAYED) == s->played0) {
            *(uint32_t volatile *)(h + BINK_SYNC_TIME) += (uint32_t)late;
            s->sync += (uint32_t)late;
            late = 0, r = 0;
        }
        if (late > most) {
            if (late > MOVIE_LATE_MAX_MS) {
                uint32_t beyond = (uint32_t)(late - MOVIE_LATE_MAX_MS);
                *(uint32_t volatile *)(h + BINK_SYNC_TIME) += beyond;
                s->sync += beyond, s->resyncs++, s->lost_ms += beyond;
                if (beyond >= 250)   // one stall; a machine too slow for the movie gives up a little every frame
                    rt_log("frame rate: the movie in slot %d is %d ms late at frame %u: it gives up %u ms and catches "
                           "up the rest (wall %.3f s)", i, late, ((const uint32_t *)h)[3], beyond, wall_s());
                late = MOVIE_LATE_MAX_MS;
            }
            r = (uint32_t)most;
        }
        atomic_store_explicit(&g_movie_late[i], late > 0 ? late : 0, memory_order_relaxed);
    }
    c->x[9] = r;
    return flags;
}

// The player's decode (main+0x9BF6F0) after BinkDoFrame and BinkNextFrame: `cbz w0` (main+0x9BF760) reads
// BinkShouldSkip's answer, and with a nonzero one the player decodes the next frame too and shows that one. x21 = the
// player, its movie at +8. Bink answers 0 for a movie without sound tracks; for a tracked movie the answer becomes 1
// while the movie is a frame and a half or more late on its clock (rt_movie_late_w9), one frame per call. The last two
// frames are never skipped, as in BinkShouldSkip.
void rt_movie_skip_w0(Cpu *c) {
    if ((uint32_t)c->x[0]) return;
    uintptr_t h = *(uintptr_t volatile *)(uintptr_t)(c->x[21] + 8);
    int i = movie_slot(h);
    if (i < 0) return;
    const uint32_t *hb = (const uint32_t *)h;
    if (hb[3] + 2 >= hb[2]) return;
    double frame_ms = movie_frame_ms(hb);
    int32_t late = atomic_load_explicit(&g_movie_late[i], memory_order_relaxed);
    if (late < 1.5 * frame_ms) return;
    atomic_store_explicit(&g_movie_late[i], late - (int32_t)(frame_ms + 0.5), memory_order_relaxed);
    g_movie_stat[i].skipped++;
    c->x[0] = 1;
}

// AC_FPS=30: the game's own timer runs, and of it the avsync set replaces the store of the fixed step alone
// (main+0x91DC48, `str s8, [x19, #0xd0]`: x19 = the timer, s8 = 1.0, the frame's time in ms at +0xE0), reached while
// the cutscene flag is set. The game shows every cutscene at 30 there, in-engine or with a movie, and steps it 1.0 a
// frame: on a machine that shows it below 30 frames per second it ran slower than its sound, as a held cutscene did
// under the 60 fps patch. It steps by the whole 30 fps frames of the real time instead (held_step), which is the
// game's 1.0 while the frames come on time. With the 60 fps patch on, rt_fps_timer's site branches past both this one
// and rt_free_step's. The timer frames count here too, for movie_playing.
void rt_fixed_step(Cpu *c) {
    uint8_t *t = (uint8_t *)(uintptr_t)c->x[19];
    double dt_ms = *(float volatile *)(t + 0xE0);
    atomic_fetch_add_explicit(&g_timer_frames, 1, memory_order_relaxed);
    int first = !fixed30;   // a new run: what the frame's time holds came before it
    fixed30 = 1;
    float step = held_step(dt_ms, first);
    cut_exact_now = step != 1.0f;
    *(float volatile *)(t + 0xD0) = step;
    run_frame(first, 0, step, dt_ms);
    test_hitch();
}

// AC_FPS=30, the same timer on a frame without the cutscene flag (main+0x91DC14, the first instruction of its own
// step): the cutscene run before it, if any, ends here and is logged.
void rt_free_step(Cpu *c) {
    (void)c;
    if (!fixed30) return;
    fixed30 = 0;
    cut_exact_now = 0;
    cutscene_report();
}

// A movie is playing while one waited for its next frame within the last MOVIE_HOLD_FRAMES game frames. Counted in
// game frames, not wall time: a stall of the game thread (a hitch while the movie decodes) passes no frames, so it
// does not end the hold. rt_fps_timer counts its own call first, so the frame count is at least 1 at any BinkWait.
static int movie_playing(void) {
    uint64_t w = atomic_load_explicit(&g_movie_wait_frame, memory_order_relaxed);
    return w && atomic_load_explicit(&g_timer_frames, memory_order_relaxed) - w <= MOVIE_HOLD_FRAMES;
}

// A full-screen movie is on screen: one was opened or waited for its next frame within the last MOVIE_HOLD_FRAMES
// presents. For the picture's shape (resolution.c res_wide_tick), at 30 fps as well as 60; any thread.
int rt_movie_showing(void) {
    uint64_t w = atomic_load_explicit(&g_movie_wait_present, memory_order_relaxed);
    return w && atomic_load_explicit(&g_presents, memory_order_relaxed) + 1 - w <= MOVIE_HOLD_FRAMES;
}


// The frame rate (menu, AC_FPS60): 60 fps runs the game's 60 fps patch, 30 fps switches it off. Read live: the patch
// set is switched in frame_limit_live, so the generated C picks the patched or the original instructions from the next
// site it runs.
static void frame_limit_live(void) {
    unsigned v = cfg_version();
    if (fps_read && v == fps_seen) return;
    fps_read = 1;
    fps_seen = v;
    unsigned hm = 0, he = 0, hr = 0, fm = 0, fe = 5000;
    const char *hitch = cfg_get("AC_TEST_HITCH");
    if (hitch && sscanf(hitch, "%u:%u:%u", &hm, &he, &hr) >= 1 &&
        (hm != hitch_ms || he != hitch_every_ms || hr != hitch_random_ms)) {
        if (he) rt_log("frame rate: test hitch of %u to %u ms in a cutscene frame every %u ms", hm, hm + hr, he);
        else rt_log("frame rate: test hitch of %u to %u ms in every cutscene frame", hm, hm + hr);
    }
    hitch_ms = hm, hitch_every_ms = he, hitch_random_ms = hr;
    const char *freeze = cfg_get("AC_TEST_FREEZE");
    if (freeze && sscanf(freeze, "%u:%u", &fm, &fe) >= 1 && (fm != freeze_ms || fe != freeze_every_ms))
        rt_log("frame rate: test freeze of %u ms, the sound with it, in a cutscene frame every %u ms", fm, fe);
    freeze_ms = fm, freeze_every_ms = fe;
    int hz = cfg_int("AC_FPS");
    int on = hz >= 60;
    if (on == fps60_on && fps_hz) return;
    fps60_on = on;
    // the timer does not run at 30: it ends the cutscene run it was in and starts anew at its next call (rt_fps_timer)
    atomic_fetch_add_explicit(&fps_switches, 1, memory_order_relaxed);
    fps_hz = on ? 60 : 30;
    rt_patch_switch(rt_patch_set("fps60"), on);
    rt_log("frame rate: %d fps%s", fps_hz, on ? " (60 fps patch)" : " (30 fps)");
}

uint64_t frame_step_cap_ns(void) {
    frame_limit_live();
    return (uint64_t)(1e9 / fps_hz + 0.5);
}

// The frame rate the limit gives, for the performance overlay: 60 or 30; 0 before the first frame read the setting.
double frame_step_limit_hz(void) { return fps_hz; }

// Cutscenes stay at 30: the game steps a cutscene by a fixed 1.0 a frame (rt_fps_timer), and the patch clears the
// frame call's second-vsync wait, so the game keeps presenting at interval 1 while it steps 1/30 s a frame. The present
// pacer (nvn_api_obj.c present_period) holds those frames to 30 per second while frame_step_cutscene says so, or the
// cutscene plays at twice its speed. The flag is the game's own, taken by the frame timer each game frame.
//
// A cutscene at 60 (cut60_now) steps by the real time and is not held; while an event movie plays in a cutscene, the
// cutscene is held. Outside a cutscene a movie is not held: the title screen's background is an
// event movie (movie/ev0001.bk2) and its prompt and menus run at 60 over it, each movie frame on two presents.
//
// The community cutscene fix also writes the game's frame-rate selector: 2 (30 fps) in a held cutscene and 1 (60 fps)
// otherwise at +0xF14 of the object the pointer chain below reaches, each frame, as the cheat engine's code does.
int frame_step_cutscene(void) { return fps60_on && in_cutscene && !cut60_now; }

static void cutscene_rate(void) {
    if (!fps60_on) return;
    uintptr_t base = (uintptr_t)MOD_MAIN_BASE;
    uint32_t want = frame_step_cutscene() ? 2u : 1u;
    void *volatile *slot = (void *volatile *)(base + 0xCCB7B08ull);
    void *obj = *slot;
    if (!obj) return;
    obj = *(void *volatile *)((uintptr_t)obj + 0x70);
    if (!obj) return;
    volatile uint32_t *selector = (volatile uint32_t *)((uintptr_t)obj + 0xF14);
    if (*selector != want) *selector = want;
}

// ---- AC_TRACE=perf speed lines
static uint64_t win_t0, win_frames, win_hist[4];   // win_hist: frames by present interval 1, 2, 3+ ([0] unused)
static double win_game;

static void speed_report(uint64_t now, double step, int interval) {
    if (!win_t0) {
        win_t0 = now;
        return;
    }
    win_frames++;
    win_game += step;
    win_hist[interval < 1 ? 1 : interval > 3 ? 3 : interval]++;
    if (win_frames < 120) return;
    double wall = (double)(now - win_t0) * 1e-9;
    rt_log("fps: %llu frames in %.3f s (%.1f fps), %.3f s game stepped, speed %.3f; interval 1/2/3+ %llu/%llu/%llu",
           (unsigned long long)win_frames, wall, (double)win_frames / wall, win_game, win_game / wall,
           (unsigned long long)win_hist[1], (unsigned long long)win_hist[2], (unsigned long long)win_hist[3]);
    win_t0 = now;
    win_frames = 0;
    win_game = 0;
    memset(win_hist, 0, sizeof win_hist);
}

static _Atomic int64_t world_us = -1;
static _Atomic uint64_t game_us;   // game time stepped since the first present (frame_step_ticks)
static uint64_t presents;

int64_t frame_step_world_us(void) { return atomic_load_explicit(&world_us, memory_order_relaxed); }
uint64_t frame_step_ticks(void) { return atomic_load_explicit(&game_us, memory_order_relaxed) * 60 / 1000000; }

static void world_clock(double dt, double step) {
    presents++;
    int64_t w = atomic_load_explicit(&world_us, memory_order_relaxed);
    if (w < 0) {
        rt_log("frame step: world clock starts at present %llu, %.2f s after the guest started",
               (unsigned long long)presents,
               rt_guest_start_ns ? (kern_now_ns() - rt_guest_start_ns) / 1e9 : 0.0);
        atomic_store_explicit(&world_us, 0, memory_order_relaxed);
        return;
    }
    if (!frame_step_realtime()) step = dt > STEP_MAX_S ? STEP_MAX_S : dt;   // whole-vsync steps: wall time
    atomic_store_explicit(&world_us, w + (int64_t)(step * 1e6 + 0.5), memory_order_relaxed);
}

void frame_step_present(uint64_t display_ns, int interval) {
    atomic_fetch_add_explicit(&g_presents, 1, memory_order_relaxed);
    frame_limit_live();
    cutscene_rate();
    double dt = last_ns ? (double)(display_ns - last_ns) * 1e-9 : 0;
    if (interval > 0) step_now = interval / 60.0;
    last_ns = display_ns;
    // The game clock of untimed script lines: what the game stepped this frame, whole vsyncs.
    double game = step_now;
    atomic_store_explicit(&game_us, atomic_load_explicit(&game_us, memory_order_relaxed) + (uint64_t)(game * 1e6 + 0.5),
                          memory_order_relaxed);
    world_clock(dt, game);
    // The speed line adds the game's own step under the 60 fps patch (the last game frame's, 30 fps units), so its
    // speed is game time against wall time: 1.000 when the frame-rate fix holds.
    if (rt_trace & RT_TRACE_PERF) speed_report(display_ns, fps60_on ? rt_fps_step() / 30.0 : game, interval);
}
