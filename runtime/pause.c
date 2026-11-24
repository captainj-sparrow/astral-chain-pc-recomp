// The pause: the game stands still while the in-game menu is open (AC_MENU_PAUSE, on by default) and, with
// AC_ASSUME_FOCUS=0, while its window is out of focus.
//
// It is taken at a present (nvn_api_obj.c nvnQueuePresentTexture), so the picture that stays is a whole frame, and it
// stops the whole guest rather than its frame loop:
//   - kern_pause (kernel.c) has every thread give its virtual core up at a point where it could be preempted, grants
//     none until kern_resume, and only then stops the guest's clock. No guest code runs, so nothing spins on a clock
//     that stands still, and the game, its timed waits, Bink and the sound engine see no time pass.
//   - The pads' motors stop at once (input_rumble_stop, after kern_pause: no guest thread can start them again).
//   - The audio output (ov_audout.c out_fill) keeps what the game queued and plays silence, so sound and picture go on
//     from the same point.
//   - The presenting thread, which holds no core here, shows the same picture again (GfxBackend repeat): the menu
//     and the performance overlay are drawn by the present path and need presents. It also polls the input, for the
//     pad's chord that closes the menu: the game, which polls it otherwise, is stopped.
//     The repeats stay a tenth under the display's rate, 54 a second at 60 Hz (repeat_period): at the display's own
//     rate a present that was once early stays queued, and the driver then waits in every vkQueuePresentKHR. Intel's
//     spins there (igvk64, mailbox and fifo): on an Arc 140T the paused game took a whole core that way, from the
//     second pause of a run on. They go out as soon as they are drawn, without the present thread's last
//     millisecond of yielding to a display time, which a still picture does not need.
//
// Host threads go on: the window, the present and replay threads, pipeline builds, a save's Commit. A host call under
// way finishes, and its thread stops where it takes its core back. So no host lock that those threads need may be
// held across kern_core_back or a call into guest code.
//
// Out of focus the pause is the Switch's: the game sets FocusHandlingMode 3 (AlwaysSuspend), so the system suspends
// the whole process and the game never runs out of focus (ov_system.c GetCurrentFocusState). Nothing changes on the
// screen then, so the picture is shown again four times a second only (a window that is uncovered or resized gets it
// back), and the pads' buttons are not looked at: they belong to the window in front.
//
// The pause waits for a save being written (fs_save_busy), so that quitting from the paused menu never cuts one in
// two, and it is taken at the next present after it.
//
// A setting whose effect is drawn by the game (the upscaler, ambient occlusion, the HUD) shows once the game goes on;
// what the present path does (colour, sharpening, HDR levels, the window) shows at once.
//
// The log has a line where a pause starts (the reason, its number, the local time), where its reason changes, where
// it ends (how long it lasted, and all of them so far), and once for each cause that holds a wanted pause back. A
// window closed during a pause says so (gfx_present.c gfx_window_closed).
//
// Another reason to pause gets a WHY_ bit in pause_reasons and its words in why_text.
#include "cfg.h"
#include "fs.h"
#include "gfx.h"
#include "host.h"
#include "input.h"
#include "pause.h"
#include "platform/platform.h"
#include "ui/menu.h"

#define PAUSE_RETRY_NS 2000000000ull          // after a guest that did not stop (kern_pause): the game runs on this long
#define AWAY_REPEAT_NS 250000000ull           // out of focus: the time between two repeats

enum { WHY_MENU = 1, WHY_FOCUS = 2 };
static const char *const why_text[] = {"", "menu", "out of focus", "menu and out of focus"};

// AC_ASSUME_FOCUS is looked up only while out of focus, and at each call, so a change takes effect at once.
static int pause_reasons(void) {
    int why = menu_is_open() && cfg_bool("AC_MENU_PAUSE") ? WHY_MENU : 0;
    if (!gfx_window_focused() && !cfg_bool("AC_ASSUME_FOCUS")) why |= WHY_FOCUS;
    return why;
}

static const char *held_by;   // presenting thread only: what holds the wanted pause back, once it was logged

static void held_back(int why, const char *by) {
    if (held_by == by) return;
    held_by = by;
    rt_log("pause: held back (%s): %s; the game runs on meanwhile", why_text[why], by);
}

// The time between two repeats: 1.1 periods of the display, or of 60 Hz when that is faster or unknown.
static uint64_t repeat_period(void) {
    double hz = gfx_window_refresh_hz();
    if (hz < 20 || hz > 60) hz = 60;
    return (uint64_t)(1.1e9 / hz);
}

void pause_frame(const tex_t *t) {
    static uint64_t retry_ns;   // presenting thread only
    static unsigned count;
    static const char saving[] = "a save is being written", no_repeat[] = "this back end cannot repeat a picture",
                      refused[] = "the threads could not be stopped, tried again every 2 s";
    int why = pause_reasons();
    if (__builtin_expect(!why, 1)) {
        if (__builtin_expect(held_by != NULL, 0)) {
            rt_log("pause: no longer wanted; the one held back was never taken");
            held_by = NULL;
        }
        return;
    }
    const GfxBackend *b = gfx_backend();
    if (!b->repeat) {
        held_back(why, no_repeat);
        return;
    }
    if (fs_save_busy()) {
        held_back(why, saving);
        return;
    }
    uint64_t asked = kern_now_ns();
    if (asked < retry_ns) return;
    if (!kern_pause()) {
        retry_ns = kern_now_ns() + PAUSE_RETRY_NS;
        held_back(why, refused);
        return;
    }
    if (fs_save_busy()) {   // a save began while the cores were coming back
        kern_resume();
        held_back(why, saving);
        return;
    }
    held_by = NULL;
    input_rumble_stop();
    uint64_t start = kern_now_ns(), next = start, shown = start;
    struct tm at;
    plat_local_time(&at);
    count++;
    rt_log("pause: the game stands still (%s; pause %u at %02d:%02d:%02d local time; its threads stopped in %.3f ms)",
           why_text[why], count, at.tm_hour, at.tm_min, at.tm_sec, (double)(start - asked) / 1e6);
    for (int now_why; (now_why = pause_reasons()) != 0;) {
        if (now_why != why) {
            rt_log("pause: still paused %.2f s in, now %s (was %s)", (double)(kern_now_ns() - start) / 1e9,
                   why_text[now_why], why_text[why]);
            why = now_why;
        }
        next += repeat_period();   // also how soon the end of the pause is seen
        uint64_t now = kern_now_ns();
        if (next < now) next = now;
        int away = why & WHY_FOCUS;
        if (!away || now - shown >= AWAY_REPEAT_NS) {
            b->repeat(t, 0);
            shown = now;
        }
        input_poll_paused(away);
        now = kern_now_ns();
        if (next > now) plat_wait_ns(next - now);
    }
    lowlat_resume();
    kern_resume();
    rt_log("pause: the game goes on after %.2f s (%s; pause %u; %.2f s paused in all)",
           (double)(kern_now_ns() - start) / 1e9, why_text[why], count, (double)kern_paused_ns() / 1e9);
}
