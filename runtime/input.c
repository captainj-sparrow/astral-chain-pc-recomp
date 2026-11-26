// Host input for nn::hid (runtime/ov_input.c): an SDL3 gamepad (no SDL window needed), the keyboard and the mouse,
// merged into Switch npad button bits and stick axes. Button order follows libnx HidNpadButton (ISC). The key and pad
// bindings are the tables in input_map.c (astralchain.ini overrides them). The keyboard and the mouse come from the window
// system's back end (input_wsys.h: input_win.c with Win32, input_sdl.c with SDL), which also owns the cursor capture.
// Keys and their map: docs/controls.md.
// The gamepad also rumbles (input_rumble) and, when it has a gyro and an accelerometer, feeds the six-axis sensor from a
// thread of its own (input_motion_start). Every SDL call on the gamepad is made under `lock`. In co-op (input_players)
// a second gamepad is opened and each player reads its own sources (player_pad).
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <SDL3/SDL.h>

#include "cfg.h"
#include "fs.h"
#include "host.h"
#include "input.h"
#include "input_map.h"
#include "input_wsys.h"
#include "nvn_api.h"
#include "ui/menu.h"
#include "vk.h"
#include "platform/platform.h"

enum { STICK_MAX = 32767, STICK_DIR = 16384, RESCAN_MS = 1000 };

static PlatMutex lock = PLAT_MUTEX_INIT;
static int state;                    // 0 not started, 1 running, -1 disabled or SDL failed
static SDL_Gamepad *pad;
static uint64_t next_scan;
static int pad_rumble;               // the open gamepad has rumble motors
static volatile int32_t pad_labels[4];  // ... and the labels of its south, east, west, north buttons (PAD_LABEL_*), any thread
static int pad_east_a;               // ... and labels its east face button A (Nintendo layout under AC_PAD_LAYOUT=auto)
static int pad_layout = -1;          // PAD_LAYOUT_* in effect for the open gamepad, -1 to pick again; live (cfg_version)
static unsigned pad_layout_seen;
static volatile int32_t pad_layout_shown = -1;   // pad_layout for the menu (any thread), -1 without a gamepad
static volatile int32_t pad_kind_shown = PAD_KIND_NONE;   // the open gamepad's kind for the menu, NONE without one
static SDL_JoystickID sensor_pad;    // the gamepad whose sensors the motion thread set up, 0 none
static int sensor_ok;                // ... and it has both sensors, enabled
static int motion_thread;            // 0 not started, 1 running, -1 off (AC_MOTION=0, no SDL)
// Co-op (input_players): the number of players the game has, and a second gamepad, opened only while it has two. pad
// stays the first gamepad (the menu's, the solo controller's, the motion thread's); when it goes away pad2 takes its
// place. The keyboard and the mouse go to the first player without a gamepad (player_pad).
static int players = 1;
static SDL_Gamepad *pad2;
static uint64_t next_scan2;
static int pad2_rumble, pad2_east_a, pad2_kind, pad2_layout = -1;
static unsigned pad2_layout_seen;

// SDL buttons of the gamepad inputs (input_map.h PAD_*), by position; PAD_ZL and PAD_ZR are the triggers (axes).
static const SDL_GamepadButton pad_sdl[PAD_COUNT] = {
    [PAD_SOUTH] = SDL_GAMEPAD_BUTTON_SOUTH, [PAD_EAST] = SDL_GAMEPAD_BUTTON_EAST, [PAD_WEST] = SDL_GAMEPAD_BUTTON_WEST,
    [PAD_NORTH] = SDL_GAMEPAD_BUTTON_NORTH, [PAD_LSTICK] = SDL_GAMEPAD_BUTTON_LEFT_STICK,
    [PAD_RSTICK] = SDL_GAMEPAD_BUTTON_RIGHT_STICK, [PAD_L] = SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
    [PAD_R] = SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, [PAD_START] = SDL_GAMEPAD_BUTTON_START, [PAD_BACK] = SDL_GAMEPAD_BUTTON_BACK,
    [PAD_DPAD_LEFT] = SDL_GAMEPAD_BUTTON_DPAD_LEFT, [PAD_DPAD_UP] = SDL_GAMEPAD_BUTTON_DPAD_UP,
    [PAD_DPAD_RIGHT] = SDL_GAMEPAD_BUTTON_DPAD_RIGHT, [PAD_DPAD_DOWN] = SDL_GAMEPAD_BUTTON_DPAD_DOWN,
    [PAD_LPADDLE1] = SDL_GAMEPAD_BUTTON_LEFT_PADDLE1, [PAD_RPADDLE1] = SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1,
    [PAD_LPADDLE2] = SDL_GAMEPAD_BUTTON_LEFT_PADDLE2, [PAD_RPADDLE2] = SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2,
};

// AC_INPUT_SCRIPT=<file>: lines "<frame> <button> [hold]" or "<frame> LSTICK|RSTICK <x> <y> [hold]", ORed into the real input.
// <frame> and its hold are 60 Hz ticks of game time since the first present (frame_step_ticks: presents at 60 fps), so
// the title menus get the presses at the same game time at any frame limit. A line "@<ms> ..." is timed instead: <ms>
// and its hold are milliseconds of the world clock (frame_step_world_us, game time since the loaded save's first
// frame), so it lands at the same game time at any frame rate and load length. A timed event shorter than a frame is
// applied once. In a script with timed lines (a recording) the untimed ones belong before the world clock: they stop
// when it starts, as the load ends at another tick in each run. "P2" before the button gives the line to co-op's
// player 2 ("@40000 P2 A 200"); the others are player 1's, which is also the solo controller.
// at, hold: ticks, or microseconds of the world clock when timed.
typedef struct { uint64_t at, hold; int bit, stick, timed, player; int32_t x, y; int logged; char text[48]; } ScriptEv;
static const struct { const char *name; int bit; } script_buttons[] = {
    {"A", B_A}, {"B", B_B}, {"X", B_X}, {"Y", B_Y}, {"LCLICK", B_STICK_L}, {"RCLICK", B_STICK_R}, {"L", B_L}, {"R", B_R},
    {"ZL", B_ZL}, {"ZR", B_ZR}, {"PLUS", B_PLUS}, {"MINUS", B_MINUS}, {"LEFT", B_LEFT}, {"UP", B_UP}, {"RIGHT", B_RIGHT}, {"DOWN", B_DOWN},
};
static ScriptEv *script_ev;
static int script_n, script_state;   // 0 not loaded, 1 loaded
static int script_mixed;             // the script has timed lines: its untimed ones end where the world clock starts
static int script_p2;                // the script has lines for player 2: player 2 is connected in co-op

static void script_load(void) {
    script_state = 1;
    const char *path = cfg_get("AC_INPUT_SCRIPT");
    if (!path || !*path) return;
    FILE *f = fopen(path, "r");
    if (!f) { rt_log("input: cannot open AC_INPUT_SCRIPT %s", path); return; }
    char line[256];
    int cap = 0, ln = 0;
    while (fgets(line, sizeof line, f)) {
        ln++;
        char *h = strchr(line, '#');
        if (h) *h = 0;
        char name[32];
        double at;
        int off = 0, timed = line[0] == '@';
        if (sscanf(line + timed, "%lf %31s%n", &at, name, &off) < 2 || at < 0) continue;
        off += timed;
        int player = 0, more = 0;
        if (!strcmp(name, "P1") || !strcmp(name, "P2")) {
            player = name[1] - '1';
            if (sscanf(line + off, "%31s%n", name, &more) < 1) continue;
            off += more;
        }
        double unit = timed ? 1000.0 : 1.0;   // ms to us, or presents
        ScriptEv e = {.at = (uint64_t)(at * unit + 0.5), .hold = timed ? 66667 : 4, .bit = -1, .timed = timed,
                      .player = player};
        long a, b;
        double c;
        if (!strcmp(name, "LSTICK") || !strcmp(name, "RSTICK")) {
            int n = sscanf(line + off, "%ld %ld %lf", &a, &b, &c);
            if (n < 2) { rt_log("input: script line %d: %s needs x y [hold]", ln, name); continue; }
            e.stick = name[0] == 'L' ? 1 : 2; e.x = (int32_t)a; e.y = (int32_t)b;
            if (n >= 3 && c > 0) e.hold = (uint64_t)(c * unit + 0.5);
            snprintf(e.text, sizeof e.text, "%s %d %d", name, e.x, e.y);
        } else {
            for (size_t i = 0; i < sizeof script_buttons / sizeof *script_buttons; i++)
                if (!strcmp(name, script_buttons[i].name)) e.bit = script_buttons[i].bit;
            if (e.bit < 0) { rt_log("input: script line %d: unknown button %s", ln, name); continue; }
            if (sscanf(line + off, "%lf", &c) == 1 && c > 0) e.hold = (uint64_t)(c * unit + 0.5);
            snprintf(e.text, sizeof e.text, "%s", name);
        }
        if (script_n == cap) {
            cap = cap ? cap * 2 : 32;
            script_ev = realloc(script_ev, (size_t)cap * sizeof *script_ev);
            if (!script_ev) { script_n = 0; fclose(f); return; }
        }
        if (player) snprintf(e.text + strlen(e.text), sizeof e.text - strlen(e.text), " (player 2)");
        script_ev[script_n++] = e;
        script_mixed |= timed;
        script_p2 |= player;
    }
    fclose(f);
    rt_log("input: script %s, %d events%s", path, script_n, script_p2 ? ", some for co-op's player 2" : "");
}

// player: 0 for the solo controller and co-op's player 1, 1 for player 2.
static void script_apply(int player, uint64_t *b, int32_t sl[2], int32_t sr[2]) {
    if (!script_state) script_load();
    if (!script_n) return;
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), tick = frame_step_ticks();
    int64_t w = frame_step_world_us();
    for (int i = 0; i < script_n; i++) {
        ScriptEv *e = &script_ev[i];
        if (e->player != player) continue;
        if (e->timed ? w < 0 : script_mixed && w >= 0) continue;
        uint64_t now = e->timed ? (uint64_t)w : tick;
        if (now < e->at || (now >= e->at + e->hold && (e->logged || !e->timed))) continue;
        if (!e->logged) {
            e->logged = 1;
            if (e->timed) rt_log("input: script frame %llu at %.1f ms: %s", (unsigned long long)fr, w / 1000.0, e->text);
            else rt_log("input: script frame %llu tick %llu: %s", (unsigned long long)fr, (unsigned long long)tick, e->text);
        }
        if (e->stick) { int32_t *s = e->stick == 1 ? sl : sr; s[0] = e->x; s[1] = e->y; }
        else *b |= 1ull << e->bit;
    }
}

// AC_INPUT_RECORD=<file>: writes the merged input as AC_INPUT_SCRIPT lines, one per button press and per stick
// segment, so a played session replays as a script. A stick segment ends when an axis moves by more than REC_STICK_STEP.
// An event that begins on the world clock is written timed ("@<ms>", hold in ms), one before it in ticks
// (frame_step_ticks). When the world clock starts, held buttons and sticks are cut there and go on as timed events.
enum { REC_STICK_STEP = 2048 };
static FILE *rec;
static int rec_state;                // 0 not opened, 1 recording, -1 off
static uint64_t rec_down[16];        // tick + 1 at which each button went down, 0 while up
static int64_t rec_down_w[16];       // the world clock then, -1 before it started
static uint64_t rec_stick_from[2];   // tick + 1 at which the current non-zero stick segment began
static int64_t rec_stick_w[2];
static int32_t rec_stick[2][2];
static int64_t rec_last_w = -1;      // the world clock at the previous apply

// "<tick> <what> <hold>" or "@<ms> <what> <hold ms>": from tick from and world clock from_w to now.
static void rec_line(uint64_t from, int64_t from_w, uint64_t fr, int64_t w, const char *what) {
    if (from_w >= 0 && w >= from_w)
        fprintf(rec, "@%.1f %s %.1f\n", from_w / 1000.0, what, (w > from_w ? w - from_w : 1000) / 1000.0);
    else
        fprintf(rec, "%llu %s %llu\n", (unsigned long long)from, what, (unsigned long long)(fr > from ? fr - from : 1));
}

static void rec_close_stick(int s, uint64_t fr, int64_t w) {
    if (!rec_stick_from[s]) return;
    char what[48];
    snprintf(what, sizeof what, "%s %d %d", s ? "RSTICK" : "LSTICK", rec_stick[s][0], rec_stick[s][1]);
    rec_line(rec_stick_from[s] - 1, rec_stick_w[s], fr, w, what);
    rec_stick_from[s] = 0;
}

static void rec_flush_all(void) {
    if (rec_state != 1) return;
    uint64_t fr = frame_step_ticks() + 1;
    int64_t w = frame_step_world_us();
    for (int i = 0; i < 16; i++)
        if (rec_down[i]) { rec_line(rec_down[i] - 1, rec_down_w[i], fr, w, script_buttons[i].name); rec_down[i] = 0; }
    rec_close_stick(0, fr, w);
    rec_close_stick(1, fr, w);
    fflush(rec);
}

static void rec_apply(uint64_t b, const int32_t sl[2], const int32_t sr[2]) {
    if (!rec_state) {
        const char *path = cfg_get("AC_INPUT_RECORD");   // no default: not recording when unset
        rec = path && *path ? fopen(path, "w") : NULL;
        rec_state = rec ? 1 : -1;
        if (path && *path && !rec) rt_log("input: cannot open AC_INPUT_RECORD %s", path);
        if (rec) { fprintf(rec, "# Recorded input (AC_INPUT_RECORD); frame = 60 Hz ticks of game time, @ms = world clock.\n"); atexit(rec_flush_all);
                   rt_log("input: recording to %s", path); }
    }
    if (rec_state != 1) return;
    uint64_t fr = frame_step_ticks();
    int64_t w = frame_step_world_us();
    int wrote = 0;
    if (w >= 0 && rec_last_w < 0) {   // the world clock starts: what is held goes on as timed events from @0
        for (int i = 0; i < 16; i++)
            if (rec_down[i]) {
                rec_line(rec_down[i] - 1, -1, fr, -1, script_buttons[i].name);
                rec_down[i] = fr + 1, rec_down_w[i] = w;
                wrote = 1;
            }
        for (int s = 0; s < 2; s++)
            if (rec_stick_from[s]) {
                rec_close_stick(s, fr, -1);
                rec_stick_from[s] = fr + 1, rec_stick_w[s] = w;
                wrote = 1;
            }
    }
    rec_last_w = w;
    for (int i = 0; i < 16; i++) {
        int bit = script_buttons[i].bit, down = (b >> bit & 1) != 0;
        if (down && !rec_down[i]) rec_down[i] = fr + 1, rec_down_w[i] = w;
        else if (!down && rec_down[i]) {
            rec_line(rec_down[i] - 1, rec_down_w[i], fr, w, script_buttons[i].name);
            rec_down[i] = 0;
            wrote = 1;
        }
    }
    const int32_t *st[2] = {sl, sr};
    for (int s = 0; s < 2; s++) {
        int moved = abs(st[s][0] - rec_stick[s][0]) > REC_STICK_STEP || abs(st[s][1] - rec_stick[s][1]) > REC_STICK_STEP ||
                    ((st[s][0] || st[s][1]) != (rec_stick[s][0] || rec_stick[s][1]));
        if (!moved) continue;
        if (rec_stick_from[s]) { rec_close_stick(s, fr, w); wrote = 1; }
        rec_stick[s][0] = st[s][0]; rec_stick[s][1] = st[s][1];
        if (st[s][0] || st[s][1]) rec_stick_from[s] = fr + 1, rec_stick_w[s] = w;
    }
    if (wrote) fflush(rec);
}

static float mouse_sens = -1;              // AC_MOUSE_SENS, 0 = mouse camera off; live (cfg_version)
static unsigned mouse_sens_seen;

static float mouse_sens_get(void) {
    unsigned ver = cfg_version();
    if (mouse_sens >= 0 && ver == mouse_sens_seen) return mouse_sens;
    mouse_sens_seen = ver;
    double v = cfg_double("AC_MOUSE_SENS");
    mouse_sens = (float)(v < 0 ? 0 : v > 10 ? 10 : v);
    return mouse_sens;
}


// ---- walk key: the keyboard's stick samples scaled down while a key is held (AC_WALK_KEY, AC_WALK_SCALE)
// A keyboard key is all or nothing, so every keyboard-driven stick sample is +/-32767 and the game only ever sees a
// full tilt. Holding this key scales those two samples, which gives the half tilt a gamepad gets from its analog range,
// so the keyboard can walk as well as run. yuzu and eden do the same with their modifier and modifier_scale settings.
// Only the keyboard is scaled: the gamepad's axes and the mouse camera sample their own values.
static int walk_key = KEY_NONE;        // AC_WALK_KEY as a KEY_*, KEY_NONE = off; live (cfg_version)
static float walk_scale = 0.51f;       // AC_WALK_SCALE, 0.2 to 0.9; live (cfg_version)
static unsigned walk_seen = ~0u;

// The ini takes the key names the Controls tab writes; a hand-written line may spell them in any case.
static int name_matches(const char *name, const char *key) {
    for (; *name && *key; name++, key++) {
        unsigned char a = (unsigned char)*name, b = (unsigned char)*key;
        if (a >= 'A' && a <= 'Z') a = (unsigned char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (unsigned char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return *name == *key;
}

static void walk_get(void) {
    unsigned ver = cfg_version();
    if (ver == walk_seen) return;
    walk_seen = ver;
    const char *name = cfg_value("AC_WALK_KEY");   // "none" (the default) turns the key off
    int key = KEY_NONE;
    if (*name && !name_matches(name, "none")) {
        for (int k = KEY_A; k < KEY_COUNT && key == KEY_NONE; k++)
            if (input_key_name(k) && name_matches(name, input_key_name(k))) key = k;
        if (key == KEY_NONE) rt_log("input: walk_key: unknown key '%s', the keyboard sticks keep their full tilt", name);
    }
    // 0.51 rather than 0.5: a keyboard sample is 32767, and 32767 * 0.5 = 16383 falls one short of stick_bits'
    // STICK_DIR (16384), which would leave the game's stick-direction bits clear while the walk key is held. The value
    // in the settings table carries that default; this code only rejects what the table's range does not allow.
    const char *sv = cfg_value("AC_WALK_SCALE");
    char *end;
    double d = strtod(sv, &end);
    double s = end == sv || *end || !(d >= 0.2 && d <= 0.9) ? strtod(cfg_meta("AC_WALK_SCALE")->def, NULL) : d;
    if (key != walk_key || (float)s != walk_scale) {
        walk_key = key;
        walk_scale = (float)s;
        rt_log("input: walk key %s, keyboard sticks at %.0f%% of their tilt (mouse and gamepad unchanged)",
               key != KEY_NONE ? input_key_name(key) : "off", s * 100);
    }
}

// The mouse camera as a right-stick sample, held for the whole presented frame. Under `lock`. The sample is divided by
// the camera speed factor the game runs with (camera_speed.c, cam_speed_applied), which multiplies every right-stick
// turn rate, so AC_MOUSE_SENS alone sets the mouse's speed; below 1 the sample stays within the stick's range, so the
// mouse's top speed drops with the factor.
static void mouse_camera(int32_t out[2]) {
    static uint64_t frame = ~0ull, t0;
    static int32_t held[2];
    uint64_t now = plat_now_ns();
    int32_t dx, dy;
    if (!ikb_captured()) {
        ikb_mouse_take(&dx, &dy);
        held[0] = held[1] = 0;
        frame = ~0ull;
    } else {
        uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        double age = frame == ~0ull ? 0 : (double)(now - t0) / 1e9;
        if (fr != frame) {
            double dt = age < 0.004 ? 0.004 : age > 0.1 ? 0.1 : age;
            ikb_mouse_take(&dx, &dy);
            input_mouse_stick(dx, dy, dt, mouse_sens_get(), held);
            frame = fr;
            t0 = now;
        } else if (age > 0.1) {
            held[0] = held[1] = 0;   // frames stopped: do not hold a deflection
        }
    }
    double k = cam_speed_applied(), x = held[0] / k, y = held[1] / k, len = sqrt(x * x + y * y);
    if (len > STICK_MAX) x *= STICK_MAX / len, y *= STICK_MAX / len;
    out[0] = (int32_t)lround(x);
    out[1] = (int32_t)lround(y);
}

static float stick_dead[2] = {-1, -1};     // input_pad_deadzone of the left and right stick; live (cfg_version)
static unsigned stick_dead_seen;

static float stick_dead_get(int s) {
    unsigned ver = cfg_version();
    if (stick_dead[0] < 0 || ver != stick_dead_seen) {
        stick_dead_seen = ver;
        stick_dead[0] = input_pad_deadzone(0);
        stick_dead[1] = input_pad_deadzone(1);
    }
    return stick_dead[s];
}

// input_pad_trigger_on, live (cfg_version). Up to 1.21.1 it was a fixed half of the travel, which made ZL and ZR wait
// for a half-pulled trigger on the PC pads (the Switch's ZL and ZR are buttons).
static int32_t trigger_on = -1;
static unsigned trigger_on_seen;

static int32_t trigger_on_get(void) {
    unsigned ver = cfg_version();
    if (trigger_on < 0 || ver != trigger_on_seen) trigger_on_seen = ver, trigger_on = input_pad_trigger_on();
    return trigger_on;
}

// One gamepad stick through the dead zone; SDL's Y grows downwards, the Switch's upwards.
static void stick(SDL_Gamepad *g, SDL_GamepadAxis ax, SDL_GamepadAxis ay, float dead, int32_t out[2]) {
    input_pad_stick(SDL_GetGamepadAxis(g, ax), -(int32_t)SDL_GetGamepadAxis(g, ay), dead, out);
}

// The prompts' kind of an open gamepad: the SDL type first, the face labels for a STANDARD pad that carries
// another maker's labels; anything else (Switch pads included) keeps the Nintendo prompts.
static int kind_of(SDL_Gamepad *gamepad) {
    switch (SDL_GetGamepadType(gamepad)) {
    case SDL_GAMEPAD_TYPE_XBOX360:
    case SDL_GAMEPAD_TYPE_XBOXONE: return PAD_KIND_XBOX;
    case SDL_GAMEPAD_TYPE_PS3:
    case SDL_GAMEPAD_TYPE_PS4:
    case SDL_GAMEPAD_TYPE_PS5: return PAD_KIND_PLAYSTATION;
    default: break;
    }
    if (SDL_GetGamepadButtonLabel(gamepad, SDL_GAMEPAD_BUTTON_SOUTH) == SDL_GAMEPAD_BUTTON_LABEL_A)
        return PAD_KIND_XBOX;
    if (SDL_GetGamepadButtonLabel(gamepad, SDL_GAMEPAD_BUTTON_SOUTH) == SDL_GAMEPAD_BUTTON_LABEL_CROSS)
        return PAD_KIND_PLAYSTATION;
    return PAD_KIND_NONE;
}

// Under lock: a gamepad other than `not`, opened, or NULL. SDL hands out the same handle for a gamepad opened twice.
static SDL_Gamepad *open_other(SDL_Gamepad *not) {
    SDL_JoystickID own = not ? SDL_GetGamepadID(not) : 0;
    SDL_Gamepad *g = NULL;
    int n = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&n);
    for (int i = 0; ids && i < n && !g; i++)
        if (ids[i] != own) g = SDL_OpenGamepad(ids[i]);
    SDL_free(ids);
    return g;
}

// Open the first gamepad when none is open or the open one went away (co-op's second gamepad takes its place);
// logs once per change.
static void scan_pads(void) {
    if (pad && SDL_GamepadConnected(pad)) return;
    if (pad) {
        SDL_CloseGamepad(pad); pad = NULL; sensor_pad = 0; sensor_ok = 0;
        plat_xchg32(&pad_layout_shown, -1);
        plat_xchg32(&pad_kind_shown, PAD_KIND_NONE);
        for (int i = 0; i < 4; i++) plat_xchg32(&pad_labels[i], PAD_LABEL_NONE);
        rt_log("input: gamepad disconnected");
    }
    if (pad2 && SDL_GamepadConnected(pad2)) {
        pad = pad2, pad2 = NULL;
        rt_log("input: the second gamepad becomes the first");
    } else
        pad = open_other(NULL);
    if (!pad) return;
    pad_rumble = SDL_GetBooleanProperty(SDL_GetGamepadProperties(pad), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
    pad_east_a = SDL_GetGamepadButtonLabel(pad, SDL_GAMEPAD_BUTTON_EAST) == SDL_GAMEPAD_BUTTON_LABEL_A;
    // The prompts' kind: the SDL type first, the face labels for a STANDARD pad that carries another maker's
    // labels; anything else (Switch pads included) keeps the Nintendo prompts.
    int kind = kind_of(pad);
    plat_xchg32(&pad_kind_shown, kind);
    for (int i = 0; i < 4; i++) {
        SDL_GamepadButtonLabel lb = SDL_GetGamepadButtonLabel(pad, pad_sdl[PAD_SOUTH + i]);
        plat_xchg32(&pad_labels[i], lb == SDL_GAMEPAD_BUTTON_LABEL_A          ? PAD_LABEL_A
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_B        ? PAD_LABEL_B
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_X        ? PAD_LABEL_X
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_Y        ? PAD_LABEL_Y
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_CROSS    ? PAD_LABEL_CROSS
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_CIRCLE   ? PAD_LABEL_CIRCLE
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_SQUARE   ? PAD_LABEL_SQUARE
                                            : lb == SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE ? PAD_LABEL_TRIANGLE
                                                                                      : PAD_LABEL_NONE);
    }
    pad_layout = -1;
    int motion = SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO) && SDL_GamepadHasSensor(pad, SDL_SENSOR_ACCEL);
    rt_log("input: gamepad \"%s\" (%s, %s, east button %s)", SDL_GetGamepadName(pad), pad_rumble ? "rumble" : "no rumble",
           motion ? "gyro and accelerometer" : "no motion sensors", pad_east_a ? "labelled A" : "not labelled A");
}

// Under lock, co-op: open a second gamepad when none is open or the open one went away; logs once per change.
static void scan_pad2(void) {
    if (pad2 && SDL_GamepadConnected(pad2)) return;
    if (pad2) {
        SDL_CloseGamepad(pad2), pad2 = NULL;
        rt_log("input: second gamepad disconnected");
    }
    if (!pad || !(pad2 = open_other(pad))) return;
    pad2_rumble = SDL_GetBooleanProperty(SDL_GetGamepadProperties(pad2), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
    pad2_east_a = SDL_GetGamepadButtonLabel(pad2, SDL_GAMEPAD_BUTTON_EAST) == SDL_GAMEPAD_BUTTON_LABEL_A;
    pad2_kind = kind_of(pad2);
    pad2_layout = -1;
    rt_log("input: second gamepad \"%s\" (%s, east button %s)", SDL_GetGamepadName(pad2),
           pad2_rumble ? "rumble" : "no rumble", pad2_east_a ? "labelled A" : "not labelled A");
}

// Under lock, state 1: scan for gamepads that went away or arrived, at most once a second each.
static void pads_scan(void) {
    uint64_t ms = plat_now_ns() / 1000000u;
    if ((!pad || !SDL_GamepadConnected(pad)) && ms >= next_scan) next_scan = ms + RESCAN_MS, scan_pads();
    if (players > 1 && (!pad2 || !SDL_GamepadConnected(pad2)) && ms >= next_scan2) next_scan2 = ms + RESCAN_MS, scan_pad2();
}

// Under lock: the layout of the open gamepad, picked again when a setting or the gamepad changed; logs each change.
// A loaded Xbox/PlayStation prompts mod forces its layout whatever the setting names (each mod is drawn for one
// layout); overlay_init runs before input starts, so romfs_prompts is set by the first pick.
static const char *layout_name(int layout) {
    return layout == PAD_LAYOUT_XBOX ? "xbox" : layout == PAD_LAYOUT_PLAYSTATION ? "playstation" : "nintendo";
}

// The forced layout of the loaded prompts folder (input_prompts_folder): the Xbox mod forces the Xbox layout, the
// PlayStation mod the PlayStation layout, the game's own prompts no forcing (-1). It follows the loaded prompts
// rather than the asked-for ones, so the buttons keep matching the glyphs on screen until a restart brings the new
// prompts.
int input_forced_layout(void) {
    const char *loaded = romfs_prompts();
    if (loaded && !strcmp(loaded, input_prompts_folder(PROMPTS_XBOX, NULL))) return input_prompts_layout(PROMPTS_XBOX);
    if (loaded && !strcmp(loaded, input_prompts_folder(PROMPTS_PLAYSTATION, NULL)))
        return input_prompts_layout(PROMPTS_PLAYSTATION);
    return -1;
}

static int pad_layout_get(void) {
    unsigned ver = cfg_version();
    if (pad_layout >= 0 && ver == pad_layout_seen) return pad_layout;
    pad_layout_seen = ver;
    int was = pad_layout;
    const char *set = cfg_value("AC_PAD_LAYOUT");
    int kind = (int)plat_cas32(&pad_kind_shown, 0, 0);
    int forced = input_forced_layout();
    pad_layout = forced >= 0 ? forced : input_pad_layout_pick(set, pad_east_a, kind);
    if (pad_layout != was) {
        if (forced >= 0)
            rt_log("input: gamepad layout %s (%s is loaded, pad_layout = %s)", layout_name(pad_layout),
                   romfs_prompts(), set);
        else
            rt_log("input: gamepad layout %s (pad_layout = %s)", layout_name(pad_layout), set);
    }
    plat_xchg32(&pad_layout_shown, pad_layout);
    return pad_layout;
}

// The same for co-op's second gamepad, picked by the same rules (not logged: the first gamepad's line names the setting).
static int pad2_layout_get(void) {
    unsigned ver = cfg_version();
    if (pad2_layout >= 0 && ver == pad2_layout_seen) return pad2_layout;
    pad2_layout_seen = ver;
    int forced = input_forced_layout();
    pad2_layout = forced >= 0 ? forced : input_pad_layout_pick(cfg_value("AC_PAD_LAYOUT"), pad2_east_a, pad2_kind);
    return pad2_layout;
}

// The AC_BIND_* values last applied, one per line; the tables are built again when one changes (the menu's remapping rows write
// them while the game runs). Under lock.
static unsigned binds_seen = ~0u;
static char binds_applied[4096];

static void binds_refresh(void) {
    unsigned v = cfg_version();
    if (v == binds_seen) return;
    binds_seen = v;
    char now[sizeof binds_applied];
    size_t len = 0;
    for (int a = 0; a < ACTION_COUNT; a++) {
        char var[32];
        snprintf(var, sizeof var, "AC_BIND_%s", input_action_name(a));
        const char *s = cfg_get(var);
        len += (size_t)snprintf(now + len, len < sizeof now ? sizeof now - len : 0, "%s\n", s ? s : "");
    }
    if (!strcmp(now, binds_applied)) return;
    memcpy(binds_applied, now, sizeof now);
    input_map_configure();
}

// AC_TEST_PAD=<lx>,<ly>,<rx>,<ry> (test runs): an SDL virtual gamepad whose sticks rest at these raw SDL axis values
// (-32768..32767, down is +y), attached before the first scan so it is the gamepad the game reads: drift and full
// deflection go through the same mapping and dead zone as a real pad. With AC_INPUT set to anything but "none" (which
// run_game.py sets) and AC_MENU_TEST it also shows that a pad resting past its dead zone still reaches the game after
// the menu closed: "3500,0,0,0" rests at 10.7 %, past the default 8 %.
static void test_pad_attach(void) {
    static SDL_Joystick *joy;   // kept open: a virtual joystick's axes are set through an open handle
    const char *e = cfg_get("AC_TEST_PAD");
    int v[4];
    if (!e || !*e) return;
    if (sscanf(e, "%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3]) != 4) {
        rt_log("input: AC_TEST_PAD=%s is not <lx>,<ly>,<rx>,<ry>; no test gamepad", e);
        return;
    }
    SDL_VirtualJoystickDesc d;
    SDL_INIT_INTERFACE(&d);
    d.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    d.naxes = SDL_GAMEPAD_AXIS_COUNT;
    d.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    d.name = "Astral Chain test gamepad";
    SDL_JoystickID id = SDL_AttachVirtualJoystick(&d);
    joy = id ? SDL_OpenJoystick(id) : NULL;
    if (!joy) {
        rt_log("input: test gamepad not attached (%s)", SDL_GetError());
        return;
    }
    static const SDL_GamepadAxis axes[4] = {SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, SDL_GAMEPAD_AXIS_RIGHTX,
                                            SDL_GAMEPAD_AXIS_RIGHTY};
    for (int i = 0; i < 4; i++)
        SDL_SetJoystickVirtualAxis(joy, axes[i], (Sint16)(v[i] < -32768 ? -32768 : v[i] > 32767 ? 32767 : v[i]));
    rt_log("input: test gamepad attached, sticks at %d,%d %d,%d", v[0], v[1], v[2], v[3]);
}

static void start(void) {
    binds_refresh();
    const char *e = cfg_get("AC_INPUT");
    if (e && !strcmp(e, "none")) { state = -1; rt_log("input: disabled (AC_INPUT=none)"); return; }
    if (!SDL_Init(SDL_INIT_GAMEPAD)) { state = -1; rt_log("input: SDL gamepad init failed (%s), keyboard only", SDL_GetError()); }
    else state = 1;
    if (state == 1) test_pad_attach();
    if (state == 1) {
        scan_pads();
        if (!pad) rt_log("input: no gamepad (keyboard only until one is connected)");
    }
    if (state == -1) state = 2;   // keyboard only
}

static void stick_bits(uint64_t *b, int32_t x, int32_t y, int first) {
    if (x < -STICK_DIR) *b |= 1ull << (first + 0);
    if (y > STICK_DIR) *b |= 1ull << (first + 1);
    if (x > STICK_DIR) *b |= 1ull << (first + 2);
    if (y < -STICK_DIR) *b |= 1ull << (first + 3);
}

static void pick(int32_t *x, int32_t *y, int32_t kx, int32_t ky) {
    int64_t a = (int64_t)*x * *x + (int64_t)*y * *y, b = (int64_t)kx * kx + (int64_t)ky * ky;
    if (b > a) { *x = kx; *y = ky; }
}

// Under lock, state 1: gamepad g's inputs by position (pd[PAD_*] != 0 held); 0 without g connected. The caller has
// updated the gamepads (pads_update).
static int pad_buttons(SDL_Gamepad *g, uint8_t pd[PAD_COUNT]) {
    if (!g || !SDL_GamepadConnected(g)) return 0;
    for (int p = PAD_SOUTH; p < PAD_ZL; p++) pd[p] = SDL_GetGamepadButton(g, pad_sdl[p]);
    int32_t on = trigger_on_get();
    pd[PAD_ZL] = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > on;
    pd[PAD_ZR] = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > on;
    for (int p = PAD_ZR + 1; p < PAD_COUNT; p++) pd[p] = SDL_GetGamepadButton(g, pad_sdl[p]);   // false without paddles
    return 1;
}

static void pads_update(void) {
    pads_scan();
    SDL_UpdateGamepads();
}

// Under lock, state 1: the first gamepad's inputs by position; 0 without a connected gamepad.
static int pad_physical(uint8_t pd[PAD_COUNT]) {
    pads_update();
    return pad_buttons(pad, pd);
}

// Under lock, state 1, after pads_update: gamepad g's buttons (Switch bits after AC_PAD_LAYOUT and the AC_BIND_* lines)
// and sticks; 0 without g connected.
static int pad_read_of(SDL_Gamepad *g, uint64_t *b, int32_t sl[2], int32_t sr[2]) {
    uint8_t pd[PAD_COUNT] = {0};
    if (!pad_buttons(g, pd)) return 0;
    *b |= input_map_pad_buttons(pd, g == pad ? pad_layout_get() : pad2_layout_get());
    stick(g, SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, stick_dead_get(0), sl);
    stick(g, SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY, stick_dead_get(1), sr);
    return 1;
}

// Under lock, state 1: the first gamepad's buttons and sticks; 0 without a connected gamepad.
static int pad_read(uint64_t *b, int32_t sl[2], int32_t sr[2]) {
    pads_update();
    return pad_read_of(pad, b, sl, sr);
}

// Under lock, co-op: player p's gamepad (NULL for none) and in *kb whether the keyboard and the mouse are theirs. The
// gamepads take the players in order (pad, then pad2), the keyboard the first player without one, or player 1 when
// both have one. Off (state <= 0, AC_INPUT=none): nothing.
static SDL_Gamepad *player_pad(int p, int *kb) {
    SDL_Gamepad *g[2];
    int n = 0;
    if (state == 1 && pad && SDL_GamepadConnected(pad)) g[n++] = pad;
    if (state == 1 && pad2 && SDL_GamepadConnected(pad2)) g[n++] = pad2;
    *kb = state > 0 && p == (n < 2 ? n : 0);
    return p < n ? g[p] : NULL;
}

int input_pad_stick_percent(double pct[2]) {
    int ok = 0;
    pct[0] = pct[1] = 0;
    plat_mutex_lock(&lock);
    if (state == 1 && pad && SDL_GamepadConnected(pad)) {   // the game's poll keeps the state fresh (SDL_UpdateGamepads)
        pct[0] = input_stick_percent(SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTX),
                                     SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTY));
        pct[1] = input_stick_percent(SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHTX),
                                     SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHTY));
        ok = 1;
    }
    plat_mutex_unlock(&lock);
    return ok;
}

int input_pad_trigger_percent(double pct[2]) {
    int ok = 0;
    pct[0] = pct[1] = 0;
    plat_mutex_lock(&lock);
    if (state == 1 && pad && SDL_GamepadConnected(pad)) {
        pct[0] = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) * 100.0 / STICK_MAX;
        pct[1] = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) * 100.0 / STICK_MAX;
        ok = 1;
    }
    plat_mutex_unlock(&lock);
    return ok;
}

// The keys and mouse buttons the last keyboard read saw down, a bit per KEY_* id (input_key_held): what the bindings
// were applied to, so a mouse button counts only while the cursor is captured, and nothing while another window is in
// front (poll_merged).
_Static_assert(KEY_COUNT <= 128, "keys_seen holds 128 keys");
static volatile int64_t keys_seen[2];
static void keys_seen_set(uint64_t lo, uint64_t hi) {
    __atomic_store_n(&keys_seen[0], (int64_t)lo, __ATOMIC_RELAXED);
    __atomic_store_n(&keys_seen[1], (int64_t)hi, __ATOMIC_RELAXED);
}

// Under lock, the window in front: the keyboard's and the mouse's buttons and sticks ORed and picked into b, sl, sr.
// *tilt: bit 0 set when keys tilt the left stick, bit 1 the right one (input_release_held: with the walk key they
// tilt it part way, under the direction bits).
static void kb_read(uint64_t *b, int32_t sl[2], int32_t sr[2], int *tilt) {
    uint8_t down[KEY_COUNT] = {0};
    // Mouse buttons count only while the cursor is captured, and a button already held when the capture began
    // (the click that refocused the window) is ignored until it is released.
    static int was_captured;
    static uint8_t mouse_ignore[KEY_COUNT];
    int cap = ikb_captured();
    for (int k = 1; k < KEY_COUNT; k++) {
        down[k] = ikb_down(k);
        if (k < KEY_MOUSE_L || k > KEY_MOUSE_X2) continue;
        if (cap && !was_captured) mouse_ignore[k] = down[k];
        if (!down[k]) mouse_ignore[k] = 0;
        if (!cap || mouse_ignore[k]) down[k] = 0;
    }
    was_captured = cap;
    uint64_t seen[2] = {0, 0};
    for (int k = 1; k < KEY_COUNT; k++) seen[k >> 6] |= (uint64_t)down[k] << (k & 63);
    keys_seen_set(seen[0], seen[1]);
    *b |= input_map_buttons(down);
    int32_t k[2];
    walk_get();
    int walk = walk_key != KEY_NONE && down[walk_key];
    input_map_stick(down, 0, k);
    *tilt |= (k[0] | k[1]) != 0;
    if (walk) k[0] = (int32_t)(k[0] * walk_scale), k[1] = (int32_t)(k[1] * walk_scale);
    pick(&sl[0], &sl[1], k[0], k[1]);
    input_map_stick(down, 1, k);
    *tilt |= ((k[0] | k[1]) != 0) << 1;
    if (walk) k[0] = (int32_t)(k[0] * walk_scale), k[1] = (int32_t)(k[1] * walk_scale);
    pick(&sr[0], &sr[1], k[0], k[1]);
    mouse_camera(k);
    pick(&sr[0], &sr[1], k[0], k[1]);
}

// who: -1 the solo controller (the first gamepad, the keyboard and the mouse merged), 0 or 1 a co-op player
// (player_pad). The recording follows the solo controller, or player 1 in co-op. *held: what the sources hold now, for
// the release gate (input_release_held), read before the chords and the open menu take anything out of the sample.
static void poll_merged(int who, uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry, uint32_t *held) {
    uint64_t b = 0;
    int32_t sl[2] = {0, 0}, sr[2] = {0, 0};
    int tilt = 0;
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state > 0) {
        binds_refresh();
        if (state == 1) pads_update();
        int kb = who < 0;
        SDL_Gamepad *g = who < 0 ? pad : player_pad(who, &kb);
        if (state == 1 && pad_read_of(g, &b, sl, sr) && who >= 0) {
            // Co-op draws its prompts as places on a Joy-Con held sideways (a dot among four), never as letters, so a
            // co-op player's face buttons go by place: the swaps of the Xbox layout (A with B, X with Y) and of the
            // PlayStation one (A with B) are taken back.
            int layout = g == pad ? pad_layout_get() : pad2_layout_get();
            uint64_t ab = layout != PAD_LAYOUT_NINTENDO ? 1ull << B_A | 1ull << B_B : 0;
            uint64_t xy = layout == PAD_LAYOUT_XBOX ? 1ull << B_X | 1ull << B_Y : 0;
            if ((b & ab) && (b & ab) != ab) b ^= ab;
            if ((b & xy) && (b & xy) != xy) b ^= xy;
        }
        if (kb && ikb_foreground()) kb_read(&b, sl, sr, &tilt);
        else if (kb) keys_seen_set(0, 0);
    }
    script_apply(who > 0, &b, sl, sr);
    if (who == (players > 1 ? 0 : -1)) rec_apply(b, sl, sr);
    uint64_t raw = b;
    stick_bits(&raw, sl[0], sl[1], B_SL_LEFT);
    stick_bits(&raw, sr[0], sr[1], B_SR_LEFT);
    *held = input_release_held(raw, tilt);
    // Minus and Plus held together (gamepad or keys) open or close the menu and never reach the game; each source of
    // polls keeps its own memory of the chord
    static int combo_held_of[3], overlay_held_of[3];
    int *combo = &combo_held_of[who + 1], *overlay_held = &overlay_held_of[who + 1];
    int32_t st[4] = {sl[0], sl[1], sr[0], sr[1]};
    // L+R+Minus+Plus step the performance overlay (F10); checked first, so it takes Minus and Plus from the menu's chord
    int overlay = *combo ? 0 : input_overlay_gate(overlay_held, &b);
    int toggle = input_menu_gate(combo, menu_is_open(), &b, st);
    sl[0] = st[0], sl[1] = st[1], sr[0] = st[2], sr[1] = st[3];
    plat_mutex_unlock(&lock);
    if (toggle) {
        rt_log("input: Minus and Plus held together%s: the menu's chord", who < 0 ? "" : who ? " (player 2)" : " (player 1)");
        menu_toggle();
    }
    if (overlay) menu_perf_cycle();
    stick_bits(&b, sl[0], sl[1], B_SL_LEFT);
    stick_bits(&b, sr[0], sr[1], B_SR_LEFT);
    *buttons = b; *lx = sl[0]; *ly = sl[1]; *rx = sr[0]; *ry = sr[1];
}

// Text entry (swkbd_*.c) and the in-game menu (ui/menu.cpp): while one is open the game reads no input, as behind the
// keyboard applet, and the cursor is released (for the menu's mouse). After it closes, what is held at that moment
// stays hidden from the game until it is released, so the press that closed it stays there: the release gate
// (input.h, which also says why the game no longer waits for a pad that reads all zero). g_wait holds one word per
// source of polls (the solo controller, co-op's players 1 and 2: each waits for its own inputs), all set when either
// closes: the inputs still waited for (INPUT_RELEASE_ALL's bits) and, above them, the log lines that close still
// owes (NOTE_*). It is 0 the rest of the time, which is all a poll in play looks at. A poll only ever clears bits,
// with one atomic AND whose old value says which lines are its own to write, so the game's thread and the cheats'
// poll on the presenting thread (cheats.c) can share the solo controller's word.
//
// The lines, a handful per close and none in play, are there to tell from a player's log why a game would not react
// after the menu: the first read after the close (how long after it, what is held and hidden, what the pad reads
// before the dead zones and the trigger threshold, and what the game gets), the release of what was held, an input
// still hidden 5 s later (nobody holds a button that long by intent: it reads pressed at rest, a trigger past
// trigger_threshold or a stick past half its travel, or it is stuck), and the first button the game gets again. No
// first read after a "menu: closed" line means the game does not read its pad; a button the game gets and does not
// act on means the fault lies past this file.
enum { NOTE_READ = 1u << 24, NOTE_BUTTON = 1u << 25, NOTE_LATE = 1u << 26 };
#define RELEASE_LATE_NS 5000000000ull
static volatile int32_t g_block;     // a text entry is open
static volatile int32_t g_menu;      // the in-game menu is open (input_menu)
static volatile uint32_t g_wait[3];  // by who + 1 (poll_merged)
static volatile uint64_t g_closed_ns;   // plat_now_ns at the last close

int input_cursor_allowed(void) {
    const char *in = cfg_get("AC_INPUT");
    return !g_block && !g_menu && mouse_sens_get() > 0 && !(in && !strcmp(in, "none"));
}

// The words first, the flag after: a poll that still finds the flag set reports nothing and leaves them alone.
static void release_arm(void) {
    __atomic_store_n(&g_closed_ns, plat_now_ns(), __ATOMIC_RELAXED);
    for (int i = 0; i < 3; i++)
        __atomic_store_n(&g_wait[i], INPUT_RELEASE_ALL | NOTE_READ | NOTE_BUTTON | NOTE_LATE, __ATOMIC_SEQ_CST);
}

void input_block(int on) {
    if (!on) release_arm();
    plat_xchg32(&g_block, on != 0);
    ikb_cursor_update();
}

void input_menu(int open) {
    if (!open) release_arm();
    plat_xchg32(&g_menu, open != 0);
    if (!open) ikb_menu_closed();   // the capture before Esc comes back
    ikb_cursor_update();
}

// The inputs of bits (buttons and stick direction bits) by name, "nothing" for none.
static const char *input_names(uint32_t bits, char *out, size_t n) {
    size_t len = 0;
    out[0] = 0;
#define NAME(s) (len += (size_t)snprintf(out + len, len < n ? n - len : 0, "%s%s", len ? ", " : "", s))
    for (size_t i = 0; i < sizeof script_buttons / sizeof *script_buttons; i++)
        if (bits >> script_buttons[i].bit & 1) NAME(script_buttons[i].name);
    if (bits & 0xfu << B_SL_LEFT) NAME("the left stick");
    if (bits & 0xfu << B_SR_LEFT) NAME("the right stick");
#undef NAME
    return len ? out : "nothing";
}

// What the gamepad of source who reads now, before the dead zones and the trigger threshold, and those.
static void pad_reads_text(int who, char *out, size_t n) {
    int kb;
    plat_mutex_lock(&lock);
    SDL_Gamepad *g = state != 1 ? NULL : who < 0 ? pad : player_pad(who, &kb);
    if (g && SDL_GamepadConnected(g)) {
        int32_t lx = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX), ly = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY);
        int32_t rx = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX), ry = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY);
        snprintf(out, n, "gamepad \"%s\" reads sticks %+d,%+d (%.1f %%) and %+d,%+d (%.1f %%), triggers %.1f %% and %.1f %%, "
                 "before dead zones of %.0f %% and %.0f %% and a trigger threshold of %.0f %%", SDL_GetGamepadName(g),
                 lx, ly, input_stick_percent(lx, ly), rx, ry, input_stick_percent(rx, ry),
                 SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) * 100.0 / STICK_MAX,
                 SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) * 100.0 / STICK_MAX,
                 stick_dead_get(0) * 100.0, stick_dead_get(1) * 100.0, trigger_on_get() * 100.0 / STICK_MAX);
    } else {
        snprintf(out, n, "no gamepad%s", state == 1 ? "" : state > 0 ? " (SDL's gamepads did not start)" : " (input off)");
    }
    plat_mutex_unlock(&lock);
}

// The game's sample of source who after a close, w its word: without what was held at the close until that is
// released, and the lines this poll owes.
static __attribute__((noinline, cold)) void game_gate_closed(int who, uint32_t w, uint32_t held, uint64_t *buttons,
                                                               int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    int32_t st[4] = {*lx, *ly, *rx, *ry};
    uint32_t n = input_release_gate(w & INPUT_RELEASE_ALL, held, buttons, st);
    *lx = st[0], *ly = st[1], *rx = st[2], *ry = st[3];
    uint64_t now = plat_now_ns(), closed = __atomic_load_n(&g_closed_ns, __ATOMIC_RELAXED);
    double ms = now > closed ? (double)(now - closed) / 1e6 : 0;
    uint32_t clear = (w & INPUT_RELEASE_ALL & ~n) | NOTE_READ;
    if (*buttons) clear |= NOTE_BUTTON;
    if (!n || ms >= RELEASE_LATE_NS / 1e6) clear |= NOTE_LATE;
    if (!(w & clear)) return;
    uint32_t old = __atomic_fetch_and(&g_wait[who + 1], ~clear, __ATOMIC_RELAXED), did = old & clear;
    const char *whose = who < 0 ? "" : who ? " (player 2)" : " (player 1)";
    char names[160], reads[320];
    if (did & NOTE_READ) {
        pad_reads_text(who, reads, sizeof reads);
        rt_log("input: first read %.0f ms after the menu or the text entry closed%s: %s held%s; %s; the game gets "
               "buttons %#llx and sticks %d,%d %d,%d", ms, whose, input_names(n, names, sizeof names),
               n ? " (hidden from the game until released)" : "", reads, (unsigned long long)*buttons, st[0], st[1],
               st[2], st[3]);
    } else if ((did & INPUT_RELEASE_ALL) && !(old & ~clear & INPUT_RELEASE_ALL)) {
        rt_log("input: what was held at the close is released %.0f ms after it%s", ms, whose);
    }
    if ((did & NOTE_LATE) && n)
        rt_log("input: %.0f s after the menu or the text entry closed the game still does not get %s%s: held since then, "
               "or reading pressed at rest (a trigger past trigger_threshold, a stick past half its travel); it comes "
               "back once it reads released", ms / 1e3, input_names(n, names, sizeof names), whose);
    if (did & NOTE_BUTTON)
        rt_log("input: the game gets a button again %.0f ms after the close%s: %s", ms, whose,
               input_names((uint32_t)*buttons & INPUT_RELEASE_ALL, names, sizeof names));
}

// The game's sample of source who (poll_merged): nothing while the text entry or the menu is open.
static inline void game_gate(int who, uint32_t held, uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    if (__builtin_expect(g_block | g_menu, 0)) {
        *buttons = 0; *lx = *ly = *rx = *ry = 0;
        return;
    }
    uint32_t w = __atomic_load_n(&g_wait[who + 1], __ATOMIC_RELAXED);
    if (__builtin_expect(w != 0, 0)) game_gate_closed(who, w, held, buttons, lx, ly, rx, ry);
}

int input_key_held(int key) {
    if (key <= KEY_NONE || key >= KEY_COUNT || (g_block | g_menu)) return 0;
    return (int)((uint64_t)__atomic_load_n(&keys_seen[key >> 6], __ATOMIC_RELAXED) >> (key & 63) & 1);
}

void input_poll(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    uint32_t held;
    poll_merged(-1, buttons, lx, ly, rx, ry, &held);
    game_gate(-1, held, buttons, lx, ly, rx, ry);
}

void input_poll_ui(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    uint32_t held;
    poll_merged(-1, buttons, lx, ly, rx, ry, &held);
}

void input_poll_paused(int away) {
    uint64_t b;
    int32_t lx, ly, rx, ry;
    uint32_t held;
    if (away) {
        // SDL sends what a pad's driver put off (a rumble stop included) and renews or expires a rumble only in here
        plat_mutex_lock(&lock);
        if (state == 1) SDL_UpdateGamepads();
        plat_mutex_unlock(&lock);
        return;
    }
    // each source keeps its own memory of the chords (poll_merged), so poll the ones the game polls: another would see
    // a chord still held from opening the menu as a new press and close it
    if (players > 1) {
        poll_merged(0, &b, &lx, &ly, &rx, &ry, &held);
        poll_merged(1, &b, &lx, &ly, &rx, &ry, &held);
    } else {
        poll_merged(-1, &b, &lx, &ly, &rx, &ry, &held);
    }
}

void input_players(int n) {
    n = n > 1 ? 2 : 1;
    plat_mutex_lock(&lock);
    if (n != players) {
        players = n;
        next_scan2 = 0;
        if (n == 1 && pad2) SDL_CloseGamepad(pad2), pad2 = NULL;
        rt_log("input: %s", n > 1 ? "two players (co-op): gamepads in order, the keyboard and the mouse to the first "
                                    "player without one" : "one player again: the first gamepad, the keyboard and the mouse");
    }
    plat_mutex_unlock(&lock);
}

unsigned input_players_connected(void) {
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state == 1) pads_scan();
    if (!script_state) script_load();
    int kb;
    unsigned m = 1u | (player_pad(1, &kb) || kb || script_p2 ? 2u : 0u);   // player 1 is the solo controller: always there
    plat_mutex_unlock(&lock);
    return m;
}

// Each player behind the text entry and the menu as the solo controller is (game_gate), with a wait word of their own.
int input_poll_player(int player, uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry) {
    int on = (int)(input_players_connected() >> (player & 1) & 1);
    *buttons = 0, *lx = *ly = *rx = *ry = 0;
    if (!on) return 0;
    uint32_t held;
    poll_merged(player & 1, buttons, lx, ly, rx, ry, &held);
    game_gate(player & 1, held, buttons, lx, ly, rx, ry);
    return 1;
}

int input_pad_read(uint64_t *buttons, int32_t *lx, int32_t *ly) {
    uint64_t b = 0;
    int32_t sl[2] = {0, 0}, sr[2] = {0, 0};
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state > 0) binds_refresh();
    int ok = state == 1 && pad_read(&b, sl, sr);
    plat_mutex_unlock(&lock);
    *buttons = b;
    *lx = sl[0];
    *ly = sl[1];
    return ok;
}

uint16_t rumble_motor(float a, float curve, float strength) {
    if (!(a > 0) || !(strength > 0)) return 0;
    float v = powf(a > 1 ? 1 : a, curve) * strength * 65535.0f + 0.5f;
    return v >= 65535.0f ? 65535 : (uint16_t)v;
}

int input_pad_layout(void) {
    return (int)plat_cas32(&pad_layout_shown, 0, 0);
}

int input_pad_kind(void) {
    return (int)plat_cas32(&pad_kind_shown, 0, 0);
}

// *none_found: the scan ran and found no gamepad at all (not: a gamepad of another kind, or input switched off).
static int detect_pad_kind(int *none_found) {
    *none_found = 0;
    const char *e = cfg_get("AC_INPUT");
    if (e && !strcmp(e, "none")) return PAD_KIND_NONE;
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        rt_log("input: auto prompts: SDL gamepad init failed (%s)", SDL_GetError());
        return PAD_KIND_NONE;
    }
    SDL_UpdateGamepads();
    int n = 0, kind = PAD_KIND_NONE;
    SDL_JoystickID *ids = SDL_GetGamepads(&n);
    SDL_Gamepad *found = NULL;
    for (int i = 0; ids && i < n && !found; i++) found = SDL_OpenGamepad(ids[i]);
    SDL_free(ids);
    if (found) {
        kind = kind_of(found);
        SDL_CloseGamepad(found);
    }
    *none_found = !found;
    return kind;
}

int input_detect_pad_kind(void) {
    int none_found;
    return detect_pad_kind(&none_found);
}

// The controller the launcher that started the game sees (input_map.h input_pad_kind_named).
static int launcher_pad_kind(void) {
    return input_pad_kind_named(getenv("RECOMPLAUNCHER_PAD"));
}

int input_prompts_auto(void) {
    int none_found, kind = detect_pad_kind(&none_found);
    if (none_found && launcher_pad_kind() != PAD_KIND_NONE) {
        kind = launcher_pad_kind();
        rt_log("input: auto prompts: no gamepad found, the launcher names a%s controller",
               kind == PAD_KIND_XBOX ? "n Xbox" : " PlayStation");
    }
    return input_prompts_auto_for_kind(kind);
}

int input_prompts_auto_live(void) {
    int kind = input_pad_kind();
    if (kind == PAD_KIND_NONE && input_pad_layout() < 0) kind = launcher_pad_kind();
    return input_prompts_auto_for_kind(kind);
}

int input_pad_label(int pad_id) {
    if (pad_id < PAD_SOUTH || pad_id > PAD_NORTH) return PAD_LABEL_NONE;
    return (int)plat_cas32(&pad_labels[pad_id - PAD_SOUTH], 0, 0);
}

void input_map_snapshot(InputMap *m) {
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state > 0) binds_refresh();
    *m = input_map;
    plat_mutex_unlock(&lock);
}

int input_capture(int reset, int *id) {
    static uint8_t prev_k[KEY_COUNT], prev_p[PAD_COUNT];
    static int prev_esc;
    uint8_t k[KEY_COUNT] = {0}, p[PAD_COUNT] = {0};
    int fg = ikb_foreground();
    for (int i = 1; i < KEY_COUNT; i++) k[i] = fg && ikb_down(i);
    int esc = fg && ikb_escape();
    plat_mutex_lock(&lock);
    if (!state) start();
    if (state == 1) pad_physical(p);
    plat_mutex_unlock(&lock);
    int got = INPUT_CAPTURE_NONE;
    if (!reset) {
        if (esc && !prev_esc) got = INPUT_CAPTURE_CANCEL;
        for (int i = 1; i < KEY_COUNT && !got; i++)
            if (k[i] && !prev_k[i]) got = INPUT_CAPTURE_KEY, *id = i;
        for (int i = 1; i < PAD_COUNT && !got; i++)
            if (p[i] && !prev_p[i]) got = INPUT_CAPTURE_PAD, *id = i;
    }
    memcpy(prev_k, k, sizeof k);
    memcpy(prev_p, p, sizeof p);
    prev_esc = esc;
    return got;
}

int input_rumble_triggers_enabled(void) {
    return cfg_bool("AC_RUMBLE_TRIGGERS");
}

int input_rumble(uint16_t low, uint16_t high, uint32_t ms) {
    int ok = 0;
    static int triggers = -1;
    static unsigned seen;
    if (triggers < 0 || seen != cfg_version()) seen = cfg_version(), triggers = input_rumble_triggers_enabled();
    plat_mutex_lock(&lock);
    if (state == 1 && pad && pad_rumble) {
        ok = SDL_RumbleGamepad(pad, low, high, ms);
        if (triggers) SDL_RumbleGamepadTriggers(pad, high, high, ms);   // unsupported pads return false: ignored
    }
    plat_mutex_unlock(&lock);
    return ok;
}

int input_rumble_player(int player, uint16_t low, uint16_t high, uint32_t ms) {
    int ok = 0, kb;
    static int triggers = -1;
    static unsigned seen;
    if (triggers < 0 || seen != cfg_version()) seen = cfg_version(), triggers = input_rumble_triggers_enabled();
    plat_mutex_lock(&lock);
    SDL_Gamepad *g = player_pad(player & 1, &kb);
    if (g && (g == pad ? pad_rumble : pad2_rumble)) {
        ok = SDL_RumbleGamepad(g, low, high, ms);
        if (triggers) SDL_RumbleGamepadTriggers(g, high, high, ms);
    }
    plat_mutex_unlock(&lock);
    return ok;
}

// Not left to the 250 ms a send lasts (ov_input.c rumble_update): that is a quarter second of vibration on a still
// picture. The triggers too, whatever rumble_triggers says now: it may have been switched off since the last send.
void input_rumble_stop(void) {
    plat_mutex_lock(&lock);
    if (state == 1) {
        SDL_Gamepad *const g[2] = {pad, pad2};
        for (int i = 0; i < 2; i++)
            if (g[i]) {
                SDL_RumbleGamepad(g[i], 0, 0, 0);
                SDL_RumbleGamepadTriggers(g[i], 0, 0, 0);
            }
    }
    plat_mutex_unlock(&lock);
}

// SDL gamepad sensor axes (SDL_sensor.h, a gamepad held in front of you): x to the right, y up out of the face (lying
// flat face up, +y points at the ceiling), z toward the player (out of the bottom edge). The accelerometer reads the
// push against gravity in m/s^2 (lying flat: y = +9.81); the gyro reads rad/s, counter-clockwise positive (right hand).
// Switch six-axis axes (nn::hid, as MacNativeGames' GCMotion feed and its Pro Controller HID path use them): x to the
// right, y toward the top edge (away from the player), z out of the face; acceleration as gravity reads (lying flat:
// z = -1 G). So Switch x = SDL x, Switch y = -SDL z, Switch z = SDL y: a rotation (right-handed to right-handed), so the
// gyro keeps its sense, while the acceleration is also negated (push -> gravity) and scaled to G.
void input_motion_axes(const float a[3], const float g[3], double acc[3], double gyro[3]) {
    acc[0] = -a[0] / SDL_STANDARD_GRAVITY;
    acc[1] = a[2] / SDL_STANDARD_GRAVITY;
    acc[2] = -a[1] / SDL_STANDARD_GRAVITY;
    gyro[0] = g[0];
    gyro[1] = -g[2];
    gyro[2] = g[1];
}

// Under lock: (re)enable the sensors when the open gamepad changed; 1 if it has both and they are on.
static int sensors_ready(void) {
    SDL_JoystickID id = SDL_GetGamepadID(pad);
    if (id == sensor_pad) return sensor_ok;
    sensor_pad = id;
    // Nothing drains the SDL event queue: no event per sample (the data still updates, SDL_SendJoystickSensor)
    SDL_SetEventEnabled(SDL_EVENT_GAMEPAD_SENSOR_UPDATE, false);
    sensor_ok =SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO) && SDL_GamepadHasSensor(pad, SDL_SENSOR_ACCEL) &&
                SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_GYRO, true) &&
                SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_ACCEL, true);
    if (sensor_ok) rt_log("input: motion sensors on (gyro %.0f Hz, accelerometer %.0f Hz)",
                          (double)SDL_GetGamepadSensorDataRate(pad, SDL_SENSOR_GYRO),
                          (double)SDL_GetGamepadSensorDataRate(pad, SDL_SENSOR_ACCEL));
    else rt_log("input: \"%s\" has no gyro and accelerometer: the six-axis sensor reads a controller at rest",
                SDL_GetGamepadName(pad));
    return sensor_ok;
}

int input_motion_enabled(void) {
    static int on = -1;
    static unsigned seen;
    unsigned v = cfg_version();
    if (on < 0 || v != seen) {
        seen = v;
        on = cfg_bool("AC_MOTION");
    }
    return on;
}

// 250 Hz: the gamepads update (reports read, rumble expiry) and the newest gyro and accelerometer sample goes into the
// fusion. Without a gamepad with sensors, or with motion off (AC_MOTION=0, live), nothing is fed and the six-axis
// sensor reads at rest.
static void motion_main(void *arg) {
    (void)arg;
    for (;;) {
        plat_wait_ns(4000000);
        float a[3], g[3];
        int have = 0;
        plat_mutex_lock(&lock);
        if (pad && SDL_GamepadConnected(pad)) {
            SDL_UpdateGamepads();
            have = sensors_ready() && SDL_GetGamepadSensorData(pad, SDL_SENSOR_ACCEL, a, 3) &&
                   SDL_GetGamepadSensorData(pad, SDL_SENSOR_GYRO, g, 3);
        }
        plat_mutex_unlock(&lock);
        if (!have || !input_motion_enabled()) continue;
        double acc[3], gyro[3];
        input_motion_axes(a, g, acc, gyro);
        static unsigned turn_seen;
        static int turn_set;
        if (!turn_set || cfg_version() != turn_seen) {   // live: read again when the settings change
            turn_set = 1;
            turn_seen = cfg_version();
            motion_set_turn(motion_global(), cfg_int("AC_MOTION_TURN"));
        }
        motion_feed(motion_global(), acc, gyro, kern_now_ns());
    }
}

void input_motion_start(void) {
    plat_mutex_lock(&lock);
    if (!state) start();
    if (!motion_thread) {
        if (!input_motion_enabled()) rt_log("input: motion off (AC_MOTION=0); it can be turned on while the game runs");
        if (state != 1) motion_thread = -1;
        else {
            if (!plat_thread_spawn(motion_main, NULL, "motion", 0)) rt_fatal("input: cannot start the motion thread");
            motion_thread = 1;
        }
    }
    plat_mutex_unlock(&lock);
}
