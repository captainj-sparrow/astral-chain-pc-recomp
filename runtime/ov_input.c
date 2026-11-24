// Adapted from MacNativeGames games/totk/runtime/input.c and hid_feedback.c @ b75be8cf (MIT, Copyright (c) 2026
// MacNativeGames contributors).
// nn::hid overrides: one connected controller fed by input_poll (gamepad and keyboard,
// runtime/input_win.c; nothing pressed with AC_INPUT=none). Docked (default) it is a full-key controller as player 1 (id 0, style bit 0); with
// AC_OPERATION_MODE=handheld it is the handheld controller (id 0x20, style bit 1) and player 1 is not connected,
// which is the controller the game reads in that mode. In co-op (coop_update) players 1 and 2 are a left and a right
// Joy-Con held sideways, fed by input_poll_player. Mouse and keyboard report no input. The controller has two
// vibration devices (left, right) played on the gamepad's motors (AC_RUMBLE=0: none) and a six-axis sensor fed by
// the gamepad's gyro and accelerometer (AC_MOTION=0, live: reads at rest). State layouts follow libnx HID.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "input.h"
#include "ov.h"
#include "platform/platform.h"

typedef struct { int64_t sampling; uint64_t buttons; int32_t lx, ly, rx, ry; uint32_t attributes, pad; } NpadState;
typedef struct { int64_t sampling; int32_t x, y, dx, dy, wx, wy; uint32_t buttons, attributes; } MouseState;
typedef struct { int64_t sampling; uint32_t modifiers, attributes; uint64_t keys[4]; } KeyboardState;
typedef struct { int8_t players; uint8_t pad[3]; uint32_t selected; } ControllerSupportResult;
_Static_assert(sizeof(NpadState) == 40 && sizeof(MouseState) == 40 && sizeof(KeyboardState) == 48, "hid state ABI");
_Static_assert(sizeof(ControllerSupportResult) == 8, "controller support result ABI");

enum { STYLE_FULL_KEY = 1, STYLE_HANDHELD = 2, STYLE_JOY_DUAL = 4, STYLE_JOY_LEFT = 8, STYLE_JOY_RIGHT = 0x10 };
// NpadAttribute: connected, and which Joy-Con of a single or dual controller is attached
enum { ATTR_CONNECTED = 1, ATTR_LEFT_CONNECTED = 4, ATTR_RIGHT_CONNECTED = 0x10 };
// Raw buttons of a single Joy-Con (libnx HidNpadButton): its side rail's SL and SR
enum { B_LEFT_SL = 24, B_LEFT_SR = 25, B_RIGHT_SL = 26, B_RIGHT_SR = 27 };

static PlatMutex lock = PLAT_MUTEX_INIT;
static uint32_t styles = STYLE_FULL_KEY;
static int enabled, allowed = 1, mouse_ready, keyboard_ready;
static int64_t sample;

// Co-op. The game sets the hid up two ways (Astral Chain 1.0.1): alone, ids {player 1, handheld} with styles
// full-key | handheld | Joy-Con pair (main+0x9af8e0); in co-op, ids {player 1, player 2} with styles handheld | left
// Joy-Con | right Joy-Con and no full-key (main+0x9afb10), then ShowControllerSupport for exactly two players
// (main+0x9afab8). Co-op is on while player 2's id and a single Joy-Con style are both supported: player 1 (id 0),
// who plays the character, is then a right Joy-Con and player 2 (id 1), who plays the Legion, a left one, as the
// game's own co-op guide draws them (Plus on the character's, Minus on the Legion's); each is held sideways and fed
// by input_poll_player (coop_player).
static uint32_t ids_supported = ~0u;   // bit per id: players 1-8 bits 0-7, other (0x10) bit 8, handheld (0x20) bit 9
static int coop;                       // lock
static volatile int64_t pad_buttons;   // the one player's buttons at the game's last read (ov_pad_buttons)

uint64_t ov_pad_buttons(void) { return (uint64_t)__atomic_load_n(&pad_buttons, __ATOMIC_RELAXED); }

static uint32_t id_bit(uint32_t id) { return id < 8 ? 1u << id : id == 0x10 ? 1u << 8 : id == 0x20 ? 1u << 9 : 0; }

static uint32_t pad_id(void) { return ov_handheld() ? 0x20u : 0u; }
static uint32_t pad_style(void) { return ov_handheld() ? STYLE_HANDHELD : STYLE_FULL_KEY; }

// Under lock, co-op: the style of player p (0 or 1), the right Joy-Con for player 1 and the left one for player 2 when
// the game supports both, else whichever single Joy-Con it supports.
static uint32_t coop_style(int p) {
    uint32_t want = p ? STYLE_JOY_LEFT : STYLE_JOY_RIGHT, other = want ^ (STYLE_JOY_LEFT | STYLE_JOY_RIGHT);
    return styles & want ? want : styles & other;
}

// After a change of the supported ids or styles: co-op on or off, told to input.c outside the lock.
static void coop_update(void) {
    plat_mutex_lock(&lock);
    int now = (ids_supported & id_bit(1)) && (styles & (STYLE_JOY_LEFT | STYLE_JOY_RIGHT));
    int was = coop;
    coop = now;
    uint32_t st = styles, ids = ids_supported;
    plat_mutex_unlock(&lock);
    if (now == was) return;
    rt_log("hid: %s (supported ids %#x, styles %#x)", now ? "co-op: two players, each on a single Joy-Con held sideways"
                                                           : "one player again", ids, st);
    input_players(now ? 2 : 1);
}

// A Switch-layout controller (input_poll's bits, sticks up +y) as the raw state of a single Joy-Con held sideways.
// The game turns that state back itself (main+0x9ae930 left, main+0x9aed80 right): left Joy-Con x = -ly, y = -lx;
// right Joy-Con x = ry, y = rx (its y grows downwards), so the stick is turned the other way here. Its buttons go
// into the bits of the full controller's buttons at the same place (main+0x9af1d0 against the readers): the bottom
// face button does what B does, the right one A, the top one X, the left one Y; SL does L, SR does R, the one system
// button Plus and the stick click the left stick's. ZL and ZR, which a single Joy-Con lacks, act as SL and SR; Minus
// as the system button; the D-pad and the right stick have no place.
static void joycon_sideways(int right, uint64_t in, int32_t sx, int32_t sy, NpadState *s) {
    static const struct { uint8_t from, left, right; } map[] = {
        {1, 12, 0},   // B -> bottom: Left / A
        {0, 15, 2},   // A -> right: Down / X
        {2, 14, 3},   // X -> top: Right / Y
        {3, 13, 1},   // Y -> left: Up / B
        {6, B_LEFT_SL, B_RIGHT_SL}, {8, B_LEFT_SL, B_RIGHT_SL},    // L, ZL -> SL
        {7, B_LEFT_SR, B_RIGHT_SR}, {9, B_LEFT_SR, B_RIGHT_SR},    // R, ZR -> SR
        {10, 11, 10}, {11, 11, 10},                                // Plus, Minus -> Minus / Plus
        {4, 4, 5}, {5, 4, 5},                                      // stick clicks -> the Joy-Con's stick click
    };
    uint64_t b = 0;
    for (size_t i = 0; i < sizeof map / sizeof *map; i++)
        if (in >> map[i].from & 1) b |= 1ull << (right ? map[i].right : map[i].left);
    int32_t x = right ? -sy : sy, y = right ? sx : -sx;   // raw axes: left (lx, ly) = (sy, -sx), right (rx, ry) = (-sy, sx)
    int first = right ? 20 : 16;             // the raw stick's direction bits (StickL/R Left, Up, Right, Down)
    if (x < -16384) b |= 1ull << first;
    if (y > 16384) b |= 1ull << (first + 1);
    if (x > 16384) b |= 1ull << (first + 2);
    if (y < -16384) b |= 1ull << (first + 3);
    s->buttons = b;
    if (right) s->rx = x, s->ry = y;
    else s->lx = x, s->ly = y;
    s->attributes = ATTR_CONNECTED | (right ? ATTR_RIGHT_CONNECTED : ATTR_LEFT_CONNECTED);
}

static const uint32_t *need_id(Cpu *c, int i, const char *fn) {
    const uint32_t *id = arg_p(c, i);
    if (!id) rt_fatal("nn::hid::%s: null controller id", fn);
    return id;
}

// void InitializeNpad()
static void ov_initialize_npad(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    enabled = 1;
    plat_mutex_unlock(&lock);
    rt_log("hid: controllers initialized (one connected %s controller)",
           ov_handheld() ? "handheld" : "full-key");
}

// void SetSupportedNpadIdType(const NpadIdType *ids, size_t count)
static void ov_supported_ids(Cpu *c) {
    const uint32_t *ids = arg_p(c, 0);
    uint64_t n = arg_x(c, 1);
    if (n > 10 || (!ids && n)) rt_fatal("nn::hid::SetSupportedNpadIdType: invalid id list (%llu ids)", (unsigned long long)n);
    int ok = 0;
    uint32_t bits = 0;
    for (uint64_t i = 0; i < n; i++) ok |= ids[i] == pad_id(), bits |= id_bit(ids[i]);
    plat_mutex_lock(&lock);
    allowed = ok;
    ids_supported = bits;
    plat_mutex_unlock(&lock);
    coop_update();
}

// void SetSupportedNpadStyleSet(NpadStyleSet styles)
static void ov_supported_styles(Cpu *c) {
    plat_mutex_lock(&lock);
    styles = (uint32_t)arg_x(c, 0);
    plat_mutex_unlock(&lock);
    coop_update();
}

// Co-op player p's state as its Joy-Con; 0 (and s left disconnected) without a source for that player.
// A single Joy-Con has one stick, and the game gives it two jobs on the Legion's side (its co-op guide): alone it turns
// the camera, with SL held it moves the Legion (SL alone calls the Legion out, SL + SR is the chain jump); the
// character's stick only moves the character. A gamepad, or the keyboard with the mouse, has two sticks, so here the
// Legion player's left stick moves the Legion (SL is held for as long as it is tilted) and the right stick turns the
// camera; the character player's right stick turns the camera too, whenever the Legion player's sticks are at rest.
static int coop_player(int p, int right, NpadState *s) {
    static PlatMutex cam_lock = PLAT_MUTEX_INIT;
    static int32_t cam[2];   // cam_lock: the character player's right stick at its last read
    uint64_t b;
    int32_t lx, ly, rx, ry;
    if (!input_poll_player(p, &b, &lx, &ly, &rx, &ry)) return 0;
    if (!p) {
        plat_mutex_lock(&cam_lock);
        cam[0] = rx, cam[1] = ry;
        plat_mutex_unlock(&cam_lock);
    } else if (lx || ly) {
        b |= 1ull << 6;   // L, which joycon_sideways turns into SL
    } else if (rx || ry) {
        lx = rx, ly = ry;
    } else {
        plat_mutex_lock(&cam_lock);
        lx = cam[0], ly = cam[1];
        plat_mutex_unlock(&cam_lock);
    }
    joycon_sideways(right, b, lx, ly, s);
    return 1;
}

// NpadStyleSet GetNpadStyleSet(const NpadIdType &id)
static void ov_style_set(Cpu *c) {
    uint32_t id = *need_id(c, 0, "GetNpadStyleSet");
    plat_mutex_lock_shared(&lock);
    int co = enabled && coop;
    uint32_t v = enabled && allowed && id == pad_id() ? styles & pad_style() : 0;
    if (co) v = id < 2 ? coop_style((int)id) : 0;
    plat_mutex_unlock_shared(&lock);
    if (co && v && !(input_players_connected() >> id & 1)) v = 0;
    ret_x(c, v);
}

// int GetNpadStates(<State> *out, int count, const NpadIdType &id): the newest state only. want: the style of the
// overload (the Joy-Con pair is never connected; the single Joy-Cons only in co-op).
static int npad_state(uint32_t want, uint32_t id, NpadState *s) {
    plat_mutex_lock(&lock);
    *s = (NpadState){.sampling = ++sample};
    int co = enabled && coop;
    int live = !co && want && enabled && allowed && id == pad_id() && want == pad_style() && (styles & want);
    int player = co && id < 2 && want && want == coop_style((int)id) ? (int)id : -1;
    plat_mutex_unlock(&lock);
    if (live || player >= 0) lowlat_input();   // vk_lowlat.c: the frame's input sample
    if (live) {
        s->attributes = ATTR_CONNECTED;
        input_poll(&s->buttons, &s->lx, &s->ly, &s->rx, &s->ry);
        __atomic_store_n(&pad_buttons, (int64_t)s->buttons, __ATOMIC_RELAXED);
    }
    if (player >= 0) live = coop_player(player, want == STYLE_JOY_RIGHT, s);
    return live;
}

static void npad_states(Cpu *c, uint32_t want) {
    NpadState *out = arg_p(c, 0);
    int max = (int)arg_x(c, 1);
    uint32_t id = *need_id(c, 2, "GetNpadStates");
    if (max <= 0) { ret_x(c, 0); return; }
    if (!out) rt_fatal("nn::hid::GetNpadStates: null output");
    NpadState s;
    npad_state(want, id, &s);
    *out = s;
    ret_x(c, 1);
}
static void ov_states_full_key(Cpu *c) { npad_states(c, STYLE_FULL_KEY); }
static void ov_states_handheld(Cpu *c) { npad_states(c, STYLE_HANDHELD); }
static void ov_states_joy_dual(Cpu *c) { npad_states(c, STYLE_JOY_DUAL); }
static void ov_states_joy_left(Cpu *c) { npad_states(c, STYLE_JOY_LEFT); }
static void ov_states_joy_right(Cpu *c) { npad_states(c, STYLE_JOY_RIGHT); }

// Result GetNpadState(<State> *out, const NpadIdType &id): the singular form the SDK's own GetNpadStates would serve.
static void npad_state_single(Cpu *c, uint32_t want) {
    NpadState *out = arg_p(c, 0);
    uint32_t id = *need_id(c, 1, "GetNpadState");
    if (!out) rt_fatal("nn::hid::GetNpadState: null output");
    NpadState s;
    npad_state(want, id, &s);
    *out = s;
    ret_x(c, 0);
}
static void ov_state_full_key(Cpu *c) { npad_state_single(c, STYLE_FULL_KEY); }
static void ov_state_handheld(Cpu *c) { npad_state_single(c, STYLE_HANDHELD); }
static void ov_state_joy_dual(Cpu *c) { npad_state_single(c, STYLE_JOY_DUAL); }
static void ov_state_joy_left(Cpu *c) { npad_state_single(c, STYLE_JOY_LEFT); }
static void ov_state_joy_right(Cpu *c) { npad_state_single(c, STYLE_JOY_RIGHT); }

// void SetNpadJoyHoldType(NpadJoyHoldType type)
static void ov_hold_type(Cpu *c) {
    int64_t t = (int32_t)arg_x(c, 0);
    if (t < 0 || t > 1) rt_fatal("nn::hid::SetNpadJoyHoldType: invalid hold type %lld", (long long)t);
}

// void DisconnectNpad(const NpadIdType &id): only the controller in use (handheld mode disconnects player 1)
static void ov_disconnect(Cpu *c) {
    uint32_t id = *need_id(c, 0, "DisconnectNpad");
    if (id != pad_id()) return;
    plat_mutex_lock(&lock);
    allowed = 0;
    plat_mutex_unlock(&lock);
}

// The controller applet, answered at once: alone the one controller is connected, so it returns with one player on
// it. In co-op it succeeds when the host has as many players as the game asks for (ControllerSupportArg's first byte,
// the minimum player count; the game asks for 2), and is cancelled otherwise (nn::hid's controller-support cancel,
// module 202 description 3101; the game only tests for success): there is no window to wait in for a second
// controller, so the game stays with one player. Returns the result, the player count in *players.
#define RESULT_CONTROLLER_SUPPORT_CANCELED 0x183acau
static uint32_t controller_support(const int8_t *arg, int *players) {
    plat_mutex_lock_shared(&lock);
    int co = coop;
    plat_mutex_unlock_shared(&lock);
    unsigned m = input_players_connected();
    int have = co ? __builtin_popcount(m) : 1, want = arg && arg[0] > 1 ? arg[0] : 1;
    *players = have < want ? 0 : have;
    if (co) rt_log("hid: controller applet for %d to %d players: %d connected, %s", want, arg ? arg[1] : 0, have,
                   have < want ? "cancelled (connect a second gamepad, or use the keyboard with one gamepad)" : "accepted");
    return have < want ? RESULT_CONTROLLER_SUPPORT_CANCELED : 0;
}

// Result ShowControllerSupport(ControllerSupportResultInfo *out, const ControllerSupportArg &arg)
static void ov_controller_support(Cpu *c) {
    ControllerSupportResult *out = arg_p(c, 0);
    int players;
    uint32_t r = controller_support(arg_p(c, 1), &players);
    if (out) *out = (ControllerSupportResult){.players = (int8_t)players, .selected = players > 1 ? 0 : pad_id()};
    ret_x(c, r);
}

// Result ShowControllerSupport(const ControllerSupportArg &arg)
static void ov_controller_support_1(Cpu *c) {
    int players;
    ret_x(c, controller_support(arg_p(c, 0), &players));
}

// The mouse-style and keyboard-style hosts above; the touch screen is never touched on a PC, so its state is always
// empty instead of serving the hid shared memory the SDK's own InitializeTouchScreen would map.
static void ov_initialize_touch(Cpu *c) {
    (void)c;
    ret_x(c, 0);
}

// void GetTouchScreenState<16>(TouchScreenState<16> *out): no touches. The state starts {s64 sampling number, s32
// count, s32 reserved} and the game reads the count at +8 (main+0x9adb44).
static void ov_touch_states(Cpu *c) {
    uint32_t *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::GetTouchScreenState: null output");
    memset(out, 0, 16);
    ret_x(c, 0);
}

static void ov_initialize_mouse(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    mouse_ready = 1;
    plat_mutex_unlock(&lock);
}

static void ov_initialize_keyboard(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    keyboard_ready = 1;
    plat_mutex_unlock(&lock);
}

// int GetMouseStates(MouseState *out, int count): a mouse that is not connected
static void ov_mouse_states(Cpu *c) {
    MouseState *out = arg_p(c, 0);
    int max = (int)arg_x(c, 1);
    if (max <= 0) { ret_x(c, 0); return; }
    if (!out) rt_fatal("nn::hid::GetMouseStates: null output");
    plat_mutex_lock(&lock);
    if (!mouse_ready) rt_fatal("nn::hid::GetMouseStates before InitializeMouse");
    *out = (MouseState){.sampling = ++sample};
    plat_mutex_unlock(&lock);
    ret_x(c, 1);
}

// Result GetMouseState(MouseState *out): the singular form of the same empty mouse.
static void ov_mouse_state(Cpu *c) {
    MouseState *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::GetMouseState: null output");
    plat_mutex_lock(&lock);
    if (!mouse_ready) rt_fatal("nn::hid::GetMouseState before InitializeMouse");
    *out = (MouseState){.sampling = ++sample};
    plat_mutex_unlock(&lock);
    ret_x(c, 0);
}

// void GetKeyboardState(KeyboardState *out): a keyboard with no key down
static void ov_keyboard_state(Cpu *c) {
    KeyboardState *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::GetKeyboardState: null output");
    plat_mutex_lock(&lock);
    if (!keyboard_ready) rt_fatal("nn::hid::GetKeyboardState before InitializeKeyboard");
    *out = (KeyboardState){.sampling = ++sample};
    plat_mutex_unlock(&lock);
}

// ---- vibration and six-axis sensor (MacNativeGames hid_feedback.c)
// Handles are the SDK's (nn::hid::utility, sdk+0x4a6e00 and sdk+0x4a6ab0): byte 0 the style index (3 full-key,
// 4 handheld), byte 1 the controller id, byte 2 the device (vibration 0 left / 1 right, six-axis 2). Every call under
// the public functions goes through nn::hid::detail (sdk's PLT), answered here instead of by the hid service, including
// the batches VibrationTarget queues through detail::EnqueueVibrationValue into detail::SendVibrationValues.
// VibrationValue {amplitude low, frequency low, amplitude high, frequency high}: 0..1 and Hz, neutral 0/160/0/320.
typedef struct { float amp_low, freq_low, amp_high, freq_high; } VibrationValue;
typedef struct { uint32_t type, position; } VibrationDeviceInfo;   // type 1 linear resonant actuator; position 1 left, 2 right
_Static_assert(sizeof(VibrationValue) == 16 && sizeof(VibrationDeviceInfo) == 8, "vibration ABI");

enum { STYLE_INDEX_FULL_KEY = 3, STYLE_INDEX_HANDHELD = 4, STYLE_INDEX_JOY_LEFT = 6, STYLE_INDEX_JOY_RIGHT = 7,
       SIX_AXIS_DEVICE = 2, RUMBLE_MS = 250 };
#define RUMBLE_STALE_NS 250000000ull     // a side the game has not updated for 250 ms is silent
#define RUMBLE_REFRESH_NS 50000000ull    // an unchanged value is sent again every 50 ms (its 250 ms start over)
#define RUMBLE_MIN_GAP_NS 10000000ull    // a changed value at most every 10 ms (the game streams every few ms)
static const VibrationValue vib_neutral = {0, 160, 0, 320};

static int env_zero(const char *name) {
    const char *e = cfg_get(name);
    return e && !strcmp(e, "0");
}

static void once(volatile int32_t *flag, const char *fmt, uint32_t v) {
    if (!plat_xchg32(flag, 1)) rt_log(fmt, v);
}

// AC_RUMBLE_STRENGTH (percent, 0..300, default 130; 0 = off) and AC_RUMBLE_CURVE (exponent on the amplitude, 0.2..1.5,
// default 0.6). HD rumble is strong at low amplitude, an ERM motor hardly turns there, so weak values are lifted by the
// curve (0.1 -> 0.25) and the strength scales the result (saturating at 65535; 0.66 and up reach full drive at 130 %).
// Both are live settings (the menu): read again whenever cfg_version changes.
static float rumble_strength = 1.3f, rumble_curve = 0.6f;
static volatile int32_t rumble_cfg_logged;
static unsigned rumble_cfg_seen = ~0u;
static float cfg_float(const char *name, float lo, float hi) {
    double v = cfg_double(name);
    return (float)(v < lo ? lo : v > hi ? hi : v);
}
static void rumble_cfg(void) {
    unsigned v = cfg_version();
    if (v == rumble_cfg_seen) return;
    rumble_cfg_seen = v;
    rumble_strength = cfg_float("AC_RUMBLE_STRENGTH", 0, 300) / 100;
    rumble_curve = cfg_float("AC_RUMBLE_CURVE", 0.2f, 1.5f);
    int trig = input_rumble_triggers_enabled();
    if (!plat_xchg32(&rumble_cfg_logged, 1))
        rt_log("hid: rumble strength %.0f%%, curve %.2f, triggers %s", rumble_strength * 100, rumble_curve, trig ? "on" : "off");
}
static uint16_t motor(float a) { return rumble_motor(a, rumble_curve, rumble_strength); }

static uint32_t pad_style_index(void) { return ov_handheld() ? STYLE_INDEX_HANDHELD : STYLE_INDEX_FULL_KEY; }

// The SDK's id check (utility functions): players 1-8, other (0x10) and handheld (0x20); the public functions abort
// on another id.
static uint32_t handle_id(Cpu *c, const char *fn) {
    uint32_t id = *need_id(c, 2, fn);
    if (id >= 8 && id != 0x10 && id != 0x20) rt_fatal("nn::hid::%s: invalid controller id %#x", fn, id);
    return id;
}

// int GetVibrationDeviceHandles(VibrationDeviceHandle *out, int count, const NpadIdType &id, NpadStyleSet style):
// utility::GetVibrationDeviceHandles' table, the first style bit in its order; none with AC_RUMBLE=0.
static void ov_vibration_handles(Cpu *c) {
    uint32_t *out = arg_p(c, 0), style = (uint32_t)arg_x(c, 3);
    int max = (int)arg_x(c, 1);
    uint32_t id = handle_id(c, "GetVibrationDeviceHandles");
    if (!out || max <= 0 || __builtin_popcount(style) != 1)
        rt_fatal("nn::hid::GetVibrationDeviceHandles: invalid arguments (count %d, styles %#x)", max, style);
    static volatile int32_t asked;
    rumble_cfg();
    // strength 0 keeps the devices (silent), so the menu can turn rumble up again
    if (env_zero("AC_RUMBLE")) { once(&asked, "hid: no vibration devices (AC_RUMBLE=%u)", 0); ret_x(c, 0); return; }
    static const struct { uint8_t bit, n; uint32_t base; } t[] = {
        {0, 2, 0x3}, {1, 2, 0x4}, {2, 2, 0x5}, {3, 1, 0x6}, {4, 1, 0x10007}, {5, 1, 0x8}, {10, 1, 0xd}, {8, 2, 0x4},
        {30, 2, 0x21}, {29, 2, 0x20}};
    int n = 0;
    for (size_t i = 0; i < sizeof t / sizeof *t && !n; i++) {
        if (!(style >> t[i].bit & 1)) continue;
        n = t[i].n < max ? t[i].n : max;
        for (int k = 0; k < n; k++) out[k] = (t[i].base | id << 8) + ((uint32_t)k << 16);
    }
    once(&asked, "hid: the game asked for its vibration devices (first handle %#x)", n ? out[0] : 0);
    ret_x(c, (uint64_t)n);
}

// int GetSixAxisSensorHandles(SixAxisSensorHandle *out, int count, const NpadIdType &id, NpadStyleSet style):
// utility::GetSixAxisSensorHandle's order (one sensor per style, two for the Joy-Con pair). Also with AC_MOTION=0: the
// sensor then reads at rest (six_fill), so the menu can turn motion on while the game runs.
static void ov_six_axis_handles(Cpu *c) {
    uint32_t *out = arg_p(c, 0), style = (uint32_t)arg_x(c, 3);
    int max = (int)arg_x(c, 1);
    uint32_t id = handle_id(c, "GetSixAxisSensorHandles");
    if (!out || max <= 0) rt_fatal("nn::hid::GetSixAxisSensorHandles: invalid arguments (count %d)", max);
    static volatile int32_t asked;
    uint32_t base = 0, n = 1;
    if (style & 1) base = 0x20003;
    else if (style & 2) base = 0x20004;
    else if (style & 4) base = 0x5, n = 2;
    else if (style & 8) base = 0x6;
    else if (style & 0x10) base = 0x10007;
    else if (style & 0x20) base = 0x20003;
    else if (style & 0xe80) n = 0;
    else if (style & 0x100) base = 0x20004;
    else if ((style & 0x60000040) == 0x40) base = 0x20003;
    else n = 0;
    if (n > (uint32_t)max) n = (uint32_t)max;
    for (uint32_t k = 0; k < n; k++) out[k] = (base | id << 8) + (k << 16);
    once(&asked, "hid: the game asked for its six-axis sensor (first handle %#x)", n ? out[0] : 0);
    ret_x(c, n);
}

static const uint32_t *need_handle(const void *p, const char *fn) {
    if (!p) rt_fatal("nn::hid::detail::%s: null handle", fn);
    return p;
}

// The vibration device a handle names: 0 left, 1 right on the connected controller; in co-op 2 * player + side, on
// the left Joy-Con of player 1 (id 0) or the right Joy-Con of player 2 (id 1). -1 for another (logged once).
static int vibration_side(const uint32_t *h, const char *fn) {
    uint32_t v = *need_handle(h, fn), dev = v >> 16 & 0xff, id = v >> 8 & 0xff, style = v & 0xff;
    plat_mutex_lock_shared(&lock);
    int co = coop;
    plat_mutex_unlock_shared(&lock);
    if (!co && style == pad_style_index() && id == pad_id() && dev < 2) return (int)dev;
    if (co && (style == STYLE_INDEX_JOY_LEFT || style == STYLE_INDEX_JOY_RIGHT) && id < 2 && dev < 2)
        return (int)(id * 2 + dev);
    static volatile int32_t warned;
    once(&warned, "hid: vibration call for handle %#x, not a device of the connected controller: ignored", v);
    return -1;
}

static PlatMutex vib_lock = PLAT_MUTEX_INIT;
// vib_lock: per device (vibration_side), then per controller (the solo one or co-op player 1, then player 2)
static VibrationValue vib_last[4] = {{0, 160, 0, 320}, {0, 160, 0, 320}, {0, 160, 0, 320}, {0, 160, 0, 320}};
static uint64_t vib_at[4], sent_at[2];
static uint16_t sent_low[2], sent_high[2];
static int vib_allowed = 1;                                                  // vib_lock: PermitVibration

static float clamp01(float v) { return v > 0 ? (v < 1 ? v : 1) : 0; }   // NaN -> 0
// Share of a band at f Hz that goes to the light (high-frequency, right) motor: none at 160 Hz and below (the low band's
// neutral frequency), all at 320 Hz and above (the high band's), log scale between.
static float light_share(float f) { return f > 160 ? (f < 320 ? log2f(f / 160) : 1) : 0; }

// Switch HD rumble -> SDL_RumbleGamepad. Each side's two bands split by frequency between the heavy (low) and the light
// (high) motor (light_share) and add up as energy per motor (root of the summed squares); the stronger side drives each
// motor. A side the game has not updated for 250 ms is silent. Each send lasts 250 ms and is renewed every 50 ms while
// the game streams, so the pad stops by itself when the game stops sending (a freeze; a pause stops it at once,
// input_rumble_stop), as long as SDL_UpdateGamepads goes on being called: only that expires a send. Under vib_lock.
// p: 0 the solo controller or co-op player 1, 1 player 2 (whose gamepad input.c picks, input_rumble_player).
static void rumble_update(uint64_t now, int p) {
    float heavy = 0, light = 0;
    rumble_cfg();
    for (int s = 2 * p; s < 2 * p + 2 && vib_allowed; s++) {
        if (!vib_at[s] || now - vib_at[s] > RUMBLE_STALE_NS) continue;
        const VibrationValue *v = &vib_last[s];
        float al = clamp01(v->amp_low), ah = clamp01(v->amp_high), sl = light_share(v->freq_low), sh = light_share(v->freq_high);
        float h = sqrtf(al * (1 - sl) * al * (1 - sl) + ah * (1 - sh) * ah * (1 - sh));
        float l = sqrtf(al * sl * al * sl + ah * sh * ah * sh);
        if (h > heavy) heavy = h;
        if (l > light) light = l;
    }
    uint16_t lo = motor(clamp01(heavy)), hi = motor(clamp01(light));
    int changed = lo != sent_low[p] || hi != sent_high[p], on = lo || hi;
    if (changed ? on && now - sent_at[p] < RUMBLE_MIN_GAP_NS : !on || now - sent_at[p] < RUMBLE_REFRESH_NS) return;
    plat_mutex_lock_shared(&lock);
    int co = coop;
    plat_mutex_unlock_shared(&lock);
    if (co) input_rumble_player(p, lo, hi, on ? RUMBLE_MS : 0);
    else input_rumble(lo, hi, on ? RUMBLE_MS : 0);
    sent_low[p] = lo, sent_high[p] = hi, sent_at[p] = now;
}

static void play(int side, const VibrationValue *v) {
    if (!v) rt_fatal("nn::hid::detail::SendVibrationValue: null value");
    if (side < 0) return;
    VibrationValue c = *v;
    if (!isfinite(c.amp_low) || !isfinite(c.amp_high) || !isfinite(c.freq_low) || !isfinite(c.freq_high)) c = vib_neutral;
    static volatile int32_t first, first_on;
    if (!plat_xchg32(&first, 1))
        rt_log("hid: first vibration value from the game: %s %.2f@%.0f Hz %.2f@%.0f Hz", side & 1 ? "right" : "left",
               (double)c.amp_low, (double)c.freq_low, (double)c.amp_high, (double)c.freq_high);
    if ((c.amp_low > 0.01f || c.amp_high > 0.01f) && !plat_xchg32(&first_on, 1))
        rt_log("hid: first non-zero vibration from the game: %s %.2f@%.0f Hz %.2f@%.0f Hz", side & 1 ? "right" : "left",
               (double)c.amp_low, (double)c.freq_low, (double)c.amp_high, (double)c.freq_high);
    uint64_t now = kern_now_ns();
    plat_mutex_lock(&vib_lock);
    vib_last[side] = c;
    vib_at[side] = now;
    rumble_update(now, side >> 1);
    plat_mutex_unlock(&vib_lock);
}

// Result detail::InitializeVibrationDevice(const VibrationDeviceHandle &)
static void ov_vib_init(Cpu *c) {
    static volatile int32_t logged;
    const uint32_t *h = arg_p(c, 0);
    if (vibration_side(h, "InitializeVibrationDevice") >= 0)
        once(&logged, "hid: the game initialized its vibration devices (handle %#x)", *h);
    ret_x(c, 0);
}

// Result detail::SendVibrationValue(const VibrationDeviceHandle &, const VibrationValue &)
static void ov_vib_send(Cpu *c) {
    play(vibration_side(arg_p(c, 0), "SendVibrationValue"), arg_p(c, 1));
    ret_x(c, 0);
}

// Result detail::SendVibrationValues(const VibrationDeviceHandle *, const VibrationValue *, int count)
static void ov_vib_send_many(Cpu *c) {
    const uint32_t *h = arg_p(c, 0);
    const VibrationValue *v = arg_p(c, 1);
    int n = (int)arg_x(c, 2);
    if (n < 0 || (n && (!h || !v))) rt_fatal("nn::hid::detail::SendVibrationValues: invalid batch (%d values)", n);
    for (int i = 0; i < n; i++) play(vibration_side(&h[i], "SendVibrationValues"), &v[i]);
    ret_x(c, 0);
}

// Result detail::GetActualVibrationValue(VibrationValue *out, const VibrationDeviceHandle &): the last value sent
static void ov_vib_actual(Cpu *c) {
    VibrationValue *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::detail::GetActualVibrationValue: null output");
    int side = vibration_side(arg_p(c, 1), "GetActualVibrationValue");
    VibrationValue v = vib_neutral;
    plat_mutex_lock_shared(&vib_lock);
    if (side >= 0) v = vib_last[side];
    plat_mutex_unlock_shared(&vib_lock);
    *out = v;
    ret_x(c, 0);
}

// Result detail::GetVibrationDeviceInfo(VibrationDeviceInfo *out, const VibrationDeviceHandle &)
static void ov_vib_info(Cpu *c) {
    VibrationDeviceInfo *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::detail::GetVibrationDeviceInfo: null output");
    int side = vibration_side(arg_p(c, 1), "GetVibrationDeviceInfo");
    *out = side >= 0 ? (VibrationDeviceInfo){1, (uint32_t)(side & 1) + 1} : (VibrationDeviceInfo){0, 0};
    ret_x(c, 0);
}

// Result detail::IsVibrationDeviceMounted(bool *out, const VibrationDeviceHandle &)
static void ov_vib_mounted(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::detail::IsVibrationDeviceMounted: null output");
    *out = vibration_side(arg_p(c, 1), "IsVibrationDeviceMounted") >= 0;
    ret_x(c, 0);
}

// Result detail::IsVibrationPermitted(bool *out)
static void ov_vib_permitted(Cpu *c) {
    uint8_t *out = arg_p(c, 0);
    if (!out) rt_fatal("nn::hid::detail::IsVibrationPermitted: null output");
    plat_mutex_lock_shared(&vib_lock);
    *out = (uint8_t)vib_allowed;
    plat_mutex_unlock_shared(&vib_lock);
    ret_x(c, 0);
}

// Result detail::PermitVibration(bool): off silences the gamepad at once
static void ov_vib_permit(Cpu *c) {
    int on = (arg_x(c, 0) & 0xff) != 0;
    plat_mutex_lock(&vib_lock);
    vib_allowed = on;
    rumble_update(kern_now_ns(), 0);
    rumble_update(kern_now_ns(), 1);
    plat_mutex_unlock(&vib_lock);
    ret_x(c, 0);
}

static volatile int32_t six_running;   // StartSixAxisSensor .. StopSixAxisSensor

// The connected controller's sensor. In co-op the Joy-Cons' sensors read at rest (the motion thread follows the first
// gamepad, which is not always player 1's), without the warning.
static int six_axis_ours(const uint32_t *h, const char *fn) {
    uint32_t v = *need_handle(h, fn);
    if ((v & 0xff) == pad_style_index() && (v >> 8 & 0xff) == pad_id() && (v >> 16 & 0xff) == SIX_AXIS_DEVICE) return 1;
    if ((v & 0xff) == STYLE_INDEX_JOY_LEFT || (v & 0xff) == STYLE_INDEX_JOY_RIGHT) return 0;
    static volatile int32_t warned;
    once(&warned, "hid: six-axis call for handle %#x, not the connected controller's sensor: reads at rest", v);
    return 0;
}

// Result detail::StartSixAxisSensor(const SixAxisSensorHandle &): angle and direction start over
static void ov_six_start(Cpu *c) {
    const uint32_t *h = arg_p(c, 0);
    if (six_axis_ours(h, "StartSixAxisSensor")) {
        static volatile int32_t logged;
        once(&logged, "hid: the game started the six-axis sensor (handle %#x)", *h);
        motion_restart(motion_global());
        input_motion_start();
        plat_xchg32(&six_running, 1);
    }
    ret_x(c, 0);
}

// Result detail::StopSixAxisSensor(const SixAxisSensorHandle &)
static void ov_six_stop(Cpu *c) {
    if (six_axis_ours(arg_p(c, 0), "StopSixAxisSensor")) plat_xchg32(&six_running, 0);
    ret_x(c, 0);
}

// All max entries, newest first: the game reads all 16 it asks for and ignores the count (main+0x2a268e0). A short
// history repeats its oldest sample. No motion (sensor not started, no gamepad, one without a gyro, AC_MOTION=0):
// a controller lying still, not marked connected.
static int six_fill(MotionState *s, int max, const uint32_t *h, const char *fn) {
    if (max <= 0) return 0;
    if (!s) rt_fatal("nn::hid::detail::%s: null output", fn);
    int live = six_axis_ours(h, fn) && six_running && input_motion_enabled();
    int n = live ? motion_read(motion_global(), s, max, kern_now_ns()) : 0;
    if (!n) {
        static const MotionState rest = {.acc = {0, 0, -1}, .dir = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
        for (int i = 0; i < max; i++) s[i] = rest;
    } else
        for (int i = n; i < max; i++) s[i] = s[n - 1];
    static volatile int32_t first_rest, first_live;
    if (!n && six_running) once(&first_rest, "hid: six-axis read with no motion samples: at rest (%u states)", (uint32_t)max);
    if (n) once(&first_live, "hid: first six-axis read with live motion (%u samples)", (uint32_t)n);
    return max;
}

// Result detail::GetSixAxisSensorStates(int *count, SixAxisSensorState *out, int max, const SixAxisSensorHandle &)
static void ov_six_states(Cpu *c) {
    int32_t *count = arg_p(c, 0);
    if (!count) rt_fatal("nn::hid::detail::GetSixAxisSensorStates: null count");
    *count = six_fill(arg_p(c, 1), (int)arg_x(c, 2), arg_p(c, 3), "GetSixAxisSensorStates");
    ret_x(c, 0);
}

// Result detail::GetSixAxisSensorState(SixAxisSensorState *out, const SixAxisSensorHandle &)
static void ov_six_state(Cpu *c) {
    six_fill(arg_p(c, 0), 1, arg_p(c, 1), "GetSixAxisSensorState");
    ret_x(c, 0);
}

HOST_OVERRIDE("_ZN2nn3hid14InitializeNpadEv", ov_initialize_npad);
HOST_OVERRIDE("_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm", ov_supported_ids);
HOST_OVERRIDE("_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE", ov_supported_styles);
HOST_OVERRIDE("_ZN2nn3hid15GetNpadStyleSetERKj", ov_style_set);
HOST_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj", ov_states_full_key);
HOST_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyDualStateEiRKj", ov_states_joy_dual);
HOST_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyLeftStateEiRKj", ov_states_joy_left);
HOST_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadJoyRightStateEiRKj", ov_states_joy_right);
HOST_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadHandheldStateEiRKj", ov_states_handheld);
HOST_OVERRIDE("_ZN2nn3hid12GetNpadStateEPNS0_16NpadFullKeyStateERKj", ov_state_full_key);
HOST_OVERRIDE("_ZN2nn3hid12GetNpadStateEPNS0_16NpadJoyDualStateERKj", ov_state_joy_dual);
HOST_OVERRIDE("_ZN2nn3hid12GetNpadStateEPNS0_16NpadJoyLeftStateERKj", ov_state_joy_left);
HOST_OVERRIDE("_ZN2nn3hid12GetNpadStateEPNS0_17NpadJoyRightStateERKj", ov_state_joy_right);
HOST_OVERRIDE("_ZN2nn3hid12GetNpadStateEPNS0_17NpadHandheldStateERKj", ov_state_handheld);
HOST_OVERRIDE("_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE", ov_hold_type);
HOST_OVERRIDE("_ZN2nn3hid14DisconnectNpadERKj", ov_disconnect);
HOST_OVERRIDE("_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE",
              ov_controller_support);
HOST_OVERRIDE("_ZN2nn3hid21ShowControllerSupportERKNS0_20ControllerSupportArgE", ov_controller_support_1);
HOST_OVERRIDE("_ZN2nn3hid15InitializeMouseEv", ov_initialize_mouse);
HOST_OVERRIDE("_ZN2nn3hid18InitializeKeyboardEv", ov_initialize_keyboard);
HOST_OVERRIDE("_ZN2nn3hid21InitializeTouchScreenEv", ov_initialize_touch);
HOST_OVERRIDE("_ZN2nn3hid19GetTouchScreenStateILm16EEEvPNS0_16TouchScreenStateIXT_EEE", ov_touch_states);
HOST_OVERRIDE("_ZN2nn3hid14GetMouseStatesEPNS0_10MouseStateEi", ov_mouse_states);
HOST_OVERRIDE("_ZN2nn3hid13GetMouseStateEPNS0_10MouseStateE", ov_mouse_state);
HOST_OVERRIDE("_ZN2nn3hid16GetKeyboardStateEPNS0_13KeyboardStateE", ov_keyboard_state);
HOST_OVERRIDE("_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_"
              "12NpadStyleTagEEE", ov_vibration_handles);
HOST_OVERRIDE("_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_"
              "12NpadStyleTagEEE", ov_six_axis_handles);
HOST_OVERRIDE("_ZN2nn3hid6detail25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE", ov_vib_init);
HOST_OVERRIDE("_ZN2nn3hid6detail18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE", ov_vib_send);
HOST_OVERRIDE("_ZN2nn3hid6detail19SendVibrationValuesEPKNS0_21VibrationDeviceHandleEPKNS0_14VibrationValueEi",
              ov_vib_send_many);
HOST_OVERRIDE("_ZN2nn3hid6detail23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE",
              ov_vib_actual);
HOST_OVERRIDE("_ZN2nn3hid6detail22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE",
              ov_vib_info);
HOST_OVERRIDE("_ZN2nn3hid6detail24IsVibrationDeviceMountedEPbRKNS0_21VibrationDeviceHandleE", ov_vib_mounted);
HOST_OVERRIDE("_ZN2nn3hid6detail20IsVibrationPermittedEPb", ov_vib_permitted);
HOST_OVERRIDE("_ZN2nn3hid6detail15PermitVibrationEb", ov_vib_permit);
HOST_OVERRIDE("_ZN2nn3hid6detail18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE", ov_six_start);
HOST_OVERRIDE("_ZN2nn3hid6detail17StopSixAxisSensorERKNS0_19SixAxisSensorHandleE", ov_six_stop);
HOST_OVERRIDE("_ZN2nn3hid6detail22GetSixAxisSensorStatesEPiPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE",
              ov_six_states);
HOST_OVERRIDE("_ZN2nn3hid6detail21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE",
              ov_six_state);
