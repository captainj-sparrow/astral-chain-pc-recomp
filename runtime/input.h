// Host input sampling (runtime/input.c, with input_win.c or input_sdl.c): SDL3 gamepad and the keyboard as one Switch npad.
#pragma once
#include <stdint.h>

// Bits of libnx HidNpadButton (A=0 B=1 X=2 Y=3 StickL=4 StickR=5 L=6 R=7 ZL=8 ZR=9 Plus=10 Minus=11 Left=12 Up=13 Right=14
// Down=15) and the stick directions (StickLLeft..Down 16-19, StickRLeft..Down 20-23, derived from the sticks).
// Sticks are -32767..32767, up is +y. AC_INPUT=none reports no input. Callable from any thread.
void input_poll(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry);
// The same for host UI (the text entry): never blocked. input_block(1) while a text entry is open: input_poll then
// reports nothing; input_block(0) when it closes: what is held then stays hidden until it is released (the release
// gate below).
void input_poll_ui(uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry);
// 1 while the key or mouse button (input_map.h KEY_*) was down at the last keyboard read, whatever it is bound to: the
// game's window in front, a mouse button only with the cursor captured, and never while the menu or a text entry is
// open. Any thread.
int input_key_held(int key);
void input_block(int on);
// The release gate (input_map.c, pure). The press that closes the menu or a text entry is still down when the game
// reads its pad again, and the game must not take it for a press of its own. So every input held at that moment
// stays hidden from the game until that input is released, and every other input reaches the game at once: wait
// starts as INPUT_RELEASE_ALL when the menu or the text entry closes, each poll keeps in it the inputs still held
// (input_release_held of what the sources read) and input_release_gate takes those out of what the game gets,
// returning the new wait. An input is a button bit or a whole stick (its four direction bits; a stick in wait reads
// centred).
// Up to 1.21.2 the game read nothing at all until one poll found every button up and all four stick axes at exactly
// 0. Analog values do not come back to 0 on every pad: a stick resting just past its dead zone, or a trigger resting
// past trigger_threshold, kept the game deaf for good after the menu closed, until the pad was unplugged (no pad
// reads 0). A test gamepad resting at 3500 of 32767 on one axis (10.7 %, 955 after the default 8 % dead zone) left
// the title screen without input after the menu closed, and one resting at 2000 did not. The menu's chord closed it
// inside the game's own poll, where the open menu had already emptied the sample, so that way out hid the fault until
// the pause moved the chord's poll to the presenting thread. Now a stick's rest is no input at all (only a stick
// past half its travel or tilted by keys is held), and an input that never reads released hides nothing but itself.
#define INPUT_RELEASE_ALL 0xffffffu   // buttons 0-15 and the stick directions 16-23
uint32_t input_release_held(uint64_t buttons, int keys_tilt);   // keys_tilt: bit 0 keys tilt the left stick, bit 1 the right
uint32_t input_release_gate(uint32_t wait, uint32_t held, uint64_t *buttons, int32_t sticks[4]);
// Co-op (ov_input.c): input_players(2) while the game plays with two players, (1) when it goes back to one. With two,
// a second gamepad is opened and each player reads its own sources: the gamepads take the players in the order they
// were opened, the keyboard and the mouse the first player without a gamepad (player 1 when both have one). Under
// AC_INPUT=none the script feeds the players (its "P2" lines player 2). input_players_connected: bit p for each player
// with a source now. input_poll_player: player 0 or 1, the same bits and sticks as input_poll (also blocked by the text
// entry and the menu, each player with a release gate of their own); 0 and nothing pressed for a player without a
// source. Any thread.
void input_players(int n);
unsigned input_players_connected(void);
int input_poll_player(int player, uint64_t *buttons, int32_t *lx, int32_t *ly, int32_t *rx, int32_t *ry);
// The in-game menu (ui/menu.h): input_menu(1) when it opens (input_poll reports nothing, the cursor is released), (0) when
// it closes (what is held then stays hidden until it is released, the release gate above). Independent of input_block.
void input_menu(int open);
// While the game is paused (pause.c) nothing else polls: one poll of every source the game reads, for the menu's and
// the overlay's chords. away (out of focus): the pads are only updated, their chords are not looked at. It reports
// nothing.
void input_poll_paused(int away);
// The start of a pause: every motor of both players' gamepads stops now.
void input_rumble_stop(void);
// The gamepad alone, for the menu: Switch button bits and the left stick (up +y). 0 without a connected gamepad.
// Minus+Plus combo and the neutral pad while the menu is open (input_map.c, window-system free): *held is the caller's memory
// of the combo, sticks are lx, ly, rx, ry. Returns 1 in the poll where the combo goes down (toggle the menu once).
int input_overlay_gate(int *held, uint64_t *buttons);   // L+R+Minus+Plus: 1 in the poll all four go down
int input_menu_gate(int *held, int menu_open, uint64_t *buttons, int32_t sticks[4]);
int input_pad_read(uint64_t *buttons, int32_t *lx, int32_t *ly);
// The open gamepad's kind (input_map.h PAD_KIND_*) for the button prompts' auto choice. Any thread.
int input_pad_kind(void);
// First-run gamepad detection for the prompts' auto choice (romfs.c overlay_init, before input starts): inits the
// SDL gamepad subsystem, opens the first gamepad and returns its kind, closing it again. Input's own start later
// opens its own handle. PAD_KIND_NONE without a gamepad or with AC_INPUT=none. Any thread; idempotent.
int input_detect_pad_kind(void);
// AUTO resolved through a first-run detection: the detected kind's mod, Nintendo for any other controller or none.
// Callers with a live gamepad use input_prompts_auto_live instead.
int input_prompts_auto(void);
// AUTO for the gamepad open now, for the settings menu. Both fall back to the controller a launcher names
// (RECOMPLAUNCHER_PAD) while there is no gamepad at all, so the menu agrees with what the start loaded.
int input_prompts_auto_live(void);
// The gamepad layout in effect (PAD_LAYOUT_*, input_map.h; AC_PAD_LAYOUT, live); -1 without a gamepad. Any thread.
int input_pad_layout(void);
// The layout the loaded button prompts force (input_map.h PAD_LAYOUT_*), or -1 with the game's own prompts: the
// Xbox mod forces the Xbox layout, the PlayStation mod the PlayStation layout (input_prompts_layout). Follows the
// loaded prompts rather than the asked-for ones (romfs.c overlay_init). Any thread.
int input_forced_layout(void);
// How far the open gamepad's left (pct[0]) and right (pct[1]) sticks are from the centre now, in percent of full
// deflection, before the dead zone: the menu shows it on the dead zone rows, so a drifting stick's resting value can be
// read off. 0 (and both 0) without a connected gamepad. Any thread.
int input_pad_stick_percent(double pct[2]);
// How far the open gamepad's left (pct[0]) and right (pct[1]) triggers are pulled now, in percent of their travel: the
// menu shows it on the trigger threshold row. 0 (and both 0) without a connected gamepad. Any thread.
int input_pad_trigger_percent(double pct[2]);
// The label on a face button of the open gamepad (input_map.h PAD_SOUTH..PAD_NORTH, as SDL reports it); PAD_LABEL_NONE for another
// button, an unknown label or no gamepad. Any thread.
enum { PAD_LABEL_NONE, PAD_LABEL_A, PAD_LABEL_B, PAD_LABEL_X, PAD_LABEL_Y, PAD_LABEL_CROSS, PAD_LABEL_CIRCLE,
       PAD_LABEL_SQUARE, PAD_LABEL_TRIANGLE };
int input_pad_label(int pad);
// The menu's remapping rows (Controls tab). input_map_snapshot copies the tables in effect (after any changed AC_BIND_* line).
// input_capture polls the keys, the mouse buttons and the gamepad for a press: reset 1 takes what is held now as the
// start (a held key counts once released and pressed again) and returns NONE; then it returns KEY or PAD with *id the
// key or pad id (input_map.h) pressed since the last call, or CANCEL for Esc. Keys count only while the game window is in front.
enum { INPUT_CAPTURE_NONE, INPUT_CAPTURE_KEY, INPUT_CAPTURE_PAD, INPUT_CAPTURE_CANCEL };
struct InputMap;   // input_map.h (not included here: swkbd_win.c has button names of its own)
void input_map_snapshot(struct InputMap *m);
int input_capture(int reset, int *id);
// The character on a punctuation key (KEY_MINUS..KEY_GRAVE) in the keyboard layout in use, 0 for another key.
uint32_t input_key_char(int key);
// F8, the flight recorder's key (no KEY_ id), held while the game window is in front.
int input_flight_key(void);

// Mouse camera and cursor (input_win.c, Win32 window thread). The window procedure calls input_window_msg first for raw
// mouse input, cursor hiding and confinement; it returns 1 with *res set when the message is handled. input_cursor_suspend(1)
// releases the cursor around an F11 window change, (0) captures it again once the client area has its new size.
void input_cursor_suspend(int on);
int input_cursor_released(void);   // Esc or a non-client activation released the cursor (probe)

// Rumble: SDL_RumbleGamepad on the open gamepad, low (heavy, left) and high (light, right) motor 0..65535 for ms
// milliseconds. 1 if a gamepad with rumble took it. Callable from any thread.
int input_rumble(uint16_t low, uint16_t high, uint32_t ms);
// The same on co-op player 0 or 1's gamepad (input_players); 0 for a player on the keyboard.
int input_rumble_player(int player, uint16_t low, uint16_t high, uint32_t ms);
// Motor value for a normalised amplitude: pow(a, curve) * strength, saturated to 0..65535. Pure.
uint16_t rumble_motor(float a, float curve, float strength);
// AC_RUMBLE_TRIGGERS=1: input_rumble also drives the impulse triggers (Xbox) with the high motor's value.
int input_rumble_triggers_enabled(void);

// Motion (AC_MOTION=0 turns it off, live). input_motion_start starts the sensor thread once: every 4 ms it reads the
// gamepad's gyro and accelerometer (when it has them) and feeds motion_global(). Callable from any thread.
void input_motion_start(void);
// AC_MOTION is not "0" (a live setting: read again when cfg_version changes). Off: the sensor reads at rest.
int input_motion_enabled(void);
// SDL sensor values (accelerometer m/s^2, gyro rad/s, SDL gamepad axes) to the Switch controller frame motion_feed
// takes (acceleration in G as gravity reads, rotation rad/s). The mapping is documented in input_win.c.
void input_motion_axes(const float sdl_acc[3], const float sdl_gyro[3], double acc_g[3], double gyro[3]);

// motion_fusion.c: nn::hid::SixAxisSensorState and the sensor fusion that produces it.
typedef struct {
    int64_t delta_ns, sampling;
    float acc[3], gyro[3], angle[3], dir[9];   // G, revolutions/s, revolutions, controller axes in the reference frame
    uint32_t attributes, reserved;             // bit 0 connected
} MotionState;
_Static_assert(sizeof(MotionState) == 0x60, "SixAxisSensorState ABI");
typedef struct MotionFusion MotionFusion;
MotionFusion *motion_new(void);
MotionFusion *motion_global(void);   // the gamepad's, fed by input_win.c
void motion_feed(MotionFusion *m, const double acc_g[3], const double gyro_rad_s[3], uint64_t now_ns);
int motion_read(MotionFusion *m, MotionState *out, int max, uint64_t now_ns);   // newest first; 0 = none in 200 ms
void motion_restart(MotionFusion *m);   // angle and direction start over at the next sample
