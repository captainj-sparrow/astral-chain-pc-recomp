// Settings: the AC_* environment variables, with astralchain.ini next to astralchain.exe behind them.
#pragma once

// getenv for a setting: the environment variable when set, else the astralchain.ini key (the name without "AC_", any
// case), else the device profile's default (cfg.c: the Steam Deck), else NULL. Same string semantics as getenv; the pointer stays valid for the process (a value cfg_set replaces
// is kept). A name without the AC_ prefix goes to getenv unchanged. Safe from any thread once cfg_init has returned.
const char *cfg_get(const char *name);

// Reads astralchain.ini once, from mem_init after the address space is claimed and before any setting is read. Logs the file, its key count, unknown keys and environment overrides.
void cfg_init(void);

// Replaces the loaded keys with those of the file at path (tests). 1 when the file was read, 0 when missing.
// Not thread-safe: nothing else may call cfg_get meanwhile.
int cfg_load(const char *path);

// Live settings (the in-game menu). cfg_set stores value as the astralchain.ini value of name (AC_ prefixed), marks it for
// cfg_save and bumps cfg_version. cfg_get returns it unless the environment sets name. Any thread.
void cfg_set(const char *name, const char *value);
// Bumped by every cfg_set. A consumer that keeps a parsed value compares this with the number it last saw (one atomic
// load) and parses again when it changed.
unsigned cfg_version(void);
// 1 when the environment sets name: cfg_set cannot change what cfg_get returns (the menu greys the setting out).
int cfg_from_env(const char *name);
// 1 when the player set name (the environment or astralchain.ini), 0 when cfg_get gives a device profile's default or NULL.
int cfg_is_set(const char *name);
// The device profile in use, "deck" or "desktop" (AC_DEVICE_PROFILE, else what the machine is).
const char *cfg_device_profile(void);
// The one-time Steam Deck move from a stored 1080p output_res to the profile's 1280x720 (cfg.c; cfg_init runs it with
// the profile and saves). deck = 0 does nothing. Marks astralchain.ini with deck_res_migrated = 1. Returns 0 when
// nothing ran, 1 when only the marker was set, 2 when output_res was reset too. Not thread-safe against cfg_load.
// For --probe-cfg only: the device profile from now on, 1 the Steam Deck's, 0 none, in place of the one the first lookup
// fixed. Not thread-safe against readers of the settings.
void cfg_force_profile(int deck);
int cfg_migrate_deck_res(int deck);
// The one-time move of a stored aspect = fit or stretch to the default, wide (cfg.c; cfg_init runs it and saves). Marks
// astralchain.ini with aspect_migrated = 1. Returns 0 when nothing ran, 1 when only the marker was set, 2 when the
// aspect line was removed too. Not thread-safe against cfg_load.
int cfg_migrate_aspect(void);
// The one-time move of a stored assume_focus other than 1 to the default, 1 (cfg.c; cfg_init runs it and saves). Marks
// astralchain.ini with assume_focus_migrated = 1. Returns as cfg_migrate_aspect. Not thread-safe against cfg_load.
int cfg_migrate_focus(void);
// Writes the keys cfg_set changed since the last save into astralchain.ini next to astralchain.exe (cfg_save_to: the file at path).
// Only those lines change: the last line of a changed key gets the new value, a missing key goes under its section
// (after the commented example line of the key when the section has one; the section is added at the end of the file
// when missing), comments and every other line stay byte for byte. 1 on success or with nothing to write.
int cfg_save(void);
int cfg_save_to(const char *path);

// Reset (the menu's "Reset all settings to defaults"). Removes the active line of each key (names without AC_, any
// case) from astralchain.ini at the next cfg_save and its value from memory, so the consumer's own default applies again; the
// commented example lines stay. Keys the environment sets are skipped. Bumps cfg_version. Returns how many it reset.
int cfg_reset(const char *const *keys, int n);
// 0 when the last cfg_save / cfg_save_to failed to write (the values stay set in memory, the next save tries again).
int cfg_last_save_ok(void);
// Every setting with a value, for a bug report ("Copy system info"): the astralchain.ini and cfg_set keys (name without
// AC_, upper case; from_env 0), then the known AC_ variables the environment sets (from_env 1).
void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx);

// What the menu knows about a setting. key: without AC_, upper case. def: what the consumer uses when the key is
// unset ("" = no fixed default, the consumer picks per machine). values: "0|1", "mailbox|fifo|immediate", a range
// "0..300", or NULL for free text. live: applies at once; the others are read at start.
typedef struct { const char *key, *def, *values; int live; } CfgMeta;
// The row of key (prefix and case ignored); NULL for a key the menu does not show.
const CfgMeta *cfg_meta(const char *key);
// The i-th row (0-based, table order); NULL past the last.
const CfgMeta *cfg_meta_at(int i);

// The setting's value for consumers: the environment or astralchain.ini value, else the CfgMeta default. name has the AC_
// prefix, like cfg_get; a name without a row is a bug (rt_fatal). A row with an empty default has no fixed value: its
// consumer reads cfg_get and keeps its own "unset" branch. An empty value counts as unset.
const char *cfg_value(const char *name);
// cfg_value parsed. A set value that does not parse (int: whole string; double: strtod) gives the default's number.
// cfg_bool: "0" is false, anything else true.
int cfg_int(const char *name);
double cfg_double(const char *name);
int cfg_bool(const char *name);

// The scene's temporal upscaler and its mode (vk_dlaa.c, resolution.c): AC_UPSCALING (off, auto, dlss, xess, fsr3) and
// AC_UPSCALING_QUALITY (native, quality, balanced, performance, ultra_performance), with 1.10.1's environment variable
// AC_ANTI_ALIASING=dlaa|xess|fsr3 read as that upscaler in its native mode while AC_UPSCALING is set nowhere.
// cfg_smaa: the game's SMAA for when no upscaler runs (AC_ANTI_ALIASING other than off).
const char *cfg_upscaling(void);
const char *cfg_upscaling_quality(void);
int cfg_smaa(void);
