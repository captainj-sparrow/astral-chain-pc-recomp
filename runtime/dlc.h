// DLCs of the port: additions to the game written as plain XML, built into the game's archives at start (dlc.c).
#pragma once
#include <stddef.h>
#include <stdint.h>

// Reads every <dir of astralchain.exe>/dlc/<name>/dlc.xml, once; later calls return at once. Any thread. With the
// setting dlc = 0 (the default; the Mods tab's "Load DLC") it reads nothing, and the game runs as it is.
void dlc_scan(void);
// 1 when this start read the DLC folders (the setting was on when dlc_scan ran).
int dlc_started_on(void);
// The DLCs this start loaded, by index (the folder names); NULL past the last one.
const char *dlc_name(int i);
// The folder of the DLCs, <dir of astralchain.exe>/dlc, into out[0..n); 0 when it cannot be named.
int dlc_dir(char *out, size_t n);
// 1 when a DLC changes the RomFS archive `rel` ("quest/quest.pkz", "Text/Text.pkz").
int dlc_changes(const char *rel);
// SHA-256 of every DLC source file and of the builder's version: what a built archive depends on besides the game's.
const uint8_t *dlc_key(void);
// The archive `rel` (in[0..n), as the game would read it) with the DLCs' entries: malloc'd, *out_n bytes; NULL when
// nothing could be added (logged).
void *dlc_build(const char *rel, const void *in, size_t n, size_t *out_n);

// The manifest entries quest.c applies, by index; 0 past the last one.
int dlc_training(int i, uint32_t *value, const char **text_id, uint32_t *quest);   // a row of the training menu
int dlc_window(int i, uint32_t *number);                                            // a tutorial window's number
int dlc_live_text(int i, const char **id);                                          // a text with {ACTION} fields
// The game's hash of a text id or of a file name in a DAT: the crc32 of its lower case, 31 bits.
uint32_t dlc_hash(const char *name);

// For --probe-dlc (dlc_probe.c) only. dlc_probe_scan reads the DLC folders of `dir` whatever the setting, in place of
// what dlc_scan read. The others write the game's containers from the probe's data, malloc'd: an archive of entries
// compressed here, a DAT, and a text file (one section, the texts in its first column).
void dlc_probe_scan(const char *dir);
void *dlc_probe_pkz(const char *const *names, const void *const *data, const size_t *sizes, int n, size_t *out_n);
void *dlc_probe_dat(const char *const *names, const void *const *data, const size_t *sizes, int n, size_t *out_n);
void *dlc_probe_ptd(const char *const *ids, const char *const *text, int n, size_t *out_n);
