// The host audio output of audout:u (ov_audout.c): one thread that asks for float frames at 48 kHz and hands them to the
// device. audio_out_win.c is WASAPI in shared mode (Windows), audio_out_sdl.c is an SDL3 stream (other hosts); the build
// picks one by its suffix.
//
// The game mixes 2 or 6 channels (FL FR FC LFE SL SR, the order of Wwise and of WAVE_FORMAT_EXTENSIBLE mask 0x60f).
// The output asks the device for that many; when the device takes no 5.1 stream it asks for stereo and the fill folds
// the 5.1 down itself. Without any device the thread still asks for frames at the rate of the clock and throws them
// away, so the game's audio thread, paced by the buffers coming back, and the device clock frame_rate.c reads go on.
//
// audio_output = auto (the default) lets the output choose what the game mixes from its first device: 5.1 when that
// device takes it for what it is (audio_out_auto_channels), stereo otherwise. The choice holds for the run, as the game
// reads its channel count once at start; a later device gets the same mix, folded down when it takes no 5.1.
#pragma once

// Fills frames frames of channels interleaved floats: the stream's channels, 2 for a fold-down of a 5.1 mix. mix: what
// the game mixes (2 or 6), passed with every call because under auto the output chooses it and starts calling before
// audio_out_start returns.
typedef void (*AudioOutFill)(float *dst, int frames, int mix, int channels);

typedef struct {
    int open;              // 1: a device plays the stream; 0: none (the null output runs, or nothing started yet)
    int started;           // audio_out_start ran: the fields below describe its last device
    char name[128];        // the device's name, UTF-8
    int device_channels;   // the channels the device's mixer runs at (WASAPI: the shared-mode mix format)
    int stream_channels;   // what the stream carries: the game's count, or 2 when a 5.1 mix is folded down here
    int spatial;           // Windows spatial sound (Windows Sonic, Dolby Atmos, DTS) on the device: 1, 0; -1 unknown
    int surround_ok;       // the device takes a 5.1 stream as it is (its own channels, or a spatial sound renderer)
} AudioOutInfo;

// What auto mixes for a device: 5.1 when its mixer runs 6 channels or more (a 5.1 or 7.1 output), or when it has a
// Windows spatial sound format on and takes a 6 channel stream (a headset with Windows Sonic, Dolby Atmos or DTS
// Headphone:X: the virtualiser places the 5.1). Not for a stereo device that merely accepts a 6 channel stream: Windows
// would fold it down there, and the game's own stereo mix is the better one. Stereo without a device.
static inline int audio_out_auto_channels(const AudioOutInfo *i) {
    return i->device_channels >= 6 || (i->spatial == 1 && i->surround_ok) ? 6 : 2;
}

#define AUDIO_OUT_AUTO 0   // audio_out_start's channels: the output chooses (audio_out_auto_channels)

// Starts the output thread for a game that mixes channels (2, 6, or AUDIO_OUT_AUTO) and returns once its first device
// has been tried, so the log and audio_out_info describe it. Returns what the game mixes (2 or 6) when the thread runs
// (with a device or the null output), 0 when it could not start: then the caller treats every buffer as played at once.
int audio_out_start(int channels, AudioOutFill fill);
// The output's last state, any thread.
void audio_out_info(AudioOutInfo *out);
