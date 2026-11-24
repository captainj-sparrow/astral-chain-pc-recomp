// The audio output on hosts other than Windows (audio_out.h): an SDL3 stream on the default playback device, filled
// from SDL's audio thread. audio_out_win.c is the Windows one; the reason it does not use SDL is in its header.
//
// Under audio_output = auto the default device chooses what the game mixes (audio_out_auto_channels: 5.1 when it has
// 6 channels or more; SDL tells nothing of a spatial sound format). The stream starts after that choice, so the fill
// never runs before it.
//
// A 5.1 game opens a 6 channel stream when the default device has 6 channels or more (SDL's 5.1 order, FL FR FC LFE
// BL BR, is Wwise's with the surrounds named back), else a stereo one that the fill folds down. SDL follows the
// default device by itself and keeps a stream without a device running at the rate of the clock, so the game keeps its
// pace without one.
#include <SDL3/SDL.h>
#include <stdio.h>
#include <string.h>

#include "audio_out.h"
#include "audren.h"
#include "host.h"
#include "platform/platform.h"

#define OUT_RATE 48000
#define OUT_MAX_FRAME 1024   // frames filled per step of the callback

static struct {
    AudioOutFill fill;
    int channels, stream_channels;   // what the game mixes, and what the stream carries
    PlatMutex info_lock;
    AudioOutInfo info;
} s = {.info_lock = PLAT_MUTEX_INIT};

static void SDLCALL out_pull(void *user, SDL_AudioStream *stream, int additional, int total) {
    (void)user;
    (void)total;
    float tmp[6 * OUT_MAX_FRAME];
    const int bytes = s.stream_channels * (int)sizeof(float);
    while (additional > 0) {
        int frames = additional / bytes;
        if (frames > OUT_MAX_FRAME) frames = OUT_MAX_FRAME;
        if (frames <= 0) break;
        s.fill(tmp, frames, s.channels, s.stream_channels);
        if (!SDL_PutAudioStreamData(stream, tmp, frames * bytes)) break;
        additional -= frames * bytes;
    }
}

int audio_out_start(int channels, AudioOutFill fill) {
    AudioOutInfo info = {.started = 1, .spatial = -1};
    s.fill = fill;
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        rt_log("audout: SDL audio init failed (%s): buffers are recycled without sound", SDL_GetError());
        return 0;
    }
    SDL_AudioSpec have = {0};
    if (SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &have, NULL)) info.device_channels = have.channels;
    info.surround_ok = info.device_channels >= 6;
    if (channels == AUDIO_OUT_AUTO) {
        channels = audio_out_auto_channels(&info);
        rt_log("audout: audio_output = auto: the default device has %d channels, the game mixes %s",
               info.device_channels, channels == 6 ? "5.1" : "stereo");
    }
    channels = channels == 6 ? 6 : 2;
    s.channels = channels;
    info.stream_channels = channels;
    s.stream_channels = channels == 6 && info.surround_ok ? 6 : 2;
    SDL_AudioSpec spec = {SDL_AUDIO_F32, s.stream_channels, OUT_RATE};
    SDL_AudioStream *stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, out_pull, NULL);
    if (!stream) {
        rt_log("audout: no output device (%s): buffers are recycled without sound", SDL_GetError());
        return 0;
    }
    if (!SDL_ResumeAudioStreamDevice(stream)) {
        rt_log("audout: starting the output failed (%s): buffers are recycled without sound", SDL_GetError());
        SDL_DestroyAudioStream(stream);
        return 0;
    }
    SDL_AudioSpec dev;
    int period = 0;
    if (SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream), &dev, &period) && period > 0 && dev.freq > 0)
        audio_device_set_period((uint64_t)period, (uint32_t)dev.freq);
    const char *name = SDL_GetAudioDeviceName(SDL_GetAudioStreamDevice(stream));
    snprintf(info.name, sizeof info.name, "%s", name ? name : "?");
    info.open = 1;
    info.stream_channels = s.stream_channels;
    plat_mutex_lock(&s.info_lock);
    s.info = info;
    plat_mutex_unlock(&s.info_lock);
    rt_log("audout: SDL device '%s' opened: %d channels; the game mixes %d channels, the stream carries %d%s, device "
           "period %d frames",
           info.name, info.device_channels, channels, s.stream_channels,
           channels == 6 && s.stream_channels == 2 ? " (the 5.1 mix folded down here)" : "", period);
    return channels;
}

void audio_out_info(AudioOutInfo *out) {
    plat_mutex_lock(&s.info_lock);
    *out = s.info;
    plat_mutex_unlock(&s.info_lock);
}
