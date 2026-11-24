// The audio output on Windows (audio_out.h): WASAPI in shared mode on the default render endpoint, fed by one
// event-driven thread that owns every COM object.
//
// Why not SDL3, which the other hosts use: SDL's WASAPI back end opens the endpoint at the channel count of its
// shared-mode mix format. A headset with a spatial sound format on (Windows Sonic for Headphones, Dolby Atmos for
// Headphones, DTS Headphone:X) keeps a stereo mix format, so SDL folds a 5.1 stream down to stereo before Windows sees
// it and the virtualiser gets nothing to place. The endpoint takes a 6 channel shared stream when the application opens
// one itself: on a Logitech PRO X Wireless with Windows Sonic on (2026-11-21) the mix format is 2 channels at 48 kHz,
// IAudioClient::IsFormatSupported answers S_OK for 6 and 8 channels, and an Initialize at 6 channels (mask 0x60f)
// succeeds without AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM. SDL, asked for 6 channels on the same endpoint, opened 2.
//
// audio_output = auto: the first endpoint tried chooses what the game mixes, from what device_probe finds there
// (audio_out_auto_channels): 5.1 on a 5.1 or 7.1 mix format or with a spatial sound format on, stereo otherwise and
// without an endpoint. audio_out_start and the thread settle it with one compare-and-swap, so a start that stops
// waiting for a slow endpoint and the thread that answers later agree on one count.
//
// Opening, per device: a 5.1 game asks for 6 float channels at 48 kHz, mask 0x60f (FL FR FC LFE SL SR, Wwise's order)
// as the endpoint takes it; refused while the mix format has 6 channels or more (a receiver at 44.1 kHz), again with
// the engine's sample-rate conversion; otherwise the stream is stereo and the fill folds the mix down. A stereo game asks
// for stereo, then with the engine's conversion (a mono or 44.1 kHz endpoint). Each open also records what the menu
// shows (audio_out_info): the mix format's channels, whether a 6 channel stream is taken, whether a spatial sound format
// is on (the spatial client offers dynamic objects: device_probe), the endpoint's name.
//
// The default endpoint is followed: a change of default (IMMNotificationClient), a change of the endpoint's format
// (PKEY_AudioEngine_DeviceFormat, which a spatial sound format switched on or off may change) and
// AUDCLNT_E_DEVICE_INVALIDATED on the stream reopen it. Without an endpoint the thread runs as a null output: it asks for
// frames at the rate of the clock and drops them, so the game and the device clock (audio_mixer.c audio_device_clock_ns)
// keep their pace, and tries an endpoint again every 500 ms. This is what SDL3 does with its default device.
//
// The GUIDs and property keys are spelled here (values from the mingw-w64 headers) so the file needs only ole32.
#define COBJMACROS
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <spatialaudioclient.h>
#include <stdio.h>
#include <string.h>

#include "audio_out.h"
#include "audren.h"
#include "host.h"
#include "platform/platform.h"

#define OUT_RATE 48000
#define OUT_MASK_51 0x60fu          // SPEAKER_FRONT_LEFT | RIGHT | CENTER | LOW_FREQUENCY | SIDE_LEFT | SIDE_RIGHT
#define OUT_MASK_STEREO 0x3u
#define OUT_NULL_STEP_MS 10         // the null output's wake-up
#define OUT_RETRY_MS 500            // the null output's interval between tries of an endpoint
#define OUT_NULL_CHUNK 1024         // frames the null output asks for per call
#define OUT_NULL_MAX_LAG (OUT_RATE / 10)   // after a longer stall the null output starts again from now (100 ms)

static const GUID ac_clsid_enumerator = {0xbcde0395, 0xe52f, 0x467c, {0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e}};
static const GUID ac_iid_enumerator = {0xa95664d2, 0x9614, 0x4f35, {0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6}};
static const GUID ac_iid_notification = {0x7991eec9, 0x7e89, 0x4d85, {0x83, 0x90, 0x6c, 0x70, 0x3c, 0xec, 0x60, 0xc0}};
static const GUID ac_iid_unknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};
static const GUID ac_iid_audio_client = {0x1cb9ad4c, 0xdbfa, 0x4c32, {0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2}};
static const GUID ac_iid_render_client = {0xf294acfc, 0x3146, 0x4483, {0xa7, 0xbf, 0xad, 0xdc, 0xa7, 0xc2, 0x60, 0xe2}};
static const GUID ac_iid_spatial_client = {0xbbf8e066, 0xaaaa, 0x49be, {0x9a, 0x4d, 0xfd, 0x2a, 0x85, 0x8e, 0xa2, 0x7f}};
static const GUID ac_iid_spatial_stream = {0xbab5f473, 0xb423, 0x477b, {0x85, 0xf5, 0xb5, 0xa3, 0x32, 0xa0, 0x41, 0x53}};
static const GUID ac_subtype_float = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
static const PROPERTYKEY ac_pkey_friendly_name = {
    {0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};
static const PROPERTYKEY ac_pkey_device_format = {
    {0xf19f064d, 0x082c, 0x4e27, {0xbc, 0x73, 0x68, 0x82, 0xa1, 0xbb, 0x8e, 0x4c}}, 0};

static struct {
    AudioOutFill fill;
    volatile int32_t channels;     // what the game mixes, 2 or 6; 0 under auto until the first endpoint was tried
    PlatMutex info_lock;
    AudioOutInfo info;
    volatile int32_t reopen;       // set by the notification client and by a failed stream call
    HANDLE wake;                   // wakes the null output when the notification client sets reopen
    HANDLE ready;                  // set once the first endpoint has been tried
    IMMDeviceEnumerator *enumerator;
    WCHAR device_id[512];          // the endpoint the stream plays on, for OnPropertyValueChanged
    IAudioClient *client;          // NULL: the null output runs
    IAudioRenderClient *render;
    HANDLE event;                  // the stream's buffer event
    UINT32 buffer;                 // the stream's buffer, frames
    int stream_channels;
    float null_buf[6 * OUT_NULL_CHUNK];
} w = {.info_lock = PLAT_MUTEX_INIT};

// ---- endpoint notifications (any thread of the audio service): they only set reopen and wake the thread
static HRESULT STDMETHODCALLTYPE nc_query(IMMNotificationClient *self, REFIID riid, void **out) {
    if (IsEqualIID(riid, &ac_iid_unknown) || IsEqualIID(riid, &ac_iid_notification)) {
        *out = self;
        return S_OK;
    }
    *out = NULL;
    return E_NOINTERFACE;
}
static ULONG STDMETHODCALLTYPE nc_addref(IMMNotificationClient *self) { (void)self; return 1; }   // static object
static ULONG STDMETHODCALLTYPE nc_release(IMMNotificationClient *self) { (void)self; return 1; }
static HRESULT STDMETHODCALLTYPE nc_state(IMMNotificationClient *self, LPCWSTR id, DWORD state) {
    (void)self, (void)id, (void)state;
    return S_OK;
}
static HRESULT STDMETHODCALLTYPE nc_added(IMMNotificationClient *self, LPCWSTR id) { (void)self, (void)id; return S_OK; }
static HRESULT STDMETHODCALLTYPE nc_removed(IMMNotificationClient *self, LPCWSTR id) { (void)self, (void)id; return S_OK; }

static void ask_reopen(void) {
    plat_xchg32(&w.reopen, 1);
    SetEvent(w.wake);
}

static HRESULT STDMETHODCALLTYPE nc_default(IMMNotificationClient *self, EDataFlow flow, ERole role, LPCWSTR id) {
    (void)self, (void)id;
    if (flow == eRender && role == eConsole) ask_reopen();
    return S_OK;
}

// The format of the endpoint in use changed. device_id is written by the output thread only while no stream runs on
// it, and a torn read here at worst skips or adds one reopen.
static HRESULT STDMETHODCALLTYPE nc_property(IMMNotificationClient *self, LPCWSTR id, const PROPERTYKEY key) {
    (void)self;
    if (id && IsEqualGUID(&key.fmtid, &ac_pkey_device_format.fmtid) && key.pid == ac_pkey_device_format.pid &&
        !wcscmp(id, w.device_id))
        ask_reopen();
    return S_OK;
}

static IMMNotificationClientVtbl nc_vtbl = {
    .QueryInterface = nc_query, .AddRef = nc_addref, .Release = nc_release, .OnDeviceStateChanged = nc_state,
    .OnDeviceAdded = nc_added, .OnDeviceRemoved = nc_removed, .OnDefaultDeviceChanged = nc_default,
    .OnPropertyValueChanged = nc_property,
};
static IMMNotificationClient nc = {&nc_vtbl};

// ---- opening and closing (output thread)
static void set_info(const AudioOutInfo *in) {
    plat_mutex_lock(&w.info_lock);
    w.info = *in;
    plat_mutex_unlock(&w.info_lock);
}

static void float_format(WAVEFORMATEXTENSIBLE *f, int channels, unsigned mask) {
    memset(f, 0, sizeof *f);
    f->Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    f->Format.nChannels = (WORD)channels;
    f->Format.nSamplesPerSec = OUT_RATE;
    f->Format.wBitsPerSample = 32;
    f->Format.nBlockAlign = (WORD)(channels * 4);
    f->Format.nAvgBytesPerSec = OUT_RATE * f->Format.nBlockAlign;
    f->Format.cbSize = sizeof *f - sizeof f->Format;
    f->Samples.wValidBitsPerSample = 32;
    f->dwChannelMask = mask;
    f->SubFormat = ac_subtype_float;
}

// A fresh client initialised for channels, or NULL. convert: let the engine convert the rate and channels.
static IAudioClient *client_open(IMMDevice *d, int channels, int convert, HRESULT *res) {
    IAudioClient *c = NULL;
    *res = IMMDevice_Activate(d, &ac_iid_audio_client, CLSCTX_ALL, NULL, (void **)&c);
    if (FAILED(*res)) return NULL;
    WAVEFORMATEXTENSIBLE f;
    float_format(&f, channels, channels == 6 ? OUT_MASK_51 : OUT_MASK_STEREO);
    DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    if (convert) flags |= AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    *res = IAudioClient_Initialize(c, AUDCLNT_SHAREMODE_SHARED, flags, 0, 0, &f.Format, NULL);
    if (FAILED(*res)) {
        IAudioClient_Release(c);
        return NULL;
    }
    return c;
}

static void device_close(void) {
    if (w.client) IAudioClient_Stop(w.client);
    if (w.render) IAudioRenderClient_Release(w.render);
    if (w.client) IAudioClient_Release(w.client);
    w.render = NULL;
    w.client = NULL;
}

// What the endpoint offers: its name, its mix format's channels, a 6 channel stream taken as it is, spatial sound on.
static void device_probe(IMMDevice *d, AudioOutInfo *info, int *mix_rate) {
    IPropertyStore *ps = NULL;
    if (SUCCEEDED(IMMDevice_OpenPropertyStore(d, STGM_READ, &ps))) {
        PROPVARIANT v;
        PropVariantInit(&v);
        if (SUCCEEDED(IPropertyStore_GetValue(ps, &ac_pkey_friendly_name, &v)) && v.vt == VT_LPWSTR)
            WideCharToMultiByte(CP_UTF8, 0, v.pwszVal, -1, info->name, sizeof info->name, NULL, NULL);
        PropVariantClear(&v);
        IPropertyStore_Release(ps);
    }
    IAudioClient *c = NULL;
    if (SUCCEEDED(IMMDevice_Activate(d, &ac_iid_audio_client, CLSCTX_ALL, NULL, (void **)&c))) {
        WAVEFORMATEX *mix = NULL;
        if (SUCCEEDED(IAudioClient_GetMixFormat(c, &mix))) {
            info->device_channels = mix->nChannels;
            *mix_rate = (int)mix->nSamplesPerSec;
            CoTaskMemFree(mix);
        }
        WAVEFORMATEXTENSIBLE f;
        float_format(&f, 6, OUT_MASK_51);
        WAVEFORMATEX *closest = NULL;
        HRESULT hr = IAudioClient_IsFormatSupported(c, AUDCLNT_SHAREMODE_SHARED, &f.Format, &closest);
        if (closest) CoTaskMemFree(closest);
        info->surround_ok = hr == S_OK || info->device_channels >= 6;
        IAudioClient_Release(c);
    }
    // A spatial sound format is on when the spatial client offers dynamic objects. IsSpatialAudioStreamAvailable alone
    // does not tell: Windows keeps a fallback renderer of static objects on every endpoint, so it answered S_OK on
    // Realtek speakers with spatial sound off in the Windows sound settings (2026-11-23), with 0 dynamic objects. Up to
    // 1.21.2 that read as spatial sound on, in the menu's line and the log.
    ISpatialAudioClient *s = NULL;
    info->spatial = 0;
    if (SUCCEEDED(IMMDevice_Activate(d, &ac_iid_spatial_client, CLSCTX_ALL, NULL, (void **)&s))) {
        UINT32 objects = 0;
        info->spatial = ISpatialAudioClient_IsSpatialAudioStreamAvailable(s, &ac_iid_spatial_stream, NULL) == S_OK &&
                        SUCCEEDED(ISpatialAudioClient_GetMaxDynamicObjectCount(s, &objects)) && objects > 0;
        ISpatialAudioClient_Release(s);
    }
}

// Under auto, the first call settles what the game mixes: from info, or stereo without an endpoint (info NULL).
static void channels_settle(const AudioOutInfo *info) {
    if (w.channels) return;
    int ch = info ? audio_out_auto_channels(info) : 2;
    if (plat_cas32(&w.channels, ch, 0) != 0) return;   // audio_out_start stopped waiting and settled it first
    w.stream_channels = ch;
    if (!info)
        rt_log("audout: audio_output = auto: no audio output to look at, the game mixes stereo");
    else
        rt_log("audout: audio_output = auto: '%s' %s, the game mixes %s", info->name,
               info->device_channels >= 6 ? "runs 6 channels or more"
               : ch == 6                  ? "has spatial sound on"
               : info->surround_ok        ? "runs 2 channels without spatial sound (Windows would fold a 5.1 mix down)"
                                          : "takes no 5.1",
               ch == 6 ? "5.1" : "stereo");
}

// Opens the default endpoint; 0 leaves the null output running.
static int device_open(int log_failure) {
    AudioOutInfo info = {.started = 1, .spatial = -1};
    IMMDevice *d = NULL;
    HRESULT hr = w.enumerator ? IMMDeviceEnumerator_GetDefaultAudioEndpoint(w.enumerator, eRender, eConsole, &d) : E_FAIL;
    if (FAILED(hr)) {
        channels_settle(NULL);
        info.stream_channels = w.channels;
        if (log_failure) rt_log("audout: no default audio output (%#lx): the null output keeps time", (unsigned long)hr);
        w.device_id[0] = 0;
        set_info(&info);
        return 0;
    }
    LPWSTR id = NULL;
    w.device_id[0] = 0;
    if (SUCCEEDED(IMMDevice_GetId(d, &id))) {
        wcsncpy(w.device_id, id, sizeof w.device_id / sizeof *w.device_id - 1);
        CoTaskMemFree(id);
    }
    int mix_rate = 0;
    device_probe(d, &info, &mix_rate);
    channels_settle(&info);
    info.stream_channels = w.channels;

    IAudioClient *c = NULL;
    const char *how = "";
    int ch = 2;
    if (w.channels == 6 && info.surround_ok) {
        ch = 6;
        how = "as the endpoint takes it";
        c = client_open(d, 6, 0, &hr);
        if (!c && info.device_channels >= 6) c = client_open(d, 6, 1, &hr), how = "converted by the engine";
    }
    if (!c) {
        ch = 2;
        how = w.channels == 6 ? "the 5.1 mix folded down here" : "as the endpoint takes it";
        c = client_open(d, 2, 0, &hr);
        if (!c) c = client_open(d, 2, 1, &hr), how = w.channels == 6 ? "the 5.1 mix folded down here, converted by the engine"
                                                                    : "converted by the engine";
    }
    IMMDevice_Release(d);
    REFERENCE_TIME period = 0;
    if (c && (FAILED(hr = IAudioClient_SetEventHandle(c, w.event)) || FAILED(hr = IAudioClient_GetBufferSize(c, &w.buffer)) ||
              FAILED(hr = IAudioClient_GetService(c, &ac_iid_render_client, (void **)&w.render)) ||
              FAILED(hr = IAudioClient_GetDevicePeriod(c, &period, NULL)))) {
        IAudioClient_Release(c);
        c = NULL;
    }
    if (c) {
        // Start with the buffer full of silence, so the first event does not find it empty.
        BYTE *dst = NULL;
        if (SUCCEEDED(IAudioRenderClient_GetBuffer(w.render, w.buffer, &dst)))
            IAudioRenderClient_ReleaseBuffer(w.render, w.buffer, AUDCLNT_BUFFERFLAGS_SILENT);
        if (FAILED(hr = IAudioClient_Start(c))) {
            IAudioRenderClient_Release(w.render);
            w.render = NULL;
            IAudioClient_Release(c);
            c = NULL;
        }
    }
    if (!c) {
        rt_log("audout: the audio output '%s' takes no float stream (%#lx): the null output keeps time", info.name,
               (unsigned long)hr);
        set_info(&info);
        return 0;
    }
    w.client = c;
    w.stream_channels = ch;
    info.open = 1;
    info.stream_channels = ch;
    set_info(&info);
    int period_frames = (int)(period * OUT_RATE / 10000000);
    if (period_frames > 0) audio_device_set_period((uint64_t)period_frames, OUT_RATE);
    rt_log("audout: WASAPI '%s': mix format %d channels at %d Hz, spatial sound %s, 5.1 stream %s; the game mixes %d "
           "channels, the stream carries %d (%s), buffer %u frames, period %d frames",
           info.name, info.device_channels, mix_rate, info.spatial ? "on" : "off", info.surround_ok ? "taken" : "not taken",
           w.channels, ch, how, w.buffer, period_frames);
    return 1;
}

// ---- the output thread
static int stream_step(void) {
    DWORD r = WaitForSingleObject(w.event, 200);
    UINT32 pad = 0;
    HRESULT hr = IAudioClient_GetCurrentPadding(w.client, &pad);
    if (FAILED(hr)) return 0;
    if (r != WAIT_OBJECT_0 || pad >= w.buffer) return 1;
    UINT32 n = w.buffer - pad;
    BYTE *dst = NULL;
    if (FAILED(IAudioRenderClient_GetBuffer(w.render, n, &dst))) return 0;
    w.fill((float *)dst, (int)n, w.channels, w.stream_channels);
    return SUCCEEDED(IAudioRenderClient_ReleaseBuffer(w.render, n, 0));
}

static void out_thread(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_HIGHEST);
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) rt_log("audout: CoInitializeEx %#lx", (unsigned long)hr);
    hr = CoCreateInstance(&ac_clsid_enumerator, NULL, CLSCTX_ALL, &ac_iid_enumerator, (void **)&w.enumerator);
    if (FAILED(hr)) {
        rt_log("audout: no WASAPI device enumerator (%#lx): the null output keeps time", (unsigned long)hr);
        w.enumerator = NULL;
    } else if (FAILED(hr = IMMDeviceEnumerator_RegisterEndpointNotificationCallback(w.enumerator, &nc))) {
        rt_log("audout: endpoint notifications unavailable (%#lx): a new default output is not followed", (unsigned long)hr);
    }
    device_open(1);
    SetEvent(w.ready);

    uint64_t null_start = plat_now_ns(), null_done = 0, last_try = null_start;
    int logged_null = 0;
    for (;;) {
        if (plat_xchg32(&w.reopen, 0)) {
            device_close();
            rt_log("audout: the default audio output changed: reopening");
            if (!device_open(1)) null_start = plat_now_ns(), null_done = 0;
            last_try = plat_now_ns();
        }
        if (w.client) {
            if (!stream_step()) {
                if (!logged_null) rt_log("audout: the audio output stopped: reopening");
                logged_null = 1;
                device_close();
                plat_xchg32(&w.reopen, 0);
                if (!device_open(1)) null_start = plat_now_ns(), null_done = 0;
                last_try = plat_now_ns();
            } else {
                logged_null = 0;
            }
            continue;
        }
        // The null output: the frames the clock says are due since it started, dropped.
        WaitForSingleObject(w.wake, OUT_NULL_STEP_MS);
        uint64_t now = plat_now_ns();
        uint64_t due = (now - null_start) * OUT_RATE / 1000000000ull;
        if (due - null_done > OUT_NULL_MAX_LAG) null_start = now, null_done = 0, due = 0;
        while (null_done < due) {
            uint64_t n = due - null_done;
            if (n > OUT_NULL_CHUNK) n = OUT_NULL_CHUNK;
            w.fill(w.null_buf, (int)n, w.channels, w.stream_channels);
            null_done += n;
        }
        if (now - last_try >= OUT_RETRY_MS * 1000000ull) {
            last_try = now;
            if (device_open(0)) rt_log("audout: an audio output is back");
        }
    }
}

int audio_out_start(int channels, AudioOutFill fill) {
    w.fill = fill;
    w.channels = channels == AUDIO_OUT_AUTO ? 0 : channels == 6 ? 6 : 2;
    w.stream_channels = w.channels ? w.channels : 2;
    w.event = CreateEventW(NULL, FALSE, FALSE, NULL);
    w.wake = CreateEventW(NULL, FALSE, FALSE, NULL);
    w.ready = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!w.event || !w.wake || !w.ready) {
        rt_log("audout: CreateEvent failed: no audio output");
        return 0;
    }
    if (!plat_thread_new(out_thread, NULL, "audout")) {
        rt_log("audout: cannot start the audio output thread");
        return 0;
    }
    if (WaitForSingleObject(w.ready, 5000) != WAIT_OBJECT_0) {
        rt_log("audout: the audio output is slow to open; going on%s", w.channels ? "" : " with the game mixing stereo");
        plat_cas32(&w.channels, 2, 0);
    }
    return w.channels;
}

void audio_out_info(AudioOutInfo *out) {
    plat_mutex_lock(&w.info_lock);
    *out = w.info;
    plat_mutex_unlock(&w.info_lock);
}
