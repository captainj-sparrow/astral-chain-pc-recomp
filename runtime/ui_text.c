// Text the host shows the player (ui_text.h): the English text is compiled in (text_en, the fallback of every id), and
// every other language comes from a compiled gettext catalog, lang/<tag>.mo beside the executable (AC_LANG_DIR moves
// it), read once at start by ui_text_init before any other thread runs.
//
// The catalogs are made from lang/<tag>.po by tools/ui_text_catalogs.py, which build_game.py runs and which copies the
// .mo files next to the executable it builds; the release ships them and the installer copies them into the install
// folder. An entry is msgctxt = the id's name (ui_text_keys.inc, generated from the enum), msgid = the English source,
// msgstr = the translation. An entry is used only when its msgid is the English text this build has, so a translation
// made before the English changed falls back to English instead of saying something else, and only when text_ok
// accepts it: valid UTF-8, the same printf conversions as the English (the menu formats it with the English
// arguments), and within the lengths the software keyboard and the description pane take. Players may drop their own
// .mo files in, so the parser bounds every offset and length by the file and trusts nothing in it.
//
// Two kinds of catalog: a menu catalog (it, ru, pt-BR, zh-Hans: every id) that the menu offers as a language, and a
// keyboard catalog (ja, fr, de, es, nl, pt, zh-Hant, ko: only UI_SWKBD_*) that the game's name-entry keyboard uses in
// that game language (ui_text). Before the catalogs were external, these texts were rows of this file; the menu
// languages' rows became lang/{ru,pt-BR,zh-Hans}.po and the keyboard rows the other .po files, with the same strings.
//
// After ui_text_init the rows never change, so readers on any thread take no lock. Reading the twelve shipped catalogs
// takes 1.4 ms at start (five starts of --probe-cfg on the Arc 140T machine, warm file cache), the id of each entry
// found by a binary search over the sorted names.
#include <ctype.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "ov.h"
#include "platform/platform.h"
#include "ui_text.h"

// The language names are the same in every row.
#define LANG_NAMES \
    [UI_MENU_LANGNAME_EN] = "English", [UI_MENU_LANGNAME_JA] = "日本語", [UI_MENU_LANGNAME_FR] = "Français", \
    [UI_MENU_LANGNAME_DE] = "Deutsch", [UI_MENU_LANGNAME_IT] = "Italiano", [UI_MENU_LANGNAME_ES] = "Español", \
    [UI_MENU_LANGNAME_NL] = "Nederlands", [UI_MENU_LANGNAME_RU] = "Русский", \
    [UI_MENU_LANGNAME_ZH_HANS] = "简体中文", [UI_MENU_LANGNAME_ZH_HANT] = "繁體中文", [UI_MENU_LANGNAME_KO] = "한국어", \
    [UI_MENU_LANGNAME_PT_BR] = "Português (Brasil)"

// SWKBD: TITLE, PROMPT, DEFAULT_NAME (at most 9 UTF-16 units, no '@' '%' '\'), OK, CANCEL, PAD_HINT.
#define SWKBD(title, prompt, name, ok, cancel, hint) \
    [UI_SWKBD_TITLE] = title, [UI_SWKBD_PROMPT] = prompt, [UI_SWKBD_DEFAULT_NAME] = name, [UI_SWKBD_OK] = ok, \
    [UI_SWKBD_CANCEL] = cancel, [UI_SWKBD_PAD_HINT] = hint, LANG_NAMES

static const char *const text_en[UI_TEXT_COUNT] = {
    SWKBD("Text entry", "Enter a name", "Horse", "OK", "Cancel", "Controller: A = OK, B = Cancel"),

    [UI_MENU_TITLE] = "Settings",
    [UI_MENU_TAB_SCREEN] = "Screen", [UI_MENU_TAB_GAME] = "Game", [UI_MENU_TAB_MODS] = "Mods", [UI_MENU_TAB_SOUND] = "Sound",
    [UI_MENU_TAB_CONTROLS] = "Controls", [UI_MENU_TAB_MENU] = "Menu", [UI_MENU_TAB_ADVANCED] = "Advanced",
    [UI_MENU_CONTINUE] = "Continue", [UI_MENU_QUIT] = "Quit game",
    [UI_MENU_ON] = "On", [UI_MENU_OFF] = "Off", [UI_MENU_AUTO] = "Automatic", [UI_MENU_DEFAULT_MARK] = "default",
    [UI_MENU_AFTER_RESTART] = "after restart", [UI_MENU_SET_BY_ENV] = "set by environment variable",
    [UI_MENU_SAVE_FAILED] = "could not save settings",
    [UI_MENU_VERSION] = "Astral Chain Recomp %s, game %s",

    [UI_MENU_FULLSCREEN] = "Fullscreen", [UI_MENU_FULLSCREEN_DESC] = "Fills the screen with the game; F11 toggles it.",
    [UI_MENU_WINDOW_SIZE] = "Window size", [UI_MENU_WINDOW_SIZE_DESC] = "Size of the game window when not fullscreen.",
    [UI_MENU_WINDOW_SIZE_CUSTOM] = "Custom",
    [UI_MENU_PRESENT] = "Presentation", [UI_MENU_PRESENT_DESC] = "How finished frames reach the screen.",
    [UI_MENU_PRESENT_MAILBOX] = "Mailbox", [UI_MENU_PRESENT_VSYNC] = "VSync", [UI_MENU_PRESENT_IMMEDIATE] = "Immediate",
    [UI_MENU_PRESENT_IN_USE] = "in use: %s",
    [UI_MENU_LOW_LATENCY] = "Low latency",
    [UI_MENU_LOW_LATENCY_DESC] = "Starts each frame as late as it can still be shown on time, so the picture answers the "
                                 "controls sooner. Uses NVIDIA Reflex or AMD Anti-Lag 2 when the driver offers them, the "
                                 "port's own timing otherwise. Boost: with Reflex the GPU also keeps its clocks up; "
                                 "otherwise shorter waits, with a dropped frame now and then.",
    [UI_MENU_LOW_LATENCY_OFF] = "Off", [UI_MENU_LOW_LATENCY_ON] = "On", [UI_MENU_LOW_LATENCY_BOOST] = "Boost",
    [UI_MENU_LOW_LATENCY_BUILTIN] = "built-in timing",
    [UI_MENU_GPU] = "Graphics card", [UI_MENU_GPU_DESC] = "The Vulkan device that draws the game.",
    [UI_MENU_GPU_AUTO] = "Automatic",
    [UI_MENU_PERF_OVERLAY] = "Performance overlay",
    [UI_MENU_PERF_OVERLAY_DESC] = "Frame rate in the top right corner. Full adds frame times with a graph, CPU, GPU and "
                                  "memory load, temperatures and the render thread's waits. F10 cycles the modes.",
    [UI_MENU_PERF_OFF] = "Off", [UI_MENU_PERF_FPS] = "FPS", [UI_MENU_PERF_FULL] = "Full",
    // printf format of the overlay line in FPS mode: the frame rate
    [UI_PERF_LINE_FPS] = "%.0f fps",
    [UI_MENU_GRAPHICS] = "Picture",
    [UI_MENU_OUTPUT_RES] = "Game resolution",
    [UI_MENU_OUTPUT_RES_DESC] = "The size the game draws its picture and interface in. 1280 x 720 is the lightest, for "
                                "weaker GPUs. Above 1920 x 1080 the game needs more memory (about 240 MB more at "
                                "2560 x 1440, 790 MB at 3840 x 2160) and GPU time. Takes effect after a restart. "
                                "Automatic takes the screen's size: up to 1920 x 1080, and up to 3840 x 2160 while an "
                                "upscaler runs.",
    [UI_MENU_UPSCALER] = "Scaling to the window",
    [UI_MENU_UPSCALER_DESC] = "How the picture is scaled to the window. FSR keeps edges and detail sharper than "
                              "bilinear, and sharpens the picture even when it already fits the window.",
    [UI_MENU_UPSCALER_FSR] = "AMD FSR 1", [UI_MENU_UPSCALER_BILINEAR] = "Bilinear",
    [UI_MENU_ASPECT] = "Picture shape",
    [UI_MENU_ASPECT_DESC] = "On a screen that is not 16:9 (21:9, 32:9, 16:10 like the Steam Deck). Fill the screen "
                            "widens the camera to the screen's shape: more of the world shows at the sides (above and "
                            "below on a taller screen), while menus and movies keep 16:9. 16:9 (original) keeps the "
                            "game's own view, with black bars. On a 16:9 screen the two are the same.",
    [UI_MENU_ASPECT_FIT] = "16:9 (original)", [UI_MENU_ASPECT_WIDE] = "Fill the screen",
    [UI_MENU_FOV] = "Field of view",
    [UI_MENU_FOV_DESC] = "How much of the world the camera shows, in percent of the game's own (50 degrees vertically in play). Cutscenes keep the game's own framing.",
    [UI_MENU_DRAW_DISTANCE] = "Draw distance",
    [UI_MENU_DRAW_DISTANCE_DESC] = "How far away objects keep their detailed model and small objects stay visible, in percent of the game's own. Higher shows less pop-in and draws more. Takes effect on what the game loads after the change, at the next loading screen.",
    [UI_MENU_ANISO] = "Texture filtering",
    [UI_MENU_ANISO_DESC] = "Anisotropic filtering of surface textures: ground, roads and walls seen at a shallow angle stay sharp into the distance instead of blurring. Game: the level the game asks for, as on the Switch. Auto: 16x with 3D render scale 1440p, twice the game's level at 2160p, the game's level otherwise.",
    [UI_MENU_ANISO_GAME] = "Game",
    [UI_MENU_ANISO_AUTO] = "Auto",
    [UI_MENU_SHARPNESS] = "FSR sharpness",
    [UI_MENU_SHARPNESS_DESC] = "Sharpening after FSR scaling; 0 turns it off.",
    [UI_MENU_COLOR] = "Colour",
    [UI_MENU_COLOR_PRESET] = "Colour profile",
    [UI_MENU_COLOR_PRESET_DESC] = "Sets the four values below. Vivid: more contrast and saturation, shadows opened a "
                                  "little (made for OLED screens). Clarity: a little more contrast, saturation and "
                                  "brightness. Moving a value makes the profile Custom. The HUD changes with the picture; "
                                  "this menu does not.",
    [UI_MENU_COLOR_DEFAULT] = "Game's own", [UI_MENU_COLOR_VIVID] = "Vivid (OLED)", [UI_MENU_COLOR_CLARITY] = "Clarity",
    [UI_MENU_COLOR_CUSTOM] = "Custom",
    [UI_MENU_COLOR_BRIGHTNESS] = "Brightness", [UI_MENU_COLOR_BRIGHTNESS_DESC] = "Added to every colour; 0 leaves it.",
    [UI_MENU_COLOR_CONTRAST] = "Contrast",
    [UI_MENU_COLOR_CONTRAST_DESC] = "Spreads colours away from mid-grey (above 100 %) or towards it; 100 % leaves them.",
    [UI_MENU_COLOR_SATURATION] = "Saturation",
    [UI_MENU_COLOR_SATURATION_DESC] = "Colour strength against grey; 100 % leaves it.",
    [UI_MENU_COLOR_GAMMA] = "Gamma",
    [UI_MENU_COLOR_GAMMA_DESC] = "Below 1 brightens dark and middle tones, above 1 darkens them; 1 leaves them.",
    [UI_MENU_HDR_GROUP] = "HDR",
    [UI_MENU_HDR] = "HDR output",
    [UI_MENU_HDR_DESC] = "Bright highlights on an HDR display. Needs HDR turned on in Windows (\"Use HDR\") or in a "
                         "Wayland compositor with colour management; without it the picture stays SDR. Applies after a "
                         "restart. The colour profile above is for SDR and is not used in HDR.",
    [UI_MENU_HDR_PEAK] = "Peak brightness",
    [UI_MENU_HDR_PEAK_DESC] = "Your display's peak in nits; the brightest highlights stop there.",
    [UI_MENU_HDR_GAME] = "Game brightness",
    [UI_MENU_HDR_GAME_DESC] = "Nits of the picture's white (paper white). 203 is the HDR reference; raise it in a "
                              "bright room.",
    [UI_MENU_HDR_UI] = "HUD and menu brightness",
    [UI_MENU_HDR_UI_DESC] = "Nits of the HUD's and this menu's white.",
    [UI_MENU_HDR_HUE] = "Highlight hue correction",
    [UI_MENU_HDR_HUE_DESC] = "Bright colours such as fire and sparks turn towards white as in the game's own SDR picture, "
                             "matching the light they cast; 0 keeps them fully saturated.",
    [UI_MENU_SHADOW_SCALE] = "Shadow resolution",
    [UI_MENU_SHADOW_SCALE_DESC] = "Draws the sun's shadow map this many times larger on each side than the game's "
                                  "2048x2048: sharper shadow edges and leaf shadows, for more video memory (about "
                                  "85 to 130 MiB at 2x, 340 to 520 MiB at 4x). Takes effect at the next start.",
    [UI_MENU_AO] = "Ambient occlusion",
    [UI_MENU_AO_DESC] = "Darkens creases, corners and the ground under characters and objects, where little of the "
                        "surrounding light reaches. High + bounce light also adds the light the surfaces on screen send "
                        "back to each other: a lit wall or floor brightens and tints what faces it, even from far away. "
                        "Computed on the screen after the world is lit; the game has none of its own. Each level costs "
                        "more GPU time than the one before; High + bounce light is demanding, so on a weak graphics card "
                        "prefer High. Takes effect at once.",
    [UI_MENU_AO_OFF] = "Off",
    [UI_MENU_AO_LOW] = "Low",
    [UI_MENU_AO_HIGH] = "High",
    [UI_MENU_AO_GI] = "High + bounce light (demanding)",
    [UI_MENU_UPSCALING] = "Upscaler",
    [UI_MENU_UPSCALING_DESC] = "A temporal upscaler in place of the game's anti-aliasing: smoother edges and less "
                               "shimmer, and with a mode below Native more frames per second. NVIDIA DLSS needs an NVIDIA "
                               "RTX GPU and nvngx_dlss.dll next to astralchain.exe, Intel XeSS a recent GPU and "
                               "libxess.dll next to astralchain.exe; AMD FSR 3 runs on any GPU, AMD FSR 4 on a Radeon "
                               "RX 6000 or newer. Automatic takes DLSS where it can run and FSR 3 elsewhere. DLSS, XeSS "
                               "and FSR 4 chosen here take effect after a restart. The HUD is not touched.",
    [UI_MENU_UPSCALING_DESC_OTHER] = "A temporal upscaler in place of the game's anti-aliasing: smoother edges and less "
                                     "shimmer, and with a mode below Native more frames per second. NVIDIA DLSS needs "
                                     "an NVIDIA RTX GPU and libnvidia-ngx-dlss.so next to astralchain, AMD FSR 3 runs "
                                     "on any GPU, AMD FSR 4 on a Radeon RX 6000 or newer; Automatic takes DLSS where it "
                                     "can run and FSR 3 elsewhere (Intel XeSS is Windows only). DLSS and FSR 4 chosen "
                                     "here take effect after a restart. The HUD is not touched.",
    [UI_MENU_UPSCALING_OFF] = "Off", [UI_MENU_UPSCALING_AUTO] = "Automatic",
    [UI_MENU_UPSCALING_UNAVAILABLE] = "not available here",
    [UI_MENU_UPSCALING_INSTEAD] = "not available here: %s in use",
    [UI_MENU_UPSCALING_MISSING] = "%s is missing: %s in use",
    [UI_MENU_UPSCALING_UNTIL] = "%s until a restart",
    [UI_MENU_UPSCALING_QUALITY] = "Upscaler mode",
    [UI_MENU_UPSCALING_QUALITY_DESC] = "The size the 3D scene is drawn at before the upscaler brings it to the game "
                                       "resolution (shown under the value). Native draws it at full size, the sharpest "
                                       "(DLAA under DLSS). Quality, Balanced, Performance and Ultra performance draw it at "
                                       "about 67, 58, 50 and 33 % for more frames per second; Ultra performance is meant "
                                       "for 3840 x 2160. Takes effect after a restart.",
    [UI_MENU_UPSCALING_NATIVE] = "Native", [UI_MENU_DLSS_QUALITY] = "Quality", [UI_MENU_DLSS_BALANCED] = "Balanced",
    [UI_MENU_DLSS_PERFORMANCE] = "Performance", [UI_MENU_UPSCALING_ULTRA] = "Ultra performance",
    [UI_MENU_UPSCALING_MODE_OFF] = "no upscaler",
    [UI_MENU_AA] = "Game anti-aliasing",
    [UI_MENU_AA_DESC] = "The game's own anti-aliasing, SMAA, for when no upscaler runs. Off leaves the edges jagged.",
    [UI_MENU_AA_OFF] = "Off",
    [UI_MENU_AA_BY_UPSCALER] = "the upscaler smooths the edges",
    [UI_MENU_FRAMEGEN] = "Frame generation",
    [UI_MENU_FRAMEGEN_DESC] = "NVIDIA DLSS Frame Generation: frames made between the game's own, for a smoother "
                              "picture; the game itself keeps its frame rate. 2x shows one frame between two of the "
                              "game's, 3x and 4x two and three (RTX 50 series). Automatic fills the display's refresh "
                              "rate. Needs an NVIDIA RTX 40 or newer GPU, the DLSS SDK's frame generation library next to "
                              "the game and an upscaler running; adds about half a frame of latency. Not with HDR output "
                              "yet. Turning it on takes effect after a restart, then at once.",
    [UI_MENU_FRAMEGEN_NEEDS_TAA] = "needs an upscaler",
    [UI_MENU_OUTPUT_RES_AUTO] = "Automatic",
    [UI_MENU_OUTPUT_RES_NEXT] = ", %s after restart",

    [UI_MENU_FPS_LIMIT] = "Frame rate",
    [UI_MENU_FPS_LIMIT_DESC] = "60 FPS runs the game's 60 fps patch; 30 FPS switches it off. Read live.",
    [UI_MENU_FPS_30] = "30 FPS", [UI_MENU_FPS_60] = "60 FPS",
    [UI_MENU_REALTIME] = "Game speed follows the clock",
    [UI_MENU_REALTIME_DESC] = "Steps the game by real time, so a slow frame does not slow the game down.",
    [UI_MENU_CLOTH_REALTIME] = "Cloth on the clock",
    [UI_MENU_CLOTH_REALTIME_DESC] = "Steps cloth and hair by real time as well.",
    [UI_MENU_HIGHFPS_FIX] = "High frame rate fixes",
    [UI_MENU_HIGHFPS_FIX_DESC] = "Corrects physics that misbehave above 30 frames per second, and objects that stay see-through or invisible above 60.",
    [UI_MENU_GAME_LANG] = "Game language", [UI_MENU_GAME_LANG_DESC] = "Language of the game's text and voices.",
    [UI_MENU_GAME_LANG_AUTO] = "Automatic",
    [UI_MENU_OPERATION_MODE] = "Console mode",
    [UI_MENU_OPERATION_MODE_DESC] = "Whether the game thinks it runs docked or handheld.",
    [UI_MENU_MODE_DOCKED] = "Docked", [UI_MENU_MODE_HANDHELD] = "Handheld",
    [UI_MENU_HUD] = "HUD",
    [UI_MENU_HUD_DESC] = "Off, the game's HUD is not drawn during play; the menus stay. F12 toggles it.",
    [UI_MENU_ASSUME_FOCUS] = "Keep running out of focus",
    [UI_MENU_ASSUME_FOCUS_DESC] = "On (the default), the game keeps running while you are in another window, and the "
                                  "keyboard stays quiet until the window is in front again. Off, the whole game, "
                                  "sound included, pauses while its window is out of focus, as on the Switch when Home "
                                  "is pressed.",

    [UI_MENU_MODS_INSTALLED] = "Installed mods",
    [UI_MENU_MODS_NONE] = "No mods yet. Unpack a mod into the mods folder, one folder per mod with its romfs folder "
                          "inside, and restart the game. TKMM can export its merged mods into a folder there too.",
    [UI_MENU_MODS_OPEN] = "Open mods folder",
    [UI_MENU_MODS_OPEN_DESC] = "Shows the folder the game loads mods from. It is created if it is missing.",
    [UI_MENU_DLC_GROUP] = "Downloadable content",
    [UI_MENU_DLC] = "Load DLC",
    [UI_MENU_DLC_DESC] = "Loads the DLCs of the dlc folder next to astralchain.exe: trainings, texts and tutorial windows written by players as XML. They are built into the game's archives from your own game files at start and kept in your profile. Off: the game runs exactly as it is.",
    [UI_MENU_DLC_LOADED] = "Loaded at this start: %s.",
    [UI_MENU_DLC_NONE] = "No DLC was found in the dlc folder at this start.",
    [UI_MENU_DLC_OPEN] = "Open DLC folder",
    [UI_MENU_DLC_OPEN_DESC] = "Shows the folder the game loads DLCs from, one folder per DLC with its dlc.xml. It is created if it is missing.",
    [UI_MENU_MODS_ROW_DESC] = "Game files it replaces: %u. New files it adds: %u.",
    [UI_MENU_MODS_OVERRIDDEN] = "%u of its files come from %s, which is higher in the list.",
    [UI_MENU_MODS_MOVE_HINT] = "The top face button or Ctrl+Up moves it up, the left face button or Ctrl+Down moves it "
                               "down. When two mods change the same file, the higher one wins. Changes apply after a "
                               "restart.",
    [UI_MENU_MODS_EMPTY] = "This folder has no romfs, exefs or cheats files.",
    [UI_MENU_MODS_SUBSDK] = "This mod brings its own game code (a subsdk plugin or a replaced main). The port cannot "
                            "load code modules, so the whole mod stays off.",
    [UI_MENU_MODS_EXEFS_LATER] = "This mod patches the game code (exefs). The game builds the patches in when it "
                                 "starts next (a rebuild of a few minutes); until then the whole mod stays off.",
    [UI_MENU_MODS_CHEATS] = "Cheats: %d.",
    [UI_MENU_MODS_CHEATS_FILE] = "Cheats for this game version are read at start from cheats\\%s in this mod.",
    [UI_MENU_MODS_BAD_NAME] = "The folder name contains |, which the mod list uses between names. Rename the folder.",
    [UI_MENU_MODS_OTHER_VERSION] = "%u of its files are made for another game version and are left out. Look for a "
                                   "version of this mod made for your game version.",
    [UI_MENU_MODS_UNUSED] = "not used",

    [UI_MENU_VOLUME] = "Volume", [UI_MENU_VOLUME_DESC] = "Master volume of the game.",
    [UI_MENU_AUDIO_OUTPUT] = "Audio output",
    [UI_MENU_AUDIO_OUTPUT_DESC] = "What the game mixes. In 5.1 the centre (the voices) and the surrounds get channels of their own; it needs a 5.1 or 7.1 output. Elsewhere the 5.1 mix is folded down to stereo. Automatic picks 5.1 when the output takes it at start, stereo otherwise.",
    [UI_MENU_AUDIO_OUTPUT_DESC_WIN] = "What the game mixes. In 5.1 the centre (the voices) and the surrounds get channels of their own; it needs a 5.1 or 7.1 output, or a headset with spatial sound on: Windows Sonic for Headphones, Dolby Atmos for Headphones or DTS Headphone:X, in the device's properties in the Windows sound settings. Elsewhere the 5.1 mix is folded down to stereo. Automatic picks 5.1 when the output takes it at start, stereo otherwise.",
    [UI_MENU_AUDIO_AUTO] = "Automatic",
    [UI_MENU_AUDIO_STEREO] = "Stereo",
    [UI_MENU_AUDIO_51] = "5.1",
    [UI_MENU_AUDIO_TAKES_51_SPATIAL] = "%s: Windows spatial sound is on and turns 5.1 into surround on headphones.",
    [UI_MENU_AUDIO_TAKES_51] = "%s: %d channels, 5.1 goes out as it is.",
    [UI_MENU_AUDIO_FOLDS_51] = "%s: %d channels, no 5.1: a 5.1 mix is folded down to stereo.",
    [UI_MENU_AUDIO_SONIC_HINT] = "With headphones, turn on Windows Sonic for Headphones (or Dolby Atmos, DTS Headphone:X) in the device's properties.",
    [UI_MENU_AUDIO_SENDING] = "The game sends %d channels.",
    [UI_MENU_AUDIO_NO_DEVICE] = "No audio output found.",
    [UI_MENU_AUDIO_MIX_HINT] = "Choose 5.1 above and restart the game to set each speaker.",
    [UI_MENU_AUDIO_GROUP_OUTPUT] = "Speakers and headphones",
    [UI_MENU_AUDIO_GROUP_MIX] = "Speaker mix",
    [UI_MENU_AUDIO_RESTART_DESC] = "Closes the game and starts it again with the new audio output. Progress since your last save is lost.",
    [UI_MENU_AUDIO_TEST] = "Test speakers",
    [UI_MENU_AUDIO_TEST_DESC] = "Plays a noise on each speaker in turn, as an A/V receiver does: check that every speaker works and sits where the diagram shows it. The game's sound is muted meanwhile. Press again to stop.",
    [UI_MENU_AUDIO_TEST_START] = "Start",
    [UI_MENU_AUDIO_TEST_STOP] = "press to stop",
    [UI_MENU_AUDIO_PRESET] = "Mix",
    [UI_MENU_AUDIO_PRESET_DESC] = "Ready-made mixes. Balanced: the recommended mix. Clear voices: the dialogue in front, softer surrounds. Immersive: stronger surrounds, and the bass to the subwoofer. Moving any speaker below makes it Custom.",
    [UI_MENU_AUDIO_PRESET_BALANCED] = "Balanced",
    [UI_MENU_AUDIO_PRESET_VOICES] = "Clear voices",
    [UI_MENU_AUDIO_PRESET_IMMERSIVE] = "Immersive",
    [UI_MENU_AUDIO_PRESET_CUSTOM] = "Custom",
    [UI_MENU_AUDIO_FRONT_LEFT] = "Front left",
    [UI_MENU_AUDIO_FRONT_RIGHT] = "Front right",
    [UI_MENU_AUDIO_CENTER] = "Centre (voices)",
    [UI_MENU_AUDIO_SURROUND_LEFT] = "Surround left",
    [UI_MENU_AUDIO_SURROUND_RIGHT] = "Surround right",
    [UI_MENU_AUDIO_SUBWOOFER] = "Subwoofer",
    [UI_MENU_AUDIO_FRONT_DESC] = "Level of this front speaker. The fronts carry mostly the music, which the game mixes in stereo, and the sounds in front of you; lowering them lowers the music most. For the music alone, use the game's own music volume.",
    [UI_MENU_AUDIO_SURROUND_DESC] = "Level of this surround speaker. The surrounds carry the ambience, the reverb and the sounds beside and behind you.",
    [UI_MENU_AUDIO_CENTER_DESC] = "Level of the centre speaker, where the voices and the interface sounds are. Lower it if voices or beeps are too loud; -2 dB keeps them near their level in stereo.",
    [UI_MENU_AUDIO_SUBWOOFER_DESC] = "Level of the subwoofer. The game sends it nothing of its own: turn on Bass to subwoofer to give it the low end of the other speakers.",
    [UI_MENU_AUDIO_BASS] = "Bass to subwoofer",
    [UI_MENU_AUDIO_BASS_DESC] = "Sends the bass (below 120 Hz) of the other five speakers to the subwoofer. Leave it off with headphones, or when your receiver already manages the bass.",
    [UI_MENU_SPK_LEFT] = "Left",
    [UI_MENU_SPK_RIGHT] = "Right",
    [UI_MENU_SPK_CENTER] = "Centre",
    [UI_MENU_SPK_SURROUND_LEFT] = "Surround L",
    [UI_MENU_SPK_SURROUND_RIGHT] = "Surround R",
    [UI_MENU_SPK_SUB] = "Sub",
    [UI_MENU_SPK_YOU] = "You",

    [UI_MENU_MOTION] = "Motion controls", [UI_MENU_MOTION_DESC] = "Aiming with the controller's gyroscope.",
    [UI_MENU_RUMBLE_STRENGTH] = "Vibration strength",
    [UI_MENU_RUMBLE_STRENGTH_DESC] = "How strong the controller vibrates.",
    [UI_MENU_RUMBLE_CURVE] = "Vibration curve",
    [UI_MENU_RUMBLE_CURVE_DESC] = "Lower values make weak vibration stronger.",
    [UI_MENU_RUMBLE_TRIGGERS] = "Trigger vibration (Xbox)",
    [UI_MENU_RUMBLE_TRIGGERS_DESC] = "Vibrates the triggers of an Xbox controller too.",
    [UI_MENU_PAD_LAYOUT] = "Button layout",
    [UI_MENU_PAD_LAYOUT_DESC] = "Which button is A. Nintendo: the right one, as on a Switch. Xbox: the bottom one, as "
                                "labelled on an Xbox controller. PlayStation: the bottom one as labelled (cross) and "
                                "the rest by position, to match the PlayStation button mod. Automatic picks Nintendo "
                                "for controllers with Nintendo labels, PlayStation for PlayStation controllers and "
                                "Xbox for the others. Which controller the game's prompts show is set under Button "
                                "prompts.",
    [UI_MENU_PAD_LAYOUT_AUTO] = "Automatic", [UI_MENU_PAD_LAYOUT_NINTENDO] = "Nintendo (A right)",
    [UI_MENU_PAD_LAYOUT_XBOX] = "Xbox (A bottom)", [UI_MENU_PAD_LAYOUT_PLAYSTATION] = "PlayStation (A bottom)",
    [UI_MENU_PAD_LAYOUT_IN_USE] = "in use: %s",
    [UI_MENU_PROMPTS] = "Button prompts",
    [UI_MENU_PROMPTS_DESC] = "Which controller's buttons the game's prompts show. Automatic follows the controller "
                             "in use (the Xbox or PlayStation mod when built, else Nintendo). Nintendo: the game's "
                             "own. Xbox and PlayStation: button mods drawn for one layout each (a loaded mod forces "
                             "its layout: Xbox for the Xbox mod, PlayStation for the PlayStation mod). The game reads "
                             "its prompts at start, so they change after a restart.",
    [UI_MENU_PROMPTS_AUTO] = "Automatic", [UI_MENU_PROMPTS_NINTENDO] = "Nintendo", [UI_MENU_PROMPTS_XBOX] = "Xbox",
    [UI_MENU_PROMPTS_PLAYSTATION] = "PlayStation",
    [UI_MENU_PROMPTS_MISSING] = "prompt files missing: run the setup again",
    [UI_MENU_PROMPTS_FAILED] = "prompt files damaged: run the setup again",
    [UI_MENU_PROMPTS_RESTART] = "prompts change after restart",
    [UI_MENU_PROMPTS_OFF_SHAPE] = "off at a game resolution not 16:9",
    [UI_MENU_PROMPTS_OFF_MODS] = "off: a RomFS mod replaces their files",
    [UI_MENU_PROMPTS_OFF_PATCHES] = "off while AC_ROMFS_PATCHES is set",
    [UI_MENU_RESTART] = "Restart now",
    [UI_MENU_RESTART_DESC] = "Closes the game and starts it again with the new prompts. Progress since your last save "
                             "is lost.",
    [UI_MENU_RESTART_CONFIRM] = "Restart the game now? Progress since your last save is lost.",
    [UI_MENU_RESTART_YES] = "Restart",
    [UI_MENU_RESTART_SAVING] = "The game is saving: Restart is possible in a moment.",
    [UI_MENU_LSTICK_DEADZONE] = "Left stick dead zone", [UI_MENU_RSTICK_DEADZONE] = "Right stick dead zone",
    [UI_MENU_STICK_DEADZONE_DESC] = "How far a gamepad stick moves before the game sees it. Raise it if the character or the camera moves on its own: let go of the stick, read \"stick now\" and set the dead zone a few percent above it.",
    [UI_MENU_STICK_DEADZONE_NOW] = "stick now: %.0f %%",
    [UI_MENU_TRIGGER_THRESHOLD] = "Trigger threshold",
    [UI_MENU_TRIGGER_THRESHOLD_DESC] = "How far a gamepad trigger (ZL, ZR) is pulled before the game sees it pressed. 0 reacts to the slightest pull. Raise it if a trigger presses on its own: let go of it, read \"triggers now\" and set the threshold a few percent above it.",
    [UI_MENU_TRIGGER_THRESHOLD_NOW] = "triggers now: L %.0f %%, R %.0f %%",
    [UI_MENU_MOUSE_SENS] = "Mouse camera sensitivity",
    [UI_MENU_MOUSE_SENS_DESC] = "How fast the mouse turns the camera.",
    [UI_MENU_WALK_KEY] = "Walk key",
    [UI_MENU_WALK_KEY_DESC] = "Hold it to tilt the keyboard's sticks part of the way, so the keyboard walks instead of "
                              "runs. The walk scale sets how far. A controller and the mouse are not affected.",
    [UI_MENU_WALK_KEY_TAKEN] = "also %s",
    [UI_MENU_WALK_SCALE] = "Walk scale",
    [UI_MENU_WALK_SCALE_DESC] = "The tilt the keyboard's sticks get while the walk key is held. Half is a walk, the "
                                "full tilt a run.",
    [UI_MENU_STICK_SENS] = "Stick camera speed",
    [UI_MENU_STICK_SENS_DESC] = "Multiplies the game's own right-stick camera speed, top speed included, at each of the "
                                "game's five camera sensitivities. The mouse keeps its own setting.",

    [UI_BINDS_HELP] = "Select a cell and press A or Enter, then the key, mouse button or controller button. Delete or "
                      "X clears a cell. A key in use moves here; a controller button swaps with the one it replaces.",
    [UI_BINDS_COL_KEY1] = "Key 1", [UI_BINDS_COL_KEY2] = "Key 2", [UI_BINDS_COL_PAD] = "Controller",
    [UI_BINDS_GROUP_BUTTONS] = "Buttons", [UI_BINDS_GROUP_MOVE] = "Movement", [UI_BINDS_GROUP_CAMERA] = "Camera",
    [UI_BINDS_PRESS_KEY] = "Press a key or mouse button for %s",
    [UI_BINDS_PRESS_PAD] = "Press a controller button for %s",
    [UI_BINDS_CANCEL_HINT] = "Esc cancels, or wait %d s",
    [UI_BINDS_TAKEN] = "%s was taken from %s", [UI_BINDS_SWAPPED] = "%s swapped with %s",
    [UI_BINDS_RESET] = "Reset all buttons",
    [UI_BINDS_RESET_DESC] = "Back to the default keys and controller buttons. The button layout above applies again: remapped controller buttons stay where you put them until this reset.",
    [UI_BIND_A] = "A", [UI_BIND_B] = "B", [UI_BIND_X] = "X", [UI_BIND_Y] = "Y",
    [UI_BIND_LCLICK] = "Left stick click", [UI_BIND_RCLICK] = "Right stick click", [UI_BIND_L] = "L",
    [UI_BIND_R] = "R", [UI_BIND_ZL] = "ZL", [UI_BIND_ZR] = "ZR", [UI_BIND_PLUS] = "Plus", [UI_BIND_MINUS] = "Minus",
    [UI_BIND_LEFT] = "D-pad left", [UI_BIND_UP] = "D-pad up", [UI_BIND_RIGHT] = "D-pad right",
    [UI_BIND_DOWN] = "D-pad down",
    [UI_BIND_LSTICK_LEFT] = "Move left", [UI_BIND_LSTICK_UP] = "Move forward", [UI_BIND_LSTICK_RIGHT] = "Move right",
    [UI_BIND_LSTICK_DOWN] = "Move back",
    [UI_BIND_RSTICK_LEFT] = "Camera left", [UI_BIND_RSTICK_UP] = "Camera up", [UI_BIND_RSTICK_RIGHT] = "Camera right",
    [UI_BIND_RSTICK_DOWN] = "Camera down",
    [UI_BIND_DESC_A] = "Interact, pick up, talk.", [UI_BIND_DESC_B] = "Run, cancel, go back.",
    [UI_BIND_DESC_X] = "Jump.", [UI_BIND_DESC_Y] = "Attack.", [UI_BIND_DESC_LCLICK] = "Crouch.",
    [UI_BIND_DESC_RCLICK] = "First-person view.", [UI_BIND_DESC_L] = "Abilities.",
    [UI_BIND_DESC_R] = "Throw a weapon.", [UI_BIND_DESC_ZL] = "Shield, lock on.", [UI_BIND_DESC_ZR] = "Bow.",
    [UI_BIND_DESC_PLUS] = "Menu, inventory.", [UI_BIND_DESC_MINUS] = "Map.",
    [UI_BIND_DESC_DPAD] = "Quick-select.",
    [UI_BIND_DESC_MOVE] = "Moves your character like the left stick. The controller's left stick always works too.",
    [UI_BIND_DESC_CAMERA] = "Turns the camera like the right stick. The mouse and the controller's right stick always "
                            "work too.",
    [UI_KEY_SPACE] = "Space", [UI_KEY_TAB] = "Tab", [UI_KEY_LSHIFT] = "Left Shift", [UI_KEY_LCTRL] = "Left Ctrl",
    [UI_KEY_UP] = "Up arrow", [UI_KEY_DOWN] = "Down arrow", [UI_KEY_LEFT] = "Left arrow",
    [UI_KEY_RIGHT] = "Right arrow",
    [UI_KEY_MOUSE_L] = "Left mouse", [UI_KEY_MOUSE_R] = "Right mouse", [UI_KEY_MOUSE_M] = "Middle mouse",
    [UI_KEY_MOUSE_X1] = "Mouse 4", [UI_KEY_MOUSE_X2] = "Mouse 5",
    [UI_KEY_ENTER] = "Enter", [UI_KEY_BACKSPACE] = "Backspace", [UI_KEY_RSHIFT] = "Right Shift",
    [UI_KEY_RCTRL] = "Right Ctrl", [UI_KEY_LALT] = "Alt", [UI_KEY_CAPSLOCK] = "Caps Lock", [UI_KEY_INSERT] = "Insert",
    [UI_KEY_DELETE] = "Delete", [UI_KEY_HOME] = "Home", [UI_KEY_END] = "End", [UI_KEY_PAGEUP] = "Page Up",
    [UI_KEY_PAGEDOWN] = "Page Down", [UI_KEY_NUM_FMT] = "Num %s",
    [UI_PAD_SOUTH] = "Bottom button", [UI_PAD_EAST] = "Right button", [UI_PAD_WEST] = "Left button",
    [UI_PAD_NORTH] = "Top button", [UI_PAD_LSTICK] = "Left stick", [UI_PAD_RSTICK] = "Right stick",
    [UI_PAD_L] = "Left bumper", [UI_PAD_R] = "Right bumper", [UI_PAD_START] = "Start", [UI_PAD_BACK] = "Back",
    [UI_PAD_DPAD_LEFT] = "D-pad left", [UI_PAD_DPAD_UP] = "D-pad up", [UI_PAD_DPAD_RIGHT] = "D-pad right",
    [UI_PAD_DPAD_DOWN] = "D-pad down", [UI_PAD_ZL] = "Left trigger", [UI_PAD_ZR] = "Right trigger",
    [UI_PAD_LPADDLE1] = "Left back button", [UI_PAD_RPADDLE1] = "Right back button",
    [UI_PAD_LPADDLE2] = "Left back button 2", [UI_PAD_RPADDLE2] = "Right back button 2",
    [UI_PAD_CROSS] = "Cross", [UI_PAD_CIRCLE] = "Circle", [UI_PAD_SQUARE] = "Square", [UI_PAD_TRIANGLE] = "Triangle",

    [UI_MENU_MENU_LANG] = "Menu language", [UI_MENU_MENU_LANG_DESC] = "Language of this menu and the overlay.",
    [UI_MENU_MENU_LANG_AUTO] = "Automatic (from Windows)",

    [UI_MENU_ADV_WARNING] = "These settings are for testing and troubleshooting. Change them only when you know why.",
    [UI_MENU_GROUP_SCHED] = "Scheduling", [UI_MENU_GROUP_SHADERS] = "Shaders and pipelines",
    [UI_MENU_GROUP_TEXMEM] = "Textures and memory", [UI_MENU_GROUP_FIXES] = "Render fixes",
    [UI_MENU_GROUP_VULKAN] = "Vulkan", [UI_MENU_GROUP_DIAG] = "Diagnostics",
    [UI_MENU_ASYNC_PREEMPT] = "Virtual core preemption",
    [UI_MENU_ASYNC_PREEMPT_DESC] = "Lets a game thread interrupt another one on the same virtual core.",
    [UI_MENU_PREEMPT_OFF] = "Off", [UI_MENU_PREEMPT_ON] = "On", [UI_MENU_PREEMPT_INHERIT] = "Inheritance only",
    [UI_MENU_REPLAY_ASYNC] = "GPU frame prep on its own thread",
    [UI_MENU_REPLAY_ASYNC_DESC] = "Prepares frames for the GPU on a separate thread instead of the game thread. Adds up to one frame of latency.",
    [UI_MENU_TIMER_1MS] = "1 ms Windows timer",
    [UI_MENU_TIMER_1MS_DESC] = "Raises the system timer resolution to 1 ms for steadier frame pacing.",
    [UI_MENU_PIPE_WAIT] = "Waiting draw handling",
    [UI_MENU_PIPE_WAIT_DESC] = "Default keeps the current scheduling with a reserved worker for waiting draws; Old (draw) uses the previous scheduling.",
    [UI_MENU_PIPE_WAIT_DEFAULT] = "Default", [UI_MENU_PIPE_WAIT_DRAW] = "Old (draw)",
    [UI_MENU_PIPE_SKIP] = "Draws skipped while a pipeline builds",
    [UI_MENU_PIPE_SKIP_DESC] = "Which draws are dropped instead of waiting for their pipeline.",
    [UI_MENU_PIPE_SKIP_STRICT] = "Strict", [UI_MENU_PIPE_SKIP_ANY] = "Any",
    [UI_MENU_PIPE_SKIP_FULLSCREEN] = "Fullscreen passes",
    [UI_MENU_LIB_PREBUILD] = "Stage library prebuild",
    [UI_MENU_LIB_PREBUILD_DESC] = "Builds shader stage libraries ahead of time in the background.",
    [UI_MENU_LIB_PREBUILD_THREADS] = "Stage library prebuild threads",
    [UI_MENU_LIB_PREBUILD_THREADS_DESC] = "Threads used for the prebuild; Automatic picks a number from your CPU.",
    [UI_MENU_PIPE_PREBUILD] = "Pipeline prebuild",
    [UI_MENU_PIPE_PREBUILD_DESC] = "Builds known pipelines ahead of time.",
    [UI_MENU_PREWARM_XLATE] = "Translator prewarm",
    [UI_MENU_PREWARM_XLATE_DESC] = "Threads that translate known shaders at start; Automatic takes half the CPU threads, at most 16.",
    [UI_MENU_GPL] = "Graphics pipeline library",
    [UI_MENU_GPL_DESC] = "Uses the Vulkan pipeline library extension; turn off if your driver misbehaves with it.",
    [UI_MENU_ASTC] = "ASTC texture decoding", [UI_MENU_ASTC_DESC] = "Where compressed ASTC textures are decoded.",
    [UI_MENU_ASTC_AUTO] = "Automatic", [UI_MENU_ASTC_GPU] = "GPU shader", [UI_MENU_ASTC_NATIVE] = "Native",
    [UI_MENU_ASTC_CPU] = "CPU",
    [UI_MENU_MEMORY] = "GPU memory mode", [UI_MENU_MEMORY_DESC] = "How game memory is shared with the GPU.",
    [UI_MENU_MEMORY_AUTO] = "Automatic", [UI_MENU_MEMORY_IMPORT] = "Import", [UI_MENU_MEMORY_MIRROR] = "Mirror",
    [UI_MENU_MEMORY_VERIFY] = "Verify",
    [UI_MENU_TEX_ALIAS] = "Texture aliasing", [UI_MENU_TEX_ALIAS_DESC] = "A new texture created over a live texture's memory starts with that content instead of empty.",
    [UI_MENU_TEX_VRAM_PRESSURE] = "Texture memory limit", [UI_MENU_TEX_VRAM_PRESSURE_DESC] = "Percent of video memory above which textures the game released are dropped (they come back from storage when used again). 0 keeps them.",
    [UI_MENU_ASTC_BC] = "Compress ASTC to BC", [UI_MENU_ASTC_BC_DESC] = "With the GPU decoder: re-encodes decoded ASTC textures to BC1 when the alpha scan proves them opaque, BC3 otherwise — a quarter to an eighth of the video memory. For GPUs with 4 GB or less; textures look slightly blockier.",
    [UI_MENU_LOD_GUARD] = "LOD guard", [UI_MENU_LOD_GUARD_DESC] = "Prevents objects flashing in the wrong place or a flat far field for a frame, by waiting for the game to finish rewriting level-of-detail data before the GPU reads it.",
    [UI_MENU_FIX_C0] = "Constant buffer fix",
    [UI_MENU_FIX_C0_DESC] = "Fills the driver-reserved constant bank c0 (storage sizes, dispatch size, point size, base instance) instead of reading zeros.",
    [UI_MENU_FIX_ZERO] = "Zero new GPU memory",
    [UI_MENU_FIX_ZERO_DESC] = "New GPU memory pools start zero-filled, as Switch memory does.",
    [UI_MENU_BIND_SKIP] = "Skip repeated binds",
    [UI_MENU_BIND_SKIP_DESC] = "Optimisation: skips bindings that did not change.",
    [UI_MENU_SHADER_MEMO] = "Shader memo", [UI_MENU_SHADER_MEMO_DESC] = "Optimisation: remembers shader lookups.",
    [UI_MENU_PIPE_MEMO] = "Pipeline memo", [UI_MENU_PIPE_MEMO_DESC] = "Optimisation: reuses the previous draw's pipeline when its state repeats.",
    [UI_MENU_TEX_MEMO] = "Texture binding memo", [UI_MENU_TEX_MEMO_DESC] = "Optimisation: remembers texture descriptors between draws.",
    [UI_MENU_DESC_TEMPLATE] = "Descriptor templates",
    [UI_MENU_DESC_TEMPLATE_DESC] = "Optimisation: updates descriptors in bulk.",
    [UI_MENU_PUSH_SKIP] = "Skip repeated push constants",
    [UI_MENU_PUSH_SKIP_DESC] = "Optimisation: skips push constants that did not change.",
    [UI_MENU_SPLIT_DRAWS] = "Early GPU submission",
    [UI_MENU_SPLIT_DRAWS_DESC] = "Submits GPU work in parts every N draws between passes so the GPU starts earlier; 0 turns it off.",
    [UI_MENU_VK_VALIDATION] = "Validation layer",
    [UI_MENU_VK_VALIDATION_DESC] = "Checks every Vulkan call and slows the game down a lot.",
    [UI_MENU_VK_VALIDATION_MISSING] = "The validation layer is not installed.",
    [UI_MENU_VK_DYN_VERTEX] = "Dynamic vertex state",
    [UI_MENU_VK_DYN_VERTEX_DESC] = "Forces dynamic vertex input on or off; the default depends on the GPU.",
    [UI_MENU_VK_DYN_A2C] = "Dynamic alpha to coverage",
    [UI_MENU_VK_DYN_A2C_DESC] = "Forces dynamic alpha to coverage on or off. Default: off on AMD's driver for RDNA 2 and "
                                "older, which draws characters black with it, on elsewhere.",
    [UI_MENU_VK_SUBGROUP_PIN] = "Pin subgroup size", [UI_MENU_VK_SUBGROUP_PIN_DESC] = "Fixes the shader subgroup size.",
    [UI_MENU_TRACE] = "Performance log", [UI_MENU_TRACE_DESC] = "Writes frame timings to a file.",
    [UI_MENU_FLIGHT] = "Flight recorder",
    [UI_MENU_FLIGHT_DESC] = "Keeps a short history and saves it when a frame freezes.",
    [UI_MENU_FLIGHT_SPIKE_MS] = "Freeze threshold (ms)",
    [UI_MENU_FLIGHT_SPIKE_MS_DESC] = "Frame time that counts as a freeze.",
    [UI_MENU_INPUT_RECORD] = "Input recording",
    [UI_MENU_INPUT_RECORD_DESC] = "Writes your inputs to a file for bug reports.",
    [UI_MENU_OPEN_LOG] = "Open log folder", [UI_MENU_OPEN_LOG_DESC] = "Shows the folder with the log files.",
    [UI_MENU_COPY_INFO] = "Copy system info",
    [UI_MENU_COPY_INFO_DESC] = "Copies GPU, driver, CPU, RAM and changed settings to the clipboard.",
    [UI_MENU_COPY_INFO_DONE] = "Copied", [UI_MENU_ACTION_FAILED] = "did not work",
    [UI_MENU_RESET] = "Reset all settings to defaults", [UI_MENU_RESET_DESC] = "Applies to every tab.",
    [UI_MENU_RESET_CONFIRM] = "Reset every setting to its default?", [UI_MENU_RESET_YES] = "Reset",
    [UI_MENU_RESET_NO] = "Cancel",

    [UI_MENU_OV_FPS] = "%.0f FPS", [UI_MENU_OV_FRAME_MS] = "%.1f ms", [UI_MENU_OV_GAME_PCT] = "game %.0f%%",
    [UI_MENU_OV_MAX_MS] = "max %.1f ms",
    [UI_MENU_OV_GPU] = "GPU", [UI_MENU_OV_CPU] = "CPU", [UI_MENU_OV_RAM] = "RAM", [UI_MENU_OV_VRAM] = "VRAM",
    [UI_MENU_OV_TEMP] = "Temperature", [UI_MENU_OV_POWER] = "Power", [UI_MENU_OV_CLOCK] = "Clock",
    [UI_MENU_OV_NONE] = "-",
    [UI_MENU_OV_REPLAY] = "GPU frame prep", [UI_MENU_OV_GPU_TIME] = "GPU busy per frame",
    [UI_MENU_OV_PIPE_WAIT] = "Pipeline waits",
    [UI_MENU_OV_SHADER_WAIT] = "Shader waits", [UI_MENU_OV_TEX_UPLOADS] = "Texture uploads",
    [UI_MENU_OV_FMT_PCT] = "%.0f%%", [UI_MENU_OV_FMT_TEMP] = "%.0f °C", [UI_MENU_OV_FMT_POWER] = "%.0f W",
    [UI_MENU_OV_FMT_MHZ] = "%.0f MHz", [UI_MENU_OV_FMT_GB] = "%.1f GB", [UI_MENU_OV_FMT_GB_OF] = "/ %.1f GB",
    [UI_MENU_OV_GAME_GB] = "game %.1f GB", [UI_MENU_OV_FMT_MS] = "%.2f ms", [UI_MENU_OV_PER_FRAME] = "%.1f per frame",
    [UI_MENU_OV_WINDOW] = "Window", [UI_MENU_OV_OUTPUT] = "Game image", [UI_MENU_OV_SCENE] = "3D scene",
    [UI_MENU_OV_SCALING] = "Scaling", [UI_MENU_OV_LIMIT] = "Frame limit", [UI_MENU_OV_PRESENT] = "Presentation",
    [UI_MENU_OV_TEX] = "Texture compression", [UI_MENU_OV_OS] = "OS",
    [UI_MENU_OV_FMT_SIZE] = "%ux%u", [UI_MENU_OV_FMT_SCENE] = "%ux%u, %.0f%%",
    [UI_MENU_OV_FMT_FSR] = "AMD FSR 1, sharpness %d%%", [UI_MENU_OV_SCALE_NONE] = "none (1:1)",
    [UI_MENU_OV_SCALE_BILINEAR] = "bilinear",
    [UI_MENU_OV_FMT_AREA] = "area average, sharpness %d%%",
    [UI_MENU_OV_FMT_PRESHARP] = "RCAS before the HUD, sharpness %d%%",
    [UI_MENU_TAB_CHEATS] = "Cheats",
    [UI_MENU_CHEATS_NONE] = "No cheats for this game version. A mod's cheats are an Atmosphere cheat file in its "
                            "cheats folder, named after the game's build: mods\\<mod>\\cheats\\%s. Restart the game to "
                            "read them. For cheats that change the game's code, the game rebuilds itself at start (a "
                            "few minutes).",
    [UI_MENU_CHEATS_MOD] = "%s: %d cheats",
    [UI_MENU_CHEAT_CODE_DESC] = "Changes the game's code. Switches at once.",
    [UI_MENU_CHEAT_KEY_DESC] = "Changes the game's code while you hold %s. Switches at once.",
    [UI_MENU_CHEAT_DATA_DESC] = "Writes the game's memory every frame while it is on.",
    [UI_MENU_CHEAT_MASTER] = "always on",
    [UI_MENU_CHEAT_MASTER_DESC] = "The file's master code: it is always on.",
    [UI_MENU_CHEAT_RESTORE] = "not needed",
    [UI_MENU_CHEAT_RESTORE_DESC] = "It only writes the game's own code back. Switching a cheat off already does that.",
    [UI_MENU_CHEAT_REBUILD] = "after the rebuild",
    [UI_MENU_CHEAT_REBUILD_DESC] = "This cheat changes the game's code and is new or edited since the game was built. "
                                   "The game builds it in when it starts next (a few minutes), or use Rebuild the game "
                                   "now.",
    [UI_MENU_CHEAT_UNSUPPORTED] = "not supported",
    [UI_MENU_CHEAT_UNSUPPORTED_DESC] = "This cheat cannot be used with this version of the game:",
    [UI_MENU_CHEATS_REBUILD] = "Rebuild the game now",
    [UI_MENU_CHEATS_REBUILD_DESC] = "Some cheats change the game's code and are not built into it: they are new or "
                                    "edited, or their last rebuild failed or was cancelled. Closes the game; the setup "
                                    "builds them in (a few minutes) and starts the game again. Progress since your "
                                    "last save is lost.",
    [UI_MENU_MODS_EXEFS_UNSUPPORTED] = "This mod patches the game code (exefs) in a way the port cannot apply, so the "
                                       "whole mod stays off. %s",
    [UI_MENU_MODS_EXEFS_OTHER] = "Its code patches (exefs) are made for another game version.",
    [UI_MENU_MODS_EXEFS_ON] = "Code patches in use: %d files.",
    [UI_MENU_MODS_REPLACES_BUILTIN] = "Code patches in use: %d files. They replace the port's own patch %s, which "
                                      "stays off while this mod is on.",
};

typedef struct { char lang[16], name[64], font[16]; const char **text; } Row;

enum { MAX_ROWS = 32, MAX_CATALOG = 1 << 20 };

static const char *const keys[UI_TEXT_COUNT] = {
#include "ui_text_keys.inc"
};

static Row rows[MAX_ROWS];
static int row_count;
static int initialized;
static uint16_t key_order[UI_TEXT_COUNT];   // the ids with a name, by name (strcmp), for id_for_mo_key
static int key_count;

static int key_cmp(const void *a, const void *b) {
    return strcmp(keys[*(const uint16_t *)a], keys[*(const uint16_t *)b]);
}

// The id a catalog key "<name>\x04<English>" stands for, or -1 for an unknown name or an English text this build does
// not have (a translation made from older English). A binary search over the names sorted once at start.
static int id_for_mo_key(const char *key) {
    const char *sep = strchr(key, '\x04');
    if (!sep) return -1;
    size_t n = (size_t)(sep - key);
    int lo = 0, hi = key_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2, id = key_order[mid];
        int c = strncmp(keys[id], key, n);
        if (!c && keys[id][n]) c = 1;   // the name is a prefix of keys[id], which sorts after it
        if (!c) return text_en[id] && !strcmp(sep + 1, text_en[id]) ? id : -1;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int valid_tag(const char *s) {
    if (!s || !*s || strlen(s) >= 16 || !isalnum((unsigned char)s[0])) return 0;
    for (const char *p = s; *p; p++)
        if (!isalnum((unsigned char)*p) && *p != '-' && *p != '_') return 0;
    return 1;
}

static int valid_utf8(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    for (; *p;) {
        if (*p < 0x80) p++;
        else if (*p >= 0xc2 && *p <= 0xdf && p[1] >= 0x80 && p[1] <= 0xbf) p += 2;
        else if (*p >= 0xe0 && *p <= 0xef && p[1] >= 0x80 && p[1] <= 0xbf && p[2] >= 0x80 && p[2] <= 0xbf &&
                 !(*p == 0xe0 && p[1] < 0xa0) && !(*p == 0xed && p[1] >= 0xa0)) p += 3;
        else if (*p >= 0xf0 && *p <= 0xf4 && p[1] >= 0x80 && p[1] <= 0xbf && p[2] >= 0x80 && p[2] <= 0xbf &&
                 p[3] >= 0x80 && p[3] <= 0xbf && !(*p == 0xf0 && p[1] < 0x90) && !(*p == 0xf4 && p[1] > 0x8f))
            p += 4;
        else return 0;
    }
    return 1;
}

static int utf16_units(const char *s) {
    int n = 0;
    const unsigned char *p = (const unsigned char *)s;
    for (; *p;) {
        unsigned char c = *p;
        if (c < 0x80) p++;
        else if (c < 0xe0) p += 2;
        else if (c < 0xf0) p += 3;
        else p += 4;
        n += c >= 0xf0 ? 2 : 1;
    }
    return n;
}

static int next_format(const char *s, const char **start, const char **end) {
    for (const char *p = s; *p; p++) {
        if (*p != '%') continue;
        if (p[1] == '%') {
            *start = p;
            *end = p + 2;
            return 1;
        }
        const char *q = p + 1;
        while (*q >= '0' && *q <= '9') q++;
        if (*q == '.') {
            q++;
            while (*q >= '0' && *q <= '9') q++;
        }
        if (*q && strchr("sdiufxX", *q)) {   // strchr also finds the terminating NUL
            *start = p;
            *end = q + 1;
            return 1;
        }
    }
    return 0;
}

static int formats_match(const char *a, const char *b) {
    for (;;) {
        const char *as, *ae, *bs, *be;
        int ha = next_format(a, &as, &ae), hb = next_format(b, &bs, &be);
        if (!ha || !hb) return !ha && !hb;
        if (ae - as != be - bs || memcmp(as, bs, (size_t)(ae - as))) return 0;
        a = ae;
        b = be;
    }
}

static int text_ok(UiTextId id, const char *value) {
    if (!value || !*value || !valid_utf8(value) || (unsigned)id >= UI_TEXT_COUNT) return 0;
    if (!formats_match(text_en[id], value)) return 0;
    if (id == UI_SWKBD_DEFAULT_NAME) {
        if (utf16_units(value) > 9 || strpbrk(value, "@%\\")) return 0;
    } else if (id == UI_SWKBD_TITLE && utf16_units(value) > 128) return 0;
    else if (id == UI_SWKBD_PROMPT && utf16_units(value) > 258) return 0;
    else if ((id == UI_SWKBD_OK || id == UI_SWKBD_CANCEL) && utf16_units(value) > 64) return 0;
    else if (id == UI_SWKBD_PAD_HINT && utf16_units(value) > 160) return 0;
    if (id >= UI_MENU_FIRST && id <= UI_MENU_LAST_ID && strlen(value) + 256 >= UI_DESC_BYTES) return 0;
    return 1;
}

enum { MO_MAGIC = 0x950412DE };

static uint32_t mo_u32(const uint8_t *p, int little) {
    if (little)
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int mo_string(const uint8_t *data, size_t size, uint32_t len, uint32_t off, const char **out) {
    size_t start = off, n = len;
    if (start > size || n > size - start || start + n >= size) return 0;
    if (memchr(data + start, 0, n) || data[start + n]) return 0;
    *out = (const char *)(data + start);
    return 1;
}

static int header_value(const char *header, const char *name, char *out, size_t cap) {
    size_t n = strlen(name);
    for (const char *p = header; *p;) {
        const char *e = strchr(p, '\n');
        if (!e) e = p + strlen(p);
        if ((size_t)(e - p) > n && !strncmp(p, name, n) && p[n] == ':') {
            const char *v = p + n + 1;
            while (v < e && isspace((unsigned char)*v)) v++;
            const char *end = e;
            while (end > v && isspace((unsigned char)end[-1])) end--;
            if ((size_t)(end - v) >= cap) return 0;
            memcpy(out, v, (size_t)(end - v));
            out[end - v] = 0;
            return 1;
        }
        if (!*e) break;
        p = e + 1;
    }
    out[0] = 0;
    return 0;
}

static int has_utf8(const char *s) {
    for (; *s; s++)
        if (tolower((unsigned char)*s) == 'u' && tolower((unsigned char)s[1]) == 't' &&
            tolower((unsigned char)s[2]) == 'f' && (s[3] == '-' || s[3] == '8'))
            return 1;
    return 0;
}

static void free_row_text(Row *row) {
    if (!row->text || row->text == (const char **)text_en) return;
    for (int i = 0; i < (int)UI_TEXT_COUNT; i++) free((void *)row->text[i]);
    free((void *)row->text);
    row->text = NULL;
}

static int parse_catalog(const char *path, const char *stem, Row *row) {
    uint64_t file_size = 0;
    uint8_t *data = plat_file_read(path, &file_size);
    if (!data || file_size < 28 || file_size > MAX_CATALOG) {
        free(data);
        return 0;
    }
    size_t size = (size_t)file_size;
    int little = !memcmp(data, "\xde\x12\x04\x95", 4);
    if (!little && memcmp(data, "\x95\x04\x12\xde", 4)) goto fail;
    if (mo_u32(data + 4, little) != 0) goto fail;
    uint32_t count = mo_u32(data + 8, little);
    uint32_t orig = mo_u32(data + 12, little), trans = mo_u32(data + 16, little);
    uint32_t hash_count = mo_u32(data + 20, little), hash_off = mo_u32(data + 24, little);
    if (!count || count > UI_TEXT_COUNT + 1) goto fail;
    if ((uint64_t)orig + 8ull * count > size || (uint64_t)trans + 8ull * count > size) goto fail;
    if (hash_count && (uint64_t)hash_off + 4ull * hash_count > size) goto fail;

    const char *header = NULL;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t key_len = mo_u32(data + orig + 8ull * i, little);
        uint32_t key_off = mo_u32(data + orig + 8ull * i + 4, little);
        uint32_t text_len = mo_u32(data + trans + 8ull * i, little);
        uint32_t text_off = mo_u32(data + trans + 8ull * i + 4, little);
        const char *key, *text;
        if (!mo_string(data, size, key_len, key_off, &key) ||
            !mo_string(data, size, text_len, text_off, &text))
            goto fail;
        if (!i) {
            if (*key) goto fail;
            header = text;
            continue;
        }
        if (!*key) goto fail;
        int id = id_for_mo_key(key);
        if (id < 0) continue;
        if (row->text[id]) goto fail;
        if (!text_ok((UiTextId)id, text)) continue;
        size_t len = strlen(text);
        char *copy = malloc(len + 1);
        if (!copy) goto fail;
        memcpy(copy, text, len + 1);
        row->text[id] = copy;
    }
    if (!header || !*header) goto fail;

    char lang[16], name[64], charset[256], font[16];
    int ok = header_value(header, "Language", lang, sizeof lang) &&
             header_value(header, "X-Astral-Language-Name", name, sizeof name) && *name &&
             header_value(header, "Content-Type", charset, sizeof charset) && has_utf8(charset) &&
             !strcmp(lang, stem);
    font[0] = 0;
    if (ok && header_value(header, "X-Astral-Font-Lang", font, sizeof font) && !valid_tag(font)) ok = 0;
    if (ok) {
        snprintf(row->lang, sizeof row->lang, "%s", lang);
        snprintf(row->name, sizeof row->name, "%s", name);
        if (*font) snprintf(row->font, sizeof row->font, "%s", font);
    }
    free(data);
    if (!ok) {
        free_row_text(row);
        return 0;
    }
    return 1;
fail:
    free(data);
    free_row_text(row);
    return 0;
}

int ui_text_catalog_check(const char *path, const char *lang) {
    Row row = {0};
    row.text = calloc(UI_TEXT_COUNT, sizeof *row.text);
    if (!row.text) return -1;
    if (!parse_catalog(path, lang, &row)) return -1;   // it freed the texts
    int n = 0;
    for (int i = 0; i < (int)UI_TEXT_COUNT; i++) n += row.text[i] != NULL;
    free_row_text(&row);
    return n;
}

typedef struct { char dir[PLAT_PATH_MAX * 2]; } LoadCtx;

static int load_catalog(const char *name, void *ctx) {
    size_t n = strlen(name);
    if (n < 3 || strcmp(name + n - 3, ".mo") || n - 3 >= 16) return 1;
    char stem[16];
    memcpy(stem, name, n - 3);
    stem[n - 3] = 0;
    if (!valid_tag(stem)) return 1;
    for (int i = 0; i < row_count; i++)
        if (!strcmp(rows[i].lang, stem)) return 1;
    if (row_count >= MAX_ROWS) return 0;

    LoadCtx *load = ctx;
    char path[PLAT_PATH_MAX * 2];
    snprintf(path, sizeof path, "%s/%s", load->dir, name);
    Row row = {0};
    row.text = calloc(UI_TEXT_COUNT, sizeof *row.text);
    if (!row.text) return 0;
    if (!parse_catalog(path, stem, &row)) return 1;
    rows[row_count++] = row;
    return 1;
}

static const char *catalog_dir(void) {
    static char dir[PLAT_PATH_MAX * 2];
    const char *set = cfg_get("AC_LANG_DIR");
    if (set && *set) return set;
    char exe[PLAT_PATH_MAX * 2];
    int n = plat_exe_path(exe, sizeof exe);
    char *sep = n > 0 && n < sizeof exe ? plat_last_sep(exe) : NULL;
    if (sep) {
        *sep = 0;
        snprintf(dir, sizeof dir, "%s/lang", exe);
        if (plat_dir_exists(dir)) return dir;
    }
    snprintf(dir, sizeof dir, "lang");
    return dir;
}

static int row_compare(const void *a, const void *b) {
    const Row *ra = a, *rb = b;
    if (!strcmp(ra->lang, "en")) return -1;
    if (!strcmp(rb->lang, "en")) return 1;
    return strcmp(ra->lang, rb->lang);
}

// Once, from ui_text_init: readers take no lock, so the rows must not change once other threads run.
static void load_catalogs(void) {
    uint64_t t0 = plat_now_ns();
    key_count = 0;
    for (int i = 0; i < (int)UI_TEXT_COUNT; i++)
        if (keys[i]) key_order[key_count++] = (uint16_t)i;
    qsort(key_order, (size_t)key_count, sizeof *key_order, key_cmp);
    for (int i = 1; i < row_count; i++) free_row_text(&rows[i]);
    row_count = 1;
    rows[0] = (Row){.lang = "en", .name = "English", .text = (const char **)text_en};
    LoadCtx ctx = {0};
    const char *dir = catalog_dir();
    snprintf(ctx.dir, sizeof ctx.dir, "%s", dir);
    if (plat_dir_exists(dir)) plat_file_list(dir, ".mo", load_catalog, &ctx);
    qsort(rows + 1, (size_t)(row_count - 1), sizeof *rows, row_compare);
    int menus = 0;
    for (int i = 1; i < row_count; i++) menus += rows[i].text[UI_MENU_TITLE] != NULL;
    rt_log("ui text: %d catalogs from %s (%d with menu text, %d keyboard only) in %.2f ms", row_count - 1, dir, menus,
           row_count - 1 - menus, (double)(plat_now_ns() - t0) / 1e6);
}

void ui_text_init(void) {
    if (initialized) return;
    initialized = 1;
    load_catalogs();
}

static const char *catalog_tag(const char *code) {
    static const struct { const char *code, *lang; } map[] = {
        {"en-US", "en"}, {"en-GB", "en"}, {"fr-CA", "fr"}, {"es-419", "es"},
        {"zh-CN", "zh-Hans"}, {"zh-TW", "zh-Hant"},
    };
    if (code)
        for (size_t i = 0; i < sizeof map / sizeof *map; i++)
            if (!strcmp(code, map[i].code)) return map[i].lang;
    return code;
}

static const Row *find_row(const char *lang) {
    lang = catalog_tag(lang);
    if (!lang) return NULL;
    for (int i = 0; i < row_count; i++)
        if (!strcmp(rows[i].lang, lang)) return &rows[i];
    return NULL;
}

const char *ui_text_lang(const char *code) {
    if (!code || !*code) return rows[0].lang;
    code = catalog_tag(code);
    const Row *r = find_row(code);
    if (r) return r->lang;
    char base[16];   // per call: the keyboard asks from guest threads while the menu asks from the render thread
    size_t n = strcspn(code, "-");
    if (n < sizeof base && code[n]) {
        memcpy(base, code, n), base[n] = 0;
        r = find_row(base);
        if (r) return r->lang;
    }
    return code;
}

static const char *fallback_text(UiTextId id) {
    if ((unsigned)id >= UI_TEXT_COUNT) return "";
    const char *text = rows[0].text && rows[0].text[id] ? rows[0].text[id] : text_en[id];
    return text && *text ? text : "";
}

int ui_text_defined(const char *lang, UiTextId id) {
    const Row *r = find_row(lang);
    return (unsigned)id < UI_TEXT_COUNT && r && r->text[id] && *r->text[id];
}

const char *ui_text_in(const char *lang, UiTextId id) {
    if ((unsigned)id >= UI_TEXT_COUNT) return "";
    const Row *r = find_row(lang);
    if (r && r->text[id] && *r->text[id]) return r->text[id];
    return fallback_text(id);
}

const char *ui_text(UiTextId id) { return ui_text_in(ui_text_lang(ov_language()), id); }

void ui_text_visit(const char *lang, void (*visit)(void *ctx, const char *text), void *ctx) {
    const Row *r = find_row(lang);
    if (!r) return;
    for (int id = 0; id < (int)UI_TEXT_COUNT; id++)
        if (r->text[id] && *r->text[id]) visit(ctx, r->text[id]);
}

int ui_text_lang_count(void) { return row_count; }

int ui_text_is_menu_lang(const char *lang) {
    const Row *r = find_row(lang);
    return r && (r == &rows[0] || r->text[UI_MENU_TITLE]);
}

const char *ui_text_lang_at(int i) { return i >= 0 && i < row_count ? rows[i].lang : NULL; }

const char *ui_text_lang_name(const char *lang) {
    const Row *r = find_row(ui_text_lang(lang));
    return r ? r->name : lang;
}

const char *ui_text_font_lang(const char *lang) {
    const Row *r = find_row(ui_text_lang(lang));
    return r ? r->font : "";
}

static const char *resolve_menu_lang(void) {
    const char *v = cfg_value("AC_MENU_LANG");
    const char *sys = plat_ui_language();
    const char *lang = ui_text_lang(v && *v ? v : sys && *sys ? sys : "en");
    if (ui_text_defined(lang, UI_MENU_TITLE)) return lang;
    char base[16];
    size_t n = strcspn(lang, "-");
    if (n < sizeof base && lang[n]) {
        memcpy(base, lang, n), base[n] = 0;
        if (ui_text_defined(base, UI_MENU_TITLE)) return find_row(base)->lang;
    }
    // Portugal's Portuguese keeps its own keyboard catalog (pt) and reads the Brazilian menu rather than English
    if (!strncmp(lang, "pt", 2) && ui_text_defined("pt-BR", UI_MENU_TITLE)) return find_row("pt-BR")->lang;
    return rows[0].lang;
}

const char *menu_text(UiTextId id) {
    static _Atomic(const char *) lang = "en";
    static _Atomic unsigned seen = ~0u;
    unsigned v = cfg_version();
    if (v != atomic_load_explicit(&seen, memory_order_relaxed)) {
        atomic_store_explicit(&lang, resolve_menu_lang(), memory_order_relaxed);
        atomic_store_explicit(&seen, v, memory_order_relaxed);
    }
    return ui_text_in(atomic_load_explicit(&lang, memory_order_relaxed), id);
}
