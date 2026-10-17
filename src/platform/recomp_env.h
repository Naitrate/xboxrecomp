/*
 * recomp_env.h - every environment variable the runtime reads, in one table.
 *
 * Three tiers:
 *
 *   config  A small set of documented, user-facing knobs, each its own
 *           variable: RECOMP_PB_BACKEND=d3d11, RECOMP_SAVE_DIR=..., ...
 *   trace   Log and report toggles, one comma list:
 *               RECOMP_TRACE=flip,tex=2,heap=0x80123000
 *   debug   Debug hacks, A/B switches, experiments, dumps and watchpoints,
 *           one comma list:
 *               RECOMP_DEBUG=pb_fast=0,fb_dump=/tmp/f_,fb_dump_at=61,121
 *
 * A list entry is `key` (value "1") or `key=value`. A later entry for the
 * same key wins. Values may contain commas: a fragment that does not start
 * with a known key continues the value before it, so `fb_dump_at=61,121`
 * and `peek_chain=0x1315A8,8,0x10,0` read as one value each.
 * RECOMP_TRACE=help or RECOMP_DEBUG=help prints the table.
 *
 * The environment is read once (recomp_env_init, or lazily on the first
 * lookup) and cached, so a lookup is an array load: safe in hot paths.
 * Values keep getenv's meaning: NULL when unset, otherwise the string.
 *
 * Transition: every variable these replace still works through an alias,
 * which prints one deprecation line naming the new spelling. See
 * cat/docs/env.md for the table with meanings.
 */
#ifndef RECOMP_ENV_H
#define RECOMP_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

/* A game adds its own keys, same row format, by defining
 * RECOMP_ENV_HAVE_GAME_KEYS on the recomp_env target (PUBLIC, so every user
 * agrees on the table) and putting a recomp_env_game.h on its include path
 * that defines RECOMP_ENV_GAME_KEYS(X). Its rows come after the toolkit's, so
 * the toolkit's ids are the same with or without them. A game key whose value
 * may contain commas is named in RECOMP_ENV_GAME_COMMA_KEYS, a string list:
 *   #define RECOMP_ENV_GAME_COMMA_KEYS "mem_dump",
 * A game can also give a toolkit key a default, used when neither its new
 * nor its old spelling is set, with RECOMP_ENV_GAME_DEFAULTS(D):
 *   #define RECOMP_ENV_GAME_DEFAULTS(D) D(APU_DSP_ACK, "auto")
 */
#ifdef RECOMP_ENV_HAVE_GAME_KEYS
#include "recomp_env_game.h"
#endif
#ifndef RECOMP_ENV_GAME_KEYS
#define RECOMP_ENV_GAME_KEYS(X)
#endif

/* The opt-in enhancements layer (src/enhance/, XBOXRECOMP_ENHANCE=ON) adds
 * its config-tier keys by defining RECOMP_ENV_HAVE_ENHANCE_KEYS on this
 * target. They mirror keys of the enhancements file (enhance.toml) and win
 * over it: the variable is RECOMP_ + the dotted key, upper case, '.' -> '_'.
 * Without the layer they do not exist, so the table is what it always was. */
#ifdef RECOMP_ENV_HAVE_ENHANCE_KEYS
#define RECOMP_ENV_ENHANCE_KEYS(X) \
    X(ENHANCE_CONFIG,    CONFIG, "RECOMP_ENHANCE_CONFIG",    NULL, "enhancements file: a path, or none (default: <root>/<title>/enhance.toml, <root>/enhance.toml)") \
    X(RENDER_SCALE,      CONFIG, "RECOMP_RENDER_SCALE",      NULL, "enhancements: render.scale, internal resolution factor (1 = stock)") \
    X(DISPLAY_ASPECT,    CONFIG, "RECOMP_DISPLAY_ASPECT",    NULL, "enhancements: display.aspect, 4:3 (stock), 16:9, 16:10, 21:9") \
    X(DISPLAY_WIDESCREEN, CONFIG, "RECOMP_DISPLAY_WIDESCREEN", NULL, "enhancements: display.widescreen, the dashboard's 16:9 setting the title reads (1/0)") \
    X(DISPLAY_HDTV,      CONFIG, "RECOMP_DISPLAY_HDTV",      NULL, "enhancements: display.hdtv, the dashboard's HDTV modes: off (stock), 480p, 720p, 1080i (each with those below)") \
    X(PRESENT_FILTER,    CONFIG, "RECOMP_PRESENT_FILTER",    NULL, "enhancements: present.filter, nearest (stock), linear, integer") \
    X(PRESENT_ASPECT,    CONFIG, "RECOMP_PRESENT_ASPECT",    NULL, "enhancements: present.aspect, the window's frame shape: auto (stock; 16:9 with display.widescreen), frame, 4:3, 16:9") \
    X(PRESENT_FULLSCREEN, CONFIG, "RECOMP_PRESENT_FULLSCREEN", NULL, "enhancements: present.fullscreen, borderless desktop fullscreen at start (1/0)") \
    X(PRESENT_PACING,    CONFIG, "RECOMP_PRESENT_PACING",    NULL, "enhancements: present.pacing, spin (stock busy-wait) or sleep (lowered spin waits sleep)") \
    X(PRESENT_SPIN_LEAD, CONFIG, "RECOMP_PRESENT_SPIN_LEAD", NULL, "enhancements: present.spin_lead, ms before the vblank a held flip spins instead of sleeping (pacing sleep; keeps the CPU clock up)") \
    X(RENDER_SHADER_CACHE, CONFIG, "RECOMP_RENDER_SHADER_CACHE", NULL, "enhancements: render.shader_cache, the pipeline-cache directory, or none (default: the per-user cache directory)")
#else
#define RECOMP_ENV_ENHANCE_KEYS(X)
#endif

/* X(ID, TIER, NAME, LEGACY, HELP)
 *   CONFIG: NAME is the variable; LEGACY an older variable, or NULL.
 *   TRACE / DEBUG: NAME is the key in RECOMP_TRACE / RECOMP_DEBUG; LEGACY
 *   the variable it replaces (an alias for the transition window). */
#define RECOMP_ENV_KEYS(X) \
    /* ── config ─────────────────────────────────────────────────────── */ \
    X(PB_EXEC,           CONFIG, "RECOMP_PB_EXEC",           NULL, "pushbuffer executor (draws frames); default on, 0: off, open: starts open, toggle=<s>,...: opens and closes at those seconds, file=<path>: whenever that file appears, aa_cycle=<s>: steps the AA setting every s seconds (scripted runs)") \
    X(PB_BACKEND,        CONFIG, "RECOMP_PB_BACKEND",        NULL, "executor backend: cpu (default), metal, vulkan, d3d11, null") \
    X(EXEC_UCLAMP,       CONFIG, "RECOMP_EXEC_UCLAMP",       NULL, "Linux: the executor thread's minimum utilisation clamp, 0..1024 (CPU clock boost without spinning)") \
    X(PRESENT_OVERSCAN,  CONFIG, "RECOMP_PRESENT_OVERSCAN",  NULL, "crop: show the 640x448 area a TV showed (y 16..464), cropping the sides alike so the aspect holds (Vulkan, Metal windows)") \
    X(PRESENT_FIT,       CONFIG, "RECOMP_PRESENT_FIT",       NULL, "fill16x10: fill the window's height, cropping the frame's sides (the Deck's 16:10 panel) (Vulkan, Metal windows)") \
    X(PRESENT_EARLY,     CONFIG, "RECOMP_PRESENT_EARLY",     NULL, "1: frames go to the host at FLIP_STALL, before the vblank hold (lower latency, same pacing)") \
    X(FLIP_PACING,       CONFIG, "RECOMP_FLIP_PACING",       NULL, "vblank: FLIP_STALL holds until the vblank the title's swap asked for (needs RECOMP_VBLANK); default off") \
    X(HEADLESS,          CONFIG, "RECOMP_HEADLESS",          NULL, "1: no window (SDL host)") \
    X(WINDOW_SCALE,      CONFIG, "RECOMP_WINDOW_SCALE",      NULL, "window scale factor (SDL host, default 2)") \
    X(PRESENT_VSYNC,     CONFIG, "RECOMP_PRESENT_VSYNC",     NULL, "vsync: SDL host 0 = off (default on); D3D11 1 = on (default off)") \
    X(WINDOW_QUIT_AFTER, CONFIG, "RECOMP_WINDOW_QUIT_AFTER", NULL, "close the window after this many seconds") \
    X(FB_WINDOW,         CONFIG, "RECOMP_FB_WINDOW",         NULL, "Windows: show the framebuffer window") \
    X(SAVE_DIR,          CONFIG, "RECOMP_SAVE_DIR",          NULL, "root for the title's UDATA/TDATA (a host using recomp_save_dir: the whole save root, instead of the per-user directory)") \
    X(SAVE_BACKUPS,      CONFIG, "RECOMP_SAVE_BACKUPS",      NULL, "save backups kept in <save dir>/backup, one per start that found the saves changed (recomp_save_backup; default 5, 0 none)") \
    X(HANG_WATCHDOG,     CONFIG, "RECOMP_HANG_WATCHDOG",     NULL, "seconds without a flip before a hang report in <save dir>/crash (default 15, 0 off; 60 before the first flip)") \
    X(MMIO_DIRECT,       CONFIG, "RECOMP_MMIO_DIRECT",       NULL, "constant-address device registers by call, not fault; default on, 0: every access traps") \
    X(AC97_READY,        CONFIG, "RECOMP_AC97_READY",        NULL, "emulated APU and audio; default on (x86-64 Windows/Proton, macOS/Linux arm64), 0: off") \
    X(VBLANK,            CONFIG, "RECOMP_VBLANK",            NULL, "deliver the vblank interrupt") \
    X(VBLANK_SYNC,       CONFIG, "RECOMP_VBLANK_SYNC",       NULL, "=display: the emulated vblank follows the display's refresh (reported by the present path) when within 5% of 60 Hz; default: its own 60.000 Hz timer") \
    X(PRESENT_DRAWABLES, CONFIG, "RECOMP_PRESENT_DRAWABLES", NULL, "Metal: CAMetalLayer maximumDrawableCount, 2 or 3 (default 2)") \
    X(PRESENT_TIMING,    CONFIG, "RECOMP_PRESENT_TIMING",    NULL, "display.present_timing: legacy (present when ready; default) or auto (each frame aimed at one refresh; the vblank locked to a fixed-rate display, timed presents under variable refresh); lock, timed force one (Metal)") \
    X(FRAME_PACING,      CONFIG, "RECOMP_FRAME_PACING",      NULL, "display.frame_pacing: auto (smooth under gamescope, low_latency elsewhere; default), low_latency (each frame at the first refresh it can make) or smooth (one finished frame kept queued: +1 refresh of latency, a late frame no longer shows as a repeated one). Linux and Windows Vulkan with FIFO and present_wait (or present_wait2); on Windows the keeper stands down under variable refresh and away from 60 Hz; elsewhere no effect yet (BO-110, BO-207)") \
    X(PRESENT_MODE,      CONFIG, "RECOMP_PRESENT_MODE",      NULL, "display.present_mode, Vulkan with vsync: auto or fifo (FIFO; default) or mailbox (MAILBOX where offered; never picked by itself, BO-110)") \
    X(PRESENT_PHASE,     CONFIG, "RECOMP_PRESENT_PHASE",     NULL, "present timing, a locked display: ms the emulated vblank sits before a display refresh (default 5)") \
    X(PRESENT_LEAD,      CONFIG, "RECOMP_PRESENT_LEAD",      NULL, "present timing, a variable-refresh display: ms from the emulated vblank to the frame's present time (default 12)") \
    X(PRESENT_DEPTH,     CONFIG, "RECOMP_PRESENT_DEPTH",     NULL, "present timing, a locked display: refreshes from the vblank's refresh to the frame's, 0-3 (default 1; raised by itself when frames slip)") \
    X(PRESENT_LATENCY,   CONFIG, "RECOMP_PRESENT_LATENCY",   NULL, "D3D11 RECOMP_PRESENT_VSYNC=fifo: SetMaximumFrameLatency, 1-3 (default 1)") \
    X(KEYBOARD,          CONFIG, "RECOMP_KEYBOARD",          NULL, "1: keyboard drives pad 1") \
    X(INPUT_RUMBLE,      CONFIG, "RECOMP_INPUT_RUMBLE",      NULL, "forward the title's rumble (USB output report) to the host pad; default 1, 0: off") \
    X(INPUT_CONTROL_SCHEME, CONFIG, "RECOMP_INPUT_CONTROL_SCHEME", NULL, "the driving controls, for a title that offers a second layout: xbox (default, the title's own), ps2 (the PlayStation 2 version's layout while driving; menus are not changed)") \
    X(INPUT_FACE_LAYOUT, CONFIG, "RECOMP_INPUT_FACE_LAYOUT", NULL, "the host pad's face buttons: xbox (default), nintendo (A and B, X and Y change places: a pad printed the Nintendo way)") \
    X(INPUT_GLYPHS,      CONFIG, "RECOMP_INPUT_GLYPHS",      NULL, "the button prompts the title draws: auto (default: those of the pad last used; a diamond-of-dots positional set for a pad nothing is known about, the title's own with no pad), xbox (the title's own, nothing replaced), xbox_modern (LB RB LT RT), playstation, playstation5 (DualSense: uncoloured symbols), nintendo, steam (the Steam Deck and Valve's controllers), positional (any pad: a diamond of four dots, the one to press filled), keyboard (key caps; auto picks it while the keyboard was the last device used)") \
    X(INPUT_RUMBLE_GAIN, CONFIG, "RECOMP_INPUT_RUMBLE_GAIN", NULL, "rumble strength, 0..3: host motor = (speed / peak) ^ (1 / strength), every backend alike; 1.0 (default): the title's speeds as the console played them, 0: off, above 1: the low speeds lifted (full stays full). Under speed 800 a motor is off, and on again from 1200; a falling level lets go over about 90 ms") \
    X(INPUT_RUMBLE_PEAK, CONFIG, "RECOMP_INPUT_RUMBLE_PEAK", NULL, "the title's own full motor speed, 1..65535 (default 65535): the speed that is full on the host motor, for a title whose strongest rumble stays under 65535") \
    X(PAD_DEADZONE,      CONFIG, "RECOMP_PAD_DEADZONE",      NULL, "stick deadzone, 0..32767 (default 0)") \
    X(PAD_TRIGGER_DEADZONE, CONFIG, "RECOMP_PAD_TRIGGER_DEADZONE", NULL, "trigger deadzone, 0..255 (default 0)") \
    X(AUDIO_VOLUME,      CONFIG, "RECOMP_AUDIO_VOLUME",      NULL, "master volume of the title's audio, 0..1 (default 1)") \
    X(AUDIO_MUTE,        CONFIG, "RECOMP_AUDIO_MUTE",        NULL, "1: no sound") \
    X(AUDIO_MUTE_BG,     CONFIG, "RECOMP_AUDIO_MUTE_BG",     NULL, "1: no sound while the window is in the background") \
    X(AUDIO_SPEAKERS,    CONFIG, "RECOMP_AUDIO_SPEAKERS",    NULL, "speaker mode: auto (default: 5.1 when the default output has 6+ channels), stereo, 5.1") \
    X(ANDROID_VIBRATION, CONFIG, "RECOMP_ANDROID_VIBRATION", NULL, "Android: where the first pad's rumble goes: auto (the pad when SDL can rumble it, else the device's vibrator when the pad is the device's built-in one; default), controller, device (the vibrator whatever the pad), off. SDL2 cannot rumble a pad on Android: an external pad does not rumble") \
    X(TOUCH_CONTROLS,    CONFIG, "RECOMP_TOUCH_CONTROLS",    NULL, "Android: the on-screen controls (src/input/touch_pad.h): auto (default: hidden while a controller is used, shown again at a touch), always, off") \
    X(TOUCH_OPACITY,     CONFIG, "RECOMP_TOUCH_OPACITY",     NULL, "Android: the on-screen controls' opacity, 0.1..1 (default 0.5)") \
    X(TOUCH_SCALE,       CONFIG, "RECOMP_TOUCH_SCALE",       NULL, "Android: the on-screen controls' size, 0.6..1.6 (default 1)") \
    X(TOUCH_AUTO_GAS,    CONFIG, "RECOMP_TOUCH_AUTO_GAS",    NULL, "Android: 1: the on-screen controls hold the accelerator while driving, BRAKE lets it go (default 0)") \
    X(SPEED_UNITS,       CONFIG, "RECOMP_SPEED_UNITS",       NULL, "speed and distance units, for a title whose build has the metric display behind a patch (--patches, recomp_patch.h): mph (default, as shipped) or kmh") \
    X(GFX_HUD_FIX,       CONFIG, "RECOMP_GFX_HUD_FIX",       NULL, "1: the title's 2D HUD keeps its shape in a 16:9 picture, for a title whose build has that behind a patch (--patches, recomp_patch.h); default 0, as shipped") \
    X(DISPLAY_ULTRAWIDE, CONFIG, "RECOMP_DISPLAY_ULTRAWIDE", NULL, "a picture wider than 16:9 for the title's 3D, for a title whose build has that behind a patch (--patches, recomp_patch.h): off (default, as shipped), 21:9, 32:9, window (the window's own shape)") \
    X(GFX_BLOOM,         CONFIG, "RECOMP_GFX_BLOOM",         NULL, "0 to 1: the strength of the title's own bloom (glow); 0 skips its passes, for a title whose build has that behind a patch (--patches, recomp_patch.h); default 1, as shipped") \
    X(GFX_MOTION_BLUR,   CONFIG, "RECOMP_GFX_MOTION_BLUR",   NULL, "0 to 1: the strength of the title's own motion blur; 0 is not drawn, for a title whose build has that behind a patch (--patches, recomp_patch.h); default 1, as shipped") \
    X(GFX_CRASH_60FPS,   CONFIG, "RECOMP_GFX_CRASH_60FPS",   NULL, "1: the title's 30 Hz phases (crash and takedown cameras) run at 60, for a title whose build has that behind a patch (--patches, recomp_patch.h); default 0, as shipped") \
    X(GFX_SPLIT_60FPS,   CONFIG, "RECOMP_GFX_SPLIT_60FPS",   NULL, "1: the title's 30 Hz split screen runs at 60, for a title whose build has that behind a patch (--patches, recomp_patch.h); default 0, as shipped") \
    X(AUDIO_SPLIT_MUSIC, CONFIG, "RECOMP_AUDIO_SPLIT_MUSIC", NULL, "1: the title's music plays in split screen too, for a title whose build has that behind a patch (--patches, recomp_patch.h); default 0, as shipped (no music there)") \
    X(GFX_VEHICLE_DETAIL, CONFIG, "RECOMP_GFX_VEHICLE_DETAIL", NULL, "how far from the camera the title's vehicles keep their detailed models, for a title whose build has its level-of-detail distances behind a patch (--patches, recomp_patch.h): original (default, as shipped), far, farthest") \
    X(GFX_SCENERY_DETAIL, CONFIG, "RECOMP_GFX_SCENERY_DETAIL", NULL, "how far from the camera the title's scenery objects keep their detailed models, for a title whose build has that distance behind a patch (--patches, recomp_patch.h): original (default, as shipped), far") \
    X(GAME_RACE_CLOCK,   CONFIG, "RECOMP_GAME_RACE_CLOCK",   NULL, "1: a race keeps real time from its start on a host that cannot hold 60 (the title's own catch-up stepping, which it otherwise starts at the first crash camera), for a title whose build has that behind a patch (--patches, recomp_patch.h); default 0, as shipped") \
    X(GFX_SCORE_BOARD,   CONFIG, "RECOMP_GFX_SCORE_BOARD",   NULL, "the title's score board in a race's HUD, for a title whose build has a hook for them: original (default, as shipped), smooth (the board fades as one)") \
    X(OVERLAY,           CONFIG, "RECOMP_OVERLAY",           NULL, "the in-game settings overlay (Back+Start, F1, Cmd+,); default on, 0: off") \
    X(OVERLAY_PERF,      CONFIG, "RECOMP_OVERLAY_PERF",      NULL, "1: the overlay's FPS and frame-time graph is shown (default off)") \
    X(OVERLAY_PERF_STEPS, CONFIG, "RECOMP_OVERLAY_PERF_STEPS", NULL, "1: the frame-time graph also marks frames in which the title ran more than one simulation step (default off)") \
    X(OVERLAY_TEST,      CONFIG, "RECOMP_OVERLAY_TEST",      NULL, "=file: a test drives the overlay through it, one line each as appended: key <up|down|left|right|enter|esc|f1|pgup|pgdn|del>, pad <a|b|y|up|down|left|right|lb|rb|back+start> (SDL events pushed into the window's loop), shot <path> (the window as presented, BMP), burst <n> <prefix> (the next n new frames as <prefix><flip>.bmp), state <path> (the open page's rows as text)") \
    X(SETTINGS_FILE,     CONFIG, "RECOMP_SETTINGS",          NULL, "the overlay's settings file: a path, or none (default: <save dir>/settings.toml)") \
    X(GFX_ANISO,         CONFIG, "RECOMP_GFX_ANISO",         NULL, "anisotropic filtering of mipmapped, linearly minified textures: off (default), 2, 4, 8, 16 (Metal, Vulkan; clamped to the device's limit)") \
    X(GFX_AA,            CONFIG, "RECOMP_GFX_AA",            NULL, "post-process anti-aliasing of the frame at the game's resolution, before the scale to the window: off (default), fxaa, smaa (Metal, Vulkan windows)") \
    X(GFX_HDR,           CONFIG, "RECOMP_GFX_HDR",           NULL, "1: HDR output (BO-110; macOS EDR on Metal, HDR10 on Vulkan where the surface offers it: gamescope with HDR on): the scene keeps light past paper white, the HUD and menus at their own white (default 0, off: the SDR path)") \
    X(GFX_HDR_PEAK,      CONFIG, "RECOMP_GFX_HDR_PEAK",      NULL, "HDR: the display's peak brightness in nits; 0 (default) what the display reports (macOS: its EDR headroom, 100 nits = the system's SDR white)") \
    X(GFX_HDR_PAPER,     CONFIG, "RECOMP_GFX_HDR_PAPER",     NULL, "HDR: the scene's paper white in nits (default 200; macOS 100, the system's SDR white)") \
    X(GFX_HDR_HUD,       CONFIG, "RECOMP_GFX_HDR_HUD",       NULL, "HDR: the HUD's and menus' white in nits; 0 (default) the paper white") \
    X(GFX_HDR_SHADOWS,   CONFIG, "RECOMP_GFX_HDR_SHADOWS",   NULL, "HDR (Vulkan): the shadows' depth, a power on luminance below mid grey (0.18), 1.0 (default) as the art; 1.1-1.2 deeper darks") \
    X(GFX_HDR_HIGHLIGHTS, CONFIG, "RECOMP_GFX_HDR_HIGHLIGHTS", NULL, "HDR (Vulkan): the gain on the scene's light past paper white before the roll-off to the peak, 1.0 (default) as the title stacks it") \
    X(GFX_HDR_SHOULDER,  CONFIG, "RECOMP_GFX_HDR_SHOULDER",  NULL, "HDR (Vulkan, Metal): how much softer the top of the roll-off is above 2.5x paper white (BO-195): 0 the plain exponential curve, 0.35 (default) a blend with a slower rational curve, so a 4-6x core keeps a gradient before the peak") \
    X(GFX_HDR_BLACK,     CONFIG, "RECOMP_GFX_HDR_BLACK",     NULL, "HDR (Vulkan, Metal): the display's black level in nits (BO-195), 0 (default): the whole frame, HUD included, is lifted by it so black is the panel's own, nits' = black + (1 - black / peak) x nits") \
    X(PCACHE_HDR,        CONFIG, "RECOMP_PCACHE_HDR",        NULL, "1: a shader pre-build also builds the HDR variants (BO-110; the game sets it for its pre-build where the display can show HDR or HDR is on); 2: pretend HDR can be reached, to test the background round of BO-208") \
    X(PCACHE_THREADS,    CONFIG, "RECOMP_PCACHE_THREADS",    NULL, "threads of the first shader pre-warm (default cores - 2, 1 to 4; BO-208)") \
    X(PCACHE_HOLD_S,     CONFIG, "RECOMP_PCACHE_HOLD_S",     NULL, "the longest the title is held for a cold first build, in seconds (default 180; BO-208)") \
    X(GFX_HDR_DECODE,    CONFIG, "RECOMP_GFX_HDR_DECODE",    NULL, "HDR (Vulkan): how the title's 8-bit values become light: 2.2 (default, the gamma the art was made for, as a TV and gamescope's SDR show it) or srgb (the piecewise sRGB curve: lighter near black)") \
    X(GFX_SHARPEN,       CONFIG, "RECOMP_GFX_SHARPEN",       NULL, "post-process sharpening (AMD FidelityFX CAS) strength, 0..1 (default 0, off) (Metal, Vulkan windows)") \
    X(GFX_LOD_BIAS,      CONFIG, "RECOMP_GFX_LOD_BIAS",      NULL, "extra mip LOD bias of mipmapped textures, added to the title's own: 0 (default, off), auto (-log2(render scale) / 2: -0.5 at 2x, -1.0 at 4x, so the texture detail keeps up with the supersampled frame), or a number such as -0.5; an alpha-tested draw (foliage, fences) takes no negative bias (Metal, Vulkan)") \
    X(GFX_UI_SCALE,      CONFIG, "RECOMP_GFX_UI_SCALE",      NULL, "the title's 2D art (HUD, menus, fonts: textures of one level that a draw without depth test binds) scaled on the CPU before upload: off (default), 2x, 4x (Metal, Vulkan)") \
    X(GFX_DITHER,        CONFIG, "RECOMP_GFX_DITHER",        NULL, "1: noise dither of +-0.5 LSB (interleaved gradient noise) in the Metal presenter's scale to the window, against banding of smooth gradients (default 0, off)") \
    X(STDIO_LOG,         CONFIG, "RECOMP_STDIO_LOG",         NULL, "send stdout/stderr to this file") \
    X(LOG_LEVEL,         CONFIG, "RECOMP_LOG_LEVEL",         "XBOX_LOG_LEVEL", "kernel log level, 0 (error) .. 4 (trace)") \
    X(MENU_ENTRY,        CONFIG, "RECOMP_MENU_ENTRY",        NULL, "interface.menu_entry: 1 (default) the game's own Settings menu lists an entry that opens this overlay, for a title that offers one (Burnout 3: TAKEDAWN SETTINGS); 0 not") \
    X(LOG_DIAG,          CONFIG, "RECOMP_LOG_DIAG",          NULL, "advanced.diagnostic_log: 1 prints the diagnostic lines (the default of a developer build), 0 only errors and the start-up facts (the default of a release build); see xr_log.h for what else switches it on") \
    X(AUDIO_MIXER,       CONFIG, "RECOMP_AUDIO_MIXER",       NULL, "the host mixer (buses over the title's voices): off (default), on") \
    X(AUDIO_MENU_SOUNDS, CONFIG, "RECOMP_AUDIO_MENU_SOUNDS", NULL, "audio.menu_sounds: 1 (default) the overlay plays the title's own menu sounds, for a title whose port names them (xbox_OverlaySetMenuSounds); 0 silent") \
    X(AUDIO_MIXER_FILE,  CONFIG, "RECOMP_AUDIO_MIXER_FILE",  NULL, "the mixer's preset file (default: mixer.toml beside settings.toml)") \
    X(AUDIO_FX_LEVEL,    CONFIG, "RECOMP_AUDIO_FX_LEVEL",    NULL, "5.1: level of the title's effect-send bins, 0..1 (default 0.25)") \
    X(AUDIO_LFE_LOWPASS, CONFIG, "RECOMP_AUDIO_LFE_LOWPASS", NULL, "low-pass on the LFE (subwoofer) feed of a title whose effects image is modelled: 120 (default: what the console's Dolby Digital encoder did), 100, 80, or off (the full-range feed)") \
    X(CONSOLE_HDKEY,     CONFIG, "RECOMP_CONSOLE_HDKEY",     NULL, "console.hdd_key: the HDD key of an original Xbox, 32 hex digits, so its saves load as they are (kernel_console_keys.h); default empty, the port's own zero keys") \
    X(CONSOLE_EEPROM,    CONFIG, "RECOMP_CONSOLE_EEPROM",    NULL, "console.eeprom: the path of an eeprom.bin to take the HDD key from (the HDD key wins when both are set); default empty") \
    X(AUDIO_MUSIC_DIR,   CONFIG, "RECOMP_AUDIO_MUSIC_DIR",   NULL, "audio.custom_music_dir: a folder of the user's music (mp3, flac, ogg, wav) offered to the title as Xbox custom soundtracks, for a title that plays them (xbox_soundtrack.h); default empty, off") \
    X(AUDIO_CUSTOM_LEVEL, CONFIG, "RECOMP_AUDIO_CUSTOM_LEVEL", NULL, "audio.custom_music_level: the level of the user's own music (custom soundtracks) against the title's music: off (the title's sends as they are), quieter (-4 dB), match (default: brought to the level of the title's own music), louder (+4 dB); for a title whose port names its custom-soundtrack voices (apu_level.h)") \
    X(AUDIO_SAFETY_LIMITER, CONFIG, "RECOMP_AUDIO_SAFETY_LIMITER", NULL, "0: the peak limiter in front of the 16-bit clamp (-0.5 dBFS, no look-ahead; apu_limit.h) is off and the clamp alone cuts a mix past full scale, as before it existed (default 1, on; it changes nothing while the mix stays under the ceiling)") \
    X(AUDIO_LEVEL_LOG,   CONFIG, "RECOMP_AUDIO_LEVEL_LOG",   NULL, "1: the custom-music leveller logs its envelope and gain once a second (default 0)") \
    X(AUDIO_BUF_SAMPLES, CONFIG, "RECOMP_AUDIO_BUF_SAMPLES", NULL, "audio output buffer size in samples") \
    X(AUDIO_BUF_COUNT,   CONFIG, "RECOMP_AUDIO_BUF_COUNT",   NULL, "audio output buffer count") \
    X(CMDLINE,           CONFIG, "RECOMP_CMDLINE",           NULL, "launch-data command line handed to the title") \
    X(FMV_HOST,          CONFIG, "RECOMP_FMV_HOST",          NULL, "play the title's movies through the host player") \
    X(RASTER_THREADS,    CONFIG, "RECOMP_RASTER_THREADS",    NULL, "CPU raster threads, 1..16 (default: cores - 4)") \
    /* ── trace: RECOMP_TRACE=key[=value],... ─────────────────────────── */ \
    X(FLIP_LOG,          TRACE, "flip",            "RECOMP_FLIP_LOG",          "one line per flip (golden.py reads it)") \
    X(FLIP_MARK,         TRACE, "flipmark",        "RECOMP_FLIP_MARK",         "one short line per flip, number and time (scene drivers count frames by it)") \
    X(PRESENT_TRACE,     TRACE, "present",         "RECOMP_PRESENT_TRACE",     "present-surface choices") \
    X(PRESENT_STATS,     TRACE, "present_stats",   "RECOMP_PRESENT_STATS",     "SDL host present statistics") \
    X(METAL_PROF,        TRACE, "metal_prof",      NULL,                       "Metal flip cost every 300 flips: GPU wait, readbacks, write-back, slot blit, hand-off, GPU time") \
    X(METAL_TRACE,       TRACE, "metal",           NULL,                       "Metal render-target events: each decode sync and stale drop, with target, texture and flip") \
    X(METAL_FRAME,       TRACE, "metal_frame",     NULL,                       "Metal backend, one line per flip: the render passes in order (target offset, guest size, C cleared / L loaded, draws) and the self copies (a target copied to be sampled while it is drawn into)") \
    X(PCACHE_TRACE,      TRACE, "pcache",          NULL,                       "pipeline cache: each shader/pipeline compile (thread, ms), the pre-warm summary; =<file>: also one CSV line per flip (t_ms, interval_ms, compiles)") \
    X(VK_PROF,           TRACE, "vk_prof",         NULL,                       "Vulkan flip cost every 300 flips: frame waits, write-backs, submits, GPU time") \
    X(VK_FRAME,          TRACE, "vk_frame",        NULL,                       "=<file>: Vulkan backend, one CSV line per flip: draws, render passes (cleared, loaded, resumed on a target already drawn that frame, empty), queue submits by reason, time in submits, timeline polls, waits and the flip's steps, GPU time from timestamp queries") \
    X(VK_PRESENT_LOG,    TRACE, "vk_present",      NULL,                       "=<file>: Vulkan window, one CSV line per present and per frame displayed (present_wait)") \
    X(POST_GPU,          TRACE, "post_gpu",        NULL,                       "window presenter: each post-process pass's GPU time (FXAA, SMAA's three, CAS) and the scale and overlay's, averaged every 300 presents (Metal; Vulkan: the post passes together)") \
    X(VK_TRACE,          TRACE, "vk",              NULL,                       "Vulkan render-target events, pipeline and shader compiles") \
    X(VSH_TRACE,         TRACE, "vsh",             "RECOMP_VSH_TRACE",         "vertex programs and their batches") \
    X(VSH_ZLOG,          TRACE, "zlog",            "RECOMP_VSH_ZLOG",          "depth clears and z-buffer use") \
    X(TEX_LOG,           TRACE, "tex",             "RECOMP_TEX_LOG",           "texture format survey; =2 also every texture") \
    X(TEX_STATE,         TRACE, "tex_state",       "RECOMP_TEX_STATE",         "texture-state methods seen, in the report") \
    X(CLIP_TRACE,        TRACE, "clip",            "RECOMP_CLIP_TRACE",        "near-plane clipping, per triangle") \
    X(ZPASS_TRACE,       TRACE, "zpass",           "RECOMP_ZPASS_TRACE",       "occlusion report writes") \
    X(PB_EXEC_VERBOSE,   TRACE, "pb_verbose",      "RECOMP_PB_EXEC_VERBOSE",   "executor bring-up detail") \
    X(SURF_TRACE,        TRACE, "surf",            "RECOMP_SURF_TRACE",        "=offset: surface methods from that colour offset") \
    X(PB_SEMA_TRACE,     TRACE, "sema",            "RECOMP_PB_SEMA_TRACE",     "every semaphore release") \
    X(PB_UNHANDLED_ALL,  TRACE, "unhandled_all",   "RECOMP_PB_UNHANDLED_ALL",  "list every unhandled method, not ten") \
    X(PB_SCAN,           TRACE, "pb_scan",         "RECOMP_PB_SCAN",           "pushbuffer survey and its report") \
    X(NV2A_TRACE,        TRACE, "nv2a",            "RECOMP_NV2A_TRACE",        "NV2A register poll") \
    X(PB_WRAP_TRACE,     TRACE, "pb_wrap",         "RECOMP_PB_WRAP_TRACE",     "pushbuffer wraps") \
    X(HEAP_TRACE,        TRACE, "heap",            "RECOMP_HEAP_TRACE",        "=va: heap blocks covering that address") \
    X(KERNEL_LOG_BUDGET, TRACE, "kernel_budget",   "RECOMP_KERNEL_LOG_BUDGET", "=n: kernel log lines per call site (200)") \
    X(TRACE_BUDGET,      TRACE, "call_budget",     "RECOMP_TRACE_BUDGET",      "=n: call-trace line budget (400000)") \
    X(TRACE_ARGS,        TRACE, "call_args",       "RECOMP_TRACE_ARGS",        "=n: call trace prints n stack args") \
    X(TRACE_DEREF,       TRACE, "call_deref",      "RECOMP_TRACE_DEREF",       "call trace follows pointer args") \
    X(TRACE_PROFILE,     TRACE, "call_profile",    "RECOMP_TRACE_PROFILE",     "call profile; =n report interval") \
    X(IRQL_TRACE,        TRACE, "irql",            "RECOMP_IRQL_TRACE",        "first IRQL transitions") \
    X(APU_TRACE,         TRACE, "apu",             "RECOMP_APU_TRACE",         "APU register and frame trace; [APU-IRQ] line every 5 s") \
    X(AUDIO_HOST,        TRACE, "audio_host",      NULL,                       "host audio playback vs wall clock; [AUDIO-HOST] starve lines (on with apu)") \
    X(AUDIO_CHANNELS,    TRACE, "audio_channels",  NULL,                       "level of every APU mixbin and output channel, once a second: [AUDIO-CH]") \
    X(MIX_TRACE,         TRACE, "mix",             "RECOMP_MIX_TRACE",         "=file: per-voice mix study: who turned each voice on, its bins, volumes, speaker shares and levels, once a second; output peaks before the clamp") \
    X(APU_RING,          TRACE, "apu_ring",        NULL,                       "looping APU buffers refilled by the title: [APU-RING] lead and stale-lap replays") \
    X(MMIO_TRACE,        TRACE, "mmio",            NULL,                       "trapped vs direct device-register accesses by device, register and host PC; [MMIO] block every 10 s") \
    X(USB_TRACE,         TRACE, "usb",             "RECOMP_USB_TRACE",         "OHCI trace") \
    X(USB_STATS,         TRACE, "usb_stats",       "RECOMP_USB_STATS",         "OHCI summary line every 5 s") \
    X(INPUT_DIAG,        TRACE, "input_diag",      "RECOMP_INPUT_DIAG",        "input chain probe once a second") \
    X(KEY_TRACE,         TRACE, "key",             "RECOMP_KEY_TRACE",         "key-down events, every window") \
    X(TITLE_STATS,       TRACE, "title",           NULL,                       "window title also shows FPS and draws, once a second") \
    X(CS_TRACE_CRT,      TRACE, "cs_crt",          "RECOMP_CS_TRACE_CRT",      "CRT critical sections; =all every one") \
    X(CS_WATCH,          TRACE, "cs_watch",        "RECOMP_CS_WATCH",          "=va: critical section at that address") \
    X(KERNEL_WATCH,      TRACE, "kernel_watch",    "RECOMP_KERNEL_WATCH",      "=va: report bridges that change it") \
    X(KERNEL_WATCH_ALL,  TRACE, "kernel_watch_all","RECOMP_KERNEL_WATCH_ALL",  "kernel_watch: every new value") \
    X(PEEK,              TRACE, "peek",            "RECOMP_PEEK",              "=va[:n],...: print guest memory at reports") \
    X(PEEK_CHAIN,        TRACE, "peek_chain",      "RECOMP_PEEK_CHAIN",        "=va,off,...: follow a pointer chain") \
    X(FIND_NAN,          TRACE, "find_nan",        "RECOMP_FIND_NAN",          "scan for NaN matrices at reports") \
    X(FIND_QUAD,         TRACE, "find_quad",       "RECOMP_FIND_QUAD",         "scan for quad vertices once") \
    X(D3D11_VERBOSE,     TRACE, "d3d11_verbose",   "RECOMP_D3D11_VERBOSE",     "D3D11 shader sources") \
    X(D3D11_PX,          TRACE, "px",              "RECOMP_D3D11_PX",          "=x,y[;x,y]: D3D11/CPU draws touching a pixel") \
    X(D3D11_PX_FLIPS,    TRACE, "px_flips",        "RECOMP_D3D11_PX_FLIPS",    "=a[-b]: px only in those flips") \
    X(D3D11_PX_MAX,      TRACE, "px_max",          "RECOMP_D3D11_PX_MAX",      "=n: px line limit (400)") \
    X(D3D11_PX_CONSTS,   TRACE, "px_consts",       "RECOMP_D3D11_PX_CONSTS",   "=a-b,c: px also prints these constants") \
    X(D3D11_PX_VERTS,    TRACE, "px_verts",        "RECOMP_D3D11_PX_VERTS",    "px also prints program and vertices") \
    X(STUB_LOG,          TRACE, "stub",            "RECOMP_STUB_LOG",          "name each unresolved stub reached") \
    /* ── debug: RECOMP_DEBUG=key[=value],... ─────────────────────────── */ \
    X(PB_VSH,            DEBUG, "pb_vsh",          "RECOMP_PB_VSH",            "=0: no vertex-program interpreter") \
    X(PB_RC,             DEBUG, "pb_rc",           "RECOMP_PB_RC",             "=0: no register combiners (MODULATE)") \
    X(PB_FAST,           DEBUG, "pb_fast",         "RECOMP_PB_FAST",           "=0: CPU raster without fast paths") \
    X(PB_FAST_AB,        DEBUG, "pb_fast_ab",      "RECOMP_PB_FAST_AB",        "compare fast and slow raster per batch") \
    X(PB_BILINEAR,       DEBUG, "pb_bilinear",     "RECOMP_PB_BILINEAR",       "=0: CPU raster samples nearest, whatever the filter") \
    X(PB_MIPS,           DEBUG, "pb_mips",         "RECOMP_PB_MIPS",           "=0: CPU raster samples mip level 0 only") \
    X(PB_CLIP,           DEBUG, "pb_clip",         "RECOMP_PB_CLIP",           "=0: no near-plane clipping") \
    X(PB_CULL_FLIP,      DEBUG, "pb_cull_flip",    "RECOMP_PB_CULL_FLIP",      "CPU raster culls the other winding") \
    X(PB_VSH_AB,         DEBUG, "pb_vsh_ab",       "RECOMP_PB_VSH_AB",         "compare vertex-program paths per batch") \
    X(PB_VSH_AB_DUMP,    DEBUG, "pb_vsh_ab_dump",  "RECOMP_PB_VSH_AB_DUMP",    "=prefix: pb_vsh_ab differing batches") \
    X(RASTER_TEST,       DEBUG, "raster_test",     "RECOMP_RASTER_TEST",       "draw a known triangle on every clear") \
    X(ZPASS_FIXED,       DEBUG, "zpass_fixed",     "RECOMP_ZPASS_FIXED",       "=n: every occlusion report reads n") \
    X(PB_NULL_DRAW_US,   DEBUG, "null_draw_us",    "RECOMP_PB_NULL_DRAW_US",   "=us: null backend cost per draw") \
    X(PB_INJECT_STOP,    DEBUG, "pb_inject_stop",  "RECOMP_PB_INJECT_STOP",    "=n: stop every nth walk (test hook)") \
    X(AUDIO_STALL,       DEBUG, "audio_stall",     NULL,                       "=at[:for]: the audio device stops taking samples at second `at`, for `for` seconds or for good (test hook; SDL sink)") \
    X(AUDIO_OPEN_HANG,   DEBUG, "audio_open_hang", NULL,                       "the audio device's reopen never returns (test hook; SDL sink)") \
    X(VK_PRESENT_FAIL,   DEBUG, "vk_present_fail", NULL,                       "=at:n[:what]: from second `at`, n acquires in a row fail with what: outdated (default), surface, timeout (test hook; Vulkan window)") \
    X(FAST_KICK,         DEBUG, "fast_kick",       "RECOMP_FAST_KICK",         "=0: ack loop ticks at Sleep(1)") \
    X(FORCE_RETURN,      DEBUG, "force_return",    "RECOMP_FORCE_RETURN",      "honour forced returns in the build") \
    X(TRAP_NULL,         DEBUG, "trap_null",       "RECOMP_TRAP_NULL",         "guest page zero faults") \
    X(DSP_ACK,           DEBUG, "dsp_ack",         "RECOMP_DSP_ACK",           "=va,...: zero these words (no APU)") \
    X(APU_DSP_ACK,       DEBUG, "apu_dsp_ack",     "RECOMP_APU_DSP_ACK",       "=auto|va,...: APU DSP acks the GP doorbell (from GPSADDR) or these words") \
    X(APU_SOLO,          DEBUG, "apu_solo",        "RECOMP_APU_SOLO",          "=voice: mix only this APU voice") \
    X(APU_VOICE_DUMP,    DEBUG, "apu_voice_dump",  "RECOMP_APU_VOICE_DUMP",    "=voice|all: raw samples of one APU voice (all: every voice) to a file") \
    X(APU_VOICE_DUMP_FILE, DEBUG, "apu_voice_dump_file", "RECOMP_APU_VOICE_DUMP_FILE", "=path: apu_voice_dump output, default voice_dump.raw") \
    X(APU_MIXDOWN_ALL,   DEBUG, "apu_mixdown_all", "RECOMP_APU_MIXDOWN_ALL",   "=0: mix only two bins") \
    X(POKE,              DEBUG, "poke",            "RECOMP_POKE",              "=va:value,...: write guest words") \
    X(WORKERS,           DEBUG, "workers",         "RECOMP_WORKERS",           "=inline: run worker threads inline") \
    X(GUEST_CPUS,        DEBUG, "guest_cpus",      "RECOMP_GUEST_CPUS",        "=all: the title's threads on every host core; =<n>: that core; =token: one at a time by a token, for hosts without thread affinity (macOS) (default: one core, as on the console)") \
    X(HOST_PRIORITY,     DEBUG, "host_priority",   NULL,                       "Windows: the process's priority class while the title's threads are pinned to one core: above (default), high, normal, or keep (as started); never lowered") \
    X(CS_MODE,           DEBUG, "cs_mode",         "RECOMP_CS_MODE",           "=single: one lock for every guest lock") \
    X(ASYNC_IO,          DEBUG, "async_io",        "RECOMP_ASYNC_IO",          "asynchronous file I/O") \
    X(UNIMPL_TRAP,       DEBUG, "unimpl_trap",     "RECOMP_UNIMPL_TRAP",       "stop at an unimplemented instruction") \
    X(CPUID_MMX,         DEBUG, "cpuid_mmx",       NULL,                       "=1: cpuid reports MMX/FXSR/SSE (D3DX JPEG takes its MMX IDCT); default masked") \
    X(D3D11_MEMO,        DEBUG, "d3d11_memo",      "RECOMP_D3D11_MEMO",        "=0: no D3D11 state memo") \
    X(D3D11_NO_CULL,     DEBUG, "d3d11_no_cull",   "RECOMP_D3D11_NO_CULL",     "D3D11 culls nothing") \
    X(D3D11_CULL_FLIP,   DEBUG, "d3d11_cull_flip", "RECOMP_D3D11_CULL_FLIP",   "D3D11 culls the other winding") \
    X(D3D11_NO_MIPS,     DEBUG, "d3d11_no_mips",   "RECOMP_D3D11_NO_MIPS",     "D3D11 uploads level 0 only") \
    X(D3D11_POINT,       DEBUG, "d3d11_point",     "RECOMP_D3D11_POINT",       "D3D11 point sampling") \
    X(D3D11_NO_RTT,      DEBUG, "d3d11_no_rtt",    "RECOMP_D3D11_NO_RTT",      "D3D11 decodes render targets from memory") \
    X(D3D11_DEBUG_PS,    DEBUG, "d3d11_debug_ps",  "RECOMP_D3D11_DEBUG_PS",    "=name: replace pixel shaders") \
    X(D3D11_DEBUG_PS_SKIP_POST, DEBUG, "d3d11_debug_ps_skip_post", "RECOMP_D3D11_DEBUG_PS_SKIP_POST", "debug_ps spares post passes") \
    X(D3D11_OCC,         DEBUG, "d3d11_occ",       "RECOMP_D3D11_OCC",         "=sync|fixed: D3D11 occlusion mode") \
    X(METAL_NO_MIPS,     DEBUG, "metal_no_mips",   "RECOMP_METAL_NO_MIPS",     "Metal uploads level 0 only") \
    X(METAL_POINT,       DEBUG, "metal_point",     "RECOMP_METAL_POINT",       "Metal point sampling") \
    X(METAL_NO_RTT,      DEBUG, "metal_no_rtt",    "RECOMP_METAL_NO_RTT",      "Metal decodes render targets from memory (implies metal_writeback=always)") \
    X(WINDOW_DISPLAY,    DEBUG, "window_display",  NULL,                       "=n: open the window centred on display n (0 = main)") \
    X(WINDOW_FLOAT,      DEBUG, "window_float",    NULL,                       "1: the window stays above other windows (macOS; a measured run needs it visible: an occluded window reports no on-glass times)") \
    X(GLASS_MEASURE,     DEBUG, "glass_measure",   NULL,                       "=0: the Metal presenter installs no on-glass handler with display.present_timing = legacy (A/B of the measurement's own cost); the graph's on-glass line then says not available") \
    X(LAT_SIG,           DEBUG, "lat_sig",         NULL,                       "=1 with RECOMP_TRACE=latency: the Vulkan presenter reads each new frame back for its 16x12 signature (G rows), so tools/perf/latency.py finds the frame that answers a press (BO-110; a full-frame copy per flip, off by default)") \
    X(EARLY_COMMIT,      DEBUG, "early_commit",    NULL,                       "=0: submit a frame's GPU work after the vblank hold, not at FLIP_STALL (A/B)") \
    X(METAL_FRAMES,      DEBUG, "metal_frames",    NULL,                       "=1: Metal waits for each frame at its flip; default 2 frames in flight") \
    X(METAL_PRESENT,     DEBUG, "metal_present",   NULL,                       "=layer|readback: Metal window present, CAMetalLayer from GPU slots (default) or the old texture readback") \
    X(METAL_WRITEBACK,   DEBUG, "metal_writeback", NULL,                       "=lazy|always: Metal writes guest memory when a dump reads it (default) or at every flip") \
    X(METAL_FB_GUARD,    DEBUG, "metal_fb_guard",  NULL,                       "trap title reads of the last Metal present surface; a read switches to metal_writeback=always") \
    X(METAL_OCC,         DEBUG, "metal_occ",       "RECOMP_METAL_OCC",         "=sync|fixed: Metal occlusion mode") \
    X(METAL_PCACHE_MAX,  DEBUG, "metal_pcache_max", NULL,                      "=n: Metal keeps at most n vertex functions, n fragment functions and n pipelines (LRU; default 512 each)") \
    X(PRESENT_STALL,     DEBUG, "present_stall",   NULL,                       "=ms[:every_s]: Vulkan, the presenter sleeps ms before a present every every_s seconds (default 10): a late present on demand, to see how the queue recovers (BO-74 test hook)") \
    X(CADENCE_DIAG,      DEBUG, "cadence_diag",    NULL,                       "=0: no [PACING] line when the frame rate drops (on by default: says whether the executor, the title or the pacing was the cause; on a pacing fault the display lock relearns the refresh; BO-210)") \
    X(VK_VALIDATION,     DEBUG, "vk_validation",   NULL,                       "Vulkan: enable VK_LAYER_KHRONOS_validation when installed") \
    X(VK_NO_MIPS,        DEBUG, "vk_no_mips",      NULL,                       "Vulkan uploads level 0 only") \
    X(VK_POINT,          DEBUG, "vk_point",        NULL,                       "Vulkan point sampling") \
    X(VK_NO_RTT,         DEBUG, "vk_no_rtt",       NULL,                       "Vulkan decodes render targets from memory (implies vk_writeback=always)") \
    X(VK_WRITEBACK,      DEBUG, "vk_writeback",    NULL,                       "=lazy|always: Vulkan writes guest memory when a dump reads it (default) or at every flip") \
    X(VK_DEVICE_LOST,    DEBUG, "vk_device_lost",  NULL,                       "=<flip>: Vulkan backend treats its first submit after that flip as VK_ERROR_DEVICE_LOST (a test of what the player is told when the driver gives up)") \
    X(VK_WAIT_LIMIT,     DEBUG, "vk_wait_limit",   NULL,                       "=<ms>: a Vulkan GPU wait (fence or timeline) that keeps timing out gives up after this long, and the run ends as for a lost device (default 10000)") \
    X(VK_WAIT_FAULT,     DEBUG, "vk_wait_fault",   NULL,                       "=<site>:<kind>[:<n>]: test hook, the Vulkan GPU wait after n (default 300) at site serial|frame|upload|present|shot|slot fails as kind: retry (two timeouts, then done), timeout (for good), lost (VK_ERROR_DEVICE_LOST)") \
    X(VK_POLL,           DEBUG, "vk_poll",         NULL,                       "=0|1: Vulkan backend asks the timeline semaphore how far the GPU is (1: default off Android) or never asks and learns it from a fence per frame, waited for when the frame's slot is reused (0: default on Android, where Turnip on KGSL answers the question by waiting for the GPU)") \
    X(VK_BATCH,          DEBUG, "vk_batch",        NULL,                       "=<draws>: Vulkan backend submits what a frame has recorded when a render pass begins and that many draws (or 2 ms with any draw) have gone since the last submit, so the GPU starts on the frame while the rest is recorded; an occlusion report then waits for the pass to end instead of ending it. 0: one submit when the frame is complete (default; Android: 300)") \
    X(VK_OCC,            DEBUG, "vk_occ",          NULL,                       "=sync|fixed: Vulkan occlusion mode") \
    X(VK_PRESENT,        DEBUG, "vk_present",      NULL,                       "=window|readback: Vulkan window present, swapchain from GPU slots (default) or texture readback into the SDL renderer") \
    X(VK_FRAMES,         DEBUG, "vk_frames",       NULL,                       "=1|2: Vulkan frames in flight (default 2)") \
    X(RT_ALIAS_CHECK,    DEBUG, "rt_alias_check",  NULL,                       "=0: Metal and D3D11 keep render targets whose guest memory the title rewrote (before rt-stale-alias)") \
    X(FB_VA,             DEBUG, "fb_va",           "RECOMP_FB_VA",             "=va: framebuffer window shows that address") \
    X(USB,               DEBUG, "ohci",            "RECOMP_USB",               "emulated OHCI/XID USB (unused path)") \
    X(USB_PORT,          DEBUG, "usb_port",        "RECOMP_USB_PORT",          "=n: USB pad port") \
    X(USB_HC,            DEBUG, "usb_hc",          "RECOMP_USB_HC",            "=1: pad on the second controller") \
    X(USB_NDP,           DEBUG, "usb_ndp",         "RECOMP_USB_NDP",           "=n: root-hub ports, 1..4") \
    X(PAD_PRESS,         DEBUG, "pad_press",       "RECOMP_PAD_PRESS",         "=mask: pulse these buttons (USB pad)") \
    X(FB_DUMP,           DEBUG, "fb_dump",         "RECOMP_FB_DUMP",           "=prefix: frame dumps (CPU/Metal, window)") \
    X(TEXREP_DUMP,       DEBUG, "texrep_dump",     NULL,                       "=dir: every texture a GPU backend (Metal, Vulkan) uploads, once per content, as <hash>_<w>x<h>_f<format>.tga with an index.tsv (the hash is the one nv2a_texrep registers)") \
    X(TEXREP_DUMP_MIPS,  DEBUG, "texrep_dump_mips", NULL,                      "with texrep_dump: a mipmapped texture's own levels 1.. as well, as <hash>_<w>x<h>_f<format>_m<level>.tga") \
    X(ALPHA_MIPS,        DEBUG, "alpha_mips",      NULL,                       "BO-171, alpha-tested (cut-out) textures: =0 the title's own mip levels and the host LOD bias as before; =mips only the levels made again with their alpha coverage kept; =bias only no negative host bias on alpha-tested draws; default both (Metal, Vulkan)") \
    X(ALPHA_MIPS_AB,     DEBUG, "alpha_mips_ab",   NULL,                       "=n: alpha_mips on for n flips, then off for n, and so on, the textures made again at each change: both pictures in one run (comparisons)") \
    X(ZBIAS,             DEBUG, "zbias",           NULL,                       "BO-176: =0 the depth offsets a title registers for its decal textures (nv2a_texrep_register_zbias) are not applied (Metal, Vulkan)") \
    X(ZBIAS_AB,          DEBUG, "zbias_ab",        NULL,                       "=n: the registered depth offsets on for n flips, then off for n, and so on: both pictures in one run (comparisons)") \
    X(ZBIAS_PROBE,       DEBUG, "zbias_probe",     NULL,                       "BO-204, =<flip>[+<flip>...]: or =@file: from each, four flips a texture: its depth-tested draws unmoved, 512 units back, unmoved, 4 back, logged per texture (Vulkan): finds decals drawn just in front of a surface in a paused scene") \
    X(ZBIAS_ADD,         DEBUG, "zbias_add",       NULL,                       "BO-204, =<hash>[+<hash>...]: those textures (texrep_dump hashes) take a -512 depth offset as if the title had registered them (Metal, Vulkan): trying a candidate") \
    X(ZTINT,             DEBUG, "ztint",           NULL,                       "BO-204, =<hash>[+<hash>...]: those textures (texrep_dump hashes) uploaded magenta, alpha kept (Metal, Vulkan): count a decal's pixels per frame") \
    X(HDR,               DEBUG, "hdr",             NULL,                       "BO-110 prototype, Metal: =1 the colour render targets are half float and the 3D of a stage view (between the title's scene marks) blends and combines without the 1.0 clamp, so stacked light keeps its value; the HUD and menus stay clamped (paper white). Off: nothing changes") \
    X(HDR_AB,            DEBUG, "hdr_ab",          NULL,                       "=n: with hdr, HDR on for n flips, then off for n, and so on (A/B in one run)") \
    X(HDR_HI,            DEBUG, "hdr_hi",          NULL,                       "=x: with hdr, the upper bound of a scene draw's combiner values (default 64)") \
    X(HDR_TEXLOG,        DEBUG, "hdr_texlog",      NULL,                       "with hdr: one line per texture (texrep_dump hash), blend and stream tag a draw uses, for finding the title's light sprites") \
    X(HDR_RT8,           DEBUG, "hdr_rt8",         NULL,                       "=0: with HDR on Vulkan, every colour target is half float (default: the small 2D targets, bloom levels and the like, stay BGRA8)") \
    X(HDR_GAINS,         DEBUG, "hdr_gains",       NULL,                       "=0: with HDR on Vulkan, the per-texture light gains a title registers (nv2a_texrep_register_hdr_gain) are not applied; =only:<hash>[+<hash>...] only those entries are (texrep_dump hashes)") \
    X(HDR_LODFADE,       DEBUG, "hdr_lodfade",     NULL,                       "=n: with HDR on Vulkan, the per-texture light gains fade out over the first n mip levels of the gained texture (BO-195: no sparkle on small, far sprites; default 2); =0 the gain is the same at every distance") \
    X(HDR_IMAGE,         DEBUG, "hdr_image",       NULL,                       "=0: the pictures a title registers for HDR (nv2a_texrep_register_hdr_image; BO-203: the sun's disc) are not used, the title's own texels in HDR too (a before/after comparison in one build)") \
    X(HDR_IMAGE_AB,      DEBUG, "hdr_image_ab",    NULL,                       "=n: with HDR on, the pictures a title registers for HDR (hdr_image) are off for every other block of n flips, so neighbouring flips are the same moment with and without them") \
    X(HDR_GAINS_AB,      DEBUG, "hdr_gains_ab",    NULL,                       "=n: with HDR on Vulkan, every other block of n flips differs: the per-texture gains are off in it, or with hdr_gains_probe one probe texture is marked in it; hdr_dump's .json says which (measuring a light class in one run)") \
    X(HDR_GAINS_PROBE,   DEBUG, "hdr_gains_probe", NULL,                       "=<hash>[+<hash>...]: with HDR on Vulkan, textures (texrep_dump hashes, registered or not) whose scene draws are drawn x4 in linear light, so an hdr_dump diff finds them: with hdr_gains_ab one per flip of its other blocks, else all on every flip") \
    X(HDR_WATCH,         DEBUG, "hdr_watch",       NULL,                       "=<hash>: with hdr and hdr_dump, also dump each flip (up to 60, 20 flips apart, each with the flip after it) where a view's 3D draws that texture (texrep_dump hash) with an additive blend, or with the alpha test on (BO-203: the sun disc; Metal, Vulkan: the .rgbaf only; Vulkan: =<hash>@<flip> from that flip on)") \
    X(HDR_DUMP,          DEBUG, "hdr_dump",        NULL,                       "=prefix: with hdr, the presented frame of every fb_dump_at flip as <prefix>flip_NNNNN.rgbaf (32-bit float RGBA, an 16-byte header: magic HDR1, w, h, hdr on)") \
    X(ALPHA_MIPS_LOG,    DEBUG, "alpha_mips_log",  NULL,                       "one line per texture an alpha-tested draw binds: texrep_dump's hash, the test, the alpha coverage of level 0 and of the levels below before and after alpha_mips (Metal, Vulkan)") \
    X(CHEATS,            DEBUG, "cheats",          NULL,                       "shows the Advanced page's cheat rows (xbox_OverlaySetCheats), a tester's tool: while one is on the title's saves go to a copy of the save") \
    X(UI_SCALE_ALGO,     DEBUG, "ui_scale_algo",   NULL,                       "=bicubic: graphics.ui_scale scales with a Catmull-Rom bicubic instead of Super-xBR (for comparison)") \
    X(UI_SCALE_HOLD,     DEBUG, "ui_scale_hold",   NULL,                       "graphics.ui_scale held off whatever the setting and the environment say: =1 on any platform (a way to rule the upscaling out when a GPU fault is chased); not held by default anywhere") \
    X(UI_SCALE_CACHE_MB, DEBUG, "ui_scale_cache_mb", NULL,                     "=n: size bound of graphics.ui_scale's disk cache of scaled textures, in MB (default 256); 0: no disk cache") \
    X(UI_SCALE_LOG,      DEBUG, "ui_scale_log",    NULL,                       "one line per texture and kind of draw that binds it (use 1: a 2D draw, 2: another, 3: both), with texrep_dump's hash and the factor it is uploaded at: what graphics.ui_scale's rule catches (Metal, Vulkan)") \
    X(FB_DUMP_AT,        DEBUG, "fb_dump_at",      "RECOMP_FB_DUMP_AT",        "=flips: fb_dump at these flips; =@file: the list in that file, read again when it grows") \
    X(FB_DUMP_FLIPS,     DEBUG, "fb_dump_flips",   "RECOMP_FB_DUMP_FLIPS",     "=n: fb_dump every nth flip") \
    X(FB_WINDOW_DUMP_EVERY, DEBUG, "fb_window_dump_every", "RECOMP_FB_WINDOW_DUMP_EVERY", "=n: window dump period") \
    X(WINDOW_SHOT,       DEBUG, "window_shot",     "RECOMP_WINDOW_SHOT",       "=flips: SDL window shots") \
    X(WINDOW_SHOT_PREFIX, DEBUG, "window_shot_prefix", "RECOMP_WINDOW_SHOT_PREFIX", "=prefix: window shot files") \
    X(AUTOPILOT,         DEBUG, "autopilot",       NULL,                       "=1: the title's own AI drives player 1's car while it races (a game's test driver; Burnout 3: src/game/autopilot.c)") \
    X(BUG_REPORT,        DEBUG, "bug_report",      NULL,                       "=seconds[:shot]: write a bug report archive that long after the start (xbox_report.h), :shot with the last frame; =@file: whenever that file appears; no window needed") \
    X(TEX_DUMP,          DEBUG, "tex_dump",        "RECOMP_TEX_DUMP",          "=prefix: textures, first use") \
    X(TEX_DUMP_EVERY,    DEBUG, "tex_dump_every",  "RECOMP_TEX_DUMP_EVERY",    "=n: tex_dump every nth bind too") \
    X(VSH_DUMP,          DEBUG, "vsh_dump",        "RECOMP_VSH_DUMP",          "=prefix: vertex programs, raw") \
    X(D3D11_DUMP,        DEBUG, "d3d11_dump",      "RECOMP_D3D11_DUMP",        "=prefix: D3D11 present 60k+1 (golden)") \
    X(FMV_DUMP,          DEBUG, "fmv_dump",        "RECOMP_FMV_DUMP",          "=prefix: two movie frames") \
    X(AUDIO_WAV,         DEBUG, "audio_wav",       "RECOMP_AUDIO_WAV",         "=path: WAV of the audio output") \
    X(AUDIO_WAV_SECS,    DEBUG, "audio_wav_secs",  "RECOMP_AUDIO_WAV_SECS",    "=s: audio_wav length cap") \
    X(WATCH,             DEBUG, "watch",           "RECOMP_WATCH",             "=spec: write watchpoint on a guest address") \
    X(WATCH_RAW,         DEBUG, "watch_raw",       "RECOMP_WATCH_RAW",         "watch also prints the raw frame") \
    X(WATCH_DEPTH,       DEBUG, "watch_depth",     "RECOMP_WATCH_DEPTH",       "=n: watch stack depth (14)") \
    X(WATCH_LEN,         DEBUG, "watch_len",       "RECOMP_WATCH_LEN",         "=bytes: watch range") \
    X(WATCHDOG_SECS,     DEBUG, "watchdog",        "RECOMP_WATCHDOG_SECS",     "=s: dump and exit after s seconds") \
    X(CRASH_TEST,        DEBUG, "crash_test",      NULL,                       "=segv[:s] or =hang[:s]: fault, or hang, the first guest kernel call after s seconds (default 5), to test the crash and hang reports") \
    /* from upstream (merge of origin/main 1409a7d), appended */ \
    X(USB_PADS,          CONFIG, "RECOMP_USB_PADS",          NULL, "emulated USB pads plugged in, 1..4 (default 1)") \
    X(PAD_SCRIPT,        DEBUG, "pad_script",      "RECOMP_PAD_SCRIPT",        "=ms:btn[+btn][:hold],... or =@file: timed pad presses") \
    X(PAD_LIVE,          DEBUG, "pad_live",        "RECOMP_PAD_LIVE",          "=file: press each line appended to it (btn[+btn][:hold])") \
    X(KEY_LIVE,          DEBUG, "key_live",        "RECOMP_KEY_LIVE",          "=file: hold keys for each line appended to it (key[+key][:hold ms]), headless") \
    X(PAD_SCRIPT_ONLY,   DEBUG, "pad_script_only", NULL,                       "=1: a scripted run with a window does not read the host's controllers either, as a headless one (a test on a handheld, whose built-in pad is always there)") \
    X(PAD_SCRIPT_KEEP,   DEBUG, "pad_script_keep", NULL,                       "=1: a pad script keeps driving a pad after a controller is used on it (merged, the script's buttons on top), and a headless scripted run reads the controllers") \
    X(PAD_SCRIPT_NOTE,   DEBUG, "pad_script_note", NULL,                       "=0: no on-screen note while a pad script drives a pad") \
    X(PB_REPORT_MS,      TRACE, "pb_report_ms",    "RECOMP_PB_REPORT_MS",      "=ms: [PB] report / fb_dump interval (10000, min 100)") \
    X(DPC_PROF,          TRACE, "dpc",             NULL,                       "[DPCPROF] time per host-run DPC/ISR routine and its waits, under the 600-vblank line") \
    X(FRAMETIME,         TRACE, "frametime",       NULL,                       "=file: one CSV line per flip: executor hold/busy/draw/present/idle and sampled guest time by section (game, d3d, xdk), spin, host, wait (tools/perf/framestats.py)") \
    X(FRAMETIME_SPIN,    TRACE, "frametime_spin",  NULL,                       "=va[+va...]: lifted functions whose sampled time counts as spin (a guest busy-wait), not as their section") \
    X(FRAMETIME_US,      TRACE, "frametime_us",    NULL,                       "=us: guest thread sampling period for frametime (500, min 100); 0: no sampling, the executor's columns only") \
    X(FRAMETIME_HOST,    TRACE, "frametime_host",  NULL,                       "=file: with frametime, the host stacks of guest-main's host-code samples in every frame where they reach frametime_host_ms (Mach, Linux)") \
    X(FRAMETIME_HOST_MS, TRACE, "frametime_host_ms", NULL,                     "=ms: the m_host a frame needs for frametime_host to write its stacks (4)") \
    X(LATENCY_TRACE,     TRACE, "latency",         NULL,                       "=file: input-to-photon CSV: pad report changes, guest ISR, swap, vblank release, executor done, GPU done + frame signature, drawable acquire/present/on-glass (tools/perf/latency.py)") \
    X(PB_SLOW,           TRACE, "pb_slow",         NULL,                       "=ms: a [PB] line for every pushbuffer method the executor spends that long in (default 5): which command a frame stood in; two clock reads a method header") \
    X(GLASS_TRACE,       TRACE, "glass",           NULL,                       "=file: one CSV line per presented frame: flip, emulated vblank, present target, commit, on-glass time, interval in ms and refreshes, steps (tools/perf/glass.py)") \
    X(KWAIT_TRACE,       TRACE, "kwait",           NULL,                       "=file: wall time of every kernel call by thread, ordinal and guest call site, a block a second; calls of 20 ms or more one line each with their arguments") \
    X(KWAIT_ALL,         TRACE, "kwait_all",       NULL,                       "=from-to: with kwait, every kernel call between those milliseconds of the trace, one line each") \
    X(PACING_TRACE,      TRACE, "pacing",          NULL,                       "[PACING] flip intervals, vblank gaps, CPU and spin-wait sites every 600 flips; =all also every flip") \
    X(TIMER_SPIN,        DEBUG, "timer_spin_us",   NULL,                       "=us: the timer thread spins this long before each vblank edge instead of sleeping to it (default 600 on Linux, 0 elsewhere; 0: sleep, as before)") \
    X(VBLANK_CLOCK,      DEBUG, "vblank_clock",    NULL,                       "=ms: the old millisecond vblank schedule and timer loop (A/B)") \
    X(MISSING,           TRACE, "missing",         NULL,                       "missing game files, one line each and an exit summary; default on, =0 off, =all also absent directories, other trees, probes, every FAILED repeat and NtOpenFile/IoCreateFile successes") \
    X(MISSING_LIST,      DEBUG, "missing_list",    NULL,                       "=path: the unique missing files with their attempt counts, written once by the first summary") \
    X(IRQ_SAFE_POINTS,   DEBUG, "irq_safe_points", NULL,                       "=0: device interrupts run beside the one-CPU gate holder after a hold-off, as before (default: posted, run by the holder at its next kernel call, spin-wait yield or gate release)") \
    X(IRQ_SAFE_MS,       DEBUG, "irq_safe_ms",     NULL,                       "=ms: how long a posted interrupt waits for the holder's safe point before the [IRQ] wait log (default 250; again every second)") \
    X(IRQ_SAFE_FORCE,    DEBUG, "irq_safe_force",  NULL,                       "=1: past irq_safe_ms, deliver beside the holder (A/B; reinstates the race)") \
    X(DPC_VBLANK_GUARD,  DEBUG, "dpc_vblank_guard_us", NULL,                   "=us: the timer thread starts no queued DPC this close before a vblank edge, so the edge is raised on time (default 1200; 0: off, as before)") \
    X(DPC_LAG,           DEBUG, "dpc_lag",         NULL,                       "=ms: the timer thread drains queued DPCs at most once per that many ms (a test: widens the window between an ISR and its DPC)") \
    X(DPC_PROMPT,        DEBUG, "dpc_prompt",      NULL,                       "=0: queued DPCs wait for the timer thread's next pass, as before (A/B; default: the timer thread is woken where the console would run them: a queue below DISPATCH_LEVEL, the one-CPU gate's release)") \
    X(DPC_STATS,         DEBUG, "dpc_stats",       NULL,                       "[DPC] queue-to-run latency p50/p99/max and the timer-thread wakes, with the [IRQ] summary and at exit") \
    /* ── the enhancements layer's keys (RECOMP_ENV_ENHANCE_KEYS, below) ── */ \
    RECOMP_ENV_ENHANCE_KEYS(X) \
    /* ── the game's own keys (RECOMP_ENV_GAME_KEYS, below) ──────────────── */ \
    RECOMP_ENV_GAME_KEYS(X)

typedef enum {
#define RECOMP_ENV_ENUM_(id, tier, name, legacy, help) RENV_##id,
    RECOMP_ENV_KEYS(RECOMP_ENV_ENUM_)
#undef RECOMP_ENV_ENUM_
    RENV_COUNT
} recomp_env_id;

/* Read the environment into the cache. Idempotent; the first lookup calls
 * it. Call it first thing in main so deprecation lines come out early. */
void recomp_env_init(void);
/* Print the reader's notes (deprecated names, unknown keys, help), held
 * since init. A host calls it once stderr is where it should be (after the
 * RECOMP_STDIO_LOG redirect); otherwise they are printed at exit. */
void recomp_env_flush_notes(void);

/* Tests that setenv() a variable after the first lookup re-read with this. */
void recomp_env_reload(void);

/* Override a value from code (NULL clears it). Takes a copy. */
void recomp_env_set(recomp_env_id id, const char *value);

extern const char *recomp_env_value_[RENV_COUNT];
extern volatile int recomp_env_ready_;

/* The value, or NULL when unset: what getenv(old name) returned. */
static inline const char *recomp_env(recomp_env_id id)
{
#if defined(_MSC_VER)
    if (!recomp_env_ready_)   /* volatile: acquire under MSVC's default /volatile:ms */
#else
    if (!__atomic_load_n(&recomp_env_ready_, __ATOMIC_ACQUIRE))
#endif
        recomp_env_init();
    return recomp_env_value_[id];
}

/* Set, non-empty and not starting with '0'. */
static inline int recomp_env_on(recomp_env_id id)
{
    const char *v = recomp_env(id);
    return v && *v && *v != '0';
}

/* atoi / strtoul(,0) / atof of the value, or `def` when unset. */
long recomp_env_int(recomp_env_id id, long def);
unsigned long recomp_env_ulong(recomp_env_id id, unsigned long def);
double recomp_env_double(recomp_env_id id, double def);

/* The variable or list key, for messages. */
const char *recomp_env_name(recomp_env_id id);
/* The older variable a key still answers to, or NULL (for a report that
 * says where a value came from). */
const char *recomp_env_legacy_name(recomp_env_id id);
/* The list variable a key belongs to ("RECOMP_TRACE", "RECOMP_DEBUG"), or
 * NULL for a config-tier variable, whose name is the variable. */
const char *recomp_env_list_var(recomp_env_id id);

#ifdef __cplusplus
}
#endif

#endif /* RECOMP_ENV_H */
