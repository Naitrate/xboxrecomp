/*
 * recomp_exit_hook.h - work that must run when a window ends the process.
 *
 * The Win32 windows end the run with ExitProcess, which skips atexit. The
 * kernel's exit-time reports (the missing-file summary) register here, and
 * each window calls recomp_exit_hook_run before ExitProcess. It lives in the
 * platform layer because the D3D library is linked without the kernel in
 * its standalone tests, so it cannot call the kernel by name.
 */
#ifndef RECOMP_EXIT_HOOK_H
#define RECOMP_EXIT_HOOK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*recomp_exit_hook_fn)(const char *why);

/* One hook; a second call replaces the first. Any thread. */
void recomp_exit_hook_set(recomp_exit_hook_fn fn);

/* One more hook beside it (the pads' motors off, src/input/input_core.c);
 * up to four, a hook added twice is there once. Any thread. */
void recomp_exit_hook_add(recomp_exit_hook_fn fn);

/* Calls the hook, if one is set, and the added ones, with why ("window"). */
void recomp_exit_hook_run(const char *why);

/* The window's way out, in order (BO-219, BO-206): the window loop has ended and
 * the title's threads are still running; they are stopped before the GPU and
 * the audio device are taken down under them. The window calls
 * recomp_quit_hook_run for each stage in turn; a hook gets every stage and
 * acts on its own. Each is bounded by its own timeout and says so in the log
 * when it is hit. */
enum {
    RECOMP_QUIT_SETTLE = 0,   /* the title still runs: a save on its way finishes */
    RECOMP_QUIT_HOLD   = 1,   /* the title's executor is parked at a frame */
    RECOMP_QUIT_HELPERS = 2   /* helper threads (the APU) stop; the executor is held */
};
typedef void (*recomp_quit_hook_fn)(int stage);

/* Up to eight, a hook added twice is there once. Any thread. */
void recomp_quit_hook_add(recomp_quit_hook_fn fn);
void recomp_quit_hook_run(int stage);

#ifdef __cplusplus
}
#endif

#endif /* RECOMP_EXIT_HOOK_H */
