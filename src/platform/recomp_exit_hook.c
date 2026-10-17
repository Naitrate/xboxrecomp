/*
 * recomp_exit_hook.c - the hook a window runs before ExitProcess
 * (recomp_exit_hook.h).
 */
#include "recomp_exit_hook.h"

#include <stddef.h>

/* Set by whichever thread notes the first miss, read by the window thread:
 * an aligned pointer store, so the reader sees the old or the new hook. */
static recomp_exit_hook_fn volatile s_hook;

void recomp_exit_hook_set(recomp_exit_hook_fn fn)
{
    s_hook = fn;
}

#define EXIT_HOOKS_MAX 4
static recomp_exit_hook_fn volatile s_more[EXIT_HOOKS_MAX];

void recomp_exit_hook_add(recomp_exit_hook_fn fn)
{
    int i;

    for (i = 0; i < EXIT_HOOKS_MAX; i++) {
        if (s_more[i] == fn)
            return;
        if (s_more[i] == NULL) {
            s_more[i] = fn;
            return;
        }
    }
}

void recomp_exit_hook_run(const char *why)
{
    recomp_exit_hook_fn fn = s_hook;
    int i;

    if (fn != NULL)
        fn(why);
    for (i = 0; i < EXIT_HOOKS_MAX; i++) {
        fn = s_more[i];
        if (fn != NULL)
            fn(why);
    }
}

#define QUIT_HOOKS_MAX 8
static recomp_quit_hook_fn volatile s_quit[QUIT_HOOKS_MAX];

void recomp_quit_hook_add(recomp_quit_hook_fn fn)
{
    int i;

    for (i = 0; i < QUIT_HOOKS_MAX; i++) {
        if (s_quit[i] == fn)
            return;
        if (s_quit[i] == NULL) {
            s_quit[i] = fn;
            return;
        }
    }
}

void recomp_quit_hook_run(int stage)
{
    int i;

    for (i = 0; i < QUIT_HOOKS_MAX; i++) {
        recomp_quit_hook_fn fn = s_quit[i];
        if (fn != NULL)
            fn(stage);
    }
}
