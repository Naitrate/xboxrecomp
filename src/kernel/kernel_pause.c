/*
 * kernel_pause.c - the host pause; see kernel_pause.h.
 */
#include "kernel.h"
#include "kernel_pause.h"
#include "xbox_crash.h"       /* g_xbox_flip_count */
#include "platform/host_time.h"
#include "recomp_exit_hook.h"

#include <stdlib.h>
#include <string.h>

#define PAUSE_LISTENERS 4
/* The hold wakes this often even without a resume: the watchdog's sign of
 * life, and a bound on a resume that was signalled before the wait began. */
#define PAUSE_TICK_MS 50

static volatile LONG s_paused;
/* xbox_HostHoldForExit: the hold is for good (no resume, no hold work), and
 * whether the executor is in it now. */
static volatile LONG s_exiting;
static volatile LONG s_held;
static CRITICAL_SECTION s_lock;
static CONDITION_VARIABLE s_cv;
static INIT_ONCE s_once = INIT_ONCE_STATIC_INIT;
static void (*s_listener[PAUSE_LISTENERS])(int);
static int s_n_listeners;

static void pause_quit_hook(int stage);

static BOOL CALLBACK pause_init(PINIT_ONCE o, PVOID p, PVOID *c)
{
    (void)o; (void)p; (void)c;
    InitializeCriticalSection(&s_lock);
    InitializeConditionVariable(&s_cv);
    recomp_quit_hook_add(pause_quit_hook);
    return TRUE;
}

static void pause_once(void)
{
    InitOnceExecuteOnce(&s_once, pause_init, NULL, NULL);
}

void xbox_HostPauseListener(void (*fn)(int))
{
    pause_once();
    EnterCriticalSection(&s_lock);
    if (fn && s_n_listeners < PAUSE_LISTENERS)
        s_listener[s_n_listeners++] = fn;
    LeaveCriticalSection(&s_lock);
}

int xbox_HostPaused(void)
{
    return s_paused != 0;
}

void xbox_HostPause(int on)
{
    int i, n, changed;
    void (*fns[PAUSE_LISTENERS])(int);

    pause_once();
    EnterCriticalSection(&s_lock);
    if (s_exiting && !on) {          /* the way out: no resume */
        LeaveCriticalSection(&s_lock);
        return;
    }
    changed = (s_paused != 0) != (on != 0);
    s_paused = on ? 1 : 0;
    n = s_n_listeners;
    for (i = 0; i < n; i++)
        fns[i] = s_listener[i];
    LeaveCriticalSection(&s_lock);
    if (!changed)
        return;
    /* Listeners outside the lock: the APU takes its own. */
    for (i = 0; i < n; i++)
        fns[i](on != 0);
    if (!on)
        WakeAllConditionVariable(&s_cv);
}

#define FLIP_RING 512
static volatile uint64_t s_flip_ns[FLIP_RING];
static volatile uint8_t s_flip_iv[FLIP_RING];  /* the swap interval each flip asked for */
static volatile uint32_t s_flip_n;       /* written by the executor only */
static volatile uint32_t s_next_iv = 1;  /* xbox_HostFlipInterval, for the next note */

/* The session's histogram: quarter-millisecond slots up to 200 ms, the last
 * slot everything longer. Written by the executor only; a reader may be one
 * flip behind. */
#define HIST_SLOTS 801
#define HIST_US    250u
static volatile uint32_t s_hist[HIST_SLOTS];
static volatile uint64_t s_hist_n, s_hist_sum_us, s_flips_all;
static volatile uint32_t s_hist_max_us, s_pauses;
static volatile uint64_t s_first_ns, s_last_ns, s_prev_ns;
static volatile uint32_t s_over[5];

static void flip_note(uint64_t ns)
{
    uint32_t n = s_flip_n;
    s_flip_ns[n % FLIP_RING] = ns;
    s_flip_iv[n % FLIP_RING] = (uint8_t)(ns ? s_next_iv : 0);
    s_flip_n = n + 1;
    if (!ns) {
        s_prev_ns = 0;          /* a pause: the next flip starts a new run */
        s_pauses++;
        return;
    }
    s_flips_all++;
    if (!s_first_ns)
        s_first_ns = ns;
    s_last_ns = ns;
    if (s_prev_ns && ns > s_prev_ns) {
        static const uint32_t over_us[5] = { 20000, 25000, 34000, 50000, 100000 };
        uint64_t us64 = (ns - s_prev_ns) / 1000u;
        uint32_t us = us64 > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)us64;
        uint32_t slot = us / HIST_US;
        int i;
        s_hist[slot < HIST_SLOTS ? slot : HIST_SLOTS - 1]++;
        s_hist_n++;
        s_hist_sum_us += us;
        if (us > s_hist_max_us)
            s_hist_max_us = us;
        for (i = 0; i < 5; i++)
            if (us > over_us[i])
                s_over[i]++;
    }
    s_prev_ns = ns;
}

/* The frame time below which `q` of the frames lie, from the histogram (a
 * slot's upper edge; the open last slot gives the maximum). */
static double hist_quantile(double q)
{
    uint64_t want = (uint64_t)(q * (double)s_hist_n), seen = 0;
    int i;
    for (i = 0; i < HIST_SLOTS; i++) {
        seen += s_hist[i];
        if (seen > want || (seen == s_hist_n && seen))
            return i == HIST_SLOTS - 1 ? s_hist_max_us / 1000.0
                                       : (double)(i + 1) * HIST_US / 1000.0;
    }
    return 0.0;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}

void xbox_HostFlipStats(xbox_flip_stats *out)
{
    static uint64_t ns[FLIP_RING];
    static uint32_t iv[FLIP_RING];
    uint32_t late_slot;
    int i, n, k = 0;

    memset(out, 0, sizeof *out);
    out->flips = s_flips_all;
    out->intervals = s_hist_n;
    out->pauses = s_pauses;
    if (!s_hist_n)
        return;
    out->span_s = (double)(s_last_ns - s_first_ns) / 1e9;
    out->mean_ms = (double)s_hist_sum_us / (double)s_hist_n / 1000.0;
    out->p50_ms = hist_quantile(0.50);
    out->p90_ms = hist_quantile(0.90);
    out->p95_ms = hist_quantile(0.95);
    out->p99_ms = hist_quantile(0.99);
    out->max_ms = s_hist_max_us / 1000.0;
    late_slot = (uint32_t)(out->p50_ms * 1.5 * 1000.0 / HIST_US);
    for (i = (int)late_slot; i < HIST_SLOTS; i++)
        out->late += s_hist[i];
    for (i = 0; i < 5; i++)
        out->over[i] = s_over[i];
    /* The flip ring: the last frames on their own (the scene on screen). */
    n = xbox_HostFlipTimes(ns, FLIP_RING);
    for (i = 1; i < n; i++)
        if (ns[i] && ns[i - 1] && ns[i] > ns[i - 1])
            iv[k++] = (uint32_t)((ns[i] - ns[i - 1]) / 1000u);
    if (k) {
        qsort(iv, (size_t)k, sizeof iv[0], cmp_u32);
        out->recent = (uint32_t)k;
        out->recent_p50_ms = iv[k / 2] / 1000.0;
        out->recent_p95_ms = iv[(int)((k - 1) * 0.95)] / 1000.0;
        out->recent_max_ms = iv[k - 1] / 1000.0;
    }
}

static void (*volatile s_hold_fn)(void);
void xbox_HostPauseHoldFn(void (*fn)(void)) { s_hold_fn = fn; }
static void (*volatile s_idle_fn)(void);
void xbox_HostPauseIdleFn(void (*fn)(void)) { s_idle_fn = fn; }

void xbox_HostFlipInterval(uint32_t iv)
{
    s_next_iv = iv ? (iv > 255u ? 255u : iv) : 1u;
}

uint32_t xbox_HostFlipIntervalAsked(void)
{
    return s_next_iv;
}

int xbox_HostFlipTimesIv(uint64_t *ns, uint8_t *iv, int max)
{
    uint32_t n = s_flip_n, have = n < FLIP_RING ? n : FLIP_RING, i, from;
    if (max <= 0)
        return 0;
    if (have > (uint32_t)max)
        have = (uint32_t)max;
    from = n - have;
    for (i = 0; i < have; i++) {
        ns[i] = s_flip_ns[(from + i) % FLIP_RING];
        iv[i] = s_flip_iv[(from + i) % FLIP_RING];
    }
    return (int)have;
}

int xbox_HostFlipTimes(uint64_t *ns, int max)
{
    uint32_t n = s_flip_n, have = n < FLIP_RING ? n : FLIP_RING, i, from;
    if (max <= 0)
        return 0;
    if (have > (uint32_t)max)
        have = (uint32_t)max;
    from = n - have;
    for (i = 0; i < have; i++)
        ns[i] = s_flip_ns[(from + i) % FLIP_RING];
    return (int)have;
}

void xbox_HostPauseHold(void)
{
    flip_note(xbox_HostNowNs());
    if (!s_paused)
        return;
    pause_once();
    EnterCriticalSection(&s_lock);
    s_held = 1;
    while (s_paused) {
        void (*fn)(void) = s_hold_fn, (*idle)(void) = s_idle_fn;
        if (s_exiting) {
            /* Held for the way out: the GPU device is being destroyed, so
             * no hold work (it may use the device) and no return. */
            SleepConditionVariableCS(&s_cv, &s_lock, PAUSE_TICK_MS);
            g_xbox_flip_count++;     /* not a hang, for the watchdog */
            continue;
        }
        if (fn || idle) {
            /* Outside the lock: it may take a while (a frame read back).
             * Not held meanwhile: the work may use the GPU. */
            s_held = 0;
            LeaveCriticalSection(&s_lock);
            if (fn)
                fn();
            if (idle)
                idle();
            EnterCriticalSection(&s_lock);
            s_held = 1;
            if (!s_paused)
                break;
        }
        SleepConditionVariableCS(&s_cv, &s_lock, PAUSE_TICK_MS);
        g_xbox_flip_count++;
    }
    s_held = 0;
    LeaveCriticalSection(&s_lock);
    flip_note(0);                /* a gap in the flip times */
}

int xbox_HostHoldForExit(int timeout_ms)
{
    int waited = 0;
    pause_once();
    EnterCriticalSection(&s_lock);
    s_exiting = 1;
    LeaveCriticalSection(&s_lock);
    xbox_HostPause(1);
    while (!s_held && waited < timeout_ms) {
        Sleep(5);
        waited += 5;
    }
    return s_held ? 1 : 0;
}

/* The window's way out (recomp_exit_hook.h, BO-219): the title's executor is
 * parked at its next frame before the GPU and the audio device are taken
 * down, so nothing of the title submits to a device that is going away. A
 * title that does not reach a frame in time (a load) is left running, as
 * before, and the log says so. The Android window holds it itself. */
#define QUIT_HOLD_MS 400

static void pause_quit_hook(int stage)
{
#if !defined(__ANDROID__)
    if (stage == RECOMP_QUIT_HOLD) {
        uint64_t t0 = xbox_HostNowNs();
        int held = xbox_HostHoldForExit(QUIT_HOLD_MS);
        double ms = (double)(xbox_HostNowNs() - t0) / 1e6;

        if (held)
            XR_NOTICE("[QUIT] the title's executor is held at a frame (%.1f ms)\n", ms);
        else
            XR_ERROR("[QUIT] the title did not reach a frame in %d ms: it runs on while the window closes\n",
                     QUIT_HOLD_MS);
    }
#else
    (void)stage;
#endif
}
