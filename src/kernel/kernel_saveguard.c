/*
 * kernel_saveguard.c - a title's save files are replaced, not overwritten.
 * See kernel_saveguard.h.
 */
#include "kernel.h"
#include "kernel_saveguard.h"
#include "recomp_exit_hook.h"
#include "recomp_save_dir.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <stdint.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define LOGP "[SAVE] "

/* ── host paths: bytes off Windows, wide there ─────────────────────── */

#if defined(_WIN32)
#define HC(s)          L##s
#define hlen           wcslen
#define hcpy(d, n, s)  _snwprintf((d), (n), L"%ls", (s))
#define hcmp           _wcsicmp
#define hncmp          _wcsnicmp
#define HSEP(c)        ((c) == L'\\' || (c) == L'/')
#else
#define HC(s)          s
#define hlen           strlen
#define hcpy(d, n, s)  snprintf((d), (n), "%s", (s))
#define hcmp           strcmp
#define hncmp          strncmp
#define HSEP(c)        ((c) == '/')
#endif

static const xbox_host_char k_suffix[] = HC(".~new");
#define SUFFIX_LEN 5
/* A new file that is whole and flushed but could not be renamed over the old
 * one (Windows: another program held the old file): "<name>.~pend". Kept,
 * tried again, and never removed unread (see "pending", below). */
static const xbox_host_char k_pend[] = HC(".~pend");
#define PEND_LEN 6

/* ── state ─────────────────────────────────────────────────────────── */

#define ROOTS    4
#define ENTRIES  16
#define HANDLES  4

typedef enum { E_FREE, E_OPEN, E_QUEUED, E_COMMITTING } entry_state;

typedef struct {
    entry_state    state;
    xbox_host_char real[MAX_PATH];
    xbox_host_char shadow[MAX_PATH];
    HANDLE         h[HANDLES];
    int            n_handles;
    int            dirty;
    int            deleted;           /* the title deleted it by handle */
    unsigned       writes;            /* the title's calls, for the log */
    unsigned long long bytes;
#if defined(_WIN32)
    HANDLE         flush;             /* a duplicate, to flush and close */
#else
    int            flush;
#endif
} guard_entry;

static xbox_host_char s_root[ROOTS][MAX_PATH];
static size_t         s_root_len[ROOTS];
static int            s_n_roots;
static guard_entry    s_entry[ENTRIES];
static void         (*s_hook)(void);
static volatile int   s_fault_at;
static int            s_fail_replace;     /* tests: replaces made to fail */

/* Pending: a save whose rename over the old file failed. Its new file is
 * kept as <name>.~pend; reads of the name see it, the next open for writing
 * starts from it, and the rename is tried again at the next save's close
 * and on the way out. After SG_TRIES failed tries the overlay is told
 * (xbox_SaveGuardProblem). What is still pending at the end stays on the
 * disk and the next start puts it in place (sweep_pend). */
#define SG_TRIES 4                        /* the failed one and three more */
typedef struct {
    int            used;
    int            tries;
    unsigned       code;                  /* the system's error of the last try */
    xbox_host_char real[MAX_PATH];
    xbox_host_char pend[MAX_PATH];
} pend_entry;
static pend_entry     s_pend[ENTRIES];
static int            s_retry_due;
static char           s_problem[400];
static int            s_problem_seq, s_problem_on;

#if defined(_WIN32)
static CRITICAL_SECTION   s_mu;
static CONDITION_VARIABLE s_cv;
#define LOCK()        EnterCriticalSection(&s_mu)
#define UNLOCK()      LeaveCriticalSection(&s_mu)
#define SIGNAL()      WakeAllConditionVariable(&s_cv)
#define WAIT(ms)      SleepConditionVariableCS(&s_cv, &s_mu, (ms))
#else
static pthread_mutex_t s_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  s_cv = PTHREAD_COND_INITIALIZER;
#define LOCK()        pthread_mutex_lock(&s_mu)
#define UNLOCK()      pthread_mutex_unlock(&s_mu)
#define SIGNAL()      pthread_cond_broadcast(&s_cv)
static void WAIT(unsigned ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    pthread_cond_timedwait(&s_cv, &s_mu, &ts);
}
#endif
static int s_ready, s_worker, s_lock_ready;

/* The process ends here when a test asked for it (kernel_saveguard.h). */
static void fault(int step)
{
    if (s_fault_at == step) {
        fflush(NULL);
        _exit(90 + step);
    }
}

void xbox_SaveGuardFaultAt(int step)
{
    s_fault_at = step;
}

static double now_ms(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static const xbox_host_char *leaf(const xbox_host_char *p)
{
    const xbox_host_char *l = p;
    for (; *p; p++)
        if (HSEP(*p))
            l = p + 1;
    return l;
}

#if defined(_WIN32)
#define LEAF_FMT "%ls"
#else
#define LEAF_FMT "%s"
#endif

/* ── files ─────────────────────────────────────────────────────────── */

static int h_exists(const xbox_host_char *p)
{
#if defined(_WIN32)
    DWORD a = GetFileAttributesW(p);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

static void h_unlink(const xbox_host_char *p)
{
#if defined(_WIN32)
    SetFileAttributesW(p, FILE_ATTRIBUTE_NORMAL);
    DeleteFileW(p);
#else
    unlink(p);
#endif
}

static FILE *h_fopen(const xbox_host_char *p, int write)
{
#if defined(_WIN32)
    /* A read shares everything, delete included: fopen does not share
     * deleting, so it cannot open a file a handle with DELETE access has open
     * (the title's save opens ask for it), and its open would make such an
     * open of the title fail meanwhile. */
    if (!write) {
        HANDLE h = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        int fd;
        FILE *f;
        if (h == INVALID_HANDLE_VALUE)
            return NULL;
        fd = _open_osfhandle((intptr_t)h, _O_RDONLY | _O_BINARY);
        if (fd < 0) {
            CloseHandle(h);
            return NULL;
        }
        f = _fdopen(fd, "rb");
        if (!f)
            _close(fd);
        return f;
    }
    return _wfopen(p, L"wb");
#else
    return fopen(p, write ? "wb" : "rb");
#endif
}

/* A whole copy; 0 and no file at `to` when it fails. */
static int h_copy(const xbox_host_char *from, const xbox_host_char *to)
{
    FILE *in = h_fopen(from, 0), *out;
    char buf[65536];
    size_t n;
    int ok = 1;

    if (!in)
        return 0;
    out = h_fopen(to, 1);
    if (!out) {
        fclose(in);
        return 0;
    }
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { ok = 0; break; }
    if (ferror(in))
        ok = 0;
    fclose(in);
    if (fclose(out) != 0)
        ok = 0;
    if (!ok)
        h_unlink(to);
    return ok;
}

/* 1 when both are there with the same bytes. */
static int h_same(const xbox_host_char *a, const xbox_host_char *b)
{
    FILE *fa = h_fopen(a, 0), *fb = h_fopen(b, 0);
    char ba[32768], bb[32768];
    int same = fa && fb;

    while (same) {
        size_t na = fread(ba, 1, sizeof ba, fa), nb = fread(bb, 1, sizeof bb, fb);
        if (na != nb || memcmp(ba, bb, na))
            same = 0;
        if (na < sizeof ba)
            break;
    }
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return same;
}

static int h_replace(const xbox_host_char *from, const xbox_host_char *to, unsigned *code)
{
#if defined(_WIN32)
    int tries;
    DWORD e = 0;
    for (tries = 0; tries < 25; tries++) {
        if (MoveFileExW(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return 1;
        e = GetLastError();
        Sleep(20);                    /* a reader has the old one open */
    }
    if (code)
        *code = (unsigned)e;
    return 0;
#else
    char dir[MAX_PATH];
    char *slash;
    int fd;

    if (rename(from, to) != 0) {
        if (code)
            *code = (unsigned)errno;
        return 0;
    }
    /* The name's change is in the folder: that goes to the disk too. */
    snprintf(dir, sizeof dir, "%s", to);
    slash = strrchr(dir, '/');
    if (slash && slash != dir) {
        *slash = 0;
        fd = open(dir, O_RDONLY);
        if (fd >= 0) {
            fsync(fd);
            close(fd);
        }
    }
    return 1;
#endif
}

/* The rename of a whole new save over the old one; the tests' fault hook
 * can make it fail (xbox_SaveGuardFailReplace). */
static int put_in_place(const xbox_host_char *from, const xbox_host_char *to, unsigned *code)
{
    if (s_fail_replace > 0) {
        s_fail_replace--;
#if defined(_WIN32)
        *code = ERROR_SHARING_VIOLATION;
#else
        *code = EBUSY;
#endif
        return 0;
    }
    return h_replace(from, to, code);
}

static long long h_mtime(const xbox_host_char *p)
{
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExW(p, GetFileExInfoStandard, &a))
        return -1;
    return ((long long)a.ftLastWriteTime.dwHighDateTime << 32) | a.ftLastWriteTime.dwLowDateTime;
#else
    struct stat st;
    if (stat(p, &st) != 0)
        return -1;
#if defined(__APPLE__)
    return (long long)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    return (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
#endif
#endif
}

/* The system's reason, in words a player can act on. */
static void reason_text(unsigned code, char *out, size_t n)
{
#if defined(_WIN32)
    switch (code) {
    case ERROR_ACCESS_DENIED:
        snprintf(out, n, "access was denied (another program, such as a virus scanner or a sync tool, "
                 "may hold the file)");
        break;
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
        snprintf(out, n, "another program has the save file open");
        break;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
        snprintf(out, n, "the disk is full");
        break;
    case ERROR_WRITE_PROTECT:
        snprintf(out, n, "the disk is write-protected");
        break;
    default:
        snprintf(out, n, "Windows error %u", code);
        break;
    }
#else
    snprintf(out, n, "%s", strerror((int)code));
#endif
}

/* ── which paths ───────────────────────────────────────────────────── */

static int guarded(const xbox_host_char *p)
{
    int i;
    for (i = 0; i < s_n_roots; i++)
        if (s_root_len[i] && !hncmp(p, s_root[i], s_root_len[i]) && HSEP(p[s_root_len[i]]))
            return 1;
    return 0;
}

int xbox_saveguard_hidden_name(const xbox_host_char *name)
{
    size_t n = hlen(name);
    return (n > SUFFIX_LEN && !hcmp(name + n - SUFFIX_LEN, k_suffix))
        || (n > PEND_LEN && !hcmp(name + n - PEND_LEN, k_pend));
}

/* ── pending saves (see s_pend) ────────────────────────────────────── */

/* With the lock held. */
static pend_entry *pend_of(const xbox_host_char *real)
{
    int i;
    for (i = 0; i < ENTRIES; i++)
        if (s_pend[i].used && !hcmp(s_pend[i].real, real))
            return &s_pend[i];
    return NULL;
}

/* The overlay's line, from the latest failure; with the lock held. */
static void problem_set(const pend_entry *p)
{
    char why[200];
    reason_text(p->code, why, sizeof why);
    snprintf(s_problem, sizeof s_problem, "A save could not be written: %s; your previous save is kept "
             "(the newer one waits beside it and is tried again).", why);
    s_problem_seq++;
    s_problem_on = 1;
    fprintf(stderr, LOGP LEAF_FMT ": %s\n", leaf(p->real), s_problem);
}

/* A pending save is done with (put in place, or overtaken by a newer one
 * that was); with the lock held. */
static void pend_drop(pend_entry *p, int remove_file)
{
    int i, left = 0;
    if (remove_file && hcmp(p->pend, p->real))
        h_unlink(p->pend);
    p->used = 0;
    for (i = 0; i < ENTRIES; i++)
        left |= s_pend[i].used && s_pend[i].tries >= SG_TRIES;
    if (!left)
        s_problem_on = 0;
}

/* The rename of `shadow` over `real` failed (code): the new file is kept as
 * <real>.~pend (or, when even that fails, where it is). Without the lock. */
static void pend_add(const xbox_host_char *real, const xbox_host_char *shadow, unsigned code)
{
    xbox_host_char pend[MAX_PATH];
    pend_entry *p;
    unsigned c2 = 0;
    int i;

    hcpy(pend, MAX_PATH, real);
    if (hlen(real) + PEND_LEN + 1 <= MAX_PATH)
        memcpy(pend + hlen(real), k_pend, sizeof k_pend);
    if (hlen(real) + PEND_LEN + 1 > MAX_PATH || !h_replace(shadow, pend, &c2)) {
        hcpy(pend, MAX_PATH, shadow);
        fprintf(stderr, LOGP LEAF_FMT ": kept as the new file beside it (it could not be renamed either)\n",
                leaf(real));
    }
    LOCK();
    p = pend_of(real);
    if (!p)
        for (i = 0; i < ENTRIES && !p; i++)
            if (!s_pend[i].used) {
                p = &s_pend[i];
                memset(p, 0, sizeof *p);
            }
    if (p) {
        if (p->used && hcmp(p->pend, pend) && hcmp(p->pend, p->real))
            h_unlink(p->pend);        /* an older pending one: this is newer */
        p->used = 1;
        hcpy(p->real, MAX_PATH, real);
        hcpy(p->pend, MAX_PATH, pend);
        p->tries++;
        p->code = code;
        if (p->tries >= SG_TRIES)
            problem_set(p);
    }
    UNLOCK();
    if (!p) {
        pend_entry tmp;
        hcpy(tmp.real, MAX_PATH, real);
        tmp.code = code;
        LOCK();
        problem_set(&tmp);
        UNLOCK();
    }
}

/* Try the pending renames again: those under SG_TRIES tries, or all of them
 * (the way out). With the lock held; a rename can take half a second on
 * Windows (h_replace), and only a save that failed pays it. */
static void pend_retry(int all)
{
    int i;
    for (i = 0; i < ENTRIES; i++) {
        pend_entry *p = &s_pend[i];
        unsigned code = 0;
        if (!p->used || (!all && p->tries >= SG_TRIES))
            continue;
        if (!h_exists(p->pend)) {
            pend_drop(p, 0);
            continue;
        }
        if (put_in_place(p->pend, p->real, &code)) {
            fprintf(stderr, LOGP LEAF_FMT ": the save that could not be put in place is in place now "
                    "(try %d)\n", leaf(p->real), p->tries + 1);
            pend_drop(p, 0);
            continue;
        }
        p->tries++;
        p->code = code;
        if (p->tries == SG_TRIES)
            problem_set(p);
    }
}

static guard_entry *entry_of_path(const xbox_host_char *real)
{
    int i;
    for (i = 0; i < ENTRIES; i++)
        if (s_entry[i].state != E_FREE && !hcmp(s_entry[i].real, real))
            return &s_entry[i];
    return NULL;
}

static guard_entry *entry_of_handle(HANDLE h, int *slot)
{
    int i, k;
    for (i = 0; i < ENTRIES; i++)
        if (s_entry[i].state == E_OPEN)
            for (k = 0; k < s_entry[i].n_handles; k++)
                if (s_entry[i].h[k] == h) {
                    if (slot) *slot = k;
                    return &s_entry[i];
                }
    return NULL;
}

/* ── the commit, on a thread of its own ────────────────────────────── */

/* Flush what the title wrote, then put it in the real file's place: the
 * rename is the one moment the save changes, and it changes whole. */
/* A newer whole save of `real` is in place (or it was deleted): what was
 * pending for it is overtaken. Without the lock. */
static void pend_done(const xbox_host_char *real)
{
    pend_entry *p;
    LOCK();
    p = pend_of(real);
    if (p)
        pend_drop(p, 1);
    UNLOCK();
}

static void commit(guard_entry *e)
{
    double t0 = now_ms(), t_flush;
    unsigned code = 0;
    int changed;

    fault(XBOX_SAVE_FAULT_BEFORE_FLUSH);
#if defined(_WIN32)
    if (e->flush && e->flush != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(e->flush);
        CloseHandle(e->flush);
    }
#else
    if (e->flush >= 0) {
#if defined(__APPLE__) && defined(F_FULLFSYNC)
        /* fsync on macOS stops at the drive's cache; this goes through it. */
        if (fcntl(e->flush, F_FULLFSYNC) != 0)
#endif
            fsync(e->flush);
        close(e->flush);
    }
#endif
    t_flush = now_ms() - t0;
    fault(XBOX_SAVE_FAULT_BEFORE_REPLACE);
    if (!h_exists(e->shadow)) {
        /* The title removed it through its handle (delete on close), or
         * renamed it away. */
        if (e->deleted) {
            h_unlink(e->real);
            pend_done(e->real);
            fprintf(stderr, LOGP LEAF_FMT ": deleted by the title\n", leaf(e->real));
        }
        return;
    }
    changed = !h_exists(e->real) || !h_same(e->shadow, e->real);
    if (!changed) {
        /* The same bytes again (a title rewrites its meta file at every
         * save): nothing to replace, nothing to keep a copy of. */
        h_unlink(e->shadow);
        pend_done(e->real);
        if (e->writes)                /* opened for writing and not written: not worth a line */
            fprintf(stderr, LOGP LEAF_FMT ": written with the same content (%u writes, %llu bytes), left as it is\n",
                leaf(e->real), e->writes, e->bytes);
        return;
    }
    /* The port's moment to keep the save as it still is (the history). */
    if (s_hook)
        s_hook();
    if (!put_in_place(e->shadow, e->real, &code)) {
        char why[200];
        reason_text(code, why, sizeof why);
        xbox_log(XBOX_LOG_ERROR, XBOX_LOG_FILE, "save: the new file could not replace the old one; "
                 "both are kept");
        fprintf(stderr, LOGP LEAF_FMT ": the new file could not be put in place (%s); the previous one "
                "stays, the new one is kept beside it and tried again\n", leaf(e->real), why);
        /* Never thrown away: kept, tried again at the next close and on the
         * way out, put in place by the next start at the latest. */
        pend_add(e->real, e->shadow, code);
        return;
    }
    /* A newer save is in place: one that waited is overtaken. */
    pend_done(e->real);
    fprintf(stderr, LOGP LEAF_FMT ": replaced (%u writes, %llu bytes; flush %.1f ms, all %.1f ms)\n",
            leaf(e->real), e->writes, e->bytes, t_flush, now_ms() - t0);
    fault(XBOX_SAVE_FAULT_AFTER_REPLACE);
}

#if defined(_WIN32)
static unsigned __stdcall worker(void *arg)
#else
static void *worker(void *arg)
#endif
{
    (void)arg;
    LOCK();
    for (;;) {
        guard_entry *e = NULL;
        int i;
        for (i = 0; i < ENTRIES; i++)
            if (s_entry[i].state == E_QUEUED) { e = &s_entry[i]; break; }
        if (!e && s_retry_due) {
            /* A save was closed after one that could not be put in place:
             * the pending ones are tried again, after that save. */
            s_retry_due = 0;
            pend_retry(0);
            SIGNAL();
            continue;
        }
        if (!e) {
            WAIT(1000);
            continue;
        }
        e->state = E_COMMITTING;
        UNLOCK();
        commit(e);
        LOCK();
        e->state = E_FREE;
        SIGNAL();
    }
    return 0;
}

static void start_worker(void)
{
    if (s_worker)
        return;
    s_worker = 1;
#if defined(_WIN32)
    {
        HANDLE t = (HANDLE)_beginthreadex(NULL, 0, worker, NULL, 0, NULL);
        if (t) CloseHandle(t); else s_worker = 0;
    }
#else
    {
        pthread_t t;
        if (pthread_create(&t, NULL, worker, NULL) == 0) pthread_detach(t); else s_worker = 0;
    }
#endif
}

/* ── the file layer's hooks ────────────────────────────────────────── */

/* Until nothing of `real` is waiting to be put in place (a read right after
 * a save must see the save). Called with the lock held. */
static void wait_committed(const xbox_host_char *real)
{
    int tries = 0;
    for (;;) {
        guard_entry *e = entry_of_path(real);
        if (!e || e->state == E_OPEN || ++tries > 50)
            return;
        WAIT(100);
    }
}

void *xbox_saveguard_open(xbox_host_char *host_path, unsigned n, int write_access, int directory)
{
    guard_entry *e;
    int i;

    if (!s_ready || directory || !guarded(host_path) || xbox_saveguard_hidden_name(host_path))
        return NULL;
    LOCK();
    wait_committed(host_path);
    if (!write_access) {
        /* A newer save waits beside the file: a read sees that one. */
        pend_entry *p = pend_of(host_path);
        if (p && hlen(p->pend) + 1 <= n && h_exists(p->pend))
            hcpy(host_path, n, p->pend);
        UNLOCK();
        return NULL;
    }
    e = entry_of_path(host_path);
    if (e && e->state == E_OPEN && e->n_handles < HANDLES) {
        /* A second handle on a file being written: the same new file. */
        e->n_handles++;
        e->h[e->n_handles - 1] = NULL;
        hcpy(host_path, n, e->shadow);
        UNLOCK();
        return e;
    }
    if (e) {                          /* no room to share it: in place, as before */
        UNLOCK();
        return NULL;
    }
    for (i = 0; i < ENTRIES && s_entry[i].state != E_FREE; i++)
        ;
    if (i == ENTRIES || hlen(host_path) + SUFFIX_LEN + 1 > MAX_PATH || hlen(host_path) + SUFFIX_LEN + 1 > n) {
        UNLOCK();
        return NULL;
    }
    e = &s_entry[i];
    memset(e, 0, sizeof *e);
    hcpy(e->real, MAX_PATH, host_path);
    hcpy(e->shadow, MAX_PATH, host_path);
    memcpy(e->shadow + hlen(e->shadow), k_suffix, sizeof k_suffix);
    /* The new file starts as the old one, so an open that keeps what is
     * there, and a write of a part, see the bytes they expect; the newest
     * one: a save that waits to be put in place (pending) when there is. */
    {
        pend_entry *p = pend_of(e->real);
        if (p && !hcmp(p->pend, e->shadow)) {
            /* Kept under the new file's own name: it is the start already. */
        } else {
            h_unlink(e->shadow);
            if (p && h_exists(p->pend) && h_copy(p->pend, e->shadow)) {
                /* from the pending one */
            } else if (h_exists(e->real) && !h_copy(e->real, e->shadow)) {
                UNLOCK();
                return NULL;          /* no room for a second copy: in place */
            }
        }
    }
    e->state = E_OPEN;
    e->n_handles = 1;
    e->h[0] = NULL;
    hcpy(host_path, n, e->shadow);
    UNLOCK();
    fault(XBOX_SAVE_FAULT_AFTER_OPEN);
    return e;
}

void xbox_saveguard_opened(void *token, HANDLE h, int ok)
{
    guard_entry *e = (guard_entry *)token;
    int k;

    if (!e)
        return;
    LOCK();
    for (k = 0; k < e->n_handles && e->h[k]; k++)
        ;
    if (ok && k < e->n_handles) {
        e->h[k] = h;
    } else {
        /* It did not open (no such file, a name collision): as if never asked. */
        if (k < e->n_handles)
            e->h[k] = e->h[e->n_handles - 1];
        e->n_handles--;
        if (e->n_handles <= 0) {
            h_unlink(e->shadow);
            e->state = E_FREE;
            SIGNAL();
        }
    }
    UNLOCK();
}

void xbox_saveguard_wrote(HANDLE h, unsigned bytes)
{
    guard_entry *e;
    int i, any = 0;

    if (!s_ready)
        return;
    for (i = 0; i < ENTRIES; i++)       /* unlocked glance: usually nothing is open */
        any |= s_entry[i].state == E_OPEN;
    if (!any)
        return;
    LOCK();
    e = entry_of_handle(h, NULL);
    if (e) {
        e->dirty = 1;
        e->writes++;
        e->bytes += bytes;
    }
    UNLOCK();
    if (e)
        fault(XBOX_SAVE_FAULT_AFTER_WRITE);
}

void xbox_saveguard_deleted(HANDLE h)
{
    guard_entry *e;

    if (!s_ready)
        return;
    LOCK();
    e = entry_of_handle(h, NULL);
    if (e)
        e->deleted = 1;
    UNLOCK();
}

void xbox_saveguard_settle(const xbox_host_char *host_path)
{
    int i, tries = 0, busy;

    if (!s_ready)
        return;
    for (i = 0, busy = 0; i < ENTRIES; i++)   /* unlocked glance, as above */
        busy |= s_entry[i].state == E_QUEUED || s_entry[i].state == E_COMMITTING;
    if (!busy || !host_path)
        return;
    LOCK();
    for (;;) {
        for (i = 0, busy = 0; i < ENTRIES; i++)
            busy |= s_entry[i].state == E_QUEUED || s_entry[i].state == E_COMMITTING;
        if (!busy || ++tries > 50)
            break;
        WAIT(100);
    }
    UNLOCK();
}

void xbox_saveguard_close(HANDLE h)
{
    guard_entry *e;
    int slot = 0;

    if (!s_ready)
        return;
    LOCK();
    e = entry_of_handle(h, &slot);
    if (!e) {
        UNLOCK();
        return;
    }
    e->h[slot] = e->h[e->n_handles - 1];
    e->n_handles--;
    if (e->n_handles > 0) {
        UNLOCK();
        return;
    }
    /* The last handle: a duplicate stays with the commit, to flush. The
     * caller closes the title's handle as it always did. */
#if defined(_WIN32)
    if (!DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &e->flush, 0, FALSE,
                         DUPLICATE_SAME_ACCESS))
        e->flush = NULL;
#else
    e->flush = dup(w32_handle_fd(h));
#endif
    e->state = E_QUEUED;
    {
        int i;
        for (i = 0; i < ENTRIES; i++)
            if (s_pend[i].used && s_pend[i].tries < SG_TRIES)
                s_retry_due = 1;
    }
    start_worker();
    if (!s_worker) {                  /* no thread: here and now */
        e->state = E_COMMITTING;
        UNLOCK();
        commit(e);
        LOCK();
        e->state = E_FREE;
        if (s_retry_due) {
            s_retry_due = 0;
            pend_retry(0);
        }
    }
    SIGNAL();
    UNLOCK();
}

/* ── the port's side ───────────────────────────────────────────────── */

int xbox_SaveGuardBusy(void)
{
    int i, n = 0;
    if (!s_ready)
        return 0;
    LOCK();
    for (i = 0; i < ENTRIES; i++)
        if (s_entry[i].state == E_QUEUED || s_entry[i].state == E_COMMITTING
            || (s_entry[i].state == E_OPEN && s_entry[i].dirty))
            n++;
    UNLOCK();
    return n;
}

int xbox_SaveGuardWait(unsigned ms)
{
    unsigned waited = 0;
    while (xbox_SaveGuardBusy() && waited < ms) {
        LOCK();
        WAIT(20);
        UNLOCK();
        waited += 20;
    }
    return !xbox_SaveGuardBusy();
}

void xbox_SaveGuardSetHook(void (*before_replace)(void))
{
    s_hook = before_replace;
}

int xbox_SaveGuardProblem(char *out, unsigned n)
{
    int seq = 0;
    if (out && n)
        out[0] = 0;
    if (!s_lock_ready)
        return 0;
    LOCK();
    if (s_problem_on) {
        seq = s_problem_seq;
        if (out && n)
            snprintf(out, n, "%s", s_problem);
    }
    UNLOCK();
    return seq;
}

int xbox_SaveGuardPending(void)
{
    int i, k = 0;
    if (!s_lock_ready)
        return 0;
    LOCK();
    for (i = 0; i < ENTRIES; i++)
        k += s_pend[i].used;
    UNLOCK();
    return k;
}

void xbox_SaveGuardFailReplace(int n)
{
    s_fail_replace = n;
}

/* A save that an earlier run could not put in place (<name>.~pend: whole
 * and flushed; the rename over the old file failed to the end). Never just
 * removed: when it is newer than the file beside it, it is put in place;
 * when that fails, or it is older, the save with it in place is kept in the
 * save history first (a whole save, restorable from the overlay), and a
 * newer one that still cannot be put in place stays pending for this run.
 * At enable, single-threaded. */
static const char *s_title_id;

static void keep_aside(const xbox_host_char *pend, const xbox_host_char *real, char *made, size_t made_n)
{
    char root8[MAX_PATH * 3], out[MAX_PATH * 3], err[600], rel8[MAX_PATH * 3];
    xbox_host_char hroot[MAX_PATH], dest[MAX_PATH];
    const char *r = xbox_PathUserRoot();
    size_t rl;

    made[0] = 0;
    if (!r || !*r || !s_title_id)
        return;
    snprintf(root8, sizeof root8, "%s", r);
#if defined(_WIN32)
    if (!MultiByteToWideChar(CP_UTF8, 0, root8, -1, hroot, MAX_PATH))
        return;
#else
    snprintf(hroot, sizeof hroot, "%s", root8);
#endif
    rl = hlen(hroot);
    while (rl > 1 && HSEP(hroot[rl - 1]))
        hroot[--rl] = 0;
    if (hncmp(real, hroot, rl) || !HSEP(real[rl]))
        return;
    {
        char hist[MAX_PATH * 3];
        snprintf(hist, sizeof hist, "%s/save-history", root8);
        if (!recomp_save_export(root8, s_title_id, hist,
                                "kept at the start: a newer save file an earlier run could not put in place "
                                "is in it", out, sizeof out, err, sizeof err)) {
            fprintf(stderr, LOGP "the waiting save could not be kept in the history: %s\n", err);
            return;
        }
    }
#if defined(_WIN32)
    if (!WideCharToMultiByte(CP_UTF8, 0, real + rl + 1, -1, rel8, (int)sizeof rel8, NULL, NULL))
        return;
    {
        char d8[MAX_PATH * 3];
        snprintf(d8, sizeof d8, "%s/%s", out, rel8);
        if (!MultiByteToWideChar(CP_UTF8, 0, d8, -1, dest, MAX_PATH))
            return;
    }
#else
    snprintf(rel8, sizeof rel8, "%s", real + rl + 1);
    snprintf(dest, sizeof dest, "%s/%s", out, rel8);
#endif
    if (h_copy(pend, dest))
        snprintf(made, made_n, "%s", out);
}

static void sweep_pend(const xbox_host_char *pend)
{
    xbox_host_char real[MAX_PATH];
    char made[MAX_PATH * 3];
    unsigned code = 0;
    int newer;
    size_t n = hlen(pend);

    hcpy(real, MAX_PATH, pend);
    real[n - PEND_LEN] = 0;
    newer = !h_exists(real) || h_mtime(pend) >= h_mtime(real);
    if (newer && put_in_place(pend, real, &code)) {
        fprintf(stderr, LOGP LEAF_FMT ": a save an earlier run could not put in place is in place now\n",
                leaf(real));
        return;
    }
    keep_aside(pend, real, made, sizeof made);
    if (newer) {
        pend_entry *p = NULL;
        int i;
        LOCK();
        for (i = 0; i < ENTRIES && !p; i++)
            if (!s_pend[i].used)
                p = &s_pend[i];
        if (p) {
            memset(p, 0, sizeof *p);
            p->used = 1;
            p->tries = SG_TRIES - 1;      /* one more try at the next save's close */
            p->code = code;
            hcpy(p->real, MAX_PATH, real);
            hcpy(p->pend, MAX_PATH, pend);
            problem_set(p);
        }
        UNLOCK();
        fprintf(stderr, LOGP LEAF_FMT ": a save an earlier run could not put in place still cannot be "
                "(kept beside it%s%s)\n", leaf(real), made[0] ? " and in " : "", made);
    } else if (made[0]) {
        h_unlink(pend);
        fprintf(stderr, LOGP LEAF_FMT ": an older save file an earlier run could not put in place was "
                "moved to %s\n", leaf(real), made);
    } else {
        fprintf(stderr, LOGP LEAF_FMT ": an older save file an earlier run could not put in place is "
                "left beside it (it could not be kept in the history)\n", leaf(real));
    }
}

static int is_pend_name(const xbox_host_char *name)
{
    size_t n = hlen(name);
    return n > PEND_LEN && !hcmp(name + n - PEND_LEN, k_pend);
}

/* A new file that never took its place (the process ended first) is not a
 * save: the old file beside it is. A pending one (whole) is: sweep_pend. */
static void sweep(const xbox_host_char *dir, int depth)
{
#if defined(_WIN32)
    WCHAR pat[MAX_PATH], path[MAX_PATH];
    WIN32_FIND_DATAW fd;
    HANDLE h;
    if (_snwprintf(pat, MAX_PATH, L"%ls\\*", dir) < 0)
        return;
    h = FindFirstFileW(pat, &fd);
    if (h == INVALID_HANDLE_VALUE)
        return;
    do {
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L".."))
            continue;
        if (_snwprintf(path, MAX_PATH, L"%ls\\%ls", dir, fd.cFileName) < 0)
            continue;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (depth < 6 && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
                sweep(path, depth + 1);
        } else if (is_pend_name(fd.cFileName)) {
            sweep_pend(path);
        } else if (xbox_saveguard_hidden_name(fd.cFileName)) {
            fprintf(stderr, LOGP "an unfinished save write was found and removed; the save before it stands\n");
            h_unlink(path);
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    struct dirent *e;
    char path[MAX_PATH];
    if (!d)
        return;
    while ((e = readdir(d)) != NULL) {
        struct stat st;
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
            continue;
        if ((size_t)snprintf(path, sizeof path, "%s/%s", dir, e->d_name) >= sizeof path
            || lstat(path, &st) != 0)
            continue;
        if (S_ISDIR(st.st_mode)) {
            if (depth < 6)
                sweep(path, depth + 1);
        } else if (S_ISREG(st.st_mode) && is_pend_name(e->d_name)) {
            sweep_pend(path);
        } else if (S_ISREG(st.st_mode) && xbox_saveguard_hidden_name(e->d_name)) {
            fprintf(stderr, LOGP "an unfinished save write was found and removed; the save before it stands\n");
            h_unlink(path);
        }
    }
    closedir(d);
#endif
}

/* Every way out: exit() (the window closed, a restart, the title's own
 * reboot) runs this before the restart's exec, which was registered first;
 * ExitProcess on the Win32 windows runs the exit hooks. A save still on its
 * way gets up to two seconds. */
static void wait_at_exit(void)
{
    int i, left = 0;
    if (xbox_SaveGuardBusy()) {
        double t0 = now_ms();
        int done = xbox_SaveGuardWait(2000);
        fprintf(stderr, LOGP "the way out waited %.0f ms for a save to reach the disk%s\n",
                now_ms() - t0, done ? "" : "; it was not done in time, the save before it stands");
        fflush(stderr);
    }
    /* A save that could not be put in place: a last try. */
    LOCK();
    pend_retry(1);
    for (i = 0; i < ENTRIES; i++)
        left += s_pend[i].used;
    UNLOCK();
    if (left) {
        fprintf(stderr, LOGP "%d save file%s could not be put in place; the newer version is kept beside "
                "the old one and is put in place at the next start\n", left, left == 1 ? "" : "s");
        fflush(stderr);
    }
}

static void wait_exit_hook(const char *why)
{
    (void)why;
    wait_at_exit();
}

/* The window's way out (recomp_exit_hook.h, BO-219): the title is still
 * running when the window loop ends, so a save on its way can finish now,
 * before the title's executor is held for the teardown. The wait is the one
 * the exit does later, moved up; this one is done and the later finds the
 * guard idle. */
static void wait_quit_hook(int stage)
{
    if (stage == RECOMP_QUIT_SETTLE && xbox_SaveGuardBusy())
        wait_at_exit();
}

void xbox_SaveGuardEnable(const char *title_id)
{
    static const char *const part[] = { "UDATA", "TDATA" };
    char guest[160];
    int i;

    if (!title_id || !*title_id)
        return;
    if (!s_lock_ready) {
#if defined(_WIN32)
        InitializeCriticalSection(&s_mu);
        InitializeConditionVariable(&s_cv);
#endif
        s_lock_ready = 1;
    }
    {
        static char id[16];
        snprintf(id, sizeof id, "%s", title_id);
        s_title_id = id;
    }
    for (i = 0; i < 2; i++) {
        xbox_host_char *r = s_root[s_n_roots < ROOTS ? s_n_roots : ROOTS - 1];
        size_t n;
        snprintf(guest, sizeof guest, "\\Device\\Harddisk0\\Partition1\\%s\\%s", part[i], title_id);
        if (!xbox_translate_path(guest, r, MAX_PATH))
            continue;
        int k, again = 0;
        n = hlen(r);
        while (n > 1 && HSEP(r[n - 1]))
            r[--n] = 0;
        for (k = 0; k < s_n_roots; k++)       /* enabled again: the sweep only */
            again |= s_root_len[k] == n && !hcmp(s_root[k], r);
        sweep(r, 0);
        if (!again && s_n_roots + 1 < ROOTS) {
            s_root_len[s_n_roots] = n;
            s_n_roots++;
        }
    }
    if (!s_ready && s_n_roots > 0) {
        atexit(wait_at_exit);
        recomp_exit_hook_add(wait_exit_hook);
        recomp_quit_hook_add(wait_quit_hook);
    }
    s_ready = s_n_roots > 0;
}
