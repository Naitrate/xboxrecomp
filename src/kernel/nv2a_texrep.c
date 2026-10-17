/**
 * nv2a_texrep.c - regions of a title's textures replaced at upload.
 * See nv2a_texrep.h.
 */
#include "nv2a_texrep.h"
#include "nv2a_backend_common.h"
#include "recomp_env.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOGP "[TEXREP] "

static const struct nv2a_texrep_tex *s_tex;
static unsigned                      s_n;
static const struct nv2a_texrep_tex *s_ht;      /* BO-203: pictures used with HDR on */
static unsigned                      s_hn;
static _Atomic int                   s_hdr_on;
static _Atomic int                   s_variant;
static _Atomic uint32_t              s_gen = 1;
static _Atomic uint64_t              s_hashed, s_matched, s_replaced;

static void gen_next(void)
{
    /* Never 0: 0 is "not a registered texture" in a backend's cache. */
    if (atomic_fetch_add_explicit(&s_gen, 1, memory_order_relaxed) == 0xFFFFFFFFu)
        atomic_store_explicit(&s_gen, 1, memory_order_relaxed);
}

void nv2a_texrep_register(const struct nv2a_texrep_tex *tex, unsigned n)
{
    s_tex = n ? tex : NULL;
    s_n = tex ? n : 0;
    gen_next();
}

void nv2a_texrep_register_hdr_image(const struct nv2a_texrep_tex *tex, unsigned n)
{
    /* RECOMP_DEBUG=hdr_image=0: as if the title had registered none. */
    const char *e = recomp_env(RENV_HDR_IMAGE);
    if (e && !strcmp(e, "0"))
        n = 0;
    s_ht = n ? tex : NULL;
    s_hn = tex ? n : 0;
    gen_next();
}

int nv2a_texrep_hdr_stage0_only(const uint8_t *guest, uint32_t color, uint32_t w,
                                uint32_t h, uint32_t pitch)
{
    const struct nv2a_texrep_tex *ht = s_ht;
    unsigned i, hn = s_hn;
    uint64_t hash = 0;

    for (i = 0; i < hn; i++) {
        if (ht[i].color != color || ht[i].width != w || ht[i].height != h
                || ht[i].n_variants < 1 || !ht[i].variants[0].stage0_only)
            continue;
        if (!hash)
            hash = nv2a_texrep_hash(guest, color, w, h, pitch);
        if (hash && hash == ht[i].hash)
            return 1;
    }
    return 0;
}

static _Atomic int s_ab_off;      /* hdr_image_ab: the pictures are off in this block */

void nv2a_texrep_set_hdr(int on)
{
    on = on != 0;
    if (atomic_exchange_explicit(&s_hdr_on, on, memory_order_relaxed) != on)
        gen_next();
}

int nv2a_texrep_hdr(void)
{
    return atomic_load_explicit(&s_hdr_on, memory_order_relaxed)
        && !atomic_load_explicit(&s_ab_off, memory_order_relaxed);
}

void nv2a_texrep_flip(uint32_t flip)
{
    static int n = -1;
    int off;
    if (n < 0) {
        const char *e = recomp_env(RENV_HDR_IMAGE_AB);
        n = e ? atoi(e) : 0;
        if (n < 0)
            n = 0;
    }
    if (!n)
        return;
    off = ((flip / (uint32_t)n) & 1u) != 0;
    if (atomic_exchange_explicit(&s_ab_off, off, memory_order_relaxed) != off)
        gen_next();
}

void nv2a_texrep_set_variant(int v)
{
    if (v < 0)
        v = 0;
    if (atomic_exchange_explicit(&s_variant, v, memory_order_relaxed) != v)
        gen_next();
}

int nv2a_texrep_variant(void)
{
    return atomic_load_explicit(&s_variant, memory_order_relaxed);
}

uint32_t nv2a_texrep_generation(void)
{
    return atomic_load_explicit(&s_gen, memory_order_relaxed);
}

uint64_t nv2a_texrep_hash(const uint8_t *guest, uint32_t color, uint32_t w,
                          uint32_t h, uint32_t pitch)
{
    uint32_t n = nv2a_tex_extent(color, w, h, pitch, 1);
    return n ? nv2a_tex_hash(guest, n, NV2A_TEX_HASH_SEED) : 0;
}

void nv2a_texrep_stats(uint64_t *hashed, uint64_t *matched, uint64_t *replaced)
{
    if (hashed)
        *hashed = atomic_load_explicit(&s_hashed, memory_order_relaxed);
    if (matched)
        *matched = atomic_load_explicit(&s_matched, memory_order_relaxed);
    if (replaced)
        *replaced = atomic_load_explicit(&s_replaced, memory_order_relaxed);
}

/* ── depth offsets (BO-176) ───────────────────────────────────────────── */

static const struct nv2a_texrep_zbias *s_zb;
static unsigned                        s_zb_n;

void nv2a_texrep_register_zbias(const struct nv2a_texrep_zbias *tex, unsigned n)
{
    s_zb = n ? tex : NULL;
    s_zb_n = tex ? n : 0;
}

int nv2a_texrep_zbias_on(uint32_t flip)
{
    static int mode = -1;               /* -1 not read, 0 off, 1 on */
    static uint32_t ab;
    if (mode < 0) {
        const char *e = recomp_env(RENV_ZBIAS), *a = recomp_env(RENV_ZBIAS_AB);
        ab = a ? (uint32_t)strtoul(a, NULL, 0) : 0;
        mode = !(e && (!strcmp(e, "0") || !strcmp(e, "off")));
    }
    if (mode && ab) {
        static uint32_t said = 0xFFFFFFFFu;
        uint32_t f = flip, half = (f / ab) & 1u;
        if (half != said) {
            said = half;
            fprintf(stderr, LOGP "zbias ab: %s from flip %u\n", half ? "off" : "on", f);
        }
        return !half;
    }
    return mode;
}

/* RECOMP_DEBUG=zbias_add=<hash>[+<hash>...] (BO-204): textures that take
 * -512 as if the title had registered them, for trying a candidate. */
static uint64_t s_za[64];
static int      s_za_n = -1;
static int hash_list(const char *e, uint64_t *out, int max);

float nv2a_texrep_zbias(const uint8_t *guest, uint32_t color, uint32_t w,
                        uint32_t h, uint32_t pitch)
{
    uint64_t hash = 0;
    unsigned i;

    if (s_za_n < 0) {
        s_za_n = hash_list(recomp_env(RENV_ZBIAS_ADD), s_za, 64);
        if (s_za_n)
            fprintf(stderr, LOGP "zbias_add: %d texture(s) at -512\n", s_za_n);
    }
    if (s_za_n > 0) {
        int k;
        hash = nv2a_texrep_hash(guest, color, w, h, pitch);
        for (k = 0; k < s_za_n; k++)
            if (s_za[k] == hash)
                return -512.0f;
    }

    for (i = 0; i < s_zb_n; i++) {
        const struct nv2a_texrep_zbias *z = &s_zb[i];
        if (z->color != color || z->width != w || z->height != h)
            continue;
        if (!hash)
            hash = nv2a_texrep_hash(guest, color, w, h, pitch);
        if (hash == z->hash) {
            static _Atomic int said;
            if (!atomic_exchange_explicit(&said, 1, memory_order_relaxed))
                fprintf(stderr, LOGP "depth offset %g for %s\n", (double)z->bias,
                        z->name ? z->name : "?");
            return z->bias;
        }
    }
    return 0.0f;
}

/* ── finding coplanar decals (BO-204, debug) ─────────────────────────── */

#define ZP_MAX 4096
#define ZP_WIN 32
static uint32_t s_zp_win[ZP_WIN];
static int      s_zp_nwin = -1, s_zp_cur = -1;
static uint64_t s_zp_hash[ZP_MAX];
static unsigned s_zp_n;
static int      s_zp_said = -1, s_zp_done;
#define ZT_MAX 64
static uint64_t s_zt[ZT_MAX];
static int      s_zt_n = -1;

static int hash_list(const char *e, uint64_t *out, int max)
{
    int n = 0;
    while (e && *e && n < max) {
        char *end;
        uint64_t v = strtoull(e, &end, 16);
        if (end == e)
            break;
        out[n++] = v;
        e = *end ? end + 1 : end;
    }
    return n;
}

static const char *s_zp_file;      /* zbias_probe=@file: the windows, read again as it grows */
static long        s_zp_fsize = -1;

static void zp_parse(const char *e)
{
    s_zp_nwin = 0;
    while (e && *e && s_zp_nwin < ZP_WIN) {
        char *end;
        unsigned long v = strtoul(e, &end, 0);
        if (end == e)
            break;
        s_zp_win[s_zp_nwin++] = (uint32_t)v;
        e = *end ? end + 1 : end;
    }
}

static void zp_reread(void)
{
    char buf[1024];
    size_t n;
    long sz;
    FILE *f = fopen(s_zp_file, "rb");
    if (!f)
        return;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    if (sz != s_zp_fsize) {
        s_zp_fsize = sz;
        fseek(f, 0, SEEK_SET);
        n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        for (size_t i = 0; i < n; i++)
            if (buf[i] == '\n' || buf[i] == ' ' || buf[i] == ',')
                buf[i] = '+';
        zp_parse(buf);
        fprintf(stderr, LOGP "zprobe: %d window(s) from %s\n", s_zp_nwin, s_zp_file);
    }
    fclose(f);
}

int nv2a_texrep_zdebug(void)
{
    if (s_zp_nwin < 0) {
        const char *e = recomp_env(RENV_ZBIAS_PROBE);
        if (e && e[0] == '@') {
            s_zp_file = e + 1;
            s_zp_nwin = 0;
        } else {
            zp_parse(e);
        }
        s_zt_n = hash_list(recomp_env(RENV_ZTINT), s_zt, ZT_MAX);
        if (s_zp_nwin || s_zt_n || s_zp_file)
            fprintf(stderr, LOGP "zprobe: %d window(s), %d tinted texture(s)\n", s_zp_nwin,
                    s_zt_n);
    }
    return s_zp_nwin > 0 || s_zt_n > 0 || s_zp_file;
}

int nv2a_texrep_ztint(uint64_t hash)
{
    int i;
    if (!nv2a_texrep_zdebug())
        return 0;
    for (i = 0; i < s_zt_n; i++)
        if (s_zt[i] == hash)
            return 1;
    return 0;
}

uint64_t nv2a_texrep_ztint_apply(const uint8_t *guest, uint32_t color, uint32_t w,
                                 uint32_t h, uint32_t pitch, uint32_t *texels, size_t n)
{
    uint64_t hash;
    size_t i;
    if (!nv2a_texrep_zdebug())
        return 0;
    hash = nv2a_texrep_hash(guest, color, w, h, pitch);
    if (nv2a_texrep_ztint(hash))
        for (i = 0; i < n; i++)
            texels[i] = (texels[i] & 0xFF000000u) | 0x00FF00FFu;
    return hash;
}

float nv2a_texrep_zprobe(uint32_t flip, uint64_t hash, uint32_t color, uint32_t w,
                         uint32_t h)
{
    int i, win = -1;
    uint32_t rel, k;
    unsigned j;

    static uint32_t last = 0xFFFFFFFFu;
    if (!nv2a_texrep_zdebug() || !hash)
        return 0.0f;
    if (s_zp_file && flip != last) {
        last = flip;
        zp_reread();
    }
    if (!s_zp_nwin)
        return 0.0f;
    for (i = 0; i < s_zp_nwin; i++)
        if (flip >= s_zp_win[i] && (win < 0 || s_zp_win[i] > s_zp_win[win]))
            win = i;
    if (win < 0)
        return 0.0f;
    if (win != s_zp_cur) {                 /* a new window: a new list */
        s_zp_cur = win;
        s_zp_n = 0;
        s_zp_said = -1;
        s_zp_done = 0;
        fprintf(stderr, LOGP "zprobe window from flip %u\n", s_zp_win[win]);
    }
    for (j = 0; j < s_zp_n; j++)
        if (s_zp_hash[j] == hash)
            break;
    if (j == s_zp_n) {
        if (s_zp_n == ZP_MAX)
            return 0.0f;
        s_zp_hash[s_zp_n++] = hash;
    }
    /* Four flips a texture: unprobed, 512 back, unprobed, 4 back. A pass
     * over the same geometry (a car's reflection, a lightmap) loses at both;
     * a decal some centimetres in front of its surface only at 512. */
    rel = flip - s_zp_win[win];
    if (!(rel & 1u))
        return 0.0f;
    k = rel / 4u;
    if (k >= s_zp_n) {
        if (!s_zp_done) {
            s_zp_done = 1;
            fprintf(stderr, LOGP "zprobe done at flip %u: %u textures\n", flip, s_zp_n);
        }
        return 0.0f;
    }
    if (j != k)
        return 0.0f;
    if ((int)k != s_zp_said) {
        s_zp_said = (int)k;
        fprintf(stderr, LOGP "zprobe flip %u #%u %016llx %ux%u f%02x\n", flip, k,
                (unsigned long long)hash, w, h, color);
    }
    return (rel & 2u) ? 4.0f : 512.0f;
}

/* ── HDR gains (BO-110) ─────────────────────────────────────────────── */

static const struct nv2a_texrep_hdr_gain *s_hg;
static unsigned                           s_hg_n;

void nv2a_texrep_register_hdr_gain(const struct nv2a_texrep_hdr_gain *tex, unsigned n)
{
    s_hg = n ? tex : NULL;
    s_hg_n = tex ? n : 0;
}

/* RECOMP_DEBUG=hdr_gains_probe=<hash>[+<hash>...] (BO-195): textures to find
 * in an hdr_dump. One the title did not register gets an entry of its own
 * here, gain 1 (nothing changes), so the backend sees its draws and can mark
 * them (nv2a_pb_vulkan.c, hdr_gain_for). */
#define HDR_PROBE_MAX 64
static struct nv2a_texrep_hdr_gain s_hp[HDR_PROBE_MAX];
static int s_hp_n = -1;

int nv2a_texrep_hdr_probes(void)
{
    if (s_hp_n < 0) {
        const char *e = recomp_env(RENV_HDR_GAINS_PROBE);
        s_hp_n = 0;
        while (e && *e && s_hp_n < HDR_PROBE_MAX) {
            char *end;
            uint64_t v = strtoull(e, &end, 16);
            if (end == e)
                break;
            s_hp[s_hp_n].name = "probe";
            s_hp[s_hp_n].hash = v;
            s_hp[s_hp_n].gain = 1.0f;
            s_hp[s_hp_n].mode = NV2A_HDR_GAIN_MUL;
            s_hp_n++;
            e = *end ? end + 1 : end;
        }
        if (s_hp_n)
            fprintf(stderr, LOGP "HDR gains: %d probe texture(s)\n", s_hp_n);
    }
    return s_hp_n;
}

int nv2a_texrep_hdr_probe_index(uint64_t hash)
{
    int i;
    for (i = 0; i < nv2a_texrep_hdr_probes(); i++)
        if (s_hp[i].hash == hash)
            return i;
    return -1;
}

uint64_t nv2a_texrep_hdr_probe_hash(int i)
{
    return i >= 0 && i < nv2a_texrep_hdr_probes() ? s_hp[i].hash : 0;
}

const struct nv2a_texrep_hdr_gain *nv2a_texrep_hdr_gain(const uint8_t *guest, uint32_t color,
                                                        uint32_t w, uint32_t h, uint32_t pitch)
{
    uint64_t hash = 0;
    unsigned i;
    int p;

    for (i = 0; i < s_hg_n; i++) {
        const struct nv2a_texrep_hdr_gain *z = &s_hg[i];
        if (z->color != color || z->width != w || z->height != h)
            continue;
        if (!hash)
            hash = nv2a_texrep_hash(guest, color, w, h, pitch);
        if (hash == z->hash) {
            static _Atomic int said;
            if (atomic_fetch_add_explicit(&said, 1, memory_order_relaxed) < 16)
                fprintf(stderr, LOGP "HDR gain %g (%s) for %s\n", (double)z->gain,
                    z->mode == NV2A_HDR_GAIN_HI ? "highlights" : "colour",
                    z->name ? z->name : "?");
            return z;
        }
    }
    if (nv2a_texrep_hdr_probes() > 0) {
        if (!hash)
            hash = nv2a_texrep_hash(guest, color, w, h, pitch);
        if ((p = nv2a_texrep_hdr_probe_index(hash)) >= 0) {
            s_hp[p].color = color;
            s_hp[p].width = w;
            s_hp[p].height = h;
            return &s_hp[p];
        }
    }
    return NULL;
}

/* ── the dump ─────────────────────────────────────────────────────────── */

#define DUMP_SEEN 4096
static uint64_t s_seen[DUMP_SEEN];
static unsigned s_n_seen;

static void tga_write(const char *path, uint32_t w, uint32_t h, const uint32_t *texels)
{
    uint8_t hdr[18] = { 0 };
    FILE *f = fopen(path, "wb");

    if (!f)
        return;
    hdr[2] = 2;                        /* true colour */
    hdr[12] = (uint8_t)w; hdr[13] = (uint8_t)(w >> 8);
    hdr[14] = (uint8_t)h; hdr[15] = (uint8_t)(h >> 8);
    hdr[16] = 32;
    hdr[17] = 0x28;                    /* top-left origin, 8 alpha bits */
    fwrite(hdr, 1, sizeof hdr, f);
    fwrite(texels, 4, (size_t)w * h, f);   /* A8R8G8B8 words: B G R A bytes */
    fclose(f);
}

static void dump(uint64_t hash, uint32_t color, uint32_t w, uint32_t h, uint32_t levels,
                 const uint32_t *texels)
{
    static const char *dir;
    static int init, mips;
    char path[1200];
    unsigned i;
    uint32_t l;
    FILE *f;

    if (!init) {
        init = 1;
        dir = recomp_env(RENV_TEXREP_DUMP);
        if (dir && !*dir)
            dir = NULL;
        mips = recomp_env(RENV_TEXREP_DUMP_MIPS) != NULL;
        if (dir)
            fprintf(stderr, LOGP "dumping every uploaded texture to %s\n", dir);
    }
    if (!dir || !hash || w > 4096 || h > 4096)
        return;
    for (i = 0; i < s_n_seen; i++)
        if (s_seen[i] == hash)
            return;
    if (s_n_seen < DUMP_SEEN)
        s_seen[s_n_seen++] = hash;
    snprintf(path, sizeof path, "%s/%016llx_%ux%u_f%02x.tga", dir,
             (unsigned long long)hash, w, h, color & 0xFFu);
    tga_write(path, w, h, texels);
    /* texrep_dump_mips: the title's own levels below it too, as decoded
     * (BO-171 compares them with levels made from level 0). */
    for (l = 1; mips && l < levels; l++) {
        texels += (size_t)nv2a_tex_level_dim(w, l - 1) * nv2a_tex_level_dim(h, l - 1);
        snprintf(path, sizeof path, "%s/%016llx_%ux%u_f%02x_m%u.tga", dir,
                 (unsigned long long)hash, w, h, color & 0xFFu, l);
        tga_write(path, nv2a_tex_level_dim(w, l), nv2a_tex_level_dim(h, l), texels);
    }
    snprintf(path, sizeof path, "%s/index.tsv", dir);
    f = fopen(path, "ab");
    if (f) {
        fprintf(f, "%016llx\t%u\t%u\t0x%02x\n", (unsigned long long)hash, w, h,
                color & 0xFFu);
        fclose(f);
    }
}

static int dump_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *d = recomp_env(RENV_TEXREP_DUMP);
        on = d && *d;
    }
    return on;
}

/* ── apply ────────────────────────────────────────────────────────────── */

static uint32_t box4(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    uint32_t o = 0;
    int s;
    for (s = 0; s < 32; s += 8)
        o |= ((((a >> s) & 0xFFu) + ((b >> s) & 0xFFu) + ((c >> s) & 0xFFu)
               + ((d >> s) & 0xFFu) + 2u) >> 2) << s;
    return o;
}

/* Levels 1.. from the one above each, 2x2 box (a level of width or height 1
 * repeats its only row or column). keep_alpha: the levels' own alpha stays
 * (only their colours are made again). */
static void remip(uint32_t *texels, uint32_t w, uint32_t h, uint32_t levels, int keep_alpha)
{
    uint32_t *src = texels, l;
    for (l = 1; l < levels; l++) {
        uint32_t dw = w > 1 ? w / 2 : 1, dh = h > 1 ? h / 2 : 1, x, y;
        uint32_t *dst = src + (size_t)w * h;
        for (y = 0; y < dh; y++) {
            const uint32_t *r0 = src + (size_t)(h > 1 ? 2 * y : 0) * w;
            const uint32_t *r1 = src + (size_t)(h > 1 ? 2 * y + 1 : 0) * w;
            for (x = 0; x < dw; x++) {
                uint32_t x0 = w > 1 ? 2 * x : 0, x1 = w > 1 ? 2 * x + 1 : 0;
                uint32_t v = box4(r0[x0], r0[x1], r1[x0], r1[x1]), *d = &dst[(size_t)y * dw + x];
                *d = keep_alpha ? (*d & 0xFF000000u) | (v & 0x00FFFFFFu) : v;
            }
        }
        src = dst;
        w = dw;
        h = dh;
    }
}

/* The variant's rectangles over level 0, clipped to the texture. */
static void copy_rects(const struct nv2a_texrep_variant *var, uint32_t w, uint32_t h,
                       uint32_t *texels)
{
    unsigned i;
    for (i = 0; i < var->n_rects; i++) {
        struct nv2a_texrep_rect r = var->rects[i];
        uint32_t y, x;
        if (r.x >= w || r.y >= h)
            continue;
        if ((uint32_t)r.x + r.w > w)
            r.w = (uint16_t)(w - r.x);
        if ((uint32_t)r.y + r.h > h)
            r.h = (uint16_t)(h - r.y);
        for (y = r.y; y < (uint32_t)r.y + r.h; y++) {
            uint32_t *d = texels + (size_t)y * w + r.x;
            const uint32_t *s = var->image + (size_t)y * w + r.x;
            if (!var->keep_alpha) {
                memcpy(d, s, (size_t)r.w * 4);
            } else {
                for (x = 0; x < r.w; x++)
                    d[x] = (d[x] & 0xFF000000u) | (s[x] & 0x00FFFFFFu);
            }
        }
    }
}

uint32_t nv2a_texrep_apply(const uint8_t *guest, uint32_t color, uint32_t w,
                           uint32_t h, uint32_t pitch, uint32_t levels,
                           uint32_t *texels)
{
    const struct nv2a_texrep_tex *tex = s_tex, *t = NULL, *ht = NULL;
    const struct nv2a_texrep_variant *var;
    unsigned n = s_n, hn = s_hn, i, cand = 0, changed = 0, rgb_only = 0;
    uint32_t gen;
    uint64_t hash = 0;
    int v;

    for (i = 0; i < n; i++)
        if (tex[i].color == color && tex[i].width == w && tex[i].height == h)
            cand = 1;
    for (i = 0; i < hn; i++)
        if (s_ht[i].color == color && s_ht[i].width == w && s_ht[i].height == h)
            cand = 1;
    if (!cand && !dump_on())
        return 0;
    hash = nv2a_texrep_hash(guest, color, w, h, pitch);
    if (dump_on())
        dump(hash, color, w, h, levels, texels);
    if (!cand || !hash)
        return 0;
    atomic_fetch_add_explicit(&s_hashed, 1, memory_order_relaxed);
    for (i = 0; i < n && !t; i++)
        if (tex[i].color == color && tex[i].width == w && tex[i].height == h
                && tex[i].hash == hash)
            t = &tex[i];
    for (i = 0; i < hn && !ht; i++)
        if (s_ht[i].color == color && s_ht[i].width == w && s_ht[i].height == h
                && s_ht[i].hash == hash)
            ht = &s_ht[i];
    if (!t && !ht)
        return 0;
    atomic_fetch_add_explicit(&s_matched, 1, memory_order_relaxed);
    /* The generation first: a variant set while this runs leaves the entry
     * with the older generation, and the next bind uploads it again. */
    gen = nv2a_texrep_generation();
    v = nv2a_texrep_variant();
    if (t && v >= 1 && (unsigned)v <= t->n_variants) {
        var = &t->variants[v - 1];
        if (var->image && var->n_rects) {
            copy_rects(var, w, h, texels);
            changed = 1;
            if (atomic_fetch_add_explicit(&s_replaced, 1, memory_order_relaxed) < 64)
                fprintf(stderr, LOGP "%s (%ux%u, %016llx): variant %d, %u regions\n",
                        t->name ? t->name : "texture", w, h, (unsigned long long)hash, v,
                        var->n_rects);
        }
    }
    if (ht && nv2a_texrep_hdr() && ht->n_variants >= 1 && ht->variants[0].image
            && ht->variants[0].n_rects) {
        copy_rects(&ht->variants[0], w, h, texels);
        rgb_only = changed ? 0 : ht->variants[0].keep_alpha != 0;
        changed = 1;
        if (atomic_fetch_add_explicit(&s_replaced, 1, memory_order_relaxed) < 64)
            fprintf(stderr, LOGP "%s (%ux%u, %016llx): HDR picture\n",
                    ht->name ? ht->name : "texture", w, h, (unsigned long long)hash);
    }
    if (changed && levels > 1)
        remip(texels, w, h, levels, rgb_only);
    return gen;
}
