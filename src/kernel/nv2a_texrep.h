/**
 * nv2a_texrep.h - regions of a title's textures replaced at upload.
 *
 * The idea is Dolphin's Dynamic Input Textures: a texture of the title is
 * known by the hash of its bytes, and before a GPU backend uploads it, images
 * the host program brings are copied over rectangles of it. The first use is
 * the title's button prompts drawn for the pad in the player's hands
 * (input.glyphs: the title's own Xbox discs, or PlayStation, Nintendo, Steam
 * Deck ones).
 *
 * The title registers its textures once, at start (nv2a_texrep_register):
 * each with the hash of its level-0 guest bytes (nv2a_texrep_hash), its
 * format and size, and for each variant an image of the texture's own size
 * and the rectangles of it that go over the title's texels. Variant 0 is the
 * title's own picture: nothing is replaced and the upload is what it was.
 *
 * A backend calls nv2a_texrep_apply() between its decode (nv2a_tex_decode,
 * A8R8G8B8) and its upload. A texture of another format or size than every
 * registered one costs a few integer compares and is not hashed; a candidate
 * is hashed once per upload (the backends upload a texture when it is first
 * drawn and when its bytes change, not per frame). The backend keeps the
 * generation apply() returned with its cache entry and uploads the texture
 * again when nv2a_texrep_generation() has moved on: that is how a change of
 * variant reaches a texture that is already on the GPU.
 *
 * No OS or graphics headers: shared by the Metal, Vulkan and D3D11 backends.
 */
#ifndef NV2A_TEXREP_H
#define NV2A_TEXREP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nv2a_texrep_rect {
    uint16_t x, y, w, h;
};

/* One variant's picture of one texture. image: width x height A8R8G8B8,
 * row-major (the decode's layout); NULL: this variant keeps the title's
 * texels. Only the rectangles are copied, all four channels. */
struct nv2a_texrep_variant {
    const uint32_t                *image;
    const struct nv2a_texrep_rect *rects;
    unsigned                       n_rects;
    /* BO-203: nonzero: only the colour channels are copied, the title's own
     * alpha stays (and the levels below are made again from the colours
     * alone, their alpha the title's). For a picture of colour under an alpha
     * the title's draw depends on: the alpha test and the occlusion count it
     * reads for its glare stay what the title made them. */
    uint32_t                       keep_alpha;
    /* BO-203, HDR pictures only: nonzero: while the picture is in use
     * (nv2a_texrep_hdr), a draw that binds this texture at stage 0 samples
     * transparent black (0, 0, 0, 0) at stages 1..3, whatever the title bound
     * there (nv2a_texrep_hdr_stage0_only). For a sprite whose combiner blends
     * in another texture the draw does not map onto the sprite: the sun's
     * disc lerps to the sky sheet by its alpha at a point (or a line) of it,
     * so where the sky is bright the whole quad passes the alpha test as a
     * flat or streaked patch, hidden in SDR's clipped white only. */
    uint32_t                       stage0_only;
};

struct nv2a_texrep_tex {
    const char *name;                 /* for the log */
    uint64_t    hash;                 /* nv2a_texrep_hash of the title's bytes */
    uint32_t    color;                /* the NV2A colour format (0x0F: DXT45) */
    uint32_t    width, height;
    /* variants[v - 1] is variant v (v >= 1); a variant past n_variants keeps
     * the title's texels. */
    const struct nv2a_texrep_variant *variants;
    unsigned    n_variants;
};

/* The textures, kept by pointer (static tables). Replaces an earlier call;
 * (NULL, 0) forgets them. Call before the first draw, or under the title's
 * own care that no backend is uploading. */
void nv2a_texrep_register(const struct nv2a_texrep_tex *tex, unsigned n);

/* The variant in use (0: the title's own). Any thread; a change moves the
 * generation on. */
void     nv2a_texrep_set_variant(int v);
int      nv2a_texrep_variant(void);
/* Never 0; changes whenever what apply() would do changes. */
uint32_t nv2a_texrep_generation(void);

/* The hash a registered texture is known by: its level-0 guest bytes
 * (nv2a_tex_extent of one level), whatever number of levels a backend
 * samples. 0 for a format with no extent. */
uint64_t nv2a_texrep_hash(const uint8_t *guest, uint32_t color, uint32_t w,
                          uint32_t h, uint32_t pitch);

/* texels: the decoded texture, nv2a_tex_texels(w, h, levels) entries, level 0
 * first. If (color, w, h) and the hash of guest name a registered texture:
 * the variant's rectangles are copied over level 0, the further levels are
 * made again from it (2x2 box), and the generation is returned (also for
 * variant 0, which copies nothing: the entry must be uploaded again when the
 * variant changes). 0: not a registered texture, texels untouched. */
uint32_t nv2a_texrep_apply(const uint8_t *guest, uint32_t color, uint32_t w,
                           uint32_t h, uint32_t pitch, uint32_t levels,
                           uint32_t *texels);

/* Counters, for a report and the tests: candidates hashed, textures matched,
 * textures a variant changed. */
void nv2a_texrep_stats(uint64_t *hashed, uint64_t *matched, uint64_t *replaced);

/* BO-203: pictures that replace a texture while HDR is on, and only
 * then. The table is a second registration of the same shape
 * (nv2a_texrep_tex; variants[0] is the picture in HDR, n_variants 1), kept
 * apart from nv2a_texrep_register so a title's button prompts and its HDR
 * pictures do not compete for the one variant number. nv2a_texrep_apply
 * copies the picture's rectangles over level 0 only while
 * nv2a_texrep_set_hdr(1) is in force; with HDR off (the SDR path) the
 * texels are the title's own and nothing is copied. A backend calls
 * nv2a_texrep_set_hdr wherever it decides HDR is on or off: a change moves
 * the generation on, so the textures already on the GPU are uploaded again,
 * in both directions. A matched texture keeps a non-zero generation in SDR
 * for that reason (stats: matched counts it, replaced does not). The first
 * use is the sun's disc sprite, whose flat alpha plateau reads as a square
 * above paper white (a profile built at start, no asset shipped). */
void nv2a_texrep_register_hdr_image(const struct nv2a_texrep_tex *tex, unsigned n);
/* 1 when the texture is one of those pictures and its variant asks for
 * stage0_only, whether HDR is on or not (a backend asks at upload and keeps
 * the answer with its cache entry; at a draw it checks nv2a_texrep_hdr()).
 * A few integer compares for any other format or size, a hash for a
 * candidate. */
int  nv2a_texrep_hdr_stage0_only(const uint8_t *guest, uint32_t color, uint32_t w,
                                 uint32_t h, uint32_t pitch);
void nv2a_texrep_set_hdr(int on);
int  nv2a_texrep_hdr(void);        /* the pictures are in use now: HDR on, and not an A/B flip with them off */
/* RECOMP_DEBUG=hdr_image_ab=n: the pictures are off for every other block of n
 * flips (a comparison in one run: neighbouring flips are the same moment). A
 * backend calls this once a flip, from its HDR flip; the generation moves at
 * each change. Without the setting it does nothing. */
void nv2a_texrep_flip(uint32_t flip);

/* BO-176: textures whose draws take a constant depth offset, as if the
 * title had set SET_POLYGON_OFFSET_BIAS for them. For a decal the title
 * draws a few centimetres in front of the surface it sits on, which the
 * original hardware keeps apart with its w-buffer (SET_CONTROL0 bit 16) and
 * the host's z-buffer, computed from the title's own screen-space z, does
 * not always: that z carries float32 noise of tens of depth units at a
 * distance (world coordinates in the thousands), more than the decal's lead.
 * bias is in depth units of the zeta format (0..zmax), negative toward the
 * eye; a title's own polygon offset on the draw adds to it.
 *
 * The title registers once, at start (static table, kept by pointer); the
 * backends ask nv2a_texrep_zbias() when they upload a texture (a few
 * integer compares for any other format or size, a hash for a candidate)
 * and keep the answer with their cache entry. */
struct nv2a_texrep_zbias {
    const char *name;                 /* for the log */
    uint64_t    hash;                 /* nv2a_texrep_hash of the title's bytes */
    uint32_t    color, width, height;
    float       bias;
};

void  nv2a_texrep_register_zbias(const struct nv2a_texrep_zbias *tex, unsigned n);
/* Whether the backends apply the offsets now: RECOMP_DEBUG=zbias=0 never,
 * zbias_ab=n on for n flips and off for n (comparisons); else always. */
int   nv2a_texrep_zbias_on(uint32_t flip);   /* flip: the executor's flip count */
/* The bias of a registered texture, 0 for any other. */
float nv2a_texrep_zbias(const uint8_t *guest, uint32_t color, uint32_t w,
                        uint32_t h, uint32_t pitch);

/* BO-204, finding more of them (debug, Vulkan). RECOMP_DEBUG=zbias_probe=
 * <flip>[+<flip>...] or =@file (read again as it grows): from each of those
 * flips (a paused scene), four flips a texture, in the order the window first
 * saw them: unprobed, its depth-tested draws 512 units back, unprobed, 4
 * units back; logged as "[TEXREP] zprobe flip F #k <hash> <w>x<h> f<format>"
 * (F: the 512 flip). A texture drawn a few centimetres in front of another
 * surface (a sign's words, a poster) vanishes at 512 only; a pass over the
 * same geometry (reflections, lightmaps) at both.
 * ztint=<hash>[+<hash>...]: those textures uploaded magenta (alpha kept),
 * for counting a decal's pixels frame by frame. */
int   nv2a_texrep_zdebug(void);              /* zbias_probe or ztint is set */
float nv2a_texrep_zprobe(uint32_t flip, uint64_t hash, uint32_t color, uint32_t w,
                         uint32_t h);
int   nv2a_texrep_ztint(uint64_t hash);
/* At upload, with zbias_probe/ztint set: the texture's hash (0 otherwise),
 * its n decoded texels made magenta when ztint lists it. */
uint64_t nv2a_texrep_ztint_apply(const uint8_t *guest, uint32_t color, uint32_t w,
                                 uint32_t h, uint32_t pitch, uint32_t *texels, size_t n);

/* BO-110: a title's light textures, a gain on their draws' colour with HDR
 * on (scene draws only; SDR never reads it). mode NV2A_HDR_GAIN_MUL: the
 * colour x gain (linear light), for additive sprites (flames, glows);
 * NV2A_HDR_GAIN_HI: only the brightest part, by smoothstep(0.6, 1.0, its
 * brightest channel)^2, towards x gain (a sky's sun disc, cloud edges).
 * Registered once at start, looked up at upload as the depth offsets. */
enum { NV2A_HDR_GAIN_MUL = 0, NV2A_HDR_GAIN_HI = 1 };
struct nv2a_texrep_hdr_gain {
    const char *name;                 /* for the log */
    uint64_t    hash;                 /* nv2a_texrep_hash of the title's bytes */
    uint32_t    color, width, height;
    float       gain;                 /* linear light; 1 = none */
    int         mode;
};
void nv2a_texrep_register_hdr_gain(const struct nv2a_texrep_hdr_gain *tex, unsigned n);
/* The entry of a registered texture, NULL for any other. */
const struct nv2a_texrep_hdr_gain *nv2a_texrep_hdr_gain(const uint8_t *guest, uint32_t color,
                                                        uint32_t w, uint32_t h, uint32_t pitch);
/* RECOMP_DEBUG=hdr_gains_probe (BO-195): the number of probe textures, and a
 * hash's index among them (-1: not a probe). nv2a_texrep_hdr_gain returns an
 * entry of gain 1 for a probe the title did not register. */
int nv2a_texrep_hdr_probes(void);
int nv2a_texrep_hdr_probe_index(uint64_t hash);
uint64_t nv2a_texrep_hdr_probe_hash(int i);   /* 0: none */

/* RECOMP_DEBUG=texrep_dump=<dir> (read by apply): every texture a backend
 * uploads, once per content, as <dir>/<hash>_<w>x<h>_f<color>.tga (level 0,
 * A8R8G8B8 as decoded, before any replacement), with one line per file in
 * <dir>/index.tsv. The hash is nv2a_texrep_hash, the one to register. A
 * debug aid for finding a title's textures: the files are the title's own
 * pictures and stay on the machine that made them. With it on every upload
 * is hashed. */

#ifdef __cplusplus
}
#endif

#endif /* NV2A_TEXREP_H */
