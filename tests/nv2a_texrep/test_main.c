/* nv2a_texrep: a registered texture is found by format, size and the hash of
 * its level-0 bytes; the variant's rectangles are copied over the decoded
 * texels and nothing else is touched; variant 0 changes nothing; the
 * generation moves when the variant does. Synthetic textures only. */
#include <stdio.h>
#include <string.h>
#include "nv2a_texrep.h"
#include "nv2a_backend_common.h"

static int s_checks, s_fail;
#define CHECK(x) do { s_checks++; if (!(x)) { s_fail++; printf("FAIL line %d: %s\n", __LINE__, #x); } } while (0)

#define FMT_DXT5 0x0Fu
#define FMT_A8R8G8B8_LIN 0x12u
#define W 16u
#define H 8u

static uint8_t  g_dxt[(W / 4) * (H / 4) * 16];   /* the "title's" bytes */
static uint8_t  g_other[sizeof g_dxt];
static uint32_t g_img1[W * H], g_img2[W * H];

static void fill(uint32_t *t, size_t n, uint32_t v)
{
    size_t i;
    for (i = 0; i < n; i++)
        t[i] = v + (uint32_t)i;
}

int main(void)
{
    static const struct nv2a_texrep_rect r1[] = { { 2, 1, 4, 3 }, { 12, 6, 8, 8 } /* clipped */ };
    static const struct nv2a_texrep_rect r2[] = { { 0, 0, W, H } };
    static struct nv2a_texrep_variant vars[3];
    static struct nv2a_texrep_tex tex[2];
    uint32_t texels[W * H + (W / 2) * (H / 2) + (W / 4) * (H / 4)], ref[W * H];
    uint32_t gen, gen2, x, y;
    uint64_t hashed, matched, replaced;
    size_t i;

    for (i = 0; i < sizeof g_dxt; i++) {
        g_dxt[i] = (uint8_t)(i * 7 + 3);
        g_other[i] = (uint8_t)(i * 5 + 1);
    }
    fill(g_img1, W * H, 0x11000000u);
    fill(g_img2, W * H, 0x22000000u);
    vars[0].image = g_img1; vars[0].rects = r1; vars[0].n_rects = 2;
    vars[1].image = NULL;   vars[1].rects = r2; vars[1].n_rects = 1;   /* keeps the title's */
    vars[2].image = g_img2; vars[2].rects = r2; vars[2].n_rects = 1;
    tex[0].name = "synthetic"; tex[0].color = FMT_DXT5; tex[0].width = W; tex[0].height = H;
    tex[0].hash = nv2a_texrep_hash(g_dxt, FMT_DXT5, W, H, 0);
    tex[0].variants = vars; tex[0].n_variants = 3;
    tex[1] = tex[0];
    tex[1].name = "never"; tex[1].hash ^= 1;

    /* The hash is the texture hash of one level's bytes. */
    CHECK(tex[0].hash == nv2a_tex_hash(g_dxt, sizeof g_dxt, NV2A_TEX_HASH_SEED));
    CHECK(tex[0].hash != nv2a_texrep_hash(g_other, FMT_DXT5, W, H, 0));
    CHECK(nv2a_texrep_hash(g_dxt, 0xEEu, W, H, 0) == 0);     /* no such format */

    /* Nothing registered: nothing happens, whatever the variant. */
    nv2a_texrep_set_variant(1);
    fill(texels, W * H, 0xAA000000u);
    memcpy(ref, texels, sizeof ref);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == 0);
    CHECK(!memcmp(ref, texels, sizeof ref));

    nv2a_texrep_register(tex, 2);
    gen = nv2a_texrep_generation();
    CHECK(gen != 0);

    /* Another size or format: not a candidate, not hashed. */
    nv2a_texrep_stats(&hashed, NULL, NULL);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H * 2, 0, 1, texels) == 0);
    CHECK(nv2a_texrep_apply(g_dxt, 0x0Cu, W, H, 0, 1, texels) == 0);
    { uint64_t h2; nv2a_texrep_stats(&h2, NULL, NULL); CHECK(h2 == hashed); }
    /* The size and format of a registered one, other bytes: hashed, no match. */
    CHECK(nv2a_texrep_apply(g_other, FMT_DXT5, W, H, 0, 1, texels) == 0);
    CHECK(!memcmp(ref, texels, sizeof ref));
    nv2a_texrep_stats(&hashed, &matched, &replaced);
    CHECK(hashed == 1 && matched == 0 && replaced == 0);

    /* The registered bytes, variant 1: its rectangles and only they. */
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == gen);
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            int in = (x >= 2 && x < 6 && y >= 1 && y < 4) || (x >= 12 && y >= 6);
            CHECK(texels[y * W + x] == (in ? g_img1 : ref)[y * W + x]);
        }
    nv2a_texrep_stats(&hashed, &matched, &replaced);
    CHECK(hashed == 2 && matched == 1 && replaced == 1);

    /* Variant 0 (the title's own): matched, so the cache follows it, but
     * untouched. The generation moved with the variant. */
    nv2a_texrep_set_variant(0);
    gen2 = nv2a_texrep_generation();
    CHECK(gen2 != gen && gen2 != 0);
    memcpy(texels, ref, sizeof ref);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == gen2);
    CHECK(!memcmp(ref, texels, sizeof ref));
    /* The same variant again: the generation stays. */
    nv2a_texrep_set_variant(0);
    CHECK(nv2a_texrep_generation() == gen2);

    /* A variant with no image, and one past the table: the title's texels. */
    nv2a_texrep_set_variant(2);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == nv2a_texrep_generation());
    CHECK(!memcmp(ref, texels, sizeof ref));
    nv2a_texrep_set_variant(9);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == nv2a_texrep_generation());
    CHECK(!memcmp(ref, texels, sizeof ref));

    /* Variant 3 replaces everything; with three levels the lower ones are
     * the box filter of the new level 0. */
    nv2a_texrep_set_variant(3);
    for (i = 0; i < W * H; i++)
        g_img2[i] = (i / W) < H / 2 ? 0xFF204060u : 0x00000000u;
    memset(texels, 0x55, sizeof texels);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 3, texels) != 0);
    CHECK(!memcmp(texels, g_img2, sizeof g_img2));
    CHECK(texels[W * H] == 0xFF204060u);                                  /* level 1, top */
    CHECK(texels[W * H + (W / 2) * (H / 2) - 1] == 0x00000000u);          /* level 1, bottom */
    CHECK(texels[W * H + (W / 2) * (H / 2)] == 0xFF204060u);              /* level 2 */
    CHECK(texels[W * H + (W / 2) * (H / 2) + (W / 4) * (H / 4) - 1] == 0);

    /* A linear format is known by pitch x height of level 0. */
    {
        static uint8_t lin[W * 4 * H];
        static struct nv2a_texrep_tex lt;
        for (i = 0; i < sizeof lin; i++)
            lin[i] = (uint8_t)(i * 3);
        lt = tex[0];
        lt.color = FMT_A8R8G8B8_LIN;
        lt.hash = nv2a_texrep_hash(lin, FMT_A8R8G8B8_LIN, W, H, W * 4);
        CHECK(lt.hash != 0);
        nv2a_texrep_register(&lt, 1);
        memcpy(texels, ref, sizeof ref);
        CHECK(nv2a_texrep_apply(lin, FMT_A8R8G8B8_LIN, W, H, W * 4, 1, texels) != 0);
        CHECK(!memcmp(texels, g_img2, sizeof g_img2));
        /* The DXT one is forgotten. */
        CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == 0);
    }

    nv2a_texrep_register(NULL, 0);
    CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == 0);

    /* BO-203: a picture that replaces the whole texture with HDR on, and
     * only then. Registered apart from the variants; matched in SDR too (the
     * cache keeps a generation, so a switch uploads it again), untouched
     * there; replaced and re-mipped with HDR on; the generation moves with
     * the switch in both directions, not when it stays. */
    {
        static struct nv2a_texrep_variant hv;
        static struct nv2a_texrep_tex ht;
        uint64_t m0, r0, m1, r1_, m2, r2_;
        uint32_t g0, g1, g2;

        for (i = 0; i < W * H; i++)
            g_img2[i] = 0x80FFFFFFu;
        hv.image = g_img2; hv.rects = r2; hv.n_rects = 1;
        ht = tex[0];
        ht.name = "hdr picture"; ht.variants = &hv; ht.n_variants = 1;
        nv2a_texrep_set_variant(0);
        nv2a_texrep_set_hdr(0);
        nv2a_texrep_register_hdr_image(&ht, 1);
        g0 = nv2a_texrep_generation();
        CHECK(g0 != 0 && !nv2a_texrep_hdr());
        memcpy(texels, ref, sizeof ref);
        nv2a_texrep_stats(NULL, &m0, &r0);
        CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == g0);   /* SDR: matched */
        CHECK(!memcmp(texels, ref, sizeof ref));                               /* ... untouched */
        nv2a_texrep_stats(NULL, &m1, &r1_);
        CHECK(m1 == m0 + 1 && r1_ == r0);
        /* Another texture of the same size: nothing. */
        CHECK(nv2a_texrep_apply(g_other, FMT_DXT5, W, H, 0, 1, texels) == 0);
        nv2a_texrep_set_hdr(0);                                                /* no change */
        CHECK(nv2a_texrep_generation() == g0);
        nv2a_texrep_set_hdr(1);                                                /* HDR on */
        g1 = nv2a_texrep_generation();
        CHECK(g1 != g0 && g1 != 0 && nv2a_texrep_hdr());
        CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 3, texels) == g1);
        CHECK(!memcmp(texels, g_img2, sizeof g_img2));
        CHECK(texels[W * H] == 0x80FFFFFFu);                                   /* level 1 */
        nv2a_texrep_stats(NULL, &m2, &r2_);
        CHECK(m2 == m1 + 1 && r2_ == r1_ + 1);
        nv2a_texrep_set_hdr(0);                                                /* and off again */
        g2 = nv2a_texrep_generation();
        CHECK(g2 != g1 && g2 != 0);
        memcpy(texels, ref, sizeof ref);
        CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == g2);
        CHECK(!memcmp(texels, ref, sizeof ref));
        /* A colour-only picture (keep_alpha): the title's alpha stays, in
         * every level; the colours are the picture's, the levels below made
         * from them. */
        hv.keep_alpha = 1;
        for (i = 0; i < W * H; i++)
            g_img2[i] = 0x00102030u + (uint32_t)(i & 7) * 0x00010101u;     /* alpha 0 in the picture */
        nv2a_texrep_register_hdr_image(&ht, 1);
        nv2a_texrep_set_hdr(1);
        for (i = 0; i < W * H + (W / 2) * (H / 2) + (W / 4) * (H / 4); i++)
            texels[i] = 0xC0000000u | (uint32_t)(i & 0xFFFF);               /* the title's levels */
        memcpy(ref, texels, sizeof ref);
        {
            uint32_t alpha0[W * H + (W / 2) * (H / 2) + (W / 4) * (H / 4)];
            memcpy(alpha0, texels, sizeof alpha0);
            CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 3, texels) != 0);
            for (i = 0; i < W * H + (W / 2) * (H / 2) + (W / 4) * (H / 4); i++)
                CHECK((texels[i] >> 24) == (alpha0[i] >> 24));              /* alpha kept, all levels */
            for (i = 0; i < W * H; i++)
                CHECK((texels[i] & 0x00FFFFFFu) == (g_img2[i] & 0x00FFFFFFu));
            CHECK((texels[W * H] & 0x00FFFFFFu) == 0x00112131u);            /* level 1: the box of the colours */
        }
        /* stage0_only (the sun's disc): asked by hash, format and size,
         * HDR on or off (the backend checks nv2a_texrep_hdr at the draw);
         * not without the flag, not for another texture. */
        CHECK(!nv2a_texrep_hdr_stage0_only(g_dxt, FMT_DXT5, W, H, 0));
        hv.stage0_only = 1;
        nv2a_texrep_register_hdr_image(&ht, 1);
        CHECK(nv2a_texrep_hdr_stage0_only(g_dxt, FMT_DXT5, W, H, 0));
        CHECK(!nv2a_texrep_hdr_stage0_only(g_other, FMT_DXT5, W, H, 0));
        CHECK(!nv2a_texrep_hdr_stage0_only(g_dxt, FMT_DXT5, W, H * 2, 0));
        nv2a_texrep_set_hdr(0);
        CHECK(nv2a_texrep_hdr_stage0_only(g_dxt, FMT_DXT5, W, H, 0) && !nv2a_texrep_hdr());
        nv2a_texrep_set_hdr(1);
        /* Forgotten: not a candidate any more. */
        nv2a_texrep_register_hdr_image(NULL, 0);
        CHECK(!nv2a_texrep_hdr_stage0_only(g_dxt, FMT_DXT5, W, H, 0));
        nv2a_texrep_set_hdr(1);
        CHECK(nv2a_texrep_apply(g_dxt, FMT_DXT5, W, H, 0, 1, texels) == 0);
        nv2a_texrep_set_hdr(0);
    }

    /* BO-176: depth offsets, by hash, format and size. */
    {
        static struct nv2a_texrep_zbias zb[2];
        zb[0].name = "decal";
        zb[0].hash = nv2a_texrep_hash(g_dxt, FMT_DXT5, W, H, 0);
        zb[0].color = FMT_DXT5; zb[0].width = W; zb[0].height = H;
        zb[0].bias = -512.0f;
        zb[1] = zb[0];
        zb[1].hash ^= 1u;                   /* a texture that is not there */
        zb[1].bias = 99.0f;
        CHECK(nv2a_texrep_zbias(g_dxt, FMT_DXT5, W, H, 0) == 0.0f);   /* none registered */
        nv2a_texrep_register_zbias(zb, 2);
        CHECK(nv2a_texrep_zbias(g_dxt, FMT_DXT5, W, H, 0) == -512.0f);
        CHECK(nv2a_texrep_zbias(g_other, FMT_DXT5, W, H, 0) == 0.0f);
        CHECK(nv2a_texrep_zbias(g_dxt, FMT_DXT5, W, H * 2, 0) == 0.0f);
        CHECK(nv2a_texrep_zbias(g_dxt, 0x0Cu, W, H, 0) == 0.0f);
        CHECK(nv2a_texrep_zbias_on(0) == 1);                           /* default: on */
        nv2a_texrep_register_zbias(NULL, 0);
        CHECK(nv2a_texrep_zbias(g_dxt, FMT_DXT5, W, H, 0) == 0.0f);
    }

    printf("nv2a_texrep: %d checks, %d failed\n", s_checks, s_fail);
    return s_fail != 0;
}
