/**
 * A Metal backend for the pushbuffer executor (struct nv2a_pb_backend), the
 * macOS sibling of nv2a_pb_d3d11.c.
 *
 * The executor (src/kernel/nv2a_pb_exec.c) decodes the title's pushbuffer into
 * the state in nv2a_pb_state.h and calls the hooks of struct nv2a_pb_backend;
 * this file answers them with Metal instead of the CPU rasteriser:
 *
 *   on_clear  -> a render pass with loadAction Clear on the surface at
 *                color_offset (one MTLTexture per guest surface, grown to
 *                its largest clip); a clear of part of it (clip, clear
 *                rect) is a scissored draw of the clear colour
 *   on_draw   -> vertex-program batches through the program as MSL
 *                (d3d8_vsh_msl.c), pre-transformed (screen-space) ones through
 *                a pass-through vertex function; pixels through the register
 *                combiners as MSL (d3d8_combiners_msl.c) when they are
 *                programmed, else MODULATE. Four stages, textured from the
 *                shared decode (nv2a_backend_common.h).
 *   on_flip   -> commit and wait; with the CAMetalLayer window up
 *                (fb_present_metal.m, whose device and queue this backend
 *                adopts), blit the surface the walker picked for this flip
 *                (nv2a_pb_present_state) into the window's back slot and
 *                publish it: no CPU copy. Under metal_present=readback the
 *                texture is read back for the SDL renderer window
 *                (xbox_FramebufferWindowPresentPixels) as before.
 *   sync_guest -> write a surface back to guest memory when the executor
 *                dumps it (RECOMP_FB_DUMP and the report): the write-back is
 *                on demand (metal_writeback=lazy, default), since the title
 *                never reads its present surface; metal_writeback=always,
 *                metal_no_rtt or a read caught by metal_fb_guard write it at
 *                every flip.
 *
 * Opt-in: RECOMP_PB_BACKEND=metal (nv2a_pb_metal_register_from_env). The CPU
 * rasteriser stays the default and the oracle.
 *
 * Rules this file keeps (from the M1 plan):
 *  - Shaders are compiled at runtime (newLibraryWithSource) with fast math off,
 *    so float results match the CPU path's IEEE arithmetic.
 *  - Constant structs are float4/uint4 only on the MSL side; the C structs they
 *    mirror are checked with _Static_assert, as the HLSL ABI is in the D3D11
 *    backend.
 *  - Encoders begin lazily (first clear or draw on a surface) and end
 *    explicitly on a render-target change, before a readback and at the flip;
 *    every pass loads (loadAction Load) unless it starts with a clear.
 *  - Only the present surface is written back to guest memory, on demand
 *    (sync_guest, a texture decode that overlaps it, its eviction), or at
 *    every flip under metal_writeback=always.
 *  - Guest memory is touched on the executor's thread only: no completion
 *    handlers; a write-back waits for the command buffer and then copies.
 *
 * Depth/stencil and render-to-texture as the D3D11 backend (M2); visibility
 * tests (GET_REPORT) from Metal visibility results through the shared
 * tracker (M3, "Visibility tests" below). Near clipping needs nothing extra:
 * the vertex programs hand Metal clip-space positions (w kept, as for D3D11)
 * and the default MTLDepthClipModeClip clips at the near and far planes, as
 * D3D11's DepthClipEnable does. Not yet: swizzled render targets. The mip
 * LOD bias is a sample() argument (MsPsConsts.tex_bias), not a sampler
 * field, so every macOS takes the same path.
 *
 * Everything runs on the NV2A ack thread (or the smoke test's main thread).
 *
 * Off macOS this file compiles to nothing.
 */
#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stddef.h>
#include "recomp_env.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <signal.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include "nv2a_pb_state.h"
#include "nv2a_backend_common.h"
#include "d3d8_msl.h"
#include "d3d8_swizzle.h"
#include "nv2a_texrep.h"
#include "nv2a_texscale.h"
#include "nv2a_texmip.h"
#include "kernel_pause.h"
static void scale_prepare_loaded(void);
static int ui_scale_log(void);
#include "nv2a_pcache.h"
#include "latency_trace.h"
#include "host_time.h"
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include "../platform/xr_log.h"   /* the print gate (also force-included) */
/* xbox_memory_layout.h pulls in the Win32 shim (BOOL, GetCurrentThread),
 * which collides with Objective-C and CoreServices: declare what is used. */
ptrdiff_t xbox_GetMemoryOffset(void);

#if __has_feature(objc_arc)
#error "nv2a_pb_metal.m is manual retain/release: build it without -fobjc-arc"
#endif

#define LOGP "[metal] "
/* Render targets at render.scale stay within Metal's 2D texture limit on
 * Apple GPUs (nv2a_host_size lowers the factor for a larger surface). */
#define METAL_MAX_DIM 16384u

extern void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
extern void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch);
extern int  xbox_FramebufferWindowRunning(void);
extern void xbox_FramebufferWindowPresentPixels(const void *px, uint32_t w, uint32_t h,
                                                uint32_t pitch, uint32_t bpp);
/* The CAMetalLayer window (fb_present_metal.m): its device and queue, which
 * this backend adopts so the slot textures and the present pass share them,
 * and the back slot a frame is blitted into. */
extern int   xbox_FramebufferWindowMetal(void **device, void **queue);
extern void *xbox_FramebufferWindowBackTexture(uint32_t w, uint32_t h);
extern void  xbox_FramebufferWindowPublishTexture(void);
static int s_layer;                  /* the window takes GPU slot textures */
static int pc_start(void);           /* the pipeline cache: tables, disk, pre-warm */
static void pc_log(uint64_t presents);
static void guard_open(void);        /* metal_fb_guard: open the pages before we touch them */
static int wb_always(void);
static uint64_t s_window_presents;   /* frames sent to the window (slot or readback) */

/* ---- shader ABI ------------------------------------------------------------
 *
 * The MSL side declares only float4/uint4 members, so its layout is the C
 * struct's by construction: every member 16-aligned, no implicit padding.
 * The asserts pin the C structs (nv2a_backend_common.h) to that layout. */

/* VpConsts (MSL): float4 c[3]; uint4 u[2]; */
_Static_assert(sizeof(struct nv2a_vp_consts) == 80, "nv2a_vp_consts is not 5 x 16 bytes");
/* A program's c[] (MSL constant float4 *c, buffer 1) is v->c as it is. */
_Static_assert(sizeof(((struct nv2a_pb_vsh *)0)->c) == NV2A_VS_MAX_CONSTANTS * 16,
               "vertex constants are not 192 float4");
_Static_assert(offsetof(struct nv2a_vp_consts, u) == 48, "nv2a_vp_consts.u moved");

/* PsConsts (MSL): float4 c0[8], c1[8], fc0, fc1, fog_color; uint4 atest
 * (x alpha_ref as float bits, y alpha_func, z alpha_test_enable, w fog_enable);
 * uint4 alpha_only; float4 tex_scale[4]; uint4 tex_mode. */
#define PS_AT(m, off) \
    _Static_assert(offsetof(struct nv2a_ps_consts, m) == (off), "nv2a_ps_consts." #m " moved")
PS_AT(c0, 0);
PS_AT(c1, 128);
PS_AT(fc0, 256);
PS_AT(fc1, 272);
PS_AT(fog_color, 288);
PS_AT(alpha_ref, 304);
PS_AT(alpha_func, 308);
PS_AT(alpha_test_enable, 312);
PS_AT(fog_enable, 316);
PS_AT(alpha_only, 320);
PS_AT(tex_scale, 336);
PS_AT(tex_mode, 400);
#undef PS_AT
_Static_assert(sizeof(struct nv2a_ps_consts) == 416, "nv2a_ps_consts size changed");

/* The MSL PsConsts: the shared constants plus Metal's own trailing float4
 * tex_bias (per stage mip LOD bias, passed to sample() as bias()). Kept out
 * of nv2a_ps_consts, whose size the D3D11 backend pins to NV2APSConstants. */
typedef struct {
    struct nv2a_ps_consts b;
    float tex_bias[4];
    float bump_mat[4][4];   /* per stage: SET_TEXTURE_SET_BUMP_ENV_MAT 00 01 10 11 */
    float bump_lum[4][4];   /* per stage: x bump-env luminance scale, y offset */
    float hdr[4];           /* BO-110: x the colour bound of the HDR combiners (unread by SDR ones) */
} MsPsConsts;
_Static_assert(offsetof(MsPsConsts, tex_bias) == 416 && offsetof(MsPsConsts, bump_mat) == 432
               && offsetof(MsPsConsts, bump_lum) == 496 && offsetof(MsPsConsts, hdr) == 560
               && sizeof(MsPsConsts) == 576,
               "MsPsConsts tail moved (D3D8_MSL_PSCONSTS)");
_Static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "alpha_ref is read as uint bits");

/* One expanded vertex of a pass-through batch: attributes 0, 3, 4, 9..12 as
 * float4, in this order (the MSL reads vb[vid * VTX_F4 + k]). */
#define VTX_F4 7
static const int s_vtx_attr[VTX_F4] = {0, 3, 4, 9, 10, 11, 12};

static const char s_msl[] =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct VpConsts { float4 c[3]; uint4 u[2]; };\n"
    D3D8_MSL_PSCONSTS
    /* The emitters' VOut (d3d8_msl.h), so vs_pass links with a combiner
     * fs_main and a program's vs_main with fs_basic. */
    D3D8_MSL_VOUT
    /* Attribute 0 is already in surface pixels (x, y, z, rhw): c[0] is
     * (2/w, -2/h, 1/zmax, 0) and c[1] (-1, 1, 0, 0), as the D3D11
     * pass-through's vp_scale / vp_off. */
    "vertex VOut vs_pass(uint vid [[vertex_id]],\n"
    "                    const device float4 *vb [[buffer(0)]],\n"
    "                    constant VpConsts &vp [[buffer(1)]]) {\n"
    "    const device float4 *v = vb + vid * 7u;\n"
    "    VOut o;\n"
    "    o.pos = float4(v[0].xy * vp.c[0].xy + vp.c[1].xy,\n"
    "                   saturate(v[0].z * vp.c[0].z + vp.c[0].w), 1.0);\n"
    "    o.d0 = v[1]; o.d1 = v[2];\n"
    "    o.t0 = v[3]; o.t1 = v[4]; o.t2 = v[5]; o.t3 = v[6];\n"
    "    o.fog = 1.0; o.psize = 1.0;\n"
    "    return o;\n"
    "}\n"
    /* Batches without register combiners (and, in M1, all of them): the CPU
     * path's MODULATE, oD0 * T0 when stage 0 is on, then the alpha test
     * (D3DCMP numbering, on the 8-bit values the way the CPU compares them).
     * Line for line the D3D11 backend's s_ps_src. */
    "fragment float4 fs_basic(VOut i [[stage_in]],\n"
    "                         constant PsConsts &pc [[buffer(0)]],\n"
    "                         texture2d<float> tex0 [[texture(0)]],\n"
    "                         sampler samp0 [[sampler(0)]]) {\n"
    "    float4 c = i.d0;\n"
    "    if (pc.tex_mode.x == 4u) {\n"
    "        c *= saturate(i.t0);\n"
    "    } else if (pc.tex_mode.x != 0u) {\n"
    "        float2 uv = i.t0.xy;\n"
    "        if (pc.tex_mode.x == 1u && i.t0.w != 0.0 && i.t0.w != 1.0) uv /= i.t0.w;\n"
    "        c *= tex0.sample(samp0, uv * pc.tex_scale[0].xy, bias(pc.tex_bias.x));\n"
    "    }\n"
    "    if (pc.atest.z != 0u) {\n"
    "        float ref = as_type<float>(pc.atest.x);\n"
    "        uint a = (uint)(saturate(c.a) * 255.0 + 0.5), r = (uint)(ref * 255.0 + 0.5);\n"
    "        bool ok = true;\n"
    "        uint f = pc.atest.y;\n"
    "        if      (f == 1u) ok = false;\n"
    "        else if (f == 2u) ok = a <  r;\n"
    "        else if (f == 3u) ok = a == r;\n"
    "        else if (f == 4u) ok = a <= r;\n"
    "        else if (f == 5u) ok = a >  r;\n"
    "        else if (f == 6u) ok = a != r;\n"
    "        else if (f == 7u) ok = a >= r;\n"
    "        if (!ok) discard_fragment();\n"
    "    }\n"
    "    return c;\n"
    "}\n"
    /* A colour clear of part of a target (metal_on_clear): a full-screen
     * triangle under a scissor, the clear colour as it is. */
    "struct ClearV { float4 pos [[position]]; };\n"
    "vertex ClearV vs_clear(uint vid [[vertex_id]]) {\n"
    "    float2 t = float2((vid << 1) & 2, vid & 2);\n"
    "    ClearV o;\n"
    "    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "    return o;\n"
    "}\n"
    "fragment float4 fs_clear(constant float4 &c [[buffer(0)]]) { return c; }\n";

/* ---- device ---------------------------------------------------------------- */

static id<MTLDevice>               s_dev;
static id<MTLCommandQueue>         s_queue;
static id<MTLFunction>             s_vs_pass, s_fs_basic;
static id<MTLFunction>             s_vs_clear, s_fs_clear;
static id<MTLRenderPipelineState>  s_clear_pso[2];   /* partial clears: without, with depth */
static id<MTLRenderPipelineState>  s_clear_pso_hdr[2];   /* the same for half-float targets (BO-110) */
static MTLCompileOptions          *s_opts;     /* fast math off, kept for M2's shaders */
static id<MTLCommandBuffer>        s_cmd;      /* the frame's, begun lazily */
static id<MTLRenderCommandEncoder> s_enc;      /* open pass, or nil */
static int                         s_failed;
static int                         s_unified;  /* shared textures readable */

/* Frames in flight (BO-57). The flip commits the frame and goes on without
 * waiting for it; the GPU draws frame N while the executor builds N+1. What
 * the CPU writes for a frame and the GPU reads later -- the vertex and index
 * rings -- is one set per frame slot, and a slot is reused only once the
 * frame that last used it has completed: the wait moves from the end of
 * this frame to the start of the one after next, and is usually free by
 * then. UnleashedRecomp does the same with NUM_FRAMES = 2
 * (UnleashedRecomp/gpu/video.cpp: per-frame command lists, upload buffers
 * and a fence waited on before the slot is reused). Everything else the
 * CPU touches is made new (textures, targets), read only after an explicit
 * wait (write-back, readback), or tracked per command buffer (visibility
 * slots). RECOMP_DEBUG=metal_frames=1 waits at every flip as before. */
#define NUM_FRAMES 2
static id<MTLBuffer>        s_vbf[NUM_FRAMES], s_ibf[NUM_FRAMES];
static id<MTLCommandBuffer> s_frame_cmd[NUM_FRAMES];  /* each slot's last commit */
static int                  s_frame;                   /* the slot being built */
static int                  s_frames = -1;             /* in flight: 1 or 2 */
static uint64_t             s_frame_waits;             /* advances that blocked */

/* The current slot's vertex and index rings, reset after every wait (a slot
 * reuse or an overflow). */
static id<MTLBuffer> s_vb, s_ib;
static NSUInteger    s_vb_size, s_vb_pos, s_ib_size, s_ib_pos;

#define VIS_SLOTS 8192

static uint64_t s_presents, s_clears, s_draws, s_draws_skipped, s_draws_prog,
                s_draws_culled, s_draws_tex, s_tex_uploads, s_tex_rehash,
                s_tex_changed, s_flushes, s_writebacks, s_cmd_errors;

/* ---- render targets: one per guest surface --------------------------------- */

/* Depth/stencil buffers, one per zeta surface and size, as the D3D11
 * backend's: Depth32Float_Stencil8, so the test compares the same normalised
 * float z (depth units / zmax) the CPU path keeps, and the 8 stencil bits
 * are Z24S8's. Never read back to guest memory. */
#define DT_MAX 4
typedef struct {
    uint32_t       zeta, w, h;
    uint32_t       hw, hh;     /* host size: w x h at render.scale */
    id<MTLTexture> tex;
    uint64_t       last_use;
    int            fresh;      /* contents undefined: the next pass clears */
} DepthTarget;

/* Burnout 3's Crash mode draws 14 surfaces a frame (the 640x480 pair, a
 * 640x480 and a 320x240 work target, two 160x120, seven 128x128 and a
 * 64x32). With 8 slots the LRU evicted every one of them every frame: a new
 * texture seeded from guest memory each time, a GPU wait and read-back for
 * each presented one evicted, and an off-screen one evicted before it was
 * sampled lost what was drawn into it. xemu keeps its surfaces in an
 * unbounded list (hw/xbox/nv2a/pgraph surface cache); 32 slots hold any frame
 * seen so far, and a slot costs only its texture. */
#define RT_MAX 32
typedef struct {
    uint32_t       offset;     /* color_offset, as the title set it */
    uint32_t       addr;       /* its guest address (nv2a_pb_dma_resolve) */
    uint32_t       w, h;       /* guest size: the cache key, the mapping */
    uint32_t       hw, hh;     /* host size: the texture (render.scale) */
    id<MTLTexture> tex;
    id<MTLTexture> opaque;     /* tex with alpha read as 1 (rt_texture, 16-bit) */
    uint64_t       last_use;
    DepthTarget   *dt;         /* the depth buffer its passes attach, or NULL */
    uint32_t       pitch;      /* the guest pitch it was made with */
    int            swz;        /* made as a swizzled surface (format type 2) */
    int            dirty;      /* drawn since its last write-back */
    uint64_t       gen;        /* s_rt_seq at its last clear or draw */
    int            presented;  /* ever picked by present_target */
    struct nv2a_rt_own own;    /* is the guest memory under it still its own */
} RenderTarget;

static RenderTarget  s_rt[RT_MAX];
static uint64_t      s_rt_seq;   /* bumped by every pass begun and draw encoded */
static RenderTarget *s_enc_rt;   /* the target s_enc draws into */
static DepthTarget  *s_enc_dt;   /* and its depth attachment, or NULL */
static DepthTarget   s_dt[DT_MAX];
static uint64_t      s_tick;
static uint64_t      s_zclears, s_draws_z, s_rtt_binds, s_self_copies;
static uint64_t      s_stale_drops, s_retired;
static uint64_t      s_grows, s_partial_clears;   /* rt_grow, metal_on_clear */

/* ---- HDR (BO-110 prototype, RECOMP_DEBUG=hdr) ---------------------------------
 *
 * The colour targets are RGBA16Float instead of BGRA8Unorm. Blending into a
 * float target does not clamp, so where the title stacks light additively
 * (flames, sparks, lights, the sun and its flare, the bloom's add) the value
 * passes 1.0 by itself. The register combiners clamp each draw to 1.0 on the
 * hardware; the HDR variant of their shader (d3d8_combiners_msl.c) takes its
 * bound from pc.hdr.x: s_hdr_hi between the title's scene marks (the 3D of a
 * stage view, nv2a_pb_stream_tag() == NV2A_HDR_TAG_SCENE), 1.0 everywhere
 * else (HUD, menus, front end, loading screens: paper white). With it off
 * nothing here runs: the formats, the shaders and the pipelines are the
 * SDR ones. */
#define NV2A_HDR_TAG_SCENE 1u
static int   s_hdr_req = -1;      /* RECOMP_DEBUG=hdr asked for (unread: -1) */
static int   s_hdr;               /* the targets are half float now */
static int   s_hdr_ab;            /* RECOMP_DEBUG=hdr_ab=n */
static float s_hdr_hi = 64.0f;
static uint64_t s_hdr_draws[2];   /* HDR draws: UI-bound, scene-bound */
/* The scene snapshot: the target the view's 3D ended in, copied when the
 * first draw after the title's hdr_ui mark comes (or at the flip), before
 * the HUD is drawn over it. The presenter tells HUD pixels (the frame differs
 * from the snapshot) from scene pixels, so paper white, HUD white and peak
 * apply live, also to a frame shown again under the overlay. */
static id<MTLTexture> s_hdr_scene;
static uint32_t       s_hdr_scene_of;     /* the snapshot's surface offset */
static int            s_hdr_scene_ok;     /* taken this frame */
static RenderTarget  *s_hdr_pend;         /* the last scene draw's target, not yet taken */
static uint64_t       s_hdr_snaps;
extern void *xbox_FramebufferWindowBackScene(uint32_t w, uint32_t h, int valid);
static void hdr_texlog(const struct nv2a_pb_gpu *g);
static void hdr_snapshot(void);
static int  s_hdr_watch_hit;   /* hdr_watch: the watched texture drawn this frame */
#define RT_FMT (s_hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm)

static int hdr_requested(void)
{
    if (s_hdr_req < 0) {
        const char *e = recomp_env(RENV_HDR), *a = recomp_env(RENV_HDR_AB),
                   *h = recomp_env(RENV_HDR_HI);
        s_hdr_req = e && *e && strcmp(e, "0") != 0;
        s_hdr_ab = s_hdr_req && a ? atoi(a) : 0;
        if (h && atof(h) >= 1.0)
            s_hdr_hi = (float)atof(h);
        s_hdr = s_hdr_req || nv2a_host_hdr_active();
        nv2a_texrep_set_hdr(s_hdr);     /* BO-203: the title's HDR pictures */
        if (s_hdr_req)
            fprintf(stderr, LOGP "HDR (BO-110 prototype): half-float targets, scene bound"
                    " %.1f%s\n", s_hdr_hi, s_hdr_ab ? ", A/B" : "");
    }
    return s_hdr_req;
}

static int tex_is_half(id<MTLTexture> t)
{
    return [t pixelFormat] == MTLPixelFormatRGBA16Float;
}

static inline uint16_t hdr_f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static inline float hdr_h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }

/* BGRA8 rows (bpr bytes apart) into a half-float texture's region. */
static void tex_put_half(id<MTLTexture> t, MTLRegion r, NSUInteger level, NSUInteger slice,
                         const void *src, NSUInteger bpr)
{
    size_t w = r.size.width, h = r.size.height, x, y;
    uint16_t *o = (uint16_t *)malloc(w * h * 8u);
    static const float k = 1.0f / 255.0f;
    if (!o)
        return;
    for (y = 0; y < h; y++) {
        const uint8_t *p = (const uint8_t *)src + y * bpr;
        uint16_t *q = o + y * w * 4u;
        for (x = 0; x < w; x++, p += 4, q += 4) {
            q[0] = hdr_f2h(p[2] * k);
            q[1] = hdr_f2h(p[1] * k);
            q[2] = hdr_f2h(p[0] * k);
            q[3] = hdr_f2h(p[3] * k);
        }
    }
    [t replaceRegion:r mipmapLevel:level slice:slice withBytes:o bytesPerRow:w * 8u
       bytesPerImage:w * h * 8u];
    free(o);
}

static inline uint8_t hdr_unorm8(float f)
{
    if (!(f > 0.0f)) return 0;
    if (f >= 1.0f) return 255;
    return (uint8_t)lrintf(f * 255.0f);
}

/* A half-float texture's region as BGRA8 rows (bpr bytes apart), clamped to
 * [0, 1] as a BGRA8Unorm target would hold it. */
static void tex_get_half(id<MTLTexture> t, void *dst, NSUInteger bpr, MTLRegion r)
{
    size_t w = r.size.width, h = r.size.height, x, y;
    uint16_t *in = (uint16_t *)malloc(w * h * 8u);
    if (!in)
        return;
    [t getBytes:in bytesPerRow:w * 8u fromRegion:r mipmapLevel:0];
    for (y = 0; y < h; y++) {
        uint8_t *p = (uint8_t *)dst + y * bpr;
        const uint16_t *q = in + y * w * 4u;
        for (x = 0; x < w; x++, p += 4, q += 4) {
            p[0] = hdr_unorm8(hdr_h2f(q[2]));
            p[1] = hdr_unorm8(hdr_h2f(q[1]));
            p[2] = hdr_unorm8(hdr_h2f(q[0]));
            p[3] = hdr_unorm8(hdr_h2f(q[3]));
        }
    }
    free(in);
}

/* getBytes of a colour target as BGRA8, whatever its format. */
static void rt_get_bytes(id<MTLTexture> t, void *dst, NSUInteger bpr, MTLRegion r)
{
    if (tex_is_half(t))
        tex_get_half(t, dst, bpr, r);
    else
        [t getBytes:dst bytesPerRow:bpr fromRegion:r mipmapLevel:0];
}

/* Visibility tests (see "Visibility tests" below): the slot counting on the
 * open pass, opened by begin_pass and closed here. */
static void occ_pass_begin(void);
static void occ_pass_end(void);
static void occ_after_wait(void);
static id<MTLBuffer> s_vis;      /* visibility result slots, 8 bytes each */
static struct OccQuery *s_occ_q;     /* the query counting now, or NULL */
static int s_occ_slot = -1;          /* its slot on s_enc */

/* RECOMP_TRACE=metal_prof: where on_flip's time goes, averaged over 300
 * flips: the wait for the frame's last command buffer, the GPU time of every
 * command buffer committed since the last flip, the host readback (scale > 1),
 * the guest write-back, the readback-mode window copy and hand-off, and the
 * layer-mode slot blit. Prints only. */
static int s_prof = -1;
#define PROF_CMDS 128
static id<MTLCommandBuffer> s_prof_cmd[PROF_CMDS];
static int s_prof_ncmd;
static struct {
    uint64_t n, flip, wait, gpu, hostrb, wb, winrb, winpush, blit, ncmd, hash;
    uint64_t texh, texh_bytes, texh_n, texup, texup_n;   /* tex_bind */
} s_pa;

/* The last committed command buffer: a sync between flips waits for it when
 * nothing is open (a GET_REPORT commit may still be running). */
static id<MTLCommandBuffer> s_last_cmd;
static uint64_t s_decode_syncs;

static uint64_t prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int prof_on(void)
{
    if (s_prof < 0)
        s_prof = recomp_env(RENV_METAL_PROF) != NULL;
    return s_prof;
}

/* A flip-phase timer: no clock reads unless metal_prof is on. */
static uint64_t prof_t(void)
{
    return prof_on() ? prof_now() : 0;
}

static void prof_add(uint64_t *acc, uint64_t t0)
{
    if (prof_on())
        *acc += prof_now() - t0;
}

static void end_encoder(void)
{
    if (s_enc) {
        occ_pass_end();
        [s_enc endEncoding];
        [s_enc release];
        s_enc = nil;
    }
    s_enc_rt = NULL;
    s_enc_dt = NULL;
}

/* End the open pass, commit the frame's command buffer and (wait) block
 * until the GPU is done with it. After a wait the rings are free again. */
static void flush(int wait)
{
    end_encoder();
    if (!s_cmd) {
        if (wait) {
            /* Nothing open, but the last commit (a GET_REPORT's, the
             * flip's) may still read the rings: wait for it first. */
            if (s_last_cmd && [s_last_cmd status] < MTLCommandBufferStatusCompleted)
                [s_last_cmd waitUntilCompleted];
            s_vb_pos = 0;
            s_ib_pos = 0;
            s_flushes++;
            occ_after_wait();
        }
        return;
    }
    [s_cmd commit];
    [s_last_cmd release];
    s_last_cmd = [s_cmd retain];
    [s_frame_cmd[s_frame] release];
    s_frame_cmd[s_frame] = [s_cmd retain];
    if (wait) {
        [s_cmd waitUntilCompleted];
        if ([s_cmd status] == MTLCommandBufferStatusError) {
            s_cmd_errors++;
            if (s_cmd_errors <= 4)
                XR_ERROR( LOGP "command buffer error: %s\n",
                        [[[s_cmd error] localizedDescription] UTF8String]);
        }
        s_vb_pos = 0;
        s_ib_pos = 0;
    }
    if (prof_on() && s_prof_ncmd < PROF_CMDS)
        s_prof_cmd[s_prof_ncmd++] = [s_cmd retain];
    [s_cmd release];
    s_cmd = nil;
    s_flushes++;
    if (wait)
        occ_after_wait();
}

static int frames_in_flight(void)
{
    if (s_frames < 0) {
        const char *e = recomp_env(RENV_METAL_FRAMES);
        int n = e ? atoi(e) : NUM_FRAMES;
        s_frames = n < 1 ? 1 : n > NUM_FRAMES ? NUM_FRAMES : n;
    }
    return s_frames;
}

static void occ_retire_poll(void);

/* At the flip, after its commit: the next frame takes the other slot, once
 * the frame that last used it is done (the only wait left in a flip). */
static void frame_advance(void)
{
    id<MTLCommandBuffer> c;

    if (frames_in_flight() < 2) {
        flush(1);
        return;
    }
    flush(0);
    s_frame = (s_frame + 1) % NUM_FRAMES;
    c = s_frame_cmd[s_frame];
    if (c) {
        if ([c status] < MTLCommandBufferStatusCompleted) {
            [c waitUntilCompleted];
            s_frame_waits++;
        }
        [c release];
        s_frame_cmd[s_frame] = nil;
    }
    s_vb = s_vbf[s_frame];
    s_ib = s_ibf[s_frame];
    s_vb_pos = 0;
    s_ib_pos = 0;
    occ_retire_poll();
}

static void ensure_cmd(void)
{
    if (!s_cmd)
        s_cmd = [[s_queue commandBuffer] retain];
}

/* RECOMP_TRACE=metal_frame: the flip's passes in order, one line a flip:
 *   [metal] frame 1234: 5 passes 2 copies | 0x03A52000 640x480 C 212 | ...
 * a pass is a target's offset and guest size, C (cleared) or L (loaded), its
 * draws and, after '>', the host size of the target (render scale applied) and
 * "i<n>", the indices those draws sent (what a level-of-detail change moves
 * when the draws stay the same); "copy WxH" a self copy (the target copied to a texture so that
 * a draw into it can sample it). What a title's post effect costs in passes
 * and full-screen copies is read off two such lines. */
#define MF_MAX 64
static struct { uint32_t off, w, h, hw, hh, draws, idx; char kind; } s_mf[MF_MAX];
static int s_mf_n, s_mf_over, s_mf_on = -1;

static void mf_add(char kind, const RenderTarget *rt)
{
    if (s_mf_on < 0)
        s_mf_on = recomp_env(RENV_METAL_FRAME) != NULL;
    if (!s_mf_on)
        return;
    if (s_mf_n == MF_MAX) {
        s_mf_over++;
        return;
    }
    s_mf[s_mf_n].off = rt->offset;
    s_mf[s_mf_n].w = rt->w;
    s_mf[s_mf_n].h = rt->h;
    s_mf[s_mf_n].hw = rt->hw;
    s_mf[s_mf_n].hh = rt->hh;
    s_mf[s_mf_n].draws = 0;
    s_mf[s_mf_n].idx = 0;
    s_mf[s_mf_n++].kind = kind;
}

static void mf_flip(void)
{
    char line[MF_MAX * 40 + 96];
    int i, n = 0, passes = 0, copies = 0;
    if (s_mf_on <= 0)
        return;
    for (i = 0; i < s_mf_n; i++)
        if (s_mf[i].kind == 'X') copies++; else passes++;
    n += snprintf(line + n, sizeof line - (size_t)n, "frame %llu: %d passes %d copies",
                  (unsigned long long)s_presents, passes + s_mf_over, copies);
    for (i = 0; i < s_mf_n; i++) {
        if (s_mf[i].kind == 'X')
            n += snprintf(line + n, sizeof line - (size_t)n, " | copy %ux%u",
                          s_mf[i].w, s_mf[i].h);
        else
            n += snprintf(line + n, sizeof line - (size_t)n, " | 0x%08X %ux%u %c %u >%ux%u i%u",
                          s_mf[i].off, s_mf[i].w, s_mf[i].h, s_mf[i].kind, s_mf[i].draws,
                          s_mf[i].hw, s_mf[i].hh, s_mf[i].idx);
    }
    fprintf(stderr, LOGP "%s\n", line);
    s_mf_n = s_mf_over = 0;
}

/* Open a pass on rt: with `clear`, loadAction Clear to c; else Load. The
 * pass attaches rt->dt when it has one; zmask (CLEAR_SURFACE's bit 0 depth,
 * bit 1 stencil) clears it to z / st, and a fresh buffer clears to the far
 * plane and stencil 0, where the CPU path starts its own. */
static void begin_pass(RenderTarget *rt, int clear, const double c[4],
                       unsigned zmask, float z, uint32_t st)
{
    MTLRenderPassDescriptor *rp;
    DepthTarget *dt = rt->dt;
    end_encoder();
    ensure_cmd();
    rt->dirty = 1;
    rt->gen = ++s_rt_seq;
    nv2a_rt_own_drawn(&rt->own, (uint32_t)s_presents);
    rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = rt->tex;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (clear) {
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(c[0], c[1], c[2], c[3]);
    } else {
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    }
    if (dt) {
        int zc = (zmask & 1u) || dt->fresh, sc = (zmask & 2u) || dt->fresh;
        rp.depthAttachment.texture = dt->tex;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.depthAttachment.loadAction = zc ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.depthAttachment.clearDepth = (zmask & 1u) ? z : 1.0;
        rp.stencilAttachment.texture = dt->tex;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
        rp.stencilAttachment.loadAction = sc ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.stencilAttachment.clearStencil = (zmask & 2u) ? (st & 0xFFu) : 0u;
        dt->fresh = 0;
    }
    rp.visibilityResultBuffer = s_vis;
    s_enc = [[s_cmd renderCommandEncoderWithDescriptor:rp] retain];
    s_enc_rt = rt;
    s_enc_dt = dt;
    occ_pass_begin();
    mf_add(clear ? 'C' : 'L', rt);
}

static void begin_encoder(RenderTarget *rt, int clear, const double c[4])
{
    begin_pass(rt, clear, c, 0, 1.0f, 0);
}

static int init(void)
{
    static int tried;
    NSError *err = nil;
    MTLCompileOptions *opts;
    id<MTLLibrary> lib;

    if (tried)
        return !s_failed;
    tried = 1;
    s_failed = 1;

    @autoreleasepool {
        void *wdev = NULL, *wq = NULL;
        /* The layer window was made before the guest started, with its own
         * device and queue: one device so its slots take this backend's
         * blits, one queue so a present pass runs after the blit it shows. */
        if (xbox_FramebufferWindowMetal(&wdev, &wq) && wdev && wq) {
            s_dev = [(id<MTLDevice>)wdev retain];
            s_queue = [(id<MTLCommandQueue>)wq retain];
            s_layer = 1;
        } else {
            s_dev = MTLCreateSystemDefaultDevice();
            if (s_dev)
                s_queue = [s_dev newCommandQueue];
        }
        if (!s_dev || !s_queue) {
            fprintf(stderr, LOGP "no Metal device\n");
            return 0;
        }
        s_unified = [s_dev hasUnifiedMemory] ? 1 : 0;

        opts = [[MTLCompileOptions alloc] init];
        /* Fast math off: the CPU path is IEEE; reassociation and flushed
         * denormals would move pixels across the golden tolerance. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
            opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        }
        s_opts = opts;
        lib = [s_dev newLibraryWithSource:[NSString stringWithUTF8String:s_msl]
                                  options:opts error:&err];
        if (!lib) {
            XR_ERROR( LOGP "MSL compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            return 0;
        }
        s_vs_pass = [lib newFunctionWithName:@"vs_pass"];
        s_fs_basic = [lib newFunctionWithName:@"fs_basic"];
        s_vs_clear = [lib newFunctionWithName:@"vs_clear"];
        s_fs_clear = [lib newFunctionWithName:@"fs_clear"];
        [lib release];
        if (!s_vs_pass || !s_fs_basic || !s_vs_clear || !s_fs_clear) {
            XR_ERROR( LOGP "MSL functions missing\n");
            return 0;
        }

        /* The vertex ring is bounded by the 16-bit vertex range (65536
         * vertices are 1 MB per float4 stream); the index ring by the cap,
         * so any batch the walker accepts fits: three 32-bit list entries
         * per index at most. */
        s_vb_size = 16u << 20;
        s_ib_size = NV_MAX_INDICES * 3u * 4u;
        for (int k = 0; k < NUM_FRAMES; k++) {
            s_vbf[k] = [s_dev newBufferWithLength:s_vb_size
                                          options:MTLResourceStorageModeShared];
            s_ibf[k] = [s_dev newBufferWithLength:s_ib_size
                                          options:MTLResourceStorageModeShared];
        }
        s_vb = s_vbf[0];
        s_ib = s_ibf[0];
        s_vis = [s_dev newBufferWithLength:VIS_SLOTS * 8u
                                   options:MTLResourceStorageModeShared];
        if (!s_vbf[0] || !s_ibf[0] || !s_vbf[NUM_FRAMES - 1] || !s_ibf[NUM_FRAMES - 1]
                || !s_vis) {
            XR_ERROR( LOGP "buffer creation failed\n");
            return 0;
        }
        hdr_requested();
        if (!pc_start()) {
            XR_ERROR( LOGP "pipeline cache allocation failed\n");
            return 0;
        }
        XR_NOTICE( LOGP "device up: %s (unified memory %d), fast math off; present %s,"
                " write-back %s\n", [[s_dev name] UTF8String], s_unified,
                s_layer ? "CAMetalLayer slots" : "texture readback",
                wb_always() ? "every flip" : "on demand");
    }
    s_failed = 0;
    /* graphics.ui_scale switched on in the overlay: the loaded 2D textures
     * are prepared while the title is held (scale_prepare_loaded). */
    xbox_HostPauseIdleFn(scale_prepare_loaded);
    return 1;
}

/* ---- surfaces --------------------------------------------------------------- */

static void rt_sync(RenderTarget *rt, uint32_t pitch);
static void rt_free(RenderTarget *rt);

/* ---- render-target ownership (nv2a_backend_common.h) -------------------------
 *
 * A target is the title's surface only while the guest memory under it holds
 * what the backend last knew there. A title that frees a target's memory and
 * loads something else into it (BLiNX 2's hub leaves two 512x512 targets
 * whose megabytes stage 1 refills with textures and tables) would otherwise
 * get the old image sampled in place of its texels, or written back over
 * them. A stale target is dropped without a write-back. A/B:
 * RECOMP_DEBUG=rt_alias_check=0. */
static int alias_check_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = recomp_env(RENV_RT_ALIAS_CHECK);
        on = !(e && e[0] == '0');
    }
    return on;
}

static void rt_own_reset(RenderTarget *rt)
{
    uint64_t t;
    if (!alias_check_on() || !rt->pitch)
        return;
    guard_open();   /* our own read, not a title read */
    t = prof_t();
    nv2a_rt_own_reset(&rt->own, (const uint8_t *)xbox_GetMemoryOffset(), rt->addr,
                      rt->pitch, rt->h, (uint32_t)s_presents);
    prof_add(&s_pa.hash, t);
}

/* Drop rt when the title has rewritten its memory; 1 if it was dropped.
 * `why` names the caller for the log: bind, sync, evict, overlap, reuse. */
static int rt_drop_if_stale(RenderTarget *rt, const char *why)
{
    static int said;
    uint64_t t;
    int stale;

    if (!alias_check_on() || !rt->tex || !rt->pitch)
        return 0;
    guard_open();
    t = prof_t();
    stale = nv2a_rt_own_stale(&rt->own, (const uint8_t *)xbox_GetMemoryOffset(), rt->addr,
                              rt->pitch, rt->h, (uint32_t)s_presents);
    prof_add(&s_pa.hash, t);
    if (!stale)
        return 0;
    s_stale_drops++;
    if (said < 20 || recomp_env(RENV_METAL_TRACE)) {
        said++;
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target dropped (%s): the title"
                " rewrote its %u bytes; last drawn at flip %u, now %llu%s\n",
                rt->offset, rt->w, rt->h, why, rt->pitch * rt->h, rt->own.draw_flip,
                (unsigned long long)s_presents, said == 20 ? " (further drops: the"
                " present summary)" : "");
    }
    rt->dirty = 0;   /* the title's bytes win: no write-back */
    rt_free(rt);
    return 1;
}

static void rt_free(RenderTarget *rt)
{
    /* A presented target's bytes are what a new target at that surface seeds
     * from, as when every flip wrote it back; off-screen ones never were. */
    if (rt->dirty && rt->presented && rt_drop_if_stale(rt, "evict"))
        return;
    if (rt->dirty && rt->presented)
        rt_sync(rt, rt->pitch);
    if (s_enc_rt == rt)
        end_encoder();
    [rt->opaque release];
    [rt->tex release];
    memset(rt, 0, sizeof *rt);
}

/* The depth buffer for the zeta surface now set, at w x h; made on first
 * use (fresh), the least recently used one giving way when all are taken. */
static DepthTarget *depth_target(uint32_t w, uint32_t h)
{
    uint32_t zeta = nv2a_pb_vsh_state()->zeta_offset;
    DepthTarget *e = NULL, *lru = &s_dt[0];
    MTLTextureDescriptor *td;
    int i;

    for (i = 0; i < DT_MAX; i++) {
        DepthTarget *d = &s_dt[i];
        if (d->tex && d->zeta == zeta && d->w == w && d->h == h) {
            d->last_use = s_tick;
            return d;
        }
        if (!d->tex && !e)
            e = d;
        if (d->last_use < lru->last_use)
            lru = d;
    }
    if (!e) {
        e = lru;
        if (s_enc_dt == e)
            end_encoder();
        for (i = 0; i < RT_MAX; i++)
            if (s_rt[i].dt == e)
                s_rt[i].dt = NULL;
        [e->tex release];
        memset(e, 0, sizeof *e);
    }
    nv2a_host_size(w, h, nv2a_host_render_scale(), METAL_MAX_DIM, &e->hw, &e->hh);
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                            width:e->hw height:e->hh mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    e->tex = [s_dev newTextureWithDescriptor:td];
    if (!e->tex)
        return NULL;
    e->zeta = zeta;
    e->w = w;
    e->h = h;
    e->last_use = s_tick;
    e->fresh = 1;
    if (e->hw != w)
        fprintf(stderr, LOGP "zeta 0x%08X: %ux%u depth buffer (%ux%u host)\n",
                zeta, w, h, e->hw, e->hh);
    else
        fprintf(stderr, LOGP "zeta 0x%08X: %ux%u depth buffer\n", zeta, w, h);
    return e;
}

/* A BGRA8 colour texture of hw x hh host pixels, as every target is made. */
static id<MTLTexture> rt_texture_new(uint32_t hw, uint32_t hh)
{
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:RT_FMT
                                                           width:hw height:hh
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    /* Managed only for a discrete GPU (Intel Macs); deprecated on 27. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    td.storageMode = s_unified ? MTLStorageModeShared : MTLStorageModeManaged;
#pragma clang diagnostic pop
    return [s_dev newTextureWithDescriptor:td];
}

/* Swizzled render targets (BO-24). A surface of format type 2 lies in guest
 * memory in the NV2A's Z order (d3d8_swizzle.h), as the CPU path draws it
 * (surface_swizzle_index): the seed reads it so, and a write-back stores it
 * so, through a linear staging copy. Only a power-of-two w x h is a swizzled
 * layout; anything else is left linear, as before. */
static int rt_swz_ok(int swz, uint32_t w, uint32_t h)
{
    return swz && w && h && !(w & (w - 1)) && !(h & (h - 1));
}

static uint8_t *s_swz_lin;    /* staging: one w x h target, linear */
static size_t   s_swz_cap;

static uint8_t *swz_staging(size_t bytes)
{
    if (s_swz_cap < bytes) {
        uint8_t *n = (uint8_t *)realloc(s_swz_lin, bytes);
        if (!n)
            return NULL;
        s_swz_lin = n;
        s_swz_cap = bytes;
    }
    return s_swz_lin;
}

/* tex (w x h guest, hw x hh host) from the guest's bytes at addr when the
 * surface is 32-bit, as the CPU path draws over whatever memory holds; 0 when
 * it is not (the texture is left as made). */
static int rt_seed(id<MTLTexture> tex, uint32_t addr, uint32_t pitch, uint32_t w,
                    uint32_t h, uint32_t hw, uint32_t hh, int swz)
{
    const uint8_t *src = (const uint8_t *)xbox_GetMemoryOffset() + addr;

    if (pitch < w * 4u)
        return 0;
    guard_open();   /* our own read of the seed is not a title read */
    if (rt_swz_ok(swz, w, h)) {
        uint8_t *lin = swz_staging((size_t)w * h * 4u);
        if (!lin)
            return 0;
        xbox_unswizzle_rect(lin, src, w, h, 4);
        src = lin;
        pitch = w * 4u;
    }
    if (tex_is_half(tex)) {
        if (hw == w) {
            tex_put_half(tex, MTLRegionMake2D(0, 0, w, h), 0, 0, src, pitch);
        } else {
            unsigned n = hw / w;
            uint8_t *up = (uint8_t *)malloc((size_t)hw * hh * 4u);
            if (up) {
                nv2a_upscale_nearest32(src, pitch, w, h, n, up, hw * 4u);
                tex_put_half(tex, MTLRegionMake2D(0, 0, hw, hh), 0, 0, up, hw * 4u);
                free(up);
            }
        }
    } else if (hw == w) {
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                 withBytes:src bytesPerRow:pitch];
    } else {
        /* Scaled: the guest bytes, each pixel an n x n block. */
        unsigned n = hw / w;
        uint8_t *up = (uint8_t *)malloc((size_t)hw * hh * 4u);
        if (up) {
            nv2a_upscale_nearest32(src, pitch, w, h, n, up, hw * 4u);
            [tex replaceRegion:MTLRegionMake2D(0, 0, hw, hh) mipmapLevel:0
                     withBytes:up bytesPerRow:hw * 4u];
            free(up);
        }
    }
    return 1;
}

/* rt made w x h (guest pixels), what it holds kept at its top left: the
 * D3D11 backend's rt_grow. The new area is seeded from the guest bytes, as a
 * new target is; replaceRegion runs now and the blit of the old contents
 * later on the GPU, so the blit wins where both write. A blit cannot
 * rescale: when the larger size takes another host factor (past
 * METAL_MAX_DIM), rt is written back first and the whole texture seeded. */
static int rt_grow(RenderTarget *rt, uint32_t w, uint32_t h)
{
    static int said;
    id<MTLTexture> tex;
    uint32_t hw, hh;
    unsigned n;
    int keep;

    n = nv2a_host_size(w, h, nv2a_host_render_scale(), METAL_MAX_DIM, &hw, &hh);
    keep = rt->hw == rt->w * n && rt->hh == rt->h * n;
    if (!keep)
        rt_sync(rt, rt->pitch);
    /* An encoder must not outlive its attachment's swap, and the passes
     * below need none open. */
    end_encoder();
    tex = rt_texture_new(hw, hh);
    if (!tex)
        return 0;
    if (!rt_seed(tex, rt->addr, rt->pitch, w, h, hw, hh, rt->swz)) {
        /* A new texture's contents are undefined: a 16-bit surface's margin
         * starts black, as D3D11's does. */
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        id<MTLRenderCommandEncoder> e;
        rp.colorAttachments[0].texture = tex;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        ensure_cmd();
        e = [s_cmd renderCommandEncoderWithDescriptor:rp];
        [e endEncoding];
    }
    if (keep) {
        id<MTLBlitCommandEncoder> b;
        ensure_cmd();
        b = [s_cmd blitCommandEncoder];
        [b copyFromTexture:rt->tex sourceSlice:0 sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(rt->hw, rt->hh, 1)
                 toTexture:tex destinationSlice:0 destinationLevel:0
         destinationOrigin:MTLOriginMake(0, 0, 0)];
        [b endEncoding];
    }
    if (said++ < 20)
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target grown to %ux%u\n",
                rt->offset, rt->w, rt->h, w, h);
    [rt->tex release];   /* an encoded command keeps its own reference */
    [rt->opaque release];
    rt->opaque = nil;
    rt->tex = tex;
    rt->w = w;
    rt->h = h;
    rt->hw = hw;
    rt->hh = hh;
    /* The cached depth buffer is the old size; begin_pass would attach it to
     * the larger colour texture. The next depth draw attaches one at the
     * grown size (depth_target), fresh. */
    rt->dt = NULL;
    rt_own_reset(rt);
    s_grows++;
    return 1;
}

/* rt (free) made as the target for the surface at offset, w x h guest,
 * seeded from the guest's bytes there. */
static int rt_make(RenderTarget *rt, uint32_t offset, uint32_t w, uint32_t h, uint32_t pitch,
                   int swz)
{
    uint32_t hw, hh;

    nv2a_host_size(w, h, nv2a_host_render_scale(), METAL_MAX_DIM, &hw, &hh);
    rt->tex = rt_texture_new(hw, hh);
    if (!rt->tex)
        return 0;
    rt->offset = offset;
    rt->addr = nv2a_pb_dma_resolve(offset);
    rt->w = w;
    rt->h = h;
    rt->hw = hw;
    rt->hh = hh;
    rt->pitch = pitch;
    rt->swz = swz;
    rt->last_use = s_tick;
    rt_own_reset(rt);   /* the bytes the seed reads, or would */
    rt_seed(rt->tex, rt->addr, pitch, w, h, hw, hh, rt->swz);
    if (hw != w)
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target (%ux%u host)\n",
                offset, w, h, hw, hh);
    else
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target\n", offset, w, h);
    return 1;
}

/* The render target for the surface at color_offset, at least the size of
 * its clip extent (the D3D11 backend's rule), made on first use. A new one
 * starts from the guest's bytes (rt_seed). */
static RenderTarget *surface(void)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    uint32_t w = g->clip_x + g->clip_w, h = g->clip_y + g->clip_h;
    RenderTarget *free_rt = NULL, *lru = &s_rt[0];
    int i, swz;

    if (!g->color_offset || !w || !h || w > 4096 || h > 4096)
        return NULL;
    swz = ((g->format >> 8) & 0xF) == 2;    /* the layout guest memory holds it in (not the identity: the title draws one surface both ways, and a target per layout retired and wrote back every frame) */
    s_tick++;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && rt->offset == g->color_offset && rt->w == w && rt->h == h) {
            /* First use this flip of a target the title may have rewritten:
             * a stale one is made again from the title's bytes, which the
             * GPU would draw over. */
            if (rt_drop_if_stale(rt, "reuse"))
                break;
            rt->last_use = s_tick;
            rt->swz = swz;   /* the guest layout for its next seed or write-back */
            return rt;
        }
    }
    /* The same surface under another clip extent: one target per surface,
     * as large as any clip it was drawn with (D3D11's rule, toolkit
     * 4773a01). Burnout 3 draws its 640x480 back buffer with clips of
     * 640x464, 623x401, 159x344 and more in one frame; a target per extent
     * made the overlap rule below write back and retire one at every clip
     * change (a GPU wait, and at render.scale > 1 the box filter's loss),
     * and a flip found no target of the frame's size. Draws map guest pixels
     * to NDC through rt->w/h, so a larger target keeps every position; a
     * clear covers only its clip and clear rect (metal_on_clear) and a flip
     * shows only the frame's extent (present_extent). Another pitch at the
     * same offset is a new allocation, not another clip: it gets a target of
     * its own and the overlap rule retires this one. */
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex || rt->offset != g->color_offset || rt->pitch != g->pitch)
            continue;
        if (rt_drop_if_stale(rt, "reuse"))
            break;
        if ((rt->w >= w && rt->h >= h)
                || rt_grow(rt, rt->w > w ? rt->w : w, rt->h > h ? rt->h : h)) {
            rt->last_use = s_tick;
            rt->swz = swz;   /* the guest layout for its next seed or write-back */
            return rt;
        }
        break;
    }
    /* No two live targets share memory: one the new target overlaps is
     * written back first (the seed below then reads what was drawn) or,
     * if the title rewrote it, dropped. */
    if (alias_check_on()) {
        uint32_t addr = nv2a_pb_dma_resolve(g->color_offset);
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, addr, g->pitch * h))
                continue;
            if (rt_drop_if_stale(rt, "overlap"))
                continue;
            rt_sync(rt, rt->pitch);
            if (s_retired++ < 20)
                fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target retired:"
                        " 0x%08X %ux%u overlaps it\n", rt->offset, rt->w, rt->h,
                        g->color_offset, w, h);
            rt_free(rt);
        }
    }
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex && !free_rt)
            free_rt = rt;
        if (rt->last_use < lru->last_use)
            lru = rt;
    }
    if (!free_rt) {
        /* Out of slots: the least recently used target goes, and what it
         * holds goes to guest memory first, presented or not. A texture
         * that samples it later decodes those bytes, and a target made
         * there again seeds from them; without the write-back an
         * off-screen pass (a blur level, a scene copy) read back whatever
         * the memory held before -- black, under Burnout 3's motion blur,
         * a dark vignette over the race (BO-47). */
        free_rt = lru;
        if (free_rt->dirty && !rt_drop_if_stale(free_rt, "evict"))
            rt_sync(free_rt, free_rt->pitch);
        if (free_rt->tex)
            rt_free(free_rt);
    }
    return rt_make(free_rt, g->color_offset, w, h, g->pitch, swz) ? free_rt : NULL;
}

static id<MTLDepthStencilState> depth_state(int depth, int stencil);

/* The partial-clear pipeline for a pass with (dt) or without a depth
 * attachment, made on first use; nil if it cannot be made. */
static id<MTLRenderPipelineState> clear_pso(int dt)
{
    static int failed[2], failed_hdr[2];
    MTLRenderPipelineDescriptor *pd;
    NSError *err = nil;

    if (s_hdr) {
        if (s_clear_pso_hdr[dt] || failed_hdr[dt])
            return s_clear_pso_hdr[dt];
        pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = s_vs_clear;
        pd.fragmentFunction = s_fs_clear;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
        pd.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
        if (dt) {
            pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        }
        s_clear_pso_hdr[dt] = [s_dev newRenderPipelineStateWithDescriptor:pd error:&err];
        [pd release];
        if (!s_clear_pso_hdr[dt])
            failed_hdr[dt] = 1;
        return s_clear_pso_hdr[dt];
    }
    if (s_clear_pso[dt] || failed[dt])
        return s_clear_pso[dt];
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = s_vs_clear;
    pd.fragmentFunction = s_fs_clear;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].writeMask = MTLColorWriteMaskAll;
    if (dt) {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    }
    s_clear_pso[dt] = [s_dev newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    if (!s_clear_pso[dt]) {
        failed[dt] = 1;
        XR_ERROR( LOGP "partial-clear pipeline failed (%s): a partial clear clears"
                " the whole target\n", err ? [[err localizedDescription] UTF8String] : "?");
    }
    return s_clear_pso[dt];
}

/* A colour clear covers its clip and clear rect only (nv2a_clear_box), as on
 * D3D11 and the CPU path: one target serves every clip of its surface
 * (rt_grow). A load action clears a whole attachment, so a box that is the
 * whole target keeps it, and a smaller one is a full-screen triangle of the
 * clear colour under a scissor, in a loading pass. */
static void metal_on_clear(uint32_t param)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    RenderTarget *rt;
    double c[4];
    uint32_t b[4];
    id<MTLRenderPipelineState> pso;

    if (!(param & 0xF0) || !init())
        return;
    @autoreleasepool {
        rt = surface();
        if (!rt || !nv2a_clear_box(g, rt->w, rt->h, b))
            return;
        c[0] = (double)((g->clear_color >> 16) & 0xFF) / 255.0;
        c[1] = (double)((g->clear_color >> 8) & 0xFF) / 255.0;
        c[2] = (double)(g->clear_color & 0xFF) / 255.0;
        c[3] = (double)(g->clear_color >> 24) / 255.0;
        s_clears++;
        if (b[0] == 0 && b[1] == 0 && b[2] == rt->w && b[3] == rt->h) {
            begin_encoder(rt, 1, c);
            return;
        }
        if (!s_enc || s_enc_rt != rt)
            begin_encoder(rt, 0, NULL);
        pso = clear_pso(s_enc_dt ? 1 : 0);
        if (!pso) {
            begin_encoder(rt, 1, c);
            return;
        }
        {
            /* byte/255: BGRA8Unorm stores the bytes the CPU path writes. */
            float fc[4] = {(float)c[0], (float)c[1], (float)c[2], (float)c[3]};
            MTLViewport vp = {0.0, 0.0, (double)rt->hw, (double)rt->hh, 0.0, 1.0};
            MTLScissorRect sr;
            sr.x = (NSUInteger)((uint64_t)b[0] * rt->hw / rt->w);
            sr.y = (NSUInteger)((uint64_t)b[1] * rt->hh / rt->h);
            sr.width = (NSUInteger)((uint64_t)b[2] * rt->hw / rt->w) - sr.x;
            sr.height = (NSUInteger)((uint64_t)b[3] * rt->hh / rt->h) - sr.y;
            [s_enc setRenderPipelineState:pso];
            if (s_enc_dt)
                [s_enc setDepthStencilState:depth_state(0, 0)];
            [s_enc setCullMode:MTLCullModeNone];
            [s_enc setViewport:vp];
            [s_enc setScissorRect:sr];
            [s_enc setFragmentBytes:fc length:sizeof fc atIndex:0];
            /* A clear is not a zpass sample: no counting for this draw. Metal
             * sets a visibility offset once per pass, so counting cannot
             * resume at the same slot: the pass ends, and the next draw's
             * pass takes a new slot in the query (occ_pass_begin), as a
             * query spanning passes does. */
            if (s_occ_slot >= 0)
                [s_enc setVisibilityResultMode:MTLVisibilityResultModeDisabled
                                        offset:(NSUInteger)s_occ_slot * 8u];
            [s_enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            if (s_occ_slot >= 0) {
                end_encoder();
            } else {
                /* The whole target again; each draw sets its own
                 * (draw_scissor). */
                sr.x = 0;
                sr.y = 0;
                sr.width = rt->hw;
                sr.height = rt->hh;
                [s_enc setScissorRect:sr];
            }
        }
        s_partial_clears++;
    }
}

/* CLEAR_SURFACE with the Z or stencil bit: the value scaled as the CPU path
 * scales it, by the zeta format's range; the stencil its low byte. A pass
 * that loads colour and clears the depth buffer. */
static void metal_on_zclear(uint32_t param)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    RenderTarget *rt;
    DepthTarget *dt;

    if (!(param & 3u) || !init())
        return;
    @autoreleasepool {
        rt = surface();
        if (!rt)
            return;
        dt = depth_target(rt->w, rt->h);
        if (!dt)
            return;
        rt->dt = dt;
        begin_pass(rt, 0, NULL, param & 3u, nv2a_zclear_depth(g->format, v->zclear),
                   v->zclear & 0xFFu);
        if (param & 1u)
            s_zclears++;
    }
}

/* ---- vertices ---------------------------------------------------------------- */

/* `bytes` from a ring, 16-aligned; on overflow the frame so far is flushed
 * (and waited for) and the ring starts over. NULL if it can never fit. */
static void *ring_alloc(id<MTLBuffer> b, NSUInteger size, NSUInteger *pos,
                        NSUInteger bytes, NSUInteger *off)
{
    NSUInteger at = (*pos + 15u) & ~(NSUInteger)15u;
    if (bytes > size)
        return NULL;
    if (at + bytes > size) {
        flush(1);   /* resets both rings */
        at = 0;
    }
    *pos = at + bytes;
    *off = at;
    return (uint8_t *)[b contents] + at;
}

/* ---- pipelines (blend state is part of them in Metal) ----------------------- */

static MTLBlendOperation blend_op(uint32_t eq)
{
    switch (eq) {
    case 0x800A: return MTLBlendOperationSubtract;
    case 0x800B:
    case 0xF005: return MTLBlendOperationReverseSubtract;
    case 0x8007: return MTLBlendOperationMin;
    case 0x8008: return MTLBlendOperationMax;
    default:     return MTLBlendOperationAdd;
    }
}

static MTLBlendFactor blend_factor(uint32_t gl, int alpha)
{
    static const MTLBlendFactor map[NV2A_BF_UNKNOWN + 1] = {
        MTLBlendFactorZero, MTLBlendFactorOne,
        MTLBlendFactorSourceColor, MTLBlendFactorOneMinusSourceColor,
        MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha,
        MTLBlendFactorDestinationAlpha, MTLBlendFactorOneMinusDestinationAlpha,
        MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor,
        MTLBlendFactorSourceAlphaSaturated,
        /* As the D3D11 backend: one RGBA constant, which
         * nv2a_blend_constant() fills for _COLOR or _ALPHA use. */
        MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
        MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor,
        MTLBlendFactorOne,                      /* unknown */
    };
    uint32_t bf = nv2a_blend_from_gl(gl);
    return map[alpha ? nv2a_blend_for_alpha(bf) : bf];
}

/* ---- pipeline cache ---------------------------------------------------------
 *
 * Three caches, each of at most s_pc_max entries (512; metal_pcache_max for
 * tests), least recently used out when full, so a title with more programs
 * than that still draws every batch (before: a full vertex-function cache
 * dropped the draw, a full fragment one fell back to fs_basic):
 *   s_vs   vertex functions, by the program's slots (vs_cache)
 *   s_fs   fragment functions, by the combiner state (fs_select)
 *   s_pso  pipelines, by the two functions' ids and the fixed state (pipeline)
 * An eviction only releases: an encoded draw holds its own reference, and a
 * pipeline keeps the code of its functions, so a pipeline outlives an evicted
 * function of its key. Ids, not pointers, key the pipelines, so a function
 * built again (or by the pre-warm thread) finds its pipelines.
 *
 * What is built is also appended to a key list on disk (nv2a_pcache.h:
 * the program's slots, the combiner state, the pipeline key), and a thread
 * started with the device builds that list at the next start, before the
 * title draws with it (pc_prewarm), so a warm start compiles nothing on the
 * ack thread. The file holds keys, not code: MSL is generated afresh from
 * each key, and Metal's own shader cache (com.apple.metal in the user cache
 * directory) turns a repeated source into a lookup.
 *
 * Locking: s_pc_mu guards the three tables' slots. The ack thread looks up
 * without it only through its memos (an entry it already holds) and is the
 * only thread that evicts; the pre-warm thread only fills free slots, so an
 * entry the ack thread holds never moves under it. */

static void pc_warm_begin(int first);
static pthread_mutex_t s_pc_mu = PTHREAD_MUTEX_INITIALIZER;
#define PC_MAX_DEFAULT 512
#define PC_MAX_LIMIT   16384
static int      s_pc_max = PC_MAX_DEFAULT;
static uint64_t s_tick = 1;             /* LRU clock: one per flip */
static uint64_t s_t0_ns;                /* process start, for the trace */
static int      s_pc_trace;             /* RECOMP_TRACE=pcache */
static FILE    *s_pc_csv;               /* RECOMP_TRACE=pcache=<file> */
/* Ids of the built-in functions in a pipeline key. */
#define PC_VS_PASS  0x7061737376730001ull
#define PC_FS_BASIC 0x6261736963660001ull

enum { PC_VS = 0, PC_FS = 1, PC_PSO = 2 };
static const char *const s_pc_kind[3] = { "vs", "fs", "pso" };
/* Compile counts and time, [ack thread, pre-warm][kind]; s_pc_mu. */
static struct {
    uint64_t n[2][3], ns[2][3];
    uint64_t rt_flip_n, rt_flip_ns;      /* ack thread, since the last flip */
    uint64_t rt_10s;                     /* ack thread, first 10 s of the process */
    uint64_t evicted[3];
    double   rt_max_ms;
} s_cs;

__attribute__((constructor)) static void pc_t0(void)
{
    s_t0_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static uint64_t pc_now(void)
{
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

/* One compile done: counted, and logged under RECOMP_TRACE=pcache. */
static void pc_note(int prewarm, int kind, uint64_t t_start, const char *what)
{
    uint64_t t = pc_now(), ns = t - t_start;
    double ms = (double)ns / 1e6, at = (double)(t - s_t0_ns) / 1e9;
    pthread_mutex_lock(&s_pc_mu);
    s_cs.n[prewarm][kind]++;
    s_cs.ns[prewarm][kind] += ns;
    if (!prewarm) {
        s_cs.rt_flip_n++;
        s_cs.rt_flip_ns += ns;
        if (at < 10.0)
            s_cs.rt_10s++;
        if (ms > s_cs.rt_max_ms)
            s_cs.rt_max_ms = ms;
    }
    nv2a_pcache_stat_built(prewarm, kind, ns);
    pthread_mutex_unlock(&s_pc_mu);
    if (s_pc_trace)
        fprintf(stderr, LOGP "pcache: compiled %s %s on the %s thread: %.2f ms at %.3f s\n",
                s_pc_kind[kind], what, prewarm ? "pre-warm" : "ack", ms, at);
}

/* The ack thread missed while the start-up pre-warm is still on the
 * pipelines of the key list and their functions (the first frames of a
 * start, which race it): what it needs is most likely on its way, so it
 * waits for that key instead of compiling the same thing beside it
 * (PC_WAIT: look again every half millisecond until it is there, the
 * pipelines' part is done, or the time is up).
 *
 * Bounded so that it cannot reach gameplay: only the pre-warm the backend
 * starts with has this part (a later one, after a pre-build has added to the
 * list, is never waited for), and all waits together end 3 s after it
 * started, which is 3 s after the title's first draw. Past that the ack
 * thread compiles what it needs itself, as without a list. Each wait is
 * logged. */
static int      s_pc_warm_done;         /* no pre-warm thread is running (atomic) */
static int      s_pc_warm_urgent;       /* the first one is on the pipelines' part (atomic) */
static uint64_t s_pc_wait_until;        /* the ack thread's waits end here */
static uint64_t s_pc_waits, s_pc_wait_ns;
#define PC_URGENT() __atomic_load_n(&s_pc_warm_urgent, __ATOMIC_ACQUIRE)
static uint64_t s_pc_cap_ns = 180000000000ull;  /* BO-208: the most the title is held, in all */
static uint64_t s_pc_hold_t0;                    /* when the first pre-warm began */
static int pc_wait_more(void)
{
    uint64_t now;
    if (!PC_URGENT())
        return 0;
    now = pc_now();
    if (now >= __atomic_load_n(&s_pc_wait_until, __ATOMIC_ACQUIRE)) {
        /* BO-208: past the first 3 s, still holding while the pre-warm is doing
         * real compiling (a driver, OS or shader-generator change that the
         * lists could not tell beforehand): a build with the caches warm is
         * lookups and has spent far less than this by now. */
        uint64_t busy = s_cs.ns[1][0] + s_cs.ns[1][1] + s_cs.ns[1][2];
        if (now >= s_pc_hold_t0 + s_pc_cap_ns || busy < 400000000ull)
            return 0;
        __atomic_store_n(&s_pc_wait_until, now + 1000000000ull, __ATOMIC_RELEASE);
    }
    nv2a_pcache_stat_waiting(1);        /* the window shows a line meanwhile */
    usleep(500);
    return 1;
}
static void pc_waited(uint64_t t0, const char *what)
{
    uint64_t t = pc_now();
    nv2a_pcache_stat_waiting(0);
    s_pc_waits++;
    s_pc_wait_ns += t - t0;
    fprintf(stderr, LOGP "pcache: the ack thread waited %.1f ms for the pre-warm's %s"
            " at %.3f s\n", (double)(t - t0) / 1e6, what, (double)(t - s_t0_ns) / 1e9);
}

/* A pipeline: its two functions (by id) and the blend, colour-mask and
 * depth-format state, which Metal bakes into the pipeline. */
typedef struct {
    uint64_t vs, fs;
    uint32_t key[6];
} PsoKey;
_Static_assert(sizeof(PsoKey) == 40, "PsoKey is written to disk as is");
typedef struct {
    PsoKey   k;
    uint64_t tick;                       /* last use, for LRU */
    int      used, next;                 /* bucket chain */
    id<MTLRenderPipelineState> pso;
} PsoEntry;
#define PSO_BUCKETS 2048
static PsoEntry *s_pso;                  /* s_pc_max, allocated in init */
static int       s_pso_head[PSO_BUCKETS];
/* The slots of the last 32 keys, by a hash of the key (nv2a_memo_slot);
 * -1 for none. pso_insert forgets a slot it evicts. */
#define PSO_MEMOS 32
static int       s_pso_last[PSO_MEMOS];
static int       s_pso_last_init;

static unsigned pso_bucket(const PsoKey *k)
{
    return (unsigned)(nv2a_fnv64(k, sizeof *k, NV2A_FNV64_INIT) & (PSO_BUCKETS - 1));
}

/* s_pc_mu held. */
static int pso_find(const PsoKey *k)
{
    int i;
    for (i = s_pso_head[pso_bucket(k)]; i >= 0; i = s_pso[i].next)
        if (!memcmp(&s_pso[i].k, k, sizeof *k))
            return i;
    return -1;
}

/* s_pc_mu held. Puts `pso` (its reference) in a free slot, or with `evict`
 * (the ack thread) in the least recently used one. -1 when full. */
static int pso_insert(const PsoKey *k, id<MTLRenderPipelineState> pso, int evict)
{
    int i, lru = -1, *pp;
    unsigned b;
    for (i = 0; i < s_pc_max; i++) {
        if (!s_pso[i].used)
            break;
        if (lru < 0 || s_pso[i].tick < s_pso[lru].tick)
            lru = i;
    }
    if (i == s_pc_max) {
        if (!evict)
            return -1;
        i = lru;
        for (pp = &s_pso_head[pso_bucket(&s_pso[i].k)]; *pp != i; pp = &s_pso[*pp].next)
            ;
        *pp = s_pso[i].next;
        [s_pso[i].pso release];          /* an encoded draw keeps its own reference */
        s_cs.evicted[PC_PSO]++;
        {
            int m;
            for (m = 0; m < PSO_MEMOS; m++)
                if (s_pso_last[m] == i)
                    s_pso_last[m] = -1;
        }
    }
    b = pso_bucket(k);
    s_pso[i].k = *k;
    s_pso[i].pso = pso;
    s_pso[i].tick = 0;
    s_pso[i].used = 1;
    s_pso[i].next = s_pso_head[b];
    s_pso_head[b] = i;
    return i;
}

/* Builds the pipeline for `pk` (a new reference), or nil. Any thread. */
static id<MTLRenderPipelineState> pso_build(const PsoKey *pk, id<MTLFunction> vs,
                                            id<MTLFunction> fs, int prewarm)
{
    const uint32_t *key = pk->key;
    MTLRenderPipelineDescriptor *pd;
    MTLRenderPipelineColorAttachmentDescriptor *ca;
    id<MTLRenderPipelineState> pso;
    NSError *err = nil;
    uint64_t t = pc_now();
    char what[48];

    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    ca = pd.colorAttachments[0];
    ca.pixelFormat = (key[5] & 2u) ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    ca.blendingEnabled = key[0] ? YES : NO;
    if (key[0]) {
        ca.sourceRGBBlendFactor = blend_factor(key[1], 0);
        ca.destinationRGBBlendFactor = blend_factor(key[2], 0);
        ca.sourceAlphaBlendFactor = blend_factor(key[1], 1);
        ca.destinationAlphaBlendFactor = blend_factor(key[2], 1);
        ca.rgbBlendOperation = (MTLBlendOperation)key[4];
        ca.alphaBlendOperation = (MTLBlendOperation)key[4];
    }
    ca.writeMask = ((key[3] & 0x00FF0000u) ? 0 : MTLColorWriteMaskRed)
                 | ((key[3] & 0x0000FF00u) ? 0 : MTLColorWriteMaskGreen)
                 | ((key[3] & 0x000000FFu) ? 0 : MTLColorWriteMaskBlue)
                 | ((key[3] & 0xFF000000u) ? 0 : MTLColorWriteMaskAlpha);
    if (key[5] & 1u) {
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    }
    pso = [s_dev newRenderPipelineStateWithDescriptor:pd error:&err];
    [pd release];
    if (!pso) {
        static int said;
        if (!said++)
            XR_ERROR( LOGP "pipeline failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
        return nil;
    }
    snprintf(what, sizeof what, "%016llX/%016llX", (unsigned long long)pk->vs,
             (unsigned long long)pk->fs);
    pc_note(prewarm, PC_PSO, t, what);
    return pso;
}

static void pc_put(uint32_t type, const void *data, uint32_t len);

/* The pipeline for the batch: functions vs and fs (ids vs_id, fs_id) and
 * the blend state now. */
static id<MTLRenderPipelineState> pipeline(uint64_t vs_id, id<MTLFunction> vs,
                                           uint64_t fs_id, id<MTLFunction> fs)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    id<MTLRenderPipelineState> pso;
    PsoKey pk;
    uint32_t *key = pk.key;
    unsigned ms;
    int i;

    memset(&pk, 0, sizeof pk);
    pk.vs = vs_id;
    pk.fs = fs_id;
    key[0] = g->blend_enable ? 1 : 0;
    key[1] = key[0] ? g->blend_sfactor : 0;
    key[2] = key[0] ? g->blend_dfactor : 0;
    key[3] = g->color_keep;             /* SET_COLOR_MASK, inverted */
    key[4] = key[0] ? (uint32_t)blend_op(g->blend_equation) : 0;
    key[5] = s_enc_dt ? 1 : 0;          /* the open pass has a depth buffer */
    if (s_hdr)
        key[5] |= 2u;                   /* BO-110: a half-float target */
    if (!s_pso_last_init) {
        for (i = 0; i < PSO_MEMOS; i++)
            s_pso_last[i] = -1;
        s_pso_last_init = 1;
    }
    {
        uint32_t w[sizeof pk / 4];
        memcpy(w, &pk, sizeof w);
        ms = nv2a_memo_slot(w, sizeof w / 4, PSO_MEMOS - 1);
    }
    i = s_pso_last[ms];
    if (i >= 0 && s_pso[i].used && !memcmp(&s_pso[i].k, &pk, sizeof pk)) {
        s_pso[i].tick = s_tick;
        return s_pso[i].pso;
    }
    pthread_mutex_lock(&s_pc_mu);
    i = pso_find(&pk);
    pthread_mutex_unlock(&s_pc_mu);
    if (i < 0 && !s_hdr && PC_URGENT()) {
        uint64_t t0 = pc_now();
        do {
            pthread_mutex_lock(&s_pc_mu);
            i = pso_find(&pk);
            pthread_mutex_unlock(&s_pc_mu);
        } while (i < 0 && pc_wait_more());
        pc_waited(t0, "pipeline");
    }
    if (i < 0) {
        if (!(pso = pso_build(&pk, vs, fs, 0)))
            return nil;
        pthread_mutex_lock(&s_pc_mu);
        if ((i = pso_find(&pk)) >= 0)
            [pso release];               /* the pre-warm thread was first */
        else
            i = pso_insert(&pk, pso, 1);
        pthread_mutex_unlock(&s_pc_mu);
        if (!s_hdr)     /* BO-110 prototype: HDR pipelines stay out of the disk cache */
            pc_put(PC_PSO, &pk, sizeof pk);
    }
    s_pso[i].tick = s_tick;
    s_pso_last[ms] = i;
    return s_pso[i].pso;
}

/* ---- textures ------------------------------------------------
 *
 * Decoded to A8R8G8B8 on the CPU by the shared decode (nv2a_tex_decode), the
 * same texels the CPU sampler and the D3D11 backend see, and uploaded as
 * BGRA8Unorm (the same bytes). Cache key and invalidation as the D3D11
 * backend: offset + format + size + pitch + palette + levels; rehashed once
 * per flip and re-uploaded if it changed. A changed texture gets a new
 * MTLTexture rather than a replaceRegion, so a draw already encoded this frame
 * keeps the texels it was encoded with. */

typedef struct nv2a_stage Stage;
static Stage s_stage[4];

#define TEX_MAX_TEXELS (2048u * 2048u)
#define TEX_CACHE 256
typedef struct {
    uint32_t       offset, color, width, height, pitch, palette, levels;
    uint32_t       sign;          /* nv2a_tex_signed_mask the bytes carry */
    uint32_t       gen, used;
    uint64_t       hash;
    uint32_t       rep;           /* nv2a_texrep generation it was uploaded at; 0: not a replaced texture */
    uint8_t        ui;            /* a 2D draw has bound it (nv2a_texscale.h); stays */
    uint8_t        scale;         /* the factor it was uploaded at: 0 (as the title's), 2, 4 */
    uint8_t        salgo;         /* and the scaler */
    uint8_t        up_scale;      /* the factor the texture on the GPU has (0 where the scaler refused) */
    uint8_t        use;           /* RECOMP_DEBUG=ui_scale_log: the kinds of draw seen */
    uint8_t        pend;          /* its scaled picture is being made (nv2a_texscale_get) */
    uint32_t       cut;           /* the alpha test its levels were made for (nv2a_texmip_cut_ref); 0: the title's */
    float          zbias;         /* nv2a_texrep_zbias of its bytes (BO-176); 0: none */
    uint8_t        solo;          /* BO-203: nv2a_texrep_hdr_stage0_only of its bytes */
    uint32_t       pal_len, pgen; /* P8 entries; the flip pend was last asked at */
    uint64_t       skey;          /* nv2a_texscale_key of the texels uploaded */
    int            live;
    id<MTLTexture> tex;
} TexEntry;

static TexEntry       s_tc[TEX_CACHE];

/* BO-45: the channels stage t's filter reads as signed (nv2a_tex_signed_mask),
 * honoured for 32-bit-decoded textures from guest memory, which tex_bind
 * uploads offset by 128 (nv2a_tex_signed_xor) for the combiner shader to
 * take back (d3d8_combiners_msl.c). Burnout 3's water bump maps (SZ_G8B8,
 * R G B signed) read wrong without it: xemu #587. */
static int s_signed_on;   /* the draw's pixel stage is a combiner shader */
static uint8_t stage_signed(const Stage *t)
{
    return s_signed_on ? nv2a_tex_signed_mask(t->filter) : 0;
}
static uint32_t       s_tc_clock;
static uint32_t      *s_texels;
static size_t         s_texels_cap;
static id<MTLTexture> s_black, s_magenta;
static id<MTLTexture> s_clear;   /* BO-203: transparent black, stages 1..3 of a stage0_only draw */

static id<MTLTexture> solid_texture(uint32_t argb)
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:1 height:1
                                 mipmapped:NO];
    id<MTLTexture> t = [s_dev newTextureWithDescriptor:td];
    [t replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&argb
         bytesPerRow:4];
    return t;
}

static int no_mips(void)
{
    static int off = -1;
    if (off < 0)
        off = recomp_env(RENV_METAL_NO_MIPS) != NULL;
    return off;
}

static void build_stages(void)
{
    const uint32_t *regs = nv2a_pb_tex_regs();
    const uint8_t  *sets = nv2a_pb_tex_regs_set();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    int n;

    for (n = 0; n < 4; n++) {
        Stage *t = &s_stage[n];
        nv2a_stage_decode(n, &regs[(0x40u * n) / 4], &sets[(0x40u * n) / 4],
                          v->shader_set, v->shader_prog, nv2a_pb_dma_resolve,
                          NV2A_STAGE_P8_PALETTE | NV2A_STAGE_PITCH_LINEAR
                          | (no_mips() ? 0u : NV2A_STAGE_MIPS), t);
        if (t->mode == 5)
            t->mode = 0;    /* CLIP_PLANE: no texture */
    }
}

static uint64_t tex_hash_pal(const uint8_t *mem, const Stage *t, uint32_t bytes)
{
    uint64_t h = nv2a_tex_hash(mem + t->offset, bytes, NV2A_TEX_HASH_SEED);
    if (t->color == NV2A_TEX_P8)
        h ^= nv2a_tex_hash(mem + t->palette, t->pal_len * 4u, NV2A_TEX_HASH_SEED) * 31u;
    return h;
}

static uint32_t s_tex_rep;    /* the last tex_upload's nv2a_texrep_apply */

/* graphics.ui_scale (nv2a_texscale.h): the draw being encoded is a 2D one,
 * and what tex_bind found for the stage it last bound: the factor its
 * texture was uploaded at (0: the title's own size). */
static int      s_draw_2d;
static unsigned s_bind_scale;
/* The depth offset the texture tex_bind returned asks for (nv2a_texrep_zbias,
 * BO-176); 0 for most. */
static float s_bind_zbias;
static int   s_bind_solo;    /* BO-203: the bound texture is a stage0_only HDR picture */
static uint32_t *s_scaled;
static size_t    s_scaled_cap;

static unsigned scale_levels(unsigned f)
{
    return f == 4 ? 3u : f == 2 ? 2u : 1u;
}

static uint32_t  s_swap_flip = 0xFFFFFFFFu;   /* the flip of the last swap */
static int       s_up_pend;    /* the last tex_upload queued the scaled picture */
static uint64_t  s_up_key;

/* The disk cache of scaled textures: beside the shader cache. Once. */
static void scale_store_init(void)
{
    static int done;
    char dir[900], sub[1000];
    if (done)
        return;
    done = 1;
    if (nv2a_pcache_dir(nv2a_host_opts()->shader_cache, nv2a_pcache_app(), dir, sizeof dir)) {
        snprintf(sub, sizeof sub, "%s/ui_scale", dir);
        nv2a_texscale_cache_dir(sub);
    }
}

static int scaled_room(const Stage *t, unsigned f)
{
    size_t n = nv2a_tex_texels(t->width * f, t->height * f, scale_levels(f));
    if (n > s_scaled_cap) {
        uint32_t *p = (uint32_t *)realloc(s_scaled, n * 4);
        if (!p)
            return 0;
        s_scaled = p;
        s_scaled_cap = n;
    }
    return 1;
}

/* s_scaled's level 0 (the texture f times as large) as a texture with the
 * levels down to the title's size: a 2D element drawn smaller than the
 * scaled texture is minified through them, not point-skipped. */
static id<MTLTexture> scaled_texture(const Stage *t, unsigned f)
{
    uint32_t w = t->width * f, h = t->height * f, levels = scale_levels(f), l;
    const uint32_t *src;
    MTLTextureDescriptor *td;
    id<MTLTexture> tex;

    nv2a_texscale_mips(s_scaled, w, h, levels);
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                            width:w
                                                           height:h
                                                        mipmapped:YES];
    td.mipmapLevelCount = levels;
    td.usage = MTLTextureUsageShaderRead;
    tex = [s_dev newTextureWithDescriptor:td];
    if (!tex)
        return nil;
    for (l = 0, src = s_scaled; l < levels; l++) {
        uint32_t lw = nv2a_tex_level_dim(w, l), lh = nv2a_tex_level_dim(h, l);
        [tex replaceRegion:MTLRegionMake2D(0, 0, lw, lh) mipmapLevel:l withBytes:src
               bytesPerRow:lw * 4];
        src += (size_t)lw * lh;
    }
    return tex;
}

/* The scaled picture of s_texels, if the store has it ready; else it is
 * queued (s_up_pend) and the title's own texture is uploaded meanwhile.
 * Nothing is scaled on this thread. */
static id<MTLTexture> tex_upload_scaled(const Stage *t, unsigned f, int algo)
{
    uint64_t t0 = ui_scale_log() ? xbox_HostNowNs() : 0;
    int r;

    s_up_pend = 0;
    /* Every texture a 2D draw binds, opaque pictures too (the World Tour's
     * photographs, the loading pictures): the owner wants the whole 2D front
     * end scaled. Transparency only steers the switch-on pass below. */
    if (!scaled_room(t, f))
        return nil;
    scale_store_init();
    s_up_key = nv2a_texscale_key(s_texels, t->width, t->height);
    r = nv2a_texscale_get(s_up_key, s_texels, t->width, t->height, f,
                          (enum nv2a_texscale_algo)algo, 0, s_scaled);
    if (ui_scale_log())
        fprintf(stderr, "[UISCALE] ask %016llx %ux%u x%u: %s, %.3f ms on the render thread,"
                " flip %llu\n", (unsigned long long)s_up_key, t->width, t->height, f,
                r == NV2A_TEXSCALE_READY ? "ready" : r == NV2A_TEXSCALE_PENDING ? "queued"
                                                                               : "not taken",
                (double)(xbox_HostNowNs() - t0) / 1e6, (unsigned long long)s_presents);
    if (r == NV2A_TEXSCALE_READY)
        return scaled_texture(t, f);
    s_up_pend = r == NV2A_TEXSCALE_PENDING;
    return nil;
}

static id<MTLTexture> tex_upload(const Stage *t, unsigned scale, int algo, uint32_t cut)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    size_t n = nv2a_tex_texels(t->width, t->height, t->levels);
    const uint32_t *pal = NULL, *src;
    MTLTextureDescriptor *td;
    id<MTLTexture> tex;
    uint32_t l;

    if (n > s_texels_cap) {
        uint32_t *p = (uint32_t *)realloc(s_texels, n * 4);
        if (!p)
            return nil;
        s_texels = p;
        s_texels_cap = n;
    }
    if (t->color == NV2A_TEX_P8)
        pal = (const uint32_t *)(mem + t->palette);
    nv2a_tex_decode(mem + t->offset, t->color, t->width, t->height, t->pitch,
                    t->levels, pal, pal ? t->pal_len : 0, s_texels);
    /* The title's button prompts as the pad in use prints them (input.glyphs,
     * nv2a_texrep.h): rectangles of a registered texture replaced here. */
    s_tex_rep = nv2a_texrep_apply(mem + t->offset, t->color, t->width, t->height,
                                  t->pitch, t->levels, s_texels);
    /* BO-204 debug: a ztint texture goes magenta (counting a decal's pixels). */
    (void)nv2a_texrep_ztint_apply(mem + t->offset, t->color, t->width, t->height, t->pitch,
                                  s_texels, n);
    /* An alpha-tested draw binds it: levels 1.. made again with level 0's
     * alpha coverage (nv2a_texmip.h). */
    (void)nv2a_texmip_apply(mem + t->offset, t->color, t->width, t->height, t->pitch,
                            t->levels, cut, s_texels);
    if (stage_signed(t)) {
        uint32_t x = nv2a_tex_signed_xor(stage_signed(t));
        size_t i;
        for (i = 0; i < n; i++)
            s_texels[i] ^= x;
    }
    s_bind_scale = 0;
    s_up_pend = 0;
    if (scale) {
        tex = tex_upload_scaled(t, scale, algo);
        if (tex) {
            s_bind_scale = scale;
            s_tex_uploads++;
            return tex;
        }
    }
    td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                            width:t->width
                                                           height:t->height
                                                        mipmapped:t->levels > 1];
    if (t->levels > 1)
        td.mipmapLevelCount = t->levels;
    td.usage = MTLTextureUsageShaderRead;
    tex = [s_dev newTextureWithDescriptor:td];
    if (!tex)
        return nil;
    for (l = 0, src = s_texels; l < t->levels; l++) {
        uint32_t w = nv2a_tex_level_dim(t->width, l), h = nv2a_tex_level_dim(t->height, l);
        [tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:l withBytes:src
               bytesPerRow:w * 4];
        src += (size_t)w * h;
    }
    s_tex_uploads++;
    return tex;
}

/* graphics.ui_scale switched on (or changed) in the overlay: while the title
 * is held for the pause, every loaded texture the rule could scale is decoded
 * and queued, to be made and written to the disk cache in the background
 * (nothing is uploaded here; a texture's own bind takes its picture from the
 * cache later). "Could scale": one level, swizzled or DXT, some transparency;
 * whether a 2D draw uses it is not known for a texture bound before the
 * setting was on, so some that never will be are made too. The executor's
 * thread (xbox_HostPauseIdleFn), which is the thread of every other use of
 * the cache. */
static void scale_prepare_loaded(void)
{
    static uint32_t done_gen;
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    unsigned f = nv2a_texscale_factor(), queued = 0;
    int algo = (int)nv2a_texscale_algo(), i;

    if (done_gen == nv2a_texscale_generation())
        return;
    done_gen = nv2a_texscale_generation();
    if (!f)
        return;
    scale_store_init();
    if (!nv2a_texscale_cache_path(NULL, 0))
        return;
    for (i = 0; i < TEX_CACHE; i++) {
        TexEntry *e = &s_tc[i];
        Stage st;
        const uint32_t *pal = NULL;
        uint32_t bytes;
        size_t n;

        if (!e->live || e->sign || e->up_scale == f || e->pend
                || !nv2a_texscale_tex_ok(nv2a_tex_size_from_format(e->color), e->width,
                                         e->height, e->levels))
            continue;
        /* Nothing says here whether a 2D draw uses a texture, and the larger
         * one-level textures with alpha are mostly the world's (car
         * liveries, tree lines, sky strips, 8 MB each at 4x): those are left
         * to their own bind, which knows. Fonts and HUD art are within this. */
        if ((uint64_t)e->width * e->height > 256u * 256u)
            continue;
        memset(&st, 0, sizeof st);
        st.offset = e->offset;
        st.color = e->color;
        st.width = e->width;
        st.height = e->height;
        st.pitch = e->pitch;
        st.palette = e->palette;
        st.pal_len = e->pal_len;
        st.levels = e->levels;
        bytes = nv2a_tex_extent(st.color, st.width, st.height, st.pitch, st.levels);
        /* Only what is still the texture that was uploaded. */
        if (!bytes || tex_hash_pal(mem, &st, bytes) != e->hash)
            continue;
        n = nv2a_tex_texels(st.width, st.height, st.levels);
        if (n > s_texels_cap) {
            uint32_t *p = (uint32_t *)realloc(s_texels, n * 4);
            if (!p)
                return;
            s_texels = p;
            s_texels_cap = n;
        }
        if (st.color == NV2A_TEX_P8)
            pal = (const uint32_t *)(mem + st.palette);
        nv2a_tex_decode(mem + st.offset, st.color, st.width, st.height, st.pitch, st.levels,
                        pal, pal ? st.pal_len : 0, s_texels);
        (void)nv2a_texrep_apply(mem + st.offset, st.color, st.width, st.height, st.pitch,
                                st.levels, s_texels);
        /* Not known to be 2D here: art cut out with alpha is the likely kind;
         * an opaque picture waits for its own 2D bind. */
        if (!nv2a_texscale_has_alpha(s_texels, (size_t)st.width * st.height))
            continue;
        if (nv2a_texscale_get(nv2a_texscale_key(s_texels, st.width, st.height), s_texels,
                              st.width, st.height, f, (enum nv2a_texscale_algo)algo,
                              NV2A_TEXSCALE_DISK_ONLY, NULL) == NV2A_TEXSCALE_PENDING)
            queued++;
    }
    fprintf(stderr, "[UISCALE] setting changed to %ux: %u loaded textures queued for the"
            " cache\n", f, queued);
}

/* RECOMP_DEBUG=ui_scale_log: one line whenever a texture is seen in a kind
 * of draw it was not seen in before (bit 0: a 2D draw, bit 1: another), with
 * the hash texrep_dump names its file by. For checking the rule. */
static int ui_scale_log(void)
{
    static int on = -1;
    if (on < 0)
        on = recomp_env(RENV_UI_SCALE_LOG) != NULL;
    return on;
}

static void ui_scale_use(TexEntry *e, const Stage *t, const uint8_t *mem)
{
    uint8_t bit = s_draw_2d ? 1 : 2;
    if (e->use & bit)
        return;
    e->use |= bit;
    fprintf(stderr, "[UISCALE] tex %016llx fmt 0x%02x %ux%u levels %u use %u scale %u flip %llu\n",
            (unsigned long long)nv2a_texrep_hash(mem + t->offset, t->color, t->width,
                                                 t->height, t->pitch),
            t->color, t->width, t->height, t->levels, e->use, e->up_scale,
            (unsigned long long)s_presents);
}

static id<MTLTexture> tex_bind(const Stage *t)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t bytes = 0, i;
    TexEntry *e = NULL, *victim = &s_tc[0];
    unsigned want;
    uint32_t cut;
    int algo;

    s_bind_scale = 0;
    s_bind_zbias = 0.0f;
    s_bind_solo = 0;
    if (t->valid && (uint64_t)t->width * t->height <= TEX_MAX_TEXELS)
        bytes = nv2a_tex_extent(t->color, t->width, t->height, t->pitch, t->levels);
    /* The guest bytes are about to be hashed and decoded: a render target
     * over them that rt_texture did not bind (another format, no_rtt, an
     * evicted name) is written back first, so the decode sees what was
     * drawn there. */
    if (bytes) {
        uint32_t lo = t->offset & 0x07FFFFFFu, hi = lo + bytes;
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            uint32_t rlo = rt->addr & 0x07FFFFFFu;
            if (rt->tex && rt->dirty && rt->pitch
                    && lo < rlo + rt->pitch * rt->h && rlo < hi) {
                if (rt_drop_if_stale(rt, "sync"))
                    continue;
                s_decode_syncs++;
                if (recomp_env(RENV_METAL_TRACE))
                    fprintf(stderr, LOGP "decode sync: target 0x%08X %ux%u (%u bytes) for"
                            " texture 0x%08X fmt 0x%02X %ux%u (%u bytes), flip %llu\n",
                            rt->offset, rt->w, rt->h, rt->pitch * rt->h, t->offset,
                            t->color, t->width, t->height, bytes,
                            (unsigned long long)s_presents);
                rt_sync(rt, rt->pitch);
            }
        }
    }
    if (!bytes) {
        static uint8_t said[256];
        if ((uint64_t)t->width * t->height > TEX_MAX_TEXELS)
            return s_black;
        if (!said[t->color & 0xFF]) {
            said[t->color & 0xFF] = 1;
            fprintf(stderr, LOGP "\x1b[1;35mtexture format 0x%02X (%ux%u at"
                    " 0x%08X) not decoded: drawn magenta\x1b[0m\n",
                    t->color, t->width, t->height, t->offset);
        }
        return s_magenta;
    }
    for (i = 0; i < TEX_CACHE; i++) {
        TexEntry *c = &s_tc[i];
        if (c->live && c->offset == t->offset && c->color == t->color
                && c->width == t->width && c->height == t->height
                && c->pitch == t->pitch && c->palette == t->palette
                && c->levels == t->levels && c->sign == stage_signed(t)) {
            e = c;
            break;
        }
        if (!c->live || (victim->live && c->used < victim->used))
            victim = c;
    }
    /* graphics.ui_scale: a texture a 2D draw binds is scaled, from then on
     * and for every draw (off: want is 0 and nothing here differs). */
    want = nv2a_texscale_factor();
    algo = (int)nv2a_texscale_algo();
    if (want && !((s_draw_2d || (e && e->ui)) && !stage_signed(t)
                  && nv2a_texscale_tex_ok(nv2a_tex_size_from_format(t->color), t->width,
                                          t->height, t->levels)))
        want = 0;
    /* BO-171: a mipmapped texture an alpha-tested draw binds gets levels
     * made for that test, once; it keeps them for every later draw, and for
     * the first test it was made for. */
    cut = 0;
    if (nv2a_texmip_mode() & NV2A_TEXMIP_LEVELS) {
        const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
        if (t->levels > 1 && nv2a_tex_filter_mip(t->filter) && !stage_signed(t))
            cut = nv2a_texmip_cut_ref(v->atest_enable, v->atest_func, v->atest_ref);
        if (e && e->cut)
            cut = e->cut;
    }
    if (e) {
        /* A replaced texture whose variant changed is uploaded again; so is
         * one whose scale is not the one wanted now. */
        int stale = (e->rep && e->rep != nv2a_texrep_generation())
                 || e->scale != want || (want && e->salgo != algo) || e->cut != cut;
        if (s_draw_2d)
            e->ui = 1;
        if (e->gen != g->flips || stale) {
            uint64_t tp = prof_t(), h = tex_hash_pal(mem, t, bytes);
            prof_add(&s_pa.texh, tp);
            s_pa.texh_bytes += bytes;
            s_pa.texh_n++;
            s_tex_rehash++;
            e->gen = g->flips;
            if (h != e->hash || stale) {
                id<MTLTexture> nt;
                tp = prof_t();
                nt = tex_upload(t, want, algo, cut);
                prof_add(&s_pa.texup, tp);
                s_pa.texup_n++;
                e->hash = h;
                e->rep = s_tex_rep;
                e->cut = cut;
                e->zbias = nv2a_texrep_zbias(mem + t->offset, t->color, t->width,
                                             t->height, t->pitch);
                e->solo = (uint8_t)nv2a_texrep_hdr_stage0_only(mem + t->offset, t->color,
                                                               t->width, t->height, t->pitch);
                s_tex_changed++;
                if (!nt)
                    return s_black;
                [e->tex release];
                e->tex = nt;
                /* The factor wanted, also when the scaler refused: it is
                 * not asked again until the bytes or the setting change. */
                e->scale = (uint8_t)want;
                e->salgo = (uint8_t)algo;
                e->up_scale = (uint8_t)s_bind_scale;
                e->pend = (uint8_t)s_up_pend;
                e->skey = s_up_key;
                e->pgen = g->flips;
                e->pal_len = t->pal_len;
            }
        }
    } else {
        id<MTLTexture> nt;
        e = victim;
        [e->tex release];
        e->tex = nil;
        e->live = 0;
        {
            uint64_t tp = prof_t();
            nt = tex_upload(t, want, algo, cut);
            prof_add(&s_pa.texup, tp);
            s_pa.texup_n++;
        }
        if (!nt)
            return s_black;
        e->ui = (uint8_t)(s_draw_2d != 0);
        e->scale = (uint8_t)want;
        e->salgo = (uint8_t)algo;
        e->up_scale = (uint8_t)s_bind_scale;
        e->pend = (uint8_t)s_up_pend;
        e->skey = s_up_key;
        e->pgen = g->flips;
        e->pal_len = t->pal_len;
        e->use = 0;
        e->cut = cut;
        e->offset = t->offset;
        e->color = t->color;
        e->width = t->width;
        e->height = t->height;
        e->pitch = t->pitch;
        e->palette = t->palette;
        e->levels = t->levels;
        e->sign = stage_signed(t);
        e->gen = g->flips;
        e->hash = tex_hash_pal(mem, t, bytes);
        e->rep = s_tex_rep;
        e->zbias = nv2a_texrep_zbias(mem + t->offset, t->color, t->width, t->height,
                                     t->pitch);
        e->solo = (uint8_t)nv2a_texrep_hdr_stage0_only(mem + t->offset, t->color, t->width,
                                                       t->height, t->pitch);
        e->tex = nt;
        e->live = 1;
    }
    /* Its scaled picture was being made: asked for once a flip, swapped in
     * when it is there, one texture a flip. A new MTLTexture, so a draw encoded with the unscaled
     * one keeps it. */
    if (e->pend && e->pgen != g->flips && s_swap_flip != g->flips) {
        e->pgen = g->flips;
        if (!scaled_room(t, e->scale)) {
            e->pend = 0;
        } else {
            uint64_t t0 = ui_scale_log() ? xbox_HostNowNs() : 0;
            int r = nv2a_texscale_get(e->skey, NULL, t->width, t->height, e->scale,
                                      (enum nv2a_texscale_algo)e->salgo, 0, s_scaled);
            if (r == NV2A_TEXSCALE_READY) {
                id<MTLTexture> nt = scaled_texture(t, e->scale);
                s_swap_flip = g->flips;   /* one a flip: its upload is the cost */
                e->pend = 0;
                if (nt) {
                    [e->tex release];
                    e->tex = nt;
                    e->up_scale = e->scale;
                    s_tex_uploads++;
                }
            } else if (r != NV2A_TEXSCALE_PENDING) {
                e->pend = 0;         /* refused or lost: stays as the title's */
            }
            if (ui_scale_log() && r != NV2A_TEXSCALE_PENDING)
                fprintf(stderr, "[UISCALE] swap %016llx %ux%u x%u: %s, %.3f ms on the render"
                        " thread, flip %llu\n", (unsigned long long)e->skey, t->width,
                        t->height, (unsigned)e->scale,
                        r == NV2A_TEXSCALE_READY ? "scaled texture in" : "no result",
                        (double)(xbox_HostNowNs() - t0) / 1e6, (unsigned long long)s_presents);
        }
    }
    e->used = ++s_tc_clock;
    s_bind_scale = e->up_scale;
    s_bind_zbias = e->zbias;
    s_bind_solo = e->solo;
    if (ui_scale_log())
        ui_scale_use(e, t, mem);
    return e->tex;
}

/* NV2A wrap modes: 1 wrap, 2 mirror, 3 clamp to edge, 4 border, 5 clamp. */
static MTLSamplerAddressMode address_mode(uint32_t m)
{
    switch (m) {
    case 1:  return MTLSamplerAddressModeRepeat;
    case 2:  return MTLSamplerAddressModeMirrorRepeat;
    default: return MTLSamplerAddressModeClampToEdge;
    }
}

/* SET_TEXTURE_FILTER as the D3D11 backend reads it (filter_linear,
 * filter_mip there). RECOMP_METAL_POINT=1 forces nearest texels. */
#define filter_linear nv2a_tex_filter_linear   /* nv2a_backend_common.h */
#define filter_mip    nv2a_tex_filter_mip
#define lod_bias      nv2a_tex_lod_bias

#define SMP_MAX 128
static struct { uint64_t key; id<MTLSamplerState> s; } s_smp[SMP_MAX];
static int s_smp_n;

static id<MTLSamplerState> sampler(const Stage *t, unsigned scale)
{
    static int point = -1;
    MTLSamplerDescriptor *sd;
    id<MTLSamplerState> st;
    uint32_t u = t->addr_u & 0xF, v = t->addr_v & 0xF;
    int mn, mg, mip, i, aniso;
    uint64_t key;

    if (point < 0)
        point = recomp_env(RENV_METAL_POINT) != NULL;
    mn = !point && filter_linear(t->filter, 0);
    mg = !point && filter_linear(t->filter, 1);
    mip = t->levels > 1 ? filter_mip(t->filter) : 0;
    /* A scaled 2D texture (one level in the title) has the levels down to
     * the title's size: blended between, so its look does not step with the
     * size it is drawn at. */
    if (scale)
        mip = 2;
    /* Anisotropy only on a mipmapped sampler with a linear minification
     * filter, as asked (RECOMP_GFX_ANISO). */
    aniso = (mip && mn && !scale) ? nv2a_host_aniso() : 1;
    key = u | v << 4 | (uint64_t)mn << 8 | (uint64_t)mg << 9 | (uint64_t)mip << 10;
    if (mip)
        key |= (uint64_t)t->lod_min << 25 | (uint64_t)t->lod_max << 29;
    key |= (uint64_t)aniso << 33;       /* the level is part of the sampler */
    key |= (uint64_t)scale << 40;
    for (i = 0; i < s_smp_n; i++)
        if (s_smp[i].key == key)
            return s_smp[i].s;
    sd = [[MTLSamplerDescriptor alloc] init];
    sd.maxAnisotropy = (NSUInteger)aniso;
    sd.minFilter = mn ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.magFilter = mg ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.mipFilter = mip == 2 ? MTLSamplerMipFilterLinear
                 : mip == 1 ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = address_mode(u);
    sd.tAddressMode = address_mode(v);
    sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.lodMinClamp = scale ? 0.0f : mip ? (float)t->lod_min : 0.0f;
    sd.lodMaxClamp = scale ? (float)(scale_levels(scale) - 1u) : mip ? (float)t->lod_max : 0.0f;
    st = [s_dev newSamplerStateWithDescriptor:sd];
    [sd release];
    if (!st)
        return s_smp_n ? s_smp[0].s : nil;
    if (s_smp_n < SMP_MAX) {
        i = s_smp_n++;
    } else {
        static unsigned next;
        i = (int)(next++ % SMP_MAX);
        [s_smp[i].s release];
    }
    s_smp[i].key = key;
    s_smp[i].s = st;
    return st;
}

/* ---- shaders (M2): vertex programs and register combiners ------------------
 *
 * As the D3D11 backend's vertex_shader and combiner_ps: the program is
 * parsed from the slots it starts at (d3d8_vsh_parse) and emitted as MSL
 * (d3d8_vsh_generate_msl, screen-space epilogue), keyed by its slots; the
 * combiner registers become NV2ACombinerState (d3d8_combiners_from_regs) and
 * MSL (d3d8_combiners_generate_msl), keyed by the state's bytes. Each
 * compiles once, with s_opts (fast math off), and stays until evicted (the
 * pipeline cache above). A source that fails to compile is logged with its
 * text, counted (s_compile_errors, in the present log) and remembered as bad. */

static uint64_t s_compile_errors, s_vs_compiled, s_fs_compiled;

/* MSL for one function, compiled (a new reference), or nil. Any thread. */
static id<MTLFunction> compile_fn(const char *src, NSString *name, const char *what)
{
    NSError *err = nil;
    id<MTLLibrary> lib;
    id<MTLFunction> fn;

    @autoreleasepool {
        lib = [s_dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                  options:s_opts error:&err];
        if (!lib) {
            uint64_t n = __atomic_add_fetch(&s_compile_errors, 1, __ATOMIC_RELAXED);
            if (n <= 4)
                XR_ERROR( LOGP "\x1b[1;31m%s: MSL compile failed: %s\x1b[0m\n%s\n",
                        what, err ? [[err localizedDescription] UTF8String] : "?", src);
            return nil;
        }
        fn = [lib newFunctionWithName:name];
        [lib release];
        if (!fn) {
            __atomic_add_fetch(&s_compile_errors, 1, __ATOMIC_RELAXED);
            fprintf(stderr, LOGP "%s: no function %s\n", what, [name UTF8String]);
        }
    }
    return fn;
}

typedef struct {
    uint64_t        id;       /* nv2a_fnv64 of the program's slots; 0: free */
    uint64_t        tick;     /* last use, for LRU */
    uint32_t        hash;     /* nv2a_vsh_program_hash, as logged */
    uint16_t        inputs;
    uint8_t         bad;
    id<MTLFunction> fn;
} VsEntry;
static VsEntry *s_vs;         /* s_pc_max, allocated in init */
static struct { int valid; uint32_t dwords, start; VsEntry *e; } s_vs_memo;

/* The vertex function for `n` slots of program (up to its FINAL), a new
 * reference or nil; its inputs in *inputs. Any thread. */
static id<MTLFunction> vs_build(const uint32_t *slots, int n, uint32_t hash,
                                uint16_t *inputs, int prewarm)
{
    NV2AVshProgram *parsed = (NV2AVshProgram *)malloc(sizeof *parsed);
    char *msl = (char *)malloc(65536);
    id<MTLFunction> fn = nil;
    uint64_t t = pc_now();
    char what[32];

    snprintf(what, sizeof what, "program %08X", hash);
    if (parsed && msl) {
        d3d8_vsh_parse(slots, n, parsed);
        if (d3d8_vsh_generate_msl(parsed, D3D8_MSL_SCREEN_SPACE, msl, 65536) <= 0) {
            __atomic_add_fetch(&s_compile_errors, 1, __ATOMIC_RELAXED);
            fprintf(stderr, LOGP "%s: no MSL\n", what);
        } else if ((fn = compile_fn(msl, @"vs_main", what)) != nil) {
            *inputs = parsed->inputs_read;
            __atomic_add_fetch(&s_vs_compiled, 1, __ATOMIC_RELAXED);
            pc_note(prewarm, PC_VS, t, what);
        }
    }
    free(parsed);
    free(msl);
    return fn;
}

/* s_pc_mu held. */
static VsEntry *vs_find(uint64_t kid)
{
    int i;
    for (i = 0; i < s_pc_max; i++)
        if (s_vs[i].id == kid)
            return &s_vs[i];
    return NULL;
}

/* s_pc_mu held. A free entry, or with `evict` (the ack thread) the least
 * recently used one, emptied; NULL when full. */
static VsEntry *vs_slot(int evict)
{
    VsEntry *lru = NULL;
    int i;
    for (i = 0; i < s_pc_max; i++) {
        if (!s_vs[i].id)
            return &s_vs[i];
        if (!lru || s_vs[i].tick < lru->tick)
            lru = &s_vs[i];
    }
    if (!evict)
        return NULL;
    [lru->fn release];
    if (s_vs_memo.e == lru)
        s_vs_memo.valid = 0;
    memset(lru, 0, sizeof *lru);
    s_cs.evicted[PC_VS]++;
    return lru;
}

static VsEntry *vs_cache(void)
{
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    const uint32_t *slots;
    id<MTLFunction> fn;
    uint32_t hash, len;
    uint16_t inputs = 0;
    uint64_t kid;
    VsEntry *e;

    /* Programs only change by upload (prog_dwords) or a new start slot. */
    if (s_vs_memo.valid && s_vs_memo.dwords == v->prog_dwords && s_vs_memo.start == v->start) {
        if (s_vs_memo.e)
            s_vs_memo.e->tick = s_tick;
        return s_vs_memo.e;
    }
    hash = nv2a_vsh_program_hash(v, &len);
    slots = &v->prog[v->start][0];
    kid = nv2a_fnv64(slots, (size_t)len * 16u, NV2A_FNV64_INIT);
    if (!kid)
        kid = 1;
    pthread_mutex_lock(&s_pc_mu);
    e = vs_find(kid);
    pthread_mutex_unlock(&s_pc_mu);
    if (!e && PC_URGENT()) {
        uint64_t t0 = pc_now();
        do {
            pthread_mutex_lock(&s_pc_mu);
            e = vs_find(kid);
            pthread_mutex_unlock(&s_pc_mu);
        } while (!e && pc_wait_more());
        pc_waited(t0, "vertex function");
    }
    if (!e) {
        fn = vs_build(slots, (int)len, hash, &inputs, 0);
        pthread_mutex_lock(&s_pc_mu);
        if ((e = vs_find(kid)) != NULL) {
            [fn release];               /* the pre-warm thread was first */
            fn = nil;
        } else {
            e = vs_slot(1);
            e->id = kid;
            e->hash = hash;
            e->inputs = inputs;
            e->bad = fn ? 0 : 1;
            e->fn = fn;
        }
        pthread_mutex_unlock(&s_pc_mu);
        if (fn) {
            pc_put(PC_VS, slots, len * 16u);
            pc_warm_begin(0);           /* BO-208: the manifest's pipelines for a program only the title assembles */
        }
    }
    e->tick = s_tick;
    s_vs_memo.valid = 1;
    s_vs_memo.dwords = v->prog_dwords;
    s_vs_memo.start = v->start;
    s_vs_memo.e = e->bad ? NULL : e;
    return s_vs_memo.e;
}

typedef struct {
    uint64_t          id;     /* nv2a_fnv64 of st; 0: free */
    uint64_t          tick;   /* last use, for LRU */
    uint8_t           bad;
    NV2ACombinerState st;
    id<MTLFunction>   fn;
} FsEntry;
static FsEntry *s_fs;         /* s_pc_max, allocated in init */
/* The last 64 choices of fs_select, by a hash of the key (nv2a_memo_slot);
 * fs_slot forgets those whose entry it evicts. */
#define FS_MEMOS 64
static struct { int valid; uint32_t r[39]; FsEntry *e; } s_fs_memo[FS_MEMOS];

/* The fragment function for combiner state `st`, a new reference or nil.
 * Any thread. */
static id<MTLFunction> fs_build(const NV2ACombinerState *st, uint64_t kid, int prewarm,
                                int hdr)
{
    char *msl = (char *)malloc(65536);
    id<MTLFunction> fn = nil;
    uint64_t t = pc_now();
    char what[40];

    snprintf(what, sizeof what, "combiners %08X", (uint32_t)kid);
    if (!msl)
        return nil;
    if ((hdr ? d3d8_combiners_generate_msl_hdr(st, msl, 65536)
             : d3d8_combiners_generate_msl(st, msl, 65536)) <= 0) {
        __atomic_add_fetch(&s_compile_errors, 1, __ATOMIC_RELAXED);
        fprintf(stderr, LOGP "%s: no MSL\n", what);
    } else if ((fn = compile_fn(msl, @"fs_main", what)) != nil) {
        __atomic_add_fetch(&s_fs_compiled, 1, __ATOMIC_RELAXED);
        pc_note(prewarm, PC_FS, t, what);
    }
    free(msl);
    return fn;
}

/* s_pc_mu held. */
static FsEntry *fs_find(uint64_t kid, const NV2ACombinerState *st)
{
    int i;
    for (i = 0; i < s_pc_max; i++)
        if (s_fs[i].id == kid && !memcmp(&s_fs[i].st, st, sizeof *st))
            return &s_fs[i];
    return NULL;
}

/* s_pc_mu held. As vs_slot. */
static FsEntry *fs_slot(int evict)
{
    FsEntry *lru = NULL;
    int i;
    for (i = 0; i < s_pc_max; i++) {
        if (!s_fs[i].id)
            return &s_fs[i];
        if (!lru || s_fs[i].tick < lru->tick)
            lru = &s_fs[i];
    }
    if (!evict)
        return NULL;
    [lru->fn release];
    for (i = 0; i < FS_MEMOS; i++)
        if (s_fs_memo[i].e == lru)
            s_fs_memo[i].valid = 0;
    memset(lru, 0, sizeof *lru);
    s_cs.evicted[PC_FS]++;
    return lru;
}

/* The fragment function for the batch, its id in *id: its register
 * combiners when they are programmed (v->rc_set and a stage count), else
 * fs_basic. As the D3D11 setup_pixel, only 2D textures are bound, so every
 * sampling mode is passed to the combiners as 2D. */
static id<MTLFunction> fs_select(const struct nv2a_pb_vsh *v, uint64_t *fid)
{
    NV2ACombinerState st;
    uint32_t key[39], modes = 0, signs = 0;
    id<MTLFunction> fn;
    uint64_t kid;
    FsEntry *e;
    unsigned ms;
    int n;

    *fid = PC_FS_BASIC;
    if (!(v->rc_set && (v->rc_ctl & 0xFF)))
        return s_fs_basic;
    for (n = 0; n < 4; n++) {
        uint32_t m = (uint32_t)s_stage[n].mode;
        if (s_stage[n].raw_mode == 5)
            m = 5;          /* CLIPPLANE reads 0, not NONE's (0, 0, 0, 1) */
        else if (m == 3 && s_stage[n].cube)
            m = 3;          /* CUBEMAP: a texturecube (cube_bind) */
        else if ((m == 6 || m == 7) && n > 0)
            ;               /* BUMPENVMAP(_LUM): offset by the previous stage */
        else if (m && m != 4) m = 1;
        modes |= m << (5 * n);
        if (m == 1 || m == 6 || m == 7)   /* 2D sampling; a cube's faces may be targets */
            signs |= (uint32_t)nv2a_tex_signed_mask(s_stage[n].filter) << (4 * n);
    }
    memcpy(key, v->rc_cicw, 32);
    memcpy(key + 8, v->rc_aicw, 32);
    memcpy(key + 16, v->rc_cocw, 32);
    memcpy(key + 24, v->rc_aocw, 32);
    key[32] = v->rc_ctl;
    key[33] = v->rc_fcw0;
    key[34] = v->rc_fcw1;
    key[35] = modes;
    key[36] = signs;
    key[37] = v->shader_ctl & 0x00FF0000u;  /* stage 2 and 3 input stages */
    key[38] = v->clip_plane_mode;
    ms = nv2a_memo_slot(key, 39, FS_MEMOS - 1);
    if (s_fs_memo[ms].valid && !memcmp(s_fs_memo[ms].r, key, sizeof key)) {
        if (!s_fs_memo[ms].e)
            return s_fs_basic;
        s_fs_memo[ms].e->tick = s_tick;
        *fid = s_fs_memo[ms].e->id;
        return s_fs_memo[ms].e->fn;
    }

    memset(&st, 0, sizeof st);
    d3d8_combiners_from_regs(v->rc_cicw, v->rc_aicw, v->rc_cocw, v->rc_aocw,
                             v->rc_ctl, v->rc_fcw0, v->rc_fcw1, modes, &st);
    st.fog_input = 1;   /* FOG.a is the interpolated vertex fog factor */
    for (n = 0; n < 4; n++) {
        st.tex_signed[n] = (uint8_t)((signs >> (4 * n)) & 0xFu);
        st.clip_mode[n] = (uint8_t)((v->clip_plane_mode >> (4 * n)) & 0xFu);
    }
    st.input_tex[2] = (uint8_t)((v->shader_ctl >> 16) & 0xFu);
    st.input_tex[3] = (uint8_t)((v->shader_ctl >> 20) & 0xFu);
    kid = nv2a_fnv64(&st, sizeof st, NV2A_FNV64_INIT);
    if (s_hdr)
        kid ^= 0x4844520000000000ull;   /* BO-110: the HDR variant, its own id */
    if (!kid)
        kid = 1;
    pthread_mutex_lock(&s_pc_mu);
    e = fs_find(kid, &st);
    pthread_mutex_unlock(&s_pc_mu);
    if (!e && !s_hdr && PC_URGENT()) {
        uint64_t t0 = pc_now();
        do {
            pthread_mutex_lock(&s_pc_mu);
            e = fs_find(kid, &st);
            pthread_mutex_unlock(&s_pc_mu);
        } while (!e && pc_wait_more());
        pc_waited(t0, "fragment function");
    }
    if (!e) {
        fn = fs_build(&st, kid, 0, s_hdr);
        pthread_mutex_lock(&s_pc_mu);
        if ((e = fs_find(kid, &st)) != NULL) {
            [fn release];               /* the pre-warm thread was first */
            fn = nil;
        } else {
            e = fs_slot(1);
            e->id = kid;
            e->st = st;
            e->bad = fn ? 0 : 1;
            e->fn = fn;
        }
        pthread_mutex_unlock(&s_pc_mu);
        if (fn && !s_hdr)
            pc_put(PC_FS, &st, sizeof st);
    }
    e->tick = s_tick;
    memcpy(s_fs_memo[ms].r, key, sizeof key);
    s_fs_memo[ms].valid = 1;
    s_fs_memo[ms].e = e->bad ? NULL : e;   /* as D3D11: the basic PS */
    if (!s_fs_memo[ms].e)
        return s_fs_basic;
    *fid = e->id;
    return e->fn;
}

/* ---- pipeline cache on disk, and the pre-warm thread ------------------------ */

static nv2a_pcache *s_pcf;              /* the key list, or NULL (off) */
static pthread_mutex_t s_pcf_mu = PTHREAD_MUTEX_INITIALIZER;   /* s_pcf itself */
static uint64_t     s_pc_loaded[3], s_pc_warm[3], s_pc_warm_ns;
static int          s_pc_rescan;        /* asked for while one ran (atomic) */
static char         s_pc_path[1100], s_pc_ver[256];

/* Appends a key the ack thread just built (it is new to this process; the
 * file drops one it already has). */
static nv2a_pcache *s_gapf;             /* BO-208: gaps.bin: what the render thread had to build */
static void pc_put_(uint32_t type, const void *data, uint32_t len, int gap)
{
    pthread_mutex_lock(&s_pcf_mu);
    int added = s_pcf ? nv2a_pcache_put(s_pcf, type, data, len) : 0;
    if (s_pcf && added < 0) {
        static int said;
        if (!said++)
            XR_ERROR( LOGP "pcache: cannot append; the key list is not written\n");
    }
    if (gap && added > 0 && s_gapf) {
        /* A key the lists lacked when the title needed it: the field's gap in
         * the manifest. A vertex program as its hash only (the microcode is
         * the title's); pcache_keys.py manifest takes the file as it is. */
        if (type == PC_VS) {
            uint64_t h = nv2a_fnv64(data, len, NV2A_FNV64_INIT);
            nv2a_pcache_put(s_gapf, NV2A_PC_MANIFEST_VS, &h, 8);
        } else {
            nv2a_pcache_put(s_gapf, type, data, len);
        }
    }
    pthread_mutex_unlock(&s_pcf_mu);
}

/* The render thread's builds: a gap if the lists had it not. */
static void pc_put(uint32_t type, const void *data, uint32_t len)
{
    pc_put_(type, data, len, 1);
}

/* The seeds of a pre-build: not gaps. */
static void pc_seed(uint32_t type, const void *data, uint32_t len)
{
    pc_put_(type, data, len, 0);
}

/* The function ids the list's pipelines name: built first. */
static uint64_t *s_pc_need;
static size_t    s_pc_need_n;

static int pc_needed(uint64_t kid)
{
    size_t k;
    for (k = 0; k < s_pc_need_n; k++)
        if (s_pc_need[k] == kid)
            return 1;
    return 0;
}

/* One record of the file built and cached, as the ack thread would have:
 * pass 0 builds the functions the list's pipelines use, pass 1 the pipelines,
 * pass 2 the other functions (keys seeded from the title's executable that
 * nothing drew yet: nobody waits for those). Never evicts: a full table
 * leaves the rest to the ack thread. */
static void pc_warm_one_(size_t i, int pass)
{
    const void *data;
    uint32_t type, len;
    uint64_t kid;

    if (!nv2a_pcache_get(s_pcf, i, &type, &data, &len))
        return;
    if (pass != 1 && type == PC_VS && len >= 16 && len % 16 == 0
            && len <= NV2A_VSH_SLOTS * 16u) {
        uint32_t hash = 2166136261u, k;
        uint16_t inputs = 0;
        id<MTLFunction> fn;
        VsEntry *e;
        for (k = 0; k < len / 4u; k++)  /* nv2a_vsh_program_hash, for the log */
            hash = (hash ^ ((const uint32_t *)data)[k]) * 16777619u;
        if (!(kid = nv2a_fnv64(data, len, NV2A_FNV64_INIT)))
            kid = 1;
        if (pass == 0 && !pc_needed(kid))
            return;
        pthread_mutex_lock(&s_pc_mu);
        e = vs_find(kid);
        pthread_mutex_unlock(&s_pc_mu);
        if (e || !(fn = vs_build((const uint32_t *)data, (int)(len / 16u), hash, &inputs, 1)))
            return;
        pthread_mutex_lock(&s_pc_mu);
        if (!vs_find(kid) && (e = vs_slot(0)) != NULL) {
            e->id = kid;
            e->hash = hash;
            e->inputs = inputs;
            e->fn = fn;
            fn = nil;
            s_pc_warm[PC_VS]++;
        }
        pthread_mutex_unlock(&s_pc_mu);
        [fn release];
    } else if (pass != 1 && type == PC_FS && len == sizeof(NV2ACombinerState)) {
        NV2ACombinerState st;
        id<MTLFunction> fn;
        FsEntry *e;
        memcpy(&st, data, sizeof st);
        if (!(kid = nv2a_fnv64(&st, sizeof st, NV2A_FNV64_INIT)))
            kid = 1;
        if (pass == 0 && !pc_needed(kid))
            return;
        pthread_mutex_lock(&s_pc_mu);
        e = fs_find(kid, &st);
        pthread_mutex_unlock(&s_pc_mu);
        if (e || !(fn = fs_build(&st, kid, 1, 0)))
            return;
        pthread_mutex_lock(&s_pc_mu);
        if (!fs_find(kid, &st) && (e = fs_slot(0)) != NULL) {
            e->id = kid;
            e->st = st;
            e->fn = fn;
            fn = nil;
            s_pc_warm[PC_FS]++;
        }
        pthread_mutex_unlock(&s_pc_mu);
        [fn release];
    } else if (pass == 1 && type == PC_PSO && len == sizeof(PsoKey)) {
        id<MTLFunction> vs = nil, fs = nil;
        id<MTLRenderPipelineState> pso;
        PsoKey pk;
        int found;
        memcpy(&pk, data, sizeof pk);
        pthread_mutex_lock(&s_pc_mu);
        found = pso_find(&pk) >= 0;
        if (!found) {
            /* The functions by id; each retained, as the ack thread may
             * evict its entry while the pipeline builds. */
            VsEntry *ve = pk.vs == PC_VS_PASS ? NULL : vs_find(pk.vs);
            int i;
            vs = pk.vs == PC_VS_PASS ? s_vs_pass : (ve && !ve->bad ? ve->fn : nil);
            if (pk.fs == PC_FS_BASIC)
                fs = s_fs_basic;
            else
                for (i = 0; i < s_pc_max; i++)
                    if (s_fs[i].id == pk.fs && !s_fs[i].bad) {
                        fs = s_fs[i].fn;
                        break;
                    }
            [vs retain];
            [fs retain];
        }
        pthread_mutex_unlock(&s_pc_mu);
        if (!found && vs && fs && (pso = pso_build(&pk, vs, fs, 1)) != nil) {
            pthread_mutex_lock(&s_pc_mu);
            if (pso_find(&pk) < 0 && pso_insert(&pk, pso, 0) >= 0) {
                pso = nil;
                s_pc_warm[PC_PSO]++;
            }
            pthread_mutex_unlock(&s_pc_mu);
            [pso release];
        }
        [vs release];
        [fs release];
    }
}


/* BO-208: the progress of the first build: records handled so far. */
static uint32_t s_pc_done, s_pc_total;
static int      s_pc_cold;
static uint64_t s_pc_hold_ns = 3000000000ull;
static void pc_warm_one(size_t i, int pass)
{
    const void *d;
    uint32_t type, len;
    int handled = 0;
    pc_warm_one_(i, pass);
    if (!nv2a_pcache_get(s_pcf, i, &type, &d, &len))
        return;
    if (type == PC_PSO)
        handled = pass == 1;
    else if (type == PC_VS || type == PC_FS) {
        uint64_t kid = nv2a_fnv64(d, len, NV2A_FNV64_INIT);
        handled = pass == 2 || (pass == 0 && pc_needed(kid ? kid : 1));
    }
    if (handled && PC_URGENT()) {
        uint32_t dn = __atomic_add_fetch(&s_pc_done, 1, __ATOMIC_RELAXED);
        nv2a_pcache_stat_hold(dn > s_pc_total ? s_pc_total : dn, s_pc_total, s_pc_cold);
    }
}

/* Builds every key of the file, functions first, on the utility queues:
 * done before the title reaches the draws that need them, the ack thread
 * finds them built. Goes round again when asked to meanwhile (pc_warm_start:
 * a pre-build in another process has appended to the file), reading the
 * file anew each time. */
static void *pc_prewarm(void *arg)
{
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_UTILITY, 0);
    (void)arg;
    do {
        uint64_t t = pc_now(), before[3];
        uint32_t listed[3] = {0, 0, 0};
        size_t n, i;
        int pass;

        __atomic_store_n(&s_pc_rescan, 0, __ATOMIC_RELEASE);
        pthread_mutex_lock(&s_pcf_mu);  /* the ack thread's puts wait for the reopen */
        if (s_pcf) {
            char msg[256];
            nv2a_pcache_close(s_pcf);
            s_pcf = nv2a_pcache_open(s_pc_path, s_pc_ver, msg, sizeof msg);
        }
        pthread_mutex_unlock(&s_pcf_mu);
        n = nv2a_pcache_count(s_pcf);
        for (i = 0; i < n; i++) {
            const void *d;
            uint32_t type, len;
            if (nv2a_pcache_get(s_pcf, i, &type, &d, &len) && type <= PC_PSO)
                listed[type]++;
        }
        for (i = 0; i < 3; i++) {
            s_pc_loaded[i] = listed[i];
            before[i] = s_pc_warm[i];
        }
        nv2a_pcache_stat_open(s_pc_path, listed);
        s_pc_total = listed[0] + listed[1] + listed[2];
        __atomic_store_n(&s_pc_done, 0, __ATOMIC_RELAXED);
        free(s_pc_need);
        s_pc_need = (uint64_t *)malloc((listed[PC_PSO] ? listed[PC_PSO] : 1) * 2 * sizeof *s_pc_need);
        s_pc_need_n = 0;
        for (i = 0; s_pc_need && i < n; i++) {
            const void *d;
            uint32_t type, len;
            if (nv2a_pcache_get(s_pcf, i, &type, &d, &len) && type == PC_PSO
                    && len == sizeof(PsoKey) && s_pc_need_n + 2 <= (size_t)listed[PC_PSO] * 2) {
                memcpy(&s_pc_need[s_pc_need_n], d, 16);     /* PsoKey's vs, fs */
                s_pc_need_n += 2;
            }
        }
        /* The compiler service's full width while the pre-warm runs (WWDC23
         * 10127, "Optimize GPU renderers with Metal"); back to balanced after. */
        if (@available(macOS 13.3, *)) {
            if (n && s_dev.maximumConcurrentCompilationTaskCount)
                s_dev.shouldMaximizeConcurrentCompilation = YES;
        }
        for (pass = 0; pass < 3; pass++) {
            dispatch_apply(n, q, ^(size_t k) {
                @autoreleasepool {
                    pc_warm_one(k, pass);
                }
            });
            if (pass == 1)              /* the pipelines are in: nobody waits now */
                __atomic_store_n(&s_pc_warm_urgent, 0, __ATOMIC_RELEASE);
        }
        s_pc_warm_ns += pc_now() - t;
        nv2a_pcache_stat_warm_ms((uint32_t)((pc_now() - t) / 1000000ull));
        nv2a_pcache_stat_hold(s_pc_total, s_pc_total, s_pc_cold);
        if (@available(macOS 13.3, *))
            s_dev.shouldMaximizeConcurrentCompilation = NO;
        fprintf(stderr, LOGP "pcache: pre-warmed %llu vs, %llu fs, %llu pipelines of %u/%u/%u"
                " on disk in %.0f ms (done at %.2f s)\n",
                (unsigned long long)(s_pc_warm[PC_VS] - before[PC_VS]),
                (unsigned long long)(s_pc_warm[PC_FS] - before[PC_FS]),
                (unsigned long long)(s_pc_warm[PC_PSO] - before[PC_PSO]),
                listed[0], listed[1], listed[2],
                (double)(pc_now() - t) / 1e6, (double)(pc_now() - s_t0_ns) / 1e9);
    } while (__atomic_load_n(&s_pc_rescan, __ATOMIC_ACQUIRE));
    __atomic_store_n(&s_pc_warm_urgent, 0, __ATOMIC_RELEASE);
    nv2a_pcache_stat_warming(0);
    __atomic_store_n(&s_pc_warm_done, 1, __ATOMIC_RELEASE);
    return NULL;
}

/* Starts the pre-warm thread, or has the one running go round again.
 * `first`: the one the backend starts with, whose pipelines the ack thread
 * may wait for (pc_wait_more). */
static void pc_warm_begin(int first)
{
    pthread_t th;
    __atomic_store_n(&s_pc_rescan, 1, __ATOMIC_RELEASE);
    if (!__atomic_exchange_n(&s_pc_warm_done, 0, __ATOMIC_ACQ_REL))
        return;                         /* one is running: it will see the flag */
    nv2a_pcache_stat_warming(1);
    if (first) {
        s_pc_hold_t0 = pc_now();
        __atomic_store_n(&s_pc_wait_until, pc_now() + s_pc_hold_ns, __ATOMIC_RELEASE);
        __atomic_store_n(&s_pc_warm_urgent, 1, __ATOMIC_RELEASE);
    }
    if (pthread_create(&th, NULL, pc_prewarm, NULL) == 0) {
        pthread_detach(th);
    } else {
        __atomic_store_n(&s_pc_warm_urgent, 0, __ATOMIC_RELEASE);
        nv2a_pcache_stat_warming(0);
        __atomic_store_n(&s_pc_warm_done, 1, __ATOMIC_RELEASE);
    }
}

/* nv2a_pcache_rescan: the list gained keys (a pre-build in another process). */
static void pc_warm_start(void)
{
    pc_warm_begin(0);
}

/* With the device up: the tables, the trace, the key list and its
 * pre-warm thread. 0 if the tables cannot be allocated. */
static int pc_start(void)
{
    const char *tr = recomp_env(RENV_PCACHE_TRACE);
    long max = recomp_env_int(RENV_METAL_PCACHE_MAX, PC_MAX_DEFAULT);
    char dir[1024], msg[256];
    size_t i;

    s_pc_max = max < 1 ? 1 : max > PC_MAX_LIMIT ? PC_MAX_LIMIT : (int)max;
    s_vs = (VsEntry *)calloc((size_t)s_pc_max, sizeof *s_vs);
    s_fs = (FsEntry *)calloc((size_t)s_pc_max, sizeof *s_fs);
    s_pso = (PsoEntry *)calloc((size_t)s_pc_max, sizeof *s_pso);
    if (!s_vs || !s_fs || !s_pso)
        return 0;
    for (i = 0; i < PSO_BUCKETS; i++)
        s_pso_head[i] = -1;
    if (tr) {
        s_pc_trace = 1;
        if (strcmp(tr, "1") != 0) {
            if ((s_pc_csv = fopen(tr, "w")) != NULL)
                fprintf(s_pc_csv, "flip,t_ms,interval_ms,ack_compiles,ack_compile_ms\n");
            else
                XR_ERROR( LOGP "pcache: cannot write %s\n", tr);
        }
    }

    s_pc_warm_done = 1;
    if (!nv2a_pcache_dir(nv2a_host_opts()->shader_cache, nv2a_pcache_app(), dir, sizeof dir)) {
        fprintf(stderr, LOGP "pcache: no disk cache (render.shader_cache=%s); %d entries"
                " per table\n", nv2a_host_opts()->shader_cache ? nv2a_host_opts()->shader_cache
                                                               : "(default, unavailable)",
                s_pc_max);
        return 1;
    }
    snprintf(s_pc_path, sizeof s_pc_path, "%s/metal_pipelines.bin", dir);
    /* What the keys mean: the key-format number (raise it when a key's
     * meaning changes at the same size) and the structures written as they
     * are. Not the build: the file holds keys, the source is generated from
     * them at every start, so a rebuild keeps it (nv2a_pcache.h). */
    snprintf(s_pc_ver, sizeof s_pc_ver, "metal pipelines 1; PsoKey %zu; NV2ACombinerState %zu;"
             " vsh slots %d", sizeof(PsoKey), sizeof(NV2ACombinerState),
             (int)NV2A_VSH_SLOTS);
    s_pcf = nv2a_pcache_open(s_pc_path, s_pc_ver, msg, sizeof msg);
    fprintf(stderr, LOGP "pcache: %s: %s; %d entries per table\n", s_pc_path, msg, s_pc_max);
    if (!s_pcf)
        return 1;
    {
        uint32_t none[3] = {0, 0, 0};
        nv2a_pcache_stat_open(s_pc_path, none);
    }
    {
        char gp[1100], gv[300], gm[128];
        snprintf(gp, sizeof gp, "%s/gaps_metal.bin", dir);
        snprintf(gv, sizeof gv, "%s; gaps", s_pc_ver);
        s_gapf = nv2a_pcache_open(gp, gv, gm, sizeof gm);
    }
    nv2a_pcache_set_rescan(pc_warm_start);
    nv2a_pcache_set_seed(pc_seed);
    {
        /* BO-208: the shipped manifest's keys join the list; a first build
         * holds the title for its pipelines behind "Preparing graphics N/M".
         * (Metal keeps its compiled code in the system's cache, which the
         * app cannot see: the manifest bringing keys is what makes it cold.) */
        nv2a_pcache_manifest_stat ms;
        uint32_t added = nv2a_pcache_manifest_apply(s_pcf, "metal", s_pc_ver, &ms);
        long hold = recomp_env_int(RENV_PCACHE_HOLD_S, 180);
        s_pc_cap_ns = (uint64_t)(hold < 3 ? 3 : hold) * 1000000000ull;
        s_pc_cold = added > 0 && (added > 64 || nv2a_pcache_count(s_pcf) > 64);
        s_pc_hold_ns = s_pc_cold ? (uint64_t)(hold < 3 ? 3 : hold) * 1000000000ull
                                 : 3000000000ull;
        fprintf(stderr, LOGP "pcache: %s start (%u keys new from the manifest), the title may be"
                " held up to %llu s\n", s_pc_cold ? "cold" : "warm", added,
                (unsigned long long)(s_pc_hold_ns / 1000000000ull));
    }
    if (nv2a_pcache_count(s_pcf) || s_pc_cold)
        pc_warm_begin(1);
    return 1;
}

/* At every flip: the LRU clock, and the CSV line. */
static void pc_flip(uint64_t flip)
{
    static uint64_t last;
    uint64_t t = pc_now(), n, ns;
    pthread_mutex_lock(&s_pc_mu);
    n = s_cs.rt_flip_n;
    ns = s_cs.rt_flip_ns;
    s_cs.rt_flip_n = s_cs.rt_flip_ns = 0;
    pthread_mutex_unlock(&s_pc_mu);
    s_tick++;
    if (s_pc_csv) {
        fprintf(s_pc_csv, "%llu,%.3f,%.3f,%llu,%.3f\n", (unsigned long long)flip,
                (double)(t - s_t0_ns) / 1e6, last ? (double)(t - last) / 1e6 : 0.0,
                (unsigned long long)n, (double)ns / 1e6);
        if ((flip % 60) == 0)
            fflush(s_pc_csv);
    }
    last = t;
}

/* The present log's pipeline-cache line. */
static void pc_log(uint64_t presents)
{
    int i, nv = 0, nf = 0, np = 0;
    pthread_mutex_lock(&s_pc_mu);
    for (i = 0; i < s_pc_max; i++) {
        nv += s_vs[i].id != 0;
        nf += s_fs[i].id != 0;
        np += s_pso[i].used;
    }
    fprintf(stderr, LOGP "pcache at present #%llu: cached %d vs, %d fs, %d pipelines (max %d);"
            " ack-thread compiles %llu vs + %llu fs + %llu pipelines in %.1f ms (max %.1f,"
            " %llu in the first 10 s); pre-warm %llu + %llu + %llu in %.1f ms (%s);"
            " evicted %llu + %llu + %llu; start-up waits %llu, %.1f ms\n",
            (unsigned long long)presents, nv, nf, np, s_pc_max,
            (unsigned long long)s_cs.n[0][PC_VS], (unsigned long long)s_cs.n[0][PC_FS],
            (unsigned long long)s_cs.n[0][PC_PSO],
            (double)(s_cs.ns[0][0] + s_cs.ns[0][1] + s_cs.ns[0][2]) / 1e6, s_cs.rt_max_ms,
            (unsigned long long)s_cs.rt_10s,
            (unsigned long long)s_cs.n[1][PC_VS], (unsigned long long)s_cs.n[1][PC_FS],
            (unsigned long long)s_cs.n[1][PC_PSO],
            (double)(s_cs.ns[1][0] + s_cs.ns[1][1] + s_cs.ns[1][2]) / 1e6,
            __atomic_load_n(&s_pc_warm_done, __ATOMIC_ACQUIRE) ? "done" : "running",
            (unsigned long long)s_cs.evicted[PC_VS], (unsigned long long)s_cs.evicted[PC_FS],
            (unsigned long long)s_cs.evicted[PC_PSO],
            (unsigned long long)s_pc_waits, (double)s_pc_wait_ns / 1e6);
    pthread_mutex_unlock(&s_pc_mu);
}

/* ---- depth / stencil state ---------------------------------------------------
 *
 * As the D3D11 backend's depth_state: SET_DEPTH_FUNC and SET_STENCIL_FUNC
 * are GL compare codes in MTLCompareFunction's order (nv2a_cmp_from_gl), the
 * stencil ops in MTLStencilOperation's (nv2a_stencil_op_from_gl). Depth and
 * stencil each only when enabled; one state per distinct key. */
#define DSS_MAX 64
static struct { uint64_t key; id<MTLDepthStencilState> dss; } s_dss[DSS_MAX];
static int s_dss_n;

static id<MTLDepthStencilState> depth_state(int depth, int stencil)
{
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    uint32_t f = nv2a_cmp_from_gl(v->depth_func);
    uint64_t key = 0;
    MTLDepthStencilDescriptor *dd;
    id<MTLDepthStencilState> st;
    int i;

    if (depth)
        key |= 1u | f << 1 | (v->depth_mask ? 1u : 0u) << 4;
    if (stencil)
        key |= (uint64_t)(1u | (v->stencil_func & 7u) << 1
                          | nv2a_stencil_op_from_gl(v->stencil_op[0]) << 4
                          | nv2a_stencil_op_from_gl(v->stencil_op[1]) << 8
                          | nv2a_stencil_op_from_gl(v->stencil_op[2]) << 12
                          | (v->stencil_rmask & 0xFFu) << 16
                          | (v->stencil_wmask & 0xFFu) << 24) << 8;
    for (i = 0; i < s_dss_n; i++)
        if (s_dss[i].key == key)
            return s_dss[i].dss;
    dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = depth ? (MTLCompareFunction)f : MTLCompareFunctionAlways;
    dd.depthWriteEnabled = depth && v->depth_mask ? YES : NO;
    if (stencil) {
        MTLStencilDescriptor *sd = [[MTLStencilDescriptor alloc] init];
        sd.stencilCompareFunction = (MTLCompareFunction)(v->stencil_func & 7u);
        sd.stencilFailureOperation = (MTLStencilOperation)nv2a_stencil_op_from_gl(v->stencil_op[0]);
        sd.depthFailureOperation = (MTLStencilOperation)nv2a_stencil_op_from_gl(v->stencil_op[1]);
        sd.depthStencilPassOperation = (MTLStencilOperation)nv2a_stencil_op_from_gl(v->stencil_op[2]);
        sd.readMask = v->stencil_rmask & 0xFFu;
        sd.writeMask = v->stencil_wmask & 0xFFu;
        dd.frontFaceStencil = sd;               /* GL one-sided stencil */
        dd.backFaceStencil = sd;
        [sd release];
    }
    st = [s_dev newDepthStencilStateWithDescriptor:dd];
    [dd release];
    if (!st)
        return nil;
    if (s_dss_n >= DSS_MAX) {
        /* Full: replace one in turn, as the sampler cache does (a state is
         * a descriptor, not a compile; an encoded draw keeps its own
         * reference). Before, a state past 64 was made per draw and leaked. */
        static unsigned next;
        i = (int)(next++ % DSS_MAX);
        [s_dss[i].dss release];
        s_dss[i].key = key;
        s_dss[i].dss = st;
        return st;
    }
    s_dss[s_dss_n].key = key;
    s_dss[s_dss_n].dss = st;
    s_dss_n++;
    return st;
}

/* ---- render-to-texture ---------------------------------------------------------
 *
 * A stage whose texture is a surface this backend draws into samples the
 * render target itself, as the D3D11 backend's rt_texture: guest memory
 * under it only changes at a flip's write-back (and then only for the
 * present surface). 32-bit, non-compressed formats only; the low 27 bits of
 * the address compare, so either memory window matches. The target being
 * drawn into is sampled through a copy (s_self), made by a blit between
 * passes. RECOMP_METAL_NO_RTT=1 decodes guest memory instead (A/B).
 *
 * A 16-bit (R5G6B5) surface sampled as (LIN_)R5G6B5 binds too, as xemu binds
 * a surface whose host texel size matches the texture's (hw/xbox/nv2a/
 * pgraph/vk/texture.c, check_surface_to_texture_compatiblity). Burnout 3's
 * Crash mode samples its 320x240 and 160x120 R5G6B5 work targets and a
 * swizzled 128x128 one every frame; decoding them meant a GPU wait and a
 * read-back each time, five a frame (the swizzled one was also written back
 * in rows and decoded in Morton order). The target is BGRA8, so it is
 * sampled through a view whose alpha reads 1, as an R5G6B5 texel's does; its
 * colour keeps 8 bits a channel where the console's would have 5-6-5. */
static RenderTarget s_self;   /* tex, w, h: the copy; offset: whose */
static RenderTarget s_opaque; /* tex: an rt's opaque view, for one bind */

/* rt (a 16-bit surface) as a texture whose alpha reads 1: a swizzled view,
 * made on first use and kept until rt's texture goes. NULL if Metal cannot
 * make one; the texture then decodes from guest memory as before. */
static RenderTarget *rt_opaque(RenderTarget *rt)
{
    if (!rt->opaque) {
        MTLTextureSwizzleChannels sw = MTLTextureSwizzleChannelsMake(
            MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
            MTLTextureSwizzleBlue, MTLTextureSwizzleOne);
        rt->opaque = [rt->tex newTextureViewWithPixelFormat:[rt->tex pixelFormat]
                                                textureType:MTLTextureType2D
                                                     levels:NSMakeRange(0, 1)
                                                     slices:NSMakeRange(0, 1)
                                                    swizzle:sw];
        if (!rt->opaque)
            return NULL;
    }
    s_opaque = *rt;
    s_opaque.tex = rt->opaque;
    return &s_opaque;
}

/* ---- cube maps (BO-8) ---------------------------------------------------------
 *
 * A CUBEMAP stage samples a texturecube (d3d8_combiners_msl.c declares one
 * for it); binding a 2D texture there, as every stage was, left the car
 * reflections wrong. The six faces sit face_stride apart in guest memory
 * (nv2a_tex_face_stride: a face with all its levels, 128-byte aligned, as
 * xemu's texture.c). A face this backend drew -- Burnout 3 renders its car
 * reflections into six 128x128 targets at 0x00AFE000 + n * 0x10000 every
 * frame -- is copied from that target on the GPU whenever it was drawn since
 * the last copy: nothing writes those targets back, so the guest bytes under
 * them are not the image. xemu likewise renders a surface into the texture
 * that reads it (hw/xbox/nv2a/pgraph/gl/surface.c,
 * pgraph_gl_render_surface_to_texture). The other faces decode from guest
 * memory, re-checked once per flip as 2D textures are; a change makes a new
 * cube, so a frame still in flight keeps the one it was encoded with. */
#define CUBE_MAX 8
typedef struct {
    uint32_t       offset, color, width, levels, side;   /* side: host face size */
    uint32_t       gen, used;     /* flip of the last memory check; LRU */
    uint64_t       hash;          /* the memory faces' bytes */
    uint64_t       face_gen[6];   /* the target's gen at its last copy */
    id<MTLTexture> tex;
} CubeEntry;
static CubeEntry      s_cube[CUBE_MAX];
static uint32_t       s_cube_clock;
static id<MTLTexture> s_black_cube;
static uint64_t       s_cube_copies, s_cube_uploads;

static id<MTLTexture> black_cube(void)
{
    if (!s_black_cube) {
        static const uint32_t px = 0xFF000000u;
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            textureCubeDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm size:1 mipmapped:NO];
        int f;
        td.usage = MTLTextureUsageShaderRead;
        s_black_cube = [s_dev newTextureWithDescriptor:td];
        for (f = 0; s_black_cube && f < 6; f++)
            [s_black_cube replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:f
                              withBytes:&px bytesPerRow:4 bytesPerImage:4];
    }
    return s_black_cube;
}

/* The target drawn at a face's address, w x w guest pixels of it, or NULL. */
static RenderTarget *cube_face_rt(uint32_t addr, uint32_t w)
{
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex || ((rt->addr ^ addr) & 0x07FFFFFFu) || rt->w < w || rt->h < w)
            continue;
        if (rt_drop_if_stale(rt, "bind"))
            continue;   /* the title rewrote it: the face decodes from memory */
        return rt;
    }
    return NULL;
}

static id<MTLTexture> cube_bind(const Stage *t)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    const uint32_t *pal = t->color == NV2A_TEX_P8 ? (const uint32_t *)(mem + t->palette) : NULL;
    RenderTarget *frt[6];
    uint32_t w = t->width, side = t->width, levels, stride, bytes, f, l;
    CubeEntry *e = NULL, *victim = &s_cube[0];
    int any = 0, all = 1, fresh = 0, ncopy = 0, i;

    if (!t->cube || !w || t->width != t->height || (uint64_t)w * w * 6u > TEX_MAX_TEXELS)
        return black_cube();
    stride = nv2a_tex_face_stride(t->color, w, w, t->pitch, t->fmt_levels);
    bytes = nv2a_tex_extent(t->color, w, w, t->pitch, t->levels);
    if (!bytes)
        return black_cube();
    for (f = 0; f < 6; f++) {
        frt[f] = cube_face_rt(t->offset + f * stride, w);
        if (frt[f]) {
            any = 1;
            side = w * frt[f]->hw / frt[f]->w;
        } else {
            all = 0;
        }
    }
    levels = any ? 1u : t->levels;     /* a drawn face has one level */
    for (i = 0; i < CUBE_MAX; i++) {
        CubeEntry *c = &s_cube[i];
        if (c->tex && c->offset == t->offset && c->color == t->color
                && c->width == w && c->levels == levels) {
            e = c;
            break;
        }
        if (!c->tex || (victim->tex && c->used < victim->used))
            victim = c;
    }
    if (!e) {
        e = victim;
        [e->tex release];
        memset(e, 0, sizeof *e);
        e->offset = t->offset;
        e->color = t->color;
        e->width = w;
        e->levels = levels;
        fresh = 1;
    }
    if (e->side != side || (e->tex && [e->tex pixelFormat] != RT_FMT))
        fresh = 1;
    if (!all && (fresh || e->gen != g->flips)) {
        uint64_t h = NV2A_TEX_HASH_SEED;
        for (f = 0; f < 6; f++)
            if (!frt[f])
                h = h * 31u + nv2a_tex_hash(mem + t->offset + f * stride, bytes,
                                            NV2A_TEX_HASH_SEED);
        if (pal)
            h ^= nv2a_tex_hash(mem + t->palette, t->pal_len * 4u, NV2A_TEX_HASH_SEED) * 31u;
        e->gen = g->flips;
        if (h != e->hash)
            fresh = 1;
        e->hash = h;
    }
    if (fresh) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            textureCubeDescriptorWithPixelFormat:RT_FMT size:side
                                       mipmapped:levels > 1];
        size_t n = nv2a_tex_texels(w, w, levels);
        if (levels > 1)
            td.mipmapLevelCount = levels;
        td.usage = MTLTextureUsageShaderRead;
        [e->tex release];
        e->tex = [s_dev newTextureWithDescriptor:td];
        e->side = side;
        memset(e->face_gen, 0, sizeof e->face_gen);
        if (!e->tex)
            return black_cube();
        if (n > s_texels_cap) {
            uint32_t *p = (uint32_t *)realloc(s_texels, n * 4);
            if (!p)
                return black_cube();
            s_texels = p;
            s_texels_cap = n;
        }
        for (f = 0; f < 6; f++) {
            const uint32_t *src = s_texels;
            if (frt[f])
                continue;
            nv2a_tex_decode(mem + t->offset + f * stride, t->color, w, w, t->pitch,
                            levels, pal, pal ? t->pal_len : 0, s_texels);

            if (side != w) {    /* beside drawn faces at render.scale: level 0, upscaled */
                uint8_t *up = (uint8_t *)malloc((size_t)side * side * 4u);
                if (up) {
                    nv2a_upscale_nearest32((const uint8_t *)s_texels, w * 4u, w, w,
                                           side / w, up, side * 4u);
                    if (tex_is_half(e->tex))
                        tex_put_half(e->tex, MTLRegionMake2D(0, 0, side, side), 0, f,
                                     up, side * 4u);
                    else
                    [e->tex replaceRegion:MTLRegionMake2D(0, 0, side, side) mipmapLevel:0
                                    slice:f withBytes:up bytesPerRow:side * 4u
                            bytesPerImage:side * side * 4u];
                    free(up);
                }
                continue;
            }
            for (l = 0; l < levels; l++) {
                uint32_t lw = nv2a_tex_level_dim(w, l);
                if (tex_is_half(e->tex))
                    tex_put_half(e->tex, MTLRegionMake2D(0, 0, lw, lw), l, f, src, lw * 4u);
                else
                [e->tex replaceRegion:MTLRegionMake2D(0, 0, lw, lw) mipmapLevel:l slice:f
                            withBytes:src bytesPerRow:lw * 4u bytesPerImage:lw * lw * 4u];
                src += (size_t)lw * lw;
            }
        }
        s_cube_uploads++;
    }
    for (f = 0; f < 6; f++)
        if (frt[f] && frt[f]->gen != e->face_gen[f])
            ncopy++;
    if (ncopy) {
        id<MTLBlitCommandEncoder> b;
        end_encoder();
        ensure_cmd();
        b = [s_cmd blitCommandEncoder];
        for (f = 0; f < 6; f++) {
            if (!frt[f] || frt[f]->gen == e->face_gen[f])
                continue;
            [b copyFromTexture:frt[f]->tex sourceSlice:0 sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(side, side, 1)
                     toTexture:e->tex destinationSlice:f destinationLevel:0
             destinationOrigin:MTLOriginMake(0, 0, 0)];
            e->face_gen[f] = frt[f]->gen;
            s_cube_copies++;
        }
        [b endEncoding];
    }
    e->used = ++s_cube_clock;
    return e->tex;
}

static RenderTarget *rt_texture(const Stage *t, RenderTarget *cur)
{
    static int off = -1;
    RenderTarget *self = NULL;
    int i, r565;

    if (off < 0)
        off = recomp_env(RENV_METAL_NO_RTT) != NULL;
    if (off || !t->offset || d3d8_format_dxt_block_bytes(t->color))
        return NULL;
    r565 = nv2a_tex_linear_twin(t->color) == 0x11;   /* (LIN_)R5G6B5 */
    if (!r565 && nv2a_tex_texel_bytes(nv2a_tex_linear_twin(t->color)) != 4)
        return NULL;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex || ((rt->addr ^ t->offset) & 0x07FFFFFFu))
            continue;
        /* 16-bit textures only on 16-bit surfaces of the same size, a linear
         * one at the same pitch, a swizzled one only on a swizzled surface
         * (xemu's rule); 32-bit as before. */
        if (r565 && (rt->pitch / rt->w != 2u || rt->w != t->width || rt->h != t->height
                     || (t->color == 0x11 ? rt->swz || t->pitch != rt->pitch : !rt->swz)))
            continue;
        if (rt_drop_if_stale(rt, "bind"))
            continue;   /* the texture decodes from the title's bytes */
        if (rt == cur) {
            if (!r565)
                self = rt;
            continue;
        }
        return r565 ? rt_opaque(rt) : rt;
    }
    if (!self)
        return NULL;
    /* copyFromTexture:toTexture: wants equal sizes: the source's host size. */
    if (!s_self.tex || s_self.w != self->w || s_self.h != self->h
            || s_self.hw != self->hw || s_self.hh != self->hh
            || [s_self.tex pixelFormat] != [self->tex pixelFormat]) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:[self->tex pixelFormat]
                                         width:self->hw height:self->hh mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        [s_self.tex release];
        s_self.tex = [s_dev newTextureWithDescriptor:td];
        if (!s_self.tex)
            return NULL;
        s_self.w = self->w;
        s_self.h = self->h;
        s_self.hw = self->hw;
        s_self.hh = self->hh;
    }
    {
        id<MTLBlitCommandEncoder> b;
        end_encoder();
        ensure_cmd();
        b = [s_cmd blitCommandEncoder];
        [b copyFromTexture:self->tex toTexture:s_self.tex];
        [b endEncoding];
    }
    s_self.offset = self->offset;
    s_self_copies++;
    mf_add('X', self);
    {
        static int said;
        if (!said++)
            fprintf(stderr, LOGP "surface 0x%08X sampled while drawn into:"
                    " bound as a copy\n", self->offset);
    }
    return &s_self;
}

/* ---- draws ------------------------------------------------------------------- */

/* The scissor a draw takes: the surface clip (SET_SURFACE_CLIP_HORIZONTAL/
 * VERTICAL), clamped to rt, in host pixels. The NV2A writes no pixel outside
 * the clip; xemu scissors every draw to it (pgraph/gl/draw.c,
 * glScissor(clip_x, clip_y, clip_width, clip_height)) and the CPU path's
 * rasteriser bounds each triangle by it. A target serves every clip of its
 * surface (rt_grow), so without the scissor a draw spills over pixels its
 * clip leaves alone: Burnout 3 draws its frame under a 640x448 clip at y 16
 * of the 640x480 back buffer, and the full-screen passes that reach past it
 * (the speed blur's scaled copies of the frame, HUD included) painted the
 * 16-row bands above and below it with displaced HUD and scenery, a
 * different picture in each back buffer. 0 when the clip leaves nothing of
 * rt. */
static int draw_scissor(const struct nv2a_pb_gpu *g, const RenderTarget *rt,
                        MTLScissorRect *sr)
{
    uint32_t x0 = g->clip_x, y0 = g->clip_y;
    uint32_t x1 = g->clip_x + g->clip_w, y1 = g->clip_y + g->clip_h;

    if (x1 > rt->w) x1 = rt->w;
    if (y1 > rt->h) y1 = rt->h;
    if (x0 >= x1 || y0 >= y1)
        return 0;
    sr->x = (NSUInteger)((uint64_t)x0 * rt->hw / rt->w);
    sr->y = (NSUInteger)((uint64_t)y0 * rt->hh / rt->h);
    sr->width = (NSUInteger)((uint64_t)x1 * rt->hw / rt->w) - sr->x;
    sr->height = (NSUInteger)((uint64_t)y1 * rt->hh / rt->h) - sr->y;
    return 1;
}

static void metal_on_draw(const struct nv2a_pb_draw *d)
{
    /* The expanded list: at most three entries per index (a fan or a quad
     * strip), grown to the largest batch so far rather than sized to
     * NV_MAX_INDICES. */
    static uint32_t *idx;
    static uint32_t idx_cap;
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    enum nv2a_topology topo;
    MTLPrimitiveType ptype;
    RenderTarget *rt;
    id<MTLRenderPipelineState> pso;
    id<MTLFunction> vfn, ffn;
    uint64_t vid, fid;
    struct nv2a_vp_consts vc;
    MsPsConsts mpc;
    NSUInteger vb_off, ib_off, vbytes;
    uint32_t lo = 0xFFFF, hi = 0, count, nidx, i, k, nf4 = 0;
    int slot[NV2A_VS_MAX_INPUTS];
    float *vf;
    void *ip;
    int cull = 0, cull_all = 0;
    float zbias = 0.0f;
    int solo = 0;                 /* BO-203: stage 0 is a stage0_only HDR picture */
    /* Stencil only on a Z24S8 zeta surface, as on the CPU. */
    int use_s = v->stencil_enable && ((g->format >> 4) & 0xF) == 2;
    int use_z = v->depth_enable || use_s;
    id<MTLTexture> stex[4];
    unsigned sscale[4] = { 0, 0, 0, 0 };

    if (!init())
        return;
    /* A batch that writes nothing is dropped, unless a visibility test is
     * counting it: a test's bounding draws typically mask every write. */
    if (g->color_keep == 0xFFFFFFFFu && !(v->depth_enable && v->depth_mask) && !use_s
            && !s_occ_q)
        return;
    if (!d->idx_count) {
        s_draws_skipped++;
        return;
    }
    @autoreleasepool {
        /* Vertex function and the float4 inputs it reads, in order. */
        if (d->program) {
            VsEntry *e = vs_cache();
            s_draws_prog++;
            if (!e) {
                s_draws_skipped++;
                return;
            }
            vfn = e->fn;
            vid = e->id;
            for (i = 0; i < NV2A_VS_MAX_INPUTS; i++)
                if (e->inputs & (1u << i))
                    slot[nf4++] = (int)i;
        } else {
            vfn = s_vs_pass;
            vid = PC_VS_PASS;
            for (k = 0; k < VTX_F4; k++)
                slot[nf4++] = s_vtx_attr[k];
        }
        rt = surface();
        if (!rt) {
            s_draws_skipped++;
            return;
        }
        build_stages();
        nv2a_ps_consts_fill(v, s_stage, &mpc.b);
        float host_bias = nv2a_host_lod_bias();   /* RECOMP_GFX_LOD_BIAS; 0 by default */
        /* BO-171: a cut-out draw (alpha test) takes no negative host bias: a
         * finer level than the pixel's own makes its edge shimmer, and the
         * levels nv2a_texmip makes keep the leaves without it. */
        if (host_bias < 0.0f && (nv2a_texmip_mode() & NV2A_TEXMIP_BIAS)
                && nv2a_texmip_cut_ref(v->atest_enable, v->atest_func, v->atest_ref))
            host_bias = 0.0f;
        for (k = 0; k < 4; k++) {
            /* As the D3D11 MipLODBias: only a mipmapped stage has one. The
             * host's own (RECOMP_GFX_LOD_BIAS, 0 by default) adds to the
             * title's; the shader applies the sum with bias(), which the
             * sampler's anisotropy and LOD clamps then act on. */
            const Stage *t = &s_stage[k];
            float b = t->levels > 1 && filter_mip(t->filter) ? lod_bias(t->filter) + host_bias : 0.0f;
            mpc.tex_bias[k] = b > 15.99f ? 15.99f : b < -16.0f ? -16.0f : b;
        }
        {   /* Bump-env registers: methods 0x1B28..0x1B3C of each stage block. */
            const uint32_t *tr = nv2a_pb_tex_regs();
            for (k = 0; k < 4; k++) {
                const uint32_t *r = &tr[(0x40u * k + 0x28u) / 4];
                memcpy(mpc.bump_mat[k], r, 16);
                memcpy(&mpc.bump_lum[k][0], &r[4], 4);
                memcpy(&mpc.bump_lum[k][1], &r[5], 4);
                mpc.bump_lum[k][2] = mpc.bump_lum[k][3] = 0.0f;
            }
        }
        /* BO-110: the colour bound of the HDR combiners, the scene's between
         * the title's scene marks, else the hardware's. */
        mpc.hdr[0] = s_hdr && nv2a_pb_stream_tag() == NV2A_HDR_TAG_SCENE ? s_hdr_hi : 1.0f;
        mpc.hdr[1] = mpc.hdr[2] = mpc.hdr[3] = 0.0f;
        if (s_hdr) {
            int scene = nv2a_pb_stream_tag() == NV2A_HDR_TAG_SCENE;
            s_hdr_draws[scene]++;
            if (scene)
                s_hdr_pend = rt;
            else if (s_hdr_pend)
                hdr_snapshot();
        }
        ffn = fs_select(v, &fid);
        s_signed_on = fid != PC_FS_BASIC;   /* fs_basic has no signed channels */
        nv2a_idx_range(d->idx, d->idx_count, &lo, &hi);
        count = hi - lo + 1;
        if (d->idx_count * 3u > idx_cap) {
            uint32_t *grown = (uint32_t *)realloc(idx, (size_t)d->idx_count * 12u);
            if (!grown) {
                s_draws_skipped++;
                return;
            }
            idx = grown;
            idx_cap = d->idx_count * 3u;
        }
        nidx = nv2a_prim_to_list(d->prim, d->idx, d->idx_count, lo, idx, &topo);
        if (!nidx) {
            s_draws_skipped++;
            return;
        }
        ptype = topo == NV2A_TOPO_POINTS ? MTLPrimitiveTypePoint
              : topo == NV2A_TOPO_LINES  ? MTLPrimitiveTypeLine
              :                            MTLPrimitiveTypeTriangle;
        if (v->cull_enable) {     /* as the D3D11 backend's rasterizer_state */
            if (v->cull_face == 0x405)      cull = 1;   /* BACK */
            else if (v->cull_face == 0x404) cull = 2;   /* FRONT */
            else if (v->cull_face == 0x408) cull_all = 1;
        }
        if (cull_all && ptype == MTLPrimitiveTypeTriangle) {
            s_draws_culled++;
            return;
        }

        /* Rings first: an overflow flushes, which ends the open pass. */
        vbytes = (NSUInteger)count * nf4 * 16u;
        if (vbytes < 16)
            vbytes = 16;
        vf = (float *)ring_alloc(s_vb, s_vb_size, &s_vb_pos, vbytes, &vb_off);
        {
            uint64_t fl = s_flushes;
            ip = vf ? ring_alloc(s_ib, s_ib_size, &s_ib_pos, (NSUInteger)nidx * 4u, &ib_off)
                    : NULL;
            if (vf && ip && fl != s_flushes) {
                /* The index ring overflowed after the vertices were placed,
                 * and its flush reset the vertex ring: place them again. */
                vf = (float *)ring_alloc(s_vb, s_vb_size, &s_vb_pos, vbytes, &vb_off);
            }
        }
        if (!vf || !ip) {
            s_draws_skipped++;
            return;
        }
        /* Each input as float4, fetched as the CPU path does; a disabled
         * one (or one without data) is its SET_VERTEX_DATA value. */
        nv2a_vtx_gather(d, slot, nf4, lo, count, v->inl,
                        (const uint8_t *)xbox_GetMemoryOffset(), vf);
        memcpy(ip, idx, (size_t)nidx * 4u);

        nv2a_vp_consts_fill(v, g->format, rt->w, rt->h, &vc);

        /* Depth: the zeta surface's buffer, attached to this target's
         * passes from now on (a new pass if the open one lacks it). */
        if (use_z) {
            DepthTarget *dt = depth_target(rt->w, rt->h);
            if (dt && rt->dt != dt) {
                rt->dt = dt;
                if (s_enc_rt == rt)
                    end_encoder();
            }
        }
        /* Textures before the pass: a self copy is a blit between passes. */
        {
            int textured = 0;
            s_draw_2d = 0;
            if (nv2a_texscale_factor() || ui_scale_log()) {
                const struct nv2a_pb_present *pr = nv2a_pb_present_state();
                struct nv2a_texscale_draw td2 = { d->program, v->depth_enable,
                                                  v->depth_enable && v->depth_mask,
                                                  rt->w, rt->h, pr->w, pr->h };
                s_draw_2d = nv2a_texscale_draw_is_2d(&td2);
            }
            for (k = 0; k < 4; k++) {
                const Stage *t = &s_stage[k];
                RenderTarget *src;
                stex[k] = s_black;
                if (!t->mode || t->mode == 4)
                    continue;
                textured = 1;
                if (t->mode == 3 && t->cube && fid != PC_FS_BASIC) {   /* texturecube */
                    stex[k] = cube_bind(t);
                } else if ((t->mode == 3 || !stage_signed(t))
                           && (src = rt_texture(t, rt)) != NULL) {
                    stex[k] = src->tex;
                    /* Linear coordinates are texels of the surface as drawn. */
                    if (!nv2a_tex_size_from_format(t->color)) {
                        mpc.b.tex_scale[k][0] = 1.0f / (float)src->w;
                        mpc.b.tex_scale[k][1] = 1.0f / (float)src->h;
                    }
                    s_rtt_binds++;
                } else {
                    stex[k] = tex_bind(t);
                    sscale[k] = s_bind_scale;
                    if (zbias == 0.0f)
                        zbias = s_bind_zbias;
                    if (k == 0)
                        solo = s_bind_solo;
                }
            }
            if (textured)
                s_draws_tex++;
            /* BO-203: an HDR picture that stands alone (the sun's disc): the
             * other stages read transparent black, whatever the title bound. */
            if (solo && s_clear && nv2a_texrep_hdr())
                for (k = 1; k < 4; k++) {
                    const Stage *t = &s_stage[k];
                    if (t->mode && t->mode != 4 && !(t->mode == 3 && t->cube))
                        stex[k] = s_clear;
                }
        }
        /* BO-176: a registered decal's depth offset, added to the constant
         * part of the title's own polygon offset (vp_scale.w). */
        if (zbias != 0.0f && use_z && nv2a_texrep_zbias_on(g->flips))
            vc.c[3] += zbias / nv2a_zmax(g->format);

        if (s_hdr)
            hdr_texlog(g);
        if (!s_enc || s_enc_rt != rt)
            begin_encoder(rt, 0, NULL);
        pso = pipeline(vid, vfn, fid, ffn);
        if (!pso) {
            s_draws_skipped++;
            return;
        }
        [s_enc setRenderPipelineState:pso];
        if (s_enc_dt) {
            id<MTLDepthStencilState> dss = use_z ? depth_state(v->depth_enable, use_s)
                                                 : depth_state(0, 0);
            [s_enc setDepthStencilState:dss];
            /* BO-10 polygon offset: the constant part is in the vertex
             * stage (vp_scale.w, nv2a_vp_consts_fill), exact in depth
             * units; the slope part is Metal's, per host pixel, so the
             * guest-pixel factor scales with render.scale. xemu applies
             * both to its own interpolated z (pgraph/glsl/psh.c:
             * zvalue += depthOffset + depthFactor * triMZ; PR #2240). */
            [s_enc setDepthBias:0.0f
                     slopeScale:(use_z && v->poly_offset)
                                ? v->poly_factor * (float)nv2a_host_render_scale() : 0.0f
                          clamp:0.0f];
            [s_enc setStencilReferenceValue:use_s ? (v->stencil_ref & 0xFFu) : 0u];
            if (use_z)
                s_draws_z++;
        }
        {
            MTLViewport vp = {0.0, 0.0, (double)rt->hw, (double)rt->hh, 0.0, 1.0};
            MTLScissorRect sr;
            if (!draw_scissor(g, rt, &sr)) {
                s_draws_skipped++;
                return;
            }
            [s_enc setViewport:vp];
            [s_enc setScissorRect:sr];
        }
        [s_enc setFrontFacingWinding:v->front_face != 0x900
                                     ? MTLWindingCounterClockwise : MTLWindingClockwise];
        [s_enc setCullMode:cull == 1 ? MTLCullModeBack
                         : cull == 2 ? MTLCullModeFront : MTLCullModeNone];
        {
            float bf[4];
            (void)nv2a_blend_constant(g->blend_enable, g->blend_sfactor,
                                      g->blend_dfactor, g->blend_color, bf, &(int){0});
            [s_enc setBlendColorRed:bf[0] green:bf[1] blue:bf[2] alpha:bf[3]];
        }
        [s_enc setVertexBuffer:s_vb offset:vb_off atIndex:0];
        if (d->program) {
            /* c[192] is 3 KB: under setVertexBytes' 4 KB. */
            [s_enc setVertexBytes:v->c length:sizeof v->c atIndex:1];
            [s_enc setVertexBytes:&vc length:sizeof vc atIndex:2];
        } else {
            [s_enc setVertexBytes:&vc length:sizeof vc atIndex:1];
        }
        [s_enc setFragmentBytes:&mpc length:sizeof mpc atIndex:0];
        for (k = 0; k < 4; k++) {
            [s_enc setFragmentTexture:stex[k] atIndex:k];
            [s_enc setFragmentSamplerState:sampler(&s_stage[k], sscale[k]) atIndex:k];
        }
        [s_enc drawIndexedPrimitives:ptype indexCount:nidx indexType:MTLIndexTypeUInt32
                         indexBuffer:s_ib indexBufferOffset:ib_off];
        s_draws++;
        rt->gen = ++s_rt_seq;
        if (s_mf_on > 0 && s_mf_n && s_mf[s_mf_n - 1].kind != 'X') {
            s_mf[s_mf_n - 1].draws++;
            s_mf[s_mf_n - 1].idx += (uint32_t)nidx;
        }
    }
}

/* ---- flip ---------------------------------------------------------------------- */

/* The render target for the surface the walker picked: its offset, and at
 * least its size when the walker knows it (a grown target covers smaller
 * frames, rt_grow). NULL when this backend never drew there; the guest's own
 * bytes are presented then. */
static RenderTarget *present_target(uint32_t surface_offset)
{
    const struct nv2a_pb_present *p = nv2a_pb_present_state();
    int i;

    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && surface_offset && rt->offset == surface_offset
                && (!p->offset || (rt->w >= p->w && rt->h >= p->h))) {
            rt->presented = 1;
            return rt;
        }
    }
    return NULL;
}

/* The part of rt a flip shows, in guest pixels: the walker's present extent
 * when it names this surface (D3D11's present_extent). A target grown for a
 * larger clip (rt_grow) can be bigger than the frame; the rest of it is not
 * the frame. */
static void present_extent(const RenderTarget *rt, uint32_t *w, uint32_t *h)
{
    const struct nv2a_pb_present *p = nv2a_pb_present_state();
    *w = rt->w;
    *h = rt->h;
    if (p->offset && p->offset == rt->offset && p->w && p->h) {
        if (p->w < *w) *w = p->w;
        if (p->h < *h) *h = p->h;
    }
}

static uint32_t s_shown_w, s_shown_h;   /* the last flip's present extent */

/* rt's pixels into the guest surface at `addr`, `pitch` bytes a row: 32-bit
 * as they are (BGRA8 is X8R8G8B8's byte order), 16-bit as R5G6B5. Runs after
 * the wait, on the executor's thread. */
/* lin (w x h, 32-bit linear) into guest memory at dst as rt's swizzled
 * layout, at bpp 4 or (R5G6B5) 2. */
static void swz_store(const RenderTarget *rt, uint8_t *dst, uint32_t *lin, uint32_t bpp)
{
    uint32_t w = rt->w, h = rt->h, i;
    if (bpp == 2) {
        uint16_t *o = (uint16_t *)lin;      /* in place: each texel shrinks */
        for (i = 0; i < w * h; i++) {
            uint32_t c = lin[i];
            o[i] = (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0) | ((c >> 3) & 0x001F));
        }
    }
    xbox_swizzle_rect(dst, lin, w, h, bpp);
}

static void write_back(RenderTarget *rt, uint32_t addr, uint32_t pitch)
{
    uint8_t *dst = (uint8_t *)xbox_GetMemoryOffset() + addr;
    uint32_t bpp = pitch / rt->w, w = rt->w, y, x;

    if (rt_swz_ok(rt->swz, rt->w, rt->h) && (bpp == 4 || bpp == 2)) {
        uint32_t *lin = (uint32_t *)swz_staging((size_t)w * rt->h * 4u);
        if (!lin)
            return;
        rt_get_bytes(rt->tex, lin, w * 4u, MTLRegionMake2D(0, 0, w, rt->h));
        swz_store(rt, dst, lin, bpp);
        s_writebacks++;
        return;
    }

    if (bpp >= 4) {
        rt_get_bytes(rt->tex, dst, pitch, MTLRegionMake2D(0, 0, w, rt->h));
    } else if (bpp == 2) {
        static uint32_t *row;
        static uint32_t row_cap;
        if (row_cap < w) {
            uint32_t *n = (uint32_t *)realloc(row, (size_t)w * 4);
            if (!n)
                return;
            row = n;
            row_cap = w;
        }
        for (y = 0; y < rt->h; y++) {
            uint16_t *o = (uint16_t *)(dst + (size_t)y * pitch);
            rt_get_bytes(rt->tex, row, w * 4, MTLRegionMake2D(0, y, w, 1));
            for (x = 0; x < w; x++) {
                uint32_t c = row[x];
                o[x] = (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0)
                                  | ((c >> 3) & 0x001F));
            }
        }
    } else {
        return;
    }
    s_writebacks++;
}

/* A scaled rt (render.scale > 1) read back once at the flip, hw x hh BGRA8
 * tightly packed: the window shows it as it is and the write-back boxes it
 * down to guest size, so one getBytes serves both. NULL on no memory. */
static const uint8_t *host_readback(RenderTarget *rt)
{
    static uint8_t *px;
    static size_t cap;
    size_t need = (size_t)rt->hw * rt->hh * 4u;
    if (cap < need) {
        uint8_t *n = (uint8_t *)realloc(px, need);
        if (!n)
            return NULL;
        px = n;
        cap = need;
    }
    rt_get_bytes(rt->tex, px, rt->hw * 4u, MTLRegionMake2D(0, 0, rt->hw, rt->hh));
    return px;
}

/* write_back for a scaled rt: the host pixels (host_readback) averaged over
 * each n x n block, so the guest -- RECOMP_FB_DUMP, fb_dump_at, the title --
 * sees a w x h frame at its pitch, 32-bit or R5G6B5 as write_back writes. */
static void write_back_scaled(RenderTarget *rt, const uint8_t *px, uint32_t addr,
                              uint32_t pitch)
{
    uint8_t *dst = (uint8_t *)xbox_GetMemoryOffset() + addr;
    uint32_t bpp = pitch / rt->w, w = rt->w, n = rt->hw / rt->w, y, x;

    if (rt_swz_ok(rt->swz, rt->w, rt->h) && (bpp == 4 || bpp == 2)) {
        uint32_t *lin = (uint32_t *)swz_staging((size_t)w * rt->h * 4u);
        if (!lin)
            return;
        nv2a_downscale_box32(px, rt->hw * 4u, w, rt->h, n, (uint8_t *)lin, w * 4u);
        swz_store(rt, dst, lin, bpp);
        s_writebacks++;
        return;
    }

    if (bpp >= 4) {
        nv2a_downscale_box32(px, rt->hw * 4u, w, rt->h, n, dst, pitch);
    } else if (bpp == 2) {
        static uint32_t *row;
        static uint32_t row_cap;
        if (row_cap < w) {
            uint32_t *nr = (uint32_t *)realloc(row, (size_t)w * 4);
            if (!nr)
                return;
            row = nr;
            row_cap = w;
        }
        for (y = 0; y < rt->h; y++) {
            uint16_t *o = (uint16_t *)(dst + (size_t)y * pitch);
            nv2a_downscale_box32(px + (size_t)y * n * rt->hw * 4u, rt->hw * 4u, w, 1, n,
                                 (uint8_t *)row, w * 4u);
            for (x = 0; x < w; x++) {
                uint32_t c = row[x];
                o[x] = (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0)
                                  | ((c >> 3) & 0x001F));
            }
        }
    } else {
        return;
    }
    s_writebacks++;
}

/* ---- write-back on demand ---------------------------------------------------
 *
 * Guest memory under a render target is written only when something is about
 * to read it: the executor's dumps (sync_guest), a texture decode over it
 * (tex_bind), the eviction of a presented target (rt_free). The title itself
 * was never seen reading its framebuffer (metal-zero-copy-present, design),
 * and D3D11 never writes back at all. metal_writeback=always, or
 * metal_no_rtt, writes the present surface at every flip as before. */
static int s_guard_always;   /* metal_fb_guard saw a title read */

static int wb_always(void)
{
    const char *e = recomp_env(RENV_METAL_WRITEBACK);
    return (e && !strcasecmp(e, "always")) || recomp_env(RENV_METAL_NO_RTT) != NULL
           || s_guard_always;
}

/* rt into guest memory now, at `pitch`, if drawn since the last write-back.
 * Waits for the GPU; the ring positions survive the wait, since a draw may
 * have placed its vertices already (tex_bind runs inside on_draw). */
/* Managed storage (a discrete GPU): queue the copy back of the GPU's
 * contents, which the next wait completes. Nothing on unified memory. */
static void rt_managed_sync(RenderTarget *rt)
{
    id<MTLBlitCommandEncoder> b;

    if (s_unified)
        return;
    end_encoder();
    ensure_cmd();
    b = [s_cmd blitCommandEncoder];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [b synchronizeResource:rt->tex];
#pragma clang diagnostic pop
    [b endEncoding];
}

static void rt_sync(RenderTarget *rt, uint32_t pitch)
{
    NSUInteger vb = s_vb_pos, ib = s_ib_pos;
    const uint8_t *px;

    if (!rt->tex || !rt->dirty || !pitch || pitch < rt->w * 2u)
        return;
    guard_open();
    @autoreleasepool {
        rt_managed_sync(rt);
        if (s_cmd)
            flush(1);
        else if (s_last_cmd && [s_last_cmd status] < MTLCommandBufferStatusCompleted)
            [s_last_cmd waitUntilCompleted];
        s_vb_pos = vb;
        s_ib_pos = ib;
        if (rt->hw == rt->w)
            write_back(rt, rt->addr, pitch);
        else if ((px = host_readback(rt)) != NULL)
            write_back_scaled(rt, px, rt->addr, pitch);
        rt->dirty = 0;
        rt_own_reset(rt);
    }
}

/* nv2a_pb_backend.sync_guest: the target present_target would pick for this
 * surface (offset, covering guest w x h), else the most recently used one at its
 * address (a surface dump of a target drawn at another clip). */
static void metal_sync_guest(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h)
{
    RenderTarget *rt = NULL;
    uint32_t addr = nv2a_pb_dma_resolve(offset) & 0x07FFFFFFu;
    int i;

    guard_open();
    if (s_failed)
        return;
    for (i = 0; i < RT_MAX && !rt; i++)
        if (s_rt[i].tex && s_rt[i].offset == offset && s_rt[i].w >= w && s_rt[i].h >= h)
            rt = &s_rt[i];
    if (!rt) {
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *c = &s_rt[i];
            if (c->tex && (c->addr & 0x07FFFFFFu) == addr
                    && (!rt || c->last_use > rt->last_use))
                rt = c;
        }
    }
    if (rt && !rt_drop_if_stale(rt, "sync"))
        rt_sync(rt, pitch);
}

/* For tests/nv2a_backend_smoke: write-backs so far, and the host pixels
 * (hw x hh BGRA8, tightly packed) of the target at a colour offset, read
 * back now, so a test can box-filter them itself. */
uint64_t nv2a_pb_metal_writebacks(void) { return s_writebacks; }

/* For tests/nv2a_backend_smoke: the size of the target at a colour offset
 * and the part of it the last flip showed, 0 when there is none (the shape
 * of nv2a_pb_d3d11_shown). */
int nv2a_pb_metal_shown(uint32_t offset, uint32_t *tw, uint32_t *th,
                        uint32_t *sw, uint32_t *sh)
{
    int i;
    for (i = 0; i < RT_MAX; i++)
        if (s_rt[i].tex && s_rt[i].offset == offset) {
            *tw = s_rt[i].w;
            *th = s_rt[i].h;
            *sw = s_shown_w;
            *sh = s_shown_h;
            return 1;
        }
    return 0;
}
const uint8_t *nv2a_pb_metal_host_pixels(uint32_t offset, uint32_t *hw, uint32_t *hh)
{
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex && rt->offset == offset) {
            rt_managed_sync(rt);
            if (s_cmd)
                flush(1);
            *hw = rt->hw;
            *hh = rt->hh;
            return host_readback(rt);
        }
    }
    return NULL;
}

/* ---- metal_fb_guard ----------------------------------------------------------
 *
 * After each lazy flip the present surface's pages are PROT_NONE; the first
 * access from any thread is recorded by the handler, which opens the pages
 * again and lets it proceed. A title READ then switches the run to
 * metal_writeback=always (that read saw the last write-back, or the seed); a
 * write needs nothing from the GPU (the allocator's bzero on re-allocation is
 * one) and is logged once. Debug only: PROT_NONE turns a kernel-side access to
 * those pages (read() into re-allocated memory) into EFAULT. */
static struct {
    uintptr_t plo, phi;          /* the page span made PROT_NONE */
    uint32_t addr;
    volatile int armed, hit;
    uintptr_t far, pc;
    uint32_t write;
    char th[24];
    struct sigaction prev_bus, prev_segv;
} s_guard;

static void guard_handler(int sig, siginfo_t *si, void *ucv)
{
    uintptr_t f = (uintptr_t)si->si_addr;
    struct sigaction *prev;

    if (s_guard.armed && f >= s_guard.plo && f < s_guard.phi) {
        uint32_t esr = 0;
        uintptr_t pc = 0;
#if defined(__APPLE__) && defined(__aarch64__)
        {
            ucontext_t *uc = (ucontext_t *)ucv;
            esr = uc->uc_mcontext->__es.__esr;
            pc = (uintptr_t)uc->uc_mcontext->__ss.__pc;
        }
#endif
        if (!s_guard.hit) {
            s_guard.far = f;
            s_guard.pc = pc;
            s_guard.write = (esr >> 6) & 1u;   /* ESR WnR */
            s_guard.th[0] = 0;
            pthread_getname_np(pthread_self(), s_guard.th, sizeof s_guard.th);
            s_guard.hit = 1;
        }
        mprotect((void *)s_guard.plo, s_guard.phi - s_guard.plo, PROT_READ | PROT_WRITE);
        s_guard.armed = 0;
        return;   /* re-executes the access */
    }
    prev = sig == SIGBUS ? &s_guard.prev_bus : &s_guard.prev_segv;
    if (prev->sa_flags & SA_SIGINFO) {
        if (prev->sa_sigaction) {
            prev->sa_sigaction(sig, si, ucv);
            return;
        }
    } else if (prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
        prev->sa_handler(sig);
        return;
    }
    signal(sig, SIG_DFL);
}

/* Open the pages before this thread reads or writes them itself. */
static void guard_open(void)
{
    if (s_guard.armed) {
        s_guard.armed = 0;
        mprotect((void *)s_guard.plo, s_guard.phi - s_guard.plo, PROT_READ | PROT_WRITE);
    }
}

/* At the flip: report what the last span saw, then close the new one. */
static void guard_flip(uint32_t addr, uint32_t len)
{
    static int on = -1, said_write;
    uintptr_t pg = (uintptr_t)getpagesize(), lo;

    if (on < 0) {
        on = recomp_env(RENV_METAL_FB_GUARD) != NULL;
        if (on) {
            /* Installed now, after the title's own fault handlers, and
             * chained to them for every fault outside the span. */
            struct sigaction sa;
            memset(&sa, 0, sizeof sa);
            sa.sa_sigaction = guard_handler;
            sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
            sigemptyset(&sa.sa_mask);
            sigaction(SIGBUS, &sa, &s_guard.prev_bus);
            sigaction(SIGSEGV, &sa, &s_guard.prev_segv);
            fprintf(stderr, LOGP "fb_guard: armed; a title read of the present surface"
                    " switches to metal_writeback=always\n");
        }
    }
    if (!on)
        return;
    guard_open();
    if (s_guard.hit) {
        Dl_info di;
        const char *sym = dladdr((void *)s_guard.pc, &di) && di.dli_sname ? di.dli_sname : "?";
        unsigned long off = sym[0] != '?' ? (unsigned long)(s_guard.pc - (uintptr_t)di.dli_saddr)
                                          : (unsigned long)s_guard.pc;
        uint32_t ga = (uint32_t)(s_guard.far - (uintptr_t)xbox_GetMemoryOffset());
        if (!s_guard.write) {
            fprintf(stderr, LOGP "fb_guard: guest READ of present surface 0x%08X (at 0x%08X)"
                    " by '%s' pc %s+0x%lX; switching to metal_writeback=always\n",
                    s_guard.addr, ga, s_guard.th, sym, off);
            s_guard_always = 1;
        } else if (!said_write++) {
            fprintf(stderr, LOGP "fb_guard: guest WRITE of present surface 0x%08X (at 0x%08X)"
                    " by '%s' pc %s+0x%lX; ignored\n", s_guard.addr, ga, s_guard.th, sym, off);
        }
        s_guard.hit = 0;
    }
    if (s_guard_always || !addr || !len)
        return;
    lo = (uintptr_t)xbox_GetMemoryOffset() + addr;
    s_guard.addr = addr;
    s_guard.plo = lo & ~(pg - 1);
    s_guard.phi = (lo + len + pg - 1) & ~(pg - 1);
    __atomic_store_n(&s_guard.armed, 1, __ATOMIC_SEQ_CST);
    if (mprotect((void *)s_guard.plo, s_guard.phi - s_guard.plo, PROT_NONE) != 0)
        s_guard.armed = 0;
}

/* ---- Visibility tests: Metal visibility-result counting ---------------------
 *
 * The shared tracker (nv2a_occ, nv2a_backend_common.h) as the D3D11 backend
 * drives it: BeginVisibilityTest is CLEAR_REPORT_VALUE +
 * SET_ZPASS_PIXEL_COUNT_ENABLE(1), EndVisibilityTest is ENABLE(0) +
 * GET_REPORT; each stretch with counting on is one query, and a report sums
 * the stretches since the last clear, completing later (poll at every batch
 * and flip, with a flush on the ack thread's idle tick) or as visible after
 * 250 ms. The count is samples passing depth and stencil, alpha-test discard
 * applied.
 *
 * A query here is a list of slots in s_vis, one per render pass it spans:
 * Metal resets, at the start of a pass, every offset that pass names (even
 * in setVisibilityResultMode:Disabled), so a slot never outlives its pass
 * and a stretch over a pass change takes a new one. Slots accumulate within
 * the pass. A result is in once every slot's command buffer has completed,
 * so GET_REPORT commits the open command buffer: a count held to the flip
 * lands a frame later than D3D11's, and light shafts whose brightness
 * comes from the count come out too bright. A slot freed while
 * its command buffer is still in flight retires until the flip's wait.
 *
 * RECOMP_METAL_OCC=sync waits at GET_REPORT; =fixed leaves reports to the
 * walker (RECOMP_ZPASS_FIXED, every test visible). */
#define OCC_QSLOTS 16
#define OCC_POOL   (NV2A_OCC_PENDING * 2 + NV2A_OCC_SEG)
typedef struct OccQuery {
    int                   slot[OCC_QSLOTS];
    id<MTLCommandBuffer>  cmd[OCC_QSLOTS];   /* retained */
    int                   n, ovf;
    struct OccQuery      *next;              /* free list */
} OccQuery;
static OccQuery          s_qpool[OCC_POOL];
static OccQuery         *s_qfree;
static int               s_qmade;
static unsigned char     s_vis_used[VIS_SLOTS];
static int               s_vis_next;
/* Slots of queries freed while their command buffer was in flight: free
 * once it completes (occ_retire_poll, at each frame advance) or at a full
 * wait (occ_after_wait). */
static struct { int slot; id<MTLCommandBuffer> cmd; } s_retire[VIS_SLOTS];
static int               s_nretire;
static struct nv2a_occ   s_occ;
static int               s_occ_mode = -1;  /* 0 defer, 1 sync, 2 fixed */
static uint64_t          s_occ_ovf;

static int occ_mode(void)
{
    if (s_occ_mode < 0) {
        const char *e = recomp_env(RENV_METAL_OCC);
        s_occ_mode = !e ? 0 : !strcasecmp(e, "sync") ? 1 : !strcasecmp(e, "fixed") ? 2 : 0;
    }
    return s_occ_mode;
}

static int vis_alloc(void)
{
    int i, k;
    for (k = 0; k < VIS_SLOTS; k++) {
        i = (s_vis_next + k) % VIS_SLOTS;
        if (!s_vis_used[i]) {
            s_vis_used[i] = 1;
            s_vis_next = (i + 1) % VIS_SLOTS;
            ((uint64_t *)[s_vis contents])[i] = 0;
            return i;
        }
    }
    return -1;
}

static void occ_pass_begin(void)
{
    OccQuery *q = s_occ_q;
    int i;
    if (!q || !s_enc || s_occ_slot >= 0)
        return;
    if (q->n >= OCC_QSLOTS || (i = vis_alloc()) < 0) {
        q->ovf = 1;
        s_occ_ovf++;
        return;
    }
    q->slot[q->n] = i;
    q->cmd[q->n] = [s_cmd retain];
    q->n++;
    [s_enc setVisibilityResultMode:MTLVisibilityResultModeCounting offset:(NSUInteger)i * 8u];
    s_occ_slot = i;
}

static void occ_pass_end(void)
{
    if (s_occ_slot >= 0 && s_enc)
        [s_enc setVisibilityResultMode:MTLVisibilityResultModeDisabled
                                offset:(NSUInteger)s_occ_slot * 8u];
    s_occ_slot = -1;
}

static uint64_t occ_now_us(void *ctx)
{
    struct timespec ts;
    (void)ctx;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static void *occ_get(void *ctx)
{
    OccQuery *q;
    (void)ctx;
    if (s_qfree) {
        q = s_qfree;
        s_qfree = q->next;
    } else if (s_qmade < OCC_POOL) {
        q = &s_qpool[s_qmade++];
    } else {
        return NULL;
    }
    q->n = 0;
    q->ovf = 0;
    return q;
}

static int cmd_done(id<MTLCommandBuffer> c)
{
    MTLCommandBufferStatus st = [c status];
    return st == MTLCommandBufferStatusCompleted || st == MTLCommandBufferStatusError;
}

static void occ_put(void *ctx, void *qv)
{
    OccQuery *q = (OccQuery *)qv;
    int i;
    (void)ctx;
    if (!q)
        return;
    if (q == s_occ_q) {     /* not expected: the tracker ends a query first */
        occ_pass_end();
        s_occ_q = NULL;
    }
    for (i = 0; i < q->n; i++) {
        if (cmd_done(q->cmd[i]))
            s_vis_used[q->slot[i]] = 0;
        else {
            s_retire[s_nretire].slot = q->slot[i];
            s_retire[s_nretire].cmd = [q->cmd[i] retain];
            s_nretire++;
        }
        [q->cmd[i] release];
    }
    q->n = 0;
    q->next = s_qfree;
    s_qfree = q;
}

static void occ_begin(void *ctx, void *qv)
{
    (void)ctx;
    occ_pass_end();
    s_occ_q = (OccQuery *)qv;
    occ_pass_begin();
}

static void occ_end(void *ctx, void *qv)
{
    (void)ctx;
    if (s_occ_q == (OccQuery *)qv) {
        occ_pass_end();
        s_occ_q = NULL;
    }
}

static int occ_result(void *ctx, void *qv, int flush_gpu, uint64_t *count)
{
    OccQuery *q = (OccQuery *)qv;
    const uint64_t *v = (const uint64_t *)[s_vis contents];
    uint64_t sum = 0;
    int i;
    (void)ctx;
    for (i = 0; i < q->n; i++) {
        if (q->cmd[i] == s_cmd) {       /* not committed yet */
            if (!flush_gpu)
                return 0;
            flush(0);
        }
        if (!cmd_done(q->cmd[i]))
            return 0;
    }
    for (i = 0; i < q->n; i++)
        sum += v[q->slot[i]];
    *count = q->ovf ? NV2A_OCC_VISIBLE : sum;
    return 1;
}

static const struct nv2a_occ_ops s_occ_ops = {
    occ_get, occ_put, occ_begin, occ_end, occ_result, occ_now_us, NULL
};

static int occ_ready_on(void)
{
    if (occ_mode() == 2 || !init())
        return 0;
    if (!s_occ.ops)
        nv2a_occ_init(&s_occ, &s_occ_ops, (uint8_t *)xbox_GetMemoryOffset());
    return 1;
}

static void metal_on_zpass(int enable)
{
    if (occ_ready_on()) {
        @autoreleasepool {
            nv2a_occ_zpass(&s_occ, enable);
        }
    }
}

static void metal_on_zpass_clear(void)
{
    if (occ_ready_on()) {
        @autoreleasepool {
            nv2a_occ_zpass_clear(&s_occ);
        }
    }
}

static uint64_t s_report_commits;    /* flushes at GET_REPORT */

/* Does the report just pushed at va count in the open command buffer? */
static int report_in_cmd(uint32_t va)
{
    const struct nv2a_occ_report *p;
    int k, i;
    if (!s_occ.npend || s_occ.pend[s_occ.npend - 1].va != va)
        return 0;
    p = &s_occ.pend[s_occ.npend - 1];
    for (k = 0; k < p->n; k++) {
        const OccQuery *q = (const OccQuery *)p->q[k];
        for (i = 0; i < q->n; i++)
            if (q->cmd[i] == s_cmd)
                return 1;
    }
    return 0;
}

static int metal_on_report(uint32_t va)
{
    if (!occ_ready_on())
        return 0;
    @autoreleasepool {
        nv2a_occ_report(&s_occ, va, occ_mode() == 1);
        /* Commit now, so the count can land within the frame, as a D3D11
         * query's does; held to the flip, it lands a frame late. Only when
         * the report counts in s_cmd: each commit ends the pass (a tile
         * store and load), and a report whose queries drew nothing, or
         * drew in a buffer already committed, gains nothing from it. */
        if (occ_mode() == 0 && s_cmd && report_in_cmd(va)) {
            s_report_commits++;
            flush(0);
        }
    }
    return 1;
}

static void metal_on_poll(void)
{
    if (s_occ.npend) {
        @autoreleasepool {
            nv2a_occ_poll(&s_occ, 1);
        }
    }
}

/* After a full wait every command buffer is done: retired slots free. */
static void occ_after_wait(void)
{
    while (s_nretire) {
        s_nretire--;
        s_vis_used[s_retire[s_nretire].slot] = 0;
        [s_retire[s_nretire].cmd release];
    }
    if (s_occ.ops)
        nv2a_occ_poll(&s_occ, 0);
}

/* At a frame advance: the retired slots whose command buffer is done, and
 * the reports that completed meanwhile. */
static void occ_retire_poll(void)
{
    int i, n = 0;
    for (i = 0; i < s_nretire; i++) {
        if (cmd_done(s_retire[i].cmd)) {
            s_vis_used[s_retire[i].slot] = 0;
            [s_retire[i].cmd release];
        } else {
            s_retire[n++] = s_retire[i];
        }
    }
    s_nretire = n;
    if (s_occ.ops)
        nv2a_occ_poll(&s_occ, 0);
}

static void flip_log(uint32_t surface_offset, RenderTarget *rt)
{
    static int on = -1;
    static uint64_t p_draws, p_prog, p_skip, p_cull, p_clears, p_tex, p_up,
                    p_chg, p_re, p_fl, p_occ, p_vis, p_late, p_super, p_sync, p_ovf;
    if (on < 0)
        on = recomp_env(RENV_FLIP_LOG) != NULL;
    if (!on)
        return;
    fprintf(stderr, "[METAL] flip %llu batches %llu prog %llu skipped %llu"
            " culled %llu clears %llu tex %llu uploads %llu changed %llu rehash %llu"
            " flushes %llu occ %llu (visible %llu, pending %d, late %llu, superseded %llu,"
            " sync %llu us, slot overflow %llu) present 0x%08X walker 0x%08X shown %ux%u\n",
            (unsigned long long)s_presents,
            (unsigned long long)(s_draws - p_draws),
            (unsigned long long)(s_draws_prog - p_prog),
            (unsigned long long)(s_draws_skipped - p_skip),
            (unsigned long long)(s_draws_culled - p_cull),
            (unsigned long long)(s_clears - p_clears),
            (unsigned long long)(s_draws_tex - p_tex),
            (unsigned long long)(s_tex_uploads - p_up),
            (unsigned long long)(s_tex_changed - p_chg),
            (unsigned long long)(s_tex_rehash - p_re),
            (unsigned long long)(s_flushes - p_fl),
            (unsigned long long)(s_occ.reports - p_occ),
            (unsigned long long)(s_occ.vis - p_vis), s_occ.npend,
            (unsigned long long)(s_occ.late - p_late),
            (unsigned long long)(s_occ.super - p_super),
            (unsigned long long)(s_occ.sync_us - p_sync),
            (unsigned long long)(s_occ_ovf - p_ovf),
            rt ? rt->offset : 0, surface_offset, rt ? s_shown_w : 0, rt ? s_shown_h : 0);
    p_occ = s_occ.reports; p_vis = s_occ.vis; p_late = s_occ.late;
    p_super = s_occ.super; p_sync = s_occ.sync_us; p_ovf = s_occ_ovf;
    p_draws = s_draws; p_prog = s_draws_prog; p_skip = s_draws_skipped;
    p_cull = s_draws_culled; p_clears = s_clears; p_tex = s_draws_tex;
    p_up = s_tex_uploads; p_chg = s_tex_changed; p_re = s_tex_rehash;
    p_fl = s_flushes;
}

static void prof_flip_end(uint64_t t_flip0)
{
    static uint64_t p_wb, p_ds;
    int i;
    if (!prof_on())
        return;
    {
        /* With frames in flight a command buffer may still run: it is
         * counted at a later flip, once complete. */
        int n = 0;
        for (i = 0; i < s_prof_ncmd; i++) {
            id<MTLCommandBuffer> c = s_prof_cmd[i];
            MTLCommandBufferStatus st = [c status];
            if (st < MTLCommandBufferStatusCompleted && s_prof_ncmd < PROF_CMDS) {
                s_prof_cmd[n++] = c;
                continue;
            }
            if (st == MTLCommandBufferStatusCompleted) {
                CFTimeInterval a = [c GPUStartTime], b = [c GPUEndTime];
                if (b > a)
                    s_pa.gpu += (uint64_t)((b - a) * 1e9);
            }
            s_pa.ncmd++;
            [c release];
        }
        s_prof_ncmd = n;
    }
    s_pa.flip += prof_now() - t_flip0;
    if (++s_pa.n == 300) {
        double n = (double)s_pa.n;
        fprintf(stderr, LOGP "prof %llu flips: on_flip %.3f ms = wait %.3f + hostrb %.3f"
                " + wb %.3f + winrb %.3f + winpush %.3f + blit %.3f (+rest); gpu %.3f ms"
                " over %.1f cmd buffers/flip; %llu write-backs, %llu decode syncs;"
                " rt ownership hashes %.3f ms/flip; textures: %.1f rehashes %.3f ms"
                " (%.0f KB), %.1f uploads %.3f ms per flip\n",
                (unsigned long long)s_presents,
                s_pa.flip / n / 1e6, s_pa.wait / n / 1e6, s_pa.hostrb / n / 1e6,
                s_pa.wb / n / 1e6, s_pa.winrb / n / 1e6, s_pa.winpush / n / 1e6,
                s_pa.blit / n / 1e6, s_pa.gpu / n / 1e6, s_pa.ncmd / n,
                (unsigned long long)(s_writebacks - p_wb),
                (unsigned long long)(s_decode_syncs - p_ds), s_pa.hash / n / 1e6,
                s_pa.texh_n / n, s_pa.texh / n / 1e6, s_pa.texh_bytes / n / 1024.0,
                s_pa.texup_n / n, s_pa.texup / n / 1e6);
        p_wb = s_writebacks;
        p_ds = s_decode_syncs;
        memset(&s_pa, 0, sizeof s_pa);
    }
}


/* RECOMP_TRACE=latency (BO-74): the G row, when the GPU has finished the
 * frame. A copy of the frame into one of a few shared buffers rides after
 * it on the queue; its completion handler (a Metal thread, nobody waits)
 * stamps the time, converts GPUEndTime and writes the frame's signature,
 * which tools/perf/latency.py diffs to find the frame that reacts to a pad
 * change. Only with the trace on. */
#define LAT_BUFS 4
static id<MTLBuffer> s_lat_buf[LAT_BUFS];
static int           s_lat_next;

/* GPUStartTime/GPUEndTime and presentedTime are host time in seconds on the
 * mach_absolute_time clock; latency rows are on xbox_HostNowNs. */
static uint64_t lat_host_s_to_ns(double s)
{
    static mach_timebase_info_data_t tb;
    uint64_t now_ns = xbox_HostNowNs(), mach_ns;
    if (!tb.denom)
        mach_timebase_info(&tb);
    mach_ns = mach_absolute_time() * tb.numer / tb.denom;
    if (s <= 0)
        return 0;
    return now_ns - mach_ns + (uint64_t)(s * 1e9);
}

static void lat_frame_done(id<MTLTexture> tex, uint32_t w, uint32_t h, uint32_t flip)
{
    NSUInteger need = (NSUInteger)w * h * 4;
    id<MTLBuffer> buf;
    id<MTLBlitCommandEncoder> b;
    id<MTLCommandBuffer> cb;

    if (!tex || !w || !h || [tex pixelFormat] != MTLPixelFormatBGRA8Unorm)
        return;
    buf = s_lat_buf[s_lat_next];
    if (!buf || [buf length] < need) {
        [buf release];
        buf = [s_dev newBufferWithLength:need options:MTLResourceStorageModeShared];
        s_lat_buf[s_lat_next] = buf;
        if (!buf)
            return;
    }
    s_lat_next = (s_lat_next + 1) % LAT_BUFS;
    cb = [s_queue commandBuffer];
    b = [cb blitCommandEncoder];
    [b copyFromTexture:tex sourceSlice:0 sourceLevel:0
          sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:buf destinationOffset:0 destinationBytesPerRow:(NSUInteger)w * 4
destinationBytesPerImage:need];
    [b endEncoding];
    [buf retain];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        uint64_t t = xbox_HostNowNs();
        char row[480];
        int n = snprintf(row, sizeof row, "%llu,",
                         (unsigned long long)lat_host_s_to_ns([c GPUEndTime]));
        xbox_LatSignature((const uint8_t *)[buf contents], w, h, w * 4, 4,
                          row + n, (int)sizeof row - n);
        xbox_LatRow('G', flip, t, row);
        [buf release];
    }];
    [cb commit];
}

/* FLIP_STALL reached: commit what the frame has recorded so the GPU draws
 * it while the executor holds for the vblank; on_flip then only adds the
 * slot blit (BO-74). Before this the frame's command buffer was committed
 * at on_flip, after the hold, and the GPU's 2-3 ms came after the vblank. */
static void metal_on_flip_ready(void)
{
    @autoreleasepool {
        if (s_cmd)
            flush(0);
    }
}

/* BO-110: the scene snapshot of s_hdr_pend (see s_hdr_scene). */
static void hdr_snapshot(void)
{
    RenderTarget *rt = s_hdr_pend;
    id<MTLBlitCommandEncoder> b;

    s_hdr_pend = NULL;
    if (!rt || !rt->tex)
        return;
    if (!s_hdr_scene || [s_hdr_scene width] != rt->hw || [s_hdr_scene height] != rt->hh
            || [s_hdr_scene pixelFormat] != [rt->tex pixelFormat]) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:[rt->tex pixelFormat]
                                         width:rt->hw height:rt->hh mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        [s_hdr_scene release];
        s_hdr_scene = [s_dev newTextureWithDescriptor:td];
        if (!s_hdr_scene)
            return;
    }
    end_encoder();
    ensure_cmd();
    b = [s_cmd blitCommandEncoder];
    [b copyFromTexture:rt->tex toTexture:s_hdr_scene];
    [b endEncoding];
    s_hdr_scene_of = rt->offset;
    s_hdr_scene_ok = 1;
    s_hdr_snaps++;
}

/* BO-110: RECOMP_DEBUG=hdr_texlog: what the title draws with which blend,
 * by texrep_dump's hash of stage 0's texture, once per combination. */
static void hdr_texlog(const struct nv2a_pb_gpu *g)
{
    static int on = -1;
    static uint64_t seen[4096];
    static int nseen;
    const Stage *t = &s_stage[0];
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint64_t h, k;
    int i;

    static uint64_t watch;
    if (on < 0) {
        const char *w = recomp_env(RENV_HDR_WATCH);
        on = recomp_env(RENV_HDR_TEXLOG) != NULL;
        watch = w ? strtoull(w, NULL, 16) : 0;
    }
    if ((!on && !watch) || !t->offset || !t->width)
        return;
    h = nv2a_texrep_hash(mem + t->offset, t->color, t->width, t->height, t->pitch);
    /* an additive draw of it, or (BO-203: the sun's disc) an alpha-tested one */
    if (watch && h == watch && g->blend_enable
            && (g->blend_dfactor == 0x1 || nv2a_pb_vsh_state()->atest_enable)
            && nv2a_pb_stream_tag() == NV2A_HDR_TAG_SCENE)
        s_hdr_watch_hit = 1;
    if (!on || nseen >= 4096)
        return;
    k = h ^ ((uint64_t)(g->blend_enable ? g->blend_sfactor : 0xFFFF) << 32)
          ^ ((uint64_t)(g->blend_enable ? g->blend_dfactor : 0xFFFF) << 16)
          ^ nv2a_pb_stream_tag();
    for (i = 0; i < nseen; i++)
        if (seen[i] == k)
            return;
    seen[nseen++] = k;
    {
        const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
        fprintf(stderr, "[HDRTEX] tex %016llx fmt 0x%02x %ux%u blend %d sf 0x%x df 0x%x tag %u flip %llu"
                " atest %d func 0x%x ref %u\n",
                (unsigned long long)h, t->color, t->width, t->height, g->blend_enable,
                g->blend_sfactor, g->blend_dfactor, nv2a_pb_stream_tag(),
                (unsigned long long)s_presents, v->atest_enable, v->atest_func, v->atest_ref);
    }
}

/* BO-110: RECOMP_DEBUG=hdr_dump=<prefix>: the presented frame of each
 * fb_dump_at flip, hcw x hch host pixels, as 32-bit float RGBA after a
 * 16-byte header ("HDR1", w, h, 1 when the targets were half float). */
static void hdr_dump(RenderTarget *rt, uint32_t w, uint32_t h)
{
    const char *prefix = recomp_env(RENV_HDR_DUMP);
    uint32_t flip = nv2a_pb_gpu_state()->flips, hdr[4], y, x;
    char path[1100];
    FILE *f;

    static uint32_t watch_last, watch_n;
    int watched = 0;
    /* and the flip after each (with hdr_image_ab=1: the other state, the same moment) */
    if (watch_n && flip == watch_last + 1u)
        watched = 1;
    if (s_hdr_watch_hit) {
        s_hdr_watch_hit = 0;
        if (watch_n < 60 && flip - watch_last >= 20) {
            watched = 1;
            watch_last = flip;
            watch_n++;
        }
    }
    if (!prefix || !*prefix || !w || !h || !(watched || nv2a_pb_dump_at_listed(flip)))
        return;
    rt_managed_sync(rt);
    if (s_cmd)
        flush(1);
    else if (s_last_cmd && [s_last_cmd status] < MTLCommandBufferStatusCompleted)
        [s_last_cmd waitUntilCompleted];
    snprintf(path, sizeof path, "%sflip_%05u.rgbaf", prefix, flip);
    f = fopen(path, "wb");
    if (!f)
        return;
    memcpy(&hdr[0], "HDR1", 4);
    hdr[1] = w;
    hdr[2] = h;
    hdr[3] = (uint32_t)tex_is_half(rt->tex);
    fwrite(hdr, 4, 4, f);
    if (tex_is_half(rt->tex)) {
        uint16_t *row = (uint16_t *)malloc((size_t)w * 8u);
        float *out = (float *)malloc((size_t)w * 16u);
        for (y = 0; row && out && y < h; y++) {
            [rt->tex getBytes:row bytesPerRow:w * 8u fromRegion:MTLRegionMake2D(0, y, w, 1)
                  mipmapLevel:0];
            for (x = 0; x < w * 4u; x++)
                out[x] = hdr_h2f(row[x]);
            fwrite(out, 16, w, f);
        }
        free(row);
        free(out);
    } else {
        uint8_t *row = (uint8_t *)malloc((size_t)w * 4u);
        float *out = (float *)malloc((size_t)w * 16u);
        for (y = 0; row && out && y < h; y++) {
            [rt->tex getBytes:row bytesPerRow:w * 4u fromRegion:MTLRegionMake2D(0, y, w, 1)
                  mipmapLevel:0];
            for (x = 0; x < w; x++) {
                out[x * 4 + 0] = row[x * 4 + 2] / 255.0f;
                out[x * 4 + 1] = row[x * 4 + 1] / 255.0f;
                out[x * 4 + 2] = row[x * 4 + 0] / 255.0f;
                out[x * 4 + 3] = row[x * 4 + 3] / 255.0f;
            }
            fwrite(out, 16, w, f);
        }
        free(row);
        free(out);
    }
    fclose(f);
    fprintf(stderr, LOGP "hdr_dump: flip %u %ux%u %s -> %s\n", flip, w, h,
            tex_is_half(rt->tex) ? "half" : "bgra8", path);
    /* The scene snapshot of the same frame, when there is one. */
    if (s_hdr_scene_ok && s_hdr_scene && s_hdr_scene_of == rt->offset
            && tex_is_half(s_hdr_scene) && [s_hdr_scene width] >= w && [s_hdr_scene height] >= h) {
        uint16_t *row = (uint16_t *)malloc((size_t)w * 8u);
        float *out = (float *)malloc((size_t)w * 16u);
        snprintf(path, sizeof path, "%sflip_%05u_scene.rgbaf", prefix, flip);
        f = fopen(path, "wb");
        if (f && row && out) {
            fwrite(hdr, 4, 4, f);
            for (y = 0; y < h; y++) {
                [s_hdr_scene getBytes:row bytesPerRow:w * 8u
                           fromRegion:MTLRegionMake2D(0, y, w, 1) mipmapLevel:0];
                for (x = 0; x < w * 4u; x++)
                    out[x] = hdr_h2f(row[x]);
                fwrite(out, 16, w, f);
            }
        }
        if (f)
            fclose(f);
        free(row);
        free(out);
    }
}

/* BO-110: at every flip the stream tag goes back to UI (a frame without the
 * title's scene marks -- menus, movies, loading -- is all paper white); with
 * hdr_ab the targets change format every n flips: every target is written
 * back (when presented) and made again in the other format from the next
 * draw on, as at a start. */
static int s_hdr_said_switch;   /* log the present after a switch */

static void hdr_flip(void)
{
    uint32_t flip = nv2a_pb_gpu_state()->flips;
    int want;
    int i, nkeep = 0;
    static RenderTarget keep[RT_MAX];

    nv2a_pb_stream_tag_set(0);
    s_hdr_scene_ok = 0;
    s_hdr_pend = NULL;
    if ((flip % 600u) == 0 && s_hdr)
        fprintf(stderr, LOGP "HDR: flip %u, %s; draws %llu UI-bound, %llu scene-bound, %llu snapshots\n",
                flip, s_hdr ? "on" : "off", (unsigned long long)s_hdr_draws[0],
                (unsigned long long)s_hdr_draws[1], (unsigned long long)s_hdr_snaps);
    want = s_hdr_ab ? ((flip / (uint32_t)s_hdr_ab) & 1u) == 0 : nv2a_host_hdr_active();
    nv2a_texrep_flip(flip);             /* BO-203, hdr_image_ab */
    if (want == s_hdr)
        return;
    @autoreleasepool {
        end_encoder();
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex)
                continue;
            if (rt->presented) {
                keep[nkeep] = *rt;
                keep[nkeep].tex = nil;
                keep[nkeep].opaque = nil;
                nkeep++;
            }
            rt_free(rt);
        }
        for (i = 0; i < CUBE_MAX; i++) {
            [s_cube[i].tex release];
            memset(&s_cube[i], 0, sizeof s_cube[i]);
        }
        [s_self.tex release];
        s_self.tex = nil;
    }
    s_hdr = want;
    nv2a_texrep_set_hdr(want);          /* BO-203: textures with an HDR picture upload again */
    memset(s_fs_memo, 0, sizeof s_fs_memo);
    for (i = 0; i < PSO_MEMOS; i++)
        s_pso_last[i] = -1;
    /* The targets a flip has shown come back at once, in the new format,
     * at their old size, from the bytes rt_free wrote back (the Vulkan
     * backend's hdr_flip: left to the next draw, one is made at that draw's
     * clip, 640x464 on Burnout 3's loading screens, which present_target
     * never takes for the 640x480 frame). */
    @autoreleasepool {
        for (i = 0; i < nkeep; i++) {
            RenderTarget *rt = &s_rt[i];
            if (rt_make(rt, keep[i].offset, keep[i].w, keep[i].h, keep[i].pitch, keep[i].swz))
                rt->presented = 1;
        }
    }
    s_hdr_said_switch = !s_hdr_ab || flip < 4000u;
    if (!s_hdr_ab || flip < 4000u)
        fprintf(stderr, LOGP "HDR%s: flip %u, HDR %s\n", s_hdr_ab ? " A/B" : "", flip, want ? "on" : "off");
}

static void metal_on_flip(uint32_t surface_offset, uint32_t pitch)
{
    RenderTarget *rt = NULL;
    uint32_t addr = surface_offset ? nv2a_pb_dma_resolve(surface_offset) : 0;
    const uint8_t *hpx = NULL;   /* a scaled rt's host pixels (host_readback) */
    uint64_t t_flip0 = prof_on() ? prof_now() : 0, t;
    int always = wb_always(), win = 0, layer = 0;
    uint32_t cw = 0, ch = 0, hcw = 0, hch = 0;   /* the frame, guest and host */

    guard_open();
    if (init()) {
        @autoreleasepool {
            int wb;
            id<MTLTexture> slot = nil;
            if (s_hdr && s_hdr_pend)
                hdr_snapshot();        /* a view with nothing drawn after it */
            rt = present_target(surface_offset);
            if (s_hdr_said_switch) {
                /* The first present after an HDR switch: on the GPU path still. */
                s_hdr_said_switch = 0;
                fprintf(stderr, LOGP "HDR: present after the switch: surface 0x%08X (%s)\n",
                        surface_offset, rt ? "drawn here" : "guest bytes");
            }
            if (rt) {
                present_extent(rt, &cw, &ch);
                hcw = (uint32_t)((uint64_t)cw * rt->hw / rt->w);
                hch = (uint32_t)((uint64_t)ch * rt->hh / rt->h);
                s_shown_w = cw;
                s_shown_h = ch;
            }
            win = rt && xbox_FramebufferWindowRunning();
            /* Layer mode: the frame goes to the window as a GPU copy into the
             * presenter's back slot (fb_present_metal.m), below. */
            layer = win && s_layer;
            wb = rt && addr && pitch && pitch >= rt->w * 2u && always;
            if (rt && (wb || (win && !layer)))
                rt_managed_sync(rt);
            /* The CPU reads this frame only for a write-back or the
             * readback window: then it waits, else the frame stays in
             * flight (frame_advance below). */
            t = prof_t();
            if (wb || (win && !layer))
                flush(1);
            else
                flush(0);
            prof_add(&s_pa.wait, t);
            if (layer) {
                /* After the frame's commit, in a command buffer of its own
                 * that nobody waits for: the queue runs it after the frame,
                 * and tracked hazards order it against a present pass still
                 * reading the slot. Published once committed, so the main
                 * thread's pass that shows it is committed after it. */
                t = prof_t();
                slot = (id<MTLTexture>)xbox_FramebufferWindowBackTexture(hcw, hch);
                if (slot && [slot pixelFormat] != [rt->tex pixelFormat]) {
                    static int said;   /* BO-110: the presenter's slots are another format */
                    if (!said++)
                        fprintf(stderr, LOGP "HDR: window slot format differs; frame not shown\n");
                    slot = nil;
                }
                if (slot) {
                    id<MTLBlitCommandEncoder> b;
                    ensure_cmd();
                    b = [s_cmd blitCommandEncoder];
                    [b copyFromTexture:rt->tex sourceSlice:0 sourceLevel:0
                          sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(hcw, hch, 1)
                             toTexture:slot destinationSlice:0 destinationLevel:0
                     destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [b endEncoding];
                    if (s_hdr) {
                        /* BO-110: the scene snapshot beside the frame. */
                        int ok = s_hdr_scene_ok && s_hdr_scene && s_hdr_scene_of == rt->offset
                                 && [s_hdr_scene width] >= hcw && [s_hdr_scene height] >= hch;
                        id<MTLTexture> ss = (id<MTLTexture>)xbox_FramebufferWindowBackScene(hcw, hch, ok);
                        if (ok && ss && [ss pixelFormat] == [s_hdr_scene pixelFormat]) {
                            b = [s_cmd blitCommandEncoder];
                            [b copyFromTexture:s_hdr_scene sourceSlice:0 sourceLevel:0
                                  sourceOrigin:MTLOriginMake(0, 0, 0)
                                    sourceSize:MTLSizeMake(hcw, hch, 1)
                                     toTexture:ss destinationSlice:0 destinationLevel:0
                             destinationOrigin:MTLOriginMake(0, 0, 0)];
                            [b endEncoding];
                        }
                    }
                    flush(0);
                    xbox_FramebufferWindowPublishTexture();
                    s_window_presents++;
                }
                prof_add(&s_pa.blit, t);
            }
            if (rt && xbox_LatOn()) {
                flush(0);
                lat_frame_done(rt->tex, hcw, hch, nv2a_pb_gpu_state()->flips);
            }
            /* The host-size readback only when something reads it: the
             * write-back below or the readback-mode window. */
            t = prof_t();
            if (rt && rt->hw != rt->w && (wb || (win && !layer)))
                hpx = host_readback(rt);
            prof_add(&s_pa.hostrb, t);
            if (wb) {
                t = prof_t();
                if (rt->hw == rt->w)
                    write_back(rt, addr, pitch);
                else if (hpx)
                    write_back_scaled(rt, hpx, addr, pitch);
                rt->dirty = 0;
                prof_add(&s_pa.wb, t);
                if (addr == rt->addr && pitch == rt->pitch)
                    rt_own_reset(rt);
            }
            if (rt && (s_hdr_req || s_hdr))
                hdr_dump(rt, hcw, hch);
            t = prof_t();
            frame_advance();
            prof_add(&s_pa.wait, t);
        }
    }
    hdr_flip();
    s_presents++;
    flip_log(surface_offset, rt);
    mf_flip();
    pc_flip(s_presents);
    if (s_presents == 1 || (s_presents % 300) == 0)
        fprintf(stderr, LOGP "present #%llu: surface 0x%08X (%s), %llu clears,"
                " %llu draws (%llu vertex-program), %llu skipped, %llu culled,"
                " %llu texture uploads, %llu rehashes, %llu write-backs (%s),"
                " %llu decode syncs, %llu stale drops, %llu grows, %llu partial clears;"
                " shaders %llu vs + %llu fs, %llu compile errors;"
                " %llu z-clears, %llu depth draws, %llu rt binds (%llu self copies);"
                " %llu window presents (%s); %llu flushes,"
                " %llu at GET_REPORT; %d frames in flight, %llu slot waits\n",
                (unsigned long long)s_presents, surface_offset,
                rt ? "drawn here" : "guest bytes",
                (unsigned long long)s_clears, (unsigned long long)s_draws,
                (unsigned long long)s_draws_prog, (unsigned long long)s_draws_skipped,
                (unsigned long long)s_draws_culled,
                (unsigned long long)s_tex_uploads, (unsigned long long)s_tex_rehash,
                (unsigned long long)s_writebacks, always ? "always" : "lazy",
                (unsigned long long)s_decode_syncs, (unsigned long long)s_stale_drops,
                (unsigned long long)s_grows, (unsigned long long)s_partial_clears,
                (unsigned long long)s_vs_compiled, (unsigned long long)s_fs_compiled,
                (unsigned long long)s_compile_errors,
                (unsigned long long)s_zclears, (unsigned long long)s_draws_z,
                (unsigned long long)s_rtt_binds, (unsigned long long)s_self_copies,
                (unsigned long long)s_window_presents,
                s_layer ? "CAMetalLayer slots" : "texture readback",
                (unsigned long long)s_flushes, (unsigned long long)s_report_commits,
                frames_in_flight(), (unsigned long long)s_frame_waits);
    if (s_presents == 1 || (s_presents % 300) == 0)
        pc_log(s_presents);
    if (!always && rt && addr && pitch)
        guard_flip(addr, pitch * rt->h);
    if (!pitch || !addr || layer) {
        prof_flip_end(t_flip0);
        return;
    }
    xbox_FramebufferWindowSet(addr, pitch);
    /* Readback mode (metal_present=readback, or no CAMetalLayer window): a
     * surface drawn here reaches the window by readback, getBytes of rt->tex
     * after the flip's wait, a copy into the window's slot, then
     * SDL_UpdateTexture. It reads the texture, not the guest bytes. A surface
     * the backend never drew (guest bytes) goes as the CPU path presents it,
     * in either mode. */
    if (rt && hpx && win) {
        /* Scaled: the frame's part of the host-size readback above. */
        t = prof_t();
        xbox_FramebufferWindowPresentPixels(hpx, hcw, hch, rt->hw * 4u, 4u);
        prof_add(&s_pa.winpush, t);
        s_window_presents++;
        prof_flip_end(t_flip0);
        return;
    }
    if (rt && rt->hw == rt->w && win) {
        static uint8_t *px;
        static size_t cap;
        size_t need = (size_t)cw * ch * 4u;
        if (cap < need) {
            uint8_t *n = (uint8_t *)realloc(px, need);
            if (n) {
                px = n;
                cap = need;
            }
        }
        if (cap >= need) {
            t = prof_t();
            rt_get_bytes(rt->tex, px, cw * 4u, MTLRegionMake2D(0, 0, cw, ch));
            prof_add(&s_pa.winrb, t);
            t = prof_t();
            xbox_FramebufferWindowPresentPixels(px, cw, ch, cw * 4u, 4u);
            prof_add(&s_pa.winpush, t);
            s_window_presents++;
            prof_flip_end(t_flip0);
            return;
        }
    }
    /* Guest bytes go to the window: our read, not the title's (the guard
     * may have armed this surface above). Headless, nothing reads them. */
    if (xbox_FramebufferWindowRunning())
        guard_open();
    xbox_FramebufferWindowPresent(addr, pitch);
    prof_flip_end(t_flip0);
}

/* ---- registration ---------------------------------------------------------- */

static const struct nv2a_pb_backend s_metal_backend = {
    metal_on_clear,
    metal_on_zclear,
    metal_on_draw,
    metal_on_flip,
    metal_on_zpass,
    metal_on_zpass_clear,
    metal_on_report,
    metal_on_poll,
    1,                  /* writes_back: dumps read guest memory, after sync_guest */
    metal_sync_guest,
    metal_on_flip_ready,
};

const struct nv2a_pb_backend *nv2a_pb_metal_backend(void)
{
    if (!s_black && init()) {
        @autoreleasepool {
            s_black = solid_texture(0xFF000000u);
            s_magenta = solid_texture(NV2A_TEX_MAGENTA);
            s_clear = solid_texture(0x00000000u);
        }
    }
    return &s_metal_backend;
}

/* RECOMP_PB_BACKEND=metal routes the executor here. Called once from the
 * executor's first method, on the ack thread. */
void nv2a_pb_metal_register_from_env(void)
{
    const char *b = recomp_env(RENV_PB_BACKEND);
    if (b && !strcasecmp(b, "metal")) {
        const struct nv2a_pb_backend *mb = nv2a_pb_metal_backend();
        if (s_failed) {
            fprintf(stderr, LOGP "RECOMP_PB_BACKEND=metal: no device, staying on"
                    " the CPU rasteriser\n");
            return;
        }
        nv2a_pb_set_backend(mb);
        fprintf(stderr, LOGP "RECOMP_PB_BACKEND=metal: executor draws through Metal\n");
    }
}

#endif /* __APPLE__ */
