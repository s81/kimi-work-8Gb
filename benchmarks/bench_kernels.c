/* bench_kernels.c - where does the arithmetic actually go?
 *
 * The engine's decode time at the memory floor splits roughly 36 s trunk read, 11 s
 * expert read, 10 s compute. The read paths are now within 10-20% of what the device can
 * deliver, so the only lossless win left is the compute. Before optimising it, measure
 * which kernel owns it -- guessing which loop is hot is how people spend a week making
 * something 2% faster.
 *
 * Dimensions here are the REAL ones, per token:
 *   bf16 trunk matmuls   the attention projections, latent down/up, shared experts and
 *                        the dense MLP. Sized from the actual layer shapes.
 *   MXFP4 expert matmuls 16 experts x 3 matrices x 92 layers, in latent space.
 *
 * Reports GFLOP/s and the projected per-token seconds for each, so the two can be
 * compared directly against the measured 10 s compute budget.
 *
 * TIMING. Each kernel runs REPS times after a warm-up and the BEST and MEDIAN single
 * runs are printed. Shared CI runners and VMs vary by tens of percent from run to run;
 * the best run is the most repeatable figure, the median shows how noisy the machine
 * was. Neither is an average, which one slow outlier would drag.
 *
 * The second half times the non-matmul kernels and the thread pool's own dispatch
 * cost, single calls at real width scaled by how often a token makes them, so a
 * platform difference (msvcrt maths, Windows locking) shows up as a number. CI runs
 * this on Linux and Windows on every push; it reports and never fails the build.
 */
#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"
#include "k3_par.h"

#define REPS 7

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Exact bits of a whole output vector.
 *
 * Building this file with and without AVX2 and comparing these hashes is the ONLY real
 * proof that a vector path is bit-identical rather than merely close. A tolerance check
 * would happily pass a kernel that quietly reassociated the reduction, which is exactly
 * the mistake worth catching: this engine's claim is that its output equals the
 * reference, and "equals" has to mean equals.
 *
 * This only means anything while the inputs are FINITE. NaN compares unequal to itself
 * and carries a payload that propagates differently through scalar and vector code, so
 * a NaN input turns this from a proof into a guaranteed false alarm. See fillbf16. */
static void fnv(const char *label, const float *v, int n)
{
    unsigned long long h = 1469598103934665603ull;
    for (int k = 0; k < n; k++) {
        union { float f; unsigned u; } b; b.f = v[k];
        for (int t = 0; t < 4; t++) { h ^= (b.u >> (8 * t)) & 0xFFu; h *= 1099511628211ull; }
    }
    printf("             %s OUTPUT FNV1a = %016llx\n", label, h);
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Best and median of REPS single timings, in seconds. */
static void best_median(double *t, double *best, double *med)
{
    qsort(t, REPS, sizeof *t, cmp_d);
    *best = t[0];
    *med = t[REPS / 2];
}

static void fillf(float *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}
static void fillb(unsigned char *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)(s >> 13);
    }
}

/* bf16 weights that are FINITE, which fillb cannot promise.
 *
 * Random bytes reinterpreted as bf16 hit the all-ones exponent about one time in 256,
 * so a 7168-term row is NaN with probability 1 - (255/256)^7168, i.e. essentially
 * always. Every output of the bf16 section then comes out NaN, and hashing NaN compares
 * PAYLOADS rather than arithmetic: those propagate differently through libm's fma() and
 * through _mm256_fmadd_pd, so the two builds disagree unconditionally and the check
 * described above reports a bit-identity failure that is not there.
 *
 * Same xorshift and the same distribution as fillf, truncated to the top half, which is
 * what bf16 storage is. No intermediate float array: at 12288 x 7168 that would be a
 * 352 MB temporary for values consumed once. */
static void fillbf16(uint16_t *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const float f = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
        unsigned u; memcpy(&u, &f, sizeof u);
        p[i] = (unsigned short)(u >> 16);
    }
}

/* One kernel call timed CALLS times per rep; prints per-call best/median and the
 * per-token cost on ONE thread (these kernels are called per head or per expert, and
 * the engine spreads the calls across the pool, so divide by the thread count for
 * wall clock). */
#define TIME_CALL(label, per_token, calls, stmt) do {                                  \
        double t_[REPS], b_, m_;                                                       \
        stmt;                                                                          \
        for (int r_ = 0; r_ < REPS; r_++) {                                            \
            const double s_ = now_s();                                                 \
            for (int c_ = 0; c_ < (calls); c_++) { stmt; }                             \
            t_[r_] = (now_s() - s_) / (calls);                                         \
        }                                                                              \
        best_median(t_, &b_, &m_);                                                     \
        printf("%-26s %9.2f us  (median %9.2f)  x %5d/token = %7.1f ms/token\n",       \
               label, b_ * 1e6, m_ * 1e6, (per_token), b_ * (per_token) * 1e3);        \
    } while (0)

static void pool_noop(void *ctx, int b, int e) { (void)ctx; (void)b; (void)e; }

/* The MXFP4 matmul on one thread and on the default count: how much the pool actually
 * buys on this machine. Thread wake-up cost differs a lot between platforms. */
static void pool_scaling(void)
{
    const int in = 3584, rows = 3072, group = K3_MXFP4_GROUP;
    const int pcols = in / 2, ngrp = in / group;
    unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
    unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
    float *x = (float *)malloc((size_t)in * sizeof(float));
    float *y = (float *)malloc((size_t)rows * sizeof(float));
    if (!pk || !sc || !x || !y) { printf("alloc failed\n"); return; }
    fillb(pk, (size_t)rows * pcols, 777u);
    memset(sc, 127, (size_t)rows * ngrp);
    fillf(x, in, 4242u);
    const int threads = k3_par_threads();
    double best[2];
    for (int pass = 0; pass < 2; pass++) {
        k3_par_set_threads(pass == 0 ? 1 : threads);
        double t[REPS], med;
        k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        for (int r = 0; r < REPS; r++) {
            const double t0 = now_s();
            k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
            t[r] = now_s() - t0;
        }
        best_median(t, &best[pass], &med);
    }
    k3_par_set_threads(threads);
    printf("\npool scaling, MXFP4 3072 x 3584: 1 thread %.2f ms, %d threads %.2f ms -> %.2fx\n",
           best[0] * 1e3, threads, best[1] * 1e3, best[0] / best[1]);
    free(pk); free(sc); free(x); free(y);
}

/* The non-matmul kernels at real width, single-threaded calls. Per-token counts:
 * 69 KDA layers x 96 heads for the step; about 17 SiTU-GLU calls (16 routed experts
 * plus the shared pair) per MoE layer; 2 rmsnorms per layer; one ShortConv over the
 * q, k and v channels and one decay per KDA layer. */
static void other_kernels(void)
{
    printf("\nother kernels, one call at real width (us), and per token on ONE thread:\n");
    const int H = 7168, dk = 128, dv = 128, NI = 3072, C = 3 * 96 * 128;
    float *S = (float *)malloc(sizeof(float) * dk * dv);
    float *o = (float *)malloc(sizeof(float) * dv), *q = (float *)malloc(sizeof(float) * dk);
    float *k = (float *)malloc(sizeof(float) * dk), *v = (float *)malloc(sizeof(float) * dv);
    float *al = (float *)malloc(sizeof(float) * dk);
    float *x = (float *)malloc(sizeof(float) * H), *y = (float *)malloc(sizeof(float) * H);
    float *w = (float *)malloc(sizeof(float) * H);
    float *gx = (float *)malloc(sizeof(float) * 2 * NI), *gy = (float *)malloc(sizeof(float) * NI);
    float *cx = (float *)malloc(sizeof(float) * C), *cy = (float *)malloc(sizeof(float) * C);
    float *cw = (float *)malloc(sizeof(float) * C * 4), *cs = (float *)malloc(sizeof(float) * C * 3);
    float *z = (float *)malloc(sizeof(float) * 96 * 128), *Al = (float *)malloc(sizeof(float) * 96);
    float *db = (float *)malloc(sizeof(float) * 96 * 128), *g = (float *)malloc(sizeof(float) * 96 * 128);
    float *a2 = (float *)malloc(sizeof(float) * 96 * 128);
    if (!S || !o || !q || !k || !v || !al || !x || !y || !w || !gx || !gy || !cx || !cy ||
        !cw || !cs || !z || !Al || !db || !g || !a2) { printf("alloc failed\n"); return; }
    fillf(S, (size_t)dk * dv, 11u); fillf(q, dk, 12u); fillf(k, dk, 13u); fillf(v, dv, 14u);
    for (int i = 0; i < dk; i++) al[i] = 0.99f;    /* decay < 1: S stays bounded */
    fillf(x, H, 15u); fillf(w, H, 16u);
    fillf(gx, 2 * NI, 17u);
    for (int i = 0; i < 2 * NI; i++) gx[i] *= 80.0f;   /* exercise tanh/exp range */
    fillf(cx, C, 18u); fillf(cw, (size_t)C * 4, 19u); memset(cs, 0, sizeof(float) * C * 3);
    fillf(z, 96 * 128, 20u); fillf(Al, 96, 21u); fillf(db, 96 * 128, 22u);

    TIME_CALL("KDA step 128x128", 69 * 96, 400, k3_kda_step(S, o, q, k, v, al, 0.5f, dk, dv));
    fnv("kda  ", o, dv);
    TIME_CALL("SiTU-GLU 3072", 17 * 92, 400, k3_situ_glu(gy, gx, NI, 4.0f, 25.0f));
    fnv("glu  ", gy, NI);
    TIME_CALL("ShortConv 36864 k4 T1", 69, 40, k3_shortconv(cy, cx, cw, cs, C, 4, 1));
    fnv("conv ", cy, C);
    TIME_CALL("KDA decay 96x128", 69, 100, k3_kda_decay(g, a2, z, Al, db, 96, 128, -5.0f));
    fnv("decay", a2, 96 * 128);
    TIME_CALL("rmsnorm 7168", 2 * 93, 400, k3_rmsnorm(y, x, w, H, 1e-5f));
    fnv("norm ", y, H);

    /* Thread pool dispatch with no work: what one parallel region costs in wake-up and
     * join. The engine opens several per layer. */
    printf("\nthread pool: %d threads\n", k3_par_threads());
    TIME_CALL("k3_par_for dispatch, empty", 0, 2000, k3_par_for(1024, 0, pool_noop, NULL));

    free(S); free(o); free(q); free(k); free(v); free(al); free(x); free(y); free(w);
    free(gx); free(gy); free(cx); free(cy); free(cw); free(cs); free(z); free(Al);
    free(db); free(g); free(a2);
}

int main(void)
{
    printf("kernel benchmark at REAL Kimi K3 dimensions\n");
#ifdef __AVX2__
    printf("built WITH AVX2\n\n");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    printf("built WITH NEON\n\n");
#else
    printf("built WITHOUT AVX2 or NEON (scalar)\n\n");
#endif

    /* ---------- bf16 trunk matmul: KDA q_proj, 7168 -> 12288 ---------- */
    {
        const int in = 7168, out = 12288;
        uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)out * sizeof(float));
        if (!W || !x || !y) { printf("alloc failed\n"); return 1; }
        fillbf16(W, (size_t)in * out, 12345u);
        fillf(x, in, 999u);

        k3_matmul_bf16(y, x, W, in, out);              /* warm */
        double t[REPS], dt, med;
        for (int r = 0; r < REPS; r++) {
            const double t0 = now_s();
            k3_matmul_bf16(y, x, W, in, out);
            t[r] = now_s() - t0;
        }
        best_median(t, &dt, &med);
        const double gflop = 2.0 * in * out / 1e9;
        printf("bf16 matmul  %5d x %-5d  %7.2f ms  %8.1f GFLOP/s  (median %.2f ms)\n",
               out, in, dt * 1e3, gflop / dt, med * 1e3);

        /* Per token the trunk is 56.74 G always-active params = 113.49 GFLOP of
         * multiply-add. Project from the rate just measured. */
        fnv("bf16 ", y, out);
        printf("             trunk is 56.74 G params/token -> %.2f s/token at this rate\n",
               2.0 * 56.74e9 / 1e9 / (gflop / dt));
        free(W); free(x); free(y);
    }

    /* ---------- MXFP4 expert matmul: w1, latent 3584 -> inter 3072 ---------- */
    {
        const int in = 3584, rows = 3072, group = K3_MXFP4_GROUP;
        const int pcols = in / 2, ngrp = in / group;
        unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
        unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)rows * sizeof(float));
        if (!pk || !sc || !x || !y) { printf("alloc failed\n"); return 1; }
        fillb(pk, (size_t)rows * pcols, 777u);
        memset(sc, 127, (size_t)rows * ngrp);          /* scale 2^0, none skipped */
        fillf(x, in, 4242u);

        k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        double t[REPS], dt, med;
        for (int r = 0; r < REPS; r++) {
            const double t0 = now_s();
            k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
            t[r] = now_s() - t0;
        }
        best_median(t, &dt, &med);
        const double gflop = 2.0 * in * rows / 1e9;
        printf("\nMXFP4 matmul %5d x %-5d  %7.2f ms  %8.1f GFLOP/s  (median %.2f ms)\n",
               rows, in, dt * 1e3, gflop / dt, med * 1e3);

        /* One expert is w1 + w3 (both 3072x3584) + w2 (3584x3072) = 3 of these.
         * 16 experts x 92 MoE layers per token. */
        fnv("mxfp4", y, rows);
        const double per_tok = dt * 3.0 * 16 * 92;
        printf("             16 experts x 3 mats x 92 layers -> %.2f s/token\n", per_tok);
        free(pk); free(sc); free(x); free(y);
    }

    printf("\nmeasured compute budget at the floor is about 10 s/token; whichever line\n"
           "above dominates it is the one worth vectorising.\n");

    pool_scaling();
    other_kernels();
    return 0;
}
