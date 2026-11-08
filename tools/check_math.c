/* check_math.c - check k3_expf and k3_tanhf on all 2^32 float inputs.
 *
 * usage (Linux x86-64, a few minutes on 4 cores):
 *   tcc -I src/core -run tools/check_math.c [threads] > close.txt
 *   python3 tools/check_math_exact.py close.txt
 *
 * Reference: glibc's long double expl and tanhl (64-bit significand, a few units in
 * the last place). For each input the reference is widened by a relative margin of
 * 2^-56, far above expl's and tanhl's error. If both ends of that interval round to
 * the same float, that float is the correctly rounded result, and k3's answer must
 * equal it bit for bit: a difference is a FAIL. If the interval straddles a rounding
 * boundary the case is "close": it is printed, with k3's answer, for
 * check_math_exact.py to settle at 300 bits.
 *
 * It also counts how often glibc's own expf and tanhf differ from k3's, which is how
 * often v1.2.3 and earlier Linux builds gave a different float.
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_math.h"

static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

typedef struct {
    uint64_t lo, hi;
    uint64_t fail[2], close[2], libm_diff[2];
    char *log; size_t len, cap;
} Job;

static void logf_(Job *j, const char *fn, uint32_t x, uint32_t got)
{
    if (j->len + 64 > j->cap) { j->cap = j->cap ? 2 * j->cap : 1 << 16; j->log = realloc(j->log, j->cap); }
    j->len += (size_t)sprintf(j->log + j->len, "%s %08x %08x\n", fn, x, got);
}

static int check_one(Job *j, int which, uint32_t ux)
{
    const float x = f_of(ux);
    const float got = which ? k3_tanhf(x) : k3_expf(x);
    const float lib = which ? tanhf(x) : expf(x);
    if (x != x) {
        if (got == got) { j->fail[which]++; return 1; }
        return 0;
    }
    if (u_of(got) != u_of(lib)) j->libm_diff[which]++;
    const long double v = which ? tanhl((long double)x) : expl((long double)x);
    const long double m = 1.0L / 72057594037927936.0L;          /* 2^-56 */
    float a = (float)(v * (1.0L - m)), b = (float)(v * (1.0L + m));
    if (u_of(a) == u_of(b)) {
        if (u_of(got) != u_of(a)) {
            j->fail[which]++;
            fprintf(stderr, "FAIL %s(%a) = %a, expected %a\n",
                    which ? "tanhf" : "expf", x, got, a);
        }
    } else {
        j->close[which]++;
        logf_(j, which ? "tanh" : "exp", ux, u_of(got));
    }
    return 0;
}

static void *run(void *p)
{
    Job *j = p;
    for (uint64_t u = j->lo; u < j->hi; u++) {
        check_one(j, 0, (uint32_t)u);
        check_one(j, 1, (uint32_t)u);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    int nt = argc > 1 ? atoi(argv[1]) : 4;
    if (nt < 1 || nt > 256) nt = 4;
    const uint64_t N = 1ULL << 32, chunk = 1ULL << 22;
    const uint64_t nchunks = N / chunk;
    Job *jobs = calloc(nchunks, sizeof *jobs);
    for (uint64_t c = 0; c < nchunks; c++) { jobs[c].lo = c * chunk; jobs[c].hi = (c + 1) * chunk; }

    /* nt workers over the chunks, in waves */
    pthread_t *th = calloc((size_t)nt, sizeof *th);
    for (uint64_t c = 0; c < nchunks; c += (uint64_t)nt) {
        int n = 0;
        for (; n < nt && c + (uint64_t)n < nchunks; n++) pthread_create(&th[n], NULL, run, &jobs[c + (uint64_t)n]);
        for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
        if ((c / (uint64_t)nt) % 64 == 0) fprintf(stderr, "  %3d%%\r", (int)(100 * c / nchunks));
    }
    uint64_t fail[2] = {0}, close[2] = {0}, diff[2] = {0};
    for (uint64_t c = 0; c < nchunks; c++) {
        for (int w = 0; w < 2; w++) { fail[w] += jobs[c].fail[w]; close[w] += jobs[c].close[w]; diff[w] += jobs[c].libm_diff[w]; }
        if (jobs[c].len) fwrite(jobs[c].log, 1, jobs[c].len, stdout);
    }
    const char *nm[2] = { "expf", "tanhf" };
    for (int w = 0; w < 2; w++)
        fprintf(stderr, "%-6s all 2^32 inputs: %llu FAIL, %llu close cases (to settle exactly), "
                "%llu differ from this libm\n", nm[w], (unsigned long long)fail[w],
                (unsigned long long)close[w], (unsigned long long)diff[w]);
    return (fail[0] || fail[1]) ? 1 : 0;
}
