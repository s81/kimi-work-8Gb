/* test_math.c - k3_expf and k3_tanhf give the correctly rounded float, on every platform.
 *
 * WHY THIS EXISTS
 *   The engine's activations, softmaxes and the KDA decay use k3_expf and k3_tanhf
 *   (src/core/k3_math.h) instead of the C library's, so that Windows and Linux compute
 *   the same bits. tools/check_math.c proved them correctly rounded on all 2^32 inputs,
 *   on Linux. This test runs on every build, on both platforms, and holds that result:
 *
 *   1. Known answers: special values, overflow and underflow edges, subnormals, and the
 *      inputs closest to a rounding boundary that the exhaustive check found. Expected
 *      bits come from a 300-bit computation (tools/check_math_exact.py --emit), not
 *      from any C library.
 *   2. Sweep: both functions over every 251st float bit pattern (17 million inputs per
 *      function), folded into a digest. The digest was recorded from the build that
 *      passed the exhaustive check; the same digest here means the same bits on those
 *      inputs.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "k3_math.h"

static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

typedef struct { const char *fn; uint32_t x, want; } Case;

static const Case CASES[] = {
#include "test_math_cases.inc"
};

#define SWEEP_STEP 251u
/* FNV-1a over (k3_expf bits, k3_tanhf bits) for x = 0, 251, 502, ... < 2^32. */
#define SWEEP_DIGEST 0x5c7379a2bb2338deULL

int main(void)
{
    int fails = 0, n = (int)(sizeof CASES / sizeof CASES[0]);
    printf("k3_expf / k3_tanhf, correctly rounded\n");
    for (int i = 0; i < n; i++) {
        const float x = f_of(CASES[i].x);
        const int is_exp = CASES[i].fn[0] == 'e';
        const uint32_t got = u_of(is_exp ? k3_expf(x) : k3_tanhf(x));
        if (got != CASES[i].want) {
            printf("  FAIL  %s(%a) = %08x, exact %08x\n", CASES[i].fn, x, got, CASES[i].want);
            fails++;
        }
    }
    printf("  %s  %d known answers (300-bit reference)\n", fails ? "FAIL" : "PASS", n);

    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint64_t u = 0; u < (1ULL << 32); u += SWEEP_STEP) {
        const float x = f_of((uint32_t)u);
        uint32_t w[2] = { u_of(k3_expf(x)), u_of(k3_tanhf(x)) };
        const unsigned char *b = (const unsigned char *)w;
        for (int k = 0; k < 8; k++) { h ^= b[k]; h *= 0x100000001b3ULL; }
    }
    const int sweep_ok = h == SWEEP_DIGEST;
    printf("  %s  sweep of every %u-th input: digest %016llx%s\n", sweep_ok ? "PASS" : "FAIL",
           SWEEP_STEP, (unsigned long long)h, sweep_ok ? "" : " (expected another value)");
    if (!sweep_ok) fails++;

    printf("\n%s\n", fails ? "MATH TESTS FAILED" : "MATH TESTS PASSED");
    return fails ? 1 : 0;
}
