/* k3_math.h - expf and tanhf, correctly rounded, the same bits on every platform.
 *
 * WHY THIS EXISTS
 *   The C library's expf and tanhf are not specified to the last bit. glibc, msvcrt and
 *   every other libm round differently on some inputs, and even glibc changes between
 *   versions. So SiTU-GLU, ShortConv, the KDA decay, the attention softmax and the
 *   router gave slightly different floats on Windows and on Linux.
 *
 *   These versions return the CORRECTLY ROUNDED result: the float nearest to the exact
 *   mathematical value (ties to even). There is only one such float, so any correct
 *   implementation, on any platform, must return it. They are written in plain C double
 *   arithmetic: no libm call, no fma, no long double, no table built at run time. TCC
 *   compiles double arithmetic to SSE2 on x86-64 in the order written, so the code runs
 *   the same operations, and gets the same bits, on Linux and Windows.
 *
 *   tools/check_math.c checks all 2^32 float inputs of both functions against an
 *   independent long double reference, with every close case re-checked at 300 bits by
 *   tools/check_math_exact.py. tests/unit/test_math.c checks a fixed sample of hard
 *   cases on every build, on both platforms.
 *
 * HOW
 *   exp: y = k*ln2/32 + r with |r| <= ln2/64, so e^y = 2^(k/32) * e^r. 2^(j/32) comes
 *   from a 32-entry table, each entry stored as two doubles (hi + lo, about 107 bits),
 *   and e^r - 1 from its Taylor series to r^8 (truncation below 2^-77). The result is
 *   kept as an unevaluated sum of two doubles, s + e, accurate to about 2^-58 relative.
 *
 *   tanh(a) = em / (em + 2) with em = e^(2a) - 1, computed as s + e in the same way,
 *   with an exact product (Dekker's split, since there is no fma) where cancellation
 *   would otherwise cost bits. Below |x| = 2^-5 its Taylor series is used directly.
 *
 *   Rounding to float: s + e is first rounded to 53 bits "to odd" (if e is not zero and
 *   s is even, step s one unit towards e), then converted to float. Rounding to odd at
 *   53 bits and then to nearest at 24 bits gives the same float as rounding s + e once,
 *   because 53 >= 24 + 2. This also covers float subnormals.
 *
 * Do not "simplify" these expressions, reorder them, or use fma(): every rounding is
 * part of the error analysis, and of the exhaustive check.
 */
#ifndef K3_MATH_H
#define K3_MATH_H

#include <stdint.h>

typedef union { uint64_t u; double d; } k3m_bits;   /* type punning through a union: C99 */
static double k3m_d(uint64_t u) { k3m_bits b; b.u = u; return b.d; }
static uint64_t k3m_u(double d) { k3m_bits b; b.d = d; return b.u; }

/* 2^(j/32) = hi + lo, both doubles, from a 300-bit computation. */
static const k3m_bits K3M_EXP2_32[32][2] = {
    { { 0x3ff0000000000000ULL }, { 0x0000000000000000ULL } },
    { { 0x3ff059b0d3158574ULL }, { 0x3c8d73e2a475b465ULL } },
    { { 0x3ff0b5586cf9890fULL }, { 0x3c98a62e4adc610bULL } },
    { { 0x3ff11301d0125b51ULL }, { 0xbc96c51039449b3aULL } },
    { { 0x3ff172b83c7d517bULL }, { 0xbc819041b9d78a76ULL } },
    { { 0x3ff1d4873168b9aaULL }, { 0x3c9e016e00a2643cULL } },
    { { 0x3ff2387a6e756238ULL }, { 0x3c99b07eb6c70573ULL } },
    { { 0x3ff29e9df51fdee1ULL }, { 0x3c8612e8afad1255ULL } },
    { { 0x3ff306fe0a31b715ULL }, { 0x3c86f46ad23182e4ULL } },
    { { 0x3ff371a7373aa9cbULL }, { 0xbc963aeabf42eae2ULL } },
    { { 0x3ff3dea64c123422ULL }, { 0x3c8ada0911f09ebcULL } },
    { { 0x3ff44e086061892dULL }, { 0x3c489b7a04ef80d0ULL } },
    { { 0x3ff4bfdad5362a27ULL }, { 0x3c7d4397afec42e2ULL } },
    { { 0x3ff5342b569d4f82ULL }, { 0xbc807abe1db13cadULL } },
    { { 0x3ff5ab07dd485429ULL }, { 0x3c96324c054647adULL } },
    { { 0x3ff6247eb03a5585ULL }, { 0xbc9383c17e40b497ULL } },
    { { 0x3ff6a09e667f3bcdULL }, { 0xbc9bdd3413b26456ULL } },
    { { 0x3ff71f75e8ec5f74ULL }, { 0xbc816e4786887a99ULL } },
    { { 0x3ff7a11473eb0187ULL }, { 0xbc841577ee04992fULL } },
    { { 0x3ff82589994cce13ULL }, { 0xbc9d4c1dd41532d8ULL } },
    { { 0x3ff8ace5422aa0dbULL }, { 0x3c96e9f156864b27ULL } },
    { { 0x3ff93737b0cdc5e5ULL }, { 0xbc675fc781b57ebcULL } },
    { { 0x3ff9c49182a3f090ULL }, { 0x3c7c7c46b071f2beULL } },
    { { 0x3ffa5503b23e255dULL }, { 0xbc9d2f6edb8d41e1ULL } },
    { { 0x3ffae89f995ad3adULL }, { 0x3c97a1cd345dcc81ULL } },
    { { 0x3ffb7f76f2fb5e47ULL }, { 0xbc75584f7e54ac3bULL } },
    { { 0x3ffc199bdd85529cULL }, { 0x3c811065895048ddULL } },
    { { 0x3ffcb720dcef9069ULL }, { 0x3c7503cbd1e949dbULL } },
    { { 0x3ffd5818dcfba487ULL }, { 0x3c82ed02d75b3707ULL } },
    { { 0x3ffdfc97337b9b5fULL }, { 0xbc91a5cd4f184b5cULL } },
    { { 0x3ffea4afa2a490daULL }, { 0xbc9e9c23179c2893ULL } },
    { { 0x3fff50765b6e4540ULL }, { 0x3c99d3e12dd8a18bULL } },
};
#define K3M_LN2_32_HI   0x3f962e42ff000000ULL  /* ln2/32, 25 significant bits: k*hi is exact */
#define K3M_LN2_32_LO   0xbd7718432a1b0e26ULL  /* ln2/32 - hi                               */
#define K3M_INV_LN2_32  0x40471547652b82feULL  /* 32/ln2 (only picks k)                     */

/* s + e = a + b exactly, for any a, b (Knuth). */
static void k3m_two_sum(double a, double b, double *s, double *e)
{
    double t = a + b;
    double bb = t - a;
    *e = (a - (t - bb)) + (b - bb);
    *s = t;
}

/* p + e = a * b exactly (Dekker; no fma). Inputs here are far from overflow. */
static void k3m_split(double a, double *hi, double *lo)
{
    double c = 134217729.0 * a;   /* 2^27 + 1 */
    double h = c - (c - a);
    *hi = h;
    *lo = a - h;
}
static void k3m_two_prod(double a, double b, double *p, double *e)
{
    double ah, al, bh, bl;
    double t = a * b;
    k3m_split(a, &ah, &al);
    k3m_split(b, &bh, &bl);
    *e = ((ah * bh - t) + ah * bl + al * bh) + al * bl;
    *p = t;
}

/* Round s + e (|e| <= half an ulp of s, s != 0) to float, correctly. */
static float k3m_round_odd_to_float(double s, double e)
{
    if (e != 0.0) {
        uint64_t u = k3m_u(s);
        if (!(u & 1)) {
            if ((e > 0.0) == (s > 0.0)) u++; else u--;
            s = k3m_d(u);
        }
    }
    return (float)s;
}

/* 2^m as a double, for -1022 <= m <= 1023. */
static double k3m_pow2(int m) { return k3m_d((uint64_t)(m + 1023) << 52); }

/* y = (k/32)*ln2 + (rh + rl), |rh| <= ln2/64 (a hair more), rh + rl = the exact
 * remainder to about 2^-100. Valid for |y| < 2^14. Returns k. */
static int k3m_exp_reduce(double y, double *rh, double *rl)
{
    const double z  = y * k3m_d(K3M_INV_LN2_32);
    const double kd = (z + 6755399441055744.0) - 6755399441055744.0;  /* round to int: 1.5*2^52 */
    const double u  = y - kd * k3m_d(K3M_LN2_32_HI);                  /* exact */
    const double v  = kd * k3m_d(K3M_LN2_32_LO);
    k3m_two_sum(u, -v, rh, rl);
    return (int)kd;
}

/* e^(rh + rl) - 1 = qh + ql (unnormalised; |ql| < 0.006 |qh| + 2^-90). */
static void k3m_expm1_small(double rh, double rl, double *qh, double *ql)
{
    const double r2 = rh * rh;
    double p = 1.0 / 40320.0;
    p = 1.0 / 5040.0 + rh * p;
    p = 1.0 / 720.0  + rh * p;
    p = 1.0 / 120.0  + rh * p;
    p = 1.0 / 24.0   + rh * p;
    p = 1.0 / 6.0    + rh * p;
    p = 0.5          + rh * p;
    *qh = rh;
    *ql = rl + (r2 * p + rl * rh);
}

/* The accurate path: e^x correctly rounded, for every float x (checked on all 2^32). */
static float k3m_expf_acc(float x)
{
    if (x != x) return x + x;                    /* NaN */
    if (x > 89.0f)   return (float)(1e300 * 1e300);   /* overflow, also +inf */
    if (x < -104.0f) return 0.0f;                /* below half the smallest subnormal, also -inf */

    double rh, rl, qh, ql;
    const int k = k3m_exp_reduce((double)x, &rh, &rl);
    const unsigned ku = (unsigned)(k + 131072);  /* biased, so j and m need no signed division */
    const int j = (int)(ku & 31u), m = (int)(ku >> 5) - 4096;
    k3m_expm1_small(rh, rl, &qh, &ql);

    /* T * (1 + q) = Th + (Tl + Th*qh + (Th*ql + Tl*qh)), T = Th + Tl */
    const double th = k3m_d(K3M_EXP2_32[j][0].u), tl = k3m_d(K3M_EXP2_32[j][1].u);
    const double rest = tl + (th * qh + (th * ql + tl * qh));
    double s, e;
    k3m_two_sum(th, rest, &s, &e);
    const double sc = k3m_pow2(m);               /* exact scaling: s*sc stays normal */
    return k3m_round_odd_to_float(s * sc, e * sc);
}

/* The accurate path: tanh(x) correctly rounded, for every float x (checked on all 2^32). */
static float k3m_tanhf_acc(float x)
{
    if (x != x) return x + x;                    /* NaN */
    const double a = x < 0.0f ? -(double)x : (double)x;
    float r;

    if (a >= 10.0) {
        r = 1.0f;                                /* 1 - tanh(10) < 2^-28: rounds to 1 */
    } else if (a < 0.03125) {
        /* tanh a = a - a^3/3 + 2a^5/15 - 17a^7/315 + 62a^9/2835 - 1382a^11/155925 + ...
         * a^2 <= 2^-10, so the next term is below 2^-62 relative. Zero and subnormal
         * inputs come out exactly. */
        const double t = a * a;
        double p = -1382.0 / 155925.0;
        p = 62.0 / 2835.0  + t * p;
        p = -17.0 / 315.0  + t * p;
        p = 2.0 / 15.0     + t * p;
        p = -1.0 / 3.0     + t * p;
        double s, e;
        k3m_two_sum(a, a * (t * p), &s, &e);
        r = k3m_round_odd_to_float(s, e);
    } else {
        /* em = e^(2a) - 1 = 2^m T (1 + q) - 1, with 2a in [1/16, 20): k >= 2, m >= 0. */
        double rh, rl, qh, ql;
        const int k = k3m_exp_reduce(2.0 * a, &rh, &rl);
        const unsigned ku = (unsigned)(k + 131072);
        const int j = (int)(ku & 31u), m = (int)(ku >> 5) - 4096;
        k3m_expm1_small(rh, rl, &qh, &ql);
        const double sc = k3m_pow2(m);
        const double th = k3m_d(K3M_EXP2_32[j][0].u), tl = k3m_d(K3M_EXP2_32[j][1].u);

        double ph, pe;                           /* th*qh exactly */
        k3m_two_prod(th, qh, &ph, &pe);
        const double big  = th * sc - 1.0;       /* exact: a multiple of 2^(m-52), below 2^(m+1) */
        const double tail = (tl + (pe + (th * ql + tl * qh))) * sc;
        double h1, l1, emh, eml;
        k3m_two_sum(big, ph * sc, &h1, &l1);
        k3m_two_sum(h1, l1 + tail, &emh, &eml);

        /* D = em + 2 */
        double dh, dl;
        k3m_two_sum(emh, 2.0, &dh, &dl);
        dl += eml;
        k3m_two_sum(dh, dl, &dh, &dl);

        /* em / D as qh + q1 */
        const double q0 = emh / dh;
        double p, pe2;
        k3m_two_prod(q0, dh, &p, &pe2);
        const double rem = (((emh - p) - pe2) + eml) - q0 * dl;
        const double q1 = rem / dh;
        double s, e;
        k3m_two_sum(q0, q1, &s, &e);
        r = k3m_round_odd_to_float(s, e);
    }
    return (k3m_u((double)x) >> 63) ? -r : r;   /* the sign bit, so tanh(-0) = -0 */
}

/* ------------------------------------------------------------- fast paths --
 * The accurate paths above cost about 30 function calls under TCC, which inlines
 * nothing. Almost every input needs far less: a plain double approximation v whose
 * relative error is known to be below eps rounds to the right float whenever v*(1-eps)
 * and v*(1+eps) round to the SAME float, because rounding is monotonic and the exact
 * value lies between them. Only when that interval straddles a rounding boundary
 * (about 2 in 2^17 calls for tanh, 2 in 2^21 for exp) is the accurate path taken. The
 * result is therefore the accurate path's result on every input, and the exhaustive
 * check covers the combination as built.
 *
 * Written for TCC's code generator, which stores every intermediate to the stack and
 * loads it back, so each dependent step costs several cycles: a 128-entry table keeps
 * the remainder small (|r| <= ln2/256 = 2^-8.5) so a degree-4 polynomial suffices, and
 * the polynomial is evaluated in two independent halves (Estrin) to halve the chain.
 * j and m come from a biased unsigned shift: a signed "/ 32" compiles to idiv.
 *
 * Error budgets (relative, rounding errors included):
 *   e^y: degree 4, truncation r^5/5! < 2^-49.4; 2^(j/128) rounded to double 2^-53;
 *        a few roundings; total below 2^-48.5. eps = 2^-46. The fast path is only
 *        taken for x > -87, where e^x is a normal float.
 *   tanh, |x| < 2^-5: series to x^11 as in the accurate path, below 2^-51. eps = 2^-46.
 *   tanh, otherwise: (e - 1)/(e + 1) with e = e^(2|x|) as above; e/(e - 1) <= 16.5
 *        magnifies e's error at most 2^4.05, so below 2^-44. eps = 2^-42. */
/* Is the double v safely away from a float rounding boundary? For a result in the float
 * normal range, rounding to float drops the low 29 bits of v's significand, and the
 * boundaries (midpoints between floats) are where those bits equal 2^28. A relative
 * error below eps is less than eps * 2^53 units of v's last place, so v rounds like the
 * exact value when its dropped bits are more than that many units from 2^28: tol = 128
 * for eps = 2^-46, 2048 for eps = 2^-42. One subtraction and one compare, where the
 * interval test v*(1-eps), v*(1+eps) cost two multiplies and two conversions. The
 * boundary between FLT_MAX and infinity is such a midpoint too. */
#define K3M_CLEAR(u, tol) \
    ((uint32_t)(((uint32_t)(u) & 0x1FFFFFFFu) - 0x10000000u + (tol)) >= 2u * (tol))

/* 2^(j/128) rounded to double, from a 300-bit computation. */
static const k3m_bits K3M_EXP2_128[128] = {
    { 0x3ff0000000000000ULL }, { 0x3ff0163da9fb3335ULL }, { 0x3ff02c9a3e778061ULL }, { 0x3ff04315e86e7f85ULL },
    { 0x3ff059b0d3158574ULL }, { 0x3ff0706b29ddf6deULL }, { 0x3ff0874518759bc8ULL }, { 0x3ff09e3ecac6f383ULL },
    { 0x3ff0b5586cf9890fULL }, { 0x3ff0cc922b7247f7ULL }, { 0x3ff0e3ec32d3d1a2ULL }, { 0x3ff0fb66affed31bULL },
    { 0x3ff11301d0125b51ULL }, { 0x3ff12abdc06c31ccULL }, { 0x3ff1429aaea92de0ULL }, { 0x3ff15a98c8a58e51ULL },
    { 0x3ff172b83c7d517bULL }, { 0x3ff18af9388c8deaULL }, { 0x3ff1a35beb6fcb75ULL }, { 0x3ff1bbe084045cd4ULL },
    { 0x3ff1d4873168b9aaULL }, { 0x3ff1ed5022fcd91dULL }, { 0x3ff2063b88628cd6ULL }, { 0x3ff21f49917ddc96ULL },
    { 0x3ff2387a6e756238ULL }, { 0x3ff251ce4fb2a63fULL }, { 0x3ff26b4565e27cddULL }, { 0x3ff284dfe1f56381ULL },
    { 0x3ff29e9df51fdee1ULL }, { 0x3ff2b87fd0dad990ULL }, { 0x3ff2d285a6e4030bULL }, { 0x3ff2ecafa93e2f56ULL },
    { 0x3ff306fe0a31b715ULL }, { 0x3ff32170fc4cd831ULL }, { 0x3ff33c08b26416ffULL }, { 0x3ff356c55f929ff1ULL },
    { 0x3ff371a7373aa9cbULL }, { 0x3ff38cae6d05d866ULL }, { 0x3ff3a7db34e59ff7ULL }, { 0x3ff3c32dc313a8e5ULL },
    { 0x3ff3dea64c123422ULL }, { 0x3ff3fa4504ac801cULL }, { 0x3ff4160a21f72e2aULL }, { 0x3ff431f5d950a897ULL },
    { 0x3ff44e086061892dULL }, { 0x3ff46a41ed1d0057ULL }, { 0x3ff486a2b5c13cd0ULL }, { 0x3ff4a32af0d7d3deULL },
    { 0x3ff4bfdad5362a27ULL }, { 0x3ff4dcb299fddd0dULL }, { 0x3ff4f9b2769d2ca7ULL }, { 0x3ff516daa2cf6642ULL },
    { 0x3ff5342b569d4f82ULL }, { 0x3ff551a4ca5d920fULL }, { 0x3ff56f4736b527daULL }, { 0x3ff58d12d497c7fdULL },
    { 0x3ff5ab07dd485429ULL }, { 0x3ff5c9268a5946b7ULL }, { 0x3ff5e76f15ad2148ULL }, { 0x3ff605e1b976dc09ULL },
    { 0x3ff6247eb03a5585ULL }, { 0x3ff6434634ccc320ULL }, { 0x3ff6623882552225ULL }, { 0x3ff68155d44ca973ULL },
    { 0x3ff6a09e667f3bcdULL }, { 0x3ff6c012750bdabfULL }, { 0x3ff6dfb23c651a2fULL }, { 0x3ff6ff7df9519484ULL },
    { 0x3ff71f75e8ec5f74ULL }, { 0x3ff73f9a48a58174ULL }, { 0x3ff75feb564267c9ULL }, { 0x3ff780694fde5d3fULL },
    { 0x3ff7a11473eb0187ULL }, { 0x3ff7c1ed0130c132ULL }, { 0x3ff7e2f336cf4e62ULL }, { 0x3ff80427543e1a12ULL },
    { 0x3ff82589994cce13ULL }, { 0x3ff8471a4623c7adULL }, { 0x3ff868d99b4492edULL }, { 0x3ff88ac7d98a6699ULL },
    { 0x3ff8ace5422aa0dbULL }, { 0x3ff8cf3216b5448cULL }, { 0x3ff8f1ae99157736ULL }, { 0x3ff9145b0b91ffc6ULL },
    { 0x3ff93737b0cdc5e5ULL }, { 0x3ff95a44cbc8520fULL }, { 0x3ff97d829fde4e50ULL }, { 0x3ff9a0f170ca07baULL },
    { 0x3ff9c49182a3f090ULL }, { 0x3ff9e86319e32323ULL }, { 0x3ffa0c667b5de565ULL }, { 0x3ffa309bec4a2d33ULL },
    { 0x3ffa5503b23e255dULL }, { 0x3ffa799e1330b358ULL }, { 0x3ffa9e6b5579fdbfULL }, { 0x3ffac36bbfd3f37aULL },
    { 0x3ffae89f995ad3adULL }, { 0x3ffb0e07298db666ULL }, { 0x3ffb33a2b84f15fbULL }, { 0x3ffb59728de5593aULL },
    { 0x3ffb7f76f2fb5e47ULL }, { 0x3ffba5b030a1064aULL }, { 0x3ffbcc1e904bc1d2ULL }, { 0x3ffbf2c25bd71e09ULL },
    { 0x3ffc199bdd85529cULL }, { 0x3ffc40ab5fffd07aULL }, { 0x3ffc67f12e57d14bULL }, { 0x3ffc8f6d9406e7b5ULL },
    { 0x3ffcb720dcef9069ULL }, { 0x3ffcdf0b555dc3faULL }, { 0x3ffd072d4a07897cULL }, { 0x3ffd2f87080d89f2ULL },
    { 0x3ffd5818dcfba487ULL }, { 0x3ffd80e316c98398ULL }, { 0x3ffda9e603db3285ULL }, { 0x3ffdd321f301b460ULL },
    { 0x3ffdfc97337b9b5fULL }, { 0x3ffe264614f5a129ULL }, { 0x3ffe502ee78b3ff6ULL }, { 0x3ffe7a51fbc74c83ULL },
    { 0x3ffea4afa2a490daULL }, { 0x3ffecf482d8e67f1ULL }, { 0x3ffefa1bee615a27ULL }, { 0x3fff252b376bba97ULL },
    { 0x3fff50765b6e4540ULL }, { 0x3fff7bfdad9cbe14ULL }, { 0x3fffa7c1819e90d8ULL }, { 0x3fffd3c22b8f71f1ULL },
};
/* 128/ln2 (only picks k); ln2/128 = hi + lo with hi to 36 bits, so k*hi is exact for
 * |k| < 2^17. */
static const k3m_bits K3M_RED128[3] = {
    { 0x40671547652b82feULL }, { 0x3f762e42fefa0000ULL }, { 0x3d0cf79abc9e3b3aULL } };

static float k3_expf(float x)
{
    if (!(x > -87.0f && x < 89.0f)) return k3m_expf_acc(x);   /* NaN, inf, subnormal results */
    /* Few statements on purpose: TCC keeps values in registers only within one
     * expression, and stores each named intermediate to the stack. */
    const double y  = (double)x;
    const double kd = (y * K3M_RED128[0].d + 6755399441055744.0) - 6755399441055744.0;
    const double r  = (y - kd * K3M_RED128[1].d) - kd * K3M_RED128[2].d;
    const unsigned ku = (unsigned)((int)kd + 131072);         /* |k| < 2^15 here */
    k3m_bits sc; sc.u = (uint64_t)((ku >> 7) - 1u) << 52;     /* 2^m: m + 1023 = (ku>>7) - 1 */
    const double th = K3M_EXP2_128[ku & 127u].d;
    k3m_bits v;
    v.d = (th + th * ((r + (r * r) * 0.5) + (r * r) * (r * (1.0 / 6.0) + (r * r) * (1.0 / 24.0)))) * sc.d;
    if (K3M_CLEAR(v.u, 128u)) return (float)v.d;
    return k3m_expf_acc(x);
}

static float k3_tanhf(float x)
{
    union { float f; uint32_t u; } ux; ux.f = x;
    const double a = (double)x < 0.0 ? -(double)x : (double)x;
    if (!(a < 10.0)) return k3m_tanhf_acc(x);                  /* NaN, |x| >= 10 */
    double v;
    unsigned tol;
    if (a < 0.03125) {
        const double t = a * a;
        v = a + a * (t * ((-1.0 / 3.0 + t * (2.0 / 15.0))
                   + (t * t) * ((-17.0 / 315.0 + t * (62.0 / 2835.0)) + (t * t) * (-1382.0 / 155925.0))));
        tol = 128u;
    } else {
        const double y  = a + a;
        const double kd = (y * K3M_RED128[0].d + 6755399441055744.0) - 6755399441055744.0;
        const double r  = (y - kd * K3M_RED128[1].d) - kd * K3M_RED128[2].d;
        const unsigned ku = (unsigned)((int)kd + 131072);
        k3m_bits sc; sc.u = (uint64_t)((ku >> 7) - 1u) << 52;
        const double th = K3M_EXP2_128[ku & 127u].d;
        const double e  = (th + th * ((r + (r * r) * 0.5)
                        + (r * r) * (r * (1.0 / 6.0) + (r * r) * (1.0 / 24.0)))) * sc.d;
        v = (e - 1.0) / (e + 1.0);
        tol = 2048u;
    }
    if (a == 0.0) return x;                                    /* +-0, sign kept */
    if (a < 1e-35) return k3m_tanhf_acc(x);                    /* float-subnormal results */
    k3m_bits vb; vb.d = v;
    if (K3M_CLEAR(vb.u, tol)) { const float f = (float)v; return (ux.u >> 31) ? -f : f; }
    return k3m_tanhf_acc(x);
}

#endif /* K3_MATH_H */
