#!/usr/bin/env python3
"""check_math_exact.py - settle the close cases from tools/check_math.c at 300 bits.

usage: python3 tools/check_math_exact.py close.txt      (needs: pip install mpmath)

Each line of close.txt is "exp|tanh <x as float bits> <k3's result as float bits>".
For each, compute e^x or tanh(x) to 300 bits, round it to the nearest float (ties to
even, subnormals and overflow included), and compare with k3's bits. Exit status 0
only if every case matches.

The same rounding routine prints the expected bits for tests/unit/test_math.c
(--emit), so the unit test's table comes from here, not from any C library.
"""
import struct
import sys

import mpmath

mpmath.mp.prec = 300


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits32(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def round_to_f32(v):
    """Nearest binary32 to the mpf v, ties to even. Returns the bits."""
    if v == 0:
        return 0
    sign = 0x80000000 if v < 0 else 0
    a = abs(v)
    if mpmath.isinf(a) or a >= mpmath.ldexp(1, 129):
        return sign | 0x7F800000
    e = int(mpmath.floor(mpmath.log(a, 2)))
    # guard against log rounding at exact powers of two
    while mpmath.ldexp(1, e) > a:
        e -= 1
    while mpmath.ldexp(1, e + 1) <= a:
        e += 1
    q = max(e, -126) - 23                       # exponent of one ulp
    n = a / mpmath.ldexp(1, q)
    lo = int(mpmath.floor(n))
    frac = n - lo
    if frac > 0.5 or (frac == 0.5 and lo % 2 == 1):
        lo += 1
    if lo == 0:
        return sign
    val = mpmath.ldexp(lo, q)
    if val >= mpmath.ldexp(1, 128) - mpmath.ldexp(1, 103):   # rounds past FLT_MAX
        return sign | 0x7F800000
    # val is exactly representable as a float, and so as a double: no double rounding
    return sign | bits32(float(val))


def exact(fn, xbits):
    x = mpmath.mpf(f32(xbits))
    if fn == "exp" and x > 200:
        return mpmath.inf          # far past overflow; avoids a 2^(10^38) number
    return mpmath.exp(x) if fn == "exp" else mpmath.tanh(x)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--emit":
        for tok in sys.argv[2:]:
            fn, xb = tok.split(":")
            xb = int(xb, 16)
            print("    { %-5s 0x%08xu, 0x%08xu },  /* %r */"
                  % ('"%s",' % fn, xb, round_to_f32(exact(fn, xb)), f32(xb)))
        return 0
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    n = bad = 0
    with open(sys.argv[1]) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            fn, xb, got = parts[0], int(parts[1], 16), int(parts[2], 16)
            want = round_to_f32(exact(fn, xb))
            n += 1
            if want != got:
                bad += 1
                print("FAIL %s(%r): k3 %08x, exact %08x" % (fn, f32(xb), got, want))
    print("%d close cases settled at 300 bits: %d FAIL" % (n, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
