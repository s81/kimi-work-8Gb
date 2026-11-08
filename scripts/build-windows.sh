#!/usr/bin/env bash
# Build Windows x86-64 binaries of k3 (k3.exe) and package them as zip files.
#
# Works in two places, picked automatically:
#
#   * MSYS2 "MinGW x64" shell on Windows (native build; uname reports MINGW64_*)
#       pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-libgomp make zip
#   * Linux / WSL (cross-compile with MinGW-w64)
#       sudo apt-get install gcc-mingw-w64-x86-64-posix make zip
#     The POSIX-threads flavour is required: the engine uses pthreads and OpenMP,
#     and the win32-threads flavour of MinGW-w64 GCC ships neither.
#
# Two variants are produced, because a distributed binary must not use -march=native
# (that ties it to the CPU of the build machine):
#
#   avx2     -mavx2 -mfma  -- the same flags as `make portable`. Fast path. Needs a CPU
#                             from roughly 2013 onwards (Intel Haswell / AMD Excavator,
#                             and every Ryzen).
#   generic  baseline x86-64 (SSE2) -- runs on any 64-bit x86 CPU, using the scalar
#                             kernels. Noticeably slower; use only if avx2 refuses to run.
#
# Usage:
#   scripts/build-windows.sh            # both variants
#   scripts/build-windows.sh avx2       # one variant
#   VERSION=1.1.0 scripts/build-windows.sh
#
# Output: dist/k3-<version>-windows-x64-<variant>.zip, each with k3.exe, LICENSE,
# NOTICE and WINDOWS.txt.
set -euo pipefail

cd "$(dirname "$0")/.."

variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(avx2 generic)

case "$(uname -s)" in
    MINGW*) CC_WIN=${CC_WIN:-gcc} ;;
    *)      CC_WIN=${CC_WIN:-x86_64-w64-mingw32-gcc} ;;
esac
command -v "$CC_WIN" >/dev/null 2>&1 || {
    echo "error: $CC_WIN not found. See the header of $0 for the packages to install." >&2
    exit 1
}

if [ -z "${VERSION:-}" ]; then
    VERSION=$(sed -n 's/^#define K3_VERSION "\(.*\)"/\1/p' src/cli/k3_run.c | head -n1)
    [ -n "$VERSION" ] || VERSION=$(git describe --tags --always 2>/dev/null || echo dev)
fi

JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
mkdir -p dist

for v in "${variants[@]}"; do
    case "$v" in
        avx2)    arch="-mavx2 -mfma" ;;
        generic) arch="" ;;
        *) echo "error: unknown variant '$v' (expected avx2 or generic)" >&2; exit 2 ;;
    esac

    build="build-win-$v"
    bin="bin-win-$v"
    echo "== building $v ($CC_WIN ${arch:-baseline x86-64}) =="
    # OMP_LDFLAGS mirrors the Makefile's own MINGW branch. It is set explicitly so a
    # Linux cross-build, where `uname` takes the Linux branch, links the same way:
    #   -static  : no libwinpthread-1.dll / libgomp-1.dll next to the exe
    #   -lpsapi  : GetProcessMemoryInfo for peak RSS reporting
    make -j"$JOBS" CC="$CC_WIN" ARCH="$arch" \
         OMP_LDFLAGS="-fopenmp -static -lpsapi" \
         BUILD="$build" BIN="$bin" all

    exe="$bin/k3.exe"
    [ -f "$exe" ] || { echo "error: $exe was not produced" >&2; exit 1; }

    stage="dist/k3-$VERSION-windows-x64-$v"
    rm -rf "$stage" "$stage.zip"
    mkdir -p "$stage"
    cp "$exe" LICENSE NOTICE "$stage/"
    cat > "$stage/WINDOWS.txt" <<EOF
k3 $VERSION -- Kimi K3 inference engine, Windows x86-64 build ($v)

Built with: $CC_WIN ${arch:-(baseline x86-64)} -O3 -fopenmp, statically linked.
No DLLs or runtime installers are needed; k3.exe depends only on KERNEL32.dll and
msvcrt.dll, which ship with every Windows version.

Variant:
  avx2     needs AVX2 + FMA (Intel Haswell 2013+, any AMD Ryzen). Use this one.
  generic  any 64-bit x86 CPU; scalar kernels, slower. Use only if the avx2 build
           exits immediately with "Illegal instruction" or 0xC000001D.

Quick check (PowerShell or cmd):
  .\\k3.exe --help
  .\\k3.exe --list-presets

Running the model needs the Kimi K3 weights downloaded separately (hundreds of GB);
see the project README, section "Quick start" / "Requirements". Pass non-ASCII prompts
with --prompt-file rather than --prompt: Windows re-encodes argv to the active code page.

Source: https://github.com/FareedKhan-dev/kimi-k3-in-c (Apache-2.0, see LICENSE/NOTICE).
EOF
    (cd dist && zip -qr "$(basename "$stage").zip" "$(basename "$stage")")
    echo "   -> $stage.zip"
done
