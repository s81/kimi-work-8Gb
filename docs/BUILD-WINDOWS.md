# Building k3 for Windows

This fork adds a packaged Windows build on top of upstream's existing MinGW-w64 port.
The engine source is unchanged. `k3.exe` is a statically linked x86-64 console program
that imports only `KERNEL32.dll` and `msvcrt.dll`, so it needs no DLLs and no installer.

## Download a prebuilt binary

- **Every push / PR:** open the run in the repository's **Actions** tab, workflow
  **windows-build**, and download the `k3-windows-x64` artifact.
- **Tagged releases (`v*`):** the zips are attached to the GitHub Release.

| Zip | CPU requirement | Notes |
|---|---|---|
| `k3-<ver>-windows-x64-avx2.zip` | AVX2 + FMA (Intel Haswell 2013+, any AMD Ryzen) | Use this one. Same flags as `make portable`. |
| `k3-<ver>-windows-x64-generic.zip` | Any 64-bit x86 CPU | Scalar kernels, slower. Only use it if the avx2 build fails with `0xC000001D` (illegal instruction). |

```powershell
.\k3.exe --help
.\k3.exe --list-presets
```

To run the model you have to download the Kimi K3 weights separately (hundreds of GB).
See `README.md` and `docs/QUICKSTART.md`. Pass non-ASCII prompts with `--prompt-file`,
because Windows re-encodes argv to the active code page.

## Build it yourself

### Option A: native, MSYS2 (Windows)

1. Install [MSYS2](https://www.msys2.org/), then open the **"MSYS2 MinGW x64"** shell.
   Don't use the plain "MSYS2 MSYS" shell.
2. Install the toolchain:
   ```bash
   pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-libgomp make zip git
   ```
3. Build and package:
   ```bash
   ./scripts/build-windows.sh        # writes dist/*.zip
   ```
   For a binary tuned to *your own* CPU only, plain `make` uses `-march=native` and
   writes `bin/k3.exe`.
4. Optional: run the weightless test suite with `make test`.

### Option B: cross-compile from Linux or WSL

```bash
sudo apt-get install gcc-mingw-w64-x86-64-posix make zip
./scripts/build-windows.sh
```

Use the `-posix` threads flavour. The engine needs pthreads and OpenMP, and the
`-win32` flavour has neither.

## How this was verified

- The Linux cross-compile (MinGW-w64 GCC 13.2, posix) builds both variants with **zero
  warnings** under the project's `-Wall -Wextra -Wshadow -Wpointer-arith -Wvla`.
- `objdump -p k3.exe` shows only `KERNEL32.dll` and `msvcrt.dll` as imports.
- The full `make test` suite passed with the cross-compiled AVX2 `.exe` files run
  under Wine: `22 passed, 0 failed, 0 skipped`, `VERDICT: ENGINE MATCHES THE REFERENCE
  EXACTLY`, `ALL WEIGHTLESS TESTS PASSED`. Two subtests skip themselves by design on
  Windows (no inode or `fork`). Tokenizer parity did not run because it needs the
  real model's `tiktoken.model`, and upstream CI skips it for the same reason.
- CI repeats the build and test natively on `windows-latest`.
- Not verified: inference on real Kimi K3 weights on Windows. That needs the
  checkpoint, which this build did not have.
