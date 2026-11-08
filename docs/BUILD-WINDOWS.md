# Building k3 on Windows

k3 builds on Windows with the **Tiny C Compiler (TCC)** and nothing else: no MSYS2,
no MinGW, no Visual Studio, no make, no shell scripts, no Python. The same `build.c`
that builds on Linux builds on Windows:

```bat
tcc -run build.c          :: builds bin\k3.exe
tcc -run build.c test     :: builds and runs every test that needs no model weights
tcc -run build.c dist     :: packages dist\k3-<version>-windows-x64.zip
```

`k3.exe` imports only `kernel32.dll` and `msvcrt.dll`, which ship with every version of
Windows, so it needs no installer or extra DLLs. Windows Vista or later is required
(the engine uses Vista's condition variables).

## One command: setup.bat

From a fresh clone (or the source zip from GitHub), in a Command Prompt or by
double-clicking:

```bat
setup.bat              :: toolchain if needed, then build k3 and run the tests
setup.bat -SkipTests   :: build only
```

It runs [`scripts/setup-windows.ps1`](../scripts/setup-windows.ps1), which does the steps
in *Get the right TCC* below by itself:

1. downloads the official TCC 0.9.27 and checks its SHA-256;
2. downloads TinyCC's source at the pinned commit as a zip (no `git` needed) and checks a
   SHA-256 over every file in it;
3. builds TCC 0.9.28rc with TCC 0.9.27;
4. builds `bin\k3.exe` and runs `tcc -run build.c test`.

Everything goes into `.tcc\` inside the repository (about 30 MB): nothing is installed
system-wide and `PATH` is not changed. Running it again skips the toolchain. It needs
Windows 10 or later (PowerShell 5.1, `curl.exe`, TLS 1.2) and, the first time, access to
`download.savannah.gnu.org` and `repo.or.cz`. CI runs it on a fresh `windows-latest`
checkout on every push.

## Download a prebuilt binary

- **Every push and pull request:** the repository's **Actions** tab, workflow **CI**,
  artifact `k3-Windows-x64`.
- **Tagged releases (`v*`):** attached to the GitHub Release, with `SHA256SUMS.txt`.

Check a download in PowerShell:

```powershell
Get-FileHash .\k3-1.3.0-windows-x64.zip -Algorithm SHA256   # compare with SHA256SUMS.txt
.\k3.exe --help
```

Running the model needs the Kimi K3 weights, downloaded separately (about 1.56 TB).
Prompts in any language work with `--prompt`: `k3.exe` reads its command line as Unicode
(builds up to v1.2.2 did not; use `--prompt-file` with those).

## Get the right TCC

Use **TCC 0.9.28rc** built from TinyCC's source at commit `e23dabed` (2026-10-16). The
official 0.9.27 Windows download is **not** enough on its own: it miscompiles negated
floating-point comparisons with NaN (`!(x > 0.0)` behaves as `x <= 0.0`), which breaks the
engine's NaN guards. `tcc -run build.c test` catches it (test_sampler, "NaN temperature is
rejected").

0.9.27 can build 0.9.28rc, so the toolchain is TCC all the way down. `setup.bat` (above)
does this for you; by hand, in a Command Prompt with `git` available:

```bat
:: 1. the official 0.9.27 release as the bootstrap compiler
::    https://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27-win64-bin.zip
::    SHA-256 34a721949a2583fdff725312da092fa0f5f1f284b702e6f811c6954714faabb2
::    unzip it to C:\tcc-boot

:: 2. the pinned source
git clone https://repo.or.cz/tinycc.git C:\tinycc
git -C C:\tinycc checkout e23dabed154a8495f3ac1ff3cc9d04fa1c207893

:: 3. 0.9.27's kernel32 import list is too old for the new TCC; give it the new lists
copy /y C:\tinycc\win32\lib\*.def C:\tcc-boot\tcc\lib\

:: 4. TinyCC's own build script, compiling TCC with TCC
cd C:\tinycc\win32
build-tcc.bat -c C:\tcc-boot\tcc\tcc.exe -t x86_64
set PATH=C:\tinycc\win32;%PATH%
tcc -v
```

`C:\tinycc\win32` then holds a self-contained toolchain (`tcc.exe`, `libtcc.dll`,
`lib\`, `include\`). CI does exactly this in
[`.github/actions/setup-tcc`](../.github/actions/setup-tcc/action.yml).

If `tcc` is not on `PATH`, tell `build.c` where it is: `set CC=C:\tinycc\win32\tcc.exe`.

## What the Windows port needs

TCC on Windows links against `msvcrt.dll` and an XP-era `kernel32` import list.
`build.c` therefore force-includes `src/compat/k3_win32.h` and puts `src/compat/win32/`
on the include path for Windows builds:

| Gap | Filled by |
|---|---|
| `pthread.h` | `src/compat/win32/pthread.h`: threads, critical sections, Vista condition variables |
| `unistd.h`, `dirent.h` | small headers in `src/compat/win32/` on msvcrt and `FindFirstFileW` |
| Unicode command line, console and file names | `k3_win_utf8_main` converts the UTF-16 command line to UTF-8 and switches the console to UTF-8 for the run; chat input is read with `ReadConsoleW`; `fopen`, `open`, `opendir`, `rename`, `unlink`, `mkdir` and `mkstemp` use the wide-character APIs with UTF-8 names. `tests/unit/test_utf8.c` checks all of it |
| `strtod` in `msvcrt.dll` | it measures the whole remaining string before parsing, so the JSON parser hands it a copy of each number; parsing in place inside a large document was quadratic |
| 32-bit `off_t`/`lseek` | `_lseeki64`; offsets are held in `int64_t`, never `off_t` (a cast would wrap past 2 GB) |
| `clock_gettime`, `mkstemp` | `QueryPerformanceCounter`, an exclusive-create loop |
| `fma`, `sqrtf` | double-precision equivalents; `fma` is exact only because every call site multiplies float-derived values, and `sqrtf` from double is exactly the correctly rounded result |
| `expf`, `tanhf` | not taken from msvcrt at all: the engine uses its own correctly rounded `k3_expf` and `k3_tanhf` (`src/core/k3_math.h`), so Windows and Linux get the same bits |
| `GlobalMemoryStatusEx`, `K32GetProcessMemoryInfo`, condition variables, `GetLogicalProcessorInformation` | looked up at run time with `GetProcAddress` |
| `fabsf`/`fabsl` in TCC's `math.h` | replaced: TCC 0.9.27's header uses x87 assembly TCC cannot compile |

## Speed

TCC builds run the scalar kernels without SIMD and with little optimisation: roughly 2-3x
slower than the old GCC `-O3` AVX2 build on the compute side, with bit-identical output.
See the README's *Building* section for the measurements.

## Verified

- `tcc -run build.c test` passes in full with TCC 0.9.28rc on Windows (here under Wine,
  and natively in CI on `windows-latest`): `22 passed, 0 failed, 0 skipped`, `VERDICT:
  ENGINE MATCHES THE REFERENCE EXACTLY`, `ALL WEIGHTLESS TESTS PASSED`. Two subtests skip
  themselves by design on Windows (no inodes, no `fork`).
- On real Kimi K3 weights the same TCC code passed `test-all`, the torch binding
  check (`verify_real_layer.py`), the expert byte check and tokenizer parity (45/45), but
  on **Linux** only; see *Without the whole checkpoint* in [TESTING.md](TESTING.md).
- Not verified: any real-weight run on Windows, and inference with all 93 layers.
