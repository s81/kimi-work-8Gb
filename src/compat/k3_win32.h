/* Windows compatibility layer for the Tiny C Compiler (TCC).
 *
 * build.c force-includes this header (tcc -include) into every translation unit of a
 * Windows build, and puts src/compat/win32/ on the include path for the POSIX headers
 * TCC's Windows runtime does not ship (unistd.h, pthread.h). Nothing here is used on
 * Linux, where TCC builds against the system's own C library.
 *
 * WHY IT IS NEEDED
 *   TCC on Windows links against msvcrt.dll, the C runtime that ships with every Windows
 *   version, and against an import list for kernel32.dll that dates from Windows XP.
 *   Several things the engine calls are therefore missing at link time even when TCC's
 *   headers declare them:
 *     - POSIX: clock_gettime, mkstemp, pthreads (src/compat/win32/pthread.h)
 *     - C99 maths: fma, which msvcrt.dll does not export
 *     - Vista-and-later kernel32 calls: GlobalMemoryStatusEx, K32GetProcessMemoryInfo,
 *       the condition-variable API, GetLogicalProcessorInformation
 *   The kernel32 ones are looked up at run time with GetProcAddress, so the binary needs
 *   no import library and runs on any Windows from Vista on.
 *
 * TEXT IS UTF-8 EVERYWHERE, as on Linux. Windows hands a C program its command line and
 * takes file names in the active ANSI code page, which cannot hold most of Unicode, so a
 * Chinese prompt or a model directory under a non-Latin user name arrived mangled. The
 * section "UTF-8 on Windows" below converts the command line to UTF-8 at the start of
 * main (k3_win_utf8_main), switches the console to UTF-8 for the run, and routes fopen,
 * open, opendir, rename, unlink, mkdir and mkstemp through the wide-character APIs, so
 * every path the engine handles is UTF-8 on both platforms. */
#ifndef K3_WIN32_H
#define K3_WIN32_H
#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <time.h>
#include <wchar.h>
#include <direct.h>

/* TCC 0.9.27's 64-bit Windows math.h defines fabsf and fabsl with x87 inline assembly
 * ("=t" constraint) that TCC itself cannot compile, so any call to them fails to build.
 * The header's versions are renamed out of the way here and replaced after it. */
#define fabsf k3_tcc_broken_fabsf
#define fabsl k3_tcc_broken_fabsl
#include <math.h>    /* before the fma macro below, which would mangle its prototype */
#undef fabsf
#undef fabsl
static float       k3_win_fabsf(float x)       { return (float)fabs((double)x); }
static long double k3_win_fabsl(long double x) { return x < 0 ? -x : x; }
#define fabsf(x) k3_win_fabsf(x)
#define fabsl(x) k3_win_fabsl(x)

/* ------------------------------------------------------------ run-time lookup -- */
/* One kernel32 export by name, or NULL. kernel32.dll is mapped into every process, so
 * GetModuleHandle never fails and there is nothing to unload. */
static FARPROC k3_win_kernel32(const char *name)
{
    return GetProcAddress(GetModuleHandleA("kernel32.dll"), name);
}

/* ------------------------------------------------------------ clock_gettime -- */
/* Only CLOCK_MONOTONIC is used (elapsed-time measurement). QueryPerformanceCounter is
 * Windows' monotonic high-resolution clock. */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef _TIMESPEC_DEFINED
#define _TIMESPEC_DEFINED
struct timespec { time_t tv_sec; long tv_nsec; };
#endif
static int k3_win_clock_gettime(int clk, struct timespec *ts)
{
    (void)clk;
    LARGE_INTEGER f, c;
    if (!QueryPerformanceFrequency(&f) || !QueryPerformanceCounter(&c)) {
        errno = EINVAL;
        return -1;
    }
    ts->tv_sec  = (time_t)(c.QuadPart / f.QuadPart);
    ts->tv_nsec = (long)((c.QuadPart % f.QuadPart) * 1000000000LL / f.QuadPart);
    return 0;
}
#define clock_gettime(clk, ts) k3_win_clock_gettime((clk), (ts))

/* --------------------------------------------------------------------- fma -- */
/* msvcrt.dll exports no fma(). Every call site in this engine multiplies two values that
 * were widened from float (or are exact powers of two times one), and a float x float
 * product needs at most 48 significant bits, which double's 53 hold exactly. The product
 * therefore never rounds, so a*b+c rounds exactly once, at the add -- the same single
 * rounding fma performs -- and the two are bit-identical. TCC never contracts a*b+c into
 * an fma instruction and uses SSE2 doubles on x86-64 (no x87 excess precision), so the
 * expression below is evaluated exactly as written. test_ops checks the kernels that use
 * it against the reference fixtures, and k3_model checks the whole engine bit-exactly.
 *
 * Do NOT call fma() on general doubles in this code base: for operands whose product
 * does round, this is not a fused multiply-add. */
static double k3_win_fma(double a, double b, double c) { return a * b + c; }
#define fma(a, b, c) k3_win_fma((a), (b), (c))

/* ---------------------------------------------------- float maths, msvcrt -- */
/* The 64-bit msvcrt.dll that TCC links against exports the double versions of these but
 * not the float ones. Each is computed in double and rounded once to float:
 *   sqrtf: exactly the correctly rounded result (double has more than twice float's
 *          precision, so the double rounding cannot change it).
 *   expf, tanhf: within the last bit of any libm's float version, which is the same
 *          freedom every C library already takes (glibc, MinGW and Apple's differ), and
 *          the test suite's tolerances were set across those libraries. */
static float k3_win_expf(float x)  { return (float)exp((double)x); }
static float k3_win_sqrtf(float x) { return (float)sqrt((double)x); }
static float k3_win_tanhf(float x) { return (float)tanh((double)x); }
#define expf(x)  k3_win_expf(x)
#define sqrtf(x) k3_win_sqrtf(x)
#define tanhf(x) k3_win_tanhf(x)

/* --------------------------------------------------------- strtoll/strtoull -- */
/* msvcrt.dll has the same conversions under Microsoft's names. Newer TCC headers
 * already map these; only define them where the headers do not. */
__int64          __cdecl _strtoi64(const char *, char **, int);
unsigned __int64 __cdecl _strtoui64(const char *, char **, int);
#ifndef strtoll
#define strtoll(s, e, b)  _strtoi64((s), (e), (b))
#endif
#ifndef strtoull
#define strtoull(s, e, b) _strtoui64((s), (e), (b))
#endif

/* ---------------------------------------------------- aligned allocation -- */
/* Declared explicitly: TCC's stdlib.h does not, and an implicitly declared function
 * returns int, which would cut every 64-bit pointer _aligned_malloc returns in half. */
void *__cdecl _aligned_malloc(size_t size, size_t alignment);
void  __cdecl _aligned_free(void *p);

/* ------------------------------------------------------------ UTF-8 on Windows -- */
/* UTF-8 to a freshly allocated UTF-16 string, or NULL. A string that is not valid UTF-8
 * is taken as the ANSI code page instead, so a name that reached here by some other
 * route (an environment variable, say) still opens. Free with free(). */
static wchar_t *k3_win_widen(const char *s)
{
    if (!s) return NULL;
    UINT cp = CP_UTF8;
    int n = MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) { cp = CP_ACP; n = MultiByteToWideChar(cp, 0, s, -1, NULL, 0); }
    if (n <= 0) return NULL;
    wchar_t *w = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!w) return NULL;
    if (MultiByteToWideChar(cp, cp == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0, s, -1, w, n) <= 0) {
        free(w);
        return NULL;
    }
    return w;
}

/* UTF-16 to a freshly allocated UTF-8 string, or NULL. Free with free(). */
static char *k3_win_narrow(const wchar_t *w)
{
    if (!w) return NULL;
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    char *s = (char *)malloc((size_t)n);
    if (!s) return NULL;
    if (WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) <= 0) { free(s); return NULL; }
    return s;
}

static FILE *k3_win_fopen(const char *path, const char *mode)
{
    wchar_t *wp = k3_win_widen(path), *wm = k3_win_widen(mode);
    FILE *f = (wp && wm) ? _wfopen(wp, wm) : NULL;
    if (!wp || !wm) errno = ENOMEM;
    free(wp); free(wm);
    return f;
}
#define fopen(path, mode) k3_win_fopen((path), (mode))

static int k3_win_unlink(const char *path)
{
    wchar_t *wp = k3_win_widen(path);
    if (!wp) { errno = ENOMEM; return -1; }
    const int r = _wunlink(wp);
    free(wp);
    return r;
}
#define unlink(path) k3_win_unlink(path)

/* The console's code pages as they were before this process changed them; a console
 * keeps its code page after the program exits, so they are put back at exit. */
static UINT k3_win_saved_out_cp, k3_win_saved_in_cp;
static void k3_win_restore_console(void)
{
    if (k3_win_saved_out_cp) SetConsoleOutputCP(k3_win_saved_out_cp);
    if (k3_win_saved_in_cp)  SetConsoleCP(k3_win_saved_in_cp);
}

/* Call first thing in main. Replaces argc/argv with the UTF-8 form of the real (UTF-16)
 * command line, and puts an attached console into UTF-8 for this run, so prompts typed
 * or passed in any language arrive intact and generated text prints as itself. On any
 * failure argv is left as the C runtime built it, which is the old behaviour. The
 * converted strings live for the whole process, like the originals. */
typedef LPWSTR *(WINAPI *K3CommandLineToArgvW)(LPCWSTR, int *);
static void k3_win_utf8_main(int *argc, char ***argv)
{
    if (GetConsoleOutputCP() != CP_UTF8) {
        const UINT out = GetConsoleOutputCP(), in = GetConsoleCP();
        if (out && SetConsoleOutputCP(CP_UTF8)) {
            k3_win_saved_out_cp = out;
            if (in && SetConsoleCP(CP_UTF8)) k3_win_saved_in_cp = in;
            atexit(k3_win_restore_console);
        }
    }

    HMODULE shell32 = LoadLibraryA("shell32.dll");
    K3CommandLineToArgvW split = shell32
        ? (K3CommandLineToArgvW)(void *)GetProcAddress(shell32, "CommandLineToArgvW") : NULL;
    if (!split) return;
    int n = 0;
    LPWSTR *wargv = split(GetCommandLineW(), &n);
    if (!wargv || n <= 0) return;
    char **out = (char **)calloc((size_t)n + 1, sizeof(char *));
    if (!out) { LocalFree(wargv); return; }
    for (int i = 0; i < n; i++) {
        out[i] = k3_win_narrow(wargv[i]);
        if (!out[i]) {                           /* keep the runtime's argv instead */
            for (int j = 0; j < i; j++) free(out[j]);
            free(out);
            LocalFree(wargv);
            return;
        }
    }
    LocalFree(wargv);
    *argc = n;
    *argv = out;
}

/* ------------------------------------------------------------------ mkstemp -- */
/* Replace the trailing XXXXXX of `tmpl` and create the file exclusively, as POSIX
 * mkstemp does. Returns a descriptor opened read/write in binary mode, or -1. */
static int k3_win_mkstemp(char *tmpl)
{
    const size_t n = strlen(tmpl);
    if (n < 6 || strcmp(tmpl + n - 6, "XXXXXX") != 0) { errno = EINVAL; return -1; }
    static const char alnum[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    unsigned long long seed = (unsigned long long)GetTickCount()
                            ^ ((unsigned long long)GetCurrentProcessId() << 32)
                            ^ (unsigned long long)(size_t)tmpl;
    for (int attempt = 0; attempt < 100; attempt++) {
        for (int i = 0; i < 6; i++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            tmpl[n - 6 + i] = alnum[(seed >> 33) % (sizeof alnum - 1)];
        }
        wchar_t *wt = k3_win_widen(tmpl);
        if (!wt) { errno = ENOMEM; return -1; }
        const int fd = _wopen(wt, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, 0600);
        free(wt);
        if (fd >= 0) return fd;
        if (errno != EEXIST) return -1;
    }
    errno = EEXIST;
    return -1;
}
#define mkstemp(t) k3_win_mkstemp(t)

/* ----------------------------------------------------- memory status (Vista+) -- */
/* GlobalMemoryStatusEx is declared by TCC's winbase.h but missing from its kernel32
 * import list, so a direct call fails to link. Resolved at run time instead. */
typedef BOOL (WINAPI *K3GlobalMemoryStatusEx)(MEMORYSTATUSEX *);
static BOOL k3_win_global_memory_status_ex(MEMORYSTATUSEX *ms)
{
    static K3GlobalMemoryStatusEx fn;
    if (!fn) fn = (K3GlobalMemoryStatusEx)(void *)k3_win_kernel32("GlobalMemoryStatusEx");
    return fn ? fn(ms) : FALSE;
}
#define GlobalMemoryStatusEx(ms) k3_win_global_memory_status_ex(ms)

/* ------------------------------------------------- peak working set (Win7+) -- */
/* Windows' peak-RSS figure. K32GetProcessMemoryInfo is kernel32's copy of psapi's
 * GetProcessMemoryInfo (Windows 7 and later); psapi.dll is tried for older systems. The
 * record is the documented PROCESS_MEMORY_COUNTERS layout, declared here because TCC
 * ships no psapi.h. Returns 0 when neither is available. */
typedef struct {
    DWORD  cb;
    DWORD  PageFaultCount;
    SIZE_T PeakWorkingSetSize;
    SIZE_T WorkingSetSize;
    SIZE_T QuotaPeakPagedPoolUsage;
    SIZE_T QuotaPagedPoolUsage;
    SIZE_T QuotaPeakNonPagedPoolUsage;
    SIZE_T QuotaNonPagedPoolUsage;
    SIZE_T PagefileUsage;
    SIZE_T PeakPagefileUsage;
} K3ProcessMemoryCounters;
typedef BOOL (WINAPI *K3GetProcessMemoryInfo)(HANDLE, K3ProcessMemoryCounters *, DWORD);
static double k3_win_peak_rss_bytes(void)
{
    K3GetProcessMemoryInfo fn =
        (K3GetProcessMemoryInfo)(void *)k3_win_kernel32("K32GetProcessMemoryInfo");
    if (!fn) {
        HMODULE psapi = LoadLibraryA("psapi.dll");
        if (psapi) fn = (K3GetProcessMemoryInfo)(void *)GetProcAddress(psapi,
                                                                     "GetProcessMemoryInfo");
    }
    if (!fn) return 0.0;
    K3ProcessMemoryCounters pmc;
    memset(&pmc, 0, sizeof pmc);
    pmc.cb = sizeof pmc;
    if (!fn(GetCurrentProcess(), &pmc, sizeof pmc)) return 0.0;
    return (double)pmc.PeakWorkingSetSize;
}

#endif /* _WIN32 */
#endif /* K3_WIN32_H */
