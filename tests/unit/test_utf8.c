/* test_utf8.c - non-ASCII text survives the trip through file names and argv.
 *
 * WHY THIS EXISTS
 *   On Linux a C program receives argv and passes file names as the bytes the user
 *   typed, which are UTF-8. On Windows the C runtime converts both to the active ANSI
 *   code page, which cannot represent most of Unicode, so a Chinese prompt, an emoji, or
 *   a model directory under a non-Latin user name arrived as '?' marks. The engine now
 *   converts the real UTF-16 command line to UTF-8 and opens files through the wide-
 *   character APIs (src/compat/k3_win32.h, "UTF-8 on Windows"). This test holds it there.
 *
 *   Part 1, every platform: create, write, list, open, rename, mkstemp and unlink files
 *   whose names are Chinese, accented Latin and an emoji (outside the BMP, so a UTF-16
 *   surrogate pair on Windows), through exactly the calls the engine uses.
 *
 *   Part 2, Windows only: start this program again with a non-ASCII argument through
 *   CreateProcessW, and have the child check that k3_win_utf8_main hands it the exact
 *   UTF-8 bytes.
 *
 * usage: test_utf8 <scratch_dir>
 *        test_utf8 --child <expected-utf8>    (internal, Windows)
 */
#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "k3_portable_io.h"   /* open/rename/mkdir shims on Windows */

/* "测试-é-😀": CJK, a precomposed accent, and U+1F600 (4 bytes in UTF-8). Spelled as
 * bytes so the test does not depend on how a compiler reads its own source file. */
static const char NAME[] =
    "\xe6\xb5\x8b\xe8\xaf\x95-\xc3\xa9-\xf0\x9f\x98\x80";

static int fails;

static void check(int ok, const char *what, const char *detail)
{
    printf("  %s  %-44s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) fails++;
}

static int files(const char *root)
{
    char dir[1024], file[1100], moved[1100], tmpl[1100];
    snprintf(dir, sizeof dir, "%s/%s", root, NAME);
    snprintf(file, sizeof file, "%s/%s.txt", dir, NAME);
    snprintf(moved, sizeof moved, "%s/moved-%s.txt", dir, NAME);
    snprintf(tmpl, sizeof tmpl, "%s/tmp-%s-XXXXXX", dir, NAME);

    mkdir(root, 0755);
    int made = mkdir(dir, 0755) == 0 || errno == EEXIST;
    check(made, "mkdir with a non-ASCII name", NULL);

    FILE *f = fopen(file, "wb");
    check(f != NULL, "fopen(\"wb\") with a non-ASCII name", NULL);
    if (f) { fputs(NAME, f); fclose(f); }

    char back[64] = {0};
    f = fopen(file, "rb");
    if (f) { size_t got = fread(back, 1, sizeof back - 1, f); back[got] = 0; fclose(f); }
    check(f != NULL && strcmp(back, NAME) == 0, "fopen(\"rb\") reads the same bytes back", NULL);

    int fd = open(file, O_RDONLY);
    check(fd >= 0, "open(O_RDONLY) with a non-ASCII name", NULL);
    if (fd >= 0) close(fd);

    int listed = 0;
    char want[1100];
    snprintf(want, sizeof want, "%s.txt", NAME);
    DIR *d = opendir(dir);
    check(d != NULL, "opendir on a non-ASCII directory", NULL);
    if (d) {
        struct dirent *e;
        while ((e = readdir(d)) != NULL) if (!strcmp(e->d_name, want)) listed = 1;
        closedir(d);
    }
    check(listed, "readdir returns the name as UTF-8", NULL);

    check(rename(file, moved) == 0, "rename between non-ASCII names", NULL);

    int tfd = mkstemp(tmpl);
    check(tfd >= 0, "mkstemp in a non-ASCII directory", NULL);
    if (tfd >= 0) { close(tfd); check(unlink(tmpl) == 0, "unlink the mkstemp file", NULL); }

    check(unlink(moved) == 0, "unlink a non-ASCII name", NULL);
    return 0;
}

#ifdef _WIN32
/* Child: argv[2] must be NAME exactly, byte for byte, after k3_win_utf8_main. */
static int child(int argc, char **argv)
{
    k3_win_utf8_main(&argc, &argv);
    if (argc != 3) return 3;
    return strcmp(argv[2], NAME) == 0 ? 0 : 4;
}

static void argv_roundtrip(void)
{
    wchar_t self[MAX_PATH];
    if (!GetModuleFileNameW(NULL, self, MAX_PATH)) { check(0, "argv: find this program", NULL); return; }
    wchar_t *wname = k3_win_widen(NAME);
    if (!wname) { check(0, "argv: widen the test name", NULL); return; }
    wchar_t cmd[2 * MAX_PATH];
    _snwprintf(cmd, sizeof cmd / sizeof cmd[0], L"\"%ls\" --child \"%ls\"", self, wname);
    cmd[sizeof cmd / sizeof cmd[0] - 1] = 0;
    free(wname);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    DWORD code = 99;
    if (CreateProcessW(NULL, cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    char detail[64];
    snprintf(detail, sizeof detail, "child exit %lu", (unsigned long)code);
    check(code == 0, "argv: a non-ASCII argument arrives as UTF-8", detail);
}
#endif

int main(int argc, char **argv)
{
#ifdef _WIN32
    if (argc >= 2 && !strcmp(argv[1], "--child")) return child(argc, argv);
#endif
    if (argc < 2) { fprintf(stderr, "usage: test_utf8 <scratch_dir>\n"); return 2; }
    printf("non-ASCII file names and arguments\n");
    files(argv[1]);
#ifdef _WIN32
    argv_roundtrip();
#else
    printf("  SKIP  argv round trip                                POSIX argv is already bytes\n");
#endif
    printf("\n%s\n", fails ? "UTF-8 TESTS FAILED" : "UTF-8 TESTS PASSED");
    return fails ? 1 : 0;
}
