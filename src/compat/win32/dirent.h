/* dirent.h for TCC on Windows: opendir/readdir/closedir on FindFirstFileA.
 *
 * TCC's own dirent.h declares these but nothing in msvcrt.dll implements them, and its
 * DIR keeps the search handle in a 32-bit long, which truncates a 64-bit HANDLE. This
 * header is found first because src/compat/win32 is on the include path (see build.c).
 * Only the fields the engine reads are provided: d_name. */
#ifndef K3_COMPAT_DIRENT_H
#define K3_COMPAT_DIRENT_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct dirent {
    char d_name[MAX_PATH];
};

typedef struct {
    HANDLE           find;
    WIN32_FIND_DATAA data;
    int              first;    /* data already holds an entry not yet returned */
    struct dirent    ent;
} DIR;

static DIR *opendir(const char *path)
{
    const size_t n = strlen(path);
    char *pattern = (char *)malloc(n + 3);
    if (!pattern) { errno = ENOMEM; return NULL; }
    memcpy(pattern, path, n);
    size_t m = n;
    if (m == 0 || (pattern[m - 1] != '/' && pattern[m - 1] != '\\')) pattern[m++] = '\\';
    pattern[m++] = '*';
    pattern[m] = '\0';

    DIR *d = (DIR *)calloc(1, sizeof *d);
    if (!d) { free(pattern); errno = ENOMEM; return NULL; }
    d->find = FindFirstFileA(pattern, &d->data);
    free(pattern);
    if (d->find == INVALID_HANDLE_VALUE) {
        free(d);
        errno = ENOENT;
        return NULL;
    }
    d->first = 1;
    return d;
}

static struct dirent *readdir(DIR *d)
{
    if (!d) return NULL;
    if (d->first) d->first = 0;
    else if (!FindNextFileA(d->find, &d->data)) return NULL;
    strncpy(d->ent.d_name, d->data.cFileName, sizeof d->ent.d_name - 1);
    d->ent.d_name[sizeof d->ent.d_name - 1] = '\0';
    return &d->ent;
}

static int closedir(DIR *d)
{
    if (!d) return -1;
    FindClose(d->find);
    free(d);
    return 0;
}

#endif /* K3_COMPAT_DIRENT_H */
