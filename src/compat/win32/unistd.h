/* unistd.h for TCC on Windows: the subset of POSIX this engine uses, mapped onto
 * msvcrt.dll. Only on the include path of Windows builds (see build.c). */
#ifndef K3_COMPAT_UNISTD_H
#define K3_COMPAT_UNISTD_H

#include <io.h>        /* close, read, write, lseek, unlink, access, _commit */
#include <process.h>   /* getpid */
#include <direct.h>    /* rmdir, getcwd */
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef F_OK
#define F_OK 0
#define W_OK 2
#define R_OK 4
#endif

/* ftruncate: _chsize_s takes a 64-bit length, so files over 2 GB truncate correctly. */
errno_t __cdecl _chsize_s(int fd, long long size);
static int k3_win_ftruncate(int fd, long long len)
{
    return _chsize_s(fd, len) == 0 ? 0 : -1;
}
#define ftruncate(fd, len) k3_win_ftruncate((fd), (long long)(len))

#endif /* K3_COMPAT_UNISTD_H */
