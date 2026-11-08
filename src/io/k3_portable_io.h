/* k3_portable_io.h - shims for the Linux-only I/O calls the readers use.
 *
 * The engine asks for things Linux gives it and the other two platforms do not spell
 * the same way:
 *
 *   O_DIRECT       bypass the page cache on the trunk and expert reads. Darwin's
 *                  equivalent is not an open() flag but fcntl(F_NOCACHE) after the
 *                  fact, so O_DIRECT is defined to 0 there (open() is unaffected) and
 *                  k3_set_direct() applies the real thing to the returned descriptor.
 *                  Windows is the opposite of Darwin: unbuffered I/O (FILE_FLAG_NO_
 *                  BUFFERING) can ONLY be set at CreateFile time, same constraint as
 *                  Linux's real O_DIRECT, but nothing in the MinGW runtime maps an
 *                  open() flag onto it -- so open() itself is intercepted below.
 *
 *   posix_fadvise  a page-cache prefetch hint with no Darwin or Windows equivalent.
 *                  Callers already treat it as advisory -- the one call site returns
 *                  early on the direct path because the hint has nothing to populate
 *                  there -- so the shim is a no-op that keeps the buffered path
 *                  compiling.
 *
 *   pread          positioned read. Native on Linux and Darwin. The MinGW runtime has
 *                  no equivalent, so Windows gets one built on ReadFile's OVERLAPPED
 *                  Offset/OffsetHigh fields -- a true positioned read that does not
 *                  touch a shared file-pointer, unlike SetFilePointerEx + ReadFile,
 *                  which would race when the trunk reader thread and the expert-cache
 *                  prefetch threads pread() the same fd concurrently.
 *
 *   posix_memalign Native on Linux and Darwin. Windows gets a thin wrapper over
 *                  _aligned_malloc, which takes its (size, align) arguments in the
 *                  opposite order.
 *
 *   getline        POSIX.1-2008 line reader used by the chat REPL. Native on Linux and
 *                  Darwin; absent from the MinGW runtime. Windows gets a small
 *                  fgetc-based implementation with the same contract: it grows *lineptr
 *                  with realloc, keeps the delimiter, and returns -1 on EOF with nothing
 *                  read, so the REPL's `< 0` check ends the session the same way on
 *                  every platform.
 *
 *   fsync          durability barrier the chat history writer uses before its atomic
 *                  rename. Native on Linux and Darwin; Windows spells the same thing
 *                  _commit(), which flushes the OS buffers for one CRT descriptor.
 *
 *   rename         the atomic replace itself. POSIX rename() overwrites an existing
 *                  destination; the CRT's does not, so every history save after the
 *                  first failed on Windows. Windows gets MoveFileEx with
 *                  MOVEFILE_REPLACE_EXISTING, which has the POSIX semantics.
 *
 *   mkdir          POSIX takes a path and a permission mode; the CRT's _mkdir takes
 *                  only the path. Windows gets a wrapper that accepts and ignores the
 *                  mode, used by test fixtures that build a scratch directory tree.
 *
 * All four call sites fall back to buffered reads (or, for pread/posix_memalign, have
 * no fallback because the shim IS the implementation) when the direct path is
 * unavailable, so none of this changes what the engine computes, only how fast it
 * reads -- except on Windows, where pread and posix_memalign are load-bearing rather
 * than a speed path, since the codebase has no buffered-only alternative to either.
 *
 * A fifth difference is not an open() flag at all: pread() itself returns EINVAL on
 * Darwin for a single request of 2^31 bytes or more, where Linux either succeeds or
 * returns a short read that a retry loop already handles. The embedding table (2.35 GB)
 * and a packed trunk layer (up to 2.37 GB) both exceed that in one contiguous span.
 * K3_PREAD_MAX is the per-syscall cap every such read loop chunks against; on platforms
 * without the limit it just turns one syscall into a few, which costs nothing measurable
 * against a multi-gigabyte transfer.
 */
#ifndef K3_PORTABLE_IO_H
#define K3_PORTABLE_IO_H

/* The readers define _POSIX_C_SOURCE, which hides Darwin's non-standard fcntl commands
 * (F_NOCACHE among them) from <fcntl.h>. _DARWIN_C_SOURCE puts them back. It must be
 * set before the first libc header is pulled in, so this header is included first. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <fcntl.h>
#include <stdint.h>

#define K3_PREAD_MAX ((int64_t)1 << 30)   /* 1 GiB per syscall; see file header */

#if defined(__APPLE__)

/* Not an open() flag on Darwin: defining it to 0 leaves open() semantics untouched. */
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

static inline int posix_fadvise(int fd, off_t off, off_t len, int advice)
{
    (void)fd; (void)off; (void)len; (void)advice;
    return 0;   /* advisory only; the buffered path is correct without it */
}

/* Darwin's O_DIRECT equivalent, applied after open(). Failure is not fatal: the caller
 * keeps the descriptor and reads through the page cache instead. */
static inline int k3_set_direct(int fd)
{
    if (fd < 0) return -1;
    return fcntl(fd, F_NOCACHE, 1);
}

#elif defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <errno.h>
#include <stdio.h>    /* FILE, fgetc: k3_getline below */
#include <stdlib.h>
#include <string.h>

#ifndef O_DIRECT
#define O_DIRECT 0x40000000   /* sentinel bit; intercepted by k3_win_open() below */
#endif

#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

static inline int posix_fadvise(int fd, long long off, long long len, int advice)
{
    (void)fd; (void)off; (void)len; (void)advice;
    return 0;   /* advisory only; no Windows equivalent */
}

/* True positioned read: OVERLAPPED with only Offset/OffsetHigh set, on a handle NOT
 * opened with FILE_FLAG_OVERLAPPED, completes synchronously and does not touch any
 * shared file-pointer state -- safe for concurrent callers on the same fd. */
static inline long long k3_pread(int fd, void *buf, size_t count, long long offset)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }

    OVERLAPPED ov;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(offset & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)((unsigned long long)offset >> 32);

    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)count, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        errno = EIO;
        return -1;
    }
    return (long long)got;
}
#define pread(fd, buf, count, offset) k3_pread((fd), (buf), (count), (offset))

/* O_DIRECT has to be intercepted at open() itself: unlike Darwin's post-hoc fcntl,
 * Windows has no way to add FILE_FLAG_NO_BUFFERING to an already-open handle. The
 * bridge back to a plain int fd -- so every other pread()/close() call site in the
 * codebase stays untouched -- goes through _open_osfhandle.
 *
 * Every reader call site in this codebase passes O_RDONLY, so the access mode below
 * matters only to a caller outside the streaming path, such as a test that opens a
 * fixture file for writing: CreateFileA needs GENERIC_WRITE for that, or a later
 * ftruncate/write on the resulting handle fails with access denied even though the
 * open() call itself succeeded. */
static inline int k3_win_open(const char *path, int flags, ...)
{
    DWORD fileFlags = FILE_ATTRIBUTE_NORMAL;
    if (flags & O_DIRECT) fileFlags |= FILE_FLAG_NO_BUFFERING;

    DWORD access = GENERIC_READ;
    int crtFlags = _O_RDONLY;
    if ((flags & (O_WRONLY | O_RDWR)) == O_WRONLY) {
        access = GENERIC_WRITE;
        crtFlags = _O_WRONLY;
    } else if (flags & O_RDWR) {
        access = GENERIC_READ | GENERIC_WRITE;
        crtFlags = _O_RDWR;
    }

    /* UTF-8 path, as everywhere in this codebase; see "UTF-8 on Windows" in
     * src/compat/k3_win32.h. */
    wchar_t *wpath = k3_win_widen(path);
    if (!wpath) { errno = ENOMEM; return -1; }
    HANDLE h = CreateFileW(wpath, access,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL, OPEN_EXISTING, fileFlags, NULL);
    free(wpath);
    if (h == INVALID_HANDLE_VALUE) { errno = ENOENT; return -1; }

    int fd = _open_osfhandle((intptr_t)h, crtFlags | _O_BINARY);
    if (fd < 0) { CloseHandle(h); errno = EMFILE; return -1; }
    return fd;
}
#define open(path, flags, ...) k3_win_open((path), (flags))

/* posix_memalign for Windows, whose C runtime has no such function: _aligned_malloc
 * provides the same aligned block. Freeing it needs _aligned_free, not free(); see
 * k3_aligned_free below. */
static inline int posix_memalign(void **out, size_t align, size_t len)
{
    *out = _aligned_malloc(len, align);
    return *out ? 0 : ENOMEM;
}

/* On real POSIX, a posix_memalign'd pointer is safe to pass to plain free() -- that is
 * the entire point of the interface. _aligned_malloc has no such guarantee: pairing it
 * with free() instead of _aligned_free() corrupts the heap (observed directly: Windows
 * terminates the process with STATUS_HEAP_CORRUPTION). k3_aligned_free() exists so the
 * two call sites that free a posix_memalign'd arena (k3_cache.c, k3_trunk.c) can free it
 * correctly on every platform without special-casing Windows at the call site itself. */
#define k3_aligned_free(p) _aligned_free(p)

/* getline(3) for MinGW, which ships no such symbol. Same contract as POSIX: *lineptr and
 * *n describe a caller-owned buffer that this function may grow with realloc; the
 * delimiter is kept; the return is the number of bytes read, or -1 at EOF with nothing
 * read or on allocation failure. fgetc rather than fgets so an embedded NUL in the line
 * cannot silently truncate the count. */
/* A line typed at a Windows console, as UTF-8. The C runtime reads console input in the
 * console's code page, and with that set to UTF-8 the byte-oriented read returns nothing
 * for non-ASCII characters on many Windows versions, so a prompt typed in Chinese came
 * through empty. ReadConsoleW returns the UTF-16 the user typed, converted here; the
 * line keeps its '\n', as getline's does. Ctrl-Z at the start of a line is EOF, as for
 * the runtime. Returns -2 when the console read itself fails, so the caller falls back
 * to the byte path. */
static inline long long k3_win_console_getline(char **lineptr, size_t *n, HANDLE in)
{
    wchar_t buf[512];
    wchar_t *w = NULL;
    size_t wlen = 0, wcap = 0;
    for (;;) {
        DWORD got = 0;
        if (!ReadConsoleW(in, buf, (DWORD)(sizeof buf / sizeof buf[0]), &got, NULL)) {
            free(w);
            return -2;
        }
        if (got == 0) break;
        if (wlen + got + 1 > wcap) {
            size_t cap = wcap ? wcap * 2 : 1024;
            while (cap < wlen + got + 1) cap *= 2;
            wchar_t *grown = (wchar_t *)realloc(w, cap * sizeof(wchar_t));
            if (!grown) { free(w); errno = ENOMEM; return -1; }
            w = grown; wcap = cap;
        }
        memcpy(w + wlen, buf, got * sizeof(wchar_t));
        wlen += got;
        if (w[wlen - 1] == L'\n') break;
    }
    if (wlen == 0 || w[0] == 0x1A) { free(w); return -1; }      /* EOF or Ctrl-Z */
    if (wlen >= 2 && w[wlen - 2] == L'\r' && w[wlen - 1] == L'\n') {
        w[wlen - 2] = L'\n'; wlen--;                           /* CRLF -> LF */
    }
    w[wlen] = 0;
    char *u = k3_win_narrow(w);
    free(w);
    if (!u) { errno = ENOMEM; return -1; }
    const size_t len = strlen(u);
    if (*n < len + 1) {
        char *grown = (char *)realloc(*lineptr, len + 1);
        if (!grown) { free(u); errno = ENOMEM; return -1; }
        *lineptr = grown; *n = len + 1;
    }
    memcpy(*lineptr, u, len + 1);
    free(u);
    return (long long)len;
}

static inline long long k3_getline(char **lineptr, size_t *n, FILE *stream)
{
    if (!lineptr || !n || !stream) { errno = EINVAL; return -1; }
    if (!*lineptr || *n == 0) {
        *n = 128;
        *lineptr = (char *)malloc(*n);
        if (!*lineptr) { errno = ENOMEM; return -1; }
    }
    if (stream == stdin) {
        HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode;
        if (in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &mode)) {
            const long long r = k3_win_console_getline(lineptr, n, in);
            if (r != -2) return r;
        }
    }
    size_t len = 0;
    int c;
    while ((c = fgetc(stream)) != EOF) {
        if (len + 2 > *n) {
            size_t cap = *n * 2;
            char *grown = (char *)realloc(*lineptr, cap);
            if (!grown) { errno = ENOMEM; return -1; }
            *lineptr = grown; *n = cap;
        }
        (*lineptr)[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0 && c == EOF) return -1;
    (*lineptr)[len] = '\0';
    return (long long)len;
}
#define getline(lineptr, n, stream) k3_getline((lineptr), (n), (stream))

/* fsync(2) for MinGW: _commit() flushes the OS write buffers for one CRT descriptor,
 * which is the durability barrier the caller wants before its atomic rename. */
#define fsync(fd) _commit(fd)

/* rename(2) for MinGW. POSIX rename replaces an existing destination atomically, and
 * the chat history writer depends on exactly that: every save after the first renames
 * a fresh temp file over the transcript it is replacing. The CRT's rename() is
 * MoveFile(), which refuses an existing destination with EEXIST, so on Windows the
 * second save of a session failed and the transcript was never updated (caught by
 * test_chat's "atomic JSONL history write", whose destination pre-exists). MoveFileEx
 * with MOVEFILE_REPLACE_EXISTING is the documented replace-in-place, and
 * MOVEFILE_WRITE_THROUGH makes it return only once the change is on disk, which is
 * the same durability the fsync above bought for the data. */
static inline int k3_win_rename(const char *from, const char *to)
{
    wchar_t *wf = k3_win_widen(from), *wt = k3_win_widen(to);
    const BOOL ok = wf && wt &&
        MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    const DWORD err = (wf && wt) ? GetLastError() : ERROR_NOT_ENOUGH_MEMORY;
    free(wf); free(wt);
    if (ok) return 0;
    switch (err) {
    case ERROR_FILE_NOT_FOUND: case ERROR_PATH_NOT_FOUND: errno = ENOENT; break;
    case ERROR_ACCESS_DENIED:  case ERROR_SHARING_VIOLATION: errno = EACCES; break;
    default: errno = EIO; break;
    }
    return -1;
}
#define rename(from, to) k3_win_rename((from), (to))

/* mkdir(2) for MinGW. POSIX mkdir takes a path and a permission mode; the CRT's
 * _mkdir (declared in direct.h, not the fcntl.h/io.h this file already pulls in)
 * takes only the path, since Windows has no equivalent of the POSIX mode bits at
 * create time. The mode argument is accepted and ignored rather than dropped at
 * every call site, so test code written against the POSIX signature needs no
 * platform branch of its own. */
#include <direct.h>
static inline int k3_win_mkdir(const char *path, int mode)
{
    (void)mode;
    wchar_t *wp = k3_win_widen(path);
    if (!wp) { errno = ENOMEM; return -1; }
    const int r = _wmkdir(wp);
    free(wp);
    return r;
}
#define mkdir(path, mode) k3_win_mkdir((path), (mode))

/* madvise(MADV_HUGEPAGE) is a Linux transparent-hugepage hint; every caller already
 * treats it as advisory ("failure is not an error" -- k3_trunk.c), so a no-op is the
 * correct port here, not a functional gap. The alternative, VirtualAlloc with
 * MEM_LARGE_PAGES, needs SeLockMemoryPrivilege and allocations sized to an exact
 * large-page multiple -- a real feature to build for a hint the code already
 * tolerates losing. */
#define madvise(addr, len, advice) ((void)0)
#define MADV_HUGEPAGE 0

/* 64-bit file offsets. The Windows C runtime's lseek and off_t are 32-bit, so a size or
 * offset past 2 GB -- every real checkpoint shard -- would silently wrap. _lseeki64 is
 * the same call with a 64-bit offset. Call sites hold offsets in int64_t, never off_t,
 * for the same reason (caught by test_model_stream's 2^60-offset short-read check). */
#define lseek(fd, off, whence) _lseeki64((fd), (long long)(off), (whence))

/* O_DIRECT already forced open()-time behavior above; nothing left to do post-open. */
static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#else   /* Linux and friends: O_DIRECT on open() already did it */

static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#endif

/* Linux and Darwin: posix_memalign's contract already makes plain free() safe. */
#ifndef k3_aligned_free
#define k3_aligned_free(p) free(p)
#endif

#endif /* K3_PORTABLE_IO_H */
