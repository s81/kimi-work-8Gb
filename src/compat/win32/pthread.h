/* pthread.h for TCC on Windows: the subset this engine uses (threads, mutexes, condition
 * variables), on native Windows primitives. Only on the include path of Windows builds
 * (see build.c); Linux uses the real pthreads.
 *
 *   pthread_t        HANDLE from CreateThread
 *   pthread_mutex_t  CRITICAL_SECTION
 *   pthread_cond_t   Windows condition variable (Vista+), resolved at run time because
 *                    the kernel32 import list TCC ships predates it
 *
 * Attributes are not supported: every call site in the engine passes NULL. */
#ifndef K3_COMPAT_PTHREAD_H
#define K3_COMPAT_PTHREAD_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <errno.h>

typedef HANDLE           pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef struct { void *ptr; } pthread_cond_t;   /* layout of CONDITION_VARIABLE */
typedef void             pthread_attr_t;
typedef void             pthread_mutexattr_t;
typedef void             pthread_condattr_t;

/* ------------------------------------------------------------------- threads -- */
typedef struct { void *(*fn)(void *); void *arg; } K3ThreadStart;

static DWORD WINAPI k3_thread_trampoline(LPVOID p)
{
    K3ThreadStart s = *(K3ThreadStart *)p;
    free(p);
    s.fn(s.arg);
    return 0;
}

static int pthread_create(pthread_t *t, const pthread_attr_t *attr,
                          void *(*fn)(void *), void *arg)
{
    (void)attr;
    K3ThreadStart *s = (K3ThreadStart *)malloc(sizeof *s);
    if (!s) return ENOMEM;
    s->fn = fn; s->arg = arg;
    *t = CreateThread(NULL, 0, k3_thread_trampoline, s, 0, NULL);
    if (!*t) { free(s); return EAGAIN; }
    return 0;
}

static int pthread_join(pthread_t t, void **ret)
{
    if (ret) *ret = NULL;   /* no call site uses the thread's return value */
    if (WaitForSingleObject(t, INFINITE) != WAIT_OBJECT_0) return EINVAL;
    CloseHandle(t);
    return 0;
}

/* ------------------------------------------------------------------- mutexes -- */
static int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a)
{ (void)a; InitializeCriticalSection(m); return 0; }
static int pthread_mutex_destroy(pthread_mutex_t *m) { DeleteCriticalSection(m); return 0; }
static int pthread_mutex_lock(pthread_mutex_t *m)    { EnterCriticalSection(m);  return 0; }
static int pthread_mutex_unlock(pthread_mutex_t *m)  { LeaveCriticalSection(m);  return 0; }

/* ------------------------------------------------------- condition variables -- */
typedef VOID (WINAPI *K3CvInit)(pthread_cond_t *);
typedef BOOL (WINAPI *K3CvSleep)(pthread_cond_t *, CRITICAL_SECTION *, DWORD);
typedef VOID (WINAPI *K3CvWake)(pthread_cond_t *);

static struct { K3CvInit init; K3CvSleep sleep; K3CvWake wake, wake_all; int ready; } k3_cv;

/* Resolve the four entry points once. Racing first calls store the same addresses, so
 * the race is benign. A Windows without them (XP or older) cannot run the engine: say
 * so and stop rather than continue with threads that cannot wait. */
static void k3_cv_load(void)
{
    if (k3_cv.ready) return;
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    k3_cv.init     = (K3CvInit)(void *)GetProcAddress(k32, "InitializeConditionVariable");
    k3_cv.sleep    = (K3CvSleep)(void *)GetProcAddress(k32, "SleepConditionVariableCS");
    k3_cv.wake     = (K3CvWake)(void *)GetProcAddress(k32, "WakeConditionVariable");
    k3_cv.wake_all = (K3CvWake)(void *)GetProcAddress(k32, "WakeAllConditionVariable");
    if (!k3_cv.init || !k3_cv.sleep || !k3_cv.wake || !k3_cv.wake_all) {
        static const char msg[] = "k3: this Windows version has no condition variables "
                                  "(Windows Vista or later is required)\n";
        DWORD n;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), msg, sizeof msg - 1, &n, NULL);
        ExitProcess(70);
    }
    k3_cv.ready = 1;
}

static int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a)
{ (void)a; k3_cv_load(); k3_cv.init(c); return 0; }
static int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; } /* nothing to free */
static int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m)
{ return k3_cv.sleep(c, m, INFINITE) ? 0 : EINVAL; }
static int pthread_cond_signal(pthread_cond_t *c)    { k3_cv.wake(c);     return 0; }
static int pthread_cond_broadcast(pthread_cond_t *c) { k3_cv.wake_all(c); return 0; }

#endif /* K3_COMPAT_PTHREAD_H */
