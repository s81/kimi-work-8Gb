/* k3_par: persistent worker pool and parallel-for. See k3_par.h for the contract. */
#include "k3_par.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <pthread.h>   /* native on Linux; src/compat/win32/pthread.h on Windows */

#define K3_PAR_MAX_THREADS 1024

/* One parallel loop in flight. Lives on the caller's stack for the duration of the
 * call; every field is read and written under pool.mu. */
typedef struct K3ParJob {
    k3_par_fn fn;
    void     *ctx;
    int       n;          /* iterations                                     */
    int       chunk;      /* iterations handed out per claim                */
    int       next;       /* first unclaimed iteration                      */
    int       done;       /* iterations finished                            */
    struct K3ParJob *link;
} K3ParJob;

static struct {
    pthread_mutex_t mu;
    pthread_cond_t  work;      /* a job with unclaimed iterations was queued   */
    pthread_cond_t  finished;  /* some job's last iteration completed          */
    K3ParJob       *queue;     /* jobs that still have unclaimed iterations    */
    int             want;      /* requested thread count, 0 = default          */
    int             started;   /* worker threads running (excludes callers)    */
    int             ready;     /* mu/conds initialised                         */
} pool;

/* One-time initialisation. The first parallel loop always comes from a single thread
 * (the engine is serial until then), so a plain flag is enough. */
static void pool_init(void)
{
    if (pool.ready) return;
    pthread_mutex_init(&pool.mu, NULL);
    pthread_cond_init(&pool.work, NULL);
    pthread_cond_init(&pool.finished, NULL);
    pool.ready = 1;
}

/* Claim the next block of `job`. Caller holds mu. Returns 0 when nothing is left. */
static int claim_locked(K3ParJob *job, int *b, int *e)
{
    if (job->next >= job->n) return 0;
    *b = job->next;
    *e = job->next + job->chunk;
    if (*e > job->n) *e = job->n;
    job->next = *e;
    if (job->next >= job->n) {             /* fully claimed: unlink from the queue */
        K3ParJob **pp = &pool.queue;
        while (*pp && *pp != job) pp = &(*pp)->link;
        if (*pp) *pp = job->link;
    }
    return 1;
}

/* Run [b, e) of job, then account for it. Caller holds mu on entry and on return. */
static void run_block_locked(K3ParJob *job, int b, int e)
{
    pthread_mutex_unlock(&pool.mu);
    job->fn(job->ctx, b, e);
    pthread_mutex_lock(&pool.mu);
    job->done += e - b;
    if (job->done >= job->n) pthread_cond_broadcast(&pool.finished);
}

static void *worker_main(void *arg)
{
    (void)arg;
    pthread_mutex_lock(&pool.mu);
    for (;;) {
        while (!pool.queue) pthread_cond_wait(&pool.work, &pool.mu);
        K3ParJob *job = pool.queue;
        int b, e;
        if (claim_locked(job, &b, &e)) run_block_locked(job, b, e);
    }
    return NULL;   /* not reached: workers live for the life of the process */
}

/* Start workers up to threads-1 (the caller is the remaining thread). Caller holds mu. */
static void ensure_workers_locked(int threads)
{
    while (pool.started < threads - 1 && pool.started < K3_PAR_MAX_THREADS - 1) {
        pthread_t t;
        if (pthread_create(&t, NULL, worker_main, NULL) != 0) {
            /* Not fatal: fewer helpers only means less speed. Say so once. */
            if (pool.started == 0)
                fprintf(stderr, "k3_par: could not start worker threads; running serially\n");
            break;
        }
        pool.started++;
    }
}

static void par_run(int n, int chunk, k3_par_fn fn, void *ctx)
{
    if (n <= 0) return;
    const int threads = k3_par_threads();
    if (threads <= 1 || n == 1) { fn(ctx, 0, n); return; }

    pool_init();
    K3ParJob job;
    memset(&job, 0, sizeof job);
    job.fn = fn; job.ctx = ctx; job.n = n; job.chunk = chunk < 1 ? 1 : chunk;

    pthread_mutex_lock(&pool.mu);
    ensure_workers_locked(threads);
    job.link = pool.queue;                 /* newest first: a nested loop is served   */
    pool.queue = &job;                     /* before the outer one it is blocking     */
    pthread_cond_broadcast(&pool.work);

    int b, e;
    while (claim_locked(&job, &b, &e)) run_block_locked(&job, b, e);
    while (job.done < job.n) pthread_cond_wait(&pool.finished, &pool.mu);
    pthread_mutex_unlock(&pool.mu);
}

void k3_par_for(int n, int serial_max, k3_par_fn fn, void *ctx)
{
    if (n <= 0) return;
    if (n <= serial_max) { fn(ctx, 0, n); return; }
    const int t = k3_par_threads();
    par_run(n, (n + t - 1) / t, fn, ctx);   /* one contiguous block per thread */
}

void k3_par_for_dynamic(int n, k3_par_fn fn, void *ctx)
{
    par_run(n, 1, fn, ctx);
}

/* --------------------------------------------------------------- thread count -- */

static int logical_cpus(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 0;
#endif
}

#ifdef _WIN32
/* GetLogicalProcessorInformation postdates the Windows import list TCC ships, so it is
 * looked up at run time. The record layout is the documented x64/x86 one: a
 * ULONG_PTR mask, a 32-bit relationship code, then a 16-byte union. */
typedef struct {
    ULONG_PTR mask;
    DWORD     relationship;      /* 0 = RelationProcessorCore */
    union { BYTE flags; ULONGLONG reserved[2]; } u;
} K3LogicalProcInfo;
typedef BOOL (WINAPI *K3GetLPI)(K3LogicalProcInfo *, DWORD *);
#endif

int k3_par_physical_cores(void)
{
#if defined(__linux__)
    /* A cpu is counted when it is the lowest-numbered member of its own sibling list,
     * which yields exactly one cpu per physical core. */
    int cores = 0;
    for (int cpu = 0; cpu < 4096; cpu++) {
        char path[128];
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE *f = fopen(path, "r");
        if (!f) {
            if (cpu == 0) return 0;   /* no topology at all */
            break;                    /* ran off the end of the online cpus */
        }
        int first = -1;
        if (fscanf(f, "%d", &first) == 1 && first == cpu) cores++;
        fclose(f);
    }
    return cores;
#elif defined(_WIN32)
    K3GetLPI get = (K3GetLPI)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"),
                                                    "GetLogicalProcessorInformation");
    if (!get) return 0;
    DWORD bytes = 0;
    get(NULL, &bytes);
    if (bytes == 0) return 0;
    K3LogicalProcInfo *info = (K3LogicalProcInfo *)malloc(bytes);
    if (!info) return 0;
    int cores = 0;
    if (get(info, &bytes)) {
        const DWORD n = bytes / (DWORD)sizeof *info;
        for (DWORD i = 0; i < n; i++)
            if (info[i].relationship == 0) cores++;
    }
    free(info);
    return cores;
#else
    return 0;
#endif
}

int k3_par_default_threads(void)
{
    const char *env = getenv("OMP_NUM_THREADS");
    if (env && *env) {
        const int n = atoi(env);
        if (n > 0) return n > K3_PAR_MAX_THREADS ? K3_PAR_MAX_THREADS : n;
    }
    /* Physical cores, not logical cpus: on an SMT part the extra threads migrate across
     * cores and cost more than they add (measured 7.41 s/token at 32 threads against
     * 5.63 at 16 on a 16-core/32-thread machine). */
    int n = k3_par_physical_cores();
    if (n <= 0) n = logical_cpus();
    if (n <= 0) n = 1;
    return n > K3_PAR_MAX_THREADS ? K3_PAR_MAX_THREADS : n;
}

void k3_par_set_threads(int n)
{
    pool.want = n > K3_PAR_MAX_THREADS ? K3_PAR_MAX_THREADS : (n > 0 ? n : 0);
}

int k3_par_threads(void)
{
    if (pool.want <= 0) pool.want = k3_par_default_threads();
    return pool.want;
}
