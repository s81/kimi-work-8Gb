/* k3_par: the engine's own parallel-for, replacing OpenMP.
 *
 * WHY THIS EXISTS
 *   The engine is built with the Tiny C Compiler (TCC), which has no OpenMP. Without a
 *   replacement every `#pragma omp parallel for` silently becomes a serial loop and the
 *   engine uses one core. This is the smallest thing that restores the eight parallel
 *   loops: a persistent pool of worker threads and a parallel-for over [0, n).
 *
 * DETERMINISM
 *   Every loop handed to k3_par_for writes only its own outputs and never reduces across
 *   iterations. Which thread runs an iteration therefore cannot change a single bit of
 *   the result, and test_ops checks the kernels against the reference to the same
 *   tolerance as before (exact where it was exact).
 *
 * CONCURRENCY AND NESTING
 *   More than one thread may start a loop at once (the trunk's reader thread and the
 *   main thread both read layers), and a loop body may itself call k3_par_for. Each call
 *   is a job on a shared queue: the caller works through its own job's iterations while
 *   idle workers help, and returns once every iteration has finished. A caller never
 *   waits on work nobody can run, so neither case can deadlock.
 *
 * Threads come from k3_thread.h: pthreads on Linux, native Windows threads on Windows. */
#ifndef K3_PAR_H
#define K3_PAR_H

/* Loop body: run iterations [begin, end). ctx is the caller's private state. */
typedef void (*k3_par_fn)(void *ctx, int begin, int end);

/* Parallel for over [0, n), split into K3_PAR_BLOCKS contiguous blocks per thread that
 * threads claim in order (OpenMP's schedule(static) with a smaller chunk). Runs serially
 * when n <= serial_max, or with one thread. Which thread runs a block never changes a
 * result; see DETERMINISM. */
void k3_par_for(int n, int serial_max, k3_par_fn fn, void *ctx);

/* Parallel for over [0, n) handing out one iteration at a time (OpenMP's
 * schedule(dynamic, 1)): for I/O loops whose iterations take very different times. */
void k3_par_for_dynamic(int n, k3_par_fn fn, void *ctx);

/* Thread count, including the calling thread. Set it before the first parallel loop;
 * workers are started lazily on first use. n <= 0 restores the default. */
void k3_par_set_threads(int n);
int  k3_par_threads(void);

/* Default thread count: OMP_NUM_THREADS if set (kept so existing scripts and docs work),
 * else the physical core count, else the logical cpu count, else 1. */
int  k3_par_default_threads(void);

/* Physical cores, or 0 when the topology cannot be read. */
int  k3_par_physical_cores(void);

#endif /* K3_PAR_H */
