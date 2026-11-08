/* k3_cancel.h - stop at a safe point on the first Ctrl-C; die on the second.
 *
 * A token on a streamed trunk takes up to two minutes, and a run is worth keeping
 * even when it is cut short: --save-state carries the conversation into the next
 * turn, --out JSON holds every id decoded so far, and the cache and trunk reports are
 * the reason many runs exist at all. SIGINT's default action throws all of that away
 * at whatever instruction the process was on. The contract here:
 *
 *   first Ctrl-C    sets a flag. The decode loop reads it once per step, at the top,
 *                   so the forward pass in flight completes (its KV rows and recurrent
 *                   state are then exact) and no further step starts. Everything a
 *                   finished run writes is still written, and the process exits 5,
 *                   distinct from 0 so a harness cannot mistake a partial run for a
 *                   complete one, and distinct from the 130 a shell reports for a
 *                   process SIGINT actually killed.
 *   second Ctrl-C   the default action: the handler re-arms SIG_DFL before returning,
 *                   so a user who does not want to wait out the step in flight is not
 *                   trapped by the first one.
 *
 * The handler touches only a sig_atomic_t and write(2), both async-signal-safe; the
 * message is written from the handler because the next safe point can be two minutes
 * away and silence for that long reads as a hang, which invites the second Ctrl-C.
 *
 * signal() rather than sigaction(): MinGW has no sigaction, and the CRT maps a console
 * Ctrl-C onto SIGINT through signal(), so one spelling covers all three platforms.
 * glibc's signal() has BSD semantics (the handler stays installed, interrupted
 * syscalls restart), which is what the trunk reader thread's blocking reads need.
 *
 * Only the batch decode loop installs this. Ctrl-C during the minutes of weight
 * loading, before the loop, still kills immediately: there is nothing to save yet. */
#ifndef K3_CANCEL_H
#define K3_CANCEL_H

#include <signal.h>
#ifdef _WIN32
#include <io.h>
#define k3_cancel_write(fd, s, n) _write((fd), (s), (unsigned)(n))
#else
#include <unistd.h>
#define k3_cancel_write(fd, s, n) write((fd), (s), (n))
#endif

static volatile sig_atomic_t k3_cancel_flag = 0;

static void k3_cancel_on_sigint(int sig)
{
    (void)sig;
    k3_cancel_flag = 1;
    signal(SIGINT, SIG_DFL);   /* the second Ctrl-C is a kill */
    static const char msg[] =
        "\ninterrupt: the step in flight will complete, then state and results are "
        "written and the run exits 5.\n           Press Ctrl-C again to kill it now.\n";
    (void)!k3_cancel_write(2, msg, sizeof msg - 1);
}

/* Arms the handler and clears any earlier request. */
static void k3_cancel_install(void)
{
    k3_cancel_flag = 0;
    signal(SIGINT, k3_cancel_on_sigint);
}

static int k3_cancel_requested(void)
{
    return k3_cancel_flag != 0;
}

#endif /* K3_CANCEL_H */
