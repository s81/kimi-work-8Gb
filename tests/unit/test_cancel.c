/* test_cancel.c - Ctrl-C must stop the run at a safe point, not in the middle of one.
 *
 * WHY THIS FILE EXISTS
 *   A token on a streamed trunk takes up to two minutes. Before src/cli/k3_cancel.h
 *   existed, SIGINT took the default action at whatever instruction the process
 *   happened to be on: the forward in flight was lost, --save-state never ran, and the
 *   --out file was never written, so the only way to keep any of a long run was to
 *   let all of it finish. The contract now is: the first Ctrl-C sets a flag the
 *   decode loop checks once per step, the step in flight completes, nothing further
 *   starts, and everything a finished run writes is still written; a second Ctrl-C
 *   kills, in case the step in flight is the thing the user wants to escape.
 *
 *   The decode loop itself needs a checkpoint to run, so what is tested here is the
 *   contract the loop relies on: the flag, its stickiness, the placement of the check
 *   (after the step in flight, before the next one), and the second-signal escape.
 *
 * usage: test_cancel
 */
#define _POSIX_C_SOURCE 200809L

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "k3_cancel.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-44s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

int main(void)
{
    printf("cancel: first interrupt is a request, not a kill\n");
    ck(!k3_cancel_requested(), "nothing requested before any signal", NULL);
    k3_cancel_install();
    ck(!k3_cancel_requested(), "install starts clean", NULL);
    raise(SIGINT);
    ck(1, "process survived SIGINT", NULL);   /* reaching this line is the assertion */
    ck(k3_cancel_requested(), "request is visible after the signal", NULL);
    ck(k3_cancel_requested(), "and stays visible (sticky)", NULL);
    k3_cancel_install();
    ck(!k3_cancel_requested(), "re-install clears it", NULL);

    printf("cancel: the loop's safe point\n");
    {
        /* A model of the decode loop's check placement: the flag is consulted at the
         * top of each step. A signal arriving DURING step 3 must let step 3 finish and
         * must keep step 4 from starting, so exactly four steps complete. */
        k3_cancel_install();
        int completed = 0;
        for (int g = 0; g < 100; g++) {
            if (k3_cancel_requested()) break;
            if (g == 3) raise(SIGINT);       /* arrives mid-step */
            completed++;                      /* the step in flight completes */
        }
        char d[64]; snprintf(d, sizeof d, "%d steps completed, want 4", completed);
        ck(completed == 4, "step in flight completes, next does not start", d);
    }

    printf("cancel: second interrupt kills\n");
#ifndef _WIN32
    {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            k3_cancel_install();
            raise(SIGINT);                    /* first: a request */
            if (!k3_cancel_requested()) _exit(3);
            raise(SIGINT);                    /* second: default action */
            _exit(4);                         /* not reached if the contract holds */
        }
        int status = 0;
        ck(pid > 0 && waitpid(pid, &status, 0) == pid, "child ran", NULL);
        char d[64];
        if (WIFSIGNALED(status)) snprintf(d, sizeof d, "killed by signal %d", WTERMSIG(status));
        else snprintf(d, sizeof d, "exited %d (not killed)", WEXITSTATUS(status));
        ck(WIFSIGNALED(status) && WTERMSIG(status) == SIGINT, "child died of SIGINT", d);
    }
#else
    printf("  SKIP  %-44s %s\n", "child died of SIGINT", "no fork on Windows");
#endif

    printf("\n%s\n", g_fail ? "CANCEL TESTS FAILED" : "CANCEL TESTS PASSED");
    return g_fail ? 1 : 0;
}
