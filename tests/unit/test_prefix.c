/* test_prefix.c - the prefix record must say "all of it" or "none of it", never a lie.
 *
 * The record decides whether a REPL turn skips its prefill. Every wrong answer here is
 * silent at the engine level: reusing a state built from different tokens produces a
 * fluent reply to a conversation that did not happen. So each rule the header states
 * is pinned by a case that would pass a plausible wrong implementation:
 *   - a longest-common-prefix implementation passes every "exact continuation" case
 *     and fails "diverges at the last recorded token" (it would return 3, not 0);
 *   - a counter-based implementation cannot tell a dropped record from a full one;
 *   - a grow that forgets to keep the fed ids makes reuse impossible on turn 2.
 *
 * usage: test_prefix
 */
#include <stdio.h>
#include <string.h>

#include "k3_prefix.h"

static int g_fail = 0;

static void ck(int ok, const char *what)
{
    printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_fail++;
}

int main(void)
{
    K3Prefix p; memset(&p, 0, sizeof p);
    const int turn1[] = { 7, 8, 9, 10, 11 };
    const int turn2[] = { 7, 8, 9, 10, 11, 12, 13 };      /* exact continuation */
    const int other[] = { 7, 8, 9, 10, 99, 12, 13 };      /* diverges at the LAST fed id */
    const int shorter[] = { 7, 8, 9 };                    /* a prefix of the record */
    const int same[] = { 7, 8, 9, 10, 11 };               /* equal to the record */

    printf("prefix record\n");

    /* ---- an empty or unallocated record never reuses ---- */
    ck(k3_prefix_reuse(&p, turn2, 7) == 0, "unallocated record reuses nothing");
    ck(k3_prefix_alloc(&p, 8) == 1, "alloc 8 positions");
    ck(k3_prefix_reuse(&p, turn2, 7) == 0, "empty record reuses nothing");

    /* ---- record as fed: the prompt in one chunk, then one token at a time ---- */
    k3_prefix_record(&p, turn1, 0, 3);          /* prefill 3 */
    k3_prefix_record(&p, turn1 + 3, 3, 1);      /* decode step */
    k3_prefix_record(&p, turn1 + 4, 4, 1);      /* decode step */
    ck(p.len == 5, "five positions recorded across three feeds");

    /* ---- the contract: all of the record, or none of it ---- */
    ck(k3_prefix_reuse(&p, turn2, 7) == 5, "exact continuation reuses all 5");
    ck(k3_prefix_reuse(&p, other, 7) == 0, "divergence at position 4 reuses NONE (not 4)");
    ck(k3_prefix_reuse(&p, shorter, 3) == 0, "a prefix of the record cannot be rewound to");
    ck(k3_prefix_reuse(&p, same, 5) == 0, "an equal prompt needs at least one new token");

    /* ---- taint: a state the ids cannot describe is never reused ---- */
    k3_prefix_taint(&p);
    ck(k3_prefix_reuse(&p, turn2, 7) == 0, "tainted state reuses nothing");
    k3_prefix_clear(&p);
    ck(p.len == 0 && !p.tainted, "clear drops both the record and the taint");

    /* ---- grow keeps the fed ids, so a longer prompt on turn 2 can still reuse ---- */
    k3_prefix_record(&p, turn1, 0, 5);
    ck(k3_prefix_grow(&p, 64, 5) == 1 && p.cap == 64 && p.len == 5,
       "grow to 64 keeps 5 positions");
    ck(k3_prefix_reuse(&p, turn2, 7) == 5, "reuse still fires after growth");
    ck(k3_prefix_grow(&p, 64, 3) == 1 && p.len == 3,
       "grow keeping fewer positions shortens the record to what was kept");
    ck(k3_prefix_reuse(&p, turn2, 7) == 3, "a shortened record reuses only what it covers");

    /* ---- a write the record cannot hold drops it rather than truncating it ---- */
    ck(k3_prefix_alloc(&p, 4) == 1, "realloc to 4 positions");
    k3_prefix_record(&p, turn1, 0, 5);
    ck(p.len == 0, "5 tokens into a 4-slot record leave it EMPTY, not 4 long");
    k3_prefix_record(&p, turn1, 0, 2);
    k3_prefix_record(&p, turn1 + 3, 3, 1);      /* skips position 2 */
    ck(p.len == 0, "a feed that skips a position drops the record");

    /* ---- re-recording an earlier position rewrites it (restart after divergence) ---- */
    ck(k3_prefix_alloc(&p, 8) == 1, "realloc to 8 positions");
    k3_prefix_record(&p, turn1, 0, 5);
    k3_prefix_clear(&p);
    k3_prefix_record(&p, other, 0, 7);
    ck(k3_prefix_reuse(&p, other, 7) == 0 && p.len == 7,
       "after a restart the record describes the new sequence");
    {
        const int other_cont[] = { 7, 8, 9, 10, 99, 12, 13, 14 };
        ck(k3_prefix_reuse(&p, other_cont, 8) == 7, "and its continuation reuses all 7");
        ck(k3_prefix_reuse(&p, turn2, 7) == 0, "while the old sequence no longer matches");
    }

    k3_prefix_free(&p);
    ck(p.fed == NULL && p.cap == 0 && p.len == 0, "free resets everything");

    printf("\n%s\n", g_fail ? "PREFIX RECORD TESTS FAILED" : "PREFIX RECORD TESTS PASSED");
    return g_fail ? 1 : 0;
}
