/* k3_prefix.h - reuse the state a previous REPL turn already built.
 *
 * WHY
 *   The chat REPL re-prefilled the whole transcript every turn (k3_run.c chat_run:
 *   "the first version re-prefills the full transcript for every REPL turn"). On a
 *   streamed trunk every replayed position re-reads its experts from disk, so the cost
 *   of turn N grew with turns 1..N-1, and the KV cache and KDA state that turn N-1 left
 *   behind were zeroed and rebuilt from the same tokens.
 *
 * THE IDEA
 *   After a turn the engine's state covers some number of positions. If the next
 *   prompt BEGINS with exactly the token ids those positions were built from, the
 *   state already IS the state at that position: skip the reset and prefill only the
 *   tail. No snapshot, no rewind. A prompt that diverges anywhere starts over.
 *
 * WHY A RECORD AND NOT A COUNTER
 *   It is tempting to derive the reusable length from the REPL's own bookkeeping
 *   ("prompt + generated - 1"). Whether the last sampled token was fed back, whether a
 *   turn stopped on <|end_of_msg|> or on --gen, whether a forward failed halfway: each
 *   changes the answer, and getting it wrong does not crash, it answers from a state
 *   that belongs to a different conversation. So the ids are recorded WHERE THEY ARE
 *   FED, and the record is the only description of the state anyone consults.
 *
 * INVARIANT: fed[0..len) are exactly the token ids the current state was built from,
 * in position order. Everything else follows from it.
 *
 * ALL OR NOTHING
 *   The KDA recurrent state is not positional: the matrix after N tokens cannot be
 *   truncated to the matrix after N-3. So a prompt that shares only part of the record
 *   reuses none of it. The one thing this must never do is answer from a state built
 *   from different tokens, and the equivalence gate (tests/unit/k3_model.c GATE 3b)
 *   requires the reused path to produce logits bit-identical to a full prefill.
 *
 * TAINT
 *   An input that its token ids do not describe (nothing in the text REPL today; an
 *   audio or image frame that reuses one placeholder id would be one) marks the state
 *   tainted, and a tainted state is never reused.
 */
#ifndef K3_PREFIX_H
#define K3_PREFIX_H

#include <stdlib.h>
#include <string.h>

typedef struct {
    int *fed;      /* token ids at positions 0..len-1 */
    int  len;      /* positions the state currently covers */
    int  cap;      /* allocated positions */
    int  tainted;  /* state consumed something token ids cannot describe */
} K3Prefix;

/* Size the record to the state it describes. Returns 0 on allocation failure, in which
 * case reuse is simply disabled: this is an optimisation and must never be the reason a
 * turn fails. */
static inline int k3_prefix_alloc(K3Prefix *p, int cap)
{
    if (!p) return 0;
    free(p->fed);
    p->fed = (cap > 0) ? (int *)calloc((size_t)cap, sizeof(int)) : NULL;
    p->cap = p->fed ? cap : 0;
    p->len = 0;
    p->tainted = 0;
    return p->fed != NULL;
}

/* Forget the state. Pair this with whatever zeroes the engine's own KV, ShortConv and
 * recurrent buffers, so the two can never disagree. */
static inline void k3_prefix_clear(K3Prefix *p)
{
    if (!p) return;
    p->len = 0;
    p->tainted = 0;
}

static inline void k3_prefix_free(K3Prefix *p)
{
    if (!p) return;
    free(p->fed);
    p->fed = NULL;
    p->cap = p->len = 0;
    p->tainted = 0;
}

/* Grow the record to `cap`, keeping the first `keep` positions.
 *
 * The REPL grows its KV cache when a longer prompt arrives. If it grows by COPYING the
 * cached positions the record must survive with them, or reuse can never fire in the
 * one case it exists for: a conversation whose prompt gets longer every turn. Returns 0
 * if the record could not be preserved, and leaves it EMPTY rather than stale, so the
 * caller then treats the state as unreusable. */
static inline int k3_prefix_grow(K3Prefix *p, int cap, int keep)
{
    if (!p || cap <= 0) return 0;
    int *grown = (int *)calloc((size_t)cap, sizeof(int));
    if (!grown) { k3_prefix_free(p); return 0; }
    if (keep > p->len) keep = p->len;
    if (keep > cap)    keep = cap;
    if (keep > 0 && p->fed) memcpy(grown, p->fed, (size_t)keep * sizeof(int));
    free(p->fed);
    p->fed = grown;
    p->cap = cap;
    p->len = keep > 0 ? keep : 0;
    return 1;
}

/* Record n tokens fed at absolute positions pos0..pos0+n-1. An out-of-range write drops
 * the record rather than truncating it: a partial record would claim coverage the state
 * does not have. A write that skips positions (pos0 > len) is the same lie and is
 * dropped for the same reason. */
static inline void k3_prefix_record(K3Prefix *p, const int *ids, int pos0, int n)
{
    if (!p || !p->fed || !ids || n <= 0 || pos0 < 0) return;
    if (pos0 + n > p->cap || pos0 > p->len) { p->len = 0; return; }
    memcpy(p->fed + pos0, ids, (size_t)n * sizeof(int));
    if (pos0 + n > p->len) p->len = pos0 + n;
}

/* Mark the state as holding something the ids do not describe (see TAINT). */
static inline void k3_prefix_taint(K3Prefix *p)
{
    if (p) p->tainted = 1;
}

/* How many leading tokens of this prompt the state already holds: either all `len` of
 * them, or none.
 *
 * Requires at least one NEW token (len < n). A prompt that is a prefix of, or equal to,
 * the recorded sequence would need the state REWOUND, which nothing here can do, and
 * prefilling zero tokens leaves the caller with no final hidden state to sample from. */
static inline int k3_prefix_reuse(const K3Prefix *p, const int *ids, int n)
{
    if (!p || !p->fed || !ids) return 0;
    if (p->tainted || p->len <= 0 || p->len >= n) return 0;
    if (memcmp(p->fed, ids, (size_t)p->len * sizeof(int)) != 0) return 0;
    return p->len;
}

#endif /* K3_PREFIX_H */
