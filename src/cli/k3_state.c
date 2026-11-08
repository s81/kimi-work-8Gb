/* k3_state.c - see k3_state.h. */
#define _POSIX_C_SOURCE 200809L
#include "k3_portable_io.h"   /* fsync() and rename() shims for MinGW; see the header */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "k3_state.h"

void k3_state_fp(const K3Cfg *c, int32_t *fp)
{
    fp[0] = c->hidden;      fp[1] = c->n_layers;  fp[2]  = c->vocab;
    fp[3] = c->kda_heads;   fp[4] = c->kda_head_dim; fp[5] = c->conv_k;
    fp[6] = c->n_heads;     fp[7] = c->qk_nope;   fp[8]  = c->qk_rope;
    fp[9] = c->v_head;      fp[10] = c->n_experts; fp[11] = c->topk;
}

/* FNV-1a folded eight bytes at a time. Each step is xor with the next word then a
 * multiply by an odd constant, and multiplying by an odd number is a bijection mod
 * 2^64, so a change to any single word always changes the running value, and every
 * later step maps distinct values to distinct values. One damaged byte anywhere in the
 * payload is therefore always caught; this is corruption detection, not an integrity
 * proof against an adversary, and it does not claim to be. Eight bytes per step rather
 * than one because the payload is 0.63 GB of recurrent state plus the KV cache at full
 * scale, hashed once on save and once on load. */
uint64_t k3_state_hash(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, b + i, 8);
        h ^= w;
        h *= 0x100000001b3ULL;
    }
    for (; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int k3_state_peek(const char *path, K3StateHdr *hd)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const size_t got = fread(hd, 1, sizeof *hd, f);
    fclose(f);
    if (got != sizeof *hd || memcmp(hd->magic, K3_STATE_MAGIC, 4) != 0) {
        fprintf(stderr, "%s is not a k3 state file\n", path);
        return -1;
    }
    if (hd->version != K3_STATE_VER) {
        fprintf(stderr, "%s is state version %d, this build writes %d\n",
                path, hd->version, K3_STATE_VER);
        return -1;
    }
    return 0;
}

/* fread that folds what it read into the running hash. */
static size_t hread(void *dst, size_t sz, size_t n, FILE *f, uint64_t *h)
{
    const size_t got = fread(dst, sz, n, f);
    *h = k3_state_hash(*h, dst, got * sz);
    return got;
}

int k3_state_load(const char *path, const K3Cfg *c, const K3StateHdr *hd,
                  int *seq, float *ks, float *kvc, float *ropec,
                  int n_bound, int n_mla, int kv_cap)
{
    int32_t fp[12];
    k3_state_fp(c, fp);
    if (memcmp(fp, hd->fp, sizeof fp) != 0) {
        fprintf(stderr, "REFUSING: %s was written by a different model architecture.\n"
                        "  Restoring it would produce fluent, wrong output.\n", path);
        return -1;
    }
    if (hd->n_bound != n_bound || hd->n_mla != n_mla) {
        fprintf(stderr, "REFUSING: %s holds %d bound layers and %d MLA layers, "
                        "this run has %d and %d\n",
                path, hd->n_bound, hd->n_mla, n_bound, n_mla);
        return -1;
    }
    if (hd->cached > kv_cap) {
        fprintf(stderr, "REFUSING: %s holds %d positions, this run's KV cache is %d.\n"
                        "  Raise --gen or shorten the prompt.\n", path, hd->cached, kv_cap);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f, (long)sizeof *hd, SEEK_SET) != 0) { fclose(f); return -1; }

    int rc = 0;
    uint64_t h = K3_STATE_HASH_INIT;
    if (hread(seq, sizeof(int), (size_t)hd->nseq, f, &h) != (size_t)hd->nseq) rc = -1;
    if (!rc && hread(ks, sizeof(float), (size_t)hd->kper * n_bound, f, &h)
               != (size_t)hd->kper * (size_t)n_bound) rc = -1;
    /* Position-major inside each layer slice, so a differently-sized destination cache
     * is written slice by slice rather than as one block. */
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = kvc + (size_t)mi * kv_cap * hd->kvpp;
        const size_t n = (size_t)hd->cached * hd->kvpp;
        if (hread(dst, sizeof(float), n, f, &h) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = ropec + (size_t)mi * kv_cap * hd->ropepp;
        const size_t n = (size_t)hd->cached * hd->ropepp;
        if (hread(dst, sizeof(float), n, f, &h) != n) rc = -1;
    }
    fclose(f);
    if (rc) { fprintf(stderr, "%s is truncated\n", path); return rc; }
    /* Every read came back full, so the file has the right SHAPE. This is the check
     * that sees a wrong BYTE: a flipped bit in a recurrent matrix restores cleanly and
     * decodes fluently, and nothing after this point can tell. */
    if (h != hd->payload_hash) {
        fprintf(stderr, "REFUSING: %s does not hash to what its header says\n"
                        "  (header %016llx, contents %016llx). The payload is damaged;\n"
                        "  restoring it would produce fluent, wrong output.\n",
                path, (unsigned long long)hd->payload_hash, (unsigned long long)h);
        return -1;
    }
    return 0;
}

/* fwrite that folds what it wrote into the running hash. */
static size_t hwrite(const void *src, size_t sz, size_t n, FILE *f, uint64_t *h)
{
    const size_t put = fwrite(src, sz, n, f);
    *h = k3_state_hash(*h, src, put * sz);
    return put;
}

int k3_state_save(const char *path, const K3Cfg *c, const int *seq, int nseq,
                  const float *ks, const float *kvc, const float *ropec,
                  int n_bound, int n_mla, int kv_cap, int cached,
                  int64_t kper, int64_t kvpp, int64_t ropepp)
{
    /* Staged next to the destination and published by rename, the same way the chat
     * history writer does it (k3_chat_history_save). Opening the destination directly
     * truncated it first, so a crash or a full disk mid-save left a file that was too
     * short to load, and the previous good state was already gone. Now the destination
     * changes exactly once, from the old complete file to the new complete file. */
    char tmp[4096];
    if (snprintf(tmp, sizeof tmp, "%s.tmp.XXXXXX", path) >= (int)sizeof tmp) {
        fprintf(stderr, "%s: path too long\n", path);
        return -1;
    }
    const int fd = mkstemp(tmp);
    if (fd < 0) {
        fprintf(stderr, "cannot stage %s next to %s: %s\n", tmp, path, strerror(errno));
        return -1;
    }
#ifndef _WIN32
    /* mkstemp creates 0600. The file used to be created by fopen, i.e. 0666 less the
     * umask, and a saved state should stay as readable as before: give the temp file
     * that mode now, so the rename publishes it. Windows has no group/other bits. */
    {
        const mode_t um = umask(0);
        umask(um);
        if (fchmod(fd, 0666 & ~um) != 0) {
            fprintf(stderr, "cannot set the mode of %s: %s\n", tmp, strerror(errno));
            close(fd); unlink(tmp);
            return -1;
        }
    }
#endif
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(tmp); perror(path); return -1; }

    K3StateHdr hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, K3_STATE_MAGIC, 4);
    hd.version = K3_STATE_VER;
    k3_state_fp(c, hd.fp);
    hd.n_bound = n_bound; hd.n_mla = n_mla; hd.cached = cached; hd.nseq = nseq;
    hd.kper = kper; hd.kvpp = kvpp; hd.ropepp = ropepp;

    /* The hash is known only after the payload has been walked, and walking it twice
     * would double the memory traffic of a save, so the header goes out first with the
     * hash still zero and is rewritten in place at the end. The file is still the temp
     * file at that point; nothing observes the intermediate header. */
    int rc = 0;
    uint64_t h = K3_STATE_HASH_INIT;
    if (fwrite(&hd, sizeof hd, 1, f) != 1) rc = -1;
    if (!rc && hwrite(seq, sizeof(int), (size_t)nseq, f, &h) != (size_t)nseq) rc = -1;
    if (!rc && hwrite(ks, sizeof(float), (size_t)kper * n_bound, f, &h)
               != (size_t)kper * (size_t)n_bound) rc = -1;
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = kvc + (size_t)mi * kv_cap * kvpp;
        const size_t n = (size_t)cached * kvpp;
        if (hwrite(src, sizeof(float), n, f, &h) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = ropec + (size_t)mi * kv_cap * ropepp;
        const size_t n = (size_t)cached * ropepp;
        if (hwrite(src, sizeof(float), n, f, &h) != n) rc = -1;
    }
    hd.payload_hash = h;
    if (!rc && fseek(f, 0, SEEK_SET) != 0) rc = -1;
    if (!rc && fwrite(&hd, sizeof hd, 1, f) != 1) rc = -1;
    if (!rc && fflush(f) != 0) rc = -1;
    if (!rc && fsync(fd) != 0) rc = -1;
    if (fclose(f) != 0) rc = -1;
    if (!rc && rename(tmp, path) != 0) rc = -1;
    if (rc) {
        fprintf(stderr, "failed writing %s: %s\n", path, strerror(errno));
        unlink(tmp);
    }
    return rc;
}
