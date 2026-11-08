/* build.c -- the whole k3 build, in C, for the Tiny C Compiler.
 *
 *   tcc -run build.c              build the engine: bin/k3 (bin\k3.exe on Windows)
 *   tcc -run build.c test         build and run every test that needs no model weights
 *   tcc -run build.c test-all     the above, plus the tests that need real shards
 *                                 (set SHARD_DIR=/path/to/shards)
 *   tcc -run build.c bench        kernel microbenchmarks
 *   tcc -run build.c dist         package bin/k3 as dist/k3-<version>-<os>-x64.zip
 *   tcc -run build.c clean        remove bin/, build/ and dist/
 *
 * Nothing else is needed: no make, no shell scripts, no Python. The same file builds on
 * Linux and on Windows; it compiles itself for whichever machine runs it.
 *
 * Environment:
 *   CC         the compiler to call (default: tcc). On Windows, the full path to tcc.exe
 *              if it is not on PATH.
 *   TOK_FILES  directory holding tiktoken.model, for the optional tokenizer and chat
 *              template tests (default: ~/k3model). Skipped, and said so, when absent.
 *   SHARD_DIR  checkpoint shards, for test-all.
 *
 * Every compile uses -Wall -Werror: a TCC warning fails the build. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define OS_NAME   "windows"
#define EXE       ".exe"
#define SEP       "\\"
#define MKDIR(p)  _mkdir(p)
#else
#include <sys/wait.h>
#include <unistd.h>
#define OS_NAME   "linux"
#define EXE       ""
#define SEP       "/"
#define MKDIR(p)  mkdir((p), 0755)
#endif

/* ----------------------------------------------------------------- settings -- */

static const char *INCLUDES =
    "-Iinclude -Iinclude/k3 -Ithird_party -Isrc/core -Isrc/io -Isrc/cache "
    "-Isrc/model -Isrc/tokenizer -Isrc/chat -Isrc/cli";

#ifdef _WIN32
/* TCC's Windows runtime lacks some POSIX and C99 pieces; see src/compat/k3_win32.h. */
static const char *PLATFORM_FLAGS = "-include src/compat/k3_win32.h -Isrc/compat/win32";
static const char *LIBS = "";
#else
static const char *PLATFORM_FLAGS = "";
static const char *LIBS = "-lm -lpthread";
#endif

#define ENGINE "src/core/k3_ops.c src/core/k3_par.c src/io/k3_st.c src/io/k3_load.c " \
               "src/io/k3_trunk.c src/cache/k3_cache.c src/model/k3_bind.c"
#define OPS    "src/core/k3_ops.c src/core/k3_par.c"

typedef struct { const char *name; const char *sources; } Target;

static const Target CLI = {
    "k3", "src/cli/k3_run.c src/cli/k3_state.c src/chat/k3_chat.c src/chat/k3_sampler.c "
          ENGINE };

/* Tests that need no checkpoint. Each links only what it needs, so a failure points at
 * one subsystem. */
static const Target TESTS[] = {
    { "test_math",         "tests/unit/test_math.c" },
    { "test_ops",          "tests/unit/test_ops.c " OPS },
    { "test_cache",        "tests/unit/test_cache.c src/cache/k3_cache.c src/io/k3_load.c "
                           "src/io/k3_st.c " OPS },
    { "test_st",           "tests/unit/test_st.c src/io/k3_st.c" },
    { "test_model_stream", "tests/unit/test_model_stream.c src/model/k3_bind.c "
                           "src/io/k3_st.c " OPS },
    { "test_st_faults",    "tests/unit/test_st_faults.c src/io/k3_st.c" },
    { "test_json",         "tests/unit/test_json.c" },
    { "test_state",        "tests/unit/test_state.c src/cli/k3_state.c" },
    { "test_cfg",          "tests/unit/test_cfg.c " OPS },
    { "test_cancel",       "tests/unit/test_cancel.c" },
    { "test_sysmem",       "tests/unit/test_sysmem.c" },
    { "test_tok",          "tests/unit/test_tok.c" },
    { "test_chat",         "tests/unit/test_chat.c src/chat/k3_chat.c src/chat/k3_sampler.c" },
    { "test_sampler",      "tests/unit/test_sampler.c src/chat/k3_sampler.c" },
    { "test_prefix",       "tests/unit/test_prefix.c" },
    { "test_utf8",         "tests/unit/test_utf8.c" },
    { "scale_test",        "tests/unit/scale_test.c " OPS },
    { "test_trunk",        "tests/unit/test_trunk.c src/io/k3_trunk.c src/io/k3_st.c "
                           "src/model/k3_bind.c " OPS },
    { "k3_model",          "tests/unit/k3_model.c " OPS },
};

/* Tests that need real shards (test-all). */
static const Target WEIGHT_TESTS[] = {
    { "test_expert",     "tests/unit/test_expert.c src/io/k3_load.c src/io/k3_st.c " OPS },
    { "test_real_layer", "tests/unit/test_real_layer.c " ENGINE },
};

static const Target BENCH = { "bench_kernels", "benchmarks/bench_kernels.c " OPS };

#define NTESTS        (int)(sizeof TESTS / sizeof *TESTS)
#define NWEIGHT_TESTS (int)(sizeof WEIGHT_TESTS / sizeof *WEIGHT_TESTS)

/* ------------------------------------------------------------------ helpers -- */

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "build: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static const char *cc(void)
{
    const char *c = getenv("CC");
    return (c && *c) ? c : "tcc";
}

/* Exit status of a command run through the system shell. */
static int status_of(int rc)
{
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
    return -1;
#endif
}

static int run(const char *cmd)
{
    fflush(stdout);
    fflush(stderr);
    return status_of(system(cmd));
}

static int file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static void ensure_dir(const char *path)
{
    if (!file_exists(path) && MKDIR(path) != 0) die("cannot create directory %s", path);
}

/* bin/NAME as a command the system shell can run: backslashes on Windows, where cmd.exe
 * reads bin/x as "bin" with a switch "/x". */
static void exe_path(char *out, size_t cap, const char *name)
{
    snprintf(out, cap, "bin" SEP "%s" EXE, name);
}

static void set_env(const char *name, const char *value)
{
#ifdef _WIN32
    char buf[512];
    snprintf(buf, sizeof buf, "%s=%s", name, value ? value : "");
    _putenv(buf);
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

/* ------------------------------------------------------------------ compile -- */

static void compile(const Target *t)
{
    char out[256], cmd[4096];
    exe_path(out, sizeof out, t->name);
    snprintf(cmd, sizeof cmd, "%s -Wall -Werror %s %s %s -o %s %s",
             cc(), PLATFORM_FLAGS, INCLUDES, t->sources, out, LIBS);
    printf("  CC  %s\n", out);
    if (run(cmd) != 0) die("compiling %s failed:\n  %s", out, cmd);
}

static void build_cli(void)
{
    ensure_dir("bin");
    compile(&CLI);
}

static void build_tests(void)
{
    ensure_dir("bin");
    ensure_dir("build");
    for (int i = 0; i < NTESTS; i++) compile(&TESTS[i]);
}

/* ------------------------------------------------------------------- checks -- */

static int g_failed;

static void step(const char *title) { printf("== %s ==\n", title); fflush(stdout); }

/* Run bin/NAME ARGS and require exit status 0. */
static void must_pass(const char *name, const char *args)
{
    char exe[256], cmd[2048];
    exe_path(exe, sizeof exe, name);
    snprintf(cmd, sizeof cmd, "%s %s", exe, args);
    const int rc = run(cmd);
    if (rc != 0) die("%s exited %d:\n  %s", name, rc, cmd);
}

/* Run bin/k3 ARGS, capturing all output, and require exit status 2 with `why` (one of
 * several '|'-separated substrings) somewhere in the output. Checking the text as well
 * as the code matters: with a fake model directory the loader would also exit 2, and an
 * exit-code-only check would pass on the wrong error. */
static void must_refuse(const char *args, const char *why)
{
    char exe[256], cmd[2048];
    exe_path(exe, sizeof exe, "k3");
    const char *log = "build" SEP "refusal.txt";
    snprintf(cmd, sizeof cmd, "%s %s > %s 2>&1", exe, args, log);
    const int rc = run(cmd);

    char text[8192] = {0};
    FILE *f = fopen(log, "rb");
    if (f) { size_t n = fread(text, 1, sizeof text - 1, f); text[n] = '\0'; fclose(f); }
    remove(log);

    int found = 0;
    char alts[256];
    snprintf(alts, sizeof alts, "%s", why);
    for (char *a = strtok(alts, "|"); a; a = strtok(NULL, "|"))
        if (strstr(text, a)) { found = 1; break; }

    if (rc != 2 || !found) {
        fprintf(stderr, "  k3 %s\n  returned %d (expected 2), output:\n%s\n", args, rc, text);
        die(rc != 2 ? "wrong exit status" : "refused, but not for the expected reason (%s)",
            why);
    }
}

static const char *tok_files(void)
{
    static char buf[1024];
    const char *t = getenv("TOK_FILES");
    if (t && *t) return t;
#ifdef _WIN32
    const char *home = getenv("USERPROFILE");
#else
    const char *home = getenv("HOME");
#endif
    snprintf(buf, sizeof buf, "%s%sk3model", home ? home : ".", SEP);
    return buf;
}

static void test(void)
{
    build_cli();
    build_tests();
    printf("\n");

    step("ultra CLI contract");
    must_refuse("fake --ids 1 --preset ultra", "--trunk|trunk");
    must_refuse("fake --ids 1 --preset ultra --trunk fake --spec 2", "--spec|spec");

    step("stop-id contract");
    must_refuse("fake --ids 1 --stop-id abc", "--stop-id");
    must_refuse("fake --ids 1 --stop-id -1", "--stop-id");
    must_refuse("fake --ids 1 --stop-id 1 --stop-id 2 --stop-id 3 --stop-id 4 --stop-id 5 "
                "--stop-id 6 --stop-id 7 --stop-id 8 --stop-id 9", "--stop-id");
    printf("  3 malformed stop lists refused, each for the right reason\n");

    step("numeric CLI contract");
    {
        static const char *bad[] = { "--ids 4294967296", "--ids 9999999999999999999999",
                                     "--layers 1junk", "--layers 9999999999999999999999",
                                     "--threads 8x", "--threads 0" };
        for (int i = 0; i < 6; i++) {
            char args[256];
            snprintf(args, sizeof args,
                     "tests/fixtures/st --config tests/fixtures/cfg/flat.json %s", bad[i]);
            must_refuse(args, "--ids|--layers|--threads");
        }
        printf("  6 malformed numerics refused, each for the right reason\n");
    }

    step("expf and tanhf");    must_pass("test_math", "");
    step("op kernels");        must_pass("test_ops", "tests/fixtures/ops");
    step("streaming cache");   must_pass("test_cache", "tests/fixtures/cache");
    step("streaming cache, split reads");
    set_env("K3_EXPERT_CHUNK", "4096");
    must_pass("test_cache", "tests/fixtures/cache");
    set_env("K3_EXPERT_CHUNK", NULL);
    step("safetensors");
    must_pass("test_st", "tests/fixtures/st build/st_index.json plain.f32.2d plain.bf16.1d "
                         "tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32");
    step("model streaming");   must_pass("test_model_stream", "tests/fixtures/st");
    step("shard faults");      must_pass("test_st_faults", "tests/fixtures/st build/stfault");
    step("json refusals");     must_pass("test_json", "");
    step("state file");        must_pass("test_state", "build/state");
    step("config reader");     must_pass("test_cfg", "fixture tests/fixtures/ref_k3.json");
    step("interrupt contract"); must_pass("test_cancel", "");
    step("memory probe");      must_pass("test_sysmem", "");
    step("config refusals");
    must_pass("test_cfg", "reject tests/fixtures/cfg/no_layermap.json");
    must_pass("test_cfg", "reject tests/fixtures/cfg/bad_layer_index.json");
    must_pass("test_cfg", "reject tests/fixtures/cfg/bad_topk.json");

    char model[1100], args[1400];
    snprintf(model, sizeof model, "%s%stiktoken.model", tok_files(), SEP);
    const int have_tok = file_exists(model);

    step("tokenizer");
    if (have_tok) {
        snprintf(args, sizeof args, "\"%s\" roundtrip src/core/k3_ops.c", tok_files());
        must_pass("test_tok", args);
    } else {
        printf("  NOT RUN: no tiktoken.model at %s\n"
               "           the vocabulary ships with the checkpoint, not with this\n"
               "           repository. Set TOK_FILES=/path/to/k3model to run it.\n",
               tok_files());
    }

    step("chat option contract");
    {
        static const char *bad[] = { "--no-think", "--thinking-effort low",
                                     "--chat --thinking-effort medium",
                                     "--chat --no-think --thinking-effort low",
                                     "--ids 1 --top-k 4" };
        for (int i = 0; i < 5; i++) {
            char a[256];
            snprintf(a, sizeof a, "fake %s", bad[i]);
            must_refuse(a, "--no-think|--thinking-effort|--top-k");
        }
        printf("  5 misuses of thinking/sampling flags refused, each for the right reason\n");
    }

    step("chat template");
    if (have_tok) {
        snprintf(args, sizeof args, "\"%s\"", tok_files());
        must_pass("test_chat", args);
    } else {
        printf("  NOT RUN: no tiktoken.model at %s\n"
               "           the template is checked against the released tokenizer, which\n"
               "           ships with the checkpoint. Set TOK_FILES to run it.\n",
               tok_files());
    }

    step("sampler");           must_pass("test_sampler", "");
    step("chat prefix record"); must_pass("test_prefix", "");
    step("non-ASCII names");   must_pass("test_utf8", "build/utf8");
    step("real dimensions");   must_pass("scale_test", "");
    step("trunk streaming");   must_pass("test_trunk", "");
    step("full-model oracle"); must_pass("k3_model", "tests/fixtures");

    printf("\n");
    if (!have_tok) printf("NOTE: tokenizer parity did NOT run; see above. Everything else did.\n");
    printf("ALL WEIGHTLESS TESTS PASSED\n");
}

static void test_all(void)
{
    const char *shards = getenv("SHARD_DIR");
    if (!shards || !*shards) die("test-all needs SHARD_DIR=/path/to/shards");
    test();
    for (int i = 0; i < NWEIGHT_TESTS; i++) compile(&WEIGHT_TESTS[i]);
    char args[1400];
    snprintf(args, sizeof args, "\"%s\" 1 64", shards);
    must_pass("test_expert", args);
    snprintf(args, sizeof args, "\"%s\" 1 4 8", shards);
    must_pass("test_real_layer", args);
    printf("ALL TESTS PASSED, including the ones on real weights\n");
}

static void bench(void)
{
    ensure_dir("bin");
    compile(&BENCH);
    must_pass("bench_kernels", "");
}

/* --------------------------------------------------------------------- dist -- */
/* A store-only zip (no compression), written here so packaging needs no zip tool. The
 * binary is small, so compression would save little. Format: PKWARE APPNOTE 6.3.x,
 * local file headers, central directory, end record; all fields little-endian. */

static unsigned long crc_table[256];

static unsigned long crc32_of(const unsigned char *p, size_t n)
{
    if (!crc_table[1])
        for (unsigned long i = 0; i < 256; i++) {
            unsigned long c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320UL ^ (c >> 1) : c >> 1;
            crc_table[i] = c;
        }
    unsigned long c = 0xFFFFFFFFUL;
    for (size_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFUL;
}

static void put16(FILE *f, unsigned v) { fputc(v & 0xFF, f); fputc((v >> 8) & 0xFF, f); }
static void put32(FILE *f, unsigned long v) { put16(f, v & 0xFFFF); put16(f, (v >> 16) & 0xFFFF); }

static unsigned char *read_all(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot read %s", path);
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(len > 0 ? (size_t)len : 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) die("cannot read %s", path);
    fclose(f);
    *n = (size_t)len;
    return buf;
}

typedef struct { const char *src; char name[256]; unsigned long crc, size, offset; int exec; } ZipEntry;

static void write_zip(const char *zip_path, ZipEntry *e, int n)
{
    FILE *f = fopen(zip_path, "wb");
    if (!f) die("cannot write %s", zip_path);

    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    const unsigned dos_time = (unsigned)((tm->tm_hour << 11) | (tm->tm_min << 5) | (tm->tm_sec / 2));
    const unsigned dos_date = (unsigned)(((tm->tm_year - 80) << 9) | ((tm->tm_mon + 1) << 5) | tm->tm_mday);

    for (int i = 0; i < n; i++) {
        size_t len;
        unsigned char *data = read_all(e[i].src, &len);
        e[i].crc = crc32_of(data, len);
        e[i].size = (unsigned long)len;
        e[i].offset = (unsigned long)ftell(f);
        const unsigned nlen = (unsigned)strlen(e[i].name);
        put32(f, 0x04034b50UL); put16(f, 20); put16(f, 0); put16(f, 0);
        put16(f, dos_time); put16(f, dos_date);
        put32(f, e[i].crc); put32(f, e[i].size); put32(f, e[i].size);
        put16(f, nlen); put16(f, 0);
        fwrite(e[i].name, 1, nlen, f);
        fwrite(data, 1, len, f);
        free(data);
    }
    const unsigned long cd_start = (unsigned long)ftell(f);
    for (int i = 0; i < n; i++) {
        const unsigned nlen = (unsigned)strlen(e[i].name);
        /* Made by: Unix (3), so the external attributes can carry the exec bit. */
        put32(f, 0x02014b50UL); put16(f, (3 << 8) | 20); put16(f, 20); put16(f, 0); put16(f, 0);
        put16(f, dos_time); put16(f, dos_date);
        put32(f, e[i].crc); put32(f, e[i].size); put32(f, e[i].size);
        put16(f, nlen); put16(f, 0); put16(f, 0); put16(f, 0); put16(f, 0);
        put32(f, (unsigned long)(e[i].exec ? 0100755 : 0100644) << 16);
        put32(f, e[i].offset);
        fwrite(e[i].name, 1, nlen, f);
    }
    const unsigned long cd_size = (unsigned long)ftell(f) - cd_start;
    put32(f, 0x06054b50UL); put16(f, 0); put16(f, 0); put16(f, (unsigned)n); put16(f, (unsigned)n);
    put32(f, cd_size); put32(f, cd_start); put16(f, 0);
    if (fclose(f) != 0) die("cannot write %s", zip_path);
}

/* K3_VERSION from src/cli/k3_run.c, so the zip name follows the source. */
static void read_version(char *out, size_t cap)
{
    FILE *f = fopen("src/cli/k3_run.c", "r");
    char line[512];
    snprintf(out, cap, "dev");
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        char *p = strstr(line, "#define K3_VERSION \"");
        if (p) {
            p += strlen("#define K3_VERSION \"");
            char *q = strchr(p, '"');
            if (q) { *q = '\0'; snprintf(out, cap, "%s", p); }
            break;
        }
    }
    fclose(f);
}

static void dist(void)
{
    build_cli();
    ensure_dir("dist");
    char version[64], dir[128], zip[192], exe[256], readme[256];
    read_version(version, sizeof version);
    snprintf(dir, sizeof dir, "k3-%s-%s-x64", version, OS_NAME);
    snprintf(zip, sizeof zip, "dist" SEP "%s.zip", dir);
    exe_path(exe, sizeof exe, "k3");

    /* README inside the zip: what it is and how to check it. */
    snprintf(readme, sizeof readme, "build" SEP "README.txt");
    ensure_dir("build");
    FILE *r = fopen(readme, "wb");
    if (!r) die("cannot write %s", readme);
    fprintf(r,
        "k3 %s -- Kimi K3 inference engine, %s x86-64\n\n"
        "Built with the Tiny C Compiler (TCC) by build.c. No installer and no extra\n"
        "libraries are needed.\n\n"
        "Quick check:\n"
        "  k3 --help\n"
        "  k3 --list-presets\n\n"
        "Running the model needs the Kimi K3 weights, downloaded separately (about\n"
        "1.56 TB); see the project README. Prompts in any language can be passed\n"
        "with --prompt: on Windows k3 reads its command line as Unicode. For an exact,\n"
        "repeatable input use --prompt-file, which is read byte for byte.\n\n"
        "Source: https://github.com/s81/kimi-work-8Gb, from\n"
        "https://github.com/FareedKhan-dev/kimi-k3-in-c (Apache-2.0; see LICENSE, NOTICE).\n",
        version, OS_NAME);
    fclose(r);

    ZipEntry e[4];
    memset(e, 0, sizeof e);
    e[0].src = exe;      snprintf(e[0].name, sizeof e[0].name, "%s/k3" EXE, dir); e[0].exec = 1;
    e[1].src = "LICENSE"; snprintf(e[1].name, sizeof e[1].name, "%s/LICENSE", dir);
    e[2].src = "NOTICE";  snprintf(e[2].name, sizeof e[2].name, "%s/NOTICE", dir);
    e[3].src = readme;    snprintf(e[3].name, sizeof e[3].name, "%s/README.txt", dir);
    write_zip(zip, e, 4);
    printf("  ZIP %s\n", zip);
}

/* -------------------------------------------------------------------- clean -- */

static void remove_bin(const char *name)
{
    char p[256];
    exe_path(p, sizeof p, name);
    remove(p);
}

static void clean(void)
{
    remove_bin(CLI.name);
    remove_bin(BENCH.name);
    for (int i = 0; i < NTESTS; i++) remove_bin(TESTS[i].name);
    for (int i = 0; i < NWEIGHT_TESTS; i++) remove_bin(WEIGHT_TESTS[i].name);
    static const char *files[] = { "build/st_index.json", "build/refusal.txt",
                                   "build/README.txt", NULL };
    for (int i = 0; files[i]; i++) remove(files[i]);
    char version[64], zip[192];
    read_version(version, sizeof version);
    snprintf(zip, sizeof zip, "dist/k3-%s-%s-x64.zip", version, OS_NAME);
    remove(zip);
    /* Test scratch directories hold files the tests name themselves; remove what is
     * left only if empty, rather than guessing at their contents. */
    rmdir("build/stfault"); rmdir("build/state"); rmdir("build"); rmdir("bin"); rmdir("dist");
    printf("cleaned\n");
}

/* --------------------------------------------------------------------- main -- */

static void usage(void)
{
    printf("usage: tcc -run build.c [k3|test|test-all|bench|dist|clean]\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) { build_cli(); return 0; }
    for (int i = 1; i < argc; i++) {
        const char *t = argv[i];
        if      (!strcmp(t, "k3") || !strcmp(t, "all")) build_cli();
        else if (!strcmp(t, "test"))     test();
        else if (!strcmp(t, "test-all")) test_all();
        else if (!strcmp(t, "bench"))    bench();
        else if (!strcmp(t, "dist"))     dist();
        else if (!strcmp(t, "clean"))    clean();
        else if (!strcmp(t, "help") || !strcmp(t, "-h") || !strcmp(t, "--help")) usage();
        else { usage(); return 2; }
    }
    return g_failed ? 1 : 0;
}
