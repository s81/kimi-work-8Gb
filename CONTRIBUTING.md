# Contributing

Real fixes and features from outside the project are credited in
[CONTRIBUTORS.md](CONTRIBUTORS.md).

## Before anything

```bash
tcc -run build.c test
```

That needs only the Tiny C Compiler (see the README's *Building* section), and it no model weights and must stay green. If it is red on `main`, that is
the bug worth fixing first.

`tools/*.py` (fixture generation, reference conformance, cache replay) needs `numpy`,
`torch`, and, for lint and the tokenizer parity check, `ruff` and `tiktoken`. Exact
versions are pinned in `pyproject.toml`; install them however you normally manage Python
dependencies, for example `pip install numpy==2.5.2 torch==2.13.0` and, for the dev
group, `pip install ruff==0.16.3 tiktoken==0.13.0`.

## The standard this codebase holds itself to

**A wrong answer that looks right is the worst failure mode here.** This engine can load
the wrong architecture, stream a corrupt expert, or mis-tokenize a prompt and still emit
fluent, plausible text. Nothing crashes. Several of the defensive checks in the code
exist because that class of failure is invisible without them.

So:

- **Fail loudly, never silently.** If a config field is missing, refuse, do not
  substitute a default. If an expert fails to load, count it and make the run fail; do
  not `continue` past it and produce a token that is missing part of its routed sum.
- **A test that cannot fail is not a test.** Make fixtures adversarial: if a plausible
  wrong implementation would still pass, change the input until it would not.
- **Comments explain why, not what.** Anywhere a plausible-looking implementation would
  be wrong, say so at that spot.

## Performance changes

The measured run-to-run spread on an identical configuration is **33%**. A single-sample
comparison proves nothing.

Report at least three runs per arm, and report all of them:

| arm | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|

Where you can, measure **counts instead of seconds**, bytes read per token, cache
evictions, pinned layers. They are immune to scheduling noise and make a much stronger
claim. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Style

- C99, 4-space indent, 90 columns. `clang-format -i` with `.clang-format` if you have it.
- Warnings are errors: `build.c` compiles with `-Wall -Werror`.
- **Never do arithmetic on a `void *`.** Weight pointers are `const void *`, and
  arithmetic on void strides by one byte, silently returning the wrong tensor. GCC's
  `-Wpointer-arith` used to reject it; TCC accepts it without a warning, so review is
  now the first line of defence (the oracle tests are the second). Cast to the element
  type first.
- Do not fuse multiply-adds by hand or rely on `fma()` for general doubles. TCC never
  contracts `a*b+c`, which keeps the scalar results bit-exact; on Windows `fma()` is
  defined as `a*b+c` (see `src/compat/k3_win32.h`), which is only exact because every
  call site multiplies float-derived values.
- **No libm transcendentals in the engine.** Use `k3_expf` and `k3_tanhf`
  (`src/core/k3_math.h`), not `expf`, `tanhf`, `exp` or `tanh`: the C library's
  versions round differently on Windows, on Linux and between glibc versions. Only
  `sqrt` is taken from libm, because IEEE 754 requires it to be correctly rounded. If
  you change `k3_math.h`, run the exhaustive check (about 15 minutes on 4 cores) and
  update the digest in `tests/unit/test_math.c` only if it passes:

  ```bash
  tcc -I src/core -run tools/check_math.c 4 > close.txt
  python3 tools/check_math_exact.py close.txt
  ```
- **Hot loops are written for TCC**, which does not inline, unroll or keep values in
  registers across statements. Follow "TCC-shaped dot products" in `src/core/k3_ops.c`:
  named accumulators, no function calls per element (`static inline` is still a call
  under TCC), activations widened to double once per call, and chained additions
  (`K3_CHAIN8`) so an accumulator stays in a register for a whole expression. Do not "simplify" them back
  into an indexed array or an `fma()` call; that costs about 5x and changes no result.
  Keep each accumulator partition and reduction tree exactly as it is: the AVX2 and NEON
  paths and the fixtures depend on that order. Check with `tcc -run build.c bench`, whose
  output digests must not change.

## Adding a kernel

1. Generate a fixture with `tools/emit_fixtures.py`, including its tolerance.
2. Add the case to `tests/unit/test_ops.c`.
3. Verify against the reference implementation in `tools/k3_ref.py`.
4. Check it still holds on a real layer: `SHARD_DIR=... tcc -run build.c test-all`.

## Commits and PRs

Present tense, imperative: `add chunked prefill`, not `added` or `adds`. Explain *why* in
the body; the diff already shows what. Fill in the PR template's verification section;
it is a checklist, not a formality.

## Releases

1. Open a PR that sets `K3_VERSION` in `src/cli/k3_run.c`, adds
   `.github/release-notes/v<version>.md`, and moves the CHANGELOG's Unreleased entries
   under the new version.
2. Merge it.
3. Only then create the release on GitHub (Releases, *Draft a new release*, new tag
   `v<version>` on `main`, empty description). `release.yml` builds and tests on Linux
   and Windows and attaches both zips, `SHA256SUMS.txt` and the notes file.

Its preflight job refuses a tag that does not match `K3_VERSION` or has no notes file,
so a release created before step 2 fails without publishing anything. If that happens,
delete the release and the tag, finish step 2, and create the release again.

## Scope

Read [docs/ROADMAP.md](docs/ROADMAP.md) first. It also lists what is deliberately *not*
planned and why, which may save you writing something that will be declined.
