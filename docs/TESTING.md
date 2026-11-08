# Testing

Every test in `tcc -run build.c test` runs **without model weights**. The checkpoint is 1.56 TB; if
correctness depended on having it, correctness would not get checked.

```bash
tcc -run build.c test                        # everything below; peak RSS ~1.7 GB
SHARD_DIR=~/k3model tcc -run build.c test-all  # adds the checkpoint-dependent tests
```

## What each gate proves

**`test_ops`**, every kernel against reference values, at a tolerance declared in the
fixture manifest rather than hardcoded. Several fixtures are adversarial by construction:
the router fixture reorders its top-2 on 5 of 6 rows, so an implementation that ignores
the routing bias fails; the SiTU-GLU fixture drives the activation to its exact
analytic cap.

**`test_cache`**, the streaming expert cache: prefetch, eviction, and mixed batch/serial
access. Uses a synthetic shard of structurally faithful experts, a few KB.

**`test_trunk`**, the streaming trunk: ring-slot budget enforcement, one-slot
guard, async prefetch, slot-isolation under concurrency, ring wrap-around, and
truncated-read failure isolation. Uses a synthetic 3-layer trunk fixture of a few
KB that is generated inline, so no checkpoint is required. The one-slot guard check
fails against any build that starts the reader thread unconditionally, which is the
condition the real model exhibited as silent token corruption: with one ring slot
the reader would write layer L+1 over layer L while the caller was still computing
on it, producing fluent but wrong tokens with no diagnostic.

It also asserts a **byte** bound rather than only a correctness one: a pinned layer is read
once for the whole run and a streamed layer once per pass, so anything above that is a slot
evicted between the prefetch that filled it and the walk that needed it. That failure
changes no token and shows up only as a device rate that flatters. It is checked per layer,
not just in aggregate, because the aggregate can say a run went over and never which
layers.

`t_sweep` then runs that same assertion across the *whole space* of pinned shapes — a
93-layer trunk, ring depths 1 to 4, budgets from a bare ring to fully resident. This is the
part worth copying elsewhere. Two hand-built fixtures had already been shaped, deliberately
and with reasoning attached, to expose this bug; both passed a build that a real run showed
was still reading 13.7% too much, and two further explanations were argued from the pinned
set's shape without evidence. The sweep failed 40 of 100 shapes on that same build and
named the worst one. **A fixture proves an arrangement is handled; only enumeration proves
the arrangement that escapes is not there.**

**`test_st`**, the safetensors reader: dtype widening, offsets, tail bytes, escaped
tensor names, and a tensor deliberately containing non-finite values.

**`test_model_stream`**, the ultra-low-memory model-table reader. It gathers one BF16
embedding row and projects through the lm_head in bounded aligned chunks, then requires
both results to be bit-identical to the resident kernels. It also injects a corrupt file
offset and requires the resulting short read to fail without being counted as valid I/O.

**`test_cfg`**, the config reader against the fixture layout, plus three malformed configs
in `tests/fixtures/cfg/` it must **refuse**: `no_layermap.json` (no `full_attn_layers` at
all), `bad_layer_index.json` (a one-based index outside `1..n_layers`), and
`bad_topk.json` (a top-k above `K3_MAX_TOPK`). Each is the working config with exactly one
field mutated, so a rejection can only come from that field. This matters more than it
looks: a config reader that substitutes defaults for missing fields produces a model that
loads, runs, and is architecturally wrong, with nothing to indicate it.

**`scale_test`**, the same kernels at the real released dimensions: 7168 wide, 93 layers,
96 heads, 896 experts. It checks the layer map really is 69 KDA + 24 MLA, that every
scratch-sizing helper returns a sane value at full width rather than only at fixture
width, and it allocates and runs one complete 7168-wide KDA layer. That last step is a
single **1.77 GB** allocation, which is the only real resource requirement anywhere in
`tcc -run build.c test`; on a machine too small for it the test says so and fails rather than
skipping.

**`test_tok`**, byte-exact roundtrip (encode then decode recovers the input exactly).
With `tools/tok_parity.py` it also compares token-for-token against the reference
tokenizer across CJK, emoji, ZWJ sequences, accents, contractions and whitespace runs.
This is the one gate that cannot run on a clean checkout: `tiktoken.model` has 163,584
entries and ships with the checkpoint, not with this repository. `tcc -run build.c test` reports it as
**NOT RUN** rather than passing it quietly.

You do not need the 1.56 TB checkpoint to run it. Four small files are enough, about
2.8 MB in total:

```bash
hf download moonshotai/Kimi-K3 \
    tiktoken.model tokenizer_config.json config.json tokenization_kimi.py \
    --local-dir ~/k3tok

TOK_FILES=~/k3tok tcc -run build.c test     # the roundtrip leg now runs
TOK_FILES=~/k3tok python3 tools/tok_parity.py bin/test_tok   # optional parity check, needs `pip install tiktoken`
./bin/test_cfg real ~/k3tok/config.json     # the released nested config
```

`tokenization_kimi.py` is required by `tools/tok_parity.py`, which reads the split regex
out of it rather than restating it; without that file the parity run stops before its
first case. The roundtrip leg in `tcc -run build.c test` needs only `tiktoken.model` and
`tokenizer_config.json`.

**`test_chat`**, the official K3 XTML text-chat contract without weights. It takes the
released tokenizer directory (`TOK_FILES`, the same one the tokenizer gate uses) and
verifies rendered bytes and token ids, assistant reasoning preservation, control-marker
injection resistance, the exact assistant turn terminator, strict JSONL
round-trip/rejection, `/reset`, and fixed-seed PCG32 sampling. Like `test_tok` it
prints NOT RUN when no `tiktoken.model` is present: the vocabulary ships with the
checkpoint, not with this repository, and is deliberately not copied into it.

**`k3_model`**, the end-to-end gate, on a tiny model whose tensor graph matches the released
architecture exactly:

- teacher forcing: every position matches the reference
- full-recompute state reuse: every logit is bit-identical with one recurrent-state slot
- greedy decode: every generated token matches
- incremental decode: same tokens as full recompute, with KV cache and carried
  recurrent state

All three must be *exact*. There is no tolerance on token identity.

## Checkpoint-dependent tests

Need `SHARD_DIR`:

- **`test_expert`**, reads an expert from released shards and compares against
  independently fetched bytes.
- **`test_real_layer`**, runs one released layer at full width against the reference.
- **`tools/conform_all.py`**, all 93 layers individually against the reference.

### Without the whole checkpoint: four shards, about 24 GB

`test-all` and the two verifiers only touch layer 1, and a short CLI run with `--layers 2`
needs just the first two layers plus the embedding and output head. The shard index says
where they live:

| Shard | Size | Holds |
|---|---|---|
| `model-00001-of-000096` | 2.3 GB | layer 0 (dense) |
| `model-00002-of-000096` | 17.0 GB | layer 1 (KDA + MoE, all 896 experts) |
| `model-00094-of-000096` | 4.7 GB | `embed_tokens`, final `norm`, `lm_head`, output attn-res |

Shards 95 and 96 hold only the vision tower and are not needed for text. Download those
three shards plus the small files (`config.json`, `model.safetensors.index.json`,
`tiktoken.model`, `tokenizer_config.json`, `tokenization_kimi.py`) into one directory, then:

```bash
SHARD_DIR=~/k3model TOK_FILES=~/k3model tcc -run build.c test-all
python3 tools/verify_real_layer.py ~/k3model real_layer.json   # needs torch, numpy
python3 tools/verify_expert.py expert_dump.json                # needs numpy
K3_TOK_FILES=~/k3model python3 tools/tok_parity.py bin/test_tok   # needs tiktoken
./bin/test_cfg real ~/k3model/config.json
./bin/k3 ~/k3model --tok ~/k3model --layers 2 --prompt "The capital of France is" --gen 4 --incremental
```

`k3_st` notes that 95 of the 96 declared shards are missing; that is expected. The
`--layers 2` output text is meaningless (two of 93 layers), it checks the pipeline from
prompt to token, not the answer.

**Last run: 2026-11-08, TCC build (v1.2.2 code), Linux x86-64, 4 cores, 15 GB RAM.**

| Check | Result |
|---|---|
| `test-all` | `ALL TESTS PASSED, including the ones on real weights`; fused MXFP4 vs dequantise-then-matmul on real expert 0 w1: max abs diff 0; all 896 experts contiguous with the expected geometry |
| `verify_real_layer.py` | `VERIFIED`; stage-by-stage against torch, relative error at most 8.4e-7 against a 5.1e-4 budget; identical expert sets for all 4 tokens |
| `verify_expert.py` | `VERIFIED END TO END`; packed bytes, scale bytes and dequantised floats identical |
| `tok_parity.py` | `tokenizer parity: 45/45 cases match` |
| `test-all` tokenizer and chat-template legs | roundtrip 109,096 bytes -> 33,358 ids -> same bytes; every chat-template check `ok` |
| `test_cfg real` | `REAL CONFIG: PASS` |
| CLI, `--layers 2` | 4 tokens generated, 2/2 layers completed, 0 expert drops, peak RSS 10.3 GB |

Not covered by this run: Windows, and the full 93-layer model (no answer quality check).

## Adding a test

Fixtures are generated by `tools/emit_fixtures.py` and carry their own tolerance in a
manifest. A test that cannot fail is not a test, make the fixture adversarial: if a
plausible wrong implementation would pass, change the input until it would not.
