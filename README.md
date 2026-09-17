# nanoinfer

An LLM inference engine written from scratch, to understand the layer underneath
the framework.

Target model: **Qwen2.5-0.5B-Instruct**, loaded from its raw `.safetensors` file.
No `transformers`, no `llama.cpp`, no `torch.generate()`. The tokenizer, the
forward pass, the KV cache, the sampler and the quantization are all implemented
here. `torch` and `transformers` appear only in `requirements-dev.txt`, used as
correctness oracles that the tests compare against and that the engine never
imports.

The engine's entire runtime dependency list is `numpy`.

## Status

| Phase | What | State |
|------:|------|-------|
| 1 | Parse safetensors, verify every tensor against the config | **done** |
| 2 | BPE tokenizer, exact round-trip vs. the reference on 10k strings | **done** |
| 3 | float32 forward pass, no cache, greedy; logits within 1e-3 of reference | **done** |
| 4 | KV cache; identical output, measured speedup | **done** |
| 5 | Temperature / top-k / top-p sampling with a seeded RNG | next |
| 6 | INT8 then INT4 quantization, perplexity delta at each level | |
| 7 | Rust port: CPU SIMD, then WebGPU compute kernels | |
| 8 | Benchmark against llama.cpp on identical hardware | |

Each phase is verified against a reference implementation before the next one
starts. Correctness first; a fast wrong answer teaches nothing.

## Hardware

All published numbers come from one machine:

| | |
|---|---|
| CPU | Intel i7-12700H, 14 cores / 20 threads, AVX2 |
| RAM | 16 GB |
| GPU | Intel Iris Xe (integrated) — no CUDA |

No discrete GPU, so phase 7 targets **WebGPU via `wgpu`** rather than CUDA or
Metal. That keeps the kernel work real and makes the result runnable on any
reviewer's machine, at the cost of peak throughput.

## Results

Numbers are appended to [`bench/results.jsonl`](bench/results.jsonl) by every
phase and never overwritten, so regressions stay visible.

```
python -m bench.benchmark --compare
```

### Phase 1 — weight loading

| Metric | Value |
|---|---|
| Parameters | 494,032,768 |
| Tensors | 290 |
| File size | 942.3 MiB (bfloat16) |
| `mmap` open | 4.6 ms |
| Widen all weights bf16 → f32 | 6.9 s |
| Peak RSS | 1,392 MB |

Mapping the file is effectively free because nothing is read. The six seconds
are spent paging 942 MiB off disk and doubling it into float32 — which is the
cost that phase 6 exists to attack.

### Phase 2 — tokenizer

| Metric | Value |
|---|---|
| Corpus | 10,000 generated strings, 240,166 characters |
| Token-ID agreement with reference | **10,000 / 10,000 exact** |
| Decode agreement | 10,000 / 10,000 exact |
| Throughput | 52,438 tok/s (pure Python) |
| Reference throughput | 116,038 tok/s (Rust `tokenizers`) |
| Gap | **2.2× slower** |
| Tokenizer load | 1.87 s |
| Peak RSS | 166 MB |

2.2× off a Rust implementation is closer than pure Python has any right to be,
and it is not cleverness in the merge loop — that loop is a deliberately
obvious O(n²) scan. It is the per-pre-token merge cache: real text repeats the
same pre-tokens relentlessly, so almost every lookup after the first thousand
strings is a dict hit.

The corpus averages 1.97 characters per token, which is far below the ~3.5–4 of
natural English. That is the corpus doing its job — a third of it is random
codepoints, control bytes and punctuation runs, which is where tokenizers break.

### Phase 3 — forward pass

Correctness, against `transformers` on the real 494M weights:

| Metric | Value |
|---|---|
| Max absolute logit difference | **6.2e-05** (gate: 1e-3) |
| Argmax agreement | every position, every prompt |
| Generated text vs. reference | token-for-token identical |
| Per-layer drift | flat at ~5e-4 across all 24 layers |

Speed, with no KV cache at all:

| Metric | Value |
|---|---|
| Model load | 2.9 s |
| Time to first token (5-token prompt) | 3,381 ms |
| Decode | **0.25 tok/s** |
| Last token vs. first | 1.26× slower |
| Peak RSS | 3,607 MB |

That 0.25 tok/s is not a disappointing result, it is the baseline. Every step
re-embeds and re-attends over the entire sequence, so the 24th token costs
noticeably more than the first — the 1.26× above is the quadratic cost becoming
visible over just 24 tokens. Phase 4 exists to delete that.

> **These numbers are superseded.** They are a mean over a single cold run, and
> phase 4 established that a single run on this machine can be off by an order
> of magnitude. Re-measured warm, with medians, the same uncached code does
> 3.46 tok/s -- see phase 4. The row is left here rather than edited, because
> the measurement really was taken and the lesson is what it cost.

### Phase 4 — KV cache

Both paths measured by alternating them in one process, three rounds each, same
prompt, same 24 generated tokens:

| | No cache | KV cache |
|---|---:|---:|
| Decode | 3.46 tok/s | **12.06 tok/s** |
| Median step | 288.7 ms | **82.9 ms** |
| Total wall time | 6.39 s | **2.06 s** |
| Time to first token | 200 ms | 152 ms |
| Arithmetic done | 396 positions | **28 positions** |

**Output is identical** — same token IDs, cached and uncached, on all 494M real
parameters across three prompts and every round. Logits agree to 1e-4 rather
than bitwise: the cache changes the shape of every matmul in the model, BLAS
sums the same products in a different order, and float addition is not
associative. Claiming bitwise equality would be claiming something false.

### The speedup is 3.5×, from 14× less arithmetic — and the gap is the point

Those two numbers not matching is the most useful thing phase 4 produced.

Decoding one token reads **every weight in the model** — 1.98 GB — to produce a
single 896-value vector. At 82.9 ms per token that is **24 GB/s effective**,
which is roughly this laptop's DRAM bandwidth. The decode step is not
arithmetic-bound, it is bandwidth-bound: the machine spends its time waiting
for weights, and most of its floating-point capacity sits idle.

So removing 93% of the arithmetic buys only 3.5×, because the arithmetic was
never the constraint. The uncached path was accidentally efficient — it pushed
17 positions through each matmul instead of one, which is exactly what makes a
GEMM worth its memory traffic.

Two consequences worth carrying forward:

- The only way past a bandwidth wall is to move fewer bytes. INT8 halves them,
  INT4 quarters them. That is phase 6, and it is now a bandwidth argument
  rather than a memory-footprint one.
- Serving engines batch many sequences through one weight read for the same
  reason. This engine is single-stream by design, so it leaves that on the
  table knowingly.

The speedup is asserted in tests as **work**, not wall time — positions pushed
through the model, which is exact and deterministic. Timings live in the
benchmark, where the machine that produced them is recorded alongside.

### Measuring on this machine

This laptop produces sporadic multi-second stalls under sustained load. Across
repeats of the *same* generation, totals have ranged from 2.97 s to 17.32 s,
and a single decode step has been 145× slower than the fastest in the same run.

I misread this at first. A matmul sweep showed a 15× cliff at exactly the
prompt length in use, which looked like a BLAS threading pathology worth
writing up. Re-running it twice put the spike at a different size each time —
there is no pathological size, only a noisy machine. The finding was an
artifact, and two more runs were cheaper than publishing it.

What the benchmark does about it:

- **Medians and minima, never means.** One stall makes a mean meaningless.
- **A/B by alternation.** Separate runs are not comparable when throughput
  drifts by more than the effect being measured, so `cache_ab` interleaves both
  paths in one process. Whatever the machine is doing, it does to both.
- **Spread is printed** next to every headline number, so the noise is visible
  instead of averaged into something that looks precise and is not.
- **BLAS thread settings are recorded** with each result. Leaving them unset
  does not mean single-threaded; it means the library chooses, based on the
  machine.

Perplexity and the llama.cpp comparison land in phases 6–8.

## Parameter budget

| | Parameters | Share |
|---|---:|---:|
| Embeddings | 136,134,656 | 27.6% |
| MLP | 313,786,368 | 63.5% |
| Attention | 44,067,840 | 8.9% |
| Norms | 43,904 | 0.0% |

Worth internalizing before optimizing anything: **the MLP is the model.** Two
thirds of the weights are in `gate_proj` / `up_proj` / `down_proj`, and
attention is under a tenth. Any optimization effort that starts with attention
is starting in the wrong place for this model size.

## What the weight file actually is

`.safetensors` is four parts and no compression:

```
+--------+---------------------+------------------------------------+
| 8 byte | N bytes             | rest of file                       |
| u64 LE | UTF-8 JSON header   | raw tensor bytes, back to back     |
| = N    |                     |                                    |
+--------+---------------------+------------------------------------+
```

For this model N is 32,280 bytes of JSON, so the tensor data starts at byte
32,288. The header maps each tensor name to its dtype, its shape, and a
`data_offsets: [begin, end)` pair — and **those offsets are relative to the
start of the data buffer, not the file.** Treating them as absolute is the
first bug everyone writes; `tests/test_safetensors.py` pins it.

The reader ([`nanoinfer/safetensors.py`](nanoinfer/safetensors.py)) memory-maps
the file and hands out zero-copy read-only views, so opening a model is
microseconds regardless of size and the OS pages weights in as they are touched.

### bfloat16 has no NumPy dtype

The weights are stored as bf16, which NumPy cannot represent. This is not a
problem, because bf16 *is* float32 with the low 16 mantissa bits removed — same
8-bit exponent, same bias. So the reader holds the raw bits as `uint16` and
widens by shifting:

```python
(raw.astype(np.uint32) << 16).view(np.float32)
```

That is exact for every value, infinities and NaNs included, and it never
rounds, because it only appends zero bits. Holding the raw bits rather than
converting on load is what makes memory-mapping worthwhile. The reverse
direction rounds half-to-even, and the tests prove the round-trip is the
identity over all 65,536 finite bf16 values.

## The tokenizer

Byte-level BPE, implemented in five pieces that each get their own module and
their own tests:

| | |
|---|---|
| [`bytelevel.py`](nanoinfer/bytelevel.py) | the 256-byte → printable-codepoint alphabet |
| [`unicode_classes.py`](nanoinfer/unicode_classes.py) | `\p{L}`, `\p{N}` and `White_Space` built from `unicodedata` |
| [`pretokenize.py`](nanoinfer/pretokenize.py) | NFC + the split regex, translated from the model's own file |
| [`bpe.py`](nanoinfer/bpe.py) | the merge loop |
| [`tokenizer.py`](nanoinfer/tokenizer.py) | added tokens, encode, decode |

Nothing about Qwen is hardcoded. The split pattern, merge table, vocabulary and
added tokens all come out of `tokenizer.json`, and unsupported spec fields are
rejected at load time rather than ignored — so pointing this at a different
model either works or fails loudly, never silently.

### Three things that cost real time

**Python's `\s` is not Rust's `\s`.** The split pattern ships as a Rust
`regex` pattern. Python's `\s` matches the C0 separators U+001C–U+001F; the
Unicode `White_Space` property, which Rust uses, does not. Building on `\s`
gives a tokenizer that is correct on everything except inputs containing a file
separator — which no small test corpus contains. The `White_Space` set is
spelled out explicitly instead, and a test asserts the divergence is real
rather than imagined.

**BPE tie-breaking is load-bearing.** When two positions hold pairs of equal
merge rank, the leftmost must merge first. The reference gets this from a
`(rank, position)` priority queue. Choosing the rightmost produces different
tokens only on repeated-character runs, which is exactly the kind of bug that
survives a hand-written test list.

**Added tokens must be extracted before the pre-tokenizer.** If `<|im_start|>`
were fed through the split regex it would be shredded into `<`, `|`, `im`,
`_start`, `|`, `>` and BPE would encode it as ordinary text. Every chat prompt
would then be wrong — and still perfectly fluent.

### Verification

The gate is exact token-ID equality on 10,000 strings, checked before any model
code exists. The corpus ([`tests/corpus.py`](tests/corpus.py)) is generated
from one seed and weighted toward where byte-level BPE actually breaks rather
than toward realism: whitespace torture, random assigned codepoints across all
planes, combining marks, long repeated runs, and control-token lookalikes.

ChatML formatting is checked the same way — by rendering the model's real Jinja
`chat_template` with Jinja and requiring an exact string match, rather than
asserting against hand-written expected output.

```
python -m bench.benchmark tokenize    # throughput vs the Rust reference
```

## The forward pass

Twenty-four identical blocks, each with two residual branches:

```python
x = x + attention(rms_norm(x))
x = x + feed_forward(rms_norm(x))
```

The norm sits *inside* the residual branch, not around it. That is "pre-norm",
and it leaves a path from the embedding to the output that no normalization
ever touches — which is what makes deep transformers trainable, and at
inference means the residual stream accumulates rather than being rescaled 48
times.

| | |
|---|---|
| [`ops.py`](nanoinfer/ops.py) | RMSNorm, SiLU, softmax, GQA head expansion, causal mask |
| [`rope.py`](nanoinfer/rope.py) | rotary position embeddings |
| [`weights.py`](nanoinfer/weights.py) | typed weight loading, shape-checked against the config |
| [`attention.py`](nanoinfer/attention.py) | grouped-query self-attention |
| [`model.py`](nanoinfer/model.py) | SwiGLU, the block, the full pass |
| [`generate.py`](nanoinfer/generate.py) | the greedy decode loop |

### Every bug here is silent

Nothing in attention fails loudly. Wrong RoPE convention, mask off by one, KV
heads tiled instead of repeated, a reshape that splits the sequence across
heads — each produces correctly-shaped finite numbers and text that still reads
like English. So each is pinned by a test that compares against a *different*
implementation rather than a restatement of the same steps.

**RoPE has two incompatible conventions.** Half-split (GPT-NeoX, HuggingFace,
Qwen2) pairs component `i` with `i + d/2`; interleaved (GPT-J, llama.cpp's
layout) pairs `2i` with `2i+1`. Both produce plausible numbers from the same
weights. A model on the wrong one emits grammatical text that degrades with
distance instead of failing. So the interleaved rotation is *also* implemented,
purely so a test can assert the two genuinely differ and that the wrong one
does not blow up.

**Grouped-query expansion must repeat, not tile.** 14 query heads share 2 KV
heads, and KV head 0 must serve query heads 0–6, not 0, 2, 4… `np.tile` gives
the identical shape with every query head paired to the wrong key.

**The causal mask boundary is `j <= i`.** A token must attend to itself.

### Verification

The gate is every logit at every position within 1e-3 of the reference, on six
prompts covering English, code, digit runs, CJK and a ChatML turn. Observed
worst case is 6.2e-05, so it passes with nearly two orders of magnitude to
spare.

Divergence is located **per layer**, not just reported at the end — a drift
starting in layer 0 and one starting in layer 19 are different bugs. Measured
drift is flat at ~5e-4 across all 24 layers, so what accumulates is float32
round-off rather than a defect, and the test fails if it ever grows abruptly.

### Two things that cost real time

**`do_sample=False` is not greedy.** Qwen2.5 ships a `generation_config.json`
declaring `repetition_penalty: 1.1`, and `transformers` applies it inside
`generate()` regardless. The obvious baseline therefore quietly downweights
tokens already in the context. Ours and the reference agree for three tokens
and then part company — "It is the largest city" against "It was founded in 7".
Chasing that as an attention bug is days aimed at the wrong thing.

**`hidden_states[-1]` is after the final norm**, not the last layer's output.
Comparing a pre-norm state against it reports a divergence of ~150 from
perfectly correct code.

### There is a floor on "identical"

Asking for the last row's logits alone reshapes the output matmul from
`[5, 896] @ [896, 151936]` to `[1, 896] @ …`. BLAS picks different blocking,
which changes the order 896 products are summed in, and float addition is not
associative — so the results differ by ~1e-5 on logits reaching 19. Phase 4
will claim the KV cache reproduces the uncached path exactly; that claim has to
be made at this tolerance, not at zero.

## The KV cache

Keys and values depend only on their own token and its position. Token 3's key
is the same whether the sequence is 4 tokens long or 400 — so once computed it
never needs computing again. Phase 3 recomputed all of them at every step
anyway.

Storage is pre-allocated with a watermark rather than appended to: growing a
list and re-concatenating each step would reintroduce exactly the copying the
cache exists to remove. `extend()` writes one layer's new keys and hands back a
zero-copy view of everything so far.

### Two invariants that turn silent corruption into exceptions

`extend()` deliberately does **not** advance the length; `commit()` does that
once, after every layer has written. Splitting them catches both ways the cache
can be driven wrong:

- **A layer that skips the cache** would leave stale values in its slots and
  attend over them next step as though they were real. `commit()` refuses
  unless every layer has written, and names the ones that did not.
- **A layer that writes twice** would put two tokens in one slot. Rejected.

### Keys are stored after RoPE

A token's position never changes, so rotating once on insert is both correct
and cheaper than re-rotating the whole history each step. Rotating again on
read would apply the rotation twice to every cached key — fluent output with a
scrambled sense of order.

The matching trap is on the query side: `positions` must continue past the
cache. Defaulting a decode step to position 0 instead of `cache.length` rotates
every generated token as though it were the first. A test injects that exact
bug and asserts it produces finite, different numbers rather than an error,
because that is what makes it dangerous.

### Verified by equality, not plausibility

Every way of getting a cache wrong produces fluent text, so the gate is
equality of output: the same 8 tokens split across five chunking patterns — one
shot, one at a time, prompt-then-decode, halves, ragged — must all give what a
single uncached pass gives. Chunk shape is what exercises the mask offsets and
the position arithmetic, which is where a cache actually breaks.

Cache size is 24 KiB per token across all 24 layers. Grouped-query attention is
what makes that affordable: with 14 KV heads instead of 2 it would be 168 KiB
per token, and a full 32k context would want 5.5 GB.

## Two things about Qwen2.5 that will break a naive implementation

**Tied embeddings.** `config.json` sets `tie_word_embeddings: true`, so there is
no `lm_head.weight` tensor in the file at all. The output projection reuses the
input embedding matrix: final logits are `hidden @ embed_tokens.T`. Searching
for an `lm_head` and not finding one is the expected outcome.

**Grouped-query attention.** 14 query heads, but only 2 key-value heads — each
KV head is shared by 7 query heads. That shrinks the KV cache by 7× (24 KiB per
token at fp32 across all 24 layers, instead of 168 KiB), and it is where a wrong
`repeat` or `reshape` produces text that is fluent and subtly wrong.

Qwen2 also keeps biases on Q/K/V but not on the output projection, unlike most
Llama-style models.

## Layout

```
nanoinfer/
  safetensors.py      container parser, bf16 <-> f32
  config.py           hyperparameters + derived shapes, with invariants asserted
  metrics.py          timing and peak-RSS measurement, no dependencies
  bytelevel.py        the 256-byte printable alphabet
  unicode_classes.py  Unicode property classes, so no `regex` dependency
  pretokenize.py      NFC + the split regex
  bpe.py              the merge loop
  tokenizer.py        encode / decode / added tokens
  chat.py             ChatML prompt formatting
  ops.py              RMSNorm, SiLU, softmax, GQA expansion, causal mask
  rope.py             rotary position embeddings
  weights.py          typed, shape-checked weight loading
  attention.py        grouped-query self-attention
  model.py            SwiGLU, the block, the forward pass
  kvcache.py          pre-allocated key/value storage
  generate.py         greedy decoding, cached or not
tools/
  download_model.py   four HTTPS GETs, no huggingface_hub
  inspect_weights.py  phase 1: prove every tensor is understood
  generate.py         run the model from the command line
bench/
  benchmark.py        append-only measurement harness
  results.jsonl       every number ever recorded
tests/
  corpus.py           the seeded 10,000-string differential corpus
  tiny.py             a 4,000-parameter model built on the fly for tests
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Scripts -> bin on Unix

python -m tools.download_model                      # ~950 MB
python -m tools.inspect_weights models/Qwen2.5-0.5B-Instruct
python -m bench.benchmark load
python -m bench.benchmark tokenize
python -m bench.benchmark generate
python -m bench.benchmark cache_ab                 # cached vs uncached, interleaved
python -m bench.benchmark --compare
python -m pytest                                    # 624 tests
```

Run the model:

```bash
python -m tools.generate --prompt "The capital of France is" --max-tokens 20
python -m tools.generate --chat "Explain RoPE in one sentence."
```

`tools.inspect_weights` builds the complete expected tensor manifest from
`config.json` alone and diffs it against the file. It exits non-zero on any
missing, unexpected, or wrongly-shaped tensor. For this model it accounts for
all 290.

## License

MIT
