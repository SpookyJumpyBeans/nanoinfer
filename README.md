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
| 5 | Temperature / top-k / top-p sampling with a seeded RNG | **done** |
| 6 | INT8 then INT4 quantization, perplexity delta at each level | **done** |
| 7 | Rust port: CPU SIMD, then WebGPU compute kernels | **CPU done**, WebGPU not started |
| 8 | Benchmark against llama.cpp on identical hardware | next |

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

### Phase 5 — sampling

| Check | Result |
|---|---|
| Temperature vs. `TemperatureLogitsWarper` | exact, 7 values |
| Top-k vs. `TopKLogitsWarper` | exact, 200 random vectors incl. forced ties |
| Top-p vs. `TopPLogitsWarper` | exact indices, 400 random vectors |
| Fixed seed reproduces text | identical, real weights |
| Temperature 0 vs. phase 4 greedy | token-for-token identical |

```
seed 1: Once upon a time, there was a small town named Greenfield. The town was known
seed 1: Once upon a time, there was a small town named Greenfield. The town was known
seed 2: Once upon a time, in the year 2000, a group of friends
```

Per-token cost, against the 83 ms decode step from phase 4:

| | |
|---|---:|
| argmax (greedy) | 0.03 ms |
| top-k alone (k=20) | 0.94 ms |
| **top-p alone (p=0.8)** | **22.36 ms** |
| full sampler, temperature → k → p → draw | **7.69 ms** |

The full pipeline is three times faster than top-p on its own. Top-p has to sort
all 151,936 logits, but running top-k first leaves 151,916 of them identical
`-inf`, and the sort is far cheaper on that. The order sampling *has* to run in
turns out to be the fast one too. It is still ~9% of a decode step, and sorting
a whole vocabulary to find a cut near the top is the obvious thing for phase 7
to attack.

### Two formulations that are equal on paper and not in float32

Nucleus sampling is usually described as "sort descending, accumulate, stop when
the sum reaches p." Implementing it that way and differential-testing against
the reference produced disagreements — all of them ties landing on opposite
sides of the cut. For 100 equiprobable tokens at p=0.5 the descending sum
reaches `0.4999997913837433`, a hair under the boundary, so `>= p` keeps one
more token than the reference's `<= 1 - p` on the ascending sum.

Neither is more correct. The implementation follows the reference, because
`top_p=0.8` selecting the same nucleus here as everywhere else is the only thing
a user of that parameter can rely on.

A second, smaller one: sorting must be **stable**. `torch.sort` orders the tied
pair in `[1, 1, 0, 0]` as `[2, 3, 0, 1]`; numpy's default quicksort gives
`[3, 2, 1, 0]`. That changes which token counts as most likely, and a `p` small
enough to keep exactly one keeps the wrong one.

### Two things I measured wrong first

Worth recording because both were confident and both were wrong:

- I asserted a trained model emits no exact float ties, and rested a test on it.
  A real forward pass here produces **about 620 exact float32 collisions** among
  its 151,936 logits. They sit in the low-probability tail, so no realistic
  nucleus boundary falls inside a tied group — but the ties are real.
- Correcting that, I then claimed the 271 untrained embedding rows past the
  tokenizer's vocabulary carry *identical* logits. I had read rounded output.
  They span a band about 6e-3 wide. What actually matters held up: they carry
  under 1e-6 of the probability mass and any truncation removes them, and that
  is now asserted as a number rather than described in prose.

### Phase 6 — quantization

Perplexity on 508 tokens of held-out prose, against an fp32 baseline of 23.2690:

| Level | Footprint | Perplexity | Δ | Weight error |
|---|---:|---:|---:|---:|
| fp32 | 1.98 GB | 23.2690 | — | — |
| **INT8** | **0.90 GB** | **23.3078** | **+0.17%** | 0.85% |
| INT8 + embeddings | **0.50 GB** | 23.3133 | +0.19% | 0.85% |
| INT4, group 32 | 0.77 GB | 30.8070 | +32% | 9.9% |
| INT4, group 128 | 0.73 GB | 35.3849 | +52% | 12.3% |

**INT8 is effectively free** — a 0.17% perplexity cost for a 4× reduction on
the weights it touches. **Naive INT4 is not.** Round-to-nearest with no error
compensation costs half the model's quality, and that gap is precisely what
GPTQ and AWQ exist to close; the number above is what a later attempt would
have to beat.

### Quantization cannot buy speed here, and that is measurable

Phase 4 established that decode is bandwidth-bound: one token reads all 1.98 GB
of weights at ~24 GB/s. Fewer bytes should mean a faster step. It does not,
because NumPy has no integer GEMM:

| Operation (4864×896 weight, M=1) | |
|---|---:|
| fp32 matmul | 0.24 ms |
| int8 → dequantize → matmul | 20.03 ms (**82× slower**) |
| int8 → widen → matmul → scale | 7.38 ms (30× slower) |

Widening the int8 weights costs far more than the matmul it feeds. So phase 6
delivers the **footprint** reduction, which is real, and not the bandwidth win,
which needs a kernel that consumes integers directly — phase 7.

The perplexity figures above therefore come from *simulated* quantization:
weights are quantized and immediately dequantized, so the model computes on
exactly the values the integer format can represent while the matmuls stay
float32. That reproduces the arithmetic of true integer storage exactly, and is
the standard way to measure this (PyTorch calls it a quantize-dequantize pass).

### The measurement changed my mind about embeddings

I excluded the embedding matrix by default, reasoning that it doubles as the
output projection where errors land straight on the logits rather than being
averaged over a hidden dimension first.

That was wrong, and the sweep says so. Quantizing embeddings costs **0.02
percentage points** more and saves another **0.41 GB** — and without them the
untouched embedding matrix dominates what is left, so linear-only gets 1.98 GB
down to 0.90 while including embeddings reaches 0.50. The flag stays, but the
conservative default was the wrong call.

### A bug the sweep caught

`--group-size 32` and `--group-size 128` reported *byte-identical* perplexity.
`quantize_model` was calling the quantizer without forwarding the group size,
so every INT4 run silently used the default. Two configurations agreeing to the
fourth decimal is not a coincidence, and that is what surfaced it. There is now
a regression test asserting the two produce different stored sizes.

### Phase 7 — Rust kernels on the CPU

Phase 6 ended with a footprint win and no speed win: NumPy has no integer GEMM,
so holding int8 and widening per call is 30–80× *slower* than float32. The
kernel that consumes integers directly is what phase 7 builds.

The crate is a cdylib reached over ctypes; the engine is still NumPy and still
runs without it. Per layer, projections only, best of 50, alternating A/B:

| | numpy fp32 | rust 1 thread | rust pooled |
|---|---:|---:|---:|
| busy machine | 1.902 ms | 3.521 ms | **1.562 ms** |
| quiet machine | 0.967 ms | 1.814 ms | **0.762 ms** |
| | — | 0.53× | **1.22× / 1.27×** |

**1.2×, not 10×.** The Rust crate's own benchmark shows 10.28× — against the
crate's own scalar loop, which is a fair baseline for judging SIMD and a
meaningless one for judging the project. The engine never ran a scalar loop; it
ran OpenBLAS on twenty cores, and OpenBLAS is very good. Beating it by 22% with
hand-written AVX2 and a thread pool is the honest result.

Single-threaded loses outright at 0.53×. The SIMD is not what wins here.

### The bug that made parallelism look like a bad idea

Output rows are independent — row *j* reads its own slice of the weights and
writes one float — so splitting them needs no synchronisation at all. The first
implementation of that was **12× slower than one thread**: 1.19 ms against
0.094 ms on a 896×896 matvec.

The decomposition was never wrong. `std::thread::scope` creates real OS threads
at the scope and destroys them at its end, so every call paid to construct
twenty threads for ~90 µs of work. Moving to a pool whose workers already exist
— the only change — took the same code from 3.453 ms to 0.798 ms per layer.

Both kernels are still in the crate. `matvec_i8_spawn_per_call` is tested for
bitwise agreement with the good one, so the benchmark compares two correct
implementations rather than a good one against a broken one.

That is also why the crate has exactly one dependency. A pool whose workers can
borrow the caller's slices needs either unsafe lifetime laundering or rayon's
scope, and the no-dependency version I wrote to avoid it is the 4–7×-slower
column.

### Two projections are slower in Rust, and that is not noise

`k_proj` and `v_proj` measure **0.32×** — a loss — in both runs.

They are 128 rows, about 10 µs of work. The ctypes boundary has a floor of
roughly 8 µs per call, measured on a 1×1 matvec where there is nothing left but
the boundary itself. Below that floor, calling Rust costs more than the work,
and no kernel improvement can fix it; the call has to be made bigger, or not
made at all.

### What phase 7 did not do

The brief says "Rust port". This is not a port — the engine is still NumPy, and
only the seven linear projections cross into Rust. Attention, the norms, RoPE
and the LM head are untouched. Scaled to a token that is 45.7 → 37.5 ms of
projection time, which is a ceiling on what these kernels can move rather than
a token rate.

**WebGPU is not started.** Phase 7 was scoped as CPU SIMD first, then compute
shaders; the CPU half is done and measured, and the GPU half is not begun.

The llama.cpp comparison lands in phase 8.

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

## Sampling

| | |
|---|---|
| [`sampling.py`](nanoinfer/sampling.py) | temperature, top-k, top-p, seeded draw |

Order is temperature → top-k → top-p → softmax → draw, and it is not
interchangeable. Temperature changes the probabilities the two truncations
threshold against, so running top-p first would nucleus-sample a distribution
the model never produced and make `top_p` mean something different at every
temperature.

`temperature=0` short-circuits to the argmax rather than dividing by zero: it is
the limit of the distribution, it is how every other engine spells greedy, and
it makes the seed irrelevant. Greedy is kept as the reference the sampled paths
are checked against, since it is the only setting whose output can be compared
token for token with another engine.

`SamplingConfig.from_model_dir` reads what the model itself ships — Qwen2.5
declares temperature 0.7, top-k 20, top-p 0.8. `repetition_penalty` is
deliberately *not* read: it is a logits processor, this phase does not implement
it, and silently honouring a declared value the engine ignores would repeat the
phase 3 trap in reverse.

Draws use an inverse-CDF search rather than `rng.choice`, which is the
definition of a multinomial draw and does not require the probabilities to sum
to exactly 1.0 in floating point.

## Quantization

| | |
|---|---|
| [`quantization.py`](nanoinfer/quantization.py) | INT8 per-channel, INT4 group-wise, packing |
| [`perplexity.py`](nanoinfer/perplexity.py) | the instrument the quality cost is read off |
| [`data/heldout.txt`](data/heldout.txt) | held-out prose, committed for reproducibility |

Three choices, each measured rather than asserted:

**Per output channel, not per tensor.** One scale per tensor is dominated by
its largest outlier and crushes every other row's resolution. On the real
`gate_proj`: **4.86% error per-tensor against 0.85% per-channel.** The cost is
one float per row — 896 floats against 802,816 weights.

**Symmetric, no zero point,** so a quantized value is just `q * scale` and every
matmul stays a plain multiply. The range is `[-127, 127]` rather than the full
`[-128, 127]`: keeping it symmetric means a weight and its negation quantize to
exact opposites.

**Group-wise scales for INT4.** Fifteen levels is far too coarse to share one
scale across a whole row — a single large weight at the end would flatten
everything before it. Groups of 128 cost ~3% overhead; group 32 cuts weight
error from 12.3% to 9.9% and perplexity from +52% to +32%.

Rounding is half-away-from-zero, not `np.round`. Banker's rounding sends 0.5 to
0, which biases small magnitudes toward zero on a symmetric grid — and small
magnitudes are most of a weight matrix.

Norms and biases are never quantized: 43,904 parameters in total, under 0.01%
of the model, and they scale everything downstream of them.

### Measuring perplexity

Scored in non-overlapping chunks rather than a sliding window. A window gives a
lower, better-looking number at many times the compute; since every precision
level is scored identically, the delta — the actual result — is unaffected. The
first token of each chunk is not scored, because it has no context.

Log-probabilities come from a direct log-softmax rather than a log of the
softmax, which would round small probabilities to zero and return `-inf`. The
NLL accumulates in float64: summing a few thousand float32 terms moves the
fourth decimal, which is exactly where the INT8 delta lives.

The harness is tested against closed forms, not just against itself — a uniform
model must score exactly the vocabulary size, and a model certain of every next
token must score exactly 1.0.

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
  sampling.py         temperature, top-k, top-p, seeded draw
  quantization.py     INT8 per-channel, INT4 group-wise, packing
  perplexity.py       held-out scoring, the quality instrument
  kernels.py          ctypes bridge to the Rust kernels, optional
  generate.py         the decode loop, greedy or sampled
tools/
  download_model.py   four HTTPS GETs, no huggingface_hub
  inspect_weights.py  phase 1: prove every tensor is understood
  generate.py         run the model from the command line
  measure_quantization.py  perplexity and footprint per precision level
  bench_kernels.py    rust kernels against OpenBLAS, alternating A/B
rust/
  src/lib.rs          fp32 and int8 matvecs, AVX2, thread pool, C ABI
  src/bin/bench.rs    the same kernels against this crate's scalar loop
  tests/matvec.rs     arithmetic; the boundary is tested from Python
bench/
  benchmark.py        append-only measurement harness
  results.jsonl       every number ever recorded
  quantization.jsonl  perplexity and footprint per precision level
  kernels.jsonl       kernel timings, rust against numpy
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

# quantization sweep, one level per process
python -m tools.measure_quantization fp32
python -m tools.measure_quantization int8 --quantize-embeddings
python -m tools.measure_quantization int4 --group-size 32

# rust kernels; entirely optional, the engine runs without them
cd rust && cargo test --release && cargo run --release --bin bench
cd .. && python -m tools.bench_kernels

python -m pytest                                    # 719 tests
```

Run the model:

```bash
python -m tools.generate --prompt "The capital of France is" --max-tokens 20
python -m tools.generate --chat "Explain RoPE in one sentence."

# sampling; --model-defaults uses Qwen2.5's own T=0.7 k=20 p=0.8
python -m tools.generate --prompt "Once upon a time" --model-defaults --seed 1
python -m tools.generate --prompt "Once upon a time" --temperature 0.9 --top-p 0.95 --seed 42
```

`tools.inspect_weights` builds the complete expected tensor manifest from
`config.json` alone and diffs it against the file. It exits non-zero on any
missing, unexpected, or wrongly-shaped tensor. For this model it accounts for
all 290.

## License

MIT
