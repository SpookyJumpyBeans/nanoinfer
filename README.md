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
| 2 | BPE tokenizer, exact round-trip vs. the reference on 10k strings | next |
| 3 | float32 forward pass, no cache, greedy; logits within 1e-3 of reference | |
| 4 | KV cache; bit-identical output, measured speedup | |
| 5 | Temperature / top-k / top-p sampling with a seeded RNG | |
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

Tokens/sec, perplexity, and the llama.cpp comparison land in phases 3–8.

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
  safetensors.py   container parser, bf16 <-> f32
  config.py        hyperparameters + derived shapes, with invariants asserted
  metrics.py       timing and peak-RSS measurement, no dependencies
tools/
  download_model.py   four HTTPS GETs, no huggingface_hub
  inspect_weights.py  phase 1: prove every tensor is understood
bench/
  benchmark.py     append-only measurement harness
  results.jsonl    every number ever recorded
tests/
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Scripts -> bin on Unix

python -m tools.download_model                      # ~950 MB
python -m tools.inspect_weights models/Qwen2.5-0.5B-Instruct
python -m bench.benchmark load
python -m pytest
```

`tools.inspect_weights` builds the complete expected tensor manifest from
`config.json` alone and diffs it against the file. It exits non-zero on any
missing, unexpected, or wrongly-shaped tensor. For this model it accounts for
all 290.

## License

MIT
