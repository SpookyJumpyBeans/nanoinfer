"""A local web server for the visualizer. Standard library only.

Loads the model once, then serves one page and a small JSON/event-stream API.
It binds to 127.0.0.1 and holds a single lock around the model, because two
generations running at once would split the CPU between them and make every
timing on the page meaningless.

Endpoints (all GET, so the page can use EventSource for the streaming ones):

    /api/info                      model shape
    /api/generate?...              stream: prompt, token..., done
    /api/attention?layer=&head=    one [seq, seq] matrix from the last run
    /api/resample?step=&...        re-rank one step's logits under new settings
    /api/race?prompt=&max=         stream: the same prompt with and without cache
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from nanoinfer.chat import ChatTemplate
from nanoinfer.generate import generate_stream
from nanoinfer.model import Qwen2
from nanoinfer.sampling import SamplingConfig
from nanoinfer.tokenizer import Tokenizer
from viz.trace import Trace, resample, traced_generate

STATIC = Path(__file__).resolve().parent / "static"

MAX_PROMPT_TOKENS = 96
MAX_GENERATE = 48
# The uncached path runs at a fraction of a token per second on a laptop CPU.
MAX_RACE = 12


class State:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.tokenizer = Tokenizer.from_model_dir(model_dir)
        self.model = Qwen2.from_model_dir(model_dir)
        self.chat = ChatTemplate.from_model_dir(model_dir)
        self.lock = threading.Lock()
        self.last: Trace | None = None
        self.eos = [
            i
            for i in (
                self.tokenizer.token_to_id("<|endoftext|>"),
                self.tokenizer.token_to_id("<|im_end|>"),
            )
            if i is not None
        ]

    def piece(self, token_id: int) -> str:
        """One token as text. Multi-byte characters split across tokens show
        as a replacement character here; the running text is decoded whole."""
        return self.tokenizer.decode([token_id], errors="replace")

    def encode(self, prompt: str, chat: bool) -> list[int]:
        text = self.chat.render([{"role": "user", "content": prompt}]) if chat else prompt
        return self.tokenizer.encode(text)


def _arg(q: dict, name: str, default, cast):
    try:
        return cast(q[name][0]) if name in q else default
    except (ValueError, TypeError):
        return default


def _sampling(q: dict) -> SamplingConfig:
    seed = _arg(q, "seed", None, int)
    return SamplingConfig(
        temperature=max(0.0, _arg(q, "temperature", 0.0, float)),
        top_k=max(0, _arg(q, "top_k", 0, int)),
        top_p=min(1.0, max(0.01, _arg(q, "top_p", 1.0, float))),
        seed=seed,
    )


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            pass

        # -- plumbing ---------------------------------------------------------

        def _json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _start_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _event(self, name: str, payload) -> None:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def _tokens(self, ids):
            return [{"id": int(i), "text": state.piece(int(i))} for i in ids]

        # -- routing ----------------------------------------------------------

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            routes = {
                "/": self.page,
                "/index.html": self.page,
                "/api/info": self.info,
                "/api/generate": self.generate,
                "/api/attention": self.attention,
                "/api/resample": self.resample_step,
                "/api/race": self.race,
            }
            handler = routes.get(url.path)
            if handler is None:
                self._json({"error": "not found"}, 404)
                return
            try:
                handler(q)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page navigated away mid-stream

        def page(self, q):
            body = (STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def info(self, q):
            c = state.model.config
            self._json(
                {
                    "model": state.model_dir.name,
                    "layers": c.num_hidden_layers,
                    "heads": c.num_attention_heads,
                    "kv_heads": c.num_key_value_heads,
                    "hidden": c.hidden_size,
                    "vocab": c.vocab_size,
                    "limits": {"generate": MAX_GENERATE, "race": MAX_RACE},
                }
            )

        # -- next-token view --------------------------------------------------

        def generate(self, q):
            prompt = _arg(q, "prompt", "", str)
            chat = _arg(q, "chat", "0", str) == "1"
            n = min(MAX_GENERATE, max(1, _arg(q, "max", 16, int)))
            config = _sampling(q)
            ids = state.encode(prompt, chat)
            self._start_stream()
            if not prompt.strip() or len(ids) > MAX_PROMPT_TOKENS:
                self._event("fail", {"message": f"Prompt must be 1 to {MAX_PROMPT_TOKENS} tokens."})
                return

            with state.lock:
                self._event("prompt", {"tokens": self._tokens(ids)})

                def on_step(trace, step):
                    text = state.tokenizer.decode(trace.generated_ids, skip_special_tokens=True)
                    self._event(
                        "token",
                        {
                            "index": len(trace.steps) - 1,
                            "id": step.token_id,
                            "text": state.piece(step.token_id),
                            "ms": round(step.ms, 1),
                            "running": text,
                            "candidates": [
                                {**c, "text": state.piece(c["id"])} for c in step.candidates
                            ],
                        },
                    )

                trace = traced_generate(
                    state.model, ids, n, stop_ids=state.eos, config=config, on_step=on_step
                )
                state.last = trace
                decode = [s.ms for s in trace.steps[1:]]
                self._event(
                    "done",
                    {
                        "prefill_ms": round(trace.prefill_ms, 1),
                        "decode_tps": round(1000 / np.mean(decode), 2) if decode else None,
                        "stop": trace.stop_reason,
                        "n_prompt": len(ids),
                    },
                )

        def resample_step(self, q):
            if state.last is None:
                self._json({"error": "generate something first"}, 409)
                return
            step = _arg(q, "step", 0, int)
            if not 0 <= step < len(state.last.logits):
                self._json({"error": "no such step"}, 400)
                return
            config = _sampling(q)
            cands = resample(state.last.logits[step], config, k=10)
            self._json(
                {
                    "step": step,
                    "chosen": state.last.steps[step].token_id,
                    "candidates": [{**c, "text": state.piece(c["id"])} for c in cands],
                }
            )

        # -- attention view ---------------------------------------------------

        def attention(self, q):
            trace = state.last
            if trace is None:
                self._json({"error": "generate something first"}, 409)
                return
            layer = min(len(trace.attention) - 1, max(0, _arg(q, "layer", 0, int)))
            head = _arg(q, "head", "avg", str)
            weights = trace.attention[layer]
            if head == "avg":
                matrix = weights.mean(axis=0)
            else:
                matrix = weights[min(weights.shape[0] - 1, max(0, int(head)))]
            self._json(
                {
                    "layer": layer,
                    "head": head,
                    "n_prompt": len(trace.prompt_ids),
                    "tokens": self._tokens(trace.all_ids),
                    "matrix": np.round(matrix, 4).tolist(),
                }
            )

        # -- cache race -------------------------------------------------------

        def race(self, q):
            prompt = _arg(q, "prompt", "", str)
            n = min(MAX_RACE, max(1, _arg(q, "max", 8, int)))
            ids = state.encode(prompt, False)
            self._start_stream()
            if not prompt.strip() or len(ids) > MAX_PROMPT_TOKENS:
                self._event("fail", {"message": f"Prompt must be 1 to {MAX_PROMPT_TOKENS} tokens."})
                return

            results = {}
            with state.lock:
                for lane, use_cache in (("cached", True), ("uncached", False)):
                    cache = state.model.new_cache(len(ids) + n) if use_cache else None
                    self._event("lane", {"lane": lane})
                    out, times = [], []
                    last = time.perf_counter()
                    # The engine's own loop, unmodified: this is the code path
                    # the equivalence tests cover.
                    for token_id in generate_stream(state.model, ids, n, cache=cache):
                        now = time.perf_counter()
                        times.append((now - last) * 1000)
                        last = now
                        out.append(token_id)
                        self._event(
                            "token",
                            {"lane": lane, "id": token_id, "text": state.piece(token_id),
                             "ms": round(times[-1], 1)},
                        )
                    results[lane] = {"ids": out, "total_ms": sum(times), "times": times}
                    self._event("lane_done", {"lane": lane, "total_ms": round(sum(times), 1)})

            c, u = results["cached"], results["uncached"]
            self._event(
                "done",
                {
                    "identical": c["ids"] == u["ids"],
                    "speedup_total": round(u["total_ms"] / c["total_ms"], 2),
                },
            )

    return Handler


def serve(model_dir: Path, port: int) -> None:
    print(f"loading {model_dir} ...", flush=True)
    started = time.perf_counter()
    state = State(model_dir)
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    print(f"open http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
