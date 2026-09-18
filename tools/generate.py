"""Run the model. Text in, text out, streamed as it is produced.

    python -m tools.generate --prompt "The capital of France is" --max-tokens 20
    python -m tools.generate --chat "Explain RoPE in one sentence."

``--chat`` wraps the prompt in the ChatML layout the Instruct model was tuned
on. Without it the model is a plain text continuer, which is a genuinely
different thing and worth seeing the difference of.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoinfer.chat import ChatTemplate  # noqa: E402
from nanoinfer.sampling import Sampler, SamplingConfig  # noqa: E402
from nanoinfer.generate import greedy  # noqa: E402
from nanoinfer.model import Qwen2  # noqa: E402
from nanoinfer.tokenizer import Tokenizer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--chat", metavar="MESSAGE", help="wrap the prompt in ChatML")
    parser.add_argument("--system", help="system prompt, with --chat")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt as tokenized")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 (default) is greedy; 0.7 is Qwen2.5's own setting")
    parser.add_argument("--top-k", type=int, default=0, help="0 disables")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 disables")
    parser.add_argument("--seed", type=int, default=None,
                        help="makes sampled output reproducible")
    parser.add_argument("--model-defaults", action="store_true",
                        help="use the sampling settings the model ships")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the KV cache; much slower, kept for comparison")
    args = parser.parse_args(argv)

    print(f"loading {args.model}", file=sys.stderr)
    start = time.perf_counter()
    tokenizer = Tokenizer.from_model_dir(args.model)
    model = Qwen2.from_model_dir(args.model)
    print(f"loaded in {time.perf_counter() - start:.2f}s", file=sys.stderr)

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.chat})
        text = ChatTemplate.from_model_dir(args.model).render(messages)
        # The assistant turn ends with <|im_end|>; stopping there is what keeps
        # the model from cheerfully inventing the user's next message too.
        stop_ids = [tokenizer.token_to_id("<|im_end|>")]
    else:
        text = args.prompt
        stop_ids = [tokenizer.token_to_id("<|endoftext|>")]

    if args.model_defaults:
        config = SamplingConfig.from_model_dir(args.model, seed=args.seed)
    else:
        config = SamplingConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
    sampler = None if config.is_greedy else Sampler(config)

    prompt_ids = tokenizer.encode(text)
    if args.show_prompt:
        print(f"prompt ({len(prompt_ids)} tokens): {text!r}", file=sys.stderr)

    print(text, end="", flush=True)

    # Decode the whole run so far and print only what is new, rather than
    # decoding each token alone. A multi-byte character routinely spans several
    # tokens, so per-token decoding prints a replacement character wherever one
    # straddles a boundary -- the same trap the tokenizer's decode() avoids.
    produced: list[int] = []
    emitted = ""

    def on_token(token_id: int) -> None:
        nonlocal emitted
        produced.append(token_id)
        text_so_far = tokenizer.decode(produced, skip_special_tokens=True)
        print(text_so_far[len(emitted) :], end="", flush=True)
        emitted = text_so_far

    result = greedy(
        model,
        prompt_ids,
        max_new_tokens=args.max_tokens,
        stop_ids=[i for i in stop_ids if i is not None],
        on_token=on_token,
        use_cache=not args.no_cache,
        sampler=sampler,
    )

    print()
    print(
        f"\n{len(prompt_ids)} prompt + {len(result.generated_ids)} generated tokens"
        f"  |  ttft {result.time_to_first_token_ms:.0f} ms"
        f"  |  decode {result.decode_tokens_per_second:.2f} tok/s"
        f"  |  kv cache {'on' if result.used_cache else 'off'}"
        f"  |  {('greedy' if not result.sampled else f'T={config.temperature} k={config.top_k} p={config.top_p} seed={result.seed}')}"
        f"  |  stopped: {result.stop_reason}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
