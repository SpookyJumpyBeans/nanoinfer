"""Check nanoinfer against llama.cpp on identical weights.

    python -m tools.compare_llamacpp --llama-cpp C:/Users/daih0/llama.cpp

Phase 8 is a benchmark, and a benchmark between two engines that disagree is
meaningless -- it would be timing two different computations. So this runs
first: same weights, same prompts, greedy decoding, and an exact comparison of
what comes out.

**Same weights, not merely the same model.** The GGUF is converted from the
very safetensors file this engine loads, at f32, by llama.cpp's own
convert_hf_to_gguf.py. Downloading a prebuilt GGUF would introduce someone
else's conversion and quantization choices as an uncontrolled variable.

**Greedy, with llama.cpp's sampler pinned.** Every sampler that could move an
argmax is set explicitly, because the defaults are not what ``--help`` says.
Asked to report the parameters it actually used, llama.cpp prints::

    repeat_penalty = 1.100, top_p = 0.800, min_p = 0.050

against a documented CLI default of ``repeat-penalty 1.00``. The model file
wins: the converter copies generation_config.json into the GGUF metadata, so
Qwen2.5's own ``repetition_penalty: 1.1`` and ``top_p: 0.8`` arrive with the
weights and silently override the flag defaults. Leaving the flag off changes
the continuation from "Paris. It is the largest city..." to "Paris. It was
founded in 789 AD...".

This is exactly the bug phase 3 hit from the other side -- the transformers
reference diverged until repetition_penalty was forced to 1.0 -- reappearing
in a second engine through a completely different mechanism.

What agreement does and does not prove: identical greedy output over many
tokens means every argmax matched at every step, which is strong evidence the
forward passes agree. It is not a claim that the logits are bitwise equal --
they are not, and phase 3 measured the floor on that at ~1e-5 on logits
reaching 19.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoinfer.generate import greedy  # noqa: E402
from nanoinfer.model import Qwen2  # noqa: E402
from nanoinfer.tokenizer import Tokenizer  # noqa: E402
from nanoinfer.weights import ModelWeights  # noqa: E402

# llama.cpp colours its output even with logging disabled.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time",
    "The three laws of robotics are",
    "1, 1, 2, 3, 5, 8,",
)


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def llama_tokenize(binary: Path, gguf: Path, prompt: str) -> list[int]:
    """Token ids llama.cpp assigns to a prompt."""
    result = subprocess.run(
        [str(binary), "--model", str(gguf), "--prompt", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ids = []
    for line in strip_ansi(result.stdout).splitlines():
        # Lines look like "  6722 -> ' capital'".
        head, _, _ = line.partition("->")
        head = head.strip()
        if head.isdigit():
            ids.append(int(head))
    return ids


def llama_generate(
    binary: Path, gguf: Path, prompt: str, n: int, threads: int
) -> tuple[str, float]:
    """Greedy continuation from llama.cpp, with every sampler pinned."""
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(binary),
            "-m", str(gguf),
            "-p", prompt,
            "-n", str(n),
            "--temp", "0",            # default is 0.80 -- without this it samples
            "--top-k", "1",
            "--top-p", "1.0",
            "--min-p", "0.0",
            "--repeat-penalty", "1.0",
            "--seed", "0",
            "-t", str(threads),
            "-no-cnv",                # otherwise it drops into an interactive prompt
            "--no-display-prompt",
            # NOT --log-disable: it silences the generated text too, but only
            # when stdout is a pipe, so it looks fine by hand and returns an
            # empty string from a script.
            "--log-colors", "off",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    return strip_ansi(result.stdout).replace("\r\n", "\n").rstrip("\n"), elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp checkout")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument(
        "--gguf", type=Path, default=Path("models/qwen2.5-0.5b-instruct-f32.gguf")
    )
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("bench/llamacpp.jsonl"))
    args = parser.parse_args(argv)

    binaries = args.llama_cpp / "build" / "bin"
    completion = binaries / "llama-completion.exe"
    tokenize = binaries / "llama-tokenize.exe"
    for path in (completion, tokenize, args.gguf):
        if not path.exists():
            print(f"missing: {path}")
            return 1

    tokenizer = Tokenizer.from_model_dir(args.model)
    print("loading weights...", flush=True)
    model = Qwen2(ModelWeights.load(args.model))

    rows = []
    tokenizer_mismatches = 0
    generation_mismatches = 0

    print(f"\n{'prompt':<34} {'tokens':>7} {'tokenize':>9} {'generate':>9}")
    print("-" * 63)

    for prompt in PROMPTS:
        ours_ids = list(tokenizer.encode(prompt))
        theirs_ids = llama_tokenize(tokenize, args.gguf, prompt)
        tokenizer_ok = ours_ids == theirs_ids
        tokenizer_mismatches += not tokenizer_ok

        result = greedy(model, ours_ids, max_new_tokens=args.tokens)
        ours_text = tokenizer.decode(list(result.generated_ids))
        theirs_text, _ = llama_generate(
            completion, args.gguf, prompt, args.tokens, args.threads
        )
        generation_ok = ours_text == theirs_text
        generation_mismatches += not generation_ok

        label = prompt if len(prompt) <= 32 else prompt[:29] + "..."
        print(
            f"{label:<34} {args.tokens:>7} "
            f"{'match' if tokenizer_ok else 'DIFFER':>9} "
            f"{'match' if generation_ok else 'DIFFER':>9}"
        )
        if not tokenizer_ok:
            print(f"    ours  : {ours_ids}")
            print(f"    llama : {theirs_ids}")
        if not generation_ok:
            print(f"    ours  : {ours_text!r}")
            print(f"    llama : {theirs_text!r}")

        rows.append(
            {
                "prompt": prompt,
                "prompt_ids": ours_ids,
                "tokenizer_match": tokenizer_ok,
                "generation_match": generation_ok,
                "ours": ours_text,
                "llama_cpp": theirs_text,
            }
        )

    print("-" * 63)
    total = len(PROMPTS)
    print(
        f"tokenizer  : {total - tokenizer_mismatches}/{total} exact\n"
        f"generation : {total - generation_mismatches}/{total} exact "
        f"({args.tokens} greedy tokens each)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tokens": args.tokens, "results": rows}) + "\n")
    print(f"\nrecorded -> {args.out}")

    return 1 if (tokenizer_mismatches or generation_mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
