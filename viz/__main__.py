"""Look inside the model while it generates.

    python -m viz
    python -m viz --port 8001 --model models/Qwen2.5-0.5B-Instruct

Serves a local page at http://127.0.0.1:8000 with three views: the candidates
the model weighed at each step, attention per layer and head, and the same
prompt run with and without the KV cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from viz.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args.model, args.port)


if __name__ == "__main__":
    main()
