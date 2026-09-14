"""Fetch a model's raw weight and tokenizer files from the HuggingFace CDN.

Deliberately not using ``huggingface_hub``: the whole project is about owning
the layer below the framework, and the download is four HTTPS GETs. It also
keeps the dependency list at "numpy" for the engine itself.

Run:  python -m tools.download_model
      python -m tools.download_model --repo Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/{repo}/resolve/main/{name}"

# The files an inference engine actually needs. Note what is absent: no
# pytorch_model.bin, no .gguf, no framework metadata.
REQUIRED = ("config.json", "model.safetensors", "tokenizer.json")
OPTIONAL = ("generation_config.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def download(repo: str, name: str, dest: Path, force: bool = False) -> bool:
    target = dest / name
    if target.exists() and not force:
        print(f"  have    {name:<28} {target.stat().st_size:>13,} B")
        return True

    url = BASE.format(repo=repo, name=name)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            digest = hashlib.sha256()
            read = 0
            with tmp.open("wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
                    digest.update(chunk)
                    read += len(chunk)
                    if total:
                        pct = 100 * read / total
                        print(f"\r  get     {name:<28} {read:>13,} B  {pct:5.1f}%", end="")
            print(f"\r  get     {name:<28} {read:>13,} B  done   ")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        print(f"  miss    {name:<28} HTTP {exc.code}")
        return False
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  fail    {name:<28} {exc}")
        return False

    tmp.replace(target)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=Path, default=None, help="destination dir (default: models/<model name>)")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args(argv)

    dest = args.out or Path("models") / args.repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{args.repo} -> {dest}")

    free = shutil.disk_usage(dest).free
    if free < 3 * 1024**3:
        print(f"  warning: only {free / 1e9:.1f} GB free", file=sys.stderr)

    ok = True
    for name in REQUIRED:
        ok &= download(args.repo, name, dest, args.force)
    for name in OPTIONAL:
        download(args.repo, name, dest, args.force)

    if not ok:
        print("\nrequired files missing", file=sys.stderr)
        return 1
    print(f"\nready: python -m tools.inspect_weights {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
