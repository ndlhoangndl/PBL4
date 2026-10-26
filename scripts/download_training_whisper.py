"""Download the public Transformers checkpoint once, pinned to its resolved SHA."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="openai/whisper-tiny")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = HfApi().model_info(args.repo, token=False).sha
    print(f"Downloading public {args.repo} at revision {revision}", flush=True)
    snapshot_download(repo_id=args.repo, revision=revision, token=False,
        local_dir=str(args.output), cache_dir=str(args.output.parent / "hfhub"), max_workers=2,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "model.safetensors", "README*", "LICENSE*"])
    if not (args.output / "model.safetensors").is_file():
        raise RuntimeError("No trainable safetensors checkpoint downloaded")
    files = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in args.output.rglob("*") if path.is_file() and ".cache" not in path.parts and path.name != "download-provenance.json"}
    (args.output / "download-provenance.json").write_text(json.dumps({"repo": args.repo,
        "revision": revision, "files_sha256": files}, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned checkpoint ready at {args.output}", flush=True)


if __name__ == "__main__":
    main()
