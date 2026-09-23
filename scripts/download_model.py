"""Download a pinned, authorized Hugging Face snapshot and record file hashes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from jev_tunix.model_registry import MODELS

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "google/gemma-3-270m"
REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="gemma-3-270m")
    args = parser.parse_args()
    spec = MODELS[args.model]
    # Configure caches before importing the Hub; importing this module has no side effects.
    os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
    from huggingface_hub import snapshot_download

    destination = ROOT / "artifacts" / args.model
    snapshot_download(
        spec["model_id"],
        revision=spec["revision"],
        local_dir=destination,
        cache_dir=ROOT / ".cache/huggingface/hub",
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "chat_template*",
            "README.md",
        ],
    )
    hashes = {}
    for path in destination.iterdir():
        if path.is_file() and path.name != "manifest.json":
            with path.open("rb") as handle:
                hashes[str(path.relative_to(destination))] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    manifest = {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "sha256": hashes,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
