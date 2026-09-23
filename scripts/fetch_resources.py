"""Fetch the pinned NanoJev code and checksum-verified data for these demos."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "76fdfc9ecdca45a9bcef17991a07d3041a87685a"
MANIFEST_SHA = "5934af3eb325b7e79e090d6b1ce30cd0c3535da6388fa2c3e8cc4fd183d2382a"
BASE = "https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/resolve/unified-games-v1/"
FILES = [
    "unified/hard/train.jsonl",
    "unified/hard/dev.jsonl",
    "unified/hard/manifest.json",
    "evaluation/test_cases.jsonl",
    "demonstrations/configs/hard_navigation_demo_v1_cases.jsonl",
    "demonstrations/web/dev/side_by_side_results.json",
    "demonstrations/web/dev/shooting_results.json",
]


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(path, url, expected, verify_only):
    if path.exists():
        if checksum(path) != expected:
            raise ValueError(f"Checksum mismatch: {path.relative_to(ROOT)}")
        return
    if verify_only:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    if checksum(partial) != expected:
        raise ValueError(f"Downloaded checksum mismatch: {path.name}")
    partial.rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--source-only", action="store_true", help="Fetch only the pinned game code"
    )
    args = parser.parse_args()
    checkout = ROOT / "references/nanojev-upstream"
    if not checkout.exists():
        if args.verify_only:
            raise FileNotFoundError(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "fetch",
                "--depth=1",
                "https://github.com/TianyuCodings/NanoJev.git",
                COMMIT,
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise ValueError("Existing NanoJev checkout does not match the pinned commit")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        raise ValueError("Tracked NanoJev source has local modifications")
    if args.source_only:
        print(json.dumps(dict(source_commit=COMMIT, source_only=True)))
        return
    manifest_path = ROOT / "references/nanojev-data-release/SHA256_MANIFEST.json"
    fetch(manifest_path, BASE + "SHA256_MANIFEST.json", MANIFEST_SHA, args.verify_only)
    manifest = json.loads(manifest_path.read_text())
    indexed = {row["path"]: row for row in manifest["files"]}
    receipts = []
    for name in FILES:
        row = indexed[name]
        fetch(ROOT / "data/nanojev-public" / name, BASE + name, row["sha256"], args.verify_only)
        receipts.append(dict(row, url=BASE + name))
    if not args.verify_only:
        destination = ROOT / "data/nanojev-public/download-manifest.json"
        if destination.exists():
            previous = json.loads(destination.read_text())
            existing = {row["path"]: row for row in previous["files"]}
            existing.update({row["path"]: row for row in receipts})
            receipts = list(existing.values())
        destination.write_text(
            json.dumps(
                dict(
                    release="unified-games-v1",
                    source_git_commit="4a68a99a921571e94246e77db132f2d777747dba",
                    files=receipts,
                ),
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(
            dict(source_commit=COMMIT, verified_files=len(FILES), verify_only=args.verify_only)
        )
    )


if __name__ == "__main__":
    main()
