"""Measure complete tokenized inputs before allocating a training GPU."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from jev_tunix.game_contract import encode_games
from jev_tunix.game_observation import format_rows
from jev_tunix.model_registry import ChatTokenizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    tokenizer = ChatTokenizer(
        AutoTokenizer.from_pretrained(ROOT / "artifacts" / cfg["model_name"], local_files_only=True)
    )
    folder = ROOT / "data" / cfg["dataset"]
    result = {}
    for split in ("train", "validation", "test"):
        if not (folder / f"{split}.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (folder / f"{split}.jsonl").read_text().splitlines()]
        maxima = {"maze": 0, "basic": 0}
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            data = encode_games(format_rows(batch, cfg.get("observation_layout", "original")),
                                tokenizer, cfg["max_length"])
            for row, lengths in zip(batch, data["lengths"], strict=True):
                maxima[row["task"]] = max(maxima[row["task"]], int(lengths.max()))
        result[split] = dict(questions=len(rows), max_tokens=maxima)
    path = folder / "token-capacity.json"
    if args.verify_only or path.exists():
        expected = json.loads(path.read_text())
        for split, actual in result.items():
            if actual != expected[split]:
                raise ValueError(f"Token capacity receipt differs for {split}")
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
