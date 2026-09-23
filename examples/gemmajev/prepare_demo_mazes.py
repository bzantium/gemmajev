"""Recreate the three fixed demonstration mazes without training targets."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "references/nanojev-upstream/scripts"))
from scaled_maze import make_maze  # noqa: E402

SEEDS = (2026121201, 2026121202, 2026121203)
EXPECTED_SHA256 = "560030ac4e8b3f3b28e244f54085f74ed0803aa053c8b24d42454b4a915984dd"


def main():
    rows = [
        dict(id=f"gemmajev-demo:50:{seed}", split="demonstration", game="scaled_maze",
             size=50, initial_state=make_maze(50, seed, "loops"))
        for seed in SEEDS
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Generated cases differ from the recorded demonstration cases")
    output = ROOT / "data/gemmajev-three-mazes"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cases.jsonl"
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError("Refusing to replace different demo cases")
    path.write_bytes(content)
    print(json.dumps(dict(cases=len(rows), sha256=digest, training_targets=False)))


if __name__ == "__main__":
    main()
