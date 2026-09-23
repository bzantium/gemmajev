"""Check an exported Transformers scorer against frozen Tunix outputs on CPU."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from gemmajev.batching import encode_compact_games
from gemmajev.transformers_backend import TransformersGameEngine


def probabilities(scores, mask):
    scores = np.where(mask, scores.astype(np.float64), -np.inf)
    values = np.exp(scores - scores.max(-1, keepdims=True))
    return values / values.sum(-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder, output = [(root / p).resolve() for p in (args.model, args.output)]
    if not all(p.is_relative_to(root) for p in (folder, output)):
        raise ValueError("Keep all artifacts under the project")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    engine = TransformersGameEngine(folder)
    fixtures = json.loads((folder / "fixtures.json").read_text())
    count, changed, max_difference = 0, 0, 0.0
    for fixture in fixtures:
        encoded = encode_compact_games(
            fixture["rows"], engine.tokenizer, engine.config["max_length"]
        )
        np.testing.assert_array_equal(encoded["tokens"], fixture["tokens"])
        np.testing.assert_array_equal(encoded["lengths"], fixture["lengths"])
        reference = probabilities(np.array(fixture["scores"]), encoded["candidate_mask"])
        actual = probabilities(engine.score(encoded), encoded["candidate_mask"])
        count += len(fixture["rows"])
        changed += int(np.sum(actual.argmax(-1) != reference.argmax(-1)))
        max_difference = max(max_difference, float(abs(actual - reference).max()))
        print(
            json.dumps(dict(questions=count, changed=changed, max_difference=max_difference)),
            flush=True,
        )
    passed = changed == 0 and max_difference < 1e-4
    report = dict(
        status="passed" if passed else "failed",
        reference_questions=count,
        changed_decisions=changed,
        max_probability_difference=max_difference,
        probability_tolerance=1e-4,
        precision="float32",
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        scope="40 fixed validation questions: 32 Maze movement and 8 ViZDoom. Runtime conversion check, not a new task benchmark.",
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise ValueError("Export differs from Tunix reference beyond the FP32 acceptance criteria")


if __name__ == "__main__":
    main()
