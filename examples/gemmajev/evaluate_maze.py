"""Compare a continued checkpoint with its parent on identical frozen questions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jev_tunix.experiment import evaluate_arrays, project_path
from jev_tunix.game_contract import encode_games
from jev_tunix.game_inference import GameEngine
from jev_tunix.game_observation import format_rows

ROOT = Path(__file__).resolve().parents[2]


def binary_details(scores, rows):
    scores = scores[:, :2].astype(np.float64)
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    truth = np.array([r["target"] for r in rows])
    predicted = probs.argmax(axis=1)
    positive, negative = truth == 1, truth == 0
    return dict(
        always_open_accuracy=float(positive.mean()),
        open_recall=float((predicted[positive] == 1).mean()),
        blocked_recall=float((predicted[negative] == 0).mean()),
        balanced_accuracy=float(((predicted[positive] == 1).mean()
                                 + (predicted[negative] == 0).mean()) / 2),
        nearly_even_fraction=float((abs(probs[:, 1] - .5) < .05).mean()),
        per_direction={direction: float((predicted[mask] == truth[mask]).mean())
                       for direction in ("north", "east", "south", "west")
                       if (mask := np.array([r["question_id"] == "clear_" + direction
                                            for r in rows])).any()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference", default="runs/baseline")
    parser.add_argument("--dataset", default="data/gemmajev-maze-v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = project_path(args.dataset)
    sources = {"original_validation": ROOT / "data/gemmajev-v1/validation.jsonl",
               "new_validation": dataset / "validation.jsonl", "new_test": dataset / "test.jsonl"}
    for name, path in sources.items():
        manifest = json.loads((path.parent / "manifest.json").read_text())
        expected = manifest["splits"][path.stem]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen evaluation data changed: {name}")
    rows = {name: [json.loads(line) for line in p.read_text().splitlines()]
            for name, p in sources.items()}
    report = dict(status="running", input_sha256={name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for name, p in sources.items()}, models={})
    for label, run in (("previous", args.reference), ("continued", args.run)):
        engine = GameEngine(run)
        result = dict(run=run, metadata_sha256=hashlib.sha256(
            (engine.run / "metadata.json").read_bytes()).hexdigest(), evaluations={})
        with engine.mesh:
            for split, examples in rows.items():
                result["evaluations"][split] = {}
                for task in ("maze", "basic"):
                    subset = [row for row in examples if row["task"] == task]
                    if not subset:
                        continue
                    data = encode_games(
                        format_rows(subset, engine.config.get("observation_layout", "original")),
                        engine.tokenizer, engine.config["max_length"])
                    metrics, scores = evaluate_arrays(engine.model, data)
                    if task == "maze":
                        metrics.update(binary_details(scores, subset))
                    np.save(output / f"{label}-{split}-{task}.npy", scores)
                    result["evaluations"][split][task] = metrics
                    print(json.dumps(dict(model=label, split=split, task=task,
                                          accuracy=metrics["accuracy"])), flush=True)
        report["models"][label] = result
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        del engine
    report["status"] = "completed"
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
