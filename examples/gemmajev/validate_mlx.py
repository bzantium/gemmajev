"""Measure real Apple GPU inference and compare frozen Tunix reference outputs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jev_tunix.game_compact import encode_compact_games
from jev_tunix.game_mlx import MLXGameEngine


def probabilities(scores, mask):
    scores = np.where(mask, scores.astype(np.float64), -np.inf)
    values = np.exp(scores - scores.max(-1, keepdims=True))
    return values / values.sum(-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--bits", type=int, choices=[8])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Keep all artifacts under the project")
    output.mkdir(parents=True, exist_ok=False)
    engine = MLXGameEngine(args.model, precision=args.precision, bits=args.bits)
    fixtures = json.loads((Path(args.model) / "fixtures.json").read_text())
    differences, changed, records = [], 0, []
    for index, fixture in enumerate(fixtures):
        encoded = encode_compact_games(fixture["rows"], engine.tokenizer, engine.config["max_length"])
        np.testing.assert_array_equal(encoded["tokens"], fixture["tokens"])
        np.testing.assert_array_equal(encoded["lengths"], fixture["lengths"])
        scores = engine.score(encoded)
        reference = probabilities(np.array(fixture["scores"]), encoded["candidate_mask"])
        actual = probabilities(scores, encoded["candidate_mask"])
        errors = abs(actual - reference)
        differences.extend(errors[encoded["candidate_mask"]].tolist())
        changed += int(np.sum(actual.argmax(-1) != reference.argmax(-1)))
        records.append(dict(reference=reference.tolist(), actual=actual.tolist(),
                            max_probability_difference=float(errors.max())))
        print(json.dumps(dict(batch=index, max_probability_difference=float(errors.max()),
                              changed_decisions_so_far=changed)), flush=True)
    # Include tokenization and answer construction, not just a lazily queued kernel.
    from jev_tunix.game_contract import decision_row

    questions = {}
    for row in fixtures[0]["rows"]:
        choices = [json.loads(c) for c in row["candidates"]]
        questions[row["question_id"]] = dict(type="boolean", instructions=row["question"],
            criteria={c["value"]: c["description"] for c in choices})
    request = dict(states=[dict(id="latency", state=fixtures[0]["rows"][0]["state"], questions=questions)])
    assert len(questions) == 4
    for row, question in zip(fixtures[0]["rows"], questions.values(), strict=True):
        converted, _ = decision_row(request["states"][0]["state"], question)
        assert converted["state"] == row["state"]
    engine.predict(request)
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        engine.predict(request)
        latencies.append(time.perf_counter() - start)
    report = dict(status="completed", precision=args.precision, bits=args.bits,
                  device=mx.device_info(), platform=platform.platform(),
                  versions={name: importlib.metadata.version(name)
                            for name in ["mlx", "mlx-lm", "transformers", "numpy"]},
                  reference_questions=sum(len(f["rows"]) for f in fixtures),
                  changed_decisions=changed, max_probability_difference=max(differences),
                  mean_probability_difference=float(np.mean(differences)), records=records,
                  mean_seconds=float(np.mean(latencies)), median_seconds=float(np.median(latencies)),
                  warm_seconds=latencies, scope="40 frozen questions, one Maze observation timed ten times. Full rollouts evaluated separately.",
                  export_manifest_sha256=hashlib.sha256((Path(args.model) / "manifest.json").read_bytes()).hexdigest())
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["records", "warm_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
