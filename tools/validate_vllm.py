"""Compare a running vLLM pooling server with the export's frozen Tunix outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from gemmajev.batching import encode_compact_games
from gemmajev.vllm_backend import VLLMGameEngine


def probabilities(scores):
    values = np.asarray(scores, dtype=np.float64)
    values = np.exp(values - values.max(-1, keepdims=True))
    return values / values.sum(-1, keepdims=True)


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Transformers export")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="gemmajev")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Keep validation output under the project")
    engine = VLLMGameEngine(
        args.model,
        base_url=args.base_url,
        model=args.served_model,
        api_key=os.environ.get("VLLM_API_KEY"),
    )
    fixtures = json.loads((engine.folder / "fixtures.json").read_text())
    count, changed, max_difference = 0, 0, 0.0
    for fixture in fixtures:
        encoded = encode_compact_games(
            fixture["rows"], engine.tokenizer, engine.config["max_length"]
        )
        np.testing.assert_array_equal(encoded["tokens"], fixture["tokens"])
        np.testing.assert_array_equal(encoded["lengths"], fixture["lengths"])
        reference = probabilities(np.where(encoded["candidate_mask"], fixture["scores"], -np.inf))
        actual = probabilities(engine.score(encoded))
        count += len(fixture["rows"])
        changed += int(np.sum(actual.argmax(-1) != reference.argmax(-1)))
        max_difference = max(max_difference, float(abs(actual - reference).max()))
    passed = count > 0 and changed == 0 and max_difference < 1e-4
    report = dict(
        status="passed" if passed else "failed",
        reference_questions=count,
        changed_decisions=changed,
        max_probability_difference=max_difference,
        probability_tolerance=1e-4,
        scope="Frozen runtime conversion check; not a new task benchmark.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("vLLM output differs from the FP32 reference")


if __name__ == "__main__":
    main()
