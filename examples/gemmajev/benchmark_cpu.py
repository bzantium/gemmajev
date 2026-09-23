"""Compare padded and compact inference on one unchanged trained checkpoint."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/maze-expanded")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cores", type=int, default=8)
    args = parser.parse_args()
    if args.cores < 1:
        parser.error("--cores must be positive")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = str(args.cores)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cores])

    import jax

    from jev_tunix.experiment import project_path
    from jev_tunix.game_compact import encode_compact_games
    from jev_tunix.game_contract import decision_row, encode_games
    from jev_tunix.game_inference import GameEngine
    from jev_tunix.game_observation import format_rows

    root = Path(__file__).resolve().parents[2]
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    engine = GameEngine(args.run)
    rows = [json.loads(line) for line in
            (root / "data/gemmajev-v1/validation.jsonl").read_text().splitlines()]
    observations = {}
    for row in rows:
        key = (row["task"], row["state"])
        observations.setdefault(key, []).append(row)
    selected = [r for (task, _), r in observations.items() if task == "maze"][:3]
    selected += [r for (task, _), r in observations.items() if task == "basic"][:1]
    requests = []
    for index, group in enumerate(selected):
        questions = {}
        for row in group:
            choices = [json.loads(c) for c in row["candidates"]]
            questions[row["question_id"]] = dict(
                type="boolean" if row["task"] == "maze" else "choice",
                instructions=row["question"],
                criteria={c["value"]: c["description"] for c in choices})
        requests.append(dict(states=[dict(id=str(index), state=group[0]["state"],
                                         questions=questions)]))
    assert len(requests) == 4 and len(requests[0]["states"][0]["questions"]) == 4
    report = dict(status="running", run=args.run, platform=platform.platform(),
                  devices=[str(d) for d in jax.devices()], jax=jax.__version__,
                  cpu_affinity=sorted(os.sched_getaffinity(0))
                  if hasattr(os, "sched_getaffinity") else None,
                  warmup={}, seconds={"padded": [], "compact": []}, checks=[])

    def invoke(request, compact):
        engine.compact = compact
        start = time.perf_counter()
        result = engine.predict(request)
        return result, time.perf_counter() - start

    for compact, label in [(False, "padded"), (True, "compact")]:
        _, report["warmup"][label] = invoke(requests[0], compact)
        print(json.dumps(dict(warmup=label, seconds=report["warmup"][label])), flush=True)
    for repeat in range(3):
        for compact, label in [(False, "padded"), (True, "compact")]:
            _, seconds = invoke(requests[0], compact)
            report["seconds"][label].append(seconds)
            print(json.dumps(dict(repeat=repeat, mode=label, seconds=seconds)), flush=True)
    for index, request in enumerate(requests):
        full, _ = invoke(request, False)
        compact, _ = invoke(request, True)
        a, b = full["states"][0]["answers"], compact["states"][0]["answers"]
        differences = []
        for key in a:
            aa = np.array(list(a[key]["probabilities"].values()))
            bb = np.array(list(b[key]["probabilities"].values()))
            np.testing.assert_allclose(aa, bb, atol=1e-5, rtol=1e-5)
            assert aa.argmax() == bb.argmax()
            differences.extend(abs(aa - bb).tolist())
        state = request["states"][0]
        raw = [decision_row(state["state"], q)[0] for q in state["questions"].values()]
        formatted = format_rows(raw, engine.config.get("observation_layout", "original"))
        shapes = {name: list(encoder(formatted, engine.tokenizer,
                                    engine.config["max_length"])["tokens"].shape)
                  for name, encoder in [("padded", encode_games),
                                        ("compact", encode_compact_games)]}
        report["checks"].append(dict(index=index, task=selected[index][0]["task"],
                                     max_probability_difference=max(differences),
                                     same_decisions=True, shapes=shapes))
        print(json.dumps(report["checks"][-1]), flush=True)
    report["mean_seconds"] = {k: float(np.mean(v)) for k, v in report["seconds"].items()}
    report["speedup"] = report["mean_seconds"]["padded"] / report["mean_seconds"]["compact"]
    report["metadata_sha256"] = hashlib.sha256((engine.run / "metadata.json").read_bytes()).hexdigest()
    report["scope"] = "Three warm timings on one observation; correctness on three Maze and one Doom observation. Not a full rollout or Mac benchmark."
    report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ["jev_tunix/game_compact.py", "jev_tunix/game_inference.py",
                                            "examples/gemmajev/benchmark_cpu.py"]}
    report["status"] = "completed"
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(mean_seconds=report["mean_seconds"], speedup=report["speedup"])), flush=True)


if __name__ == "__main__":
    main()
