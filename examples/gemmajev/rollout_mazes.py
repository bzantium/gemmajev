"""Record three fixed mazes with the existing trained Gemma checkpoint."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from jev_tunix.experiment import project_path
from jev_tunix.game_inference import GameEngine

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))
from evaluate_model_edges_maze import run_exploration  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases_path, output = project_path(args.cases), project_path(args.output)
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    assert len(cases) == 3 and len({c["id"] for c in cases}) == 3
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases.jsonl").write_bytes(cases_path.read_bytes())
    engine = GameEngine(args.run)
    identity = dict(engine="checkpoint", model="GemmaJev", backend="Gemma + Tunix",
                    checkpoint_step=engine.config["steps"],
                    training_metadata_sha256=sha(engine.run / "metadata.json"))
    write(output / "provenance.json", dict(
        model=identity, training_run=args.run, case_sha256=sha(cases_path),
        selection="Three seeds fixed before inference; no outcome-based selection.",
        slurm_job=os.environ.get("SLURM_JOB_ID"), nodes=os.environ.get("SLURM_JOB_NODELIST"),
        checkpoint_sha256={str(p.relative_to(engine.run)): sha(p)
                           for p in sorted((engine.run / "checkpoint").rglob("*")) if p.is_file()},
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       [Path(__file__), ROOT / "jev_tunix/game_inference.py",
                        ROOT / "jev_tunix/game_contract.py", UPSTREAM / "evaluate_model_edges_maze.py",
                        UPSTREAM / "evaluate_composed_maze.py", UPSTREAM / "scaled_maze.py"]},
    ))
    summaries = []
    for index, case in enumerate(cases, 1):
        print(json.dumps({"starting": index, "case": case["id"]}), flush=True)
        result = run_exploration([case], engine, window_size=5, max_steps=0,
                                 batch_states=1, batch_questions=4)
        result.update(model=identity, source_episodes_sha256=sha(cases_path),
                      implementation_sha256=sha(UPSTREAM / "evaluate_model_edges_maze.py"),
                      local_renderer_sha256=sha(UPSTREAM / "evaluate_composed_maze.py"))
        write(output / f"maze-{index}.json", result)
        episode = result["episodes"][0]
        summaries.append({"case": case["id"], **{key: episode[key] for key in
                          ("status", "goal_completion", "attempts", "successful_moves", "collisions")},
                          "seconds": result["elapsed_seconds"]})
        write(output / "summary.json", summaries)
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
