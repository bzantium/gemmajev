"""Run all three fixed Maze cases using the exported model on Apple GPU."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from jev_tunix.game_mlx import MLXGameEngine

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))
from evaluate_model_edges_maze import run_exploration  # noqa: E402


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--bits", type=int, choices=[8])
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Keep all artifacts under the project")
    output.mkdir(parents=True, exist_ok=False)
    engine = MLXGameEngine(ROOT / args.model, precision=args.precision, bits=args.bits)
    cases_path = ROOT / "data/gemmajev-three-mazes/cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    assert len(cases) == 3
    identity = dict(engine="checkpoint", model="GemmaJev", backend="MLX; trained with Tunix",
                    checkpoint_step=engine.config["steps"], precision=args.precision, bits=args.bits,
                    training_metadata_sha256=engine.manifest["metadata_sha256"])
    sources = [Path(__file__), ROOT / "jev_tunix/game_mlx.py", ROOT / "jev_tunix/tokenization.py",
               ROOT / "jev_tunix/game_compact.py", ROOT / "jev_tunix/game_contract.py",
               UPSTREAM / "evaluate_model_edges_maze.py", UPSTREAM / "evaluate_composed_maze.py",
               UPSTREAM / "scaled_maze.py"]
    provenance = dict(model=identity, training_run=engine.manifest["source_run"],
                      case_sha256=sha(cases_path),
                      export_manifest_sha256=sha(engine.folder / "manifest.json"),
                      source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "cases.jsonl").write_bytes(cases_path.read_bytes())
    summaries = []
    for index, case in enumerate(cases, 1):
        print(json.dumps(dict(starting=index, case=case["id"])), flush=True)
        result = run_exploration([case], engine, window_size=5, max_steps=0,
                                 batch_states=1, batch_questions=4)
        result.update(model=identity, source_episodes_sha256=sha(cases_path),
                      implementation_sha256=sha(UPSTREAM / "evaluate_model_edges_maze.py"),
                      local_renderer_sha256=sha(UPSTREAM / "evaluate_composed_maze.py"))
        (output / f"maze-{index}.json").write_text(json.dumps(result, indent=2) + "\n")
        episode = result["episodes"][0]
        summaries.append({"case": case["id"], **{key: episode[key] for key in
                          ("status", "goal_completion", "attempts", "successful_moves", "collisions")},
                          "seconds": result["elapsed_seconds"]})
        (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
