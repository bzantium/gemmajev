"""Run memory-conditioned movement selection with identical exploration guards."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from gemmajev.navigation import NavigationMemory, select_action

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))
from evaluate_model_edges_maze import MazeEnvironment  # noqa: E402


def replay(case, steps):
    env = MazeEnvironment(case["initial_state"], window_size=5)
    memory = NavigationMemory(**env.public_coordinates())
    for row in steps:
        memory.observe(env.observe()["state"])
        assert memory.available() == row["available"]
        assert row["action"] in row["available"]
        if row["request"] is not None:
            assert memory.request(row["request"]["id"]) == row["request"]
        assert list(memory.position) == row["position"]
        feedback = env.attempt(row["action"])
        for key, value in feedback.items():
            assert value == row[key], (row["step"], key)
        memory.transition(row["action"], feedback["next_position"], feedback["collision"])
    return env.reached_goal()


def run_case(case, engine):
    env = MazeEnvironment(case["initial_state"], window_size=5)
    memory = NavigationMemory(**env.public_coordinates())
    steps, model_calls, masked_argmax = [], 0, 0
    start = time.perf_counter()
    for step in range(2 * memory.size**2):
        if env.reached_goal():
            break
        memory.observe(env.observe()["state"])
        available = memory.available()
        if not available:
            break
        request, probabilities = None, None
        if len(available) == 1:
            action = available[0]
        else:
            # Requests depend on changing memory; never cache solely by coordinate.
            request = memory.request(f"{case['id']}:{step}")
            response = engine.predict({"states": [request]}, batch_questions=1)
            probabilities = response["states"][0]["answers"]["move"]["probabilities"]
            raw_choice = max(probabilities, key=probabilities.get)
            masked_argmax += raw_choice not in available
            action = select_action(probabilities, available)
            model_calls += 1
        before = list(memory.position)
        feedback = env.attempt(action)
        assert feedback["position"] == before
        memory.transition(action, feedback["next_position"], feedback["collision"])
        steps.append(
            dict(
                step=step,
                action=action,
                available=available,
                request=request,
                probabilities=probabilities,
                forced=len(available) == 1,
                **feedback,
            )
        )
    completed = replay(case, steps)
    assert completed == env.reached_goal()
    summary = dict(
        case=case["id"],
        scope=case["scope"],
        goal=completed,
        attempts=len(steps),
        collisions=sum(row["collision"] for row in steps),
        model_calls=model_calls,
        forced_moves=len(steps) - model_calls,
        masked_argmax=masked_argmax,
        seconds=time.perf_counter() - start,
        replay_verified=True,
    )
    return dict(case=case, summary=summary, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["jax", "mlx"], default="jax")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--scope", choices=["demo_fit", "generated"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, cases_path, model_path = [
        (ROOT / p).resolve() for p in (args.output, args.cases, args.model)
    ]
    if not all(p.is_relative_to(ROOT) for p in (output, cases_path, model_path)):
        raise ValueError("Keep all artifacts under the project")
    output.mkdir(parents=True, exist_ok=False)
    if args.backend == "jax":
        from gemmajev.jax_backend import GameEngine

        engine = GameEngine(args.model, compact=True)
        identity = model_path / "metadata.json"
    else:
        from gemmajev.mlx_backend import MLXGameEngine

        engine = MLXGameEngine(model_path, precision="float32")
        identity = model_path / "manifest.json"
    all_cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    cases = [case for case in all_cases if not args.scope or case["scope"] == args.scope]
    if not cases:
        raise ValueError("No cases selected")
    report = dict(
        model=args.model,
        backend=args.backend,
        model_metadata_sha256=hashlib.sha256(identity.read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        policy="Model ranks available unvisited legal passages; DFS stack backtracks; forced moves skip inference",
        hidden_map_used_by_policy=False,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "gemmajev/navigation.py",
                ROOT / "gemmajev/interface.py",
            )
        },
        cases=[],
    )
    for i, case in enumerate(cases, 1):
        print(json.dumps(dict(starting=i, case=case["id"])), flush=True)
        episode = run_case(case, engine)
        (output / f"maze-{i}.json").write_text(json.dumps(episode) + "\n")
        report["cases"].append(episode["summary"])
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(episode["summary"]), flush=True)


if __name__ == "__main__":
    main()
