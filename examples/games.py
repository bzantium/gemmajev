"""Run the two fixed README cases with the same trained Gemma checkpoint."""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

from gemmajev.jax_backend import GameEngine
from gemmajev.runtime import project_path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))

from evaluate_composed_maze import digest  # noqa: E402
from evaluate_model_edges_maze import run_exploration  # noqa: E402
from unified_doom_env import UnifiedDoomEnv  # noqa: E402
from unified_game_pipeline import behavior_distribution, choose, policy_request  # noqa: E402


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basic-only", action="store_true", help="Check only the ViZDoom demo")
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    engine = GameEngine(args.run)
    identity = dict(
        engine="checkpoint",
        model="GemmaJev",
        backend="Gemma + Tunix",
        checkpoint_step=engine.config["steps"],
        training_metadata_sha256=sha(engine.run / "metadata.json"),
    )
    write(
        output / "provenance.json",
        dict(
            model=identity,
            training_run=args.run,
            checkpoint_sha256={
                str(p.relative_to(engine.run)): sha(p)
                for p in sorted((engine.run / "checkpoint").rglob("*"))
                if p.is_file()
            },
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    Path(__file__),
                    ROOT / "gemmajev/jax_backend.py",
                    ROOT / "gemmajev/interface.py",
                    *sorted(UPSTREAM.glob("*.py")),
                ]
            },
        ),
    )
    public = ROOT / "data/nanojev-public"
    cases = [
        json.loads(line)
        for line in (public / "evaluation/test_cases.jsonl").read_text().splitlines()
    ]
    case = next(row for row in cases if row["id"] == "test-appo_basic-9030060")
    rng = random.Random(int(digest([case["id"], 17])[:16], 16))
    env = UnifiedDoomEnv(copy.deepcopy(case["spec"]))
    episode = dict(
        case=case,
        steps=[],
        complete=False,
        controller=dict(mode="greedy", epsilon=0.1, sampling_seed=17),
        model=identity,
    )
    try:
        obs, info = env.reset(case["seed"])
        while not (info["terminated"] or info["truncated"]):
            request = policy_request(obs, f"{case['id']}:{len(episode['steps'])}")
            response = engine.predict({"states": [request]})
            answer = response["states"][0]["answers"]["action"]
            scores = answer["probabilities"]
            probs = behavior_distribution(scores, "greedy", 0.1)
            action = choose(probs, rng)
            previous = obs
            obs, reward, terminated, truncated, info = env.step(action)
            episode["steps"].append(
                dict(
                    observation=previous,
                    action=action,
                    scores=scores,
                    behavior_probs=probs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
        episode.update(
            complete=True, success=info["success"], final_info=info, final_observation=obs
        )
        write(output / "basic.json", episode)
        print(json.dumps(dict(basic=info["episode_metrics"])), flush=True)
    finally:
        env.close()
    if args.basic_only:
        from replay_unified_episodes import replay_episode

        audit = replay_episode(episode)
        write(output / "replay.json", audit)
        if not audit["passed"]:
            raise ValueError("ViZDoom trajectory did not replay exactly")
        write(
            output / "result.json",
            dict(basic_success=episode["success"], actual_gameplay=True, replay_verified=True),
        )
        return
    case_path = public / "demonstrations/configs/hard_navigation_demo_v1_cases.jsonl"
    cases = [json.loads(line) for line in case_path.read_text().splitlines()]
    case = next(row for row in cases if row["id"] == "maze:ood:50:24310922")
    maze = run_exploration(
        [case], engine, window_size=5, max_steps=0, batch_states=1, batch_questions=4
    )
    maze.update(
        model=identity,
        source_episodes_sha256=sha(case_path),
        implementation_sha256=sha(UPSTREAM / "evaluate_model_edges_maze.py"),
        local_renderer_sha256=sha(UPSTREAM / "evaluate_composed_maze.py"),
    )
    write(output / "maze.json", maze)
    print(json.dumps(dict(maze=maze["summary"])), flush=True)
    write(
        output / "result.json",
        dict(
            basic_success=episode["success"],
            maze_success=maze["episodes"][0]["goal_completion"],
            actual_gameplay=True,
            replay_verified=False,
        ),
    )


if __name__ == "__main__":
    main()
