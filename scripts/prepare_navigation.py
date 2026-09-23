"""Generate supervised movement examples, separating demo fitting from new maps."""

import hashlib
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path

from gemmajev.interface import decision_row
from gemmajev.navigation import DIRECTIONS, NavigationMemory, destination

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))
from evaluate_model_edges_maze import MazeEnvironment  # noqa: E402
from scaled_maze import make_maze  # noqa: E402


def distances(initial):
    """Privileged training teacher only; never used by the deployed policy."""
    goal = tuple(initial["goal"])
    walls = set(map(tuple, initial["walls"]))
    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        position = queue.popleft()
        for action in DIRECTIONS:
            nxt = destination(position, action)
            if (
                all(0 <= x < initial["size"] for x in nxt)
                and nxt not in walls
                and nxt not in distance
            ):
                distance[nxt] = distance[position] + 1
                queue.append(nxt)
    return distance


def examples(case, rollouts, rng):
    rows, receipts = [], []
    distance = distances(case["initial_state"])
    for repeat in range(rollouts):
        env = MazeEnvironment(case["initial_state"], window_size=5)
        memory = NavigationMemory(**env.public_coordinates())
        for step in range(2 * memory.size**2):
            if env.reached_goal():
                break
            memory.observe(env.observe()["state"])
            allowed = memory.available()
            if not allowed:
                raise RuntimeError("Connected teacher map exhausted before reaching its goal")
            teacher = min(
                allowed,
                key=lambda a: (
                    distance[destination(memory.position, a)],
                    list(DIRECTIONS).index(a),
                ),
            )
            request = memory.request(f"{case['id']}:{repeat}:{step}")
            row, keys = decision_row(request["state"], request["questions"]["move"])
            # Include all junctions plus a small sample of forced corridor/backtrack moves.
            if len(allowed) > 1 or rng.random() < 0.2:
                row.update(
                    target=keys.index(teacher),
                    task="maze",
                    question_id="move",
                    map_id=case["id"],
                    training_scope=case["scope"],
                    available=allowed,
                    junction=len(allowed) > 1,
                )
                rows.append(row)
            action = teacher
            # Teacher labels are retained on off-path states to teach recovery.
            if repeat and len(allowed) > 1 and rng.random() < 0.15:
                action = rng.choice(allowed)
            feedback = env.attempt(action)
            memory.transition(action, feedback["next_position"], feedback["collision"])
        receipts.append(dict(case=case["id"], repeat=repeat, goal=env.reached_goal(), steps=step))
    unique = {}
    for row in rows:
        key = json.dumps({k: row[k] for k in ("state", "question", "candidates")}, sort_keys=True)
        if key in unique and unique[key]["target"] != row["target"]:
            raise ValueError("Conflicting movement labels")
        unique[key] = row
    return list(unique.values()), receipts


def main():
    output = ROOT / "data/gemmajev-navigation-v1"
    output.mkdir(parents=True, exist_ok=False)
    demo_path = ROOT / "data/gemmajev-three-mazes/cases.jsonl"
    demos = [
        dict(json.loads(line), scope="demo_fit") for line in demo_path.read_text().splitlines()
    ]
    cases = {"train": demos, "validation": [], "test": []}
    for split, count, seed_base in (
        ("train", 20, 202609231000),
        ("validation", 4, 202609232000),
        ("test", 4, 202609233000),
    ):
        for i in range(count):
            size = (12, 20, 32, 50)[i % 4] if split == "train" else 50
            seed = seed_base + i
            cases[split].append(
                dict(
                    id=f"navigation:{size}:{seed}",
                    split=split,
                    scope="generated",
                    game="scaled_maze",
                    size=size,
                    initial_state=make_maze(size, seed, "loops"),
                )
            )
    rng = random.Random(20260923)
    manifest = dict(
        interface="local-window-memory-action-v1",
        demo_maps_in_training=True,
        teacher="Full-map distance labels for training only; absent from inference inputs",
        student="5x5 local observation, goal offset, visits, attempted edges, previous move",
        controller="Legal unvisited passages; stack backtracking for exhausted branches",
        splits={},
        inputs_sha256={},
    )
    receipts = []
    for split, maps in cases.items():
        rows = []
        for case in maps:
            n = 4 if case["scope"] == "demo_fit" else (2 if split == "train" else 1)
            prepared, receipt = examples(case, n, rng)
            rows.extend(prepared)
            receipts.extend(receipt)
        if split != "test":
            old = ROOT / f"data/gemmajev-v1/{split}.jsonl"
            rows.extend(
                json.loads(line)
                for line in old.read_text().splitlines()
                if json.loads(line)["task"] == "basic"
            )
        content = "".join(json.dumps(row) + "\n" for row in rows)
        (output / f"{split}.jsonl").write_text(content)
        (output / f"{split}-maps.jsonl").write_text("".join(json.dumps(c) + "\n" for c in maps))
        manifest["splits"][split] = dict(
            questions=len(rows),
            tasks=dict(Counter(r["task"] for r in rows)),
            maze_scopes=dict(Counter(r.get("training_scope") for r in rows if r["task"] == "maze")),
            junction_questions=sum(r.get("junction", False) for r in rows),
            maps=len(maps),
            sha256=hashlib.sha256(content.encode()).hexdigest(),
        )
    for path in (
        Path(__file__),
        ROOT / "gemmajev/navigation.py",
        ROOT / "gemmajev/interface.py",
        demo_path,
        ROOT / "data/gemmajev-v1/train.jsonl",
        ROOT / "data/gemmajev-v1/validation.jsonl",
        UPSTREAM / "scaled_maze.py",
    ):
        manifest["inputs_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "teacher-rollouts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
