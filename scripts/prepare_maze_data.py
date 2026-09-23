"""Generate local Maze supervision with map-separated evaluation and contrast pairs."""

import collections
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from gemmajev.interface import decision_row

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))
from evaluate_composed_maze import local_questions, local_truth, render_local_request  # noqa: E402
from scaled_maze import DIRECTIONS, make_maze, source_group_id  # noqa: E402


def window(text):
    return text.split("Local map:\n", 1)[1].splitlines()


def rotate(lines):
    return ["".join(row) for row in zip(*lines[::-1], strict=True)]


def family(text):
    """Keep reflected/rotated copies of a local window in the same split."""
    lines = window(text)
    variants = []
    for _ in range(4):
        variants.extend(("\n".join(lines), "\n".join(s[::-1] for s in lines)))
        lines = rotate(lines)
    return min(variants)


def rotate_observation(text, size):
    row, col = map(int, re.search(r"Agent coordinate: \((\d+),(\d+)\)", text).groups())
    prefix = text.split("Local map:\n", 1)[0]
    prefix = prefix.replace(f"({row},{col})", f"({col},{size - 1 - row})", 1)
    return prefix + "Local map:\n" + "\n".join(rotate(window(text)))


def contrast(text, action):
    """Flip only one visible neighbor; outside cells cannot become walkable."""
    lines = [list(row) for row in window(text)]
    dr, dc = DIRECTIONS[action]
    if lines[2 + dr][2 + dc] == "X":
        return None
    lines[2 + dr][2 + dc] = "." if lines[2 + dr][2 + dc] == "#" else "#"
    return (
        text.split("Local map:\n", 1)[0] + "Local map:\n" + "\n".join("".join(row) for row in lines)
    )


def labeled_rows(text, metadata):
    truth = local_truth({"state": text})
    rows = []
    for qid, question in local_questions().items():
        row, keys = decision_row(text, question)
        action = qid.removeprefix("clear_")
        label = "true" if truth[action] else "false"
        rows.append(dict(row, target=keys.index(label), task="maze", question_id=qid, **metadata))
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "data/gemmajev-maze-v2"
    if output.exists():
        raise FileExistsError(output)
    rng = random.Random(20260923)
    old = {
        split: [
            json.loads(s)
            for s in (ROOT / f"data/gemmajev-v1/{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "validation")
    }
    old_train = {family(r["state"]) for r in old["train"] if r["task"] == "maze"}
    old_val = {family(r["state"]) for r in old["validation"] if r["task"] == "maze"}
    protected = old_train | old_val
    split_families, groups, prepared, maps = {}, {}, {}, []
    # Evaluation is frozen before training examples are generated. No rollout data is read.
    for split, seed_base, map_count, per_map in (
        ("test", 2026092500, 16, 24),
        ("validation", 2026092400, 16, 24),
        ("train", 2026092300, 64, 40),
    ):
        excluded = protected | set().union(*split_families.values())
        if split == "train":
            excluded -= old_train - old_val
        families, source_groups, texts, rows = set(), set(), set(), []
        for index in range(map_count):
            size = (12, 20, 32, 50)[index % 4]
            topology = ("corridor", "tree", "loops", "random_obstacle")[(index // 4) % 4]
            state = make_maze(size, seed_base + index, topology)
            group = source_group_id(state)
            assert all(group not in previous for previous in groups.values())
            source_groups.add(group)
            maps.append(dict(split=split, initial_state=state, source_group_id=group))
            walls = set(map(tuple, state["walls"]))
            cells = [(r, c) for r in range(size) for c in range(size) if (r, c) not in walls]
            rng.shuffle(cells)
            # Include edges, corners and dead ends, not just cells on a successful route.
            boundary = [p for p in cells if min(p[0], p[1], size - 1 - p[0], size - 1 - p[1]) <= 1]
            interior = [p for p in cells if p not in set(boundary)]
            cells = []
            while boundary or interior:
                cells.extend(boundary[:1] + interior[:3])
                boundary, interior = boundary[1:], interior[3:]
            accepted = 0
            for cell in cells:
                current = dict(state, position=list(cell))
                text = render_local_request(current)["state"]
                if family(text) in excluded or text in texts:
                    continue
                base_meta = dict(
                    source_id=f"generated:{seed_base + index}:{cell[0]},{cell[1]}",
                    episode_id=f"generated:{seed_base + index}",
                    source_group_id=group,
                    size=size,
                    topology=topology,
                    augmentation="observed",
                )

                def add(value, kind):
                    if value is None or value in texts or family(value) in excluded:
                        return
                    texts.add(value)
                    families.add(family(value))
                    rows.extend(labeled_rows(value, dict(base_meta, augmentation=kind)))

                add(text, "observed")
                if split == "train":
                    if accepted % 4 == 0:
                        rotated = rotate_observation(text, size)
                        add(rotated, "rotation_90")
                    if accepted % 4 == 1:
                        action = list(DIRECTIONS)[(accepted // 4) % 4]
                        add(contrast(text, action), f"contrast_{action}")
                accepted += 1
                if accepted == per_map:
                    break
        groups[split] = source_groups
        split_families[split] = families
        # Preserve the exact Doom train/validation records from the original run.
        if split != "test":
            rows.extend(copy.deepcopy(r) for r in old[split] if r["task"] == "basic")
        prepared[split] = rows
    assert not split_families["train"] & split_families["validation"]
    assert not split_families["train"] & split_families["test"]
    assert not split_families["validation"] & split_families["test"]
    assert not split_families["train"] & old_val
    output.mkdir(parents=True)
    manifest = dict(
        scope="Maze data expansion with the unchanged local Boolean input/output contract",
        seed=20260923,
        evaluation="Disjoint maps and D4-equivalent local windows; frozen first",
        demo_seeds_excluded=[24310922, 2026121201, 2026121202, 2026121203],
        contrast="One adjacent visible cell toggled; all four labels recomputed from the observation",
        splits={},
        inputs_sha256={},
    )
    for split, rows in prepared.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        maze = [r for r in rows if r["task"] == "maze"]
        manifest["splits"][split] = dict(
            questions=len(rows),
            tasks=dict(collections.Counter(r["task"] for r in rows)),
            maps=len(groups[split]),
            local_families=len(split_families[split]),
            label_counts=dict(
                collections.Counter(f"{r['question_id']}:{r['target']}" for r in maze)
            ),
            augmentation=dict(collections.Counter(r["augmentation"] for r in maze)),
            sha256=sha(path),
        )
    (output / "maps.jsonl").write_text("".join(json.dumps(m) + "\n" for m in maps))
    for path in [
        Path(__file__),
        ROOT / "gemmajev/interface.py",
        UPSTREAM / "evaluate_composed_maze.py",
        UPSTREAM / "scaled_maze.py",
        ROOT / "data/gemmajev-v1/train.jsonl",
        ROOT / "data/gemmajev-v1/validation.jsonl",
    ]:
        manifest["inputs_sha256"][str(path.relative_to(ROOT))] = sha(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
