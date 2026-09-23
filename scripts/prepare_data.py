"""Prepare only Basic actions and local Maze safety for the two teaching demos."""

import collections
import hashlib
import json
import sys
from pathlib import Path

from gemmajev.interface import decision_row

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references/nanojev-upstream/scripts"
sys.path.insert(0, str(UPSTREAM))
from evaluate_composed_maze import local_questions, local_truth  # noqa: E402


def identity(row):
    return json.dumps({k: row[k] for k in ("state", "question", "candidates")}, sort_keys=True)


def main():
    output = ROOT / "data/gemmajev-v1"
    if output.exists():
        raise FileExistsError(output)
    manifest = dict(
        scope="ViZDoom Basic and local Maze Boolean questions for two fixed demonstrations",
        basic_labels="Public expert action labels; no relabelling from successful demo actions",
        maze_labels="Geometric truth from the visible 5x5 map, with upstream local question wording",
        splits={},
        inputs_sha256={},
    )
    downloaded = json.loads((ROOT / "data/nanojev-public/download-manifest.json").read_text())
    seen = {}
    groups = []
    prepared = {}
    for source_split, split in (("train", "train"), ("dev", "validation")):
        source = ROOT / f"data/nanojev-public/unified/hard/{source_split}.jsonl"
        expected = next(
            r["sha256"]
            for r in downloaded["files"]
            if r["path"] == f"unified/hard/{source_split}.jsonl"
        )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        manifest["inputs_sha256"][str(source.relative_to(ROOT))] = expected
        rows, source_groups, duplicates = [], set(), 0
        split_inputs, ambiguous = set(), set()
        for raw in source.read_text().splitlines():
            item = json.loads(raw)
            meta = item["metadata"]
            if meta["task"] == "maze":
                state = item["state"].split("Agent coordinate:", 1)[1]
                prefix, grid = state.split("Local map:\n", 1)
                lines = grid.splitlines()[:5]
                assert len(lines) == 5 and all(len(line) == 5 for line in lines)
                public = dict(
                    state="Agent coordinate:" + prefix + "Local map:\n" + "\n".join(lines)
                )
                truth = local_truth(public)
                examples = []
                for qid, question in local_questions().items():
                    row, keys = decision_row(public["state"], question)
                    label = "true" if truth[qid.removeprefix("clear_")] else "false"
                    examples.append(
                        dict(row, target=keys.index(label), task="maze", question_id=qid)
                    )
            elif meta.get("spec", {}).get("scenario") == "basic":
                if meta.get("policy_target_kind") != "expert_action":
                    raise ValueError("Basic examples must carry an explicit expert action")
                row, keys = decision_row(item["state"], item["questions"]["action"])
                examples = [
                    dict(
                        row,
                        target=keys.index(item["gold"]["action"]),
                        task="basic",
                        question_id="action",
                    )
                ]
            else:
                continue
            source_groups.add(meta["source_group_id"])
            for row in examples:
                key = identity(row)
                if key in seen:
                    if key in split_inputs and seen[key] != row["target"]:
                        ambiguous.add(key)
                    duplicates += 1
                    continue
                seen[key] = row["target"]
                split_inputs.add(key)
                rows.append(dict(row, source_id=item["id"], episode_id=meta["episode_id"]))
        rows = [row for row in rows if identity(row) not in ambiguous]
        prepared[split] = rows
        groups.append(source_groups)
        manifest["splits"][split] = dict(
            questions=len(rows),
            tasks=dict(collections.Counter(r["task"] for r in rows)),
            duplicate_inputs_removed=duplicates,
            ambiguous_inputs_removed=len(ambiguous),
            source_groups=len(source_groups),
        )
    assert not groups[0] & groups[1]
    for path in [
        Path(__file__),
        ROOT / "gemmajev/interface.py",
        UPSTREAM / "evaluate_composed_maze.py",
    ]:
        manifest["inputs_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    output.mkdir(parents=True)
    for split, rows in prepared.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["splits"][split]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
