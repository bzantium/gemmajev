"""Compare every fixed maze under the same controller and input contract."""

import argparse
import json
from pathlib import Path


def compare(reference, candidate, allow_observation_layout_change=False):
    def read(folder, name):
        return json.loads((folder / name).read_text())

    old_provenance = read(reference, "provenance.json")
    new_provenance = read(candidate, "provenance.json")
    if old_provenance["case_sha256"] != new_provenance["case_sha256"]:
        raise ValueError("The comparison requires identical cases")
    changes = []
    for name, digest in old_provenance["source_sha256"].items():
        if new_provenance["source_sha256"].get(name) != digest:
            if allow_observation_layout_change and name == "gemmajev/jax_backend.py":
                changes.append(name)
            else:
                raise ValueError(f"Controller, adapter or runner changed: {name}")
    old, new = [read(folder, "summary.json") for folder in (reference, candidate)]
    if [row["case"] for row in old] != [row["case"] for row in new]:
        raise ValueError("Cases were omitted or reordered")
    cases = []
    for index, (before, after) in enumerate(zip(old, new, strict=True), 1):
        old_episode, new_episode = [
            read(folder, f"maze-{index}.json")["episodes"][0] for folder in (reference, candidate)
        ]
        if old_episode["horizon"] != new_episode["horizon"]:
            raise ValueError("Attempt limit changed")
        if old_episode["initial_state"] != new_episode["initial_state"]:
            raise ValueError("Maze geometry changed")
        cases.append(
            dict(
                case=before["case"],
                before=before,
                after=after,
                attempt_change=after["attempts"] - before["attempts"],
                collision_change=after["collisions"] - before["collisions"],
            )
        )
    old_attempts, new_attempts = [sum(row["attempts"] for row in rows) for rows in (old, new)]
    return dict(
        status="matched",
        same_cases_controller_and_limit=True,
        inference_source_changes=changes,
        reference_run=old_provenance["training_run"],
        candidate_run=new_provenance["training_run"],
        all_candidates_finish=all(row["goal_completion"] for row in new),
        attempts=dict(
            before=old_attempts,
            after=new_attempts,
            reduction_fraction=1 - new_attempts / old_attempts,
        ),
        collisions={
            label: sum(row["collisions"] for row in rows)
            for label, rows in [("before", old), ("after", new)]
        },
        cases=cases,
        scope="Previously inspected development demonstrations; not a blind maze benchmark",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-observation-layout-change",
        action="store_true",
        help="Allow a declared inference adapter change; controller must match",
    )
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.allow_observation_layout_change)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
