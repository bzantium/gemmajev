"""Replay-audit three actual model runs and build a side-by-side maze viewer."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references/nanojev-upstream/scripts"))
import build_arcade_demo as arcade  # noqa: E402


def navigation_system(source, path):
    from examples.navigate import replay

    case, steps = source["case"], source["steps"]
    reached = replay(case, steps)
    summary = source["summary"]
    assert reached == summary["goal"]
    assert len(steps) == summary["attempts"]
    assert sum(s["collision"] for s in steps) == summary["collisions"]
    initial = case["initial_state"]
    frames = [dict(position=initial["position"], probabilities={}, collision=False)]
    for step in steps:
        frames.append(
            dict(
                position=step["next_position"],
                probabilities=step["probabilities"] or {},
                collision=step["collision"],
                action=step["action"],
                forced=step["forced"],
                available=step["available"],
            )
        )
    system = dict(
        frames=frames,
        summary={**summary, "outcome": "goal" if reached else "horizon_exhausted"},
    )
    audit = dict(
        path=str(path.relative_to(ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        replay_verified=True,
        scope=case["scope"],
    )
    return initial, system, audit, case["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    run, output = [(ROOT / x).resolve() for x in (args.rollout, args.output)]
    if not all(p.is_relative_to(ROOT) for p in (run, output)):
        raise ValueError("Keep artifacts under the project root")
    output.mkdir(parents=True, exist_ok=False)
    cases, audits, modes = [], [], set()
    arcade.COLORS["gemmajev"] = "#2879ff"
    for i in range(1, 4):
        path = run / f"maze-{i}.json"
        source = json.loads(path.read_text())
        if "steps" in source:
            modes.add("navigation")
            initial, system, audit, case_id = navigation_system(source, path)
        else:
            modes.add("local-safety")
            case_id = arcade.MAZE_CASE = source["episodes"][0]["id"]
            initial, system, audit, _ = arcade.maze_system(
                path,
                "gemmajev",
                "GemmaJev",
                "Gemma local probabilities with remembered-edge exploration",
            )
        cases.append(
            dict(
                label=f"Maze {i}",
                id=case_id,
                size=initial["size"],
                initial=initial,
                system=system,
            )
        )
        # Store source evidence relative to the project for portability.
        audit["path"] = str(path.relative_to(ROOT))
        audits.append(audit)
    assert len({c["id"] for c in cases}) == 3
    assert len(modes) == 1, "Do not mix movement and local-safety interfaces"
    (output / "demo.json").write_text(json.dumps({"mazes": cases, "mode": modes.pop()}) + "\n")
    (output / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    for name in ("index.html", "demo.css", "demo.js"):
        shutil.copy2(ROOT / "web/mazes" / name, output / name)
    if args.font_dir:
        fonts = args.font_dir.resolve()
        if not fonts.is_relative_to(ROOT):
            raise ValueError("Font assets must remain under the project")
        (output / "fonts").mkdir()
        for name in ("KakaoSmall.woff2", "KakaoBig-Bold.woff2"):
            shutil.copy2(fonts / name, output / "fonts" / name)
    else:
        css = output / "demo.css"
        css.write_text(re.sub(r"@font-face\s*\{[^}]*\}\s*", "", css.read_text()))

    shutil.copy2(ROOT / "references/nanojev-upstream/LICENSE", output / "NANOJEV-LICENSE")
    (output / "NOTICE.txt").write_text(
        "Maze generator and environment: TianyuCodings/NanoJev (MIT).\n"
        "Movement mode uses GemmaJev's local-view exploration memory.\n"
        "All three trajectories: the same GemmaJev checkpoint, trained with Tunix.\n"
        "Replay speed is accelerated and does not measure inference latency.\n"
        "Compare held-out predictions and matched rollouts; a successful replay alone "
        "does not establish a training benefit.\n"
        "Kakao fonts retain their own license and are not part of the public source bundle.\n"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob("*")
                if p.is_file()
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "replay_verified": True,
                "cases": [{"id": c["id"], **c["system"]["summary"]} for c in cases],
            }
        )
    )


if __name__ == "__main__":
    main()
