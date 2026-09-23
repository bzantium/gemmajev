"""Verify real Gemma trajectories and assemble attributed three-panel replays."""

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references/nanojev-upstream/scripts"))
import build_arcade_demo as arcade  # noqa: E402
import build_shooting_demo as shooting  # noqa: E402
from build_predict_position_demo import PredictCapture, add_events  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font-dir", type=Path, help="Optional licensed Kakao font directory")
    args = parser.parse_args()
    run, output = [(ROOT / p).resolve() for p in (args.rollout, args.output)]
    if not all(p.is_relative_to(ROOT) for p in (run, output)):
        raise ValueError("Artifacts must remain under the project")
    output.mkdir(parents=True, exist_ok=False)
    public = ROOT / "data/nanojev-public/demonstrations/web/dev"
    basic = json.loads((public / "shooting_results.json").read_text())
    basic_case = copy.deepcopy(
        next(c for c in basic["cases"] if c["id"] == "test-appo_basic-9030060")
    )
    basic_case["systems"] = [s for s in basic_case["systems"] if s["id"] in ("jev", "nanojev")]
    shooting.MODEL_TEXT["gemmajev"] = ("GemmaJev", "Gemma 3 270M IT trained with Tunix")
    atlas = shooting.AtlasWriter(output / "media", "gemmajev_basic")
    system, basic_audit, deadline = shooting.render_episode(
        json.loads((run / "basic.json").read_text()),
        "gemmajev",
        atlas,
        capture_factory=PredictCapture,
    )
    assert deadline == basic_case["max_ticks"]
    basic_case["systems"].append(add_events(system))
    for baseline in basic_case["systems"][:2]:
        for name in {f["sprite"]["src"] for f in baseline["frames"]}:
            shutil.copy2(ROOT / "references/nanojev-upstream/web/dev" / name, output / name)
    maze_data = json.loads((public / "side_by_side_results.json").read_text())
    maze_case = copy.deepcopy(next(c for c in maze_data["examples"] if c["id"] == arcade.MAZE_CASE))
    maze_case["systems"] = [s for s in maze_case["systems"] if s["id"] in ("jev", "nanojev")]
    arcade.COLORS["gemmajev"] = "#7357e8"
    initial, system, maze_audit, _ = arcade.maze_system(
        run / "maze.json", "gemmajev", "GemmaJev", "Gemma local judgments + shared exploration"
    )
    for key, value in maze_case["initial"].items():
        assert value == initial[key], key
    maze_case["systems"].append(system)
    for case in (basic_case, maze_case):
        for system in case["systems"]:
            system["provenance"] = (
                "Our trained Gemma, recorded gameplay"
                if system["id"] == "gemmajev"
                else "Public NanoJev recording"
            )
    payload = dict(
        basic=basic_case,
        maze=maze_case,
        source="https://github.com/TianyuCodings/NanoJev",
        note="Fixed README illustrations, not aggregate benchmark results. All panels are replays.",
    )
    (output / "demo.json").write_text(json.dumps(payload) + "\n")
    (output / "audit.json").write_text(
        json.dumps(dict(basic=basic_audit, maze=maze_audit), indent=2) + "\n"
    )
    for name in ("index.html", "demo.css", "demo.js"):
        shutil.copy2(ROOT / "web/games" / name, output / name)
    if args.font_dir:
        fonts = args.font_dir.resolve()
        if not fonts.is_relative_to(ROOT):
            raise ValueError("Font assets must remain under the project")
        (output / "fonts").mkdir()
        for name in ("KakaoSmall.woff2", "KakaoBig-Bold.woff2"):
            shutil.copy2(fonts / name, output / "fonts" / name)
    else:
        # Public source does not redistribute presentation fonts.
        css = output / "demo.css"
        css.write_text(re.sub(r"@font-face\s*\{[^}]*\}\s*", "", css.read_text()))

    shutil.copy2(ROOT / "references/nanojev-upstream/LICENSE", output / "NANOJEV-LICENSE")
    (output / "NOTICE.txt").write_text(
        "NanoJev source and public Jev/NanoJev recordings: TianyuCodings/NanoJev (MIT).\n"
        "GemmaJev adaptation and recordings: this project (Apache-2.0 code).\n"
        "Gemma weights remain subject to Gemma terms. Kakao fonts retain their own license.\n"
        "This presentation preview is not a redistribution clearance for font assets.\n"
    )
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                replay_verified=True,
                basic_success=basic_case["systems"][-1]["success"],
                maze_outcome=maze_case["systems"][-1]["summary"]["outcome"],
            )
        )
    )


if __name__ == "__main__":
    main()
