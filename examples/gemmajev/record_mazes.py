"""Record the audited three-maze viewer and remove only browser startup frames."""

import argparse
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from record_demo import clean_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8796")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True, help="Actual rollout summary.json to verify")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    summary = (root / args.summary).resolve()
    if not summary.is_relative_to(root):
        raise ValueError("Keep source evidence inside the project")
    evidence = json.loads(summary.read_text())
    if len(evidence) != 3 or any(r["status"] not in ("goal", "horizon_exhausted") for r in evidence):
        raise ValueError("Expected three completed or step-limited recorded runs")
    expected_outcomes = ["Exit reached" if r["status"] == "goal" else "Step limit reached"
                         for r in evidence]
    expected_metrics = [f"{r['attempts']} attempts · {r['collisions']} wall hits" for r in evidence]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Keep artifacts under the project root")
    output.mkdir(parents=True, exist_ok=False)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1920, "height": 1080},
                                      record_video_dir=str(output),
                                      record_video_size={"width": 1920, "height": 1080})
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.wait_for_function("window.demoReady === true")
        assert page.locator('.panel h2').all_text_contents() == ["Maze 1", "Maze 2", "Maze 3"]
        assert not page.evaluate("document.documentElement.scrollHeight > innerHeight")
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path=str(output / "three-mazes-ready.png"))
        page.wait_for_timeout(1500)
        page.click('#play')
        page.wait_for_timeout(12000)
        page.screenshot(path=str(output / "three-mazes-progress.png"))
        page.wait_for_function("tick >= maximum", timeout=20000)
        outcomes = page.locator('.outcome').all_text_contents()
        assert outcomes == expected_outcomes, outcomes
        metrics = page.locator('.metric').all_text_contents()
        assert metrics == expected_metrics, metrics
        page.screenshot(path=str(output / "three-mazes-complete.png"))
        page.wait_for_timeout(2500)
        recording = page.video
        context.close()
        recording.save_as(str(output / "raw-three-mazes.webm"))
        recording.delete()
        browser.close()
    assert not errors, errors
    cleanup = clean_start(output / "raw-three-mazes.webm", output / "three-mazes-ready.png",
                          output / "three-mazes.mp4")
    report = dict(browser_errors=errors, outcomes=outcomes, metrics=metrics, no_overflow=True,
                  summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                  startup_cleanup=cleanup, playback="Each complete recorded run is accelerated to 24 seconds; rates differ.",
                  sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
