"""Record the verified replay UI, including its source and replay labels."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def clean_start(raw, reference, destination):
    """Trim only startup, locating three frames that match the ready UI."""
    size = (320, 200)
    target = np.asarray(
        Image.open(reference).convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.int16
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(raw),
        "-frames:v",
        "200",
        "-vf",
        "scale=320:200:flags=lanczos",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    consecutive, first_ready, errors = 0, None, []
    try:
        for index in range(200):
            block = process.stdout.read(size[0] * size[1] * 3)
            if len(block) != size[0] * size[1] * 3:
                break
            frame = np.frombuffer(block, np.uint8).reshape(200, 320, 3).astype(np.int16)
            error = float(np.abs(frame - target).mean())
            errors.append(error)
            consecutive = consecutive + 1 if error < 3.0 else 0
            if consecutive == 3:
                first_ready = index
                break
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()
    if first_ready is None:
        raise ValueError(
            f"No stable ready frame found; minimum image error: {min(errors, default=-1)}"
        )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(raw),
            "-vf",
            f"trim=start_frame={first_ready},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )
    block = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(destination),
            "-frames:v",
            "1",
            "-vf",
            "scale=320:200:flags=lanczos",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    first = np.frombuffer(block, np.uint8).reshape(200, 320, 3).astype(np.int16)
    error = float(np.abs(first - target).mean())
    if error >= 3.0:
        raise ValueError("Encoded first frame no longer matches the ready UI")

    def frames(path):
        value = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )
        return int(value["streams"][0]["nb_read_frames"])

    before, after = frames(raw), frames(destination)
    if before - first_ready != after:
        raise ValueError("Unexpected frame loss after trimming startup")
    return dict(
        removed_startup_frames=first_ready,
        source_frames=before,
        output_frames=after,
        first_frame_image_error=error,
        first_frame_matches_ready_ui=True,
        gameplay_frames_preserved=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8794")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root):
        raise ValueError("Recordings must remain under the project")
    output.mkdir(parents=True, exist_ok=False)
    errors, receipts = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        for game in ("basic", "maze"):
            context = browser.new_context(
                viewport={"width": 1600, "height": 1000},
                record_video_dir=str(output),
                record_video_size={"width": 1600, "height": 1000},
            )
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url)
            page.wait_for_function("window.demoReady === true")
            page.click(f'[data-game="{game}"]')
            page.evaluate("document.fonts.ready")
            page.screenshot(path=str(output / f"{game}-ready.png"))
            page.wait_for_timeout(1500)
            assert page.locator(".panel h2").all_text_contents() == ["Jev", "NanoJev", "GemmaJev"]
            assert page.locator(".tag").inner_text() == "Recorded gameplay"
            page.click("#play")
            if game == "maze":
                page.wait_for_timeout(8500)
                page.screenshot(path=str(output / "maze-gemma-complete.png"), full_page=True)
                page.select_option("#speed", "20")
            page.wait_for_function("tick >= Number(timeline.max)", timeout=45000)
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            assert not page.evaluate("document.documentElement.scrollHeight > innerHeight")
            outcomes = page.locator(".outcome").all_text_contents()
            assert outcomes[-1] == ("Target hit" if game == "basic" else "Exit reached")
            page.screenshot(path=str(output / f"{game}-complete.png"), full_page=True)
            receipts.append(dict(game=game, outcomes=outcomes, no_overflow=True))
            page.wait_for_timeout(2000)
            video = page.video
            context.close()
            video.save_as(str(output / f"raw-{game}.webm"))
            video.delete()
        browser.close()
    assert not errors, errors
    for game in ("basic", "maze"):
        receipt = clean_start(
            output / f"raw-{game}.webm", output / f"{game}-ready.png", output / f"{game}.mp4"
        )
        receipts.append(dict(game=game, startup_cleanup=receipt))
    report = dict(
        browser_errors=errors,
        checks=receipts,
        playback="Basic at 35 ticks/s; Maze at 30 attempts/s then 20x. Not inference latency.",
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    )
    (output / "recording.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(receipts))


if __name__ == "__main__":
    main()
