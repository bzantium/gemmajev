# Run and replay the games

Prepare resources and train or restore a compatible checkpoint first, following
the [README](../README.md) and [training guide](training.md).

To watch the bundled recordings without installing dependencies, run
`python3 -m http.server 8000 --directory demos`. The steps below execute the model.

## The two NanoJev examples

```bash
source scripts/env.sh
.venv/bin/python examples/games.py \
  --run runs/maze-expanded --output runs/two-games
.venv/bin/python tools/build_demo.py \
  --rollout runs/two-games --output demo-output
.venv/bin/python -m http.server 8794 --directory demo-output
```

Open http://localhost:8794. The viewer compares public Jev/NanoJev recordings
with the trained model's saved trajectory. These are replays, not three live
services. The builder independently checks the environment transitions.
ViZDoom requires the `games` extra. Default replay fonts use system fallbacks;
optional font files are not distributed here.

## Three fixed mazes

The seeds were fixed before inference, without selecting successful outcomes:
`2026121201`, `2026121202`, `2026121203`. Each generates a 50×50 loop maze.

```bash
.venv/bin/python examples/prepare_mazes.py
.venv/bin/python examples/maze.py \
  --run runs/maze-expanded \
  --cases data/gemmajev-three-mazes/cases.jsonl \
  --output runs/three-mazes
.venv/bin/python tools/build_maze_triptych.py \
  --rollout runs/three-mazes --output demo-output-mazes
.venv/bin/python -m http.server 8796 --directory demo-output-mazes
```

For Apple GPU inference, export the checkpoint and use `examples/maze_mlx.py` as
described in [Mac inference](mlx.md). Its output works with the same replay
builder. Prepare the three cases before running it.

## Record video

Browser recording needs the `demo` extra, Playwright Chromium and `ffmpeg` on
PATH. Keep any locally downloaded tools inside the project directory.

```bash
source scripts/env.sh
.venv/bin/python -m playwright install chromium
# Keep the maze replay server running in another terminal.
.venv/bin/python tools/record_mazes.py \
  --url http://127.0.0.1:8796 \
  --summary runs/three-mazes/summary.json \
  --output demo-output-recording
```

For the two-game viewer, use `record_demo.py --url http://127.0.0.1:8794
--output demo-output-two-games-video`. Recording scripts verify ready frames
before exporting. Playback duration is not model inference latency.

## CPU check

`GameEngine(run, compact=True)` removes padded candidates and rounds the actual
token capacity to small buckets without truncating inputs. The same checkpoint
can run on CPU:

```bash
JAX_PLATFORMS=cpu .venv/bin/python tools/benchmark_cpu.py \
  --run runs/maze-expanded --cores 8 --output runs/cpu-check
```

The recorded eight-core Xeon measurement fell from 7.87 seconds to 1.26 seconds
per observation after compact batching. These are warm measurements on a small
fixed input set. See the separate MLX measurements for the M2 Max.
