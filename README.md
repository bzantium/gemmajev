# GemmaJev

**Jev-style game decisions with Gemma 3 270M IT and Tunix.**

Give the model an observation, a question and possible answers. Get answer
probabilities without decoding response tokens.
Inspired by [NanoJev](https://github.com/TianyuCodings/NanoJev).

[Watch the demos](#demos) · [Train a model](docs/training.md) · [Run on a Mac](docs/mlx.md) · [Input and output](docs/interface.md)

## Demos

### ViZDoom · Aim, then fire

[![Jev, NanoJev and GemmaJev aiming and firing in ViZDoom](demos/media/vizdoom.gif)](demos/media/vizdoom.mp4)

The model scores **left, right, shoot and noop** from a text observation.
The three panels replay Jev, NanoJev and our trained Gemma on the same scenario.

### Maze · Find the exit

[![Jev, NanoJev and GemmaJev exploring a 50×50 maze](demos/media/maze.gif)](demos/media/maze.mp4)

Gemma sees a **5×5 local window** and estimates whether each adjacent cell is
open. The shared controller explores and remembers paths; the full map is for
viewers. [Three more mazes with the continued model →](demos/media/three-mazes.mp4)

Watch all three recordings locally with just Python:

```bash
python3 -m http.server 8000 --directory demos
```

Open **http://localhost:8000**. No model download or GPU needed for playback.
The two comparison clips use the initial checkpoint; the three-maze clip uses
the continued checkpoint. Videos are accelerated recordings, not live inference.

## How it works

```text
Observation + question + candidates
                ↓
       Gemma → scoring head → probabilities → game controller
```

[model.py](gemmajev/model.py) adds a shared scoring head to Gemma's final hidden
states. [train.py](scripts/train.py) trains the backbone and head with Tunix's
custom-loss interface. The same checkpoint runs through JAX or an MLX export.

```python
import json
from pathlib import Path
from gemmajev.mlx_backend import MLXGameEngine

engine = MLXGameEngine("artifacts/maze-expanded-mlx")
request = json.loads(Path("examples/maze_request.json").read_text())
response = engine.predict(request)
```

See the [actual request and response](docs/interface.md) and
[Mac setup](docs/mlx.md). Weights are generated locally and are not bundled.

## Train and run

Start with [setup and training](docs/training.md), then
[run the games](docs/demos.md). The recipe prepares NanoJev data, adds local Maze
examples, and trains Gemma on one GPU. Checkpoints, data and caches stay inside
this repository.

The continued model reached **80.6% Maze** and **88.1% ViZDoom** validation question
accuracy. On an M2 Max, MLX FP32 took **54.4 ms** per local observation and completed
all three fixed mazes. See [measurements and limitations](docs/results.md) for
sample sizes, controller behavior and timing details.

## Code

```text
gemmajev/      Model, question interface, JAX and MLX inference
scripts/       Fetch data, prepare inputs, train and evaluate
examples/      Run games or send a single request
demos/         Ready-to-watch recordings, no model required
web/           Templates for replaying your own game traces
tools/         Benchmarks, replay builders and video recording
```

[Development](docs/development.md) · [Training configs](configs)

## Acknowledgments

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[NanoJev](https://github.com/TianyuCodings/NanoJev) and
[JevHarness](https://github.com/TianyuCodings/JevHarness) inspired this project.
Built with [Gemma](https://ai.google.dev/gemma),
[Tunix](https://github.com/google/tunix) and [MLX](https://github.com/ml-explore/mlx).

Code: [Apache-2.0](LICENSE). Models, game assets and datasets retain their own
terms; see [NOTICE](NOTICE) and [demo credits](demos/README.md).
