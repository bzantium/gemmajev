# GemmaJev

Build a Jev-style game agent with **Gemma 3 270M IT** and **Tunix**. This project
adapts the two NanoJev README examples: aim and fire in ViZDoom, and explore a
50×50 maze using a local 5×5 view.

Gemma scores a supplied set of answers. A shared game controller turns those
probabilities into actions. Training uses a learned candidate head and Tunix SFT;
inference runs with JAX or locally on Apple silicon with MLX.

## Start reading

| Code | What it does |
| --- | --- |
| [game_contract.py](jev_tunix/game_contract.py) | Convert game questions into candidate inputs and normalized answers |
| [decision.py](jev_tunix/decision.py) | Gemma hidden states, candidate scoring head, supervised loss |
| [prepare_data.py](examples/gemmajev/prepare_data.py) | Prepare the original two-game training data |
| [prepare_maze_data.py](examples/gemmajev/prepare_maze_data.py) | Generate additional local Maze observations with checked labels |
| [train.py](examples/gemmajev/train.py) | Train with Tunix and verify checkpoint restoration |
| [game_inference.py](jev_tunix/game_inference.py) | Restore the model and answer game requests with JAX |
| [game_mlx.py](jev_tunix/game_mlx.py) | Run the same candidate model on an Apple GPU |
| [rollout_mlx.py](examples/gemmajev/rollout_mlx.py) | Solve three fixed mazes and save complete trajectories |

For Maze, the questions ask whether the cells north, east, south and west are
open. Each has `true` and `false` candidates. The controller remembers explored
edges and chooses where to move. Gemma never receives the audience's full map.
For ViZDoom's upstream `basic` scenario, the candidates are left, right, shoot
and noop. This interface returns candidate probabilities, rather than generating
an open-ended chat response.

## Reproduce

Use Python 3.12 for training and JAX inference. Run these commands from the
repository root. `env.sh` keeps downloads, caches and temporary files here.

```bash
source scripts/env.sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.lock.txt
.venv/bin/python -m pip install -e '.[dev,games,demo]'
.venv/bin/python examples/gemmajev/fetch_resources.py
.venv/bin/python scripts/download_model.py --model gemma-3-270m-it
```

Gemma downloads require access under the model's terms and Hugging Face
authentication. Supply credentials through your environment; keep them out of Git.
The GPU lock records the Linux CUDA training environment. MLX uses a separate
[Mac environment](docs/mlx.md), tested with Python 3.13.

1. [Prepare data and train](docs/training.md).
2. [Run the games and build browser replays](docs/demos.md).
3. [Export the trained model and run it with MLX](docs/mlx.md).

Checkpoints, datasets, generated recordings and presentation assets are not
included in this repository. The commands fetch or create the required artifacts.

## Recorded results

The selected model continues from the initial two-game model, through a short
Maze warmup, then expanded Maze data with ViZDoom rehearsal. The exact lineage
and configs are documented in [training](docs/training.md).

| Check | Selected model |
| --- | ---: |
| Expanded Maze validation question accuracy | 80.6% |
| ViZDoom validation question accuracy | 88.1% |
| Three fixed 50×50 mazes, MLX FP32 | 3/3 reached the goal |
| One Maze observation, MLX FP32 on M2 Max | 54.4 ms |

The latency is the mean of ten warm calls on one observation, including all four
direction questions. The complete three-maze exploration took 53.2 seconds,
excluding loading. It is separate from accelerated video playback.

These demos are fixed illustrations, not an aggregate comparison against Jev or
NanoJev. The controller contributes substantially to navigation. Validation
windows are correlated and the 50×50 loop-maze subset is small, so question
accuracy is not evidence of broad generalization or efficient pathfinding.
Machine-readable measurements are in [results](results).

## Checks

After installing the training dependencies:

```bash
source scripts/env.sh
.venv/bin/python examples/gemmajev/fetch_resources.py --source-only
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check .
.venv/bin/python -m build
```

Tests cover candidate encoding, capacity checks, masked probabilities, observation
formatting, sampling and spatial labels. They do not require pretrained weights
or a GPU. Full training and gameplay are separate checks.

## Attribution

Inspired by [Jev](https://www.typesafe.ai/) and
[NanoJev](https://github.com/TianyuCodings/NanoJev); see also
[JevHarness](https://github.com/TianyuCodings/JevHarness).
Built with [Gemma](https://ai.google.dev/gemma),
[Tunix](https://github.com/google/tunix), and
[MLX](https://github.com/ml-explore/mlx).

Project code is **Apache-2.0**. Gemma weights, public datasets and third-party
code keep their own terms. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
This is an independent reproduction of the public examples, with no claim to
Jev's proprietary training method or affiliation.
