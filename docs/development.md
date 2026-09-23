# Development

Use the Python 3.12 environment from [training setup](training.md). From the
repository root:

```bash
source scripts/env.sh
.venv/bin/python scripts/fetch_resources.py --source-only
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m build
```

The unit tests need the pinned NanoJev source, but no weights or GPU. They cover
candidate encoding, capacity limits, probability formatting, observation layouts,
sampling and spatial labels. Training, output parity and gameplay are separate
checks described in the task guides.

## Where to make a change

| Area | Files |
| --- | --- |
| Gemma forward pass and supervised loss | `gemmajev/model.py` |
| Request encoding and answer probabilities | `gemmajev/interface.py`, `gemmajev/tokenization.py` |
| Checkpoint loading and evaluation helpers | `gemmajev/runtime.py` |
| Inference | `gemmajev/jax_backend.py`, `gemmajev/mlx_backend.py` |
| Data and training workflow | `scripts/`, `configs/` |
| Game execution | `examples/games.py`, `examples/maze.py`, `examples/maze_mlx.py` |
| Replay templates and recording tools | `web/`, `tools/` |

`source scripts/env.sh` sets `PYTHONPATH` for direct script execution and keeps
caches and temporary files inside the repository. The MLX environment uses Python
3.13 and its own dependency lock; it does not install the JAX training package.

The `demos/` page is a static video gallery. The `web/` directories are templates
used by the replay builders to visualize newly generated trajectories. Neither
requires a frontend build system.
