# Data and training

The selected model is Gemma 3 270M IT with a shared scalar candidate head.
For each question, the model encodes the state, question and each complete
candidate separately. The last valid token representation receives a scalar
score. Masked softmax and question-level cross entropy train the supplied-answer
distribution. The backbone and head are both trained; this recipe does not use RL.

## Prepare the inputs

Run the setup commands in the [README](../README.md) first:

```bash
source scripts/env.sh
.venv/bin/python examples/gemmajev/prepare_data.py
.venv/bin/python examples/gemmajev/prepare_maze_data.py
.venv/bin/python examples/gemmajev/check_inputs.py --config configs/baseline.json
.venv/bin/python examples/gemmajev/check_inputs.py --config configs/maze-expanded.json
```

The fetcher pins NanoJev source to commit
`76fdfc9ecdca45a9bcef17991a07d3041a87685a` and checks the data release manifest.
Model revisions are pinned in [model_registry.py](../jev_tunix/model_registry.py).

| Dataset | Train | Validation | Test |
| --- | --- | --- | --- |
| Original | 1,516 Maze + 2,174 ViZDoom | 496 Maze + 201 ViZDoom | Separate gameplay demos |
| Expanded | 13,516 Maze + 2,174 ViZDoom | 1,296 Maze + 201 ViZDoom | 1,420 Maze |

The expanded Maze set is newly generated; it is not the original 1,516 rows
concatenated with an extra 12,000. ViZDoom examples are preserved. Maze labels
come from visible adjacent cells, with rotated views and single-cell contrasts.
Training examples are sampled from whole maps, including boundaries and dead
ends. Validation and test maps are frozen first, and equivalent local patterns
under rotation/reflection are kept apart by the preparation script. Demo seeds
are excluded from these generated maps.

The original validation set contains correlated questions and local patterns
shared with its training set. The expanded split improves separation, but its
50×50 loop-maze validation subset contains only one map. Treat reported accuracy
as a limited classification check, not a broad navigation benchmark.

## Restore the selected training lineage

All runs use seed 17, batch size 8 and a maximum of 512 tokens per candidate.
Run each command in a **single-GPU environment**, with a fresh output folder:

```bash
.venv/bin/python examples/gemmajev/train.py \
  --config configs/baseline.json --output runs/baseline
.venv/bin/python examples/gemmajev/train.py \
  --config configs/maze-warmup.json --output runs/maze-warmup
.venv/bin/python examples/gemmajev/train.py \
  --config configs/maze-expanded.json --output runs/maze-expanded
```

| Config | Parent | Updates | Learning rate | Questions per batch |
| --- | --- | ---: | ---: | --- |
| [baseline](../configs/baseline.json) | Pretrained Gemma | 800 | 1e-5 | 4 Maze + 4 ViZDoom |
| [maze-warmup](../configs/maze-warmup.json) | Baseline | 240 | 1e-4 | 8 Maze, first 64 training questions |
| [maze-expanded](../configs/maze-expanded.json) | Warmup | 1,600 | 1e-5 | 4 expanded Maze + 4 ViZDoom |

The warmup began as a small training-set fit check. It is part of the selected
checkpoint's ancestry, so it is retained here for reproducibility. Its training
accuracy is not a performance result. Total ancestry is 2,640 updates. The last
run samples 6,400 Maze questions from the larger prepared pool.

Tunix's `PeftTrainer.with_loss_fn` accepts the candidate loss; the trainer manages
updates and checkpointing while the game interface remains outside it. The
script checks finite losses, records the batch schedule, hashes its inputs and
sources, and restores the saved checkpoint to check identical scores.
Historical measured summaries are under [results](../results). Hardware and
floating-point differences can affect a rerun.

To compare the baseline and continued model on frozen questions:

```bash
.venv/bin/python examples/gemmajev/evaluate_maze.py \
  --reference runs/baseline --run runs/maze-expanded \
  --output runs/evaluation
```
