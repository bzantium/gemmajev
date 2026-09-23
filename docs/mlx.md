# Run the trained model on a Mac

These measurements use the `maze-expanded` local-safety checkpoint on an Apple
M2 Max GPU. They do not measure the movement checkpoint in the three-maze video.
Use **FP32** by default: it closely matches the original candidate probabilities,
while FP16 and 8-bit weights offered little additional speed in this measurement.
Export preserves the trained weights; it does not perform additional training.

The latest movement checkpoint is available as a [verified MLX download](huggingface.md#mlx).
It matches 40 Tunix reference decisions and averages 35.1 ms for one movement
question on one timed observation. The results below concern four local-safety
questions per observation, so the two timing numbers describe different workloads.

## Results

Each latency measurement includes tokenization, scoring all four Maze direction
questions, GPU synchronization and answer construction. It is the mean of ten
warm calls on one observation, not a broad latency benchmark.

| Runtime | Mean per observation | Changed choices on 40 reference questions | Largest probability difference |
| --- | ---: | ---: | ---: |
| MLX FP32 | 54.4 ms | 0 | 0.0000122 |
| MLX FP16 | 49.0 ms | 0 | 0.0637 |
| MLX 8-bit weights, FP16 activations | 52.3 ms | 0 | 0.0688 |

The earlier compact JAX CPU measurement was 1.26 seconds on eight Intel Xeon
cores. That is a different machine and backend; this is not an isolated software
speed comparison. The MLX machine is an M2 Max with 32 GiB memory.

Both FP32 and FP16 finished all three fixed maps with the original controller,
geometry and attempt limits. Independent replay checks verified the recorded
observations, decisions and environment transitions.

| Maze | Original GPU recording: attempts / wall hits | MLX FP32 | MLX FP16 |
| --- | ---: | ---: | ---: |
| 1 | 627 / 163 | 624 / 160 | 626 / 162 |
| 2 | 401 / 108 | 400 / 107 | 399 / 106 |
| 3 | 751 / 253 | 751 / 253 | 753 / 255 |

Sequential exploration took 53.2 seconds in FP32 and 48.0 seconds in FP16,
excluding model loading. Small trajectory differences can follow from different
floating-point arithmetic. They are not new training gains. The 8-bit version
was checked on the 40 questions only; it was not selected for full rollouts.
The bundled three-maze video uses a different checkpoint and controller. Its
playback duration is not an inference-time measurement.

## Run with a prepared Mac environment

From the project directory:

```bash
source scripts/env.sh
.venv-mlx/bin/python scripts/fetch_resources.py --source-only
.venv-mlx/bin/python examples/prepare_mazes.py
.venv-mlx/bin/python examples/maze_mlx.py \
  --model artifacts/maze-expanded-mlx \
  --precision float32 \
  --output runs/mlx-three-mazes
```

Choose a fresh output directory. This executes the model and saves complete
trajectories; it does not play a prerecorded video.

To repeat the output and latency check:

```bash
source scripts/env.sh
.venv-mlx/bin/python tools/validate_mlx.py \
  --model artifacts/maze-expanded-mlx \
  --output runs/mlx-check
```

The optional `--precision float16 --bits 8` evaluates post-training 8-bit
quantization. The small scoring head stays FP32. Quantization occurs in memory
from the verified exported weights and does not overwrite them.

## Recreate the environment and export

The Mac inference environment was tested with Python 3.13, MLX 0.32.2,
MLX-LM 0.29.1 and Transformers 4.57.6. It is separate from the Python 3.12
Tunix/JAX training environment. All packages and caches stay under the project:

```bash
source scripts/env.sh
export PIP_CACHE_DIR="$JEV_PROJECT_ROOT/.cache/pip"
python3.13 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements/mlx.lock.txt
```

Export a completed checkpoint in the existing Tunix environment:

```bash
source scripts/env.sh
JAX_PLATFORMS=cpu .venv/bin/python scripts/export_mlx.py \
  --run runs/maze-expanded \
  --output artifacts/maze-expanded-mlx
```

The exporter restores the trained backbone and learned head, converts projection
layouts, preserves all 268,098,816 parameters for this checkpoint, and writes
FP32 safetensors. It also saves reference inputs, exact token IDs, Tunix scores,
checkpoint hashes and an export manifest. Copy the complete export directory
to the Mac project. The exported model retains the Gemma weight terms.

The MLX adapter uses the same tokenizer, complete candidate strings, causal
attention, last-valid-token pooling and candidate softmax. Its embedding scaling
matches Tunix's activation dtype instead of MLX-LM's BF16-rounded constant. It
does not use the vocabulary generation head. The original JAX inference path
remains available.

Measured summaries are in [results](results/). Model exports and full trajectories are generated locally and are not included in Git.
