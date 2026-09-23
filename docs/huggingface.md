# Download a trained model

Both repositories contain the same `navigation-rehearsal` checkpoint, including
the trained backbone and scalar candidate head. No retraining is needed to run it.

| Repository | Runtime | Weight format |
| --- | --- | --- |
| [bzantium/gemma-3-270m-jev](https://huggingface.co/bzantium/gemma-3-270m-jev) | Transformers, CPU or CUDA | FP32 `model.safetensors` |
| [bzantium/gemma-3-270m-jev-mlx](https://huggingface.co/bzantium/gemma-3-270m-jev-mlx) | MLX, Apple Silicon | FP32 backbone and head safetensors |

The repositories are private during code review. Authenticate with a Hugging Face
account that has access. Model weights use the Gemma terms; the code is Apache-2.0.

## Transformers

From this source repository, create a separate inference environment:

```bash
source scripts/env.sh
python3.12 -m venv .venv-hf
.venv-hf/bin/python -m pip install -r requirements/transformers.txt
.venv-hf/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "bzantium/gemma-3-270m-jev",
    local_dir="artifacts/gemma-3-270m-jev",
)
PY
.venv-hf/bin/python examples/predict.py \
  --backend transformers --model artifacts/gemma-3-270m-jev \
  --request examples/navigation_request.json
```

Use `--request examples/doom_request.json` for Doom. Add `--device cuda` for a
CUDA-enabled PyTorch installation. The default is CPU FP32 with eight threads; adjust with `--threads`.

The checkpoint loads with `AutoModelForSequenceClassification`. Its single
output is a **scalar candidate score**. The adapter batches candidates and applies
softmax across their scores. Do not use a text-generation pipeline or interpret
the single score as an action probability by itself.

## MLX

Use the [Mac environment](mlx.md#recreate-the-environment-and-export), then:

```bash
source scripts/env.sh
.venv-mlx/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "bzantium/gemma-3-270m-jev-mlx",
    local_dir="artifacts/gemma-3-270m-jev-mlx",
)
PY
.venv-mlx/bin/python examples/predict.py \
  --backend mlx --model artifacts/gemma-3-270m-jev-mlx \
  --request examples/navigation_request.json
```

The export uses `MLXGameEngine`, not `mlx_lm.generate`. It retains all trained
parameters in FP32; it is not quantized. Both repositories also bundle the small
inference adapter and request examples for use without the training code.

## Reproduce the conversion

From a completed Tunix run:

```bash
.venv/bin/python scripts/export_mlx.py \
  --run runs/navigation-rehearsal --output artifacts/navigation-mlx
.venv/bin/python scripts/export_transformers.py \
  --source artifacts/navigation-mlx --output artifacts/navigation-transformers
.venv-hf/bin/python tools/validate_transformers.py \
  --model artifacts/navigation-transformers --output runs/transformers-check
```

For Mac verification, copy the MLX export into the Mac project and run
`tools/validate_mlx.py`. Both checks compare 40 frozen Tunix reference questions:
32 Maze movement and 8 Doom. They check input token IDs, top-ranked candidates
and numerical probability differences; this is conversion verification, not a
new gameplay benchmark. Export manifests record the source checkpoint hashes.

The [training data and limitations](navigation.md) apply to both exports. The
three recorded Maze maps are training examples. See [input/output](interface.md)
for exact requests and the distinction between model scores and controller actions.
