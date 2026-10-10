# vLLM serving

GemmaJev uses a learned scalar head on the final valid token of each candidate
input. It must run as a **pooling / classification model**, preserving that head
and the training prompt. A causal-LM server reading answer-token probabilities
would perform a different computation.

## Status of the new decision API

Checked on October 11, 2026 against vLLM main commit
`276fbcff2717bd934cfa37c8a2e4c391f3e7237b` and release `v0.31.0`.

- [`/v1/systemone`](https://github.com/vllm-project/vllm/pull/59299) merged on
  October 8, after the v0.31.0 release. Installing v0.31.0 does not add this route.
- The development branch's
  [read-strategy registry](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/entrypoints/generate/structured_decisions/strategies.py)
  supports selected Qwen architectures and DiffusionGemma. It does not include
  `Gemma3TextForSequenceClassification` or a generic candidate-head strategy.
- The separate [`/v1/decisions` proposal](https://github.com/vllm-project/vllm/pull/60465)
  was still open at this check.

The new official decision route therefore cannot serve this checkpoint directly.
Its request also uses one `state` and `questions`, whereas the game adapter accepts
`states: [...]`. Renaming fields would not resolve the model/readout difference.
Supporting that route would require a candidate-scoring read strategy and pooling
engine integration upstream or in an extension. No retraining is needed for the
pooling route below.

## Run the pooling server

Use the **Transformers export**, not the MLX files. Follow the
[model download guide](huggingface.md#transformers) first. Keep the vLLM environment
separate from the pinned Tunix training environment because their dependencies
differ.

```bash
source scripts/env.sh
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python -r requirements/vllm.txt

export VLLM_CACHE_ROOT="$JEV_PROJECT_ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JEV_PROJECT_ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$JEV_PROJECT_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$JEV_PROJECT_ROOT/.cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$JEV_PROJECT_ROOT/.cache/flashinfer"
```

The original export uses Transformers' `Gemma3TextForSequenceClassification`
architecture and keeps weights under `model.*` plus `score.weight`. vLLM 0.31.0's
automatic resolution selects a bare Gemma backbone for that name and fails while
loading the `model.*` prefix. Select its wrapped Gemma classification model at
launch; this leaves the exported config and weights unchanged:

```bash
.venv-vllm/bin/vllm serve artifacts/gemma-3-270m-jev \
  --served-model-name gemmajev \
  --host 127.0.0.1 --port 8000 \
  --runner pooling --convert classify \
  --hf-overrides '{"architectures":["Gemma3ForSequenceClassification"]}' \
  --pooler-config '{"seq_pooling_type":"LAST","use_activation":false}' \
  --dtype float32 --attention-backend FLEX_ATTENTION \
  --max-model-len 512 --max-num-seqs 32 \
  --gpu-memory-utilization 0.1 --enforce-eager
```

The model remains a scalar candidate scorer. The override does not restore or
use a language-model output head.

## Send a game request

The supplied client sends the exact chat-templated **token IDs** from training to
[`/classify`](https://docs.vllm.ai/en/v0.31.0/models/pooling_models/classify/), with
`use_activation: false`. vLLM returns one raw scalar per candidate in the API's
`probs` field. The client applies softmax across the two to four candidates for
that question and builds the existing game response.

With the server running, use the repository's inference environment:

```bash
source scripts/env.sh
.venv-hf/bin/python examples/predict.py \
  --backend vllm \
  --model artifacts/gemma-3-270m-jev \
  --base-url http://127.0.0.1:8000 \
  --request examples/navigation_request.json
```

`--served-model` defaults to `gemmajev`. If the server requires authentication,
set `VLLM_API_KEY` in the client environment. The client reports HTTP request count;
it does not infer the server's GPU forward-pass count.

Check the server against the export's frozen reference inputs:

```bash
.venv-hf/bin/python tools/validate_vllm.py \
  --model artifacts/gemma-3-270m-jev \
  --base-url http://127.0.0.1:8000 \
  --output artifacts/vllm-validation.json
```

This checks token identity, selected candidates and FP32 probabilities. It is a
runtime compatibility check, not a new gameplay evaluation.

## Verified run

On one NVIDIA B300, with the FP32 command above:

- All 40 frozen reference questions kept the same selected candidate. Maximum
  absolute probability difference from Tunix: `3.40e-6`.
- The movement, ViZDoom and Boolean Maze requests also matched the Transformers
  reference: six questions, maximum probability difference `9.50e-7`.
- A four-candidate movement request took **57.8 ms median** over ten warm,
  sequential HTTP calls on the server host. This measurement uses pretokenized
  inputs, FP32 and eager execution; it excludes client tokenization, game logic
  and network travel from another machine. It is not an optimized latency claim.

[Recorded versions, settings and results](results/navigation-vllm-fp32.json).
The server was stopped after verification. No training or weight changes were
needed. The client explicitly requests token lists so both Transformers 4.x and
5.x preserve the training inputs.

`FLEX_ATTENTION` is intentional for this FP32 check: the v0.31.0 Triton attention
kernel exceeded the B300's per-block shared-memory limit at Gemma's head size.
