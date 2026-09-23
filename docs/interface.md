# An observation in, probabilities out

GemmaJev scores supplied candidates. It does not decode a JSON answer or generate
a route through the maze. The current movement model and the original
local-safety model use different questions:

| Task | Questions per state | Candidates per question |
| --- | ---: | --- |
| Maze movement | One next-direction question | `north`, `east`, `south`, `west` |
| Maze local safety | Four questions, one per adjacent cell | `false`, `true` |
| ViZDoom | One next-action question | `left`, `noop`, `right`, `shoot` |

## How a response is built

1. Build one input sequence per candidate: state, question and complete candidate.
2. Apply Gemma's chat template and score the sequences together in a batch.
3. Pool the last valid token and apply the shared scalar head.
4. Normalize the candidate scores with softmax.
5. Pair the numbers with the original candidate names in Python.

One batch uses one forward pass, with no autoregressive response decoding. This
still processes a separate input sequence for each candidate. More questions
are split into batches by `batch_questions`; this is not a constant-cost operation.
Padding candidates are masked. Inputs over the trained 512-token capacity are
rejected rather than truncated.

The response-building part of [format_answer](../gemmajev/interface.py), with
validation omitted here, is:

```python
answer = {
    "type": kind,
    "probabilities": dict(zip(keys, values.tolist(), strict=True)),
}
if kind == "boolean":
    answer["p_true"] = answer["probabilities"]["true"]
```

The backend keeps state and question IDs around this answer. `json.dumps(response)`
serializes the resulting dictionary; the model never generates the field names.
The game controller separately decides how to act on these probabilities.

## Maze movement

The [complete request](../examples/navigation_request.json) is the first decision
in the three-maze recording. The model sees this local state plus directional
visit and attempt history:

```text
Position: row 31, column 9.
Goal offset: -30 rows south, -8 columns east.
Previous move: none. Current visits: 1.
Local map:
.#.#.
.#.#.
.#A#.
.#.##
...#.
Available directions: north, south.
```

The question asks for the next movement direction toward the goal. The final
`navigation-rehearsal` checkpoint produced these [recorded probabilities](../examples/navigation_response.json):

```json
{
  "type": "choice",
  "probabilities": {
    "east": 0.000259,
    "north": 0.645754,
    "south": 0.352776,
    "west": 0.001211
  }
}
```

Values above are rounded. The controller chooses north from the available moves.
It tracks explored branches and handles forced moves and backtracking without
calling the model. **This map is included in training.** The example explains the
interface; it is not an unseen-map evaluation.

## ViZDoom

The [complete request](../examples/doom_request.json) contains text describing
visible object boxes, ammunition, remaining time and recent observations. It does
not contain an image. Each action has a description, such as "Strafe right for 4
Doom ticks."

The final checkpoint's [recorded first response](../examples/doom_response.json)
assigns about 0.999987 to `right` on the fixed `test-appo_basic-9030060` case.
The recorded controller moves right. This is a model score, not a verified
99.9987% chance of eventual success. The Doom controller uses a seeded
0.1 epsilon-greedy policy, so its action sampling distribution is distinct from
the model's candidate probabilities.

## Run either request

After [training the movement checkpoint](navigation.md), use the training Python
environment:

```bash
source scripts/env.sh
.venv/bin/python examples/predict.py \
  --backend jax --model runs/navigation-rehearsal \
  --request examples/navigation_request.json
.venv/bin/python examples/predict.py \
  --backend jax --model runs/navigation-rehearsal \
  --request examples/doom_request.json
```

These commands execute the model. They never read the saved response files.
[Example provenance](../examples/recorded_examples.json) records the source
trajectory hashes. The response files wrap exact recorded probabilities in the
current API structure; their values can vary slightly across numerical backends.

## Original local-safety interface

[maze_request.json](../examples/maze_request.json) asks four Boolean questions
about neighboring cells. Its [MLX FP32 response](../examples/maze_response.json)
comes from `maze-expanded`, on an existing validation observation:

| Question | P(true) |
| --- | ---: |
| North is open | 0.7735 |
| East is open | 0.4011 |
| South is open | 0.7785 |
| West is open | 0.3991 |

After [setting up MLX and exporting that checkpoint](mlx.md):

```bash
.venv-mlx/bin/python examples/predict.py \
  --backend mlx --model artifacts/maze-expanded-mlx \
  --request examples/maze_request.json
```

This example uses four independent Boolean distributions, not one distribution
over directions. The controller combines them with its exploration memory.

## Scope

The backbone and scoring head are trained together with supervised candidate
cross entropy. This project does not implement Jev's RLCD, calibrated confidence,
or a switch between reasoning and non-reasoning modes. An encoder backbone could
also implement this scoring interface, but would need its own training and evaluation.
