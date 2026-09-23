# An observation in, probabilities out

The controller sends a state string and named questions. Each question supplies
its candidate descriptions. Gemma scores those candidates; it does not generate
a chat answer or a route through the maze.

Here is the local view in [the example request](../examples/maze_request.json):

```text
Agent coordinate: (3,1)

X#...
X#.##
X#A#.
X#.#.
X#...
```

`A` is the agent, `#` a wall, `.` an open cell and `X` outside the maze. The request
asks whether one step north, east, south or west would reach a traversable cell.
For each question the candidates are `false` and `true`, with descriptions of what
each means. Read the JSON file for the exact instructions and coordinate context.

The [recorded MLX FP32 response](../examples/maze_response.json) from the continued
checkpoint gives these probabilities, rounded here:

| Question | P(true) |
| --- | ---: |
| North is open | 0.7735 |
| East is open | 0.4011 |
| South is open | 0.7785 |
| West is open | 0.3991 |

All four questions share one forward pass. The controller uses these judgments
and its exploration memory to decide where to move. This example comes from an
existing validation observation, not a new generalization test.

## Run the request

After [exporting the model and setting up MLX](mlx.md):

```bash
source scripts/env.sh
.venv-mlx/bin/python examples/predict.py \
  --backend mlx --model artifacts/maze-expanded-mlx \
  --request examples/maze_request.json
```

For JAX, use `--backend jax --model runs/maze-expanded` with the training Python
environment. This executes the model; `maze_response.json` is only a recorded
reference and is never read by the inference command.

## Encoding

The official Gemma chat template wraps the observation, question and one complete
candidate. The last valid token's hidden state goes through a shared scalar head.
Softmax normalizes scores over the supplied candidates. Padding is masked, and
inputs exceeding the token limit are rejected rather than truncated.

Maze has two candidates per question. ViZDoom has four action candidates:
`left`, `right`, `shoot` and `noop`. See [interface.py](../gemmajev/interface.py)
and [model.py](../gemmajev/model.py) for the implementation.
