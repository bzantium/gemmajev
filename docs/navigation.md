# Maze navigation with memory

This recipe adds a movement question to the existing Gemma checkpoint. The model
sees a 5×5 window, the goal offset, previous move and directional visit/attempt
history. It scores north, east, south and west.

The controller derives legal directions from the visible window and excludes
already explored branches. Gemma ranks the available passages at junctions. A
DFS stack handles required backtracking; forced single-direction moves do not
invoke the model. This controller differs from the original local-safety demo,
so compare the parent and trained model using the same new controller.

## Data

The three demonstration maps **are included in training**. Recordings on these
maps are demonstrations of fitted behavior, not evidence of unseen-map performance.
Four separate 50×50 maps are reserved for validation and four more for gameplay
testing. The test maps do not supply training examples. The run lengths are fixed
in advance; test results are reported without selecting an intermediate checkpoint.

Movement training contains 1,322 examples: 441 from the demo maps and 881 from
20 generated maps. Another 2,174 original ViZDoom examples rehearse the existing
task. Training samples three demo movement questions, three generated movement
questions and two ViZDoom questions per batch.

An offline teacher uses the full map to label useful directions. Occasional
off-route moves expose recovery states. **The teacher's map and distance values
are not included in model inputs or available to the inference controller.**
Question accuracy alone does not establish efficient navigation under partial
observability; complete rollouts are evaluated separately.

```bash
source scripts/env.sh
.venv/bin/python examples/prepare_mazes.py
.venv/bin/python scripts/prepare_navigation.py
.venv/bin/python scripts/check_inputs.py --config configs/navigation.json
.venv/bin/python scripts/train.py \
  --config configs/navigation.json --output runs/navigation-v1
```

The continuation uses `runs/maze-expanded` as its parent, 1,200 Tunix SFT updates,
batch size 8 and learning rate 3e-5 on one GPU. Existing checkpoints and the
original safety interface remain available.

An additional 800-update continuation uses `configs/navigation-rehearsal.json`.
It lowers the learning rate to 1e-5 and samples four movement questions and four
ViZDoom questions per batch after the first continuation reduced ViZDoom accuracy.
This reuses the same 2,174 expert-labelled ViZDoom examples from 419 episodes;
it does not add synthetic labels or change the validation data.

```bash
.venv/bin/python scripts/train.py \
  --config configs/navigation-rehearsal.json --output runs/navigation-rehearsal
```

## Compare the same controller before and after training

```bash
for model in maze-expanded navigation-rehearsal; do
  .venv/bin/python examples/navigate.py \
    --model "runs/$model" \
    --cases data/gemmajev-navigation-v1/train-maps.jsonl --scope demo_fit \
    --output "runs/navigation-eval/$model-demo"
  .venv/bin/python examples/navigate.py \
    --model "runs/$model" \
    --cases data/gemmajev-navigation-v1/test-maps.jsonl \
    --output "runs/navigation-eval/$model-test"
done
.venv/bin/python tools/compare_mazes.py \
  --reference runs/navigation-eval/maze-expanded-demo \
  --candidate runs/navigation-eval/navigation-rehearsal-demo \
  --output runs/navigation-eval/comparison-demo.json
```

Reports distinguish model calls, forced moves and cases where the controller
masks an unavailable top-ranked action. Every trajectory is replayed against the
environment. No cached answer is reused solely because coordinates match: the
visit history changes the input.

For Mac inference, export the continuation using `scripts/export_mlx.py`, then
use `examples/navigate.py --backend mlx --model artifacts/navigation-mlx` with the
same case/output arguments. See [MLX setup](mlx.md).

## Results

Both checkpoints were evaluated on the same frozen questions:

| Task | Before movement training | After movement training and rehearsal |
| --- | ---: | ---: |
| Maze next direction, 196 questions | 24.0% | 70.4% |
| ViZDoom expert action, 201 questions | 88.1% | 90.0% |

The Maze task here is next-direction selection, not the original local-safety
Boolean task. Their accuracies should not be compared directly. ViZDoom's README
case succeeded with both checkpoints: 13 decisions, 49 ticks and one shot. Each
recorded episode replayed without an environment mismatch.

With the **same new exploration controller** before and after training:

| Demonstration seed | Parent attempts | Trained attempts |
| --- | ---: | ---: |
| 2026121201 | 756 | 226 |
| 2026121202 | 729 | 131 |
| 2026121203 | 1,856 | 200 |

All three finish, with no collisions. The original controller's recordings used
627 / 401 / 751 attempts. Comparing with those videos mixes a controller change
with training; the table above isolates the checkpoint change. Legal-action
masking prevents wall collisions, and the DFS stack supplies backtracking. Goal
completion and zero collisions alone are therefore not evidence of model skill.

On four separate maps, total attempts fell from 4,866 to 3,218. Two routes got
shorter and two got longer. All four completed under both checkpoints. These
maps were excluded from training but inspected after the first continuation;
the final results are development evaluation, not a fresh blind benchmark.

The three-map recording uses the final GPU trajectories, with accelerated
playback. It is a demonstration of fitted behavior, not unseen-map generalization.
The new checkpoint has not yet been benchmarked through MLX.

[Training result](results/navigation-rehearsal.json) ·
[Matched demo comparison](results/navigation-comparison-demo.json) ·
[Separate-map comparison](results/navigation-comparison-test.json) ·
[ViZDoom gameplay check](results/navigation-doom.json)
