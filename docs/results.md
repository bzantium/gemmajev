# Measurements

## Movement model

The bundled three-maze recording uses `navigation-rehearsal`. On the same frozen
validation questions, its parent and final checkpoint score:

| Task | Parent | Movement model |
| --- | ---: | ---: |
| Maze next direction, 196 questions | 24.0% | 70.4% |
| ViZDoom action, 201 questions | 88.1% | 90.0% |

The three demo maps are included in training. Four separate development maps
produce two shorter and two longer routes under the same controller. See the
[data, matched gameplay results and limitations](navigation.md#results).

## Movement model on an Apple GPU

The FP32 MLX export matches all 40 Tunix reference decisions (32 Maze movement,
8 Doom), with maximum probability difference 0.00000681. One movement question
averaged 35.1 ms over ten warm calls on an M2 Max. This covers tokenization,
scoring and answer construction on one observation, not complete gameplay.

[MLX conversion report](results/navigation-mlx-fp32.json) · [Download weights](huggingface.md)

## Local-safety model

| Checkpoint | Maze validation | ViZDoom validation |
| --- | ---: | ---: |
| Initial two-game model | 49.8% on 496 questions | 60.2% on 201 questions |
| Continued model | 80.6% on 1,296 questions | 88.1% on 201 questions |

**The two Maze columns use different validation sets.** These values are not a
matched before/after comparison. Use `scripts/evaluate.py` to evaluate both models
on the same frozen inputs. The original set has correlated questions and some
local patterns shared with training. The expanded set separates maps and local
pattern families, but its 50×50 loop-maze subset contains only one map.

[Initial result](results/baseline.json) · [Continued result](results/maze-expanded.json) · [Data and training lineage](training.md)

## Apple GPU inference

On an Apple M2 Max with 32 GiB RAM, one observation includes tokenization, the four
Maze direction questions, GPU synchronization and answer construction. Timings
below average ten warm calls on one observation.

| Precision | Mean | Changed choices on 40 questions | Maximum probability difference from JAX |
| --- | ---: | ---: | ---: |
| FP32 | 54.4 ms | 0 | 0.0000122 |
| FP16 | 49.0 ms | 0 | 0.0637 |
| 8-bit weights, FP16 activations | 52.3 ms | 0 | 0.0688 |

FP32 is the default because it most closely preserves the original probabilities.
Quantization gave little additional speed in this small measurement; the 8-bit
model was not run through complete games.

## Three fixed mazes

Each is a 50×50 loop maze with a seed fixed before inference. All rows use the
continued checkpoint with the same exploration controller. Attempts include
moves into walls.

| Maze seed | Original JAX GPU attempts | MLX FP32 attempts | MLX result |
| --- | ---: | ---: | --- |
| 2026121201 | 627 | 624 | Goal reached |
| 2026121202 | 401 | 400 | Goal reached |
| 2026121203 | 751 | 751 | Goal reached |

The sequential MLX FP32 runs took **53.2 seconds**, excluding loading. Different
floating-point arithmetic can change trajectories; these are runtime differences,
not training gains. These measurements describe the original local-safety model.
The bundled three-maze video now uses the movement-with-memory continuation,
which finishes in 226 / 131 / 200 attempts. See the [matched comparison](navigation.md#results).

Navigation combines model probabilities with shared exploration code and remembered
paths. Completing these maps does not demonstrate optimal routing, broad OOD
generalization or superiority over other models. The two comparison recordings
in the README use the initial checkpoint, while the three-maze recording uses
the movement-trained checkpoint. Its three demonstration maps are included in
training, and its new controller masks blocked moves and handles backtracking.

[Full timing summaries](results/) · [MLX reproduction](mlx.md)
