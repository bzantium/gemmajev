"""Direction-balanced training batches with optional difficult-example replay."""

import numpy as np


def maze_schedule(rows, steps, rng, margins):
    """Use each direction once per batch, balancing labels and replaying errors.

    Margins must come from the parent model on these training rows only. Half
    of each direction/label stream cycles through its lowest-margin quartile;
    the other half cycles through the full stratum. Row indices are preserved.
    """
    margins = np.asarray(margins)
    if margins.shape != (len(rows),) or not np.isfinite(margins).all():
        raise ValueError("Expected a finite parent margin per training question")
    schedule = np.empty((steps, 4), dtype=np.int64)
    report = {}
    for column, direction in enumerate(("north", "east", "south", "west")):
        for target in (0, 1):
            pool = np.array([i for i, row in enumerate(rows)
                             if row['task'] == 'maze'
                             and row['question_id'] == 'clear_' + direction
                             and row['target'] == target])
            if not len(pool):
                raise ValueError(f"Empty training stratum: {direction}/{target}")
            difficult = pool[np.argsort(margins[pool], kind='stable')[:max(1, len(pool) // 4)]]
            positions = np.flatnonzero((np.arange(steps) + column) % 2 == target)
            for parity, choices in enumerate((pool, difficult)):
                selected = positions[parity::2]
                order = np.concatenate([rng.permutation(choices)
                                        for _ in range(1 + len(selected) // len(choices))])
                schedule[selected, column] = order[:len(selected)]
            report[f'{direction}:{target}'] = dict(pool=len(pool), difficult=len(difficult),
                parent_accuracy=float((margins[pool] > 0).mean()), presentations=len(positions))
        # Balance over the schedule without making labels or difficulty follow
        # a fixed four-update cycle shared by every direction.
        schedule[:, column] = schedule[rng.permutation(steps), column]
    return schedule, report
