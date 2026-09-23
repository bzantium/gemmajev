import unittest

import numpy as np

from jev_tunix.game_schedule import maze_schedule


class MazeScheduleTest(unittest.TestCase):
    def test_balanced_replay_uses_training_strata_and_low_margin_examples(self):
        rows = [dict(task='maze', question_id='clear_' + direction, target=target)
                for direction in ('north', 'east', 'south', 'west')
                for target in (0, 1) for _ in range(8)]
        rows.append(dict(task='basic', target=0))
        margins = np.tile(np.arange(8), 8).tolist() + [0]
        schedule, report = maze_schedule(rows, 32, np.random.default_rng(17), margins)
        repeated, _ = maze_schedule(rows, 32, np.random.default_rng(17), margins)
        np.testing.assert_array_equal(schedule, repeated)
        for batch in schedule:
            selected = [rows[i] for i in batch]
            self.assertEqual(len({r['question_id'] for r in selected}), 4)
            self.assertTrue(all(r['task'] == 'maze' for r in selected))
        for column in schedule.T:
            self.assertEqual(sum(rows[i]['target'] for i in column), 16)
        self.assertTrue(any(sum(rows[i]['target'] for i in batch) != 2 for batch in schedule))
        self.assertTrue(all(v['presentations'] == 16 for v in report.values()))
        # Half the stream explicitly draws from the two lowest margins;
        # its uniform half also includes examples from this quartile.
        self.assertEqual(int((np.asarray(margins)[schedule] < 2).sum()), 80)

    def test_missing_stratum_and_invalid_margins_fail(self):
        with self.assertRaises(ValueError):
            maze_schedule([], 4, np.random.default_rng(1), [])
        with self.assertRaises(ValueError):
            maze_schedule([dict(task='basic')], 4, np.random.default_rng(1), [np.nan])


if __name__ == '__main__':
    unittest.main()
