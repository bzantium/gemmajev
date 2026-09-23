import unittest

from jev_tunix.game_observation import format_rows


class ObservationLayoutTest(unittest.TestCase):
    def test_all_cells_are_preserved_without_answer_dependent_formatting(self):
        grid = ['XXXXX', '##..#', '#.A.#', '.##..', '..#..']
        state = 'Agent coordinate: (1,27).\nLocal map:\n' + '\n'.join(grid)
        rows = [dict(state=state, target=label) for label in (0, 1)]
        before = repr(rows)
        formatted = format_rows(rows, 'indexed_grid')
        self.assertEqual(repr(rows), before)
        self.assertEqual(formatted[0]['state'], formatted[1]['state'])
        table = formatted[0]['state'].split('Local map:\n')[1].splitlines()[1:]
        restored = [''.join(line.split(':', 1)[1].split()) for line in table]
        self.assertEqual(restored, grid)
        self.assertTrue(formatted[0]['state'].startswith('Agent coordinate: (1,27).'))

    def test_original_layout_and_doom_are_unchanged(self):
        rows = [dict(state='Doom observation', target=3)]
        self.assertIs(format_rows(rows), rows)
        self.assertEqual(format_rows(rows, 'indexed_grid'), rows)

    def test_malformed_and_unknown_layout_fail(self):
        with self.assertRaises(ValueError):
            format_rows([dict(state='Local map:\nA')], 'indexed_grid')
        with self.assertRaises(ValueError):
            format_rows([], 'other')


if __name__ == '__main__':
    unittest.main()
