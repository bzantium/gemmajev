"""Verify spatial labels independently of the augmentation implementation."""

import json
import unittest

from scripts.prepare_maze_data import (
    contrast,
    family,
    labeled_rows,
    rotate_observation,
    window,
)


class MazeAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.text = "Agent coordinate: (3,4); zero-based row and column.\nLocal map:\n" + "\n".join(
            [".#.#.", "##.##", ".#A..", "#####", "...#."]
        )

    def labels(self, text):
        result = {}
        for row in labeled_rows(text, {}):
            result[row["question_id"]] = json.loads(row["candidates"][row["target"]])["value"]
        return result

    def test_known_neighbors(self):
        self.assertEqual(
            self.labels(self.text),
            {
                "clear_north": "true",
                "clear_east": "true",
                "clear_south": "false",
                "clear_west": "false",
            },
        )

    def test_rotation_moves_labels_and_coordinates(self):
        rotated = rotate_observation(self.text, 12)
        self.assertIn("(4,8)", rotated)
        self.assertEqual(
            self.labels(rotated),
            {
                "clear_north": "false",
                "clear_east": "true",
                "clear_south": "true",
                "clear_west": "false",
            },
        )
        self.assertEqual(family(self.text), family(rotated))
        for _ in range(3):
            rotated = rotate_observation(rotated, 12)
        self.assertEqual(rotated, self.text)

    def test_contrast_changes_only_requested_label_and_cell(self):
        for direction in ("north", "east", "south", "west"):
            changed = contrast(self.text, direction)
            before, after = self.labels(self.text), self.labels(changed)
            differences = [k for k in before if before[k] != after[k]]
            self.assertEqual(differences, ["clear_" + direction])
            self.assertEqual(
                sum(
                    a != b
                    for a, b in zip(
                        "".join(window(self.text)), "".join(window(changed)), strict=True
                    )
                ),
                1,
            )

    def test_outside_cell_cannot_be_toggled_open(self):
        boundary = self.text.replace("##.##", "XXXXX")
        self.assertIsNone(contrast(boundary, "north"))
        self.assertEqual(self.labels(boundary)["clear_north"], "false")


if __name__ == "__main__":
    unittest.main()
