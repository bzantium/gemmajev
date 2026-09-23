"""Protect Boolean polarity and mask padding at the external game boundary."""

import unittest

import numpy as np

from gemmajev.interface import decision_row, encode_games, format_answer


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class GameContractTest(unittest.TestCase):
    def test_boolean_polarity_survives_reordered_criteria(self):
        row, keys = decision_row(
            "map",
            dict(
                type="boolean", instructions="Open?", criteria={"true": "Open", "false": "Blocked"}
            ),
        )
        self.assertEqual(keys, ["false", "true"])
        batch = encode_games([row], Tokenizer(), 256)
        np.testing.assert_array_equal(batch["candidate_mask"], [[True, True, False, False]])
        np.testing.assert_array_equal(batch["tokens"][0, 2], batch["tokens"][0, 0])
        self.assertEqual(format_answer("boolean", keys, [0.2, 0.8])["p_true"], 0.8)

    def test_choice_labels_keep_probability_identity(self):
        _, keys = decision_row(
            "state",
            dict(type="choice", instructions="Act", criteria={"shoot": "Fire", "left": "Move"}),
        )
        answer = format_answer("choice", keys, [0.75, 0.25])
        self.assertEqual(answer["probabilities"], {"left": 0.75, "shoot": 0.25})

    def test_invalid_probabilities_are_rejected(self):
        for values in ([0.2, 0.2], [float("nan"), 0.5], [-0.1, 1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                format_answer("boolean", ["false", "true"], values)


if __name__ == "__main__":
    unittest.main()
