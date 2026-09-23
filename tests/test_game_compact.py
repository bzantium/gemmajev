"""Protect complete candidate inputs when removing inference padding."""

import unittest

import numpy as np

from gemmajev.batching import encode_compact_games
from gemmajev.interface import encode_games
from gemmajev.model import TokenCapacityError


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class CompactGameTest(unittest.TestCase):
    def test_real_inputs_lengths_and_candidate_order_are_preserved(self):
        for counts in [(2, 2), (2, 4)]:
            rows = [
                dict(
                    state="grid",
                    question="open?",
                    candidates=["x" * n for n in range(1, count + 1)],
                )
                for count in counts
            ]
            full = encode_games(rows, Tokenizer(), 512)
            compact = encode_compact_games(rows, Tokenizer(), 512)
            self.assertEqual(compact["tokens"].shape[1], max(counts))
            self.assertLess(compact["tokens"].shape[2], 512)
            for row, count in enumerate(counts):
                for candidate in range(count):
                    length = full["lengths"][row, candidate]
                    self.assertEqual(compact["lengths"][row, candidate], length)
                    np.testing.assert_array_equal(
                        full["tokens"][row, candidate, :length],
                        compact["tokens"][row, candidate, :length],
                    )
            np.testing.assert_array_equal(
                full["candidate_mask"][:, : max(counts)], compact["candidate_mask"]
            )

    def test_over_capacity_input_is_rejected_not_truncated(self):
        rows = [dict(state="x" * 512, question="open?", candidates=["yes", "no"])]
        with self.assertRaises(TokenCapacityError):
            encode_compact_games(rows, Tokenizer(), 512)


if __name__ == "__main__":
    unittest.main()
