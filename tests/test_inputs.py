"""Input failures must be rejected before an expensive GPU step."""

import unittest

import numpy as np

from jev_tunix.decision import make_batch


class TokenizerStub:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [2] + [3 + ord(char) % 32 for char in text]


class InputContractTest(unittest.TestCase):
    def batch(self, candidates, target=0, max_length=256):
        return make_batch(
            [dict(state={"key": True}, question="Choose", candidates=candidates, target=target)],
            TokenizerStub(),
            max_length=max_length,
        )

    def test_variable_candidate_counts(self):
        rows = [
            dict(state={}, question="Choose", candidates=["a", "b"], target=1),
            dict(state={}, question="Choose", candidates=["a", "b", "c"], target=2),
        ]
        batch = make_batch(rows, TokenizerStub(), max_length=64)
        np.testing.assert_array_equal(
            batch["candidate_mask"], [[True, True, False], [True, True, True]]
        )
        np.testing.assert_array_equal(batch["targets"], [1, 2])
        self.assertTrue((batch["lengths"] > 0).all())

    def test_rejects_empty_questions(self):
        with self.assertRaises(ValueError):
            make_batch([], TokenizerStub())

    def test_rejects_empty_candidates(self):
        with self.assertRaises(ValueError):
            self.batch([])

    def test_rejects_invalid_targets(self):
        for target in [-1, 2, 0.5]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.batch(["a", "b"], target)

    def test_rejects_ambiguous_or_blank_candidates(self):
        for candidates in [["a", "a"], ["a", " "]]:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.batch(candidates)

    def test_rejects_silent_truncation(self):
        with self.assertRaises(ValueError):
            self.batch(["very long candidate"], max_length=4)


if __name__ == "__main__":
    unittest.main()
