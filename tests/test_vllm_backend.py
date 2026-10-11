"""Guard candidate identity and raw-logit semantics at the HTTP boundary."""

import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from gemmajev.vllm_backend import VLLMGameEngine


class VLLMTransportTest(unittest.TestCase):
    def setUp(self):
        self.engine = VLLMGameEngine.__new__(VLLMGameEngine)
        self.engine.url = "http://127.0.0.1:8000/classify"
        self.engine.model = "gemmajev"
        self.engine.headers = {"Content-Type": "application/json"}
        self.encoded = dict(
            tokens=np.array([[[2, 10, 0], [2, 20, 21]], [[2, 30, 0], [2, 30, 0]]]),
            lengths=np.array([[2, 3], [2, 2]]),
            candidate_mask=np.array([[True, True], [True, False]]),
        )

    def test_raw_scores_keep_candidate_identity_and_skip_padding(self):
        data = [
            {"index": 2, "probs": [-3.0]},
            {"index": 0, "probs": [4.0]},
            {"index": 1, "probs": [1.0]},
        ]
        with patch(
            "gemmajev.vllm_backend.urlopen",
            return_value=io.StringIO(json.dumps({"data": data})),
        ) as send:
            scores = self.engine.score(self.encoded)
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["input"], [[2, 10], [2, 20, 21], [2, 30]])
        self.assertIs(payload["use_activation"], False)
        np.testing.assert_array_equal(scores, [[4.0, 1.0], [-3.0, -np.inf]])

    def test_rejects_ambiguous_indices_or_wrong_head(self):
        for data in [
            [{"index": 0, "probs": [1.0]}] * 3,
            [{"index": i, "probs": [0.5, 0.5]} for i in range(3)],
            [{"index": i, "probs": [float("nan")]} for i in range(3)],
        ]:
            with self.subTest(data=data):
                with patch(
                    "gemmajev.vllm_backend.urlopen",
                    return_value=io.StringIO(json.dumps({"data": data})),
                ):
                    with self.assertRaises(ValueError):
                        self.engine.score(self.encoded)


if __name__ == "__main__":
    unittest.main()
