"""Preserve the trained candidate head through vLLM's /classify API."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from transformers import AutoTokenizer

from gemmajev.backbones import ChatTokenizer
from gemmajev.batching import encode_compact_games
from gemmajev.interface import decision_row, format_answer


class VLLMGameEngine:
    """Client for a pooling server, using the exact training tokenizer and inputs.

    This is not vLLM's /v1/systemone endpoint. The server returns raw candidate
    scores; this client normalizes them across each question's actual options.
    """

    def __init__(self, folder, *, base_url="http://127.0.0.1:8000", model="gemmajev", api_key=None):
        self.folder = Path(folder).resolve()
        self.config = json.loads((self.folder / "training_config.json").read_text())
        self.tokenizer = ChatTokenizer(
            AutoTokenizer.from_pretrained(self.folder, local_files_only=True)
        )
        self.url = base_url.rstrip("/") + "/classify"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def score(self, encoded):
        """Send complete, unpadded token IDs and disable per-candidate activation."""
        positions = list(zip(*np.nonzero(encoded["candidate_mask"]), strict=True))
        inputs = [
            encoded["tokens"][b, c, : encoded["lengths"][b, c]].tolist() for b, c in positions
        ]
        if not inputs:
            raise ValueError("At least one real candidate is required")
        payload = dict(model=self.model, input=inputs, use_activation=False)
        request = Request(self.url, json.dumps(payload).encode(), self.headers)
        with urlopen(request, timeout=120) as response:
            data = json.load(response)["data"]
        if len(data) != len(inputs) or {d["index"] for d in data} != set(range(len(inputs))):
            raise ValueError("vLLM returned missing, duplicate or unexpected candidate indices")
        scores = np.full(encoded["candidate_mask"].shape, -np.inf, dtype=np.float64)
        for item in data:
            values = np.asarray(item["probs"], dtype=np.float64)
            if values.shape != (1,) or not np.isfinite(values).all():
                raise ValueError("Expected one finite, unactivated scalar per candidate")
            scores[positions[item["index"]]] = values[0]
        return scores

    def predict(self, request, batch_questions=4, temperature=1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        if type(batch_questions) is not int or batch_questions < 1:
            raise ValueError("Invalid question batch size")
        states = request["states"]
        if len({s["id"] for s in states}) != len(states):
            raise ValueError("Duplicate state IDs")
        returned = [{"id": s["id"], "answers": {}} for s in states]
        pending = []
        for index, state in enumerate(states):
            for qid, question in state["questions"].items():
                row, keys = decision_row(state["state"], question)
                pending.append((index, qid, question["type"], keys, row))
        calls = 0
        for start in range(0, len(pending), batch_questions):
            batch = pending[start : start + batch_questions]
            encoded = encode_compact_games(
                [item[-1] for item in batch], self.tokenizer, self.config["max_length"]
            )
            scores = self.score(encoded) / temperature
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            for (index, qid, kind, keys, _), values in zip(batch, probs, strict=True):
                returned[index]["answers"][qid] = format_answer(kind, keys, values[: len(keys)])
            calls += 1
        # GPU forward-pass count belongs to the server's scheduler, not this client.
        return dict(states=returned, execution=dict(network_model_calls=calls))
