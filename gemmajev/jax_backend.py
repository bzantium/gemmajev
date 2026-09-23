"""Gemma candidate probabilities behind the NanoJev game request interface."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tunix.sft.checkpoint_manager import CheckpointManager

from gemmajev.batching import encode_compact_games
from gemmajev.interface import decision_row, encode_games, format_answer
from gemmajev.observations import format_rows
from gemmajev.runtime import load_model, logits, project_path


class GameEngine:
    def __init__(self, run, *, compact=False):
        self.compact = compact
        self.run = project_path(run)
        result = json.loads((self.run / "result.json").read_text())
        if result["status"] != "completed" or not result["checkpoint_restore_exact"]:
            raise ValueError("A completed, restore-verified training run is required")
        metadata = json.loads((self.run / "metadata.json").read_text())
        self.config = metadata["config"]
        if self.config.get("observation_layout", "original") != "original":
            source = Path(__file__).with_name("observations.py")
            if (
                hashlib.sha256(source.read_bytes()).hexdigest()
                != metadata["source_sha256"]["gemmajev/observations.py"]
            ):
                raise ValueError("Observation formatter differs from the training source")
        self.model, self.tokenizer, self.mesh = load_model(
            self.config["seed"], self.config["model_name"], "chat"
        )
        with self.mesh:
            manager = CheckpointManager(str(self.run / "checkpoint"))
            try:
                step, _ = manager.maybe_restore(self.model, step=self.config["steps"])
                if step != self.config["steps"]:
                    raise ValueError("Checkpoint step mismatch")
            finally:
                manager.close()

    def predict(self, request, batch_questions=4, temperature=1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        batch_questions = batch_questions or 4
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
        forwards = 0
        with self.mesh:
            for start in range(0, len(pending), batch_questions):
                batch = pending[start : start + batch_questions]
                encoder = encode_compact_games if self.compact else encode_games
                encoded = encoder(
                    format_rows(
                        [item[-1] for item in batch],
                        self.config.get("observation_layout", "original"),
                    ),
                    self.tokenizer,
                    self.config["max_length"],
                )
                scores = np.asarray(logits(self.model, encoded), dtype=np.float64) / temperature
                scores = np.where(encoded["candidate_mask"], scores, -np.inf)
                probs = np.exp(scores - scores.max(-1, keepdims=True))
                probs /= probs.sum(-1, keepdims=True)
                for (index, qid, kind, keys, _), values in zip(batch, probs, strict=True):
                    returned[index]["answers"][qid] = format_answer(kind, keys, values[: len(keys)])
                forwards += 1
        return {
            "states": returned,
            "execution": {"forward_passes": forwards, "network_model_calls": 0},
        }
