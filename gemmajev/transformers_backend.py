"""Candidate scoring with the exported Hugging Face sequence-scoring model."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gemmajev.backbones import ChatTokenizer
from gemmajev.batching import encode_compact_games
from gemmajev.interface import decision_row, format_answer


class TransformersGameEngine:
    def __init__(self, folder, *, device="cpu"):
        self.folder = Path(folder).resolve()
        self.config = json.loads((self.folder / "training_config.json").read_text())
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.folder,
                dtype=torch.float32,
                attn_implementation="eager",
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        if self.model.config.num_labels != 1:
            raise ValueError("Expected one scalar score per candidate")
        self.tokenizer = ChatTokenizer(
            AutoTokenizer.from_pretrained(self.folder, local_files_only=True)
        )

    @torch.inference_mode()
    def score(self, encoded):
        b, c, length = encoded["tokens"].shape
        tokens = torch.as_tensor(
            encoded["tokens"].reshape(b * c, length),
            dtype=torch.long,
            device=self.model.device,
        )
        lengths = torch.as_tensor(encoded["lengths"].reshape(-1), device=tokens.device)
        mask = torch.arange(length, device=tokens.device)[None, :] < lengths[:, None]
        scores = self.model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        return scores.reshape(b, c).float().cpu().numpy()

    def predict(self, request, batch_questions=4, temperature=1.0):
        """Return the same game response as JAX and MLX, without text generation."""
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
        forwards = 0
        for start in range(0, len(pending), batch_questions):
            batch = pending[start : start + batch_questions]
            encoded = encode_compact_games(
                [item[-1] for item in batch], self.tokenizer, self.config["max_length"]
            )
            scores = self.score(encoded).astype(np.float64) / temperature
            scores = np.where(encoded["candidate_mask"], scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            for (index, qid, kind, keys, _), values in zip(batch, probs, strict=True):
                returned[index]["answers"][qid] = format_answer(kind, keys, values[: len(keys)])
            forwards += 1
        return dict(states=returned, execution=dict(forward_passes=forwards, network_model_calls=0))
