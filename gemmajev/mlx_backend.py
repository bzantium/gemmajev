"""Apple GPU inference for an exported Gemma candidate-scoring checkpoint."""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gemma3_text import Gemma3Model, ModelArgs
from transformers import AutoTokenizer

from gemmajev.backbones import ChatTokenizer
from gemmajev.batching import encode_compact_games
from gemmajev.interface import decision_row, format_answer


class MLXGameEngine:
    def __init__(self, folder, *, precision="float32", bits=None):
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        if self.manifest["format"] != "gemmajev-mlx-v1":
            raise ValueError("Expected an exported GemmaJev checkpoint with a scoring head")
        for name, digest in self.manifest["files_sha256"].items():
            with (self.folder / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Export integrity check failed: {name}")
        if precision not in ("float32", "float16") or bits not in (None, 8):
            raise ValueError("Use float32/float16, optionally with 8-bit backbone weights")
        self.precision, self.bits = precision, bits
        self.config = self.manifest["training_config"]
        cfg = json.loads((self.folder / "config.json").read_text())
        self.args = ModelArgs.from_dict(cfg)
        self.model = Gemma3Model(self.args)
        self.model.load_weights(
            list(mx.load(str(self.folder / "backbone.safetensors")).items()), strict=True
        )
        self.model.set_dtype(getattr(mx, precision))
        # This tiny learned head retains FP32 even when the backbone is quantized.
        self.head = mx.load(str(self.folder / "head.safetensors"))["weight"].astype(mx.float32)
        if bits:
            nn.quantize(self.model, group_size=64, bits=bits)
        mx.eval(self.model.parameters(), self.head)
        self.tokenizer = ChatTokenizer(
            AutoTokenizer.from_pretrained(self.folder, local_files_only=True)
        )

    def score(self, encoded):
        """Return raw candidate scores, synchronizing actual GPU computation."""
        b, c, length = encoded["tokens"].shape
        tokens = mx.array(encoded["tokens"].reshape(b * c, length))
        lengths = mx.array(encoded["lengths"].reshape(-1))
        positions = mx.arange(length)
        causal = positions[:, None] >= positions[None, :]
        valid = positions[None, :] < lengths[:, None]
        mask = (causal[None, :, :] & valid[:, None, :])[:, None, :, :]
        local = (
            mask
            & ((positions[:, None] - positions[None, :]) < self.args.sliding_window)[
                None, None, :, :
            ]
        )
        x = self.model.embed_tokens(tokens)
        # Tunix FP32 training uses the true sqrt, not MLX-LM's BF16-rounded scalar.
        x = x * mx.array(math.sqrt(self.args.hidden_size), dtype=x.dtype)
        for i, layer in enumerate(self.model.layers):
            attention_mask = mask if (i + 1) % self.args.sliding_window_pattern == 0 else local
            x = layer(x, attention_mask)
        x = self.model.norm(x)
        pooled = x[mx.arange(b * c), lengths - 1].astype(mx.float32)
        scores = (pooled @ self.head.T).reshape(b, c)
        mx.eval(scores)
        return np.asarray(scores)

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
