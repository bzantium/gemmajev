"""Shared experiment I/O, model loading and probability metrics."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoTokenizer
from tunix.models.gemma3 import model as gemma
from tunix.models.gemma3 import params_safetensors

from gemmajev.backbones import MODELS, ChatTokenizer
from gemmajev.model import DecisionModel

ROOT = Path(os.environ.get("JEV_PROJECT_ROOT", Path.cwd())).resolve()


def project_path(path):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Artifacts must remain inside the project")
    return path


def load_model(seed=17, model_name="gemma-3-270m", input_format="plain", *, for_restore=False):
    """Load pretrained weights, or initialize a tree for immediate checkpoint restoration.

    Continuation and inference restore every trained parameter from Orbax, so
    they need the verified tokenizer/configuration but not a second base-weight copy.
    A for_restore model must be restored before it is used for scoring or training.
    """
    spec = MODELS[model_name]
    folder = ROOT / "artifacts" / model_name
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["model_id"] == spec["model_id"] and manifest["revision"] == spec["revision"]
    for name, expected in manifest["sha256"].items():
        if for_restore and (
            name.endswith(".safetensors") or name.endswith(".safetensors.index.json")
        ):
            continue
        with (folder / name).open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, name
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with mesh:
        cfg = dataclasses.replace(
            getattr(gemma.ModelConfig, spec["config"])(), param_dtype=jnp.float32
        )
        if for_restore:
            backbone = gemma.Gemma3(cfg, rngs=nnx.Rngs(seed))
        else:
            backbone = params_safetensors.create_model_from_safe_tensors(
                str(folder), cfg, mesh, dtype=jnp.float32
            )
        model = DecisionModel(backbone, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if input_format == "chat":
        tokenizer = ChatTokenizer(tokenizer)
    elif input_format != "plain":
        raise ValueError("Input format must be plain or chat")
    return model, tokenizer, mesh


@nnx.jit
def logits(model, batch):
    return model(batch["tokens"], batch["lengths"], batch["candidate_mask"])


def metrics(scores, targets, mask=None, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scores = np.asarray(scores, np.float64) / temperature
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    shifted = scores - scores.max(-1, keepdims=True)
    logprobs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    probs = np.exp(logprobs)
    targets = np.asarray(targets)
    correct = probs.argmax(-1) == targets
    onehot = np.eye(probs.shape[-1])[targets]
    confidence = probs.max(-1)
    bins = []
    bin_index = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        lo = index / 10
        membership = bin_index == index
        count = int(membership.sum())
        bins.append(
            dict(
                lower=float(lo),
                upper=float(lo + 0.1),
                count=count,
                accuracy=float(correct[membership].mean()) if count else None,
                confidence=float(confidence[membership].mean()) if count else None,
            )
        )
    ece = sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]) / len(
        scores
    )
    return dict(
        count=len(scores),
        accuracy=float(correct.mean()),
        ce=float(-logprobs[np.arange(len(scores)), targets].mean()),
        brier=float(np.square(probs - onehot).sum(-1).mean()),
        ece=float(ece),
        reliability=bins,
    )


def evaluate_arrays(model, data, batch_size=8):
    values = []
    for start in range(0, len(data["targets"]), batch_size):
        batch = {key: value[start : start + batch_size] for key, value in data.items()}
        values.append(np.asarray(logits(model, batch)))
    scores = np.concatenate(values)
    return metrics(scores, data["targets"]), scores
