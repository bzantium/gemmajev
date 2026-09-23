"""Adapt the two NanoJev game interfaces to a shared Gemma candidate head."""

import json

import numpy as np

from jev_tunix.tokenization import make_batch


def decision_row(state, question):
    """Use only observed state, instructions and supplied candidate descriptions."""
    kind = question["type"]
    criteria = question["criteria"]
    if kind == "boolean":
        if set(criteria) != {"false", "true"}:
            raise ValueError("Boolean criteria must contain false and true")
        keys = ["false", "true"]
    elif kind == "choice":
        keys = sorted(criteria)
    else:
        raise ValueError("These demos use only Choice and Boolean questions")
    if not 2 <= len(keys) <= 4:
        raise ValueError("Expected two to four candidates")
    row = dict(
        state=state,
        question=question["instructions"],
        candidates=[
            json.dumps(dict(value=k, description=criteria[k]), sort_keys=True) for k in keys
        ],
    )
    return row, keys


def encode_games(rows, tokenizer, max_length=768):
    """One stable four-candidate shape, including masked Boolean padding."""
    data = make_batch([dict(r, target=r.get("target", 0)) for r in rows], tokenizer, max_length)
    count = data["tokens"].shape[1]
    if count > 4:
        raise ValueError("Game adapter supports at most four candidates")
    for key in ("tokens", "lengths"):
        data[key] = np.concatenate(
            [data[key], np.repeat(data[key][:, :1], 4 - count, axis=1)], axis=1
        )
    data["candidate_mask"] = np.pad(data["candidate_mask"], ((0, 0), (0, 4 - count)))
    return data


def format_answer(kind, keys, probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(keys),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid candidate probabilities")
    if not np.isclose(values.sum(), 1, atol=1e-6):
        raise ValueError("Candidate probabilities must sum to one")
    answer = dict(type=kind, probabilities=dict(zip(keys, values.tolist(), strict=True)))
    if kind == "boolean":
        answer["p_true"] = answer["probabilities"]["true"]
    return answer
