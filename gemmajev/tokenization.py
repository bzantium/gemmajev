"""Backend-independent candidate tokenization with strict capacity checks."""

import json

import numpy as np


class TokenCapacityError(ValueError):
    """A complete observed input exceeds the requested padding capacity."""


def make_batch(examples, tokenizer, max_length=128):
    """Right-pad complete questions; reject truncation and invalid targets."""
    if not examples:
        raise ValueError("At least one question is required")
    candidates = max(len(row["candidates"]) for row in examples)
    tokens = np.full((len(examples), candidates, max_length), tokenizer.pad_token_id, np.int32)
    lengths = np.zeros((len(examples), candidates), np.int32)
    mask = np.zeros((len(examples), candidates), bool)
    targets = []
    for b, row in enumerate(examples):
        choices = row["candidates"]
        target = row["target"]
        if not choices or not isinstance(target, int) or not 0 <= target < len(choices):
            raise ValueError("Each question needs candidates and a valid target index")
        if len(set(choices)) != len(choices) or any(not choice.strip() for choice in choices):
            raise ValueError("Candidates must be nonempty, distinct strings")
        targets.append(target)
        for c, choice in enumerate(choices):
            text = (
                f"State: {json.dumps(row['state'], sort_keys=True)}\n"
                f"Question: {row['question']}\nCandidate: {choice}"
            )
            ids = tokenizer.encode(text, add_special_tokens=True)
            if not ids:
                raise ValueError("Tokenizer returned an empty candidate input")
            if len(ids) > max_length:
                raise TokenCapacityError(
                    f"Question {b}, candidate {c}: {len(ids)} tokens exceeds {max_length}"
                )
            tokens[b, c, : len(ids)] = ids
            lengths[b, c] = len(ids)
            mask[b, c] = True
        # Dummy candidates reuse a real input, avoiding fully masked attention.
        tokens[b, len(choices) :] = tokens[b, 0]
        lengths[b, len(choices) :] = lengths[b, 0]
    return dict(
        tokens=tokens, lengths=lengths, candidate_mask=mask, targets=np.asarray(targets, np.int32)
    )
