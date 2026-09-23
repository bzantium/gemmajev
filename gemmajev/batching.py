"""Reduce inference padding without changing any observed input or candidate."""

from gemmajev.tokenization import make_batch


def encode_compact_games(rows, tokenizer, max_length, bucket_size=32):
    """Keep actual candidates and round the longest input up to a small bucket.

    The training capacity is still enforced before slicing. Causal attention and
    last-valid-token pooling do not depend on discarded right-padding positions.
    Buckets avoid compiling a different JAX executable for every input length.
    """
    if type(bucket_size) is not int or bucket_size < 1:
        raise ValueError("Bucket size must be a positive integer")
    data = make_batch([dict(r, target=r.get("target", 0)) for r in rows], tokenizer, max_length)
    if data["tokens"].shape[1] > 4:
        raise ValueError("Game adapter supports at most four candidates")
    longest = int(data["lengths"].max())
    capacity = min(max_length, ((longest + bucket_size - 1) // bucket_size) * bucket_size)
    data["tokens"] = data["tokens"][:, :, :capacity]
    return data
