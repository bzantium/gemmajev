"""Text-only Gemma candidate scoring; one categorical loss per question."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from gemmajev.tokenization import TokenCapacityError as TokenCapacityError
from gemmajev.tokenization import make_batch as make_batch


def hidden_states(backbone, tokens, lengths):
    """Tunix 0.1.7 text forward through final norm, without vocabulary decode.

    Uses the same layer order as Gemma3.__call__; public-forward equivalence is
    checked by the smoke runner. This adapter is tied to the pinned Tunix release.
    """
    positions = jnp.broadcast_to(jnp.arange(tokens.shape[-1]), tokens.shape)
    valid_keys = positions < lengths[:, None]
    causal = positions[:, :, None] >= positions[:, None, :]
    attention_mask = causal & valid_keys[:, None, :]
    x = backbone.embedder.encode(tokens)
    for layer in backbone.layers:
        _, x = layer(x, positions, None, attention_mask)
    return backbone.final_norm(x)


class DecisionModel(nnx.Module):
    def __init__(self, backbone, seed=17):
        self.backbone = backbone
        # No shared bias: it cancels under candidate softmax and has no signal.
        self.head = nnx.Linear(
            backbone.config.embed_dim,
            1,
            use_bias=False,
            param_dtype=jnp.float32,
            rngs=nnx.Rngs(seed),
        )

    def __call__(self, tokens, lengths, candidate_mask):
        b, c, length = tokens.shape
        flat_lengths = lengths.reshape(-1)
        hidden = hidden_states(self.backbone, tokens.reshape(b * c, length), flat_lengths)
        pooled = hidden[jnp.arange(b * c), flat_lengths - 1].astype(jnp.float32)
        scores = self.head(pooled).reshape(b, c)
        return jnp.where(candidate_mask, scores, -1e9)


def question_loss(model, tokens, lengths, candidate_mask, targets):
    scores = model(tokens, lengths, candidate_mask)
    return optax.softmax_cross_entropy_with_integer_labels(scores, targets).mean()


def loss_with_aux(model, tokens, lengths, candidate_mask, targets):
    """Explicit parameters are required by Flax NNX keyword resolution."""
    loss = question_loss(model, tokens, lengths, candidate_mask, targets)
    return loss, loss


@nnx.jit
def gradient_report(model, batch):
    loss, grads = nnx.value_and_grad(question_loss)(model, **batch)
    report = {"loss": loss}
    for name in ("backbone", "head"):
        leaves = jax.tree.leaves(grads[name])
        report[name] = {
            "all_finite": jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])),
            "norm": optax.global_norm(jax.tree.map(lambda x: x.astype(jnp.float32), grads[name])),
        }
    return report


@nnx.jit
def predict(model, batch):
    return jax.nn.softmax(
        model(batch["tokens"], batch["lengths"], batch["candidate_mask"]), axis=-1
    )
