"""Export a restored Tunix checkpoint, including its learned candidate head."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    from flax import nnx
    from safetensors.numpy import save_file

    from gemmajev.batching import encode_compact_games
    from gemmajev.jax_backend import GameEngine
    from gemmajev.runtime import logits, project_path

    root = Path(__file__).resolve().parents[1]
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    engine = GameEngine(args.run, compact=True)
    if engine.config.get("observation_layout", "original") != "original":
        raise ValueError("This export currently supports the selected original input layout")
    model = engine.model.backbone
    cfg = model.config
    if cfg.num_heads == cfg.num_kv_heads:
        raise ValueError("This exporter supports separate Q and KV Gemma projections")

    def array(param):
        return np.asarray(param.value, dtype=np.float32)

    weights = {
        "embed_tokens.weight": array(model.embedder.input_embedding),
        "norm.weight": array(model.final_norm.scale),
    }
    for i, layer in enumerate(model.layers):
        prefix = f"layers.{i}."
        q = array(layer.attn.q_einsum.w)
        kv = array(layer.attn.kv_einsum.w)
        weights[prefix + "self_attn.q_proj.weight"] = q.transpose(0, 2, 1).reshape(
            -1, cfg.embed_dim
        )
        for name, value in zip(("k", "v"), kv, strict=True):
            weights[prefix + f"self_attn.{name}_proj.weight"] = value.transpose(0, 2, 1).reshape(
                -1, cfg.embed_dim
            )
        weights[prefix + "self_attn.o_proj.weight"] = (
            array(layer.attn.attn_vec_einsum.w).reshape(-1, cfg.embed_dim).T
        )
        for name in ("gate", "up", "down"):
            weights[prefix + f"mlp.{name}_proj.weight"] = array(
                getattr(layer.mlp, name + "_proj").kernel
            ).T
        for source, target in [
            ("pre_attention_norm", "input_layernorm"),
            ("post_attention_norm", "post_attention_layernorm"),
            ("pre_ffw_norm", "pre_feedforward_layernorm"),
            ("post_ffw_norm", "post_feedforward_layernorm"),
        ]:
            weights[prefix + target + ".weight"] = array(getattr(layer, source).scale)
        weights[prefix + "self_attn.q_norm.weight"] = array(layer.attn._query_norm.scale)
        weights[prefix + "self_attn.k_norm.weight"] = array(layer.attn._key_norm.scale)
    head = {"weight": array(engine.model.head.kernel).T}
    expected = sum(p.size for p in jax.tree.leaves(nnx.state(engine.model, nnx.Param)))
    assert sum(a.size for a in [*weights.values(), *head.values()]) == expected
    save_file(
        {k: np.ascontiguousarray(v) for k, v in weights.items()},
        str(output / "backbone.safetensors"),
    )
    save_file(head, str(output / "head.safetensors"))
    original = root / "artifacts" / engine.config["model_name"]
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        shutil.copy2(original / name, output / name)
    print(json.dumps(dict(exported_parameters=expected, backbone_tensors=len(weights))), flush=True)

    # Frozen, small cross-runtime check. Actual rollouts are evaluated separately.
    source = root / "data/gemmajev-v1/validation.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    selected = [r for r in rows if r["task"] == "maze"][:32] + [
        r for r in rows if r["task"] == "basic"
    ][:8]
    fixtures = []
    for start in range(0, len(selected), 4):
        subset = selected[start : start + 4]
        data = encode_compact_games(subset, engine.tokenizer, engine.config["max_length"])
        with engine.mesh:
            scores = np.asarray(logits(engine.model, data))
        fixtures.append(
            dict(
                rows=subset,
                tokens=data["tokens"].tolist(),
                lengths=data["lengths"].tolist(),
                scores=scores.tolist(),
            )
        )
        print(json.dumps(dict(reference_questions=start + len(subset))), flush=True)
    (output / "fixtures.json").write_text(json.dumps(fixtures) + "\n")

    def sha(p):
        with p.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    manifest = dict(
        format="gemmajev-mlx-v1",
        source_run=args.run,
        training_config=engine.config,
        metadata_sha256=sha(engine.run / "metadata.json"),
        exported_parameters=expected,
        export_dtype="float32",
        checkpoint_sha256={
            str(p.relative_to(engine.run)): sha(p)
            for p in sorted((engine.run / "checkpoint").rglob("*"))
            if p.is_file()
        },
        files_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))},
        embedding_scale="sqrt(hidden_size) rounded to the activation dtype, matching Tunix",
        fixture_source_sha256=sha(source),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
