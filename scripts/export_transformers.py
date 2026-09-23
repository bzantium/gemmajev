"""Convert a verified FP32 export to a standard Transformers scalar scorer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors.numpy import load_file, save_file


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Verified export from export_mlx.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, output = [(root / p).resolve() for p in (args.source, args.output)]
    if not all(p.is_relative_to(root) for p in (source, output)):
        raise ValueError("Keep all artifacts under the project")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["format"] != "gemmajev-mlx-v1" or manifest["export_dtype"] != "float32":
        raise ValueError("Expected the complete FP32 backbone and scoring-head export")
    for name, expected in manifest["files_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Source export checksum mismatch: {name}")
    weights = {f"model.{k}": v for k, v in load_file(source / "backbone.safetensors").items()}
    weights["score.weight"] = load_file(source / "head.safetensors")["weight"]
    count = sum(value.size for value in weights.values())
    if count != manifest["exported_parameters"]:
        raise ValueError("Export parameter count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    save_file(weights, output / "model.safetensors", metadata={"format": "pt"})
    for name in manifest["files_sha256"]:
        if name.startswith("tokenizer") or name in (
            "chat_template.jinja",
            "special_tokens_map.json",
            "fixtures.json",
        ):
            shutil.copy2(source / name, output / name)
    config = json.loads((source / "config.json").read_text())
    config.update(
        architectures=["Gemma3TextForSequenceClassification"],
        num_labels=1,
        id2label={"0": "candidate_score"},
        label2id={"candidate_score": 0},
        problem_type="regression",
        torch_dtype="float32",
        use_cache=False,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "training_config.json").write_text(
        json.dumps(manifest["training_config"], indent=2) + "\n"
    )
    result = dict(
        format="gemmajev-transformers-v1",
        source_manifest_sha256=digest(source / "manifest.json"),
        source_run=manifest["source_run"],
        exported_parameters=count,
        export_dtype="float32",
        checkpoint_sha256=manifest["checkpoint_sha256"],
        files_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output.relative_to(root)), exported_parameters=count)))


if __name__ == "__main__":
    main()
