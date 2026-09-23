"""Train the shared Basic/Maze candidate model with Tunix, then verify restore."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tunix.sft.checkpoint_manager import CheckpointManager
from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig

from jev_tunix.decision import loss_with_aux
from jev_tunix.experiment import evaluate_arrays, load_model, logits, project_path
from jev_tunix.game_contract import encode_games
from jev_tunix.game_observation import format_rows
from jev_tunix.game_schedule import maze_schedule

ROOT = Path(__file__).resolve().parents[2]


class GameTrainer(PeftTrainer):
    def _post_process_train_step(self, aux):
        loss = float(aux)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        row = dict(step=self.train_steps + 1, loss=loss)
        with (self.output / "loss.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["step"] % 100 == 0:
            print(json.dumps(row), flush=True)
        if getattr(self, "monitor", None) and row["step"] % self.monitor_every == 0:
            self.monitor(row["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-run", help="Override the source checkpoint in the configuration")
    args = parser.parse_args()
    cfg_path, output = project_path(args.config), project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(cfg_path.read_text())
    if args.initial_run:
        cfg["initial_run"] = args.initial_run
    folder = ROOT / "data" / cfg["dataset"]
    manifest = json.loads((folder / "manifest.json").read_text())
    sources = [
        Path(__file__),
        cfg_path,
        folder / "manifest.json",
        folder / "token-capacity.json",
        ROOT / "jev_tunix/game_contract.py",
        ROOT / "jev_tunix/game_schedule.py",
        ROOT / "jev_tunix/game_observation.py",
        ROOT / "jev_tunix/decision.py",
        ROOT / "jev_tunix/tokenization.py",
        ROOT / "jev_tunix/experiment.py",
        ROOT / "jev_tunix/model_registry.py",
        ROOT / "requirements-gpu.lock.txt",
        ROOT / "artifacts" / cfg["model_name"] / "manifest.json",
    ]
    if cfg.get("maze_replay_cache"):
        cache_run = project_path(cfg["maze_replay_cache"])
        sources.extend([cache_run / "metadata.json", cache_run / "parent-training-margins.npy"])
    for name, expected in manifest["inputs_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    rows = {}
    for split in ("train", "validation"):
        path = folder / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["splits"][split]["sha256"]
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        sources.append(path)
    assert jax.default_backend() == "gpu" and jax.device_count() == 1
    jax.config.update("jax_default_matmul_precision", "highest")
    metadata = dict(config=cfg, device=str(jax.devices()[0]), source_sha256={})
    for path in sources:
        name = str(path.relative_to(ROOT))
        content = path.read_bytes()
        metadata["source_sha256"][name] = hashlib.sha256(content).hexdigest()
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Loading Gemma for Basic and Maze", flush=True)
    model, tokenizer, mesh = load_model(cfg["seed"], cfg["model_name"], "chat")
    if cfg.get("initial_run"):
        parent = project_path(cfg["initial_run"])
        parent_result = json.loads((parent / "result.json").read_text())
        parent_cfg = json.loads((parent / "metadata.json").read_text())["config"]
        assert parent_result["checkpoint_restore_exact"]
        assert parent_cfg["model_name"] == cfg["model_name"]
        metadata["initial_checkpoint_sha256"] = {
            str(p.relative_to(parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((parent / "checkpoint").rglob("*")) if p.is_file()
        }
        with mesh:
            manager = CheckpointManager(str(parent / "checkpoint"))
            try:
                step, _ = manager.maybe_restore(model, step=parent_cfg["steps"])
                assert step == parent_cfg["steps"]
            finally:
                manager.close()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    data = {name: encode_games(format_rows(items, cfg.get("observation_layout", "original")),
                               tokenizer, cfg["max_length"]) for name, items in rows.items()}
    rng = np.random.default_rng(cfg["seed"])
    task_batches = cfg.get("task_batches", dict(maze=cfg["batch_size"] // 2,
                                               basic=cfg["batch_size"] // 2))
    assert sum(task_batches.values()) == cfg["batch_size"]
    priority_schedule = None
    if cfg.get("maze_error_replay"):
        assert task_batches["maze"] == 4
        if cfg.get("maze_replay_cache"):
            cached = json.loads((cache_run / "metadata.json").read_text())
            assert cached["config"].get("observation_layout", "original") == cfg.get(
                "observation_layout", "original")
            assert cached["initial_checkpoint_sha256"] == metadata["initial_checkpoint_sha256"]
            for source in (str((folder / "train.jsonl").relative_to(ROOT)),
                           "jev_tunix/game_contract.py", "jev_tunix/decision.py",
                           "jev_tunix/experiment.py", "jev_tunix/model_registry.py"):
                assert cached["source_sha256"][source] == metadata["source_sha256"][source], source
            margins = np.load(cache_run / "parent-training-margins.npy")
        else:
            maze_ids = np.array([i for i, row in enumerate(rows["train"]) if row["task"] == "maze"])
            with mesh:
                _, parent_scores = evaluate_arrays(
                    model, {k: v[maze_ids] for k, v in data["train"].items()}, batch_size=32
                )
            targets = np.array([rows["train"][i]["target"] for i in maze_ids])
            margins = np.zeros(len(rows["train"]), dtype=np.float64)
            margins[maze_ids] = (parent_scores[np.arange(len(maze_ids)), targets]
                                - parent_scores[np.arange(len(maze_ids)), 1 - targets])
        np.save(output / "parent-training-margins.npy", margins)
        priority_schedule, report = maze_schedule(rows["train"], cfg["steps"], rng, margins)
        (output / "maze-sampling.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"maze_error_replay": report}), flush=True)
    indices = []
    probes = {}
    for task in ("maze", "basic"):
        pool = np.array([i for i, row in enumerate(rows["train"]) if row["task"] == task])
        if cfg.get("train_limit_per_task"):
            pool = pool[:cfg["train_limit_per_task"]]
        probes[task] = pool[:64]
        per_task = task_batches[task]
        if not per_task:
            continue
        if task == "maze" and priority_schedule is not None:
            indices.append(priority_schedule)
            continue
        needed = cfg["steps"] * per_task
        order = np.concatenate(
            [rng.permutation(pool) for _ in range((needed + len(pool) - 1) // len(pool))]
        )[:needed]
        indices.append(order.reshape(cfg["steps"], per_task))
    schedule = np.concatenate(indices, axis=1)
    np.save(output / "schedule.npy", schedule)

    def monitor(step):
        report = {"step": step}
        for task, ids in probes.items():
            metrics, _ = evaluate_arrays(model, {k: v[ids] for k, v in data["train"].items()})
            report[task] = metrics
        with (output / "train-probe.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps({"train_probe": step, **{k: report[k]["accuracy"] for k in probes}}),
              flush=True)

    def evaluate(label):
        result = {}
        for task in ("maze", "basic"):
            mask = np.array([r["task"] == task for r in rows["validation"]])
            metrics, scores = evaluate_arrays(
                model, {k: v[mask] for k, v in data["validation"].items()}, batch_size=8
            )
            result[task] = metrics
            np.save(output / f"{label}-{task}.npy", scores)
        print(json.dumps({label: {k: v["accuracy"] for k, v in result.items()}}), flush=True)
        return result

    with mesh:
        before = evaluate("before")
        monitor(0)
        config = TrainingConfig(
            max_steps=cfg["steps"],
            eval_every_n_steps=cfg["steps"] + 1,
            checkpoint_root_directory=str(output / "checkpoint"),
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=cfg["steps"], max_to_keep=1, enable_async_checkpointing=False
            ),
        )
        trainer = GameTrainer(
            model,
            optax.chain(
                optax.clip_by_global_norm(1.0), optax.adamw(cfg["learning_rate"], weight_decay=0.01)
            ),
            config,
        ).with_loss_fn(loss_with_aux, has_aux=True)
        trainer.is_managed_externally = True
        trainer.output = output
        trainer.monitor = monitor
        trainer.monitor_every = cfg.get("monitor_every", 200)
        started = time.perf_counter()
        trainer.train(({k: v[batch] for k, v in data["train"].items()} for batch in schedule))
        trainer.close()
        elapsed = time.perf_counter() - started
        assert trainer.train_steps == cfg["steps"]
        check = {k: v[:8] for k, v in data["validation"].items()}
        saved = np.asarray(logits(model, check))
        nnx.update(model, jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param)))
        manager = CheckpointManager(str(output / "checkpoint"))
        step, _ = manager.maybe_restore(model, step=cfg["steps"])
        manager.close()
        assert step == cfg["steps"]
        np.testing.assert_array_equal(saved, np.asarray(logits(model, check)))
        after = evaluate("after")
        result = dict(
            status="completed",
            updates=cfg["steps"],
            before=before,
            after=after,
            training_seconds=elapsed,
            checkpoint_restore_exact=True,
            gameplay_verified=False,
        )
        for name, expected in metadata["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
