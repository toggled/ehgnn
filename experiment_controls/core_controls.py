#!/usr/bin/env python3
"""Feature-only MLP and Random-Resampled controls."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from node_classification_utils import fixed_split, prepare_dataset
from training_utils import clone_state_dict, load_v2e_dataset, make_args, setup_seed
from models_sparse import HCHA, MLP_model
from walmart_feature_check import load_walmart


DEFAULT_MLP_PROTOCOL = Path("experiment_specs/mlp_baseline.json")
DEFAULT_RANDOM_RESAMPLED_PROTOCOL = Path("experiment_specs/random_resampled.json")


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        value = state[name].detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_hash(split: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in ("train", "valid", "test"):
        digest.update(name.encode("ascii"))
        digest.update(split[name].detach().cpu().long().numpy().tobytes())
    return digest.hexdigest()


def verify_dependencies(protocol_path: Path, protocol: Mapping[str, object]) -> None:
    dependencies = protocol.get("dependencies", {})
    for key, value in dependencies.items():
        if key == "policy" or key.endswith("_sha256"):
            continue
        hash_key = f"{key}_sha256"
        if hash_key not in dependencies:
            continue
        path = Path(str(value))
        expected = str(dependencies[hash_key])
        observed = file_hash(path)
        if observed != expected:
            raise RuntimeError(
                f"Dependency changed for {key}: {observed} != {expected}"
            )
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)


def load_dataset(dataset: str, protocol: Mapping[str, object]):
    if dataset in {"actor", "twitch", "pokec", "yelp"}:
        data, num_features, num_classes = prepare_dataset(dataset)
    elif dataset == "coauthor_dblp":
        data, num_features, num_classes = load_v2e_dataset(dataset)
    elif dataset == "walmart-trips":
        cfg = protocol["data"]
        data, num_features, num_classes = load_walmart(
            str(cfg["walmart_feature_noise"]),
            int(cfg["walmart_feature_seed"]),
            int(cfg["walmart_feature_dimension"]),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return data, int(num_features), int(num_classes)


def dataset_context(dataset: str, protocol: Mapping[str, object]):
    data, num_features, num_classes = load_dataset(dataset, protocol)
    cfg = protocol["data"]
    split = fixed_split(
        data,
        int(cfg["split_seed"]),
        float(cfg["train_prop"]),
        float(cfg["valid_prop"]),
    )
    diagnostics = {
        "feature_hash": tensor_hash(data.x),
        "label_hash": tensor_hash(data.y),
        "edge_index_hash": tensor_hash(data.edge_index),
        "split_hash": split_hash(split),
        "num_nodes": int(data.n_x),
        "num_hyperedges": int(data.num_hyperedges),
        "num_original_incidences": int(data.edge_index.size(1)),
        "num_features": num_features,
        "num_classes": num_classes,
    }
    return data, num_features, num_classes, split, diagnostics


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confusion = np.bincount(
        labels * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    present = support > 0
    return {
        "accuracy": 100.0 * float((labels == predictions).mean()),
        "macro_f1": 100.0 * float(f1[present].mean()),
        "balanced_accuracy": 100.0 * float(recall[present].mean()),
        "per_class_recall": [float(value) for value in recall],
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    split: Mapping[str, torch.Tensor],
    num_classes: int,
    include_test: bool,
) -> Dict[str, object]:
    logp = F.log_softmax(logits, dim=1)
    result: Dict[str, object] = {}
    names = ("train", "valid", "test") if include_test else ("train", "valid")
    for name in names:
        idx = split[name]
        key = "val" if name == "valid" else name
        current_labels = labels[idx]
        loss = F.nll_loss(logp[idx], current_labels)
        predictions = logp[idx].argmax(dim=1)
        result[f"{key}_loss"] = float(loss.item())
        result[f"{key}_acc"] = 100.0 * float(
            (predictions == current_labels).float().mean().item()
        )
        if name == "test":
            detailed = classification_metrics(
                current_labels.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
                num_classes,
            )
            result["test_macro_f1"] = detailed["macro_f1"]
            result["test_balanced_accuracy"] = detailed["balanced_accuracy"]
            result["test_per_class_recall"] = detailed["per_class_recall"]
            result["test_predictions"] = predictions.detach().cpu()
            result["test_labels"] = current_labels.detach().cpu()
    return result


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_ci(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float(array.mean()), float(array.mean())
    radius = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return float(array.mean() - radius), float(array.mean() + radius)


def wilcoxon_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.allclose(array, 0.0):
        return 1.0
    return float(stats.wilcoxon(array, alternative="two-sided").pvalue)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * float(p[index]))
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def selected(values: Optional[Sequence], allowed: Sequence, label: str):
    result = list(allowed if values is None else values)
    if not set(result) <= set(allowed):
        raise ValueError(f"Requested {label} outside frozen protocol: {result}")
    return result


def mlp_configs(protocol: Mapping[str, object]) -> List[Dict[str, object]]:
    grid = protocol["validation_grid"]
    configs: List[Dict[str, object]] = []
    index = 0
    for hidden in grid["hidden"]:
        for lr in grid["learning_rate"]:
            for dropout in grid["dropout"]:
                for wd in grid["weight_decay"]:
                    configs.append({
                        "config_index": index,
                        "hidden": int(hidden),
                        "learning_rate": float(lr),
                        "dropout": float(dropout),
                        "weight_decay": float(wd),
                    })
                    index += 1
    return configs


def make_mlp(
    protocol: Mapping[str, object],
    config: Mapping[str, object],
    num_features: int,
    num_classes: int,
) -> MLP_model:
    model_cfg = protocol["model"]
    args = SimpleNamespace(
        num_features=int(num_features),
        num_classes=int(num_classes),
        MLP_hidden=int(config["hidden"]),
        All_num_layers=int(model_cfg["layers"]),
        dropout=float(config["dropout"]),
        normalization=str(model_cfg["normalization"]),
    )
    return MLP_model(args)


def train_mlp(
    *,
    protocol: Mapping[str, object],
    config: Mapping[str, object],
    data,
    split: Mapping[str, torch.Tensor],
    num_features: int,
    num_classes: int,
    seed: int,
    device: torch.device,
    include_test: bool,
) -> Tuple[torch.nn.Module, Dict[str, object], str]:
    setup_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: idx.to(device) for name, idx in split.items()}
    if not include_test:
        local_data.y[local_split["test"]] = -1
    model = make_mlp(protocol, config, num_features, num_classes).to(device)
    model.reset_parameters()
    initial_hash = state_hash(clone_state_dict(model))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    maximum_epochs = int(protocol["model"]["maximum_epochs"])
    patience = int(protocol["model"]["patience"])
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    start = time.perf_counter()
    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(local_data)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            logp[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite MLP training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            val_logits = model(local_data)
            val_loss = F.cross_entropy(
                val_logits[local_split["valid"]], local_data.y[local_split["valid"]]
            ).item()
        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No finite MLP checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(local_data)
        metrics = evaluate_logits(
            logits, local_data.y, local_split, num_classes, include_test
        )
    metrics["best_epoch"] = best_epoch
    metrics["epochs_run"] = epoch + 1
    metrics["end_to_end_seconds"] = elapsed
    return model, metrics, initial_hash


def majority_metrics(data, split: Mapping[str, torch.Tensor], num_classes: int):
    train_labels = data.y[split["train"]].detach().cpu().numpy()
    majority = int(np.bincount(train_labels, minlength=num_classes).argmax())
    result = {"majority_class": majority}
    for name in ("valid", "test"):
        labels = data.y[split[name]].detach().cpu().numpy()
        predictions = np.full(labels.shape, majority, dtype=np.int64)
        detail = classification_metrics(labels, predictions, num_classes)
        key = "val" if name == "valid" else name
        result[f"{key}_accuracy"] = detail["accuracy"]
        result[f"{key}_macro_f1"] = detail["macro_f1"]
        result[f"{key}_balanced_accuracy"] = detail["balanced_accuracy"]
    return result


def tune_mlp(args, protocol: Mapping[str, object], protocol_path: Path, device):
    outdir = Path(str(protocol["output_root"]))
    run_dir = outdir / "tune_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    configs = mlp_configs(protocol)
    config_indices = set(
        int(value) for value in selected(args.config_indices, [c["config_index"] for c in configs], "configs")
    )
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    seeds = [int(value) for value in selected(args.seeds, protocol["validation_grid"]["seeds"], "seeds")]
    for dataset in datasets:
        data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
        for config in configs:
            if int(config["config_index"]) not in config_indices:
                continue
            for seed in seeds:
                path = run_dir / f"tune_c{config['config_index']}_{dataset}_s{seed}.json"
                if path.exists() and not args.force:
                    print(f"[mlp:tune:skip] {dataset} c={config['config_index']} s={seed}", flush=True)
                    continue
                print(f"[mlp:tune] {dataset} c={config['config_index']} s={seed}", flush=True)
                model, metrics, initial_hash = train_mlp(
                    protocol=protocol,
                    config=config,
                    data=data,
                    split=split,
                    num_features=num_features,
                    num_classes=num_classes,
                    seed=seed,
                    device=device,
                    include_test=False,
                )
                if any("test" in key for key in metrics):
                    raise RuntimeError("MLP tuning exposed a test field")
                write_json(path, {
                    "stage": "mlp_validation_only_tuning",
                    "created_at": timestamp(),
                    "protocol_hash": file_hash(protocol_path),
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "seed": seed,
                    **config,
                    "initial_state_hash": initial_hash,
                    **diagnostics,
                    **metrics,
                })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del data


def select_mlp(protocol: Mapping[str, object], protocol_path: Path):
    outdir = Path(str(protocol["output_root"]))
    rows = [read_json(path) for path in sorted((outdir / "tune_runs").glob("tune_*.json"))]
    configs = mlp_configs(protocol)
    expected = {
        (dataset, int(config["config_index"]), int(seed))
        for dataset in protocol["datasets"]
        for config in configs
        for seed in protocol["validation_grid"]["seeds"]
    }
    observed = {
        (str(row["dataset_id"]), int(row["config_index"]), int(row["seed"]))
        for row in rows
    }
    if expected != observed:
        raise RuntimeError(
            f"Incomplete MLP grid: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    expected_hash = file_hash(protocol_path)
    if any(row["protocol_hash"] != expected_hash for row in rows):
        raise RuntimeError("MLP tuning protocol hash mismatch")
    if any(any("test" in key for key in row) for row in rows):
        raise RuntimeError("MLP tuning output contains a test field")

    summary: List[Dict[str, object]] = []
    winners: Dict[str, object] = {}
    for dataset in protocol["datasets"]:
        dataset_summaries = []
        for config in configs:
            group = [
                row for row in rows
                if row["dataset_id"] == dataset
                and int(row["config_index"]) == int(config["config_index"])
            ]
            mean, std = mean_std([float(row["val_acc"]) for row in group])
            item = {
                "dataset_id": dataset,
                "dataset": protocol["display_names"][dataset],
                **config,
                "n": len(group),
                "mean_val_accuracy": mean,
                "std_val_accuracy": std,
            }
            summary.append(item)
            dataset_summaries.append(item)
        winner = min(
            dataset_summaries,
            key=lambda row: (
                -float(row["mean_val_accuracy"]),
                int(row["hidden"]),
                float(row["learning_rate"]),
                float(row["dropout"]),
                float(row["weight_decay"]),
                int(row["config_index"]),
            ),
        )
        winners[dataset] = {
            key: winner[key]
            for key in (
                "config_index",
                "hidden",
                "learning_rate",
                "dropout",
                "weight_decay",
                "mean_val_accuracy",
                "std_val_accuracy",
            )
        }
    frozen = {
        "status": "selected",
        "selected_at": timestamp(),
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": expected_hash,
        "selection_metric": protocol["validation_grid"]["selection_metric"],
        "selected_by_dataset": winners,
    }
    write_csv(outdir / "validation_grid_runs.csv", rows)
    write_csv(outdir / "validation_grid_summary.csv", summary)
    write_json(outdir / "selected_hyperparameters.json", frozen)
    for dataset, winner in winners.items():
        print(f"[mlp:select] {dataset} -> c={winner['config_index']} val={winner['mean_val_accuracy']:.3f}", flush=True)


def save_predictions(path: Path, labels: torch.Tensor, predictions: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        labels=labels.detach().cpu().long().numpy(),
        predictions=predictions.detach().cpu().long().numpy(),
    )
    temporary.replace(path)


def evaluate_mlp(args, protocol: Mapping[str, object], protocol_path: Path, device):
    outdir = Path(str(protocol["output_root"]))
    selected_path = Path(
        args.selected_hyperparameters
        or protocol["dependencies"]["selected_hyperparameters"]
    )
    selected_hyperparameters = read_json(selected_path)
    missing = set(protocol["datasets"]) - set(selected_hyperparameters["selected_by_dataset"])
    if missing:
        raise RuntimeError(f"Missing selected MLP settings for: {sorted(missing)}")
    run_dir = outdir / "evaluation_runs"
    pred_dir = outdir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    seeds = [int(value) for value in selected(args.seeds, protocol["evaluation"]["seeds"], "seeds")]
    for dataset in datasets:
        data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
        config = selected_hyperparameters["selected_by_dataset"][dataset]
        majority = majority_metrics(data, split, num_classes)
        for seed in seeds:
            path = run_dir / f"eval_mlp_{dataset}_s{seed}.json"
            prediction_path = pred_dir / f"pred_mlp_{dataset}_s{seed}.npz"
            if path.exists() and prediction_path.exists() and not args.force:
                print(f"[mlp:evaluate:skip] {dataset} s={seed}", flush=True)
                continue
            print(f"[mlp:evaluate] {dataset} s={seed}", flush=True)
            model, metrics, initial_hash = train_mlp(
                protocol=protocol,
                config=config,
                data=data,
                split=split,
                num_features=num_features,
                num_classes=num_classes,
                seed=seed,
                device=device,
                include_test=True,
            )
            predictions = metrics.pop("test_predictions")
            labels = metrics.pop("test_labels")
            save_predictions(prediction_path, labels, predictions)
            record = {
                "stage": "mlp_test_evaluation",
                "created_at": timestamp(),
                "protocol_hash": file_hash(protocol_path),
                "selected_hyperparameters_hash": file_hash(selected_path),
                "dataset_id": dataset,
                "dataset": protocol["display_names"][dataset],
                "seed": seed,
                "method": "MLP",
                "initial_state_hash": initial_hash,
                "prediction_path": str(prediction_path.resolve()),
                "prediction_hash": file_hash(prediction_path),
                "majority_class": majority["majority_class"],
                **{
                    f"majority_{key}": value
                    for key, value in majority.items()
                    if key != "majority_class"
                },
                **config,
                **diagnostics,
                **metrics,
            }
            write_json(path, record)
            print(
                f"[mlp:evaluate:done] {dataset} s={seed} test={record['test_acc']:.2f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del data


def current_method_summaries() -> List[Dict[str, str]]:
    rows = read_csv(
        Path(".work/structural_baselines/all_method_budget_summary.csv")
    )
    rows.extend(read_csv(
        Path(".work/large_dataset_accuracy/new_dataset_method_summary.csv")
    ))
    return rows


def current_seed_accuracy() -> Dict[Tuple[str, float, int, str], float]:
    lookup: Dict[Tuple[str, float, int, str], float] = {}
    core_rows = read_csv(
        Path(".work/main_accuracy/frozen_evaluation_runs.csv")
    )
    for row in core_rows:
        if row["dataset_id"] not in {"actor", "twitch", "pokec", "yelp"}:
            continue
        if row["method"] not in {"EHGNN-F", "Random-Fixed"}:
            continue
        lookup[(row["dataset_id"], float(row["keep_ratio"]), int(row["seed"]), row["method"])] = float(row["test_acc"])
    for path in sorted(Path(
        ".work/large_dataset_accuracy/evaluation_runs"
    ).glob("eval_*.json")):
        row = read_json(path)
        if row.get("method") not in {"EHGNN-F", "Random-Fixed"}:
            continue
        lookup[(str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), str(row["method"]))] = float(row["test_acc"])
    return lookup


def summarize_mlp(protocol: Mapping[str, object], protocol_path: Path):
    outdir = Path(str(protocol["output_root"]))
    paths = sorted((outdir / "evaluation_runs").glob("eval_mlp_*.json"))
    rows = [read_json(path) for path in paths]
    expected = {
        (dataset, int(seed))
        for dataset in protocol["datasets"]
        for seed in protocol["evaluation"]["seeds"]
    }
    observed = {(str(row["dataset_id"]), int(row["seed"])) for row in rows}
    if expected != observed:
        raise RuntimeError(
            f"Incomplete MLP evaluation: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    method_summary: List[Dict[str, object]] = []
    for dataset in protocol["datasets"]:
        group = [row for row in rows if row["dataset_id"] == dataset]
        item: Dict[str, object] = {
            "dataset_id": dataset,
            "dataset": protocol["display_names"][dataset],
            "method": "MLP",
            "n": len(group),
            "majority_class": int(group[0]["majority_class"]),
        }
        for metric in (
            "test_acc",
            "test_macro_f1",
            "test_balanced_accuracy",
            "majority_test_accuracy",
            "majority_test_macro_f1",
            "majority_test_balanced_accuracy",
            "end_to_end_seconds",
        ):
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(
                [float(row[metric]) for row in group]
            )
        method_summary.append(item)
    write_csv(outdir / "evaluation_runs.csv", rows)
    write_csv(outdir / "method_summary.csv", method_summary)

    current = current_method_summaries()
    comparison: List[Dict[str, object]] = []
    summary_lookup = {row["dataset_id"]: row for row in method_summary}
    for row in current:
        dataset = row["dataset_id"]
        mlp = summary_lookup[dataset]
        comparison.append({
            "dataset_id": dataset,
            "dataset": row["dataset"],
            "budget": float(row["keep_ratio"]),
            "mlp_accuracy": float(mlp["test_acc_mean"]),
            "majority_accuracy": float(mlp["majority_test_accuracy_mean"]),
            "full_accuracy": float(row["Full_mean"]),
            "ehgnnf_accuracy": float(row["EHGNN-F_mean"]),
            "random_fixed_accuracy": float(row["Random-Fixed_mean"]),
            "ehgnnf_minus_mlp": float(row["EHGNN-F_mean"]) - float(mlp["test_acc_mean"]),
            "random_fixed_minus_mlp": float(row["Random-Fixed_mean"]) - float(mlp["test_acc_mean"]),
            "full_minus_mlp": float(row["Full_mean"]) - float(mlp["test_acc_mean"]),
        })
    write_csv(outdir / "comparison_summary.csv", comparison)

    lines = [
        "# Feature-only MLP result",
        "",
        "MLP configurations were selected separately per dataset using validation accuracy only. Values are ten-seed test mean +/- sample standard deviation.",
        "",
        "| Dataset | Majority accuracy | MLP accuracy | MLP macro-F1 | MLP balanced accuracy | Full accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in method_summary:
        current_rows = [row for row in current if row["dataset_id"] == item["dataset_id"]]
        full = float(current_rows[0]["Full_mean"])
        lines.append(
            f"| {item['dataset']} | {item['majority_test_accuracy_mean']:.2f} "
            f"| {item['test_acc_mean']:.2f} +/- {item['test_acc_std']:.2f} "
            f"| {item['test_macro_f1_mean']:.2f} +/- {item['test_macro_f1_std']:.2f} "
            f"| {item['test_balanced_accuracy_mean']:.2f} +/- {item['test_balanced_accuracy_std']:.2f} "
            f"| {full:.2f} |"
        )
    lines.extend([
        "",
        "## EHGNN-F minus MLP by structural budget",
        "",
        "| Dataset | 10% | 20% | 30% | 50% |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in protocol["datasets"]:
        values = sorted(
            [row for row in comparison if row["dataset_id"] == dataset],
            key=lambda row: float(row["budget"]),
        )
        lines.append(
            f"| {protocol['display_names'][dataset]} | "
            + " | ".join(f"{row['ehgnnf_minus_mlp']:+.2f}" for row in values)
            + " |"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_hash": file_hash(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "run_count": len(rows),
        "outputs": {
            name: file_hash(outdir / name)
            for name in (
                "evaluation_runs.csv",
                "method_summary.csv",
                "comparison_summary.csv",
                "summary.md",
            )
        },
    })


def uniform_exact_mask(total: int, k: int, generator: torch.Generator, device):
    if not 0 < k <= total:
        raise ValueError(f"Invalid exact mask budget {k}/{total}")
    indices = torch.randperm(total, generator=generator, device=device)[:k]
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    mask[indices] = True
    if int(mask.sum()) != k:
        raise RuntimeError("Uniform exact mask has wrong cardinality")
    return mask, indices


def train_random_resampled(
    *,
    protocol: Mapping[str, object],
    data,
    split: Mapping[str, torch.Tensor],
    num_features: int,
    num_classes: int,
    keep_ratio: float,
    seed: int,
    device: torch.device,
):
    setup_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: idx.to(device) for name, idx in split.items()}
    cfg = protocol["training"]
    total = int(local_data.edge_index.size(1))
    k = max(1, int(keep_ratio * total))
    base_edge_index = local_data.edge_index
    args = make_args(
        mode=str(cfg["model_mode"]),
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=keep_ratio,
        hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        sampling="multinomial",
    )
    args.sparse_self_loop_policy = str(cfg["sparse_self_loop_policy"])
    model = HCHA(args).to(device)
    model.reset_parameters()
    initial_hash = state_hash(clone_state_dict(model))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    stream_cfg = protocol["random_streams"]
    train_seed = 410000 + int(seed)
    eval_seed = 310000 + int(seed)
    if str(stream_cfg["training_mask_seed"]) != "410000 + model seed":
        raise RuntimeError("Unexpected frozen training-mask seed rule")
    train_generator = torch.Generator(device=device).manual_seed(train_seed)
    eval_generator = torch.Generator(device=device).manual_seed(eval_seed)
    eval_mask, eval_indices = uniform_exact_mask(total, k, eval_generator, device)
    eval_edge_index = base_edge_index[:, eval_indices]

    maximum_epochs = int(cfg["maximum_epochs"])
    patience = int(cfg["patience"])
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    previous_mask = None
    jaccard_sum = 0.0
    jaccard_count = 0
    start = time.perf_counter()
    for epoch in range(maximum_epochs):
        train_mask, train_indices = uniform_exact_mask(
            total, k, train_generator, device
        )
        if previous_mask is not None:
            intersection = int((previous_mask & train_mask).sum().item())
            union = 2 * k - intersection
            jaccard_sum += intersection / union
            jaccard_count += 1
        previous_mask = train_mask
        local_data.edge_index = base_edge_index[:, train_indices]
        if int(local_data.edge_index.size(1)) != k:
            raise RuntimeError("Training graph violates exact K")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(local_data, is_test=False)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            logp[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Random-Resampled loss at epoch {epoch}")
        loss.backward()
        optimizer.step()

        local_data.edge_index = eval_edge_index
        with torch.no_grad():
            model.eval()
            val_logits = model(local_data, is_test=True)
            val_loss = F.cross_entropy(
                val_logits[local_split["valid"]], local_data.y[local_split["valid"]]
            ).item()
        if int(getattr(model, "last_fixed_self_loop_count", -1)) != int(local_data.n_x):
            raise RuntimeError("Fixed self-loop count check failed")
        if int(getattr(model, "last_forward_incidence_count", -1)) != k + int(local_data.n_x):
            raise RuntimeError("Forward incidence-count audit failed")
        if val_loss < best_loss:
            best_loss = float(val_loss)
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No finite Random-Resampled checkpoint")
    model.load_state_dict(best_state)
    local_data.edge_index = eval_edge_index
    model.eval()
    with torch.no_grad():
        logits = model(local_data, is_test=True)
        metrics = evaluate_logits(
            logits, local_data.y, local_split, num_classes, include_test=True
        )
    metrics["best_epoch"] = best_epoch
    metrics["epochs_run"] = epoch + 1
    metrics["end_to_end_seconds"] = elapsed
    diagnostics = {
        "num_selected": k,
        "num_mask_units": total,
        "structural_density": k / total,
        "num_fixed_self_loops": int(local_data.n_x),
        "forward_incidences": k + int(local_data.n_x),
        "effective_forward_density": (k + int(local_data.n_x)) / (total + int(local_data.n_x)),
        "training_mask_seed": train_seed,
        "evaluation_mask_seed": eval_seed,
        "mean_consecutive_training_mask_jaccard": (
            jaccard_sum / jaccard_count if jaccard_count else float("nan")
        ),
        "training_mask_draws": epoch + 1,
    }
    return model, metrics, diagnostics, initial_hash, eval_mask.detach().cpu()


def save_mask(path: Path, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, mask=mask.detach().cpu().bool().numpy().astype(np.uint8))
    temporary.replace(path)


def evaluate_random_resampled(args, protocol: Mapping[str, object], protocol_path: Path, device):
    outdir = Path(str(protocol["output_root"]))
    run_dir = outdir / "evaluation_runs"
    mask_dir = outdir / "masks"
    pred_dir = outdir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    budgets = [float(value) for value in selected(args.budgets, protocol["evaluation"]["budgets"], "budgets")]
    seeds = [int(value) for value in selected(args.seeds, protocol["evaluation"]["model_seeds"], "seeds")]
    for dataset in datasets:
        data, num_features, num_classes, split, base_diagnostics = dataset_context(dataset, protocol)
        for budget in budgets:
            ratio = int(round(100 * budget))
            for seed in seeds:
                path = run_dir / f"eval_random_resampled_{dataset}_r{ratio:03d}_s{seed}.json"
                mask_path = mask_dir / f"mask_random_resampled_{dataset}_r{ratio:03d}_s{seed}.npz"
                prediction_path = pred_dir / f"pred_random_resampled_{dataset}_r{ratio:03d}_s{seed}.npz"
                if path.exists() and mask_path.exists() and prediction_path.exists() and not args.force:
                    print(f"[random-resampled:skip] {dataset} rho={budget} s={seed}", flush=True)
                    continue
                print(f"[random-resampled] {dataset} rho={budget} s={seed}", flush=True)
                model, metrics, diagnostics, initial_hash, eval_mask = train_random_resampled(
                    protocol=protocol,
                    data=data,
                    split=split,
                    num_features=num_features,
                    num_classes=num_classes,
                    keep_ratio=budget,
                    seed=seed,
                    device=device,
                )
                predictions = metrics.pop("test_predictions")
                labels = metrics.pop("test_labels")
                save_mask(mask_path, eval_mask)
                save_predictions(prediction_path, labels, predictions)
                record = {
                    "stage": "random_resampled_test_evaluation",
                    "created_at": timestamp(),
                    "protocol_hash": file_hash(protocol_path),
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "method": "Random-Resampled",
                    "budget": budget,
                    "seed": seed,
                    "initial_state_hash": initial_hash,
                    "evaluation_mask_path": str(mask_path.resolve()),
                    "evaluation_mask_hash": file_hash(mask_path),
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_hash": file_hash(prediction_path),
                    **base_diagnostics,
                    **diagnostics,
                    **metrics,
                }
                write_json(path, record)
                print(
                    f"[random-resampled:done] {dataset} rho={budget} s={seed} test={record['test_acc']:.2f}",
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del data


def summarize_random_resampled(protocol: Mapping[str, object], protocol_path: Path):
    outdir = Path(str(protocol["output_root"]))
    rows = [read_json(path) for path in sorted((outdir / "evaluation_runs").glob("eval_*.json"))]
    expected = {
        (dataset, float(budget), int(seed))
        for dataset in protocol["datasets"]
        for budget in protocol["evaluation"]["budgets"]
        for seed in protocol["evaluation"]["model_seeds"]
    }
    observed = {
        (str(row["dataset_id"]), float(row["budget"]), int(row["seed"]))
        for row in rows
    }
    if expected != observed:
        raise RuntimeError(
            f"Incomplete Random-Resampled evaluation: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    fixed_accuracy = current_seed_accuracy()
    method_summary: List[Dict[str, object]] = []
    comparisons: List[Dict[str, object]] = []
    for dataset in protocol["datasets"]:
        for budget in protocol["evaluation"]["budgets"]:
            budget = float(budget)
            group = [
                row for row in rows
                if row["dataset_id"] == dataset and float(row["budget"]) == budget
            ]
            for method in ("EHGNN-F", "Random-Fixed", "Random-Resampled"):
                if method == "Random-Resampled":
                    values = [float(row["test_acc"]) for row in group]
                else:
                    source_name = "Random-Fixed" if method == "Random-Fixed" else method
                    values = [
                        fixed_accuracy[(dataset, budget, int(seed), source_name)]
                        for seed in protocol["evaluation"]["model_seeds"]
                    ]
                mean, std = mean_std(values)
                item = {
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "method": method,
                    "n": len(values),
                    "test_accuracy_mean": mean,
                    "test_accuracy_std": std,
                }
                if method == "Random-Resampled":
                    for metric in ("test_macro_f1", "test_balanced_accuracy", "end_to_end_seconds", "mean_consecutive_training_mask_jaccard"):
                        item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(
                            [float(row[metric]) for row in group]
                        )
                method_summary.append(item)

            resampled = {int(row["seed"]): float(row["test_acc"]) for row in group}
            contrast_specs = (
                ("EHGNN-F minus Random-Resampled", "EHGNN-F", "resampled"),
                ("Random-Resampled minus Random-Fixed", "resampled", "Random-Fixed"),
                ("EHGNN-F minus Random-Fixed", "EHGNN-F", "Random-Fixed"),
            )
            for name, left, right in contrast_specs:
                values = []
                for seed in protocol["evaluation"]["model_seeds"]:
                    seed = int(seed)
                    left_value = resampled[seed] if left == "resampled" else fixed_accuracy[(dataset, budget, seed, left)]
                    right_value = resampled[seed] if right == "resampled" else fixed_accuracy[(dataset, budget, seed, right)]
                    values.append(left_value - right_value)
                low, high = paired_ci(values)
                comparisons.append({
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "comparison": name,
                    "n": len(values),
                    "paired_delta_mean": float(np.mean(values)),
                    "paired_delta_ci95_low": low,
                    "paired_delta_ci95_high": high,
                    "wins": int(np.sum(np.asarray(values) > 0)),
                    "ties": int(np.sum(np.isclose(values, 0.0))),
                    "losses": int(np.sum(np.asarray(values) < 0)),
                    "wilcoxon_p_raw": wilcoxon_p(values),
                })
    for family in (
        "EHGNN-F minus Random-Resampled",
        "Random-Resampled minus Random-Fixed",
    ):
        family_rows = [row for row in comparisons if row["comparison"] == family]
        adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in family_rows])
        for row, value in zip(family_rows, adjusted):
            row["wilcoxon_p_holm_24"] = value
    for row in comparisons:
        row.setdefault("wilcoxon_p_holm_24", float("nan"))
    write_csv(outdir / "evaluation_runs.csv", rows)
    write_csv(outdir / "method_summary.csv", method_summary)
    write_csv(outdir / "paired_comparisons.csv", comparisons)

    summary_lookup = {
        (row["dataset_id"], float(row["budget"]), row["method"]): row
        for row in method_summary
    }
    comparison_lookup = {
        (row["dataset_id"], float(row["budget"]), row["comparison"]): row
        for row in comparisons
    }
    lines = [
        "# Random-Resampled result",
        "",
        "All methods use the same exact original-incidence budget and the same fixed self-loops. Accuracy is ten-seed mean +/- sample standard deviation.",
        "",
        "| Dataset | Budget | EHGNN-F | Random-Resampled | Random-Fixed | EHGNN-F - Resampled (95% CI) | Holm p | Resampled - Fixed (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in protocol["datasets"]:
        for budget in protocol["evaluation"]["budgets"]:
            budget = float(budget)
            learned = summary_lookup[(dataset, budget, "EHGNN-F")]
            resampled = summary_lookup[(dataset, budget, "Random-Resampled")]
            fixed = summary_lookup[(dataset, budget, "Random-Fixed")]
            primary = comparison_lookup[(dataset, budget, "EHGNN-F minus Random-Resampled")]
            secondary = comparison_lookup[(dataset, budget, "Random-Resampled minus Random-Fixed")]
            lines.append(
                f"| {protocol['display_names'][dataset]} | {100*budget:.0f}% "
                f"| {learned['test_accuracy_mean']:.2f} +/- {learned['test_accuracy_std']:.2f} "
                f"| {resampled['test_accuracy_mean']:.2f} +/- {resampled['test_accuracy_std']:.2f} "
                f"| {fixed['test_accuracy_mean']:.2f} +/- {fixed['test_accuracy_std']:.2f} "
                f"| {primary['paired_delta_mean']:+.2f} [{primary['paired_delta_ci95_low']:+.2f}, {primary['paired_delta_ci95_high']:+.2f}] "
                f"| {primary['wilcoxon_p_holm_24']:.4f} "
                f"| {secondary['paired_delta_mean']:+.2f} [{secondary['paired_delta_ci95_low']:+.2f}, {secondary['paired_delta_ci95_high']:+.2f}] |"
            )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_hash": file_hash(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "run_count": len(rows),
        "outputs": {
            name: file_hash(outdir / name)
            for name in (
                "evaluation_runs.csv",
                "method_summary.csv",
                "paired_comparisons.csv",
                "summary.md",
            )
        },
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "mlp-tune",
            "mlp-select",
            "mlp-evaluate",
            "mlp-summarize",
            "random-resampled-evaluate",
            "random-resampled-summarize",
        ),
    )
    parser.add_argument("--protocol")
    parser.add_argument("--selected-hyperparameters")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--config-indices", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_mlp = args.stage.startswith("mlp-")
    default = DEFAULT_MLP_PROTOCOL if is_mlp else DEFAULT_RANDOM_RESAMPLED_PROTOCOL
    protocol_path = Path(args.protocol or default)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "fixed":
        raise RuntimeError("Experiment specification is not fixed")
    verify_dependencies(protocol_path, protocol)
    device = torch.device(args.device)
    if args.stage == "mlp-tune":
        tune_mlp(args, protocol, protocol_path, device)
    elif args.stage == "mlp-select":
        select_mlp(protocol, protocol_path)
    elif args.stage == "mlp-evaluate":
        evaluate_mlp(args, protocol, protocol_path, device)
    elif args.stage == "mlp-summarize":
        summarize_mlp(protocol, protocol_path)
    elif args.stage == "random-resampled-evaluate":
        evaluate_random_resampled(args, protocol, protocol_path, device)
    elif args.stage == "random-resampled-summarize":
        summarize_random_resampled(protocol, protocol_path)


if __name__ == "__main__":
    main()
