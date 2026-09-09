#!/usr/bin/env python3
"""Evaluate HGNN+HERALD with validation-selected settings."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from node_classification_utils import fixed_split
from training_utils import clone_state_dict, setup_seed
from experiment_controls.core_controls import (
    classification_metrics,
    load_dataset,
    split_hash,
    tensor_hash,
)


VENDOR_ROOT = Path("third_party/HERALD").resolve()
DEFAULT_PROTOCOL = Path("experiment_specs/herald.json")
Generate_G_from_H = None
HERALD = None


def load_herald_components() -> None:
    global Generate_G_from_H, HERALD
    if Generate_G_from_H is not None:
        return
    if not VENDOR_ROOT.exists():
        raise RuntimeError("HERALD is not installed; run scripts/fetch_third_party.sh first")
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    from herald import Generate_G_from_H as VendorGenerateG
    from herald import HERALD as VendorHERALD

    Generate_G_from_H = VendorGenerateG
    HERALD = VendorHERALD


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def verify_protocol(path: Path) -> Dict[str, object]:
    protocol = read_json(path)
    if protocol.get("protocol_version") != "herald-v1":
        raise RuntimeError("Unexpected HERALD specification version")
    if protocol.get("status") != "fixed":
        raise RuntimeError("HERALD experiment specification is not fixed")
    for item in protocol["dependencies"]:
        source = Path(str(item["path"]))
        if file_hash(source) != item["sha256"]:
            raise RuntimeError(f"HERALD dependency changed: {source}")
    return protocol


def selected(values: Optional[Sequence], allowed: Sequence, label: str) -> List:
    result = list(allowed if values is None else values)
    if not set(result) <= set(allowed):
        raise ValueError(f"Requested {label} outside the experiment specification: {result}")
    return result


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
        "num_features": int(num_features),
        "num_classes": int(num_classes),
    }
    return data, int(num_features), int(num_classes), split, diagnostics


def dense_incidence(data, device: torch.device) -> torch.Tensor:
    incidence = torch.zeros(
        (int(data.n_x), int(data.num_hyperedges)),
        dtype=torch.float32,
        device=device,
    )
    edge_index = data.edge_index.to(device)
    incidence[edge_index[0], edge_index[1]] = 1.0
    return incidence


class HeraldHGNN(nn.Module):
    """Paper-specified three-layer HGNN with HERALD in layers two and three."""

    def __init__(
        self,
        num_features: int,
        hidden: int,
        num_classes: int,
        herald_hidden: int,
        dropout: float,
        base_theta: float,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.layer1 = nn.Linear(num_features, hidden, bias=True)
        self.layer2 = nn.Linear(hidden, hidden, bias=True)
        self.layer3 = nn.Linear(hidden, num_classes, bias=True)
        self.herald2 = HERALD(
            in_feature=hidden,
            hidden=herald_hidden,
            only_G=True,
            theta=base_theta,
        )
        self.herald3 = HERALD(
            in_feature=hidden,
            hidden=herald_hidden,
            only_G=True,
            theta=base_theta,
        )

    def forward(
        self,
        incidence: torch.Tensor,
        original_g: torch.Tensor,
        features: torch.Tensor,
    ):
        x = F.relu(original_g.mm(self.layer1(features)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        learned_g2 = self.herald2(
            adj=incidence,
            G=original_g,
            feats=x,
            num=2,
        )
        residual_g2 = self.herald2.adj
        x = F.relu(learned_g2.mm(self.layer2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        learned_g3 = self.herald3(
            adj=incidence,
            G=original_g,
            feats=x,
            num=3,
        )
        residual_g3 = self.herald3.adj
        logits = learned_g3.mm(self.layer3(x))
        return logits, (residual_g2, residual_g3)

    def clear_residual_references(self) -> None:
        # The released module stores its dense residual operator on each block.
        self.herald2.adj = None
        self.herald3.adj = None


def stability_regularizer(
    original_g: torch.Tensor,
    residual_operators: Sequence[torch.Tensor],
) -> torch.Tensor:
    return sum(
        torch.linalg.vector_norm(original_g - residual)
        for residual in residual_operators
    )


@torch.no_grad()
def evaluate(
    model: HeraldHGNN,
    incidence: torch.Tensor,
    original_g: torch.Tensor,
    data,
    split: Mapping[str, torch.Tensor],
    include_test: bool,
):
    model.eval()
    logits, residuals = model(incidence, original_g, data.x)
    logp = F.log_softmax(logits, dim=1)
    names = ("train", "valid", "test") if include_test else ("train", "valid")
    metrics: Dict[str, float] = {}
    for name in names:
        index = split[name]
        metrics[f"{name}_loss"] = float(F.nll_loss(logp[index], data.y[index]).item())
        metrics[f"{name}_acc"] = 100.0 * float(
            (logp[index].argmax(dim=1) == data.y[index]).float().mean().item()
        )
    residual_distances = [
        float(torch.linalg.vector_norm(original_g - residual).item())
        for residual in residuals
    ]
    model.clear_residual_references()
    return metrics, logp, residual_distances


def train_run(
    *,
    dataset: str,
    config: Mapping[str, object],
    seed: int,
    stage: str,
    protocol: Mapping[str, object],
    protocol_hash: str,
    device: torch.device,
) -> Dict[str, object]:
    load_herald_components()
    data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
    setup_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    local_split = {name: index.to(device) for name, index in split.items()}
    include_test = stage == "test_evaluation"
    if not include_test:
        local_data.y[local_split["test"]] = -1

    incidence = dense_incidence(local_data, device)
    with torch.no_grad():
        original_g = Generate_G_from_H().to(device)(incidence)

    fixed = protocol["model"]
    model = HeraldHGNN(
        num_features=num_features,
        hidden=int(config["classifier_hidden"]),
        num_classes=num_classes,
        herald_hidden=int(fixed["herald_hidden"]),
        dropout=float(config["dropout"]),
        base_theta=float(fixed["base_theta"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    maximum_epochs = int(protocol["training"]["maximum_epochs"])
    patience = int(protocol["training"]["patience"])
    regularizer_weight = float(protocol["training"]["stability_weight"])
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    start = time.perf_counter()
    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, residuals = model(incidence, original_g, local_data.x)
        task_loss = F.cross_entropy(
            logits[local_split["train"]],
            local_data.y[local_split["train"]],
        )
        regularizer = stability_regularizer(original_g, residuals)
        loss = task_loss + regularizer_weight * regularizer
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite HERALD loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        model.clear_residual_references()
        del logits, residuals, task_loss, regularizer, loss

        validation, _, _ = evaluate(
            model,
            incidence,
            original_g,
            local_data,
            local_split,
            include_test=False,
        )
        val_loss = float(validation["valid_loss"])
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("No finite HERALD validation checkpoint")
    model.load_state_dict(best_state)
    metrics, logp, residual_distances = evaluate(
        model,
        incidence,
        original_g,
        local_data,
        local_split,
        include_test=include_test,
    )
    result: Dict[str, object] = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": stage,
        "status": "complete",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "config_id": config["config_id"],
        "classifier_hidden": int(config["classifier_hidden"]),
        "herald_hidden": int(fixed["herald_hidden"]),
        "learning_rate": float(config["learning_rate"]),
        "dropout": float(config["dropout"]),
        "seed": int(seed),
        **diagnostics,
        "learned_structure": "dense_weighted_all_node_hyperedge_pairs",
        "learned_incidence_candidate_count": int(
            diagnostics["num_nodes"] * diagnostics["num_hyperedges"]
        ),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "end_to_end_seconds": float(elapsed),
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "train_acc": metrics["train_acc"],
        "val_acc": metrics["valid_acc"],
        "val_loss": metrics["valid_loss"],
        "residual_layer2_distance": residual_distances[0],
        "residual_layer3_distance": residual_distances[1],
    }
    if include_test:
        index = local_split["test"]
        predictions = logp[index].argmax(dim=1).detach().cpu().numpy()
        labels = local_data.y[index].detach().cpu().numpy()
        detail = classification_metrics(labels, predictions, num_classes)
        result.update({
            "test_acc": metrics["test_acc"],
            "test_macro_f1": float(detail["macro_f1"]),
            "test_balanced_accuracy": float(detail["balanced_accuracy"]),
            "test_per_class_recall": detail["per_class_recall"],
        })
    return result


def run_path(outdir: Path, stage: str, dataset: str, config_id: str, seed: int) -> Path:
    return outdir / stage / "run_records" / f"{dataset}_{config_id}_s{seed}.json"


def failure_result(protocol, protocol_hash, dataset, config, seed, stage, error):
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": stage,
        "status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "config_id": config["config_id"],
        "classifier_hidden": int(config["classifier_hidden"]),
        "herald_hidden": int(protocol["model"]["herald_hidden"]),
        "learning_rate": float(config["learning_rate"]),
        "dropout": float(config["dropout"]),
        "seed": int(seed),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def inherited_oom_result(protocol, protocol_hash, dataset, config, seed, source):
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": "validation_tuning",
        "status": "oom_inherited_from_minimum_configuration",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "config_id": config["config_id"],
        "classifier_hidden": int(config["classifier_hidden"]),
        "herald_hidden": int(protocol["model"]["herald_hidden"]),
        "learning_rate": float(config["learning_rate"]),
        "dropout": float(config["dropout"]),
        "seed": int(seed),
        "source_oom_record": str(source),
        "reason": protocol["failure_policy"]["minimum_configuration_inheritance"],
    }


def run_tuning(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    config_ids = selected(
        args.configs,
        [config["config_id"] for config in protocol["validation_grid"]],
        "configs",
    )
    seeds = selected(args.seeds, protocol["tuning"]["seeds"], "seeds")
    configs = {config["config_id"]: config for config in protocol["validation_grid"]}
    ordered_configs = [configs[config_id] for config_id in config_ids]
    outdir = Path(str(protocol["output_root"]))
    protocol_hash = file_hash(protocol_path)
    device = torch.device(args.device)
    for dataset in datasets:
        inherited_from = None
        for config in ordered_configs:
            for seed in seeds:
                path = run_path(outdir, "validation", dataset, config["config_id"], int(seed))
                if path.exists() and not args.force:
                    observed = read_json(path)
                    if observed["status"] == "oom" and config["config_id"] == protocol["failure_policy"]["minimum_config_id"]:
                        inherited_from = path
                    print(f"[herald:tune:skip] {dataset} {config['config_id']} s={seed}", flush=True)
                    continue
                print(f"[herald:tune] {dataset} {config['config_id']} s={seed}", flush=True)
                if inherited_from is not None:
                    result = inherited_oom_result(
                        protocol, protocol_hash, dataset, config, int(seed), inherited_from
                    )
                else:
                    try:
                        result = train_run(
                            dataset=dataset,
                            config=config,
                            seed=int(seed),
                            stage="validation_tuning",
                            protocol=protocol,
                            protocol_hash=protocol_hash,
                            device=device,
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                        result = failure_result(
                            protocol, protocol_hash, dataset, config, int(seed),
                            "validation_tuning", error,
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        if (
                            result["status"] == "oom"
                            and config["config_id"] == protocol["failure_policy"]["minimum_config_id"]
                        ):
                            inherited_from = path
                write_json(path, result)
                print(f"[herald:tune:done] status={result['status']}", flush=True)


def freeze_selection(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    outdir = Path(str(protocol["output_root"]))
    manifest_path = outdir / "frozen_selection.json"
    if manifest_path.exists() and not args.force:
        raise RuntimeError("HERALD selection is already frozen")
    choices = {}
    rows = []
    for dataset in protocol["datasets"]:
        candidates = []
        for config in protocol["validation_grid"]:
            run_rows = []
            for seed in protocol["tuning"]["seeds"]:
                path = run_path(outdir, "validation", dataset, config["config_id"], int(seed))
                if not path.exists():
                    raise RuntimeError(f"Missing HERALD validation run: {path}")
                run_rows.append(read_json(path))
            complete = [row for row in run_rows if row["status"] == "complete"]
            mean_val = float(np.mean([row["val_acc"] for row in complete])) if complete else -math.inf
            row = {
                "dataset_id": dataset,
                "config_id": config["config_id"],
                "complete_seeds": len(complete),
                "mean_val_acc": mean_val,
                "classifier_hidden": config["classifier_hidden"],
                "learning_rate": config["learning_rate"],
                "dropout": config["dropout"],
            }
            rows.append(row)
            if len(complete) == len(protocol["tuning"]["seeds"]):
                candidates.append(row)
        if not candidates:
            choices[dataset] = {"status": "oom_no_feasible_configuration"}
            continue
        winner = sorted(
            candidates,
            key=lambda row: (
                -row["mean_val_acc"],
                row["classifier_hidden"],
                row["dropout"],
                row["config_id"],
            ),
        )[0]
        choices[dataset] = {"status": "selected", **winner}
    write_csv(outdir / "validation_grid_summary.csv", rows)
    write_json(manifest_path, {
        "status": "frozen_before_test_evaluation",
        "created_at": timestamp(),
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "selection_rule": protocol["tuning"]["selection_rule"],
        "choices": choices,
    })


def run_evaluation(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    outdir = Path(str(protocol["output_root"]))
    manifest = read_json(outdir / "frozen_selection.json")
    if manifest.get("status") != "frozen_before_test_evaluation":
        raise RuntimeError("HERALD selection manifest is not frozen")
    if manifest.get("protocol_hash") != file_hash(protocol_path):
        raise RuntimeError("HERALD selection manifest protocol hash mismatch")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    seeds = selected(args.seeds, protocol["evaluation"]["seeds"], "seeds")
    configs = {config["config_id"]: config for config in protocol["validation_grid"]}
    device = torch.device(args.device)
    for dataset in datasets:
        choice = manifest["choices"][dataset]
        if choice["status"] != "selected":
            print(f"[herald:evaluate:oom] {dataset}", flush=True)
            continue
        config = configs[choice["config_id"]]
        for seed in seeds:
            path = run_path(outdir, "evaluation", dataset, config["config_id"], int(seed))
            if path.exists() and not args.force:
                print(f"[herald:evaluate:skip] {dataset} s={seed}", flush=True)
                continue
            print(f"[herald:evaluate] {dataset} {config['config_id']} s={seed}", flush=True)
            try:
                result = train_run(
                    dataset=dataset,
                    config=config,
                    seed=int(seed),
                    stage="test_evaluation",
                    protocol=protocol,
                    protocol_hash=file_hash(protocol_path),
                    device=device,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                result = failure_result(
                    protocol, file_hash(protocol_path), dataset, config, int(seed),
                    "test_evaluation", error,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            write_json(path, result)
            print(f"[herald:evaluate:done] status={result['status']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "freeze-selection", "evaluate"])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "tune":
        run_tuning(args)
    elif args.command == "freeze-selection":
        freeze_selection(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
