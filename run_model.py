#!/usr/bin/env python3
"""Run one node-classification model using the paper's fixed specification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from structural_baselines import edge_scores, prefix_mask
from node_classification_utils import train_learned_with_trajectory
from training_utils import train_model, unit_scores_and_mask_for_baseline
from experiment_controls.core_controls import (
    dataset_context,
    majority_metrics,
    train_mlp,
    train_random_resampled,
)
from main_accuracy import forward_diagnostics


ROOT = Path(__file__).resolve().parent
CORE_PROTOCOL = ROOT / "experiment_specs/main_accuracy.json"
CORE_CONFIG = ROOT / "experiment_specs/main_selected_hyperparameters.json"
STRUCTURAL_PROTOCOL = ROOT / "experiment_specs/structural_baselines.json"
MLP_PROTOCOL = ROOT / "experiment_specs/mlp_baseline.json"
MLP_CONFIG = ROOT / "experiment_specs/mlp_selected_hyperparameters.json"
RESAMPLED_PROTOCOL = ROOT / "experiment_specs/random_resampled.json"

MODELS = (
    "full",
    "ehgnn-f",
    "random-fixed",
    "random-resampled",
    "cardinality",
    "laplacian-proxy",
    "mlp",
    "majority",
)
DATASETS = (
    "actor",
    "twitch",
    "pokec",
    "yelp",
    "coauthor_dblp",
    "walmart-trips",
)
SPARSE_MODELS = {
    "ehgnn-f",
    "random-fixed",
    "random-resampled",
    "cardinality",
    "laplacian-proxy",
}


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_args(protocol: Mapping[str, object]) -> Dict[str, object]:
    cfg = protocol["training"]
    return {
        "hidden": int(cfg["hidden"]),
        "dropout": float(cfg["dropout"]),
        "lr": float(cfg["learning_rate"]),
        "wd": float(cfg["weight_decay"]),
        "epochs": int(cfg["maximum_epochs"]),
        "patience": int(cfg["patience"]),
        "sampling": "multinomial",
        "sparse_self_loop_policy": str(cfg["sparse_self_loop_policy"]),
    }


def save_mask(path: Path, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        mask=mask.detach().cpu().bool().numpy().astype(np.uint8),
    )
    temporary.replace(path)


def train_fixed_mask(
    *,
    data,
    split,
    num_features: int,
    num_classes: int,
    selected: torch.Tensor,
    budget: float,
    seed: int,
    device: torch.device,
    train_cfg: Mapping[str, object],
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    sparse_data = copy.deepcopy(data)
    sparse_data.edge_index = data.edge_index[:, selected.bool()]
    return train_model(
        data=sparse_data,
        split_idx=split,
        mode="random",
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=budget,
        seed=seed,
        device=device,
        **train_cfg,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--budget", type=float, default=0.5, choices=(0.1, 0.2, 0.3, 0.5))
    parser.add_argument("--seed", type=int, default=0, choices=range(10))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model not in SPARSE_MODELS and args.budget != 0.5:
        raise ValueError(f"--budget does not apply to {args.model}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly for a CPU run")

    invocation_dir = Path.cwd()
    output = args.output
    if output is None:
        budget = f"_r{int(round(100 * args.budget)):03d}" if args.model in SPARSE_MODELS else ""
        output = invocation_dir / "outputs" / "single_runs" / (
            f"{args.model}_{args.dataset}{budget}_s{args.seed}.json"
        )
    elif not output.is_absolute():
        output = invocation_dir / output
    output = output.resolve()

    os.chdir(ROOT)
    resampled_protocol = read_json(RESAMPLED_PROTOCOL)
    core_protocol = read_json(CORE_PROTOCOL)
    structural_protocol = read_json(STRUCTURAL_PROTOCOL)
    data, num_features, num_classes, split, data_info = dataset_context(
        args.dataset, resampled_protocol
    )
    device = torch.device(args.device)
    train_cfg = training_args(resampled_protocol)
    mask: Optional[torch.Tensor] = None
    diagnostics: Dict[str, object] = {}

    if args.model == "full":
        model, metrics = train_model(
            data=data,
            split_idx=split,
            mode="full",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=1.0,
            seed=args.seed,
            device=device,
            **train_cfg,
        )
        val_accuracy = 100.0 * float(metrics["val_acc"])
        test_accuracy = 100.0 * float(metrics["test_acc"])
    elif args.model == "ehgnn-f":
        selected = read_json(CORE_CONFIG)["selected"]
        model, metrics, initial_mask, _, mask, final_weights, _ = train_learned_with_trajectory(
            data=data,
            split_idx=split,
            mode="learnmask",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=args.budget,
            seed=args.seed,
            device=device,
            trajectory_every=int(train_cfg["epochs"]) + 1,
            mask_init_std=float(selected["mask_init_std"]),
            mask_lr_multiplier=float(selected["mask_lr_multiplier"]),
            include_test_metrics=True,
            **train_cfg,
        )
        intersection = int((initial_mask & mask).sum().item())
        union = int((initial_mask | mask).sum().item())
        diagnostics.update(
            {
                "initial_final_jaccard": intersection / max(1, union),
                "minimum_final_sampling_weight": float(final_weights.min().item()),
                **forward_diagnostics(model, data, int(mask.sum().item())),
            }
        )
        val_accuracy = 100.0 * float(metrics["val_acc"])
        test_accuracy = 100.0 * float(metrics["test_acc"])
    elif args.model == "random-fixed":
        _, mask, _ = unit_scores_and_mask_for_baseline(
            data,
            method="Random-Fixed",
            unit="incidence",
            keep_ratio=args.budget,
            seed=int(core_protocol["evaluation"]["random_mask_seed_offset"]) + args.seed,
        )
        model, metrics = train_fixed_mask(
            data=data,
            split=split,
            num_features=num_features,
            num_classes=num_classes,
            selected=mask,
            budget=args.budget,
            seed=args.seed,
            device=device,
            train_cfg=train_cfg,
        )
        diagnostics.update(forward_diagnostics(model, data, int(mask.sum().item())))
        val_accuracy = 100.0 * float(metrics["val_acc"])
        test_accuracy = 100.0 * float(metrics["test_acc"])
    elif args.model == "random-resampled":
        model, metrics, diagnostics, _, mask = train_random_resampled(
            protocol=resampled_protocol,
            data=data,
            split=split,
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=args.budget,
            seed=args.seed,
            device=device,
        )
        metrics.pop("test_predictions")
        metrics.pop("test_labels")
        val_accuracy = float(metrics["val_acc"])
        test_accuracy = float(metrics["test_acc"])
    elif args.model in {"cardinality", "laplacian-proxy"}:
        method = "Degree-prefix" if args.model == "cardinality" else "Spectral-prefix"
        scores = edge_scores(data, method, structural_protocol, device)
        mask, diagnostics = prefix_mask(data, scores, args.budget)
        model, metrics = train_fixed_mask(
            data=data,
            split=split,
            num_features=num_features,
            num_classes=num_classes,
            selected=mask,
            budget=args.budget,
            seed=args.seed,
            device=device,
            train_cfg=train_cfg,
        )
        diagnostics.update(forward_diagnostics(model, data, int(mask.sum().item())))
        val_accuracy = 100.0 * float(metrics["val_acc"])
        test_accuracy = 100.0 * float(metrics["test_acc"])
    elif args.model == "mlp":
        mlp_protocol = read_json(MLP_PROTOCOL)
        config = read_json(MLP_CONFIG)["selected_by_dataset"][args.dataset]
        model, metrics, _ = train_mlp(
            protocol=mlp_protocol,
            config=config,
            data=data,
            split=split,
            num_features=num_features,
            num_classes=num_classes,
            seed=args.seed,
            device=device,
            include_test=True,
        )
        metrics.pop("test_predictions")
        metrics.pop("test_labels")
        val_accuracy = float(metrics["val_acc"])
        test_accuracy = float(metrics["test_acc"])
        diagnostics["mlp_config"] = {
            key: config[key]
            for key in ("hidden", "learning_rate", "dropout", "weight_decay")
        }
    else:
        model = None
        metrics = majority_metrics(data, split, num_classes)
        val_accuracy = float(metrics["val_accuracy"])
        test_accuracy = float(metrics["test_accuracy"])
        diagnostics["majority_class"] = int(metrics["majority_class"])

    if mask is not None:
        mask_path = output.with_suffix(".mask.npz")
        save_mask(mask_path, mask)
        diagnostics.update(
            {
                "mask_path": str(mask_path),
                "selected_original_incidences": int(mask.sum().item()),
                "realized_structural_density": float(mask.float().mean().item()),
            }
        )

    record = {
        "model": args.model,
        "dataset": args.dataset,
        "nominal_budget": args.budget if args.model in SPARSE_MODELS else None,
        "seed": args.seed,
        "device": str(device),
        "validation_accuracy_percent": val_accuracy,
        "test_accuracy_percent": test_accuracy,
        "best_epoch": int(metrics.get("best_epoch", -1)),
        "epochs_run": int(metrics.get("epochs_run", 0)),
        "protocol_sha256": {
            "core": sha256(CORE_PROTOCOL),
            "structural": sha256(STRUCTURAL_PROTOCOL),
            "resampled": sha256(RESAMPLED_PROTOCOL),
        },
        "data": data_info,
        "diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    temporary.replace(output)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
