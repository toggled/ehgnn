#!/usr/bin/env python3
"""Evaluate robustness across independently generated data splits."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from node_classification_utils import fixed_split, train_learned_with_trajectory
from training_utils import train_model, unit_scores_and_mask_for_baseline
from experiment_controls.core_controls import (
    classification_metrics,
    file_hash,
    holm_adjust,
    load_dataset,
    mean_std,
    paired_ci,
    read_json,
    save_mask,
    save_predictions,
    selected,
    split_hash,
    state_hash,
    tensor_hash,
    timestamp,
    train_mlp,
    train_random_resampled,
    verify_dependencies,
    wilcoxon_p,
    write_csv,
    write_json,
)


DEFAULT_PROTOCOL = Path("experiment_specs/split_robustness.json")


def dataset_diagnostics(data, num_features: int, num_classes: int) -> Dict[str, object]:
    return {
        "feature_hash": tensor_hash(data.x),
        "label_hash": tensor_hash(data.y),
        "edge_index_hash": tensor_hash(data.edge_index),
        "num_nodes": int(data.n_x),
        "num_hyperedges": int(data.num_hyperedges),
        "num_original_incidences": int(data.edge_index.size(1)),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
    }


def class_counts(labels: torch.Tensor, mask: torch.Tensor, num_classes: int) -> List[int]:
    values = labels[mask].detach().cpu().long()
    return torch.bincount(values, minlength=num_classes).tolist()


def audit_splits(protocol: Mapping[str, object], protocol_path: Path, force: bool) -> None:
    if protocol.get("status") != "fixed":
        raise RuntimeError("Experiment specification is not fixed")
    outdir = Path(str(protocol["output_root"]))
    manifest_path = outdir / "split_manifest.json"
    if manifest_path.exists() and not force:
        raise RuntimeError("Split manifest already exists; use --force to replace it")
    cfg = protocol["data"]
    candidates = [int(value) for value in cfg["candidate_split_seeds"]]
    acceptance_by_seed = {seed: True for seed in candidates}
    rows: List[Dict[str, object]] = []
    split_hashes: Dict[str, Dict[str, str]] = {}
    for dataset in protocol["datasets"]:
        data, num_features, num_classes = load_dataset(dataset, protocol)
        diagnostics = dataset_diagnostics(data, num_features, num_classes)
        split_hashes[dataset] = {}
        for seed in candidates:
            split = fixed_split(
                data,
                seed,
                float(cfg["train_prop"]),
                float(cfg["valid_prop"]),
            )
            counts = {
                name: class_counts(data.y, split[name], num_classes)
                for name in ("train", "valid", "test")
            }
            accepted = all(value > 0 for value in counts["train"])
            acceptance_by_seed[seed] = acceptance_by_seed[seed] and accepted
            current_hash = split_hash(split)
            split_hashes[dataset][str(seed)] = current_hash
            rows.append({
                "dataset_id": dataset,
                "dataset": protocol["display_names"][dataset],
                "split_seed": seed,
                "accepted_for_dataset": accepted,
                "split_hash": current_hash,
                "train_size": int(split["train"].sum()),
                "valid_size": int(split["valid"].sum()),
                "test_size": int(split["test"].sum()),
                "train_class_counts": counts["train"],
                "valid_class_counts": counts["valid"],
                "test_class_counts": counts["test"],
                **diagnostics,
            })
    accepted = [seed for seed in candidates if acceptance_by_seed[seed]]
    required = int(cfg["required_accepted_splits"])
    if len(accepted) != required:
        raise RuntimeError(
            f"Expected {required} accepted splits, observed {len(accepted)}: {accepted}"
        )
    audit_path = outdir / "split_audit.csv"
    write_csv(audit_path, rows)
    manifest = {
        "status": "fixed_for_evaluation",
        "created_at": timestamp(),
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "split_audit_path": str(audit_path.resolve()),
        "split_audit_hash": file_hash(audit_path),
        "accepted_split_seeds": accepted,
        "acceptance_rule": cfg["acceptance_rule"],
        "split_hashes": split_hashes,
    }
    write_json(manifest_path, manifest)
    print(f"[split-robustness:audit] fixed {len(accepted)} splits: {accepted}", flush=True)


def verify_manifest(protocol: Mapping[str, object], protocol_path: Path) -> Dict[str, object]:
    outdir = Path(str(protocol["output_root"]))
    path = outdir / "split_manifest.json"
    manifest = read_json(path)
    if manifest.get("status") != "fixed_for_evaluation":
        raise RuntimeError("Split manifest is not fixed for evaluation")
    if manifest.get("protocol_hash") != file_hash(protocol_path):
        raise RuntimeError("Split-manifest specification hash mismatch")
    if file_hash(Path(str(manifest["split_audit_path"]))) != manifest["split_audit_hash"]:
        raise RuntimeError("Split-audit output changed after the manifest was created")
    required = int(protocol["data"]["required_accepted_splits"])
    if len(manifest["accepted_split_seeds"]) != required:
        raise RuntimeError("Fixed split count is incorrect")
    return manifest


def split_context(
    dataset: str,
    split_seed: int,
    protocol: Mapping[str, object],
    manifest: Mapping[str, object],
):
    data, num_features, num_classes = load_dataset(dataset, protocol)
    cfg = protocol["data"]
    split = fixed_split(
        data,
        split_seed,
        float(cfg["train_prop"]),
        float(cfg["valid_prop"]),
    )
    observed = split_hash(split)
    expected = manifest["split_hashes"][dataset][str(split_seed)]
    if observed != expected:
        raise RuntimeError(f"Fixed split hash changed for {dataset}, seed {split_seed}")
    return data, num_features, num_classes, split, {
        **dataset_diagnostics(data, num_features, num_classes),
        "split_hash": observed,
    }


def detailed_from_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[Dict[str, object], torch.Tensor]:
    predictions = probabilities.argmax(dim=1).detach().cpu()
    labels = labels.detach().cpu()
    return classification_metrics(labels.numpy(), predictions.numpy(), num_classes), predictions


@torch.no_grad()
def learned_outputs(model, data, split, num_classes: int, device: torch.device):
    local_data = copy.deepcopy(data).to(device)
    test_mask = split["test"].to(device)
    model.eval()
    logits = model(local_data, is_test=True)
    probabilities = F.softmax(logits[test_mask], dim=1)
    labels = local_data.y[test_mask]
    detail, predictions = detailed_from_probabilities(probabilities, labels, num_classes)
    return detail, labels.detach().cpu(), predictions


def normalized_sparse_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        "val_acc": 100.0 * float(metrics["val_acc"]),
        "test_acc": 100.0 * float(metrics["test_acc"]),
        "best_epoch": int(metrics["best_epoch"]),
        "epochs_run": int(metrics["epochs_run"]),
    }


def base_training(protocol: Mapping[str, object]) -> Dict[str, object]:
    cfg = protocol["training"]
    return {
        "hidden": int(cfg["hidden"]),
        "dropout": float(cfg["dropout"]),
        "lr": float(cfg["learning_rate"]),
        "wd": float(cfg["weight_decay"]),
        "epochs": int(cfg["maximum_epochs"]),
        "patience": int(cfg["patience"]),
        "sampling": str(cfg["sampling"]),
        "sparse_self_loop_policy": str(cfg["sparse_self_loop_policy"]),
    }


def train_one(
    *,
    method: str,
    budget: Optional[float],
    model_seed: int,
    data,
    num_features: int,
    num_classes: int,
    split,
    protocol: Mapping[str, object],
    device: torch.device,
):
    cfg = protocol["training"]
    common = base_training(protocol)
    start = time.perf_counter()
    mask = None
    structural: Dict[str, object] = {}

    if method == "EHGNN-F":
        model, metrics, _, _, mask, _, _ = train_learned_with_trajectory(
            data=data,
            split_idx=split,
            mode="learnmask",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=float(budget),
            seed=model_seed,
            device=device,
            trajectory_every=int(cfg["maximum_epochs"]) + 1,
            mask_init_std=float(cfg["ehgnnf_mask_init_std"]),
            mask_lr_multiplier=float(cfg["ehgnnf_mask_lr_multiplier"]),
            include_test_metrics=True,
            **common,
        )
        detail, labels, predictions = learned_outputs(
            model, data, split, num_classes, device
        )
        result = normalized_sparse_metrics(metrics)
        k = max(1, int(float(budget) * int(data.edge_index.size(1))))
        if int(mask.sum()) != k:
            raise RuntimeError("EHGNN-F violated its exact incidence budget")
        structural = {
            "num_selected": k,
            "num_mask_units": int(mask.numel()),
            "num_fixed_self_loops": int(getattr(model, "last_fixed_self_loop_count", -1)),
            "forward_incidences": int(getattr(model, "last_forward_incidence_count", -1)),
        }
    elif method == "Random-Fixed":
        mask_seed = 100000 + model_seed
        edge_index, mask, _ = unit_scores_and_mask_for_baseline(
            data,
            method="Random-Fixed",
            unit="incidence",
            keep_ratio=float(budget),
            seed=mask_seed,
        )
        random_data = copy.deepcopy(data)
        random_data.edge_index = edge_index
        model, metrics = train_model(
            data=random_data,
            split_idx=split,
            mode="random",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=float(budget),
            seed=model_seed,
            device=device,
            return_outputs=True,
            **common,
        )
        detail, predictions = detailed_from_probabilities(
            metrics.pop("_test_probs"), metrics.pop("_test_labels"), num_classes
        )
        labels = data.y[split["test"]].detach().cpu()
        result = normalized_sparse_metrics(metrics)
        k = max(1, int(float(budget) * int(data.edge_index.size(1))))
        if int(mask.sum()) != k:
            raise RuntimeError("Random-Fixed violated its exact incidence budget")
        structural = {
            "num_selected": k,
            "num_mask_units": int(mask.numel()),
            "num_fixed_self_loops": int(getattr(model, "last_fixed_self_loop_count", -1)),
            "forward_incidences": int(getattr(model, "last_forward_incidence_count", -1)),
            "mask_seed": mask_seed,
        }
    elif method == "Random-Resampled":
        resampled_protocol = {
            "training": {
                "hidden": int(cfg["hidden"]),
                "dropout": float(cfg["dropout"]),
                "learning_rate": float(cfg["learning_rate"]),
                "weight_decay": float(cfg["weight_decay"]),
                "maximum_epochs": int(cfg["maximum_epochs"]),
                "patience": int(cfg["patience"]),
                "sparse_self_loop_policy": str(cfg["sparse_self_loop_policy"]),
                "model_mode": "random",
            },
            "random_streams": {"training_mask_seed": "410000 + model seed"},
        }
        model, metrics, structural, _, mask = train_random_resampled(
            protocol=resampled_protocol,
            data=data,
            split=split,
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=float(budget),
            seed=model_seed,
            device=device,
        )
        predictions = metrics.pop("test_predictions")
        labels = metrics.pop("test_labels")
        detail = {
            "accuracy": metrics["test_acc"],
            "macro_f1": metrics["test_macro_f1"],
            "balanced_accuracy": metrics["test_balanced_accuracy"],
            "per_class_recall": metrics["test_per_class_recall"],
        }
        result = {
            key: metrics[key]
            for key in ("val_acc", "test_acc", "best_epoch", "epochs_run")
        }
    elif method == "Full":
        model, metrics = train_model(
            data=data,
            split_idx=split,
            mode="full",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=1.0,
            seed=model_seed,
            device=device,
            return_outputs=True,
            **common,
        )
        detail, predictions = detailed_from_probabilities(
            metrics.pop("_test_probs"), metrics.pop("_test_labels"), num_classes
        )
        labels = data.y[split["test"]].detach().cpu()
        result = normalized_sparse_metrics(metrics)
    elif method == "MLP":
        dependencies = protocol["dependencies"]
        selected_hyperparameters = read_json(
            Path(str(dependencies["mlp_selected_hyperparameters"]))
        )
        mlp_protocol = read_json(Path(str(dependencies["mlp_protocol"])))
        model, metrics, initial_hash = train_mlp(
            protocol=mlp_protocol,
            config=selected_hyperparameters["selected_by_dataset"][protocol["_active_dataset"]],
            data=data,
            split=split,
            num_features=num_features,
            num_classes=num_classes,
            seed=model_seed,
            device=device,
            include_test=True,
        )
        predictions = metrics.pop("test_predictions")
        labels = metrics.pop("test_labels")
        detail = {
            "accuracy": metrics["test_acc"],
            "macro_f1": metrics["test_macro_f1"],
            "balanced_accuracy": metrics["test_balanced_accuracy"],
            "per_class_recall": metrics["test_per_class_recall"],
        }
        result = {
            key: metrics[key]
            for key in ("val_acc", "test_acc", "best_epoch", "epochs_run")
        }
        structural["initial_state_hash"] = initial_hash
    else:
        raise ValueError(method)

    elapsed = time.perf_counter() - start
    result.update({
        "test_macro_f1": float(detail["macro_f1"]),
        "test_balanced_accuracy": float(detail["balanced_accuracy"]),
        "test_per_class_recall": detail["per_class_recall"],
        "end_to_end_seconds": elapsed,
        **structural,
    })
    if budget is not None and method != "MLP" and method != "Full":
        k = int(result["num_selected"])
        n = int(data.n_x)
        total = int(data.edge_index.size(1))
        if int(result["num_fixed_self_loops"]) != n:
            raise RuntimeError(f"{method} fixed self-loop check failed")
        if int(result["forward_incidences"]) != k + n:
            raise RuntimeError(f"{method} forward-count audit failed")
        result["structural_density"] = k / total
        result["effective_forward_density"] = (k + n) / (total + n)
    return model, result, labels, predictions, mask


def run_path(outdir: Path, method: str, dataset: str, split_seed: int, model_seed: int, budget):
    tag = method.lower().replace("-", "_")
    budget_tag = "full" if budget is None else f"r{int(round(100 * budget)):03d}"
    stem = f"eval_{tag}_{dataset}_split{split_seed:02d}_{budget_tag}_s{model_seed}"
    return outdir / "evaluation_runs" / f"{stem}.json", stem


def evaluate(args, protocol: Mapping[str, object], protocol_path: Path, device: torch.device):
    manifest = verify_manifest(protocol, protocol_path)
    outdir = Path(str(protocol["output_root"]))
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    methods = selected(
        args.methods,
        list(protocol["evaluation"]["sparse_methods"]) + list(protocol["evaluation"]["references"]),
        "methods",
    )
    split_seeds = [int(value) for value in selected(
        args.split_seeds, manifest["accepted_split_seeds"], "split seeds"
    )]
    model_seeds = [int(value) for value in selected(
        args.model_seeds, protocol["evaluation"]["model_seeds"], "model seeds"
    )]
    budgets = [float(value) for value in selected(
        args.budgets, protocol["evaluation"]["budgets"], "budgets"
    )]
    for dataset in datasets:
        protocol["_active_dataset"] = dataset
        for split_seed in split_seeds:
            data, num_features, num_classes, split, diagnostics = split_context(
                dataset, split_seed, protocol, manifest
            )
            for method in methods:
                method_budgets = budgets if method in protocol["evaluation"]["sparse_methods"] else [None]
                for budget in method_budgets:
                    for model_seed in model_seeds:
                        path, stem = run_path(
                            outdir, method, dataset, split_seed, model_seed, budget
                        )
                        prediction_path = outdir / "predictions" / f"{stem}.npz"
                        mask_path = outdir / "masks" / f"{stem}.npz"
                        needs_mask = method in protocol["evaluation"]["sparse_methods"]
                        complete = path.exists() and prediction_path.exists() and (
                            mask_path.exists() if needs_mask else True
                        )
                        if complete and not args.force:
                            print(
                                f"[split-robustness:skip] {dataset} split={split_seed} {method} "
                                f"rho={budget} s={model_seed}", flush=True
                            )
                            continue
                        print(
                            f"[split-robustness] {dataset} split={split_seed} {method} "
                            f"rho={budget} s={model_seed}", flush=True
                        )
                        model, metrics, labels, predictions, mask = train_one(
                            method=method,
                            budget=budget,
                            model_seed=model_seed,
                            data=data,
                            num_features=num_features,
                            num_classes=num_classes,
                            split=split,
                            protocol=protocol,
                            device=device,
                        )
                        save_predictions(prediction_path, labels, predictions)
                        record = {
                            "stage": "split_robustness_evaluation",
                            "created_at": timestamp(),
                            "protocol_hash": file_hash(protocol_path),
                            "split_manifest_hash": file_hash(outdir / "split_manifest.json"),
                            "dataset_id": dataset,
                            "dataset": protocol["display_names"][dataset],
                            "split_seed": split_seed,
                            "model_seed": model_seed,
                            "method": method,
                            "budget": budget if budget is not None else 1.0,
                            "prediction_path": str(prediction_path.resolve()),
                            "prediction_hash": file_hash(prediction_path),
                            **diagnostics,
                            **metrics,
                        }
                        if mask is not None:
                            save_mask(mask_path, mask)
                            record["mask_path"] = str(mask_path.resolve())
                            record["mask_hash"] = file_hash(mask_path)
                        write_json(path, record)
                        print(
                            f"[split-robustness:done] {dataset} split={split_seed} {method} "
                            f"rho={budget} s={model_seed} test={record['test_acc']:.2f}",
                            flush=True,
                        )
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            del data


def hierarchical_bootstrap_interval(
    seed_differences: np.ndarray,
    draws: int,
    seed: int,
) -> Tuple[float, float]:
    values = np.asarray(seed_differences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Expected split by model-seed differences")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    split_count, seed_count = values.shape
    for draw in range(draws):
        sampled_splits = rng.integers(0, split_count, size=split_count)
        sampled_values = []
        for split_index in sampled_splits:
            sampled_seeds = rng.integers(0, seed_count, size=seed_count)
            sampled_values.extend(values[split_index, sampled_seeds])
        estimates[draw] = np.mean(sampled_values)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def summarize(protocol: Mapping[str, object], protocol_path: Path) -> None:
    manifest = verify_manifest(protocol, protocol_path)
    outdir = Path(str(protocol["output_root"]))
    rows = [read_json(path) for path in sorted((outdir / "evaluation_runs").glob("eval_*.json"))]
    sparse = list(protocol["evaluation"]["sparse_methods"])
    references = list(protocol["evaluation"]["references"])
    splits = [int(value) for value in manifest["accepted_split_seeds"]]
    seeds = [int(value) for value in protocol["evaluation"]["model_seeds"]]
    budgets = [float(value) for value in protocol["evaluation"]["budgets"]]
    expected = set()
    for dataset in protocol["datasets"]:
        for split_seed in splits:
            for model_seed in seeds:
                for method in sparse:
                    for budget in budgets:
                        expected.add((dataset, split_seed, model_seed, method, budget))
                for method in references:
                    expected.add((dataset, split_seed, model_seed, method, 1.0))
    observed = {
        (str(row["dataset_id"]), int(row["split_seed"]), int(row["model_seed"]), str(row["method"]), float(row["budget"]))
        for row in rows
    }
    if expected != observed:
        raise RuntimeError(
            f"Incomplete split-robustness matrix: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    protocol_hash = file_hash(protocol_path)
    manifest_hash = file_hash(outdir / "split_manifest.json")
    if any(row["protocol_hash"] != protocol_hash for row in rows):
        raise RuntimeError("Split-robustness specification hash mismatch")
    if any(row["split_manifest_hash"] != manifest_hash for row in rows):
        raise RuntimeError("Split-manifest hash mismatch")
    lookup = {
        (row["dataset_id"], int(row["split_seed"]), int(row["model_seed"]), row["method"], float(row["budget"])): row
        for row in rows
    }
    method_summary: List[Dict[str, object]] = []
    for dataset in protocol["datasets"]:
        for budget in budgets:
            for method in sparse + references:
                source_budget = budget if method in sparse else 1.0
                group = [
                    lookup[(dataset, split_seed, model_seed, method, source_budget)]
                    for split_seed in splits for model_seed in seeds
                ]
                item = {
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "method": method,
                    "run_n": len(group),
                    "split_n": len(splits),
                }
                for metric in ("test_acc", "test_macro_f1", "test_balanced_accuracy", "end_to_end_seconds"):
                    item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(
                        [float(row[metric]) for row in group]
                    )
                method_summary.append(item)

    split_rows: List[Dict[str, object]] = []
    comparisons: List[Dict[str, object]] = []
    bootstrap_cfg = protocol["statistics"]["hierarchical_bootstrap"]
    contrasts = (
        ("EHGNN-F minus Random-Fixed", "Random-Fixed"),
        ("EHGNN-F minus Random-Resampled", "Random-Resampled"),
    )
    for dataset_index, dataset in enumerate(protocol["datasets"]):
        for budget_index, budget in enumerate(budgets):
            for contrast_index, (name, comparator) in enumerate(contrasts):
                seed_matrix = np.empty((len(splits), len(seeds)), dtype=np.float64)
                for split_index, split_seed in enumerate(splits):
                    for seed_index, model_seed in enumerate(seeds):
                        learned = float(lookup[(dataset, split_seed, model_seed, "EHGNN-F", budget)]["test_acc"])
                        baseline = float(lookup[(dataset, split_seed, model_seed, comparator, budget)]["test_acc"])
                        seed_matrix[split_index, seed_index] = learned - baseline
                    split_rows.append({
                        "dataset_id": dataset,
                        "dataset": protocol["display_names"][dataset],
                        "budget": budget,
                        "comparison": name,
                        "split_seed": split_seed,
                        "model_seed_0_difference": seed_matrix[split_index, 0],
                        "model_seed_1_difference": seed_matrix[split_index, 1],
                        "split_mean_difference": float(seed_matrix[split_index].mean()),
                    })
                split_values = seed_matrix.mean(axis=1)
                low, high = paired_ci(split_values)
                bootstrap_seed = (
                    int(bootstrap_cfg["seed"])
                    + 100 * dataset_index + 10 * budget_index + contrast_index
                )
                bootstrap_low, bootstrap_high = hierarchical_bootstrap_interval(
                    seed_matrix, int(bootstrap_cfg["draws"]), bootstrap_seed
                )
                comparisons.append({
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "comparison": name,
                    "split_n": len(splits),
                    "model_seeds_per_split": len(seeds),
                    "split_mean_difference": float(split_values.mean()),
                    "split_difference_ci95_low": low,
                    "split_difference_ci95_high": high,
                    "hierarchical_bootstrap_ci95_low": bootstrap_low,
                    "hierarchical_bootstrap_ci95_high": bootstrap_high,
                    "positive_splits": int(np.sum(split_values > 0)),
                    "tied_splits": int(np.sum(np.isclose(split_values, 0.0))),
                    "negative_splits": int(np.sum(split_values < 0)),
                    "wilcoxon_p_raw": wilcoxon_p(split_values),
                })
    adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["wilcoxon_p_holm_24"] = value

    write_csv(outdir / "evaluation_runs.csv", rows)
    write_csv(outdir / "method_summary.csv", method_summary)
    write_csv(outdir / "split_level_differences.csv", split_rows)
    write_csv(outdir / "paired_comparisons.csv", comparisons)
    method_lookup = {
        (row["dataset_id"], float(row["budget"]), row["method"]): row
        for row in method_summary
    }
    comparison_lookup = {
        (row["dataset_id"], float(row["budget"]), row["comparison"]): row
        for row in comparisons
    }
    lines = [
        "# Robustness across data splits",
        "",
        "Values aggregate two model seeds within each of 15 frozen data splits. Inferential comparisons use the data split as the unit.",
        "",
        "| Dataset | Budget | EHGNN-F | Random-Fixed | Random-Resampled | MLP | Full | EHGNN-F - Fixed (95% split CI; positive splits; Holm p) | EHGNN-F - Resampled (95% split CI; positive splits; Holm p) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in protocol["datasets"]:
        for budget in budgets:
            values = {
                method: method_lookup[(dataset, budget, method)]
                for method in sparse + references
            }
            fixed = comparison_lookup[(dataset, budget, "EHGNN-F minus Random-Fixed")]
            resampled = comparison_lookup[(dataset, budget, "EHGNN-F minus Random-Resampled")]
            lines.append(
                f"| {protocol['display_names'][dataset]} | {100*budget:.0f}% "
                f"| {values['EHGNN-F']['test_acc_mean']:.2f} "
                f"| {values['Random-Fixed']['test_acc_mean']:.2f} "
                f"| {values['Random-Resampled']['test_acc_mean']:.2f} "
                f"| {values['MLP']['test_acc_mean']:.2f} "
                f"| {values['Full']['test_acc_mean']:.2f} "
                f"| {fixed['split_mean_difference']:+.2f} [{fixed['split_difference_ci95_low']:+.2f}, {fixed['split_difference_ci95_high']:+.2f}]; {fixed['positive_splits']}/15; {fixed['wilcoxon_p_holm_24']:.4f} "
                f"| {resampled['split_mean_difference']:+.2f} [{resampled['split_difference_ci95_low']:+.2f}, {resampled['split_difference_ci95_high']:+.2f}]; {resampled['positive_splits']}/15; {resampled['wilcoxon_p_holm_24']:.4f} |"
            )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    output_names = (
        "evaluation_runs.csv",
        "method_summary.csv",
        "split_level_differences.csv",
        "paired_comparisons.csv",
        "summary.md",
    )
    write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_hash": protocol_hash,
        "split_manifest_hash": manifest_hash,
        "run_count": len(rows),
        "split_count": len(splits),
        "outputs": {name: file_hash(outdir / name) for name in output_names},
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit-splits", "evaluate", "summarize"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--split-seeds", nargs="+", type=int)
    parser.add_argument("--model-seeds", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    verify_dependencies(protocol_path, protocol)
    if args.stage == "audit-splits":
        audit_splits(protocol, protocol_path, args.force)
    elif args.stage == "evaluate":
        evaluate(args, protocol, protocol_path, torch.device(args.device))
    elif args.stage == "summarize":
        summarize(protocol, protocol_path)


if __name__ == "__main__":
    main()
