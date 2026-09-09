#!/usr/bin/env python3
"""Run the conditioned and low-rank scorer comparison in Table 15."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from node_classification_utils import (
    PRETTY_DATASET,
    fixed_split,
    prepare_dataset,
    train_learned_with_trajectory,
)
from main_accuracy import (
    atomic_save_mask,
    atomic_write_json,
    holm_adjust,
    mean_ci,
    read_json,
    timestamp,
    write_csv,
)


DEFAULT_PROTOCOL = Path("experiment_specs/conditioned_scorers.json")
DEFAULT_SELECTION = Path("experiment_specs/conditioned_scorer_selection.json")
DEFAULT_OUTDIR = Path(".work/conditioned_scorers")
CORE_RESULTS = Path(".work/main_accuracy")


VARIANTS = {
    "f_cond": {
        "name": "EHGNN-F(cond)",
        "mode": "learnmask_cond",
        "granularity": "incidence",
        "capacity_name": "hidden",
    },
    "f_cond_lr": {
        "name": "EHGNN-F(cond,LR)",
        "mode": "NeuralF",
        "granularity": "incidence",
        "capacity_name": "rank",
    },
}
PAPER_VARIANTS = ("f_cond", "f_cond_lr")


def load_protocol(path: Path) -> Tuple[Dict[str, object], str]:
    protocol = read_json(path)
    if protocol.get("status") != "fixed":
        raise RuntimeError("The conditioned-scorer specification is not fixed")
    return protocol, hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_context(dataset: str, protocol: Dict[str, object]):
    data, num_features, num_classes = prepare_dataset(dataset)
    settings = protocol["data"]
    split = fixed_split(
        data,
        int(settings["split_seed"]),
        float(settings["train_prop"]),
        float(settings["valid_prop"]),
    )
    return data, num_features, num_classes, split


def configs(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    out = []
    for variant, spec in protocol["variants"].items():
        for capacity in spec["capacity_grid"]:
            for lr_multiplier in protocol["training"]["mask_lr_multiplier_grid"]:
                out.append({
                    "variant": variant,
                    "config_index": sum(row["variant"] == variant for row in out),
                    "capacity": int(capacity),
                    "mask_lr_multiplier": float(lr_multiplier),
                })
    return out


def training_settings(protocol: Dict[str, object]) -> Dict[str, object]:
    settings = protocol["training"]
    return {
        "hidden": int(settings["hidden"]),
        "dropout": float(settings["dropout"]),
        "lr": float(settings["classifier_lr"]),
        "wd": float(settings["weight_decay"]),
        "epochs": int(settings["maximum_epochs"]),
        "patience": int(settings["patience"]),
        "sampling": str(settings["sampling"]).replace("_without_replacement", ""),
        "sparse_self_loop_policy": str(settings["sparse_self_loop_policy"]),
    }


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))


def initialize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(0, device=device)


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def raw_tensor_estimate_gib(variant: str, incidences: int, features: int) -> float:
    if variant != "f_cond":
        return 0.0
    return 2.0 * incidences * features * 4.0 / (1024.0 ** 3)


def selected_edge_index(data, hard_mask: torch.Tensor, granularity: str) -> torch.Tensor:
    if granularity != "incidence":
        raise ValueError(f"Unsupported mask granularity: {granularity}")
    hard = hard_mask.detach().cpu().bool()
    return data.edge_index[:, hard]


def structure_diagnostics(data, hard_mask: torch.Tensor, granularity: str) -> Dict[str, object]:
    selected = selected_edge_index(data, hard_mask, granularity)
    n_nodes = int(data.n_x)
    n_edges = int(data.num_hyperedges)
    n_incidences = int(data.edge_index.size(1))
    selected_units = int(hard_mask.sum())
    total_units = n_incidences
    retained_incidences = int(selected.size(1))
    node_coverage = (
        float(selected[0].unique().numel() / n_nodes) if retained_incidences else 0.0
    )
    edge_coverage = (
        float(selected[1].unique().numel() / n_edges) if retained_incidences else 0.0
    )
    return {
        "mask_unit": granularity,
        "num_mask_units": total_units,
        "num_selected_units": selected_units,
        "selected_unit_density": selected_units / total_units,
        "num_retained_original_incidences": retained_incidences,
        "realized_original_incidence_density": retained_incidences / n_incidences,
        "node_coverage": node_coverage,
        "active_hyperedge_coverage": edge_coverage,
        "num_fixed_self_loops": n_nodes,
        "effective_forward_incidences": retained_incidences + n_nodes,
        "effective_forward_density": (retained_incidences + n_nodes) / (n_incidences + n_nodes),
    }


def learned_run(
    *,
    data,
    split,
    num_features: int,
    num_classes: int,
    variant: str,
    capacity: int,
    lr_multiplier: float,
    seed: int,
    keep_ratio: float,
    include_test: bool,
    protocol: Dict[str, object],
    device: torch.device,
):
    spec = VARIANTS[variant]
    settings = training_settings(protocol)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, metrics, initial_mask, initial_prob, final_mask, final_prob, _ = (
        train_learned_with_trajectory(
            data=data,
            split_idx=split,
            mode=spec["mode"],
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=keep_ratio,
            seed=seed,
            device=device,
            trajectory_every=int(settings["epochs"]) + 1,
            mask_init_std=float(protocol["training"]["mask_init_std"]),
            mask_lr_multiplier=lr_multiplier,
            scorer_hidden_dim=capacity,
            low_rank=capacity,
            include_test_metrics=include_test,
            **settings,
        )
    )
    expected_units = int(
        keep_ratio
        * data.edge_index.size(1)
    )
    expected_units = max(1, expected_units)
    if int(final_mask.sum()) != expected_units:
        raise RuntimeError(
            f"Exact unit budget failure for {variant}: {int(final_mask.sum())} != {expected_units}"
        )
    diagnostics = structure_diagnostics(data, final_mask, spec["granularity"])
    if int(model.last_forward_incidence_count) != diagnostics["effective_forward_incidences"]:
        raise RuntimeError("Fixed self-loop forward count mismatch")
    extra = {
        **diagnostics,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "mask_parameter_count": sum(
            parameter.numel() for parameter in model.mask_module.parameters()
        ),
        "peak_allocated_memory_mb": peak_memory_mb(device),
        "initial_final_jaccard": float(
            (initial_mask & final_mask).sum().item()
            / max(1, (initial_mask | final_mask).sum().item())
        ),
        "mean_abs_probability_change": float((final_prob - initial_prob).abs().mean()),
        "final_probability_mean": float(final_prob.mean()),
        "final_probability_std": float(final_prob.std(unbiased=False)),
    }
    return model, metrics, final_mask, extra


def result_path(outdir: Path, stage: str, variant: str, config_index: int,
                dataset: str, seed: int) -> Path:
    return outdir / f"{stage}_runs" / (
        f"{stage}_{variant}_c{config_index}_{dataset}_s{seed}.json"
    )


def tune(args, protocol: Dict[str, object], study_hash: str, device: torch.device) -> None:
    all_configs = configs(protocol)
    selected_variants = set(args.variants or PAPER_VARIANTS)
    selected_datasets = list(args.datasets or protocol["datasets"])
    selected_seeds = list(args.seeds or protocol["tuning"]["seeds"])
    chosen_indices = set(args.config_indices) if args.config_indices else None
    run_dir = Path(args.outdir) / "tune_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for config in all_configs:
        variant = str(config["variant"])
        if variant not in selected_variants:
            continue
        if chosen_indices is not None and int(config["config_index"]) not in chosen_indices:
            continue
        for dataset in selected_datasets:
            data, num_features, num_classes, split = dataset_context(dataset, protocol)
            for seed_value in selected_seeds:
                seed = int(seed_value)
                path = result_path(
                    Path(args.outdir), "tune", variant,
                    int(config["config_index"]), dataset, seed,
                )
                if path.exists() and not args.force:
                    print(f"[tune:skip] {variant} c={config['config_index']} {dataset} s={seed}", flush=True)
                    continue
                print(f"[tune] {variant} c={config['config_index']} {dataset} s={seed}", flush=True)
                common = {
                    "stage": "validation_only_tuning",
                    "created_at": timestamp(),
                    "study_hash": study_hash,
                    "variant": variant,
                    "method": VARIANTS[variant]["name"],
                    "mode": VARIANTS[variant]["mode"],
                    "granularity": VARIANTS[variant]["granularity"],
                    "config_index": int(config["config_index"]),
                    "capacity": int(config["capacity"]),
                    "capacity_name": VARIANTS[variant]["capacity_name"],
                    "mask_lr_multiplier": float(config["mask_lr_multiplier"]),
                    "dataset_id": dataset,
                    "dataset": PRETTY_DATASET.get(dataset, dataset),
                    "seed": seed,
                    "keep_ratio": float(protocol["tuning"]["keep_ratio"]),
                    "num_nodes": int(data.n_x),
                    "num_hyperedges": int(data.num_hyperedges),
                    "num_incidences": int(data.edge_index.size(1)),
                    "num_features": int(num_features),
                    "estimated_raw_conditioning_tensor_gib": raw_tensor_estimate_gib(
                        variant, int(data.edge_index.size(1)), int(num_features)
                    ),
                }
                try:
                    model, metrics, _, diagnostics = learned_run(
                        data=data,
                        split=split,
                        num_features=num_features,
                        num_classes=num_classes,
                        variant=variant,
                        capacity=int(config["capacity"]),
                        lr_multiplier=float(config["mask_lr_multiplier"]),
                        seed=seed,
                        keep_ratio=float(protocol["tuning"]["keep_ratio"]),
                        include_test=False,
                        protocol=protocol,
                        device=device,
                    )
                    if any("test" in key.lower() for key in metrics):
                        raise RuntimeError("Validation tuning exposed a test metric")
                    record = {
                        **common,
                        "status": "ok",
                        "train_acc": 100.0 * float(metrics["train_acc"]),
                        "val_acc": 100.0 * float(metrics["val_acc"]),
                        "train_loss": float(metrics["train_loss"]),
                        "val_loss": float(metrics["val_loss"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_run": int(metrics["epochs_run"]),
                        **diagnostics,
                    }
                    del model
                except BaseException as error:
                    if not is_oom(error):
                        raise
                    record = {
                        **common,
                        "status": "oom",
                        "error_type": type(error).__name__,
                        "error_summary": str(error).splitlines()[0][:1000],
                        "peak_allocated_memory_mb": peak_memory_mb(device),
                    }
                    print(f"[tune:oom] {variant} {dataset} c={config['config_index']} s={seed}", flush=True)
                atomic_write_json(path, record)
                clear_memory(device)


def collect_records(directory: Path, pattern: str) -> List[Dict[str, object]]:
    return [read_json(path) for path in sorted(directory.rglob(pattern))]


def select(args, protocol: Dict[str, object], study_hash: str) -> None:
    selected_variants = list(args.variants or PAPER_VARIANTS)
    all_configs = [
        row for row in configs(protocol) if row["variant"] in selected_variants
    ]
    records = collect_records(Path(args.outdir) / "tune_runs", "tune_*.json")
    expected = {
        (row["variant"], row["config_index"], dataset, int(seed))
        for row in all_configs
        for dataset in protocol["datasets"]
        for seed in protocol["tuning"]["seeds"]
    }
    lookup = {
        (row["variant"], int(row["config_index"]), row["dataset_id"], int(row["seed"])): row
        for row in records
    }
    missing = expected - set(lookup)
    if missing:
        raise RuntimeError(f"Incomplete validation grid: {len(missing)} cells missing")
    if any(row["study_hash"] != study_hash for row in records):
        raise RuntimeError("Tuning study hash mismatch")
    if any(
        any("test" in key.lower() for key in row)
        for row in records
    ):
        raise RuntimeError("A tuning record contains a test field")

    summaries = []
    frozen_variants = {}
    for variant in selected_variants:
        variant_summaries = []
        for config in [row for row in all_configs if row["variant"] == variant]:
            cells = [
                lookup[(variant, int(config["config_index"]), dataset, int(seed))]
                for dataset in protocol["datasets"]
                for seed in protocol["tuning"]["seeds"]
            ]
            feasible = all(cell["status"] == "ok" for cell in cells)
            row = {
                **config,
                "feasible_all_cells": feasible,
                "ok_cells": sum(cell["status"] == "ok" for cell in cells),
                "oom_cells": sum(cell["status"] == "oom" for cell in cells),
            }
            if feasible:
                dataset_means = {
                    dataset: float(np.mean([
                        cell["val_acc"] for cell in cells if cell["dataset_id"] == dataset
                    ]))
                    for dataset in protocol["datasets"]
                }
                row["macro_val_acc"] = float(np.mean(list(dataset_means.values())))
                row.update({f"{dataset}_val_acc": value for dataset, value in dataset_means.items()})
                variant_summaries.append(row)
            summaries.append(row)
        if not variant_summaries:
            frozen_variants[variant] = {
                "status": "no_globally_feasible_configuration",
                "reason": "At least one validation-only cell OOMed for every configuration.",
            }
            continue
        best = max(float(row["macro_val_acc"]) for row in variant_summaries)
        tolerance = float(protocol["tuning"]["tie_tolerance_accuracy_points"])
        tied = [row for row in variant_summaries if float(row["macro_val_acc"]) >= best - tolerance]
        winner = min(
            tied,
            key=lambda row: (
                int(row["capacity"]),
                float(row["mask_lr_multiplier"]),
                int(row["config_index"]),
            ),
        )
        frozen_variants[variant] = {
            "status": "selected",
            "config_index": int(winner["config_index"]),
            "capacity": int(winner["capacity"]),
            "mask_lr_multiplier": float(winner["mask_lr_multiplier"]),
            "macro_val_acc": float(winner["macro_val_acc"]),
        }
    frozen = {
        "status": "frozen",
        "frozen_at": timestamp(),
        "study_hash": study_hash,
        "selection_scope": "one_global_configuration_per_variant",
        "variants": frozen_variants,
    }
    write_csv(Path(args.outdir) / "validation_grid_runs.csv", records)
    write_csv(Path(args.outdir) / "validation_grid_summary.csv", summaries)
    atomic_write_json(Path(args.outdir) / "frozen_config.json", frozen)
    print(json.dumps(frozen_variants, indent=2), flush=True)


def verify_frozen(path: Path, study_hash: str) -> Dict[str, object]:
    frozen = read_json(path)
    if frozen.get("status") not in {"fixed", "frozen"} or frozen.get("study_hash") != study_hash:
        raise RuntimeError("Evaluation requires a matching selected configuration")
    return frozen


def evaluate(args, protocol: Dict[str, object], study_hash: str, device: torch.device) -> None:
    frozen_path = Path(args.frozen_config or DEFAULT_SELECTION)
    frozen = verify_frozen(frozen_path, study_hash)
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    methods = list(args.variants or PAPER_VARIANTS)
    datasets = list(args.datasets or protocol["datasets"])
    seeds = list(args.seeds or protocol["evaluation"]["seeds"])
    keep_ratio = float(protocol["evaluation"]["keep_ratio"])
    run_dir = Path(args.outdir) / "eval_runs"
    mask_dir = Path(args.outdir) / "masks"
    run_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        data, num_features, num_classes, split = dataset_context(dataset, protocol)
        for seed_value in seeds:
            seed = int(seed_value)
            for method in methods:
                selected = frozen["variants"][method]
                if selected["status"] != "selected":
                    print(f"[evaluate:infeasible] {method} {dataset} s={seed}", flush=True)
                    continue
                config_index = int(selected["config_index"])
                path = result_path(
                    Path(args.outdir), "eval", method, config_index, dataset, seed
                )
                if path.exists() and not args.force:
                    print(f"[evaluate:skip] {method} {dataset} s={seed}", flush=True)
                    continue
                print(f"[evaluate] {method} {dataset} s={seed}", flush=True)
                model, metrics, hard, diagnostics = learned_run(
                    data=data,
                    split=split,
                    num_features=num_features,
                    num_classes=num_classes,
                    variant=method,
                    capacity=int(selected["capacity"]),
                    lr_multiplier=float(selected["mask_lr_multiplier"]),
                    seed=seed,
                    keep_ratio=keep_ratio,
                    include_test=True,
                    protocol=protocol,
                    device=device,
                )
                variant_name = VARIANTS[method]["name"]
                capacity = int(selected["capacity"])
                lr_multiplier = float(selected["mask_lr_multiplier"])
                mask_path = mask_dir / f"mask_{method}_{dataset}_s{seed}.npz"
                atomic_save_mask(mask_path, hard)
                atomic_write_json(path, {
                    "stage": "frozen_test_evaluation",
                    "created_at": timestamp(),
                    "study_hash": study_hash,
                    "frozen_config_hash": frozen_hash,
                    "variant": method,
                    "method": variant_name,
                    "dataset_id": dataset,
                    "dataset": PRETTY_DATASET.get(dataset, dataset),
                    "seed": seed,
                    "keep_ratio": keep_ratio,
                    "capacity": capacity,
                    "mask_lr_multiplier": lr_multiplier,
                    "val_acc": 100.0 * float(metrics["val_acc"]),
                    "test_acc": 100.0 * float(metrics["test_acc"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "epochs_run": int(metrics["epochs_run"]),
                    "mask_path": str(mask_path.resolve()),
                    **diagnostics,
                })
                del model
                clear_memory(device)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def paired_test(left: Sequence[float], right: Sequence[float]) -> Dict[str, object]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    low, high = mean_ci(differences)
    if np.allclose(differences, 0.0):
        wilcoxon = 1.0
    else:
        wilcoxon = float(stats.wilcoxon(differences).pvalue)
    return {
        "n": int(differences.size),
        "paired_delta_mean": float(differences.mean()),
        "paired_delta_std": float(differences.std(ddof=1)),
        "paired_delta_ci95_low": low,
        "paired_delta_ci95_high": high,
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "wilcoxon_p": wilcoxon,
        "paired_t_p": float(stats.ttest_rel(left, right).pvalue),
    }


def control_rows(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for dataset in protocol["datasets"]:
        for seed in protocol["evaluation"]["seeds"]:
            for prefix, method in (("ehgnnf", "EHGNN-F"), ("random", "Random-Fixed")):
                source = CORE_RESULTS / "evaluation_runs" / f"eval_{prefix}_{dataset}_r050_s{seed}.json"
                record = read_json(source)
                rows.append({
                    "dataset_id": dataset,
                    "seed": int(seed),
                    "method": method,
                    "test_acc": float(record["test_acc"]),
                    "selected_unit_density": float(record["keep_ratio"]),
                    "realized_original_incidence_density": float(record["keep_ratio"]),
                    "effective_forward_density": float(record["effective_forward_density"]),
                    "node_coverage": float("nan"),
                    "peak_allocated_memory_mb": float("nan"),
                })
    return rows


def summarize(args, protocol: Dict[str, object], study_hash: str) -> None:
    frozen_path = Path(args.frozen_config or DEFAULT_SELECTION)
    frozen = verify_frozen(frozen_path, study_hash)
    requested_variants = set(args.variants or PAPER_VARIANTS)
    new_rows = [
        row
        for row in collect_records(Path(args.outdir) / "eval_runs", "eval_*.json")
        if row["variant"] in requested_variants
    ]
    expected_methods = [
        variant for variant, value in frozen["variants"].items()
        if variant in requested_variants and value["status"] == "selected"
    ]
    expected = {
        (method, dataset, int(seed))
        for method in expected_methods
        for dataset in protocol["datasets"]
        for seed in protocol["evaluation"]["seeds"]
    }
    observed = {
        (row["variant"], row["dataset_id"], int(row["seed"])) for row in new_rows
    }
    missing = expected - observed
    if missing:
        raise RuntimeError(f"Incomplete test evaluation: {len(missing)} cells missing")
    if any(row["study_hash"] != study_hash for row in new_rows):
        raise RuntimeError("Evaluation study hash mismatch")
    rows = [*new_rows, *control_rows(protocol)]
    write_csv(Path(args.outdir) / "all_method_seed_results.csv", rows)

    summaries = []
    methods = sorted(set(row["method"] for row in rows))
    for dataset in protocol["datasets"]:
        for method in methods:
            group = [row for row in rows if row["dataset_id"] == dataset and row["method"] == method]
            if not group:
                continue
            mean, std = mean_std([row["test_acc"] for row in group])
            summary = {
                "dataset_id": dataset,
                "dataset": PRETTY_DATASET.get(dataset, dataset),
                "method": method,
                "n": len(group),
                "test_acc_mean": mean,
                "test_acc_std": std,
            }
            for metric in (
                "selected_unit_density",
                "realized_original_incidence_density",
                "effective_forward_density",
                "node_coverage",
                "peak_allocated_memory_mb",
                "model_parameter_count",
                "mask_parameter_count",
            ):
                finite = [float(row[metric]) for row in group if np.isfinite(float(row.get(metric, np.nan)))]
                if finite:
                    summary[f"{metric}_mean"] = float(np.mean(finite))
            summaries.append(summary)
    write_csv(Path(args.outdir) / "method_summary.csv", summaries)

    name_by_variant = {variant: spec["name"] for variant, spec in VARIANTS.items()}
    comparisons = []
    learned_variants = [variant for variant in expected_methods if variant in VARIANTS]
    for variant in learned_variants:
        method = name_by_variant[variant]
        comparators = ["EHGNN-F", "Random-Fixed"]
        for comparator in comparators:
            for dataset in protocol["datasets"]:
                left = sorted(
                    [row for row in rows if row["dataset_id"] == dataset and row["method"] == method],
                    key=lambda row: int(row["seed"]),
                )
                right = sorted(
                    [row for row in rows if row["dataset_id"] == dataset and row["method"] == comparator],
                    key=lambda row: int(row["seed"]),
                )
                comparisons.append({
                    "variant": variant,
                    "method": method,
                    "comparator": comparator,
                    "dataset_id": dataset,
                    **paired_test(
                        [row["test_acc"] for row in left],
                        [row["test_acc"] for row in right],
                    ),
                })
    for variant in learned_variants:
        method = name_by_variant[variant]
        comparators = sorted({
            row["comparator"] for row in comparisons if row["method"] == method
        })
        for comparator in comparators:
            family = [
                row for row in comparisons
                if row["method"] == method and row["comparator"] == comparator
            ]
            adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in family])
            for row, value in zip(family, adjusted):
                row["wilcoxon_p_holm_4"] = value
    write_csv(Path(args.outdir) / "paired_comparisons.csv", comparisons)

    across_dataset = []
    families = sorted({
        (row["method"], row["comparator"]) for row in comparisons
    })
    for method, comparator in families:
        family = sorted(
            [
                row for row in comparisons
                if row["method"] == method and row["comparator"] == comparator
            ],
            key=lambda row: row["dataset_id"],
        )
        deltas = np.asarray([row["paired_delta_mean"] for row in family], dtype=float)
        low, high = mean_ci(deltas)
        across_dataset.append({
            "method": method,
            "comparator": comparator,
            "n_datasets": len(deltas),
            "equal_dataset_delta_mean": float(deltas.mean()),
            "equal_dataset_delta_ci95_low": low,
            "equal_dataset_delta_ci95_high": high,
            "positive_datasets": int((deltas > 0).sum()),
            "negative_datasets": int((deltas < 0).sum()),
            "wilcoxon_p_over_dataset_means": (
                1.0 if np.allclose(deltas, 0.0)
                else float(stats.wilcoxon(deltas).pvalue)
            ),
        })
    write_csv(Path(args.outdir) / "across_dataset_comparisons.csv", across_dataset)

    feasibility = []
    validation_rows = collect_records(Path(args.outdir) / "tune_runs", "tune_*.json")
    for variant in sorted(requested_variants):
        cells = [row for row in validation_rows if row["variant"] == variant]
        feasibility.append({
            "variant": variant,
            "method": VARIANTS[variant]["name"],
            "selection_status": frozen["variants"][variant]["status"],
            "ok_cells": sum(row["status"] == "ok" for row in cells),
            "oom_cells": sum(row["status"] == "oom" for row in cells),
            "datasets_with_oom": ",".join(sorted({
                row["dataset_id"] for row in cells if row["status"] == "oom"
            })),
            "maximum_peak_allocated_memory_mb": max(
                [float(row.get("peak_allocated_memory_mb", 0.0)) for row in cells],
                default=0.0,
            ),
        })
    write_csv(Path(args.outdir) / "feasibility_summary.csv", feasibility)
    atomic_write_json(Path(args.outdir) / "summary_manifest.json", {
        "created_at": timestamp(),
        "study_hash": study_hash,
        "datasets": protocol["datasets"],
        "test_seeds": protocol["evaluation"]["seeds"],
        "selected_variants": expected_methods,
        "files": [
            "validation_grid_summary.csv",
            "feasibility_summary.csv",
            "method_summary.csv",
            "paired_comparisons.csv",
            "across_dataset_comparisons.csv",
            "all_method_seed_results.csv",
        ],
    })
    print("[summarize] complete", flush=True)


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("stage", choices=["tune", "select", "evaluate", "summarize"])
    out.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    out.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    out.add_argument("--frozen-config")
    out.add_argument("--device", default="cuda:0")
    out.add_argument("--datasets", nargs="+")
    out.add_argument("--variants", nargs="+")
    out.add_argument("--seeds", nargs="+", type=int)
    out.add_argument("--config-indices", nargs="+", type=int)
    out.add_argument("--force", action="store_true")
    return out


def main() -> None:
    args = parser().parse_args()
    protocol_path = Path(args.protocol)
    protocol, study_hash = load_protocol(protocol_path)
    unknown_datasets = set(args.datasets or []) - set(protocol["datasets"])
    if unknown_datasets:
        raise ValueError(f"Datasets outside protocol: {sorted(unknown_datasets)}")
    unknown_variants = set(args.variants or []) - set(VARIANTS)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    device = torch.device(args.device)
    initialize_device(device)
    if args.stage == "tune":
        tune(args, protocol, study_hash, device)
    elif args.stage == "select":
        select(args, protocol, study_hash)
    elif args.stage == "evaluate":
        evaluate(args, protocol, study_hash, device)
    else:
        summarize(args, protocol, study_hash)


if __name__ == "__main__":
    main()
