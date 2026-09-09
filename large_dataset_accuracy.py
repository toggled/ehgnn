#!/usr/bin/env python3
"""Run the DBLP-CA and Walmart portion of the five-method Table 2.

Stages are deliberately resumable so independent seed partitions can run on
separate GPUs without sharing mutable state:

* prepare: construct deterministic Degree/Spectral exact-budget masks;
* evaluate-core: run Full, EHGNN-F, and Random-Fixed;
* evaluate-structural: train Cardinality and Laplacian-proxy masks;
* summarize: require and audit the complete 340-run matrix.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from structural_baselines import (
    edge_scores,
    prefix_mask,
    structural_diagnostics,
)
from node_classification_utils import (
    fixed_split,
    jaccard,
    train_learned_with_trajectory,
)
from training_utils import (
    load_v2e_dataset,
    train_model,
    unit_scores_and_mask_for_baseline,
)
from main_accuracy import (
    atomic_save_mask,
    atomic_write_json,
    forward_diagnostics,
    holm_adjust,
    load_mask,
    mean_ci,
    mean_std,
    protocol_hash,
    read_json,
    timestamp,
    write_csv,
)
from walmart_feature_check import feature_diagnostics, load_walmart


DEFAULT_PROTOCOL = Path("experiment_specs/large_dataset_accuracy.json")
DEFAULT_OUTDIR = Path(".work/large_dataset_accuracy")
DEFAULT_FROZEN = Path("experiment_specs/main_selected_hyperparameters.json")

DISPLAY = {
    "coauthor_dblp": "DBLP-CA",
    "walmart-trips": "Walmart",
}
CORE_METHODS = ("Full", "EHGNN-F", "Random-Fixed")
STRUCTURAL_METHODS = ("Degree-prefix", "Spectral-prefix")
RECORD_METHOD = {
    "Full": "Full",
    "EHGNN-F": "EHGNN-F",
    "Random-Fixed": "Random-Fixed",
    "Degree-prefix": "Degree-prefix",
    "Spectral-prefix": "Spectral-prefix",
}
METHOD_TAG = {
    "Full": "full",
    "EHGNN-F": "ehgnnf",
    "Random-Fixed": "random",
    "Degree-prefix": "degree_prefix",
    "Spectral-prefix": "spectral_prefix",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fixed_settings(protocol: Mapping[str, object]) -> Mapping[str, object]:
    return protocol["fixed_budget_accuracy_evaluation"]


def train_kwargs(protocol: Mapping[str, object]) -> Dict[str, object]:
    cfg = fixed_settings(protocol)["training"]
    return {
        "hidden": int(cfg["hidden"]),
        "dropout": float(cfg["dropout"]),
        "lr": float(cfg["lr"]),
        "wd": float(cfg["weight_decay"]),
        "epochs": int(cfg["epochs"]),
        "patience": int(cfg["patience"]),
        "sampling": str(cfg["sampling"]),
        "sparse_self_loop_policy": str(cfg["sparse_self_loop_policy"]),
    }


def verify_prerequisites(
    protocol_path: Path,
    protocol: Mapping[str, object],
    frozen_path: Path,
) -> Dict[str, object]:
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    if protocol.get("status") != "fixed":
        raise RuntimeError("The experiment specification is not fixed")
    feature_hash = str(protocol["walmart_validation_only_audit"]["feature_hash"])

    frozen = read_json(frozen_path)
    if frozen.get("status") != "fixed":
        raise RuntimeError("The EHGNN-F configuration is not fixed")
    requested = fixed_settings(protocol)["frozen_ehgnnf_configuration"]
    selected = frozen["selected"]
    for key in ("mask_init_std", "mask_lr_multiplier"):
        if not math.isclose(float(requested[key]), float(selected[key]), abs_tol=0.0):
            raise RuntimeError(f"Frozen EHGNN-F mismatch for {key}")

    parent_path = Path(str(protocol["parent_accuracy_protocol"]))
    parent = read_json(parent_path)
    if frozen.get("protocol_hash") != protocol_hash(parent_path):
        raise RuntimeError("Frozen configuration does not match its parent protocol")
    if str(parent["training"]["sparse_self_loop_policy"]) != "fixed_self_loops_implicit":
        raise RuntimeError("The main experiment does not use the required fixed self-loops")

    baseline_path = Path(str(protocol["parent_structural_baseline_protocol"]))
    baseline = read_json(baseline_path)
    if baseline.get("status") != "fixed":
        raise RuntimeError("The structural-baseline specification is not fixed")
    return {
        "audit": {"feature_hash": feature_hash},
        "frozen": frozen,
        "parent": parent,
        "baseline": baseline,
        "protocol_hash": protocol_hash(protocol_path),
        "frozen_hash": sha256(frozen_path),
        "baseline_protocol_hash": protocol_hash(baseline_path),
    }


def dataset_context(dataset: str, protocol: Mapping[str, object], expected_feature_hash: str):
    settings = fixed_settings(protocol)
    if dataset == "walmart-trips":
        audit_cfg = protocol["walmart_validation_only_audit"]
        data, num_features, num_classes = load_walmart(
            str(audit_cfg["feature_noise"]),
            int(audit_cfg["feature_seed"]),
            int(audit_cfg["feature_dimension"]),
        )
        observed_hash = tensor_hash(data.x)
        if observed_hash != expected_feature_hash:
            raise RuntimeError(
                f"Deterministic Walmart feature hash changed: {observed_hash}"
            )
    elif dataset == "coauthor_dblp":
        data, num_features, num_classes = load_v2e_dataset(dataset)
        observed_hash = tensor_hash(data.x)
    else:
        raise ValueError(f"Dataset outside frozen protocol: {dataset}")
    split = fixed_split(
        data,
        int(settings["split_seed"]),
        float(settings["train_prop"]),
        float(settings["valid_prop"]),
    )
    diagnostics = {
        "feature_hash": observed_hash,
        "label_hash": tensor_hash(data.y),
        "edge_index_hash": tensor_hash(data.edge_index),
        "num_nodes": int(data.n_x),
        "num_hyperedges": int(data.num_hyperedges),
        "num_original_incidences": int(data.edge_index.size(1)),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
    }
    return data, num_features, num_classes, split, diagnostics


def requested(values: Sequence | None, allowed: Sequence, label: str) -> List:
    result = list(allowed if values is None else values)
    if not set(result) <= set(allowed):
        raise ValueError(f"Requested {label} are outside the frozen protocol")
    return result


def record_path(outdir: Path, method: str, dataset: str, seed: int, ratio=None) -> Path:
    if method == "Full":
        name = f"eval_full_{dataset}_s{seed}.json"
    else:
        ratio_tag = f"{int(round(100 * float(ratio))):03d}"
        name = f"eval_{METHOD_TAG[method]}_{dataset}_r{ratio_tag}_s{seed}.json"
    return outdir / "evaluation_runs" / name


def mask_path(outdir: Path, method: str, dataset: str, ratio: float, seed=None) -> Path:
    ratio_tag = f"{int(round(100 * ratio)):03d}"
    suffix = "" if seed is None else f"_s{seed}"
    return outdir / "masks" / f"mask_{METHOD_TAG[method]}_{dataset}_r{ratio_tag}{suffix}.npz"


def common_record(
    *,
    args,
    prereq: Mapping[str, object],
    dataset: str,
    seed: int,
    method: str,
    ratio: float,
    metrics: Mapping[str, float],
    selected: int,
    units: int,
    model,
    data,
    diagnostics: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "stage": "large_dataset_test_evaluation",
        "created_at": timestamp(),
        "protocol_hash": prereq["protocol_hash"],
        "frozen_config_hash": prereq["frozen_hash"],
        "dataset_id": dataset,
        "dataset": DISPLAY[dataset],
        "seed": int(seed),
        "method": RECORD_METHOD[method],
        "unit": "incidence",
        "keep_ratio": float(ratio),
        "sparse_self_loop_policy": "fixed_self_loops_implicit",
        "val_acc": 100.0 * float(metrics["val_acc"]),
        "test_acc": 100.0 * float(metrics["test_acc"]),
        "best_epoch": int(metrics["best_epoch"]),
        "epochs_run": int(metrics["epochs_run"]),
        "num_selected": int(selected),
        "num_mask_units": int(units),
        "structural_density": float(selected / units),
        **diagnostics,
        **forward_diagnostics(model, data, int(selected)),
    }


def prepare(args, protocol, prereq, device: torch.device) -> None:
    outdir = Path(args.outdir)
    settings = fixed_settings(protocol)
    datasets = requested(args.datasets, settings["datasets"], "datasets")
    methods = requested(args.methods, STRUCTURAL_METHODS, "methods")
    budgets = [float(value) for value in settings["budgets"]]
    baseline_protocol = prereq["baseline"]
    expected_feature_hash = str(prereq["audit"]["feature_hash"])
    for dataset in datasets:
        data, _, _, _, diagnostics = dataset_context(
            dataset, protocol, expected_feature_hash
        )
        for method in methods:
            score_file = outdir / "scores" / f"scores_{METHOD_TAG[method]}_{dataset}.npz"
            if score_file.exists() and not args.force:
                with np.load(score_file) as archive:
                    scores = torch.from_numpy(archive["scores"])
                print(f"[prepare:score-skip] {method} dataset={dataset}", flush=True)
            else:
                print(f"[prepare:score] {method} dataset={dataset}", flush=True)
                scores = edge_scores(data, method, baseline_protocol, device)
                score_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = score_file.with_name(score_file.name + ".tmp.npz")
                np.savez_compressed(temporary, scores=scores.numpy())
                temporary.replace(score_file)
            if scores.numel() != int(data.num_hyperedges) or not torch.isfinite(scores).all():
                raise RuntimeError(f"Invalid {method} scores for {dataset}")
            for ratio in budgets:
                saved_mask = mask_path(outdir, method, dataset, ratio)
                metadata_path = outdir / "mask_metadata" / (
                    f"mask_{METHOD_TAG[method]}_{dataset}_r{int(round(100 * ratio)):03d}.json"
                )
                if saved_mask.exists() and metadata_path.exists() and not args.force:
                    mask = load_mask(saved_mask)
                    print(f"[prepare:mask-skip] {method} dataset={dataset} rho={ratio}", flush=True)
                else:
                    mask, details = prefix_mask(data, scores, ratio)
                    atomic_save_mask(saved_mask, mask)
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(metadata_path, {
                        "created_at": timestamp(),
                        "protocol_hash": prereq["protocol_hash"],
                        "baseline_protocol_hash": prereq["baseline_protocol_hash"],
                        "dataset_id": dataset,
                        "dataset": DISPLAY[dataset],
                        "method": method,
                        "keep_ratio": ratio,
                        "num_selected": int(mask.sum()),
                        "num_mask_units": int(mask.numel()),
                        "structural_density": float(mask.float().mean()),
                        **diagnostics,
                        **details,
                        **structural_diagnostics(data, mask),
                    })
                    print(
                        f"[prepare:mask] {method} dataset={dataset} rho={ratio} "
                        f"K={int(mask.sum())}", flush=True
                    )
                expected = max(1, int(ratio * data.edge_index.size(1)))
                if mask.numel() != data.edge_index.size(1) or int(mask.sum()) != expected:
                    raise RuntimeError(f"Stored exact budget failed for {method}/{dataset}/{ratio}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate_core(args, protocol, prereq, device: torch.device) -> None:
    outdir = Path(args.outdir)
    (outdir / "evaluation_runs").mkdir(parents=True, exist_ok=True)
    settings = fixed_settings(protocol)
    datasets = requested(args.datasets, settings["datasets"], "datasets")
    seeds = [int(value) for value in requested(args.seeds, settings["model_seeds"], "seeds")]
    methods = requested(args.methods, CORE_METHODS, "methods")
    budgets = [float(value) for value in settings["budgets"]]
    cfg = train_kwargs(protocol)
    selected_cfg = settings["frozen_ehgnnf_configuration"]
    random_offset = int(prereq["parent"]["evaluation"]["random_mask_seed_offset"])
    expected_feature_hash = str(prereq["audit"]["feature_hash"])
    for dataset in datasets:
        data, num_features, num_classes, split, diagnostics = dataset_context(
            dataset, protocol, expected_feature_hash
        )
        total = int(data.edge_index.size(1))
        for seed in seeds:
            if "Full" in methods:
                path = record_path(outdir, "Full", dataset, seed)
                if path.exists() and not args.force:
                    print(f"[core:skip] Full dataset={dataset} seed={seed}", flush=True)
                else:
                    print(f"[core] Full dataset={dataset} seed={seed}", flush=True)
                    model, metrics = train_model(
                        data=data, split_idx=split, mode="full",
                        num_features=num_features, num_classes=num_classes,
                        keep_ratio=1.0, seed=seed, device=device, **cfg,
                    )
                    record = common_record(
                        args=args, prereq=prereq, dataset=dataset, seed=seed,
                        method="Full", ratio=1.0, metrics=metrics, selected=total,
                        units=total, model=model, data=data, diagnostics=diagnostics,
                    )
                    atomic_write_json(path, record)
                    print(f"[core:done] Full dataset={dataset} seed={seed} test={record['test_acc']:.2f}", flush=True)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            for ratio in budgets:
                expected = max(1, int(ratio * total))
                if "EHGNN-F" in methods:
                    path = record_path(outdir, "EHGNN-F", dataset, seed, ratio)
                    saved_mask = mask_path(outdir, "EHGNN-F", dataset, ratio, seed)
                    if path.exists() and saved_mask.exists() and not args.force:
                        print(f"[core:skip] EHGNN-F dataset={dataset} rho={ratio} seed={seed}", flush=True)
                    else:
                        print(f"[core] EHGNN-F dataset={dataset} rho={ratio} seed={seed}", flush=True)
                        model, metrics, init_mask, init_probs, final_mask, final_probs, _ = (
                            train_learned_with_trajectory(
                                data=data, split_idx=split, mode="learnmask",
                                num_features=num_features, num_classes=num_classes,
                                keep_ratio=ratio, seed=seed, device=device,
                                trajectory_every=int(cfg["epochs"]) + 1,
                                mask_init_std=float(selected_cfg["mask_init_std"]),
                                mask_lr_multiplier=float(selected_cfg["mask_lr_multiplier"]),
                                include_test_metrics=True, **cfg,
                            )
                        )
                        if final_mask.numel() != total or int(final_mask.sum()) != expected:
                            raise RuntimeError("EHGNN-F exact hard budget failed")
                        atomic_save_mask(saved_mask, final_mask)
                        record = common_record(
                            args=args, prereq=prereq, dataset=dataset, seed=seed,
                            method="EHGNN-F", ratio=ratio, metrics=metrics,
                            selected=int(final_mask.sum()), units=total, model=model,
                            data=data, diagnostics=diagnostics,
                        )
                        record.update({
                            "mask_init_std": float(selected_cfg["mask_init_std"]),
                            "mask_lr_multiplier": float(selected_cfg["mask_lr_multiplier"]),
                            "initial_final_jaccard": jaccard(init_mask, final_mask),
                            "mean_abs_probability_change": float((final_probs - init_probs).abs().mean()),
                            "mask_path": str(saved_mask.resolve()),
                        })
                        atomic_write_json(path, record)
                        print(
                            f"[core:done] EHGNN-F dataset={dataset} rho={ratio} "
                            f"seed={seed} test={record['test_acc']:.2f}", flush=True
                        )
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                if "Random-Fixed" in methods:
                    path = record_path(outdir, "Random-Fixed", dataset, seed, ratio)
                    saved_mask = mask_path(outdir, "Random-Fixed", dataset, ratio, seed)
                    if path.exists() and saved_mask.exists() and not args.force:
                        print(f"[core:skip] Random-Fixed dataset={dataset} rho={ratio} seed={seed}", flush=True)
                    else:
                        print(f"[core] Random-Fixed dataset={dataset} rho={ratio} seed={seed}", flush=True)
                        edge_index, mask, _ = unit_scores_and_mask_for_baseline(
                            data, method="Random-Fixed", unit="incidence", keep_ratio=ratio,
                            seed=random_offset + seed,
                        )
                        if mask.numel() != total or int(mask.sum()) != expected:
                            raise RuntimeError("Random-Fixed exact hard budget failed")
                        sparse = copy.deepcopy(data)
                        sparse.edge_index = edge_index
                        model, metrics = train_model(
                            data=sparse, split_idx=split, mode="random",
                            num_features=num_features, num_classes=num_classes,
                            keep_ratio=ratio, seed=seed, device=device, **cfg,
                        )
                        atomic_save_mask(saved_mask, mask)
                        record = common_record(
                            args=args, prereq=prereq, dataset=dataset, seed=seed,
                            method="Random-Fixed", ratio=ratio, metrics=metrics,
                            selected=int(mask.sum()), units=total, model=model,
                            data=data, diagnostics=diagnostics,
                        )
                        record["random_mask_seed"] = random_offset + seed
                        record["mask_path"] = str(saved_mask.resolve())
                        atomic_write_json(path, record)
                        print(
                            f"[core:done] Random-Fixed dataset={dataset} rho={ratio} "
                            f"seed={seed} test={record['test_acc']:.2f}", flush=True
                        )
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()


def evaluate_structural(args, protocol, prereq, device: torch.device) -> None:
    outdir = Path(args.outdir)
    (outdir / "evaluation_runs").mkdir(parents=True, exist_ok=True)
    settings = fixed_settings(protocol)
    datasets = requested(args.datasets, settings["datasets"], "datasets")
    seeds = [int(value) for value in requested(args.seeds, settings["model_seeds"], "seeds")]
    methods = requested(args.methods, STRUCTURAL_METHODS, "methods")
    budgets = [float(value) for value in settings["budgets"]]
    cfg = train_kwargs(protocol)
    expected_feature_hash = str(prereq["audit"]["feature_hash"])
    for dataset in datasets:
        data, num_features, num_classes, split, diagnostics = dataset_context(
            dataset, protocol, expected_feature_hash
        )
        total = int(data.edge_index.size(1))
        for method in methods:
            for ratio in budgets:
                stored_mask = mask_path(outdir, method, dataset, ratio)
                if not stored_mask.exists():
                    raise FileNotFoundError(f"Run prepare first: {stored_mask}")
                mask = load_mask(stored_mask)
                expected = max(1, int(ratio * total))
                if mask.numel() != total or int(mask.sum()) != expected:
                    raise RuntimeError(f"Stored exact budget failed for {method}/{dataset}/{ratio}")
                sparse = copy.deepcopy(data)
                sparse.edge_index = data.edge_index[:, mask]
                for seed in seeds:
                    path = record_path(outdir, method, dataset, seed, ratio)
                    if path.exists() and not args.force:
                        print(f"[structural:skip] {method} dataset={dataset} rho={ratio} seed={seed}", flush=True)
                        continue
                    print(f"[structural] {method} dataset={dataset} rho={ratio} seed={seed}", flush=True)
                    model, metrics = train_model(
                        data=sparse, split_idx=split, mode="random",
                        num_features=num_features, num_classes=num_classes,
                        keep_ratio=ratio, seed=seed, device=device, **cfg,
                    )
                    record = common_record(
                        args=args, prereq=prereq, dataset=dataset, seed=seed,
                        method=method, ratio=ratio, metrics=metrics,
                        selected=int(mask.sum()), units=total, model=model,
                        data=data, diagnostics=diagnostics,
                    )
                    record.update(structural_diagnostics(data, mask))
                    record["mask_path"] = str(stored_mask.resolve())
                    atomic_write_json(path, record)
                    print(
                        f"[structural:done] {method} dataset={dataset} rho={ratio} "
                        f"seed={seed} test={record['test_acc']:.2f}", flush=True
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def load_records(path: Path) -> List[Dict[str, object]]:
    return [read_json(item) for item in sorted(path.glob("eval_*.json"))]


def paired_interval(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float(array.mean()), float(array.mean())
    radius = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return float(array.mean() - radius), float(array.mean() + radius)


def summarize(args, protocol, prereq) -> None:
    outdir = Path(args.outdir)
    rows = load_records(outdir / "evaluation_runs")
    settings = fixed_settings(protocol)
    datasets = list(settings["datasets"])
    budgets = [float(value) for value in settings["budgets"]]
    seeds = [int(value) for value in settings["model_seeds"]]
    methods = [RECORD_METHOD[value] for value in (*CORE_METHODS, *STRUCTURAL_METHODS)]
    expected = {
        (dataset, 1.0, seed, "Full")
        for dataset in datasets for seed in seeds
    }
    expected |= {
        (dataset, ratio, seed, method)
        for dataset in datasets for ratio in budgets for seed in seeds
        for method in methods if method != "Full"
    }
    lookup = {}
    for row in rows:
        key = (
            str(row["dataset_id"]), float(row["keep_ratio"]),
            int(row["seed"]), str(row["method"]),
        )
        if key in lookup:
            raise RuntimeError(f"Duplicate evaluation record: {key}")
        lookup[key] = row
    missing = sorted(expected - set(lookup))
    extra = sorted(set(lookup) - expected)
    if missing or extra:
        raise RuntimeError(
            f"Incomplete 340-run matrix: missing={len(missing)} extra={len(extra)}; "
            f"examples={missing[:5]}"
        )

    expected_hash = prereq["protocol_hash"]
    expected_feature_hash = str(prereq["audit"]["feature_hash"])
    audit_errors: List[str] = []
    for key, row in lookup.items():
        dataset, ratio, _, method = key
        total = int(row["num_original_incidences"])
        nodes = int(row["num_nodes"])
        selected = int(row["num_selected"])
        expected_selected = total if method == "Full" else max(1, int(ratio * total))
        checks = {
            "protocol_hash": row.get("protocol_hash") == expected_hash,
            "feature_hash": dataset != "walmart-trips" or row.get("feature_hash") == expected_feature_hash,
            "finite_accuracy": math.isfinite(float(row["val_acc"])) and math.isfinite(float(row["test_acc"])),
            "exact_budget": selected == expected_selected,
            "mask_length": int(row["num_mask_units"]) == total,
            "fixed_self_loops": int(row["num_fixed_self_loops"]) == nodes,
            "forward_count": int(row["forward_incidences"]) == selected + nodes,
            "effective_density": math.isclose(
                float(row["effective_forward_density"]),
                (selected + nodes) / (total + nodes), abs_tol=1e-12,
            ),
        }
        audit_errors.extend(f"{key}: {name}" for name, passed in checks.items() if not passed)
        if method != "Full":
            stored = Path(str(row["mask_path"]))
            if not stored.exists():
                audit_errors.append(f"{key}: missing mask")
            else:
                mask = load_mask(stored)
                if mask.numel() != total or int(mask.sum()) != selected:
                    audit_errors.append(f"{key}: mask artifact mismatch")
    if audit_errors:
        raise RuntimeError("Accuracy-matrix audit failed:\n" + "\n".join(audit_errors[:20]))

    summary_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    for dataset in datasets:
        full_values = [float(lookup[(dataset, 1.0, seed, "Full")]["test_acc"]) for seed in seeds]
        full_mean, full_std = mean_std(full_values)
        for ratio in budgets:
            summary: Dict[str, object] = {
                "dataset_id": dataset,
                "dataset": DISPLAY[dataset],
                "keep_ratio": ratio,
                "Full_mean": full_mean,
                "Full_std": full_std,
            }
            values_by_method = {}
            for method in methods:
                if method == "Full":
                    continue
                values = [float(lookup[(dataset, ratio, seed, method)]["test_acc"]) for seed in seeds]
                values_by_method[method] = np.asarray(values)
                mean, std = mean_std(values)
                summary[f"{method}_mean"] = mean
                summary[f"{method}_std"] = std
            summary_rows.append(summary)
            learned = values_by_method["EHGNN-F"]
            for baseline in ("Random-Fixed", "Degree-prefix", "Spectral-prefix", "Full"):
                base = np.asarray(full_values) if baseline == "Full" else values_by_method[baseline]
                differences = learned - base
                low, high = paired_interval(differences)
                pvalue = 1.0 if np.allclose(differences, 0.0) else float(
                    stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue
                )
                comparison_rows.append({
                    "dataset_id": dataset,
                    "dataset": DISPLAY[dataset],
                    "keep_ratio": ratio,
                    "baseline": baseline,
                    "n": len(seeds),
                    "paired_delta_mean": float(differences.mean()),
                    "paired_delta_ci95_low": low,
                    "paired_delta_ci95_high": high,
                    "wins": int((differences > 0).sum()),
                    "ties": int(np.isclose(differences, 0.0).sum()),
                    "wilcoxon_p_raw": pvalue,
                })

    random_rows = [row for row in comparison_rows if row["baseline"] == "Random-Fixed"]
    structural_rows = [row for row in comparison_rows if row["baseline"] in STRUCTURAL_METHODS]
    full_rows = [row for row in comparison_rows if row["baseline"] == "Full"]
    for family, label in ((random_rows, "new8"), (structural_rows, "new16"), (full_rows, "new8")):
        adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in family])
        for row, value in zip(family, adjusted):
            row[f"wilcoxon_p_holm_{label}"] = value

    write_csv(outdir / "new_dataset_method_summary.csv", summary_rows)
    write_csv(outdir / "new_dataset_paired_comparisons.csv", comparison_rows)
    audit_summary = {
        "created_at": timestamp(),
        "protocol_hash": expected_hash,
        "audit_pass": True,
        "number_of_records": len(rows),
        "expected_records": 340,
        "core_records": 180,
        "structural_records": 160,
        "walmart_feature_hash": expected_feature_hash,
        "all_exact_budgets": True,
        "all_fixed_self_loop_counts": True,
        "all_forward_counts": True,
        "all_metrics_finite": True,
    }
    atomic_write_json(outdir / "accuracy_matrix_audit.json", audit_summary)

    lines = [
        "# Table 2 scale-dataset panel", "",
        "All values are mean +/- sample standard deviation over ten paired model seeds.", "",
        "| Dataset | Budget | Full | EHGNN-F | Random-Fixed | Cardinality | Laplacian-proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def cell(method: str) -> str:
            return f"{float(row[f'{method}_mean']):.2f} +/- {float(row[f'{method}_std']):.2f}"
        lines.append(
            f"| {row['dataset']} | {100 * float(row['keep_ratio']):.0f}% | "
            f"{cell('Full')} | {cell('EHGNN-F')} | {cell('Random-Fixed')} | "
            f"{cell('Degree-prefix')} | {cell('Spectral-prefix')} |"
        )
    (outdir / "table2_scale_panel.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit_summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "evaluate-core", "evaluate-structural", "summarize"),
        required=True,
    )
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--frozen-config", default=str(DEFAULT_FROZEN))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    prereq = verify_prerequisites(protocol_path, protocol, Path(args.frozen_config))
    device = torch.device(
        args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    )
    if args.stage == "prepare":
        prepare(args, protocol, prereq, device)
    elif args.stage == "evaluate-core":
        evaluate_core(args, protocol, prereq, device)
    elif args.stage == "evaluate-structural":
        evaluate_structural(args, protocol, prereq, device)
    else:
        summarize(args, protocol, prereq)


if __name__ == "__main__":
    main()
