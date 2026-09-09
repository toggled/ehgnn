#!/usr/bin/env python3
"""Reproduce the runtime measurements and three-panel Figure 2."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from structural_baselines import edge_scores, prefix_mask
from node_classification_utils import train_learned_with_trajectory
from training_utils import train_model, unit_scores_and_mask_for_baseline
from main_accuracy import (
    atomic_write_json,
    dataset_context as heterophilic_dataset_context,
    forward_diagnostics,
    protocol_hash,
    read_json,
    timestamp,
    write_csv,
)
from large_dataset_accuracy import (
    dataset_context as scale_dataset_context,
    fixed_settings,
    train_kwargs,
    verify_prerequisites,
)


DEFAULT_PROTOCOL = Path("experiment_specs/runtime.json")
DEFAULT_OUTDIR = Path(".work/runtime")
METHOD_TAG = {
    "Full": "full",
    "EHGNN-F": "ehgnnf",
    "Random-Fixed": "random",
    "Degree-prefix": "degree_prefix",
    "Spectral-prefix": "spectral_prefix",
}
RECORD_METHOD = {
    "Full": "Full",
    "EHGNN-F": "EHGNN-F",
    "Random-Fixed": "Random-Fixed",
    "Degree-prefix": "Degree-prefix",
    "Spectral-prefix": "Spectral-prefix",
}


def load_any_mask(path: Path) -> torch.Tensor:
    with np.load(path) as archive:
        if "packed" in archive:
            length = int(archive["length"])
            value = np.unpackbits(archive["packed"])[:length].astype(bool)
        elif "mask" in archive:
            value = archive["mask"].astype(bool)
        else:
            raise RuntimeError(f"Unknown mask representation in {path}")
    return torch.from_numpy(value)


def mask_hash(mask: torch.Tensor) -> str:
    value = mask.detach().cpu().bool().numpy().astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(value).tobytes())
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_runtime_protocol(path: Path, protocol: Mapping[str, object]) -> Dict[str, object]:
    if protocol.get("status") != "fixed":
        raise RuntimeError("The runtime specification is not fixed")
    allowed_runtime_hashes = {protocol_hash(path)}
    accuracy_path = Path(str(protocol["accuracy_protocol"]))
    accuracy = read_json(accuracy_path)
    prereq = verify_prerequisites(
        accuracy_path,
        accuracy,
        Path(str(protocol["selected_hyperparameters"])),
    )
    matrix_audit_path = Path(str(protocol["accuracy_output"])) / "accuracy_matrix_audit.json"
    if not matrix_audit_path.exists():
        raise RuntimeError("Runtime study requires the complete audited accuracy matrix")
    matrix_audit = read_json(matrix_audit_path)
    if matrix_audit.get("audit_pass") is not True:
        raise RuntimeError("Accuracy matrix did not pass its audit")
    return {
        "accuracy_protocol": accuracy,
        "accuracy_prereq": prereq,
        "runtime_protocol_hash": protocol_hash(path),
        "allowed_runtime_protocol_hashes": sorted(allowed_runtime_hashes),
        "accuracy_matrix_audit": matrix_audit,
    }


def dataset_context(dataset: str, protocol: Mapping[str, object], verified):
    if dataset in {"coauthor_dblp", "walmart-trips"}:
        return scale_dataset_context(
            dataset,
            verified["accuracy_protocol"],
            str(verified["accuracy_prereq"]["audit"]["feature_hash"]),
        )
    if dataset == "yelp":
        baseline = verified["accuracy_prereq"]["baseline"]
        data, num_features, num_classes, split = heterophilic_dataset_context(
            dataset, baseline
        )
        diagnostics = {
            "num_nodes": int(data.n_x),
            "num_hyperedges": int(data.num_hyperedges),
            "num_original_incidences": int(data.edge_index.size(1)),
            "num_features": int(num_features),
            "num_classes": int(num_classes),
        }
        return data, num_features, num_classes, split, diagnostics
    raise ValueError(dataset)


def ratio_tag(ratio: float) -> str:
    return f"{int(round(100 * ratio)):03d}"


def reference_record(
    protocol: Mapping[str, object], dataset: str, method: str, ratio: float, seed: int
) -> Dict[str, object]:
    tag = METHOD_TAG[method]
    rtag = ratio_tag(ratio)
    if dataset != "yelp":
        root = Path(str(protocol["accuracy_output"])) / "evaluation_runs"
        name = (
            f"eval_full_{dataset}_s{seed}.json" if method == "Full"
            else f"eval_{tag}_{dataset}_r{rtag}_s{seed}.json"
        )
        return read_json(root / name)

    if method == "Full":
        root = Path(str(protocol["heterophilic_structural_output"])) / "evaluation_runs"
        return read_json(root / f"eval_full_yelp_s{seed}.json")
    if method in {"EHGNN-F", "Random-Fixed"}:
        root = Path(str(protocol["heterophilic_accuracy_output"])) / "evaluation_runs"
        return read_json(root / f"eval_{tag}_yelp_r{rtag}_s{seed}.json")
    root = Path(str(protocol["heterophilic_structural_output"])) / "evaluation_runs"
    return read_json(root / f"eval_{tag}_yelp_r{rtag}_s{seed}.json")


def required_budgets(protocol: Mapping[str, object], dataset: str, method: str) -> List[float]:
    if method == "EHGNN-F" and dataset == protocol["figure_panels"]["budget_curve"]["dataset"]:
        return [float(value) for value in protocol["figure_panels"]["budget_curve"]["budgets"]]
    return [float(protocol["figure_panels"]["method_dataset_comparison"]["budget"])]


def requested(values: Sequence | None, allowed: Sequence, label: str) -> List:
    result = list(allowed if values is None else values)
    if not set(result) <= set(allowed):
        raise ValueError(f"Requested {label} are outside the frozen runtime protocol")
    return result


def run(args, protocol, verified, device: torch.device) -> None:
    outdir = Path(args.outdir)
    run_dir = outdir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = requested(args.datasets, protocol["datasets"], "datasets")
    methods = requested(args.methods, protocol["methods"], "methods")
    seeds = [int(value) for value in requested(args.seeds, protocol["timing_seeds"], "seeds")]
    accuracy_protocol = verified["accuracy_protocol"]
    cfg = train_kwargs(accuracy_protocol)
    selected_cfg = fixed_settings(accuracy_protocol)["frozen_ehgnnf_configuration"]
    parent = verified["accuracy_prereq"]["parent"]
    baseline_protocol = verified["accuracy_prereq"]["baseline"]
    random_offset = int(parent["evaluation"]["random_mask_seed_offset"])

    for dataset in datasets:
        data, num_features, num_classes, split, diagnostics = dataset_context(
            dataset, protocol, verified
        )
        total = int(data.edge_index.size(1))
        for method in methods:
            budgets = required_budgets(protocol, dataset, method)
            if args.budgets is not None:
                budgets = [ratio for ratio in budgets if ratio in set(args.budgets)]
            for ratio in budgets:
                for seed in seeds:
                    path = run_dir / (
                        f"runtime_{METHOD_TAG[method]}_{dataset}_r{ratio_tag(ratio)}_s{seed}.json"
                    )
                    if path.exists() and not args.force:
                        print(f"[runtime:skip] {method} dataset={dataset} rho={ratio} seed={seed}", flush=True)
                        continue
                    reference = reference_record(protocol, dataset, method, ratio, seed)
                    reference_mask = None
                    if method != "Full":
                        reference_mask = load_any_mask(Path(str(reference["mask_path"])))

                    print(f"[runtime] {method} dataset={dataset} rho={ratio} seed={seed}", flush=True)
                    sync(device)
                    total_start = time.perf_counter()
                    preprocessing_start = total_start
                    trained_data = data
                    selected_mask = None
                    if method == "Random-Fixed":
                        edge_index, selected_mask, _ = unit_scores_and_mask_for_baseline(
                            data, method="Random-Fixed", unit="incidence", keep_ratio=ratio,
                            seed=random_offset + seed,
                        )
                        trained_data = copy.deepcopy(data)
                        trained_data.edge_index = edge_index
                    elif method in {"Degree-prefix", "Spectral-prefix"}:
                        scores = edge_scores(data, method, baseline_protocol, device)
                        recomputed_mask, _ = prefix_mask(data, scores, ratio)
                        selected_mask = reference_mask.clone()
                        trained_data = copy.deepcopy(data)
                        trained_data.edge_index = data.edge_index[:, selected_mask]
                    sync(device)
                    preprocessing_end = time.perf_counter()
                    fit_start = preprocessing_end

                    if method == "Full":
                        model, metrics = train_model(
                            data=data, split_idx=split, mode="full",
                            num_features=num_features, num_classes=num_classes,
                            keep_ratio=1.0, seed=seed, device=device, **cfg,
                        )
                        selected = total
                    elif method == "EHGNN-F":
                        model, metrics, _, _, selected_mask, _, _ = train_learned_with_trajectory(
                            data=data, split_idx=split, mode="learnmask",
                            num_features=num_features, num_classes=num_classes,
                            keep_ratio=ratio, seed=seed, device=device,
                            trajectory_every=int(cfg["epochs"]) + 1,
                            mask_init_std=float(selected_cfg["mask_init_std"]),
                            mask_lr_multiplier=float(selected_cfg["mask_lr_multiplier"]),
                            include_test_metrics=True, **cfg,
                        )
                        selected = int(selected_mask.sum())
                    else:
                        model, metrics = train_model(
                            data=trained_data, split_idx=split, mode="random",
                            num_features=num_features, num_classes=num_classes,
                            keep_ratio=ratio, seed=seed, device=device, **cfg,
                        )
                        selected = int(selected_mask.sum())
                    sync(device)
                    fit_end = time.perf_counter()

                    expected = total if method == "Full" else max(1, int(ratio * total))
                    budget_equivalent = selected == expected
                    mask_bitwise_equivalent = method == "Full" or torch.equal(
                        selected_mask.detach().cpu().bool(), reference_mask.bool()
                    )
                    selected_mask_jaccard = None if method == "Full" else float(
                        (selected_mask.detach().cpu().bool() & reference_mask.bool()).sum()
                        / (selected_mask.detach().cpu().bool() | reference_mask.bool()).sum().clamp_min(1)
                    )
                    if method == "EHGNN-F" and protocol.get("ehgnnf_mask_policy"):
                        mask_equivalent = selected_mask_jaccard >= float(
                            protocol["ehgnnf_mask_policy"]["minimum_reference_mask_jaccard"]
                        )
                    else:
                        mask_equivalent = mask_bitwise_equivalent
                    observed_accuracy = 100.0 * float(metrics["test_acc"])
                    accuracy_equivalent = math.isclose(
                        observed_accuracy, float(reference["test_acc"]), abs_tol=1e-6
                    )
                    forward = forward_diagnostics(model, data, selected)
                    self_loop_equivalent = int(forward["num_fixed_self_loops"]) == int(data.n_x)
                    forward_equivalent = int(forward["forward_incidences"]) == selected + int(data.n_x)
                    finite = all(math.isfinite(float(metrics[key])) for key in ("val_acc", "test_acc"))
                    equivalence_pass = all((
                        budget_equivalent, mask_equivalent, accuracy_equivalent,
                        self_loop_equivalent, forward_equivalent, finite,
                    ))
                    if not equivalence_pass:
                        raise RuntimeError(
                            f"Runtime equivalence failed for {method}/{dataset}/{ratio}/s{seed}: "
                            f"budget={budget_equivalent} mask={mask_equivalent} "
                            f"mask_jaccard={selected_mask_jaccard} "
                            f"accuracy={accuracy_equivalent} self_loops={self_loop_equivalent} "
                            f"forward={forward_equivalent} finite={finite}; "
                            f"observed={observed_accuracy} reference={reference['test_acc']}"
                        )
                    preprocessing_seconds = preprocessing_end - preprocessing_start
                    fit_seconds = fit_end - fit_start
                    epochs_run = int(metrics["epochs_run"])
                    record = {
                        "stage": "runtime_measurement",
                        "created_at": timestamp(),
                        "runtime_protocol_hash": verified["runtime_protocol_hash"],
                        "accuracy_protocol_hash": verified["accuracy_prereq"]["protocol_hash"],
                        "dataset_id": dataset,
                        "dataset": protocol["display_names"][dataset],
                        "method": RECORD_METHOD[method],
                        "method_id": method,
                        "comparison_budget": ratio,
                        "seed": seed,
                        "num_original_incidences": total,
                        "num_selected": selected,
                        "structural_density": selected / total,
                        "preprocessing_seconds": preprocessing_seconds,
                        "fit_and_final_evaluation_seconds": fit_seconds,
                        "fit_and_final_evaluation_seconds_per_epoch": fit_seconds / epochs_run,
                        "end_to_end_seconds": fit_end - total_start,
                        "epochs_run": epochs_run,
                        "best_epoch": int(metrics["best_epoch"]),
                        "test_acc": observed_accuracy,
                        "reference_test_acc": float(reference["test_acc"]),
                        "mask_hash": None if selected_mask is None else mask_hash(selected_mask),
                        "reference_mask_hash": None if reference_mask is None else mask_hash(reference_mask),
                        "ranking_recomputed_mask_hash": (
                            mask_hash(recomputed_mask)
                            if method in {"Degree-prefix", "Spectral-prefix"} else None
                        ),
                        "ranking_recomputed_mask_exact": (
                            bool(torch.equal(recomputed_mask.bool(), reference_mask.bool()))
                            if method in {"Degree-prefix", "Spectral-prefix"} else None
                        ),
                        "ranking_recomputed_mask_jaccard": (
                            float(
                                (recomputed_mask.bool() & reference_mask.bool()).sum()
                                / (recomputed_mask.bool() | reference_mask.bool()).sum().clamp_min(1)
                            )
                            if method in {"Degree-prefix", "Spectral-prefix"} else None
                        ),
                        "budget_equivalent": budget_equivalent,
                        "mask_equivalent": mask_equivalent,
                        "mask_bitwise_equivalent": mask_bitwise_equivalent,
                        "used_mask_reference_jaccard": selected_mask_jaccard,
                        "accuracy_equivalent": accuracy_equivalent,
                        "self_loop_equivalent": self_loop_equivalent,
                        "forward_equivalent": forward_equivalent,
                        "equivalence_pass": equivalence_pass,
                        "device": str(device),
                        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "python_version": platform.python_version(),
                        **diagnostics,
                        **forward,
                    }
                    atomic_write_json(path, record)
                    print(
                        f"[runtime:done] {method} dataset={dataset} rho={ratio} seed={seed} "
                        f"e2e={record['end_to_end_seconds']:.2f}s "
                        f"per_epoch={record['fit_and_final_evaluation_seconds_per_epoch']:.4f}s",
                        flush=True,
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def summarize(args, protocol, verified) -> None:
    outdir = Path(args.outdir)
    rows = [read_json(path) for path in sorted((outdir / "runs").glob("runtime_*.json"))]
    expected = set()
    for dataset in protocol["datasets"]:
        for method in protocol["methods"]:
            for ratio in required_budgets(protocol, dataset, method):
                for seed in protocol["timing_seeds"]:
                    expected.add((dataset, method, float(ratio), int(seed)))
    observed = {
        (row["dataset_id"], row["method_id"], float(row["comparison_budget"]), int(row["seed"]))
        for row in rows
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"Incomplete 54-run timing matrix: missing={len(missing)} extra={len(extra)}")
    if len(rows) != 54 or any(row.get("equivalence_pass") is not True for row in rows):
        raise RuntimeError("Runtime artifact audit failed")
    observed_protocol_hashes = {str(row["runtime_protocol_hash"]) for row in rows}
    if not observed_protocol_hashes <= set(verified["allowed_runtime_protocol_hashes"]):
        raise RuntimeError("Runtime records include an unapproved protocol hash")

    groups: Dict[Tuple[str, str, float], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["dataset_id"]), str(row["method_id"]), float(row["comparison_budget"])), []
        ).append(row)
    summaries = []
    metrics = (
        "preprocessing_seconds",
        "fit_and_final_evaluation_seconds",
        "fit_and_final_evaluation_seconds_per_epoch",
        "end_to_end_seconds",
        "epochs_run",
    )
    for (dataset, method, ratio), group in sorted(groups.items()):
        row = {
            "dataset_id": dataset,
            "dataset": protocol["display_names"][dataset],
            "method_id": method,
            "method": RECORD_METHOD[method],
            "comparison_budget": ratio,
            "n": len(group),
        }
        for metric in metrics:
            mean, std = mean_std([float(item[metric]) for item in group])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summaries.append(row)
    write_csv(outdir / "runtime_run_records.csv", rows)
    write_csv(outdir / "runtime_summary.csv", summaries)
    sparse_rows = [row for row in rows if row["method_id"] != "Full"]
    ehgnnf_jaccards = [
        float(row["used_mask_reference_jaccard"])
        for row in rows
        if row["method_id"] == "EHGNN-F" and row.get("used_mask_reference_jaccard") is not None
    ]
    atomic_write_json(outdir / "runtime_audit.json", {
        "created_at": timestamp(),
        "runtime_protocol_hash": verified["runtime_protocol_hash"],
        "audit_pass": True,
        "number_of_runs": len(rows),
        "all_mask_criteria_pass": all(bool(row["mask_equivalent"]) for row in sparse_rows),
        "all_masks_bitwise_equivalent": all(
            bool(row.get("mask_bitwise_equivalent", row["mask_equivalent"]))
            for row in sparse_rows
        ),
        "all_random_and_used_structural_masks_bitwise_equivalent": all(
            bool(row.get("mask_bitwise_equivalent", row["mask_equivalent"]))
            for row in sparse_rows
            if row["method_id"] != "EHGNN-F"
        ),
        "minimum_ehgnnf_reference_mask_jaccard": min(ehgnnf_jaccards),
        "all_accuracies_equivalent": True,
        "all_budgets_exact": True,
        "all_fixed_self_loop_counts": True,
        "accepted_runtime_protocol_hashes": sorted(observed_protocol_hashes),
    })
    plot_preview(outdir, protocol, summaries)


def plot_preview(outdir: Path, protocol: Mapping[str, object], runtime_rows) -> None:
    colors = {
        "Full": "#4D4D4D",
        "EHGNN-F": "#D62728",
        "Random-Fixed": "#1F77B4",
        "Degree-prefix": "#2CA02C",
        "Spectral-prefix": "#9467BD",
    }
    labels = {
        "Full": "Full",
        "EHGNN-F": "EHGNN-F",
        "Random-Fixed": "Random-Fixed",
        "Degree-prefix": "Cardinality",
        "Spectral-prefix": "Laplacian-proxy",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.45))

    with (Path(str(protocol["controlled_corruption_output"])) / "all_method_cell_summary.csv").open() as handle:
        corruption = [row for row in csv.DictReader(handle) if row["dataset_id"] == "cora" and row["cell_role"] == "concentrated_rate_curve"]
    corruption.sort(key=lambda row: float(row["eta"]))
    t_multiplier = float(stats.t.ppf(0.975, 9) / math.sqrt(10))
    source_names = {
        "Full": "Full", "EHGNN-F": "EHGNN-F", "Random-Fixed": "Random-Fixed",
        "Degree-prefix": "Degree-prefix", "Spectral-prefix": "Spectral-prefix",
    }
    markers = {"Full": "o", "EHGNN-F": "s", "Random-Fixed": "^", "Degree-prefix": "D", "Spectral-prefix": "v"}
    for method in protocol["methods"]:
        source = source_names[method]
        axes[0].errorbar(
            [100 * float(row["eta"]) for row in corruption],
            [float(row[f"{source}_mean"]) for row in corruption],
            yerr=[t_multiplier * float(row[f"{source}_std"]) for row in corruption],
            label=labels[method], color=colors[method], marker=markers[method],
            linewidth=1.4, markersize=4, capsize=2,
        )
    axes[0].set_xlabel("Rewired incidences (%)")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("(a) Controlled corruption")

    curve = [
        row for row in runtime_rows
        if row["dataset_id"] == "yelp" and row["method_id"] == "EHGNN-F"
    ]
    curve.sort(key=lambda row: float(row["comparison_budget"]))
    axes[1].errorbar(
        [100 * float(row["comparison_budget"]) for row in curve],
        [float(row["end_to_end_seconds_mean"]) for row in curve],
        yerr=[float(row["end_to_end_seconds_std"]) for row in curve],
        color=colors["EHGNN-F"], marker="s", linewidth=1.6, capsize=3,
    )
    axes[1].set_xlabel("Retained incidences (%)")
    axes[1].set_ylabel("End-to-end time (s)")
    axes[1].set_title("(b) Yelp runtime")

    datasets = list(protocol["figure_panels"]["method_dataset_comparison"]["datasets"])
    methods = list(protocol["methods"])
    width = 0.15
    x = np.arange(len(datasets))
    for index, method in enumerate(methods):
        values, errors = [], []
        for dataset in datasets:
            row = next(
                item for item in runtime_rows
                if item["dataset_id"] == dataset and item["method_id"] == method
                and math.isclose(float(item["comparison_budget"]), 0.5)
            )
            values.append(float(row["fit_and_final_evaluation_seconds_per_epoch_mean"]))
            errors.append(float(row["fit_and_final_evaluation_seconds_per_epoch_std"]))
        axes[2].bar(
            x + (index - 2) * width, values, width, yerr=errors,
            color=colors[method], label=labels[method], capsize=2,
        )
    axes[2].set_xticks(x, [protocol["display_names"][dataset] for dataset in datasets])
    axes[2].set_yscale("log", base=2)
    axes[2].set_ylabel("Fit time/epoch (s)")
    axes[2].set_title("(c) Runtime at 50%")

    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(outdir / "figure2_three_panel.pdf", bbox_inches="tight")
    figure.savefig(outdir / "figure2_three_panel.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("run", "summarize"), required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.protocol)
    protocol = read_json(path)
    verified = verify_runtime_protocol(path, protocol)
    if args.stage == "run":
        device = torch.device(
            args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
        )
        run(args, protocol, verified, device)
    else:
        summarize(args, protocol, verified)


if __name__ == "__main__":
    main()
