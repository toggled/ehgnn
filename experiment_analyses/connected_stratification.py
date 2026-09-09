#!/usr/bin/env python3
"""Compare test accuracy on originally connected and isolated nodes."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

from node_classification_utils import train_learned_with_trajectory
from training_utils import train_model, unit_scores_and_mask_for_baseline
from main_accuracy import dataset_context, training_kwargs


DEFAULT_PROTOCOL = Path("experiment_specs/connected_nodes.json")


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
    if protocol.get("protocol_version") != "connected-nodes-v1":
        raise RuntimeError(f"Unexpected specification version in {path}")
    if protocol.get("status") != "fixed":
        raise RuntimeError(f"Experiment specification is not fixed: {path}")
    for dependency in protocol["dependencies"]:
        dependency_path = Path(str(dependency["path"]))
        observed = file_hash(dependency_path)
        if observed != dependency["sha256"]:
            raise RuntimeError(
                f"Dependency changed: {dependency_path}: "
                f"{observed} != {dependency['sha256']}"
            )
    return protocol


def selected(values: Sequence | None, allowed: Sequence, label: str) -> list:
    result = list(allowed if values is None else values)
    if not set(result) <= set(allowed):
        raise ValueError(f"Requested {label} outside the experiment specification: {result}")
    return result


def cell_path(outdir: Path, dataset: str, budget: float, seed: int, method: str) -> Path:
    budget_tag = f"{int(round(100 * budget)):03d}"
    return outdir / "evaluation_runs" / f"{method}_{dataset}_r{budget_tag}_s{seed}.json"


def original_node_groups(data) -> Dict[str, torch.Tensor]:
    node_ids = data.edge_index[0].detach().cpu()
    degree = torch.bincount(node_ids, minlength=int(data.n_x))
    return {"connected": degree > 0, "isolated": degree == 0}


@torch.no_grad()
def prediction_metrics(model, data, split_idx, groups, device: torch.device) -> Dict[str, object]:
    local_data = copy.deepcopy(data).to(device)
    model.eval()
    predictions = model(local_data, is_test=True).argmax(dim=-1).detach().cpu()
    labels = data.y.detach().cpu()
    test = split_idx["test"].detach().cpu()
    if test.dtype == torch.bool:
        test_mask = test.clone()
        test_indices = test.nonzero(as_tuple=False).view(-1)
    else:
        test_indices = test.long().view(-1)
        test_mask = torch.zeros(int(data.n_x), dtype=torch.bool)
        test_mask[test_indices] = True

    result: Dict[str, object] = {}
    for group_name, group_mask in groups.items():
        indices = (test_mask & group_mask).nonzero(as_tuple=False).view(-1)
        if indices.numel() == 0:
            raise RuntimeError(f"No {group_name} test nodes")
        result[f"test_{group_name}_count"] = int(indices.numel())
        result[f"test_{group_name}_acc"] = 100.0 * float(
            (predictions[indices] == labels[indices]).float().mean().item()
        )

    result["test_count"] = int(test_indices.numel())
    result["test_acc"] = 100.0 * float(
        (predictions[test_indices] == labels[test_indices]).float().mean().item()
    )
    weighted = sum(
        float(result[f"test_{name}_acc"]) * int(result[f"test_{name}_count"])
        for name in groups
    ) / int(result["test_count"])
    result["weighted_subgroup_test_acc"] = weighted
    result["subgroup_reconstruction_abs_error"] = abs(weighted - float(result["test_acc"]))
    return result


def archive_path(protocol, dataset: str, budget: float, seed: int, method: str) -> Path:
    budget_tag = f"{int(round(100 * budget)):03d}"
    archive_method = "ehgnnf" if method == "ehgnnf" else "random"
    return Path(str(protocol["source_confirmation"]["output_root"])) / "evaluation_runs" / (
        f"eval_{archive_method}_{dataset}_r{budget_tag}_s{seed}.json"
    )


def run_cell(
    protocol: Mapping[str, object],
    protocol_path: Path,
    dataset: str,
    budget: float,
    seed: int,
    method: str,
    device: torch.device,
) -> Dict[str, object]:
    source = protocol["source_confirmation"]
    source_protocol = read_json(Path(str(source["protocol_path"])))
    frozen = read_json(Path(str(source["frozen_config_path"])))
    data, num_features, num_classes, split_idx = dataset_context(dataset, source_protocol)
    groups = original_node_groups(data)
    train_args = training_kwargs(source_protocol)
    random_seed = int(source_protocol["evaluation"]["random_mask_seed_offset"]) + seed

    if method == "ehgnnf":
        model, metrics, _, _, final_mask, _, _ = train_learned_with_trajectory(
            data=data,
            split_idx=split_idx,
            mode="learnmask",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=budget,
            seed=seed,
            device=device,
            trajectory_every=int(train_args["epochs"]) + 1,
            mask_init_std=float(frozen["selected"]["mask_init_std"]),
            mask_lr_multiplier=float(frozen["selected"]["mask_lr_multiplier"]),
            include_test_metrics=True,
            **train_args,
        )
        evaluation_data = data
        selected_mask = final_mask
    elif method == "random":
        random_edge_index, selected_mask, _ = unit_scores_and_mask_for_baseline(
            data,
            method="Random-Fixed",
            unit="incidence",
            keep_ratio=budget,
            seed=random_seed,
        )
        evaluation_data = copy.deepcopy(data)
        evaluation_data.edge_index = random_edge_index
        model, metrics = train_model(
            data=evaluation_data,
            split_idx=split_idx,
            mode="random",
            num_features=num_features,
            num_classes=num_classes,
            keep_ratio=budget,
            seed=seed,
            device=device,
            **train_args,
        )
    else:
        raise ValueError(method)

    prediction = prediction_metrics(model, evaluation_data, split_idx, groups, device)
    archive = read_json(archive_path(protocol, dataset, budget, seed, method))
    archive_difference = float(prediction["test_acc"]) - float(archive["test_acc"])
    expected_selected = max(1, int(budget * int(data.edge_index.size(1))))
    selected_nodes = torch.bincount(
        data.edge_index[0, selected_mask.bool()].detach().cpu(),
        minlength=int(data.n_x),
    ) > 0
    originally_connected = groups["connected"]

    row = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "budget": budget,
        "seed": seed,
        "method": method,
        "num_nodes": int(data.n_x),
        "num_original_incidences": int(data.edge_index.size(1)),
        "num_originally_connected_nodes": int(originally_connected.sum().item()),
        "num_originally_isolated_nodes": int((~originally_connected).sum().item()),
        "originally_isolated_fraction": float((~originally_connected).float().mean().item()),
        "num_selected": int(selected_mask.sum().item()),
        "expected_selected": expected_selected,
        "selected_originally_connected_nodes": int((selected_nodes & originally_connected).sum().item()),
        "selected_originally_connected_fraction": float(
            (selected_nodes & originally_connected).sum().item()
            / max(1, originally_connected.sum().item())
        ),
        "best_epoch": int(metrics["best_epoch"]),
        "epochs_run": int(metrics["epochs_run"]),
        "archive_test_acc": float(archive["test_acc"]),
        "archive_test_acc_difference": archive_difference,
        **prediction,
    }
    if row["num_selected"] != expected_selected:
        raise RuntimeError(f"Exact budget failed: {row}")
    if row["subgroup_reconstruction_abs_error"] > 1e-5:
        raise RuntimeError(f"Subgroup reconstruction failed: {row}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return mean, mean - half, mean + half


def paired_pvalues(differences: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(differences, dtype=float)
    t_p = float(stats.ttest_1samp(array, popmean=0.0).pvalue)
    try:
        w_p = float(stats.wilcoxon(array, alternative="two-sided").pvalue)
    except ValueError:
        w_p = 1.0
    return t_p, w_p


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def summarize(protocol: Mapping[str, object], protocol_path: Path) -> None:
    outdir = Path(str(protocol["output_root"]))
    methods = list(protocol["evaluation"]["methods"])
    paths = [
        cell_path(outdir, dataset, float(budget), int(seed), method)
        for dataset in protocol["datasets"]
        for budget in protocol["evaluation"]["budgets"]
        for seed in protocol["evaluation"]["seeds"]
        for method in methods
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete result matrix; missing {len(missing)} cells")
    rows = [read_json(path) for path in paths]
    write_csv(outdir / "evaluation_records.csv", rows)

    summaries = []
    for dataset in protocol["datasets"]:
        for budget in protocol["evaluation"]["budgets"]:
            cell = [
                row for row in rows
                if row["dataset_id"] == dataset and float(row["budget"]) == float(budget)
            ]
            by_method = {
                method: sorted(
                    [row for row in cell if row["method"] == method],
                    key=lambda row: int(row["seed"]),
                )
                for method in methods
            }
            if any(len(group) != len(protocol["evaluation"]["seeds"]) for group in by_method.values()):
                raise RuntimeError(f"Incomplete paired cell: {dataset}, {budget}")
            for group_name in ("all", "connected", "isolated"):
                metric = "test_acc" if group_name == "all" else f"test_{group_name}_acc"
                learned = [float(row[metric]) for row in by_method["ehgnnf"]]
                random = [float(row[metric]) for row in by_method["random"]]
                differences = [left - right for left, right in zip(learned, random)]
                delta, low, high = mean_ci(differences)
                t_p, w_p = paired_pvalues(differences)
                summaries.append({
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": float(budget),
                    "group": group_name,
                    "test_node_count": int(by_method["ehgnnf"][0][
                        "test_count" if group_name == "all" else f"test_{group_name}_count"
                    ]),
                    "ehgnnf_mean": float(np.mean(learned)),
                    "ehgnnf_std": float(np.std(learned, ddof=1)),
                    "random_mean": float(np.mean(random)),
                    "random_std": float(np.std(random, ddof=1)),
                    "paired_delta_mean": delta,
                    "paired_delta_ci95_low": low,
                    "paired_delta_ci95_high": high,
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                    "paired_wins": int(sum(value > 0 for value in differences)),
                })

    subgroup_indices = [
        index for index, row in enumerate(summaries) if row["group"] in {"connected", "isolated"}
    ]
    adjusted = holm_adjust([float(summaries[index]["paired_t_p"]) for index in subgroup_indices])
    for index, value in zip(subgroup_indices, adjusted):
        summaries[index]["paired_t_p_holm_16"] = value
    write_csv(outdir / "subgroup_summary.csv", summaries)

    max_archive_difference = max(abs(float(row["archive_test_acc_difference"])) for row in rows)
    max_reconstruction_error = max(float(row["subgroup_reconstruction_abs_error"]) for row in rows)
    metadata = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "run_count": len(rows),
        "complete": True,
        "exact_budget_all_runs": all(row["num_selected"] == row["expected_selected"] for row in rows),
        "maximum_archive_accuracy_difference_points": max_archive_difference,
        "maximum_subgroup_reconstruction_error_points": max_reconstruction_error,
        "holm_family": "16 connected/isolated dataset-budget comparisons",
    }
    write_json(outdir / "summary_metadata.json", metadata)

    table_rows = [row for row in summaries if row["group"] in {"connected", "isolated"}]
    lines = [
        "# Connected and isolated test-node results",
        "",
        "Groups are fixed from the original hypergraph before sparsification. Values are",
        "ten-seed test accuracy means and paired EHGNN-F minus Random-Fixed differences in",
        "percentage points.",
        "",
        "| Dataset | $\\rho$ | Group | Test nodes | EHGNN-F | Random-Fixed | $\\Delta$ | 95% CI | Holm $p$ |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['dataset']} | {row['budget']:.1f} | {row['group'].title()} | "
            f"{row['test_node_count']:,} | {row['ehgnnf_mean']:.2f} | "
            f"{row['random_mean']:.2f} | {row['paired_delta_mean']:+.2f} | "
            f"[{row['paired_delta_ci95_low']:.2f}, {row['paired_delta_ci95_high']:.2f}] | "
            f"{row['paired_t_p_holm_16']:.3g} |"
        )
    lines.extend([
        "",
        f"Maximum difference from the archived aggregate accuracies: {max_archive_difference:.6g} points.",
        f"Maximum subgroup-to-overall reconstruction error: {max_reconstruction_error:.6g} points.",
    ])
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--budgets", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    if args.command == "summarize":
        summarize(protocol, protocol_path)
        return

    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    budgets = selected(args.budgets, protocol["evaluation"]["budgets"], "budgets")
    seeds = selected(args.seeds, protocol["evaluation"]["seeds"], "seeds")
    methods = selected(args.methods, protocol["evaluation"]["methods"], "methods")
    outdir = Path(str(protocol["output_root"]))
    device = torch.device(args.device)
    for dataset in datasets:
        for seed in seeds:
            for budget in budgets:
                for method in methods:
                    path = cell_path(outdir, dataset, float(budget), int(seed), method)
                    if path.exists() and not args.force:
                        print(f"[connected-nodes:skip] {dataset} rho={budget} seed={seed} {method}", flush=True)
                        continue
                    print(f"[connected-nodes] {dataset} rho={budget} seed={seed} {method}", flush=True)
                    row = run_cell(
                        protocol,
                        protocol_path,
                        dataset,
                        float(budget),
                        int(seed),
                        method,
                        device,
                    )
                    write_json(path, row)
                    print(
                        f"[connected-nodes:done] aggregate={row['test_acc']:.3f} "
                        f"connected={row['test_connected_acc']:.3f} "
                        f"isolated={row['test_isolated_acc']:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
