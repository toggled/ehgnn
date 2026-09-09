#!/usr/bin/env python3
"""Summarize effective density and retained multi-node structure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from node_classification_utils import prepare_dataset
from training_utils import load_v2e_dataset
from main_accuracy import load_mask
from walmart_feature_check import load_walmart


DEFAULT_PROTOCOL = Path("experiment_specs/retained_structure.json")

METHOD_TAG = {
    "EHGNN-F": "ehgnnf",
    "Random-Fixed": "random",
}


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_dataset(dataset: str, protocol: Mapping[str, object]):
    if dataset in {"actor", "twitch", "pokec", "yelp"}:
        data, _, num_classes = prepare_dataset(dataset)
    elif dataset == "coauthor_dblp":
        data, _, num_classes = load_v2e_dataset(dataset)
    elif dataset == "walmart-trips":
        cfg = protocol["data"]
        data, _, num_classes = load_walmart(
            str(cfg["walmart_feature_noise"]),
            int(cfg["walmart_feature_seed"]),
            int(cfg["walmart_feature_dimension"]),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    data.edge_index = data.edge_index.detach().cpu().long()
    data.y = data.y.detach().cpu().long()
    return data, int(num_classes)


def incidence_agreement(
    node_ids: np.ndarray,
    edge_ids: np.ndarray,
    labels: np.ndarray,
    num_edges: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_sizes = np.bincount(edge_ids, minlength=num_edges).astype(np.int64)
    flat = edge_ids * num_classes + labels[node_ids]
    class_counts = np.bincount(
        flat, minlength=num_edges * num_classes
    ).reshape(num_edges, num_classes)
    same_count = class_counts[edge_ids, labels[node_ids]] - 1
    denominators = edge_sizes[edge_ids] - 1
    agreement = np.full(node_ids.size, np.nan, dtype=np.float64)
    valid = denominators > 0
    agreement[valid] = same_count[valid] / denominators[valid]
    return agreement, edge_sizes


def compute_mask_metrics(
    *,
    node_ids: np.ndarray,
    edge_ids: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    agreement: np.ndarray,
    original_edge_sizes: np.ndarray,
    num_nodes: int,
    num_edges: int,
    num_classes: int,
) -> Dict[str, float]:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.size != node_ids.size:
        raise ValueError("Mask and incidence arrays differ in length")
    selected = int(mask.sum())
    if selected <= 0 or selected >= mask.size:
        raise ValueError("Audit requires a nonempty, nonfull mask")

    retained_edges = edge_ids[mask]
    retained_nodes = node_ids[mask]
    retained_sizes = np.bincount(retained_edges, minlength=num_edges).astype(np.int64)
    zero_edges = int((retained_sizes == 0).sum())
    singleton_edges = int((retained_sizes == 1).sum())
    relational_edges = int((retained_sizes >= 2).sum())

    selected_in_singletons = retained_sizes[retained_edges] == 1
    singleton_budget_fraction = float(selected_in_singletons.mean())

    originally_connected = np.zeros(num_nodes, dtype=bool)
    originally_connected[np.unique(node_ids)] = True
    ordinary_nodes = np.zeros(num_nodes, dtype=bool)
    ordinary_nodes[np.unique(retained_nodes)] = True
    relational_selected = retained_sizes[retained_edges] >= 2
    relational_nodes = np.zeros(num_nodes, dtype=bool)
    relational_nodes[np.unique(retained_nodes[relational_selected])] = True
    connected_count = int(originally_connected.sum())

    eligible = np.isfinite(agreement)
    retained_eligible = mask & eligible
    removed_eligible = (~mask) & eligible

    flat = retained_edges * num_classes + labels[retained_nodes]
    retained_class_counts = np.bincount(
        flat, minlength=num_edges * num_classes
    ).reshape(num_edges, num_classes)
    relational = retained_sizes >= 2
    if relational.any():
        counts = retained_class_counts[relational].astype(np.float64)
        sizes = retained_sizes[relational].astype(np.float64)
        probabilities = counts / sizes[:, None]
        purity = probabilities.max(axis=1)
        if num_classes > 1:
            entropy_terms = np.zeros_like(probabilities)
            positive = probabilities > 0
            entropy_terms[positive] = -probabilities[positive] * np.log(
                probabilities[positive]
            )
            entropy = entropy_terms.sum(axis=1) / math.log(num_classes)
        else:
            entropy = np.zeros_like(sizes)
        relational_purity = float(purity.mean())
        relational_entropy = float(entropy.mean())
    else:
        relational_purity = float("nan")
        relational_entropy = float("nan")

    return {
        "num_selected": selected,
        "structural_density": selected / mask.size,
        "eligible_agreement_incidences": int(eligible.sum()),
        "retained_agreement": float(agreement[retained_eligible].mean()),
        "removed_agreement": float(agreement[removed_eligible].mean()),
        "retained_minus_removed_agreement": float(
            agreement[retained_eligible].mean() - agreement[removed_eligible].mean()
        ),
        "original_singleton_hyperedges": int((original_edge_sizes == 1).sum()),
        "retained_zero_hyperedges": zero_edges,
        "retained_singleton_hyperedges": singleton_edges,
        "retained_relational_hyperedges": relational_edges,
        "retained_zero_hyperedge_fraction": zero_edges / num_edges,
        "retained_singleton_hyperedge_fraction": singleton_edges / num_edges,
        "retained_relational_hyperedge_fraction": relational_edges / num_edges,
        "singleton_budget_fraction": singleton_budget_fraction,
        "ordinary_node_coverage": float(ordinary_nodes[originally_connected].mean()),
        "relational_node_coverage": float(relational_nodes[originally_connected].mean()),
        "retained_relational_edge_purity": relational_purity,
        "retained_relational_edge_entropy": relational_entropy,
        "retained_edge_cardinality_mean_nonempty": float(
            retained_sizes[retained_sizes > 0].mean()
        ),
        "retained_edge_cardinality_median_nonempty": float(
            np.median(retained_sizes[retained_sizes > 0])
        ),
    }


def mask_path(
    protocol: Mapping[str, object], dataset: str, method: str, budget: float, seed: int
) -> Path:
    root = Path(str(protocol["mask_sources"][dataset]))
    tag = METHOD_TAG[method]
    ratio = int(round(100 * budget))
    return root / f"mask_{tag}_{dataset}_r{ratio:03d}_s{seed}.npz"


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_ci(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float(array.mean()), float(array.mean())
    radius = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return float(array.mean() - radius), float(array.mean() + radius)


def run_analysis(protocol: Mapping[str, object], protocol_path: Path, outdir: Path) -> None:
    rows: List[Dict[str, object]] = []
    metrics = [
        "retained_agreement",
        "removed_agreement",
        "retained_minus_removed_agreement",
        "retained_zero_hyperedges",
        "retained_singleton_hyperedges",
        "retained_relational_hyperedges",
        "retained_zero_hyperedge_fraction",
        "retained_singleton_hyperedge_fraction",
        "retained_relational_hyperedge_fraction",
        "singleton_budget_fraction",
        "ordinary_node_coverage",
        "relational_node_coverage",
        "retained_relational_edge_purity",
        "retained_relational_edge_entropy",
        "retained_edge_cardinality_mean_nonempty",
        "retained_edge_cardinality_median_nonempty",
    ]
    for dataset in protocol["datasets"]:
        print(f"[retained-structure:data] {dataset}", flush=True)
        data, num_classes = load_dataset(str(dataset), protocol)
        node_ids = data.edge_index[0].numpy().astype(np.int64, copy=False)
        edge_ids = data.edge_index[1].numpy().astype(np.int64, copy=False)
        labels = data.y.numpy().astype(np.int64, copy=False)
        num_nodes = int(data.n_x)
        num_edges = int(data.num_hyperedges)
        if edge_ids.min() < 0 or edge_ids.max() >= num_edges:
            raise RuntimeError(f"Noncontiguous hyperedge range for {dataset}")
        agreement, original_sizes = incidence_agreement(
            node_ids, edge_ids, labels, num_edges, num_classes
        )
        for budget in protocol["budgets"]:
            expected = max(1, int(float(budget) * node_ids.size))
            for method in protocol["methods"]:
                for seed in protocol["seeds"]:
                    path = mask_path(
                        protocol, str(dataset), str(method), float(budget), int(seed)
                    )
                    if not path.exists():
                        raise FileNotFoundError(path)
                    mask = load_mask(path).numpy().astype(bool, copy=False)
                    if mask.size != node_ids.size or int(mask.sum()) != expected:
                        raise RuntimeError(f"Exact-budget failure: {path}")
                    result = compute_mask_metrics(
                        node_ids=node_ids,
                        edge_ids=edge_ids,
                        labels=labels,
                        mask=mask,
                        agreement=agreement,
                        original_edge_sizes=original_sizes,
                        num_nodes=num_nodes,
                        num_edges=num_edges,
                        num_classes=num_classes,
                    )
                    rows.append({
                        "protocol_hash": file_hash(protocol_path),
                        "dataset_id": dataset,
                        "dataset": protocol["display_names"][dataset],
                        "budget": float(budget),
                        "method": method,
                        "seed": int(seed),
                        "mask_path": str(path.resolve()),
                        "mask_hash": file_hash(path),
                        "num_nodes": num_nodes,
                        "num_hyperedges": num_edges,
                        "num_original_incidences": int(node_ids.size),
                        **result,
                    })
        del data

    expected_rows = (
        len(protocol["datasets"])
        * len(protocol["budgets"])
        * len(protocol["methods"])
        * len(protocol["seeds"])
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Retained-structure matrix incomplete: {len(rows)} != {expected_rows}")
    write_csv(outdir / "seed_results.csv", rows)

    summary: List[Dict[str, object]] = []
    paired: List[Dict[str, object]] = []
    for dataset in protocol["datasets"]:
        for budget in protocol["budgets"]:
            cell = [
                row for row in rows
                if row["dataset_id"] == dataset and row["budget"] == float(budget)
            ]
            by_method_seed = {
                (str(row["method"]), int(row["seed"])): row for row in cell
            }
            for method in protocol["methods"]:
                group = [row for row in cell if row["method"] == method]
                result: Dict[str, object] = {
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": float(budget),
                    "method": method,
                    "n": len(group),
                }
                for metric in metrics:
                    result[f"{metric}_mean"], result[f"{metric}_std"] = mean_std(
                        [float(row[metric]) for row in group]
                    )
                summary.append(result)
            difference: Dict[str, object] = {
                "dataset_id": dataset,
                "dataset": protocol["display_names"][dataset],
                "budget": float(budget),
                "comparison": "EHGNN-F minus Random-Fixed",
                "n": len(protocol["seeds"]),
            }
            for metric in metrics:
                values = [
                    float(by_method_seed[("EHGNN-F", int(seed))][metric])
                    - float(by_method_seed[("Random-Fixed", int(seed))][metric])
                    for seed in protocol["seeds"]
                ]
                low, high = paired_ci(values)
                difference[f"{metric}_delta_mean"] = float(np.mean(values))
                difference[f"{metric}_delta_ci95_low"] = low
                difference[f"{metric}_delta_ci95_high"] = high
            paired.append(difference)
    write_csv(outdir / "method_summary.csv", summary)
    write_csv(outdir / "paired_differences.csv", paired)

    ten = [row for row in paired if math.isclose(float(row["budget"]), 0.1)]
    summary_lookup = {
        (row["dataset_id"], float(row["budget"]), row["method"]): row
        for row in summary
    }
    lines = [
        "# Retained structure at 10%",
        "",
        "Values are EHGNN-F minus paired Random-Fixed means. Coverage and budget values are percentage points; label agreement is on [0,1].",
        "",
        "| Dataset | EHGNN-F hyperedges <=1 | Random-Fixed hyperedges <=1 | Retained agreement delta | EHGNN-F singleton budget | Random-Fixed singleton budget | Relational coverage delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ten:
        learned = summary_lookup[(row["dataset_id"], 0.1, "EHGNN-F")]
        random = summary_lookup[(row["dataset_id"], 0.1, "Random-Fixed")]
        learned_low_cardinality = (
            float(learned["retained_zero_hyperedges_mean"])
            + float(learned["retained_singleton_hyperedges_mean"])
        )
        random_low_cardinality = (
            float(random["retained_zero_hyperedges_mean"])
            + float(random["retained_singleton_hyperedges_mean"])
        )
        lines.append(
            f"| {row['dataset']} | {learned_low_cardinality:.1f} "
            f"| {random_low_cardinality:.1f} "
            f"| {float(row['retained_agreement_delta_mean']):+.4f} "
            f"| {100*float(learned['singleton_budget_fraction_mean']):.2f}% "
            f"| {100*float(random['singleton_budget_fraction_mean']):.2f}% "
            f"| {100*float(row['relational_node_coverage_delta_mean']):+.2f} |"
        )
    (outdir / "summary_10pct.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--outdir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    if protocol.get("status") != "fixed":
        raise RuntimeError("Experiment specification is not fixed")
    outdir = Path(args.outdir or str(protocol["output_root"]))
    outdir.mkdir(parents=True, exist_ok=True)
    run_analysis(protocol, protocol_path, outdir)
    required = [outdir / name for name in protocol["required_outputs"] if name != "summary_metadata.json"]
    if any(not path.exists() for path in required):
        raise RuntimeError("At least one required output is missing")
    write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "outputs": {
            path.name: file_hash(path) for path in required
        },
    })


if __name__ == "__main__":
    main()
