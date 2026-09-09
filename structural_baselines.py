#!/usr/bin/env python3
"""Run the exact-budget Cardinality and Laplacian-proxy baselines.

Stages:
  prepare   Compute structural rankings and exact-budget masks.
  evaluate  Train the common HGNN on fixed masks and train Full once per seed.
  summarize Combine new runs with the fixed EHGNN-F/Random-Fixed evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from node_classification_utils import PRETTY_DATASET
from models_sparse import approximate_L_inv_diag_hutchinson_edgeindex
from main_accuracy import dataset_context, forward_diagnostics, training_kwargs
from training_utils import train_model


METHOD_TAGS = {
    "Degree-prefix": "degree_prefix",
    "Spectral-prefix": "spectral_prefix",
}


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_descending(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu()
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:
        indices = np.arange(values.numel())
        order = np.lexsort((indices, -values.numpy()))
        return torch.from_numpy(order.astype(np.int64, copy=False))


def edge_scores(
    data,
    method: str,
    protocol: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    edge_index = data.edge_index.to(device)
    v_idx, e_idx = edge_index
    num_edges = int(data.num_hyperedges)
    edge_degree = torch.bincount(e_idx, minlength=num_edges).float()
    if method == "Degree-prefix":
        return edge_degree.cpu()
    if method != "Spectral-prefix":
        raise ValueError(method)

    config = protocol["spectral_prefix"]
    torch.manual_seed(int(config["score_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["score_seed"]))
    diagonal, _, _ = approximate_L_inv_diag_hutchinson_edgeindex(
        v_idx,
        e_idx,
        int(data.n_x),
        num_edges,
        num_probes=int(config["num_probes"]),
        cg_tol=float(config["cg_tolerance"]),
        cg_max_iter=int(config["cg_max_iterations"]),
        reg=float(config["regularization"]),
    )
    scores = torch.zeros(num_edges, device=device)
    scores.index_add_(0, e_idx, diagonal[v_idx])
    scores[edge_degree <= 1] = 0.0
    if not torch.isfinite(scores).all():
        raise RuntimeError("Spectral-prefix produced non-finite hyperedge scores")
    return scores.cpu()


def prefix_mask(data, scores: torch.Tensor, keep_ratio: float) -> Tuple[torch.Tensor, Dict[str, int]]:
    e_idx = data.edge_index[1].detach().cpu()
    total = int(e_idx.numel())
    budget = max(1, int(keep_ratio * total))
    edge_order = stable_descending(scores)
    edge_rank = torch.empty(edge_order.numel(), dtype=torch.long)
    edge_rank[edge_order] = torch.arange(edge_order.numel(), dtype=torch.long)
    try:
        incidence_order = torch.argsort(edge_rank[e_idx], stable=True)
    except TypeError:
        incidence_order = torch.from_numpy(
            np.argsort(edge_rank[e_idx].numpy(), kind="stable").astype(np.int64, copy=False)
        )
    selected_indices = incidence_order[:budget]
    mask = torch.zeros(total, dtype=torch.bool)
    mask[selected_indices] = True
    if int(mask.sum()) != budget:
        raise RuntimeError(f"Exact budget failed: observed {int(mask.sum())}, expected {budget}")

    selected_counts = torch.bincount(e_idx[mask], minlength=edge_order.numel())
    edge_counts = torch.bincount(e_idx, minlength=edge_order.numel())
    complete = int(((selected_counts == edge_counts) & (edge_counts > 0)).sum())
    partial = int(((selected_counts > 0) & (selected_counts < edge_counts)).sum())
    return mask, {
        "budget": budget,
        "complete_hyperedges": complete,
        "partial_hyperedges": partial,
        "active_hyperedges": int((selected_counts > 0).sum()),
    }


def structural_diagnostics(data, mask: torch.Tensor) -> Dict[str, float]:
    v_idx = data.edge_index[0].detach().cpu()
    e_idx = data.edge_index[1].detach().cpu()
    selected_nodes = torch.zeros(int(data.n_x), dtype=torch.bool)
    selected_edges = torch.zeros(int(data.num_hyperedges), dtype=torch.bool)
    selected_nodes[v_idx[mask].unique()] = True
    selected_edges[e_idx[mask].unique()] = True
    return {
        "active_node_coverage": float(selected_nodes.float().mean()),
        "active_hyperedge_coverage": float(selected_edges.float().mean()),
    }


def prepare(args, protocol: Dict[str, object], device: torch.device) -> None:
    outdir = Path(args.outdir)
    datasets = list(args.datasets or protocol["datasets"])
    methods = list(args.methods or protocol["evaluation"]["methods"])
    budgets = [float(value) for value in protocol["evaluation"]["budgets"]]
    for dname in datasets:
        data, _, _, _ = dataset_context(dname, protocol)
        for method in methods:
            tag = METHOD_TAGS[method]
            score_path = outdir / "scores" / f"scores_{tag}_{dname}.npz"
            if score_path.exists() and not args.force:
                scores = torch.from_numpy(np.load(score_path)["scores"])
                print(f"[prepare:score-skip] {method} dataset={dname}", flush=True)
            else:
                print(f"[prepare:score] {method} dataset={dname}", flush=True)
                scores = edge_scores(data, method, protocol, device)
                atomic_save_npz(score_path, scores=scores.numpy())
            for ratio in budgets:
                ratio_tag = f"{int(round(100 * ratio)):03d}"
                mask_path = outdir / "masks" / f"mask_{tag}_{dname}_r{ratio_tag}.npz"
                metadata_path = outdir / "mask_metadata" / f"mask_{tag}_{dname}_r{ratio_tag}.json"
                if mask_path.exists() and metadata_path.exists() and not args.force:
                    print(f"[prepare:mask-skip] {method} dataset={dname} rho={ratio}", flush=True)
                    continue
                mask, details = prefix_mask(data, scores, ratio)
                atomic_save_npz(mask_path, mask=mask.numpy())
                atomic_write_json(metadata_path, {
                    "created_at": timestamp(),
                    "protocol_hash": protocol_hash(Path(args.protocol)),
                    "dataset_id": dname,
                    "dataset": PRETTY_DATASET.get(dname, dname),
                    "method": method,
                    "keep_ratio": ratio,
                    "num_selected": int(mask.sum()),
                    "num_mask_units": int(mask.numel()),
                    "structural_density": float(mask.float().mean()),
                    **details,
                    **structural_diagnostics(data, mask),
                })
                print(f"[prepare:mask] {method} dataset={dname} rho={ratio} K={int(mask.sum())}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate(args, protocol: Dict[str, object], device: torch.device) -> None:
    outdir = Path(args.outdir)
    datasets = list(args.datasets or protocol["datasets"])
    methods = list(args.methods or protocol["evaluation"]["methods"])
    seeds = list(args.seed_values or protocol["evaluation"]["seeds"])
    budgets = [float(value) for value in protocol["evaluation"]["budgets"]]
    train_args = training_kwargs(protocol)
    result_dir = outdir / "evaluation_runs"
    result_dir.mkdir(parents=True, exist_ok=True)

    for dname in datasets:
        data, num_features, num_classes, split_idx = dataset_context(dname, protocol)
        for seed in seeds:
            if args.include_full:
                full_path = result_dir / f"eval_full_{dname}_s{seed}.json"
                if not full_path.exists() or args.force:
                    print(f"[evaluate] Full dataset={dname} seed={seed}", flush=True)
                    model, metrics = train_model(
                        data=data,
                        split_idx=split_idx,
                        mode="full",
                        num_features=num_features,
                        num_classes=num_classes,
                        keep_ratio=1.0,
                        seed=int(seed),
                        device=device,
                        **train_args,
                    )
                    atomic_write_json(full_path, {
                        "stage": "baseline_extension_test_evaluation",
                        "created_at": timestamp(),
                        "protocol_hash": protocol_hash(Path(args.protocol)),
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "seed": int(seed),
                        "method": "Full",
                        "unit": "incidence",
                        "keep_ratio": 1.0,
                        "test_acc": 100.0 * float(metrics["test_acc"]),
                        "val_acc": 100.0 * float(metrics["val_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_run": int(metrics["epochs_run"]),
                        "num_selected": int(data.edge_index.size(1)),
                        "num_mask_units": int(data.edge_index.size(1)),
                        **forward_diagnostics(model, data, int(data.edge_index.size(1))),
                    })
                    del model
                else:
                    print(f"[evaluate:skip] Full dataset={dname} seed={seed}", flush=True)

            for method in methods:
                tag = METHOD_TAGS[method]
                for ratio in budgets:
                    ratio_tag = f"{int(round(100 * ratio)):03d}"
                    result_path = result_dir / f"eval_{tag}_{dname}_r{ratio_tag}_s{seed}.json"
                    if result_path.exists() and not args.force:
                        print(f"[evaluate:skip] {method} dataset={dname} rho={ratio} seed={seed}", flush=True)
                        continue
                    mask_path = outdir / "masks" / f"mask_{tag}_{dname}_r{ratio_tag}.npz"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Prepare masks first: {mask_path}")
                    mask = torch.from_numpy(np.load(mask_path)["mask"]).bool()
                    expected = max(1, int(ratio * data.edge_index.size(1)))
                    if int(mask.sum()) != expected:
                        raise RuntimeError(
                            f"Stored budget mismatch for {method}/{dname}/{ratio}: "
                            f"{int(mask.sum())} != {expected}"
                        )
                    sparse_data = copy.deepcopy(data)
                    sparse_data.edge_index = data.edge_index[:, mask]
                    print(f"[evaluate] {method} dataset={dname} rho={ratio} seed={seed}", flush=True)
                    model, metrics = train_model(
                        data=sparse_data,
                        split_idx=split_idx,
                        mode="random",
                        num_features=num_features,
                        num_classes=num_classes,
                        keep_ratio=ratio,
                        seed=int(seed),
                        device=device,
                        **train_args,
                    )
                    atomic_write_json(result_path, {
                        "stage": "baseline_extension_test_evaluation",
                        "created_at": timestamp(),
                        "protocol_hash": protocol_hash(Path(args.protocol)),
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "seed": int(seed),
                        "method": method,
                        "unit": "incidence",
                        "keep_ratio": ratio,
                        "test_acc": 100.0 * float(metrics["test_acc"]),
                        "val_acc": 100.0 * float(metrics["val_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_run": int(metrics["epochs_run"]),
                        "num_selected": int(mask.sum()),
                        "num_mask_units": int(mask.numel()),
                        "mask_path": str(mask_path.resolve()),
                        **structural_diagnostics(data, mask),
                        **forward_diagnostics(model, data, int(mask.sum())),
                    })
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_ci(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    radius = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return mean - radius, mean + radius


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def load_records(paths: Iterable[Path]) -> List[Dict[str, object]]:
    return [read_json(path) for path in sorted(paths)]


def summarize(args, protocol: Dict[str, object]) -> None:
    outdir = Path(args.outdir)
    new_records = load_records((outdir / "evaluation_runs").glob("eval_*.json"))
    parent_records = load_records(Path(args.parent_results).glob("eval_*.json"))
    datasets = list(protocol["datasets"])
    budgets = [float(value) for value in protocol["evaluation"]["budgets"]]
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    new_methods = list(protocol["evaluation"]["methods"])

    lookup: Dict[Tuple[str, float, int, str], float] = {}
    full_lookup: Dict[Tuple[str, int], float] = {}
    for row in new_records:
        method = str(row["method"])
        if method == "Full":
            full_lookup[(str(row["dataset_id"]), int(row["seed"]))] = float(row["test_acc"])
        else:
            lookup[(str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), method)] = float(row["test_acc"])
    if args.full_results:
        for row in load_records(Path(args.full_results).glob("run_full_*.json")):
            key = (str(row["dataset_id"]), int(row["seed"]))
            full_lookup.setdefault(key, float(row["test_accuracy"]))
    for row in parent_records:
        method = str(row["method"])
        if method in {"EHGNN-F", "Random-Fixed"}:
            lookup[(str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), method)] = float(row["test_acc"])

    expected = {
        (dname, ratio, seed, method)
        for dname in datasets
        for ratio in budgets
        for seed in seeds
        for method in ["EHGNN-F", "Random-Fixed", *new_methods]
    }
    missing = sorted(expected - set(lookup))
    missing_full = sorted({(dname, seed) for dname in datasets for seed in seeds} - set(full_lookup))
    if missing or missing_full:
        raise RuntimeError(
            f"Incomplete result matrix: missing={len(missing)}, missing_full={len(missing_full)}; "
            f"examples={missing[:3] + missing_full[:3]}"
        )

    summary_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    for dname in datasets:
        full_values = [full_lookup[(dname, seed)] for seed in seeds]
        full_mean, full_std = mean_std(full_values)
        for ratio in budgets:
            row: Dict[str, object] = {
                "dataset_id": dname,
                "dataset": PRETTY_DATASET.get(dname, dname),
                "keep_ratio": ratio,
                "Full_mean": full_mean,
                "Full_std": full_std,
            }
            for method in ["EHGNN-F", "Random-Fixed", *new_methods]:
                values = [lookup[(dname, ratio, seed, method)] for seed in seeds]
                mean, std = mean_std(values)
                row[f"{method}_mean"] = mean
                row[f"{method}_std"] = std
            summary_rows.append(row)

            learned = np.asarray(
                [lookup[(dname, ratio, seed, "EHGNN-F")] for seed in seeds], dtype=float
            )
            for baseline in new_methods:
                base = np.asarray(
                    [lookup[(dname, ratio, seed, baseline)] for seed in seeds], dtype=float
                )
                differences = learned - base
                low, high = paired_ci(differences)
                comparison_rows.append({
                    "dataset_id": dname,
                    "dataset": PRETTY_DATASET.get(dname, dname),
                    "keep_ratio": ratio,
                    "baseline": baseline,
                    "paired_delta_mean": float(differences.mean()),
                    "paired_delta_ci95_low": low,
                    "paired_delta_ci95_high": high,
                    "wilcoxon_p_raw": (
                        1.0 if np.allclose(differences, 0.0)
                        else float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
                    ),
                })

    adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in comparison_rows])
    for row, value in zip(comparison_rows, adjusted):
        row["wilcoxon_p_holm_32"] = value

    write_csv(outdir / "all_method_budget_summary.csv", summary_rows)
    write_csv(outdir / "new_baseline_paired_comparisons.csv", comparison_rows)
    atomic_write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_hash": protocol_hash(Path(args.protocol)),
        "new_comparison_family_size": len(comparison_rows),
        "number_of_new_records": len(new_records),
        "number_of_parent_records": len(parent_records),
    })
    print(f"[summarize] wrote results to {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "evaluate", "summarize"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--parent-results")
    parser.add_argument("--full-results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seed-values", nargs="+", type=int)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "summarize" and not args.parent_results:
        parser.error("--parent-results is required for summarize")
    return args


def main() -> None:
    args = parse_args()
    protocol = read_json(Path(args.protocol))
    device = torch.device(
        args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    )
    if args.stage == "prepare":
        prepare(args, protocol, device)
    elif args.stage == "evaluate":
        evaluate(args, protocol, device)
    else:
        summarize(args, protocol)


if __name__ == "__main__":
    main()
