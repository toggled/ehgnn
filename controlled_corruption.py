#!/usr/bin/env python3
"""Paired controlled-corruption study with exact incidence budgets.

Stages:
  prepare    Build deterministic corruption traces and fixed baseline masks.
  evaluate   Train one or more methods on prepared traces.
  summarize  Require the complete result matrix and write statistical reports.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from structural_baselines import (
    atomic_save_npz,
    atomic_write_json,
    edge_scores,
    prefix_mask,
    protocol_hash,
    read_json,
    structural_diagnostics,
)
from node_classification_utils import train_learned_with_trajectory
from training_utils import train_model
from main_accuracy import dataset_context, forward_diagnostics, training_kwargs


PRETTY_DATASET = {
    "cora": "Cora",
    "actor": "Actor",
    "pokec": "Pokec",
}

METHOD_TAGS = {
    "Full": "full",
    "EHGNN-F": "ehgnnf",
    "Random-Fixed": "random",
    "Degree-prefix": "degree_prefix",
    "Spectral-prefix": "spectral_prefix",
}

FIXED_MASK_METHODS = {
    "Random-Fixed",
    "Degree-prefix",
    "Spectral-prefix",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def cell_id(eta: float, alpha: float) -> str:
    return f"eta{int(round(100 * eta)):03d}_a{int(round(100 * alpha)):03d}"


def protocol_cells(protocol: Dict[str, object]) -> List[Dict[str, object]]:
    corruption = protocol["corruption"]
    rate = corruption["rate_curve"]
    concentration = corruption["concentration_slice"]
    cells: List[Dict[str, object]] = []
    seen = set()
    for eta in rate["etas"]:
        key = (float(eta), float(rate["alpha"]))
        seen.add(key)
        cells.append({
            "cell_id": cell_id(*key),
            "eta": key[0],
            "alpha": key[1],
            "cell_role": "concentrated_rate_curve",
        })
    for alpha in concentration["alphas"]:
        key = (float(concentration["eta"]), float(alpha))
        if key in seen:
            continue
        cells.append({
            "cell_id": cell_id(*key),
            "eta": key[0],
            "alpha": key[1],
            "cell_role": "concentration_slice",
        })
    expected = int(corruption["unique_cell_count_per_dataset"])
    if len(cells) != expected:
        raise RuntimeError(f"Protocol defines {len(cells)} cells, expected {expected}")
    return cells


def resolve_cells(protocol: Dict[str, object], requested: Optional[Sequence[str]]) -> List[Dict[str, object]]:
    cells = protocol_cells(protocol)
    if requested is None:
        return cells
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    unknown = set(requested) - set(by_id)
    if unknown:
        raise ValueError(f"Unknown cells: {sorted(unknown)}")
    return [by_id[value] for value in requested]


def trace_path(outdir: Path, dname: str, cell: str, seed: int) -> Path:
    return outdir / "corruption_traces" / f"trace_{dname}_{cell}_s{seed}.npz"


def trace_metadata_path(outdir: Path, dname: str, cell: str, seed: int) -> Path:
    return outdir / "corruption_metadata" / f"trace_{dname}_{cell}_s{seed}.json"


def mask_path(outdir: Path, method: str, dname: str, cell: str, seed: int) -> Path:
    return outdir / "masks" / f"mask_{METHOD_TAGS[method]}_{dname}_{cell}_s{seed}.npz"


def mask_metadata_path(outdir: Path, method: str, dname: str, cell: str, seed: int) -> Path:
    return outdir / "mask_metadata" / f"mask_{METHOD_TAGS[method]}_{dname}_{cell}_s{seed}.json"


def result_path(outdir: Path, method: str, dname: str, cell: str, seed: int) -> Path:
    return outdir / "evaluation_runs" / f"eval_{METHOD_TAGS[method]}_{dname}_{cell}_s{seed}.json"


def dataset_seed(protocol: Dict[str, object], dname: str, trial: int, key: str) -> int:
    datasets = list(protocol["datasets"])
    offset = int(protocol[key]) if key in protocol else int(protocol["corruption"][key])
    return offset + 10000 * datasets.index(dname) + int(trial)


def class_balanced_replacements(
    data,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Assign one unique, non-member replacement to every incidence.

    Replacement labels are balanced within each hyperedge. This prevents a
    fully rewired binary-class edge from becoming a new pure edge of the
    opposite class, which would not be a reliable harmful intervention.
    """
    nodes = data.edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    edges = data.edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    labels = data.y.detach().cpu().numpy().astype(np.int64, copy=False)
    num_nodes = int(data.n_x)
    classes = np.unique(labels)
    class_nodes = {int(c): np.flatnonzero(labels == c).astype(np.int64) for c in classes}
    replacements = np.empty_like(nodes)
    fallback_count = 0

    incidence_order = np.argsort(edges, kind="stable")
    ordered_edges = edges[incidence_order]
    boundaries = np.flatnonzero(np.diff(ordered_edges)) + 1
    position_groups = np.split(incidence_order, boundaries)

    for positions in position_groups:
        if positions.size == 0:
            continue
        edge = int(edges[positions[0]])
        original = set(int(value) for value in nodes[positions])
        used = set()
        target_classes: List[int] = []
        while len(target_classes) < positions.size:
            target_classes.extend(int(value) for value in rng.permutation(classes))
        target_classes = target_classes[: positions.size]
        rng.shuffle(target_classes)

        for position, target_class in zip(positions, target_classes):
            pool = class_nodes[target_class]
            candidate = -1
            for _ in range(128):
                value = int(pool[int(rng.integers(0, pool.size))])
                if value not in original and value not in used:
                    candidate = value
                    break
            if candidate < 0:
                for value in pool:
                    value = int(value)
                    if value not in original and value not in used:
                        candidate = value
                        break
            if candidate < 0:
                fallback_count += 1
                for value in rng.permutation(num_nodes):
                    value = int(value)
                    if value not in original and value not in used:
                        candidate = value
                        break
            if candidate < 0:
                raise RuntimeError(f"No unique replacement for hyperedge {edge}")
            replacements[position] = candidate
            used.add(candidate)

    if np.any(replacements == nodes):
        raise RuntimeError("Replacement map contains unchanged incidence nodes")
    return replacements, {"replacement_fallback_count": fallback_count}


def trial_latents(
    data,
    protocol: Dict[str, object],
    dname: str,
    trial: int,
) -> Dict[str, object]:
    seed = dataset_seed(protocol, dname, trial, "corruption_seed_offset")
    rng = np.random.default_rng(seed)
    total = int(data.edge_index.size(1))
    num_edges = int(data.num_hyperedges)
    edge_priority = rng.random(num_edges)
    incidence_priority = rng.random(total)
    replacements, diagnostics = class_balanced_replacements(data, rng)
    return {
        "seed": seed,
        "edge_priority": edge_priority,
        "incidence_priority": incidence_priority,
        "replacement_nodes": replacements,
        **diagnostics,
    }


def corruption_order(data, latents: Dict[str, object], alpha: float) -> np.ndarray:
    edges = data.edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    edge_priority = np.asarray(latents["edge_priority"], dtype=float)
    incidence_priority = np.asarray(latents["incidence_priority"], dtype=float)
    if math.isclose(alpha, 1.0):
        return np.lexsort((-incidence_priority, -edge_priority[edges])).astype(np.int64)
    if math.isclose(alpha, 0.0):
        return np.argsort(-incidence_priority, kind="stable").astype(np.int64)
    scores = alpha * edge_priority[edges] + (1.0 - alpha) * incidence_priority
    return np.lexsort((np.arange(scores.size), -scores)).astype(np.int64)


def incidence_weighted_purity(node_ids: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray) -> float:
    num_classes = int(labels.max()) + 1
    num_edges = int(edge_ids.max()) + 1
    edge_class = edge_ids.astype(np.int64) * num_classes + labels[node_ids].astype(np.int64)
    pairs, counts = np.unique(edge_class, return_counts=True)
    majority = np.zeros(num_edges, dtype=np.int64)
    np.maximum.at(majority, pairs // num_classes, counts)
    return float(majority.sum() / max(1, node_ids.size))


def corruption_diagnostics(
    data,
    node_ids: np.ndarray,
    corrupt_mask: np.ndarray,
    replacement_fallback_count: int,
) -> Dict[str, object]:
    base_nodes = data.edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    edge_ids = data.edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    labels = data.y.detach().cpu().numpy().astype(np.int64, copy=False)
    total = int(base_nodes.size)
    num_edges = int(data.num_hyperedges)
    edge_sizes = np.bincount(edge_ids, minlength=num_edges)
    corrupt_counts = np.bincount(edge_ids[corrupt_mask], minlength=num_edges)
    carriers = corrupt_counts > 0
    corrupted = int(corrupt_mask.sum())
    proportions = corrupt_counts[carriers] / max(1, corrupted)
    hhi = float(np.square(proportions).sum()) if corrupted else 0.0
    pair_ids = edge_ids.astype(np.int64) * int(data.n_x) + node_ids.astype(np.int64)
    duplicate_count = int(pair_ids.size - np.unique(pair_ids).size)
    changed = int((node_ids != base_nodes).sum())
    expected_changed = corrupted
    if changed != expected_changed:
        raise RuntimeError(f"Changed-node audit failed: {changed} != {expected_changed}")
    if duplicate_count:
        raise RuntimeError(f"Corruption created {duplicate_count} duplicate incidences")
    return {
        "num_incidences": total,
        "num_hyperedges": num_edges,
        "num_corrupted": corrupted,
        "realized_corruption_rate": corrupted / total,
        "carrier_hyperedges": int(carriers.sum()),
        "carrier_hyperedge_fraction": float(carriers.mean()),
        "mean_carrier_severity": (
            float((corrupt_counts[carriers] / edge_sizes[carriers]).mean()) if carriers.any() else 0.0
        ),
        "fully_corrupted_hyperedges": int(((corrupt_counts == edge_sizes) & carriers).sum()),
        "corruption_count_hhi": hhi,
        "effective_carrier_count": 1.0 / hhi if hhi > 0.0 else 0.0,
        "base_incidence_weighted_label_purity": incidence_weighted_purity(base_nodes, edge_ids, labels),
        "corrupt_incidence_weighted_label_purity": incidence_weighted_purity(node_ids, edge_ids, labels),
        "changed_node_count": changed,
        "duplicate_incidence_count": duplicate_count,
        "replacement_fallback_count": int(replacement_fallback_count),
    }


def build_corruption(
    data,
    latents: Dict[str, object],
    eta: float,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    base_nodes = data.edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    total = int(base_nodes.size)
    count = int(math.floor(eta * total))
    order = corruption_order(data, latents, alpha)
    corrupt_mask = np.zeros(total, dtype=bool)
    corrupt_mask[order[:count]] = True
    node_ids = base_nodes.copy()
    replacements = np.asarray(latents["replacement_nodes"], dtype=np.int64)
    node_ids[corrupt_mask] = replacements[corrupt_mask]
    diagnostics = corruption_diagnostics(
        data,
        node_ids,
        corrupt_mask,
        int(latents["replacement_fallback_count"]),
    )
    if diagnostics["num_corrupted"] != count:
        raise RuntimeError("Exact corruption count failed")
    return node_ids, corrupt_mask, diagnostics


def corrupted_data_from_arrays(data, node_ids: np.ndarray):
    out = copy.deepcopy(data)
    out.edge_index = torch.stack([
        torch.from_numpy(node_ids.astype(np.int64, copy=False)),
        data.edge_index[1].detach().cpu().long(),
    ])
    out.num_hyperedges = int(data.num_hyperedges)
    out.n_x = int(data.n_x)
    return out


def load_corrupted_data(data, path: Path):
    arrays = np.load(path)
    node_ids = arrays["node_ids"].astype(np.int64, copy=False)
    corrupt_mask = arrays["corrupt_mask"].astype(bool, copy=False)
    return corrupted_data_from_arrays(data, node_ids), torch.from_numpy(corrupt_mask).bool()


def fixed_random_order(protocol: Dict[str, object], dname: str, trial: int, total: int) -> np.ndarray:
    offset = int(protocol["evaluation"]["random_mask_seed_offset"])
    dataset_index = list(protocol["datasets"]).index(dname)
    rng = np.random.default_rng(offset + 10000 * dataset_index + int(trial))
    return rng.permutation(total).astype(np.int64)


def exact_mask_from_order(order: np.ndarray, budget: int, total: int) -> torch.Tensor:
    mask = torch.zeros(total, dtype=torch.bool)
    mask[torch.from_numpy(order[:budget]).long()] = True
    if int(mask.sum()) != budget:
        raise RuntimeError(f"Exact mask budget failed: {int(mask.sum())} != {budget}")
    return mask


def prepare_mask(
    method: str,
    data,
    random_order: np.ndarray,
    protocol: Dict[str, object],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    total = int(data.edge_index.size(1))
    budget = max(1, int(float(protocol["evaluation"]["keep_ratio"]) * total))
    details: Dict[str, object] = {"budget": budget}
    if method == "Random-Fixed":
        mask = exact_mask_from_order(random_order, budget, total)
    elif method in {"Degree-prefix", "Spectral-prefix"}:
        scores = edge_scores(data, method, protocol, device)
        mask, prefix_details = prefix_mask(data, scores, float(protocol["evaluation"]["keep_ratio"]))
        details.update(prefix_details)
    else:
        raise ValueError(method)
    return mask, {**details, **structural_diagnostics(data, mask)}


def prepare(args: argparse.Namespace, protocol: Dict[str, object], device: torch.device) -> None:
    outdir = Path(args.outdir)
    datasets = list(args.datasets or protocol["datasets"])
    seeds = list(args.seed_values or protocol["evaluation"]["seeds"])
    cells = resolve_cells(protocol, args.cell_ids)
    requested_methods = list(args.methods or protocol["evaluation"]["methods"])
    methods = [method for method in requested_methods if method in FIXED_MASK_METHODS]
    p_hash = protocol_hash(Path(args.protocol))

    for dname in datasets:
        data, _, _, _ = dataset_context(dname, protocol)
        total = int(data.edge_index.size(1))
        for seed in seeds:
            latents = trial_latents(data, protocol, dname, int(seed))
            concentrated_masks: List[Tuple[float, np.ndarray]] = []
            random_order = fixed_random_order(protocol, dname, int(seed), total)
            for cell in cells:
                cid = str(cell["cell_id"])
                eta = float(cell["eta"])
                alpha = float(cell["alpha"])
                t_path = trace_path(outdir, dname, cid, int(seed))
                m_path = trace_metadata_path(outdir, dname, cid, int(seed))
                if t_path.exists() and m_path.exists() and not args.force:
                    arrays = np.load(t_path)
                    node_ids = arrays["node_ids"].astype(np.int64, copy=False)
                    corrupt_np = arrays["corrupt_mask"].astype(bool, copy=False)
                    corrupted_data = corrupted_data_from_arrays(data, node_ids)
                    diagnostics = read_json(m_path)
                    print(f"[prepare:trace-skip] {dname} {cid} seed={seed}", flush=True)
                else:
                    node_ids, corrupt_np, diagnostics = build_corruption(data, latents, eta, alpha)
                    corrupted_data = corrupted_data_from_arrays(data, node_ids)
                    atomic_save_npz(t_path, node_ids=node_ids, corrupt_mask=corrupt_np)
                    atomic_write_json(m_path, {
                        "stage": "controlled_corruption_trace",
                        "created_at": timestamp(),
                        "protocol_hash": p_hash,
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "trial_seed": int(seed),
                        "corruption_seed": int(latents["seed"]),
                        **cell,
                        **diagnostics,
                    })
                    print(
                        f"[prepare:trace] {dname} {cid} seed={seed} "
                        f"B={int(corrupt_np.sum())} carriers={diagnostics['carrier_hyperedges']}",
                        flush=True,
                    )
                if math.isclose(alpha, 1.0):
                    concentrated_masks.append((eta, corrupt_np.copy()))

                corrupt_mask = torch.from_numpy(corrupt_np).bool()
                for method in methods:
                    path = mask_path(outdir, method, dname, cid, int(seed))
                    metadata_path = mask_metadata_path(outdir, method, dname, cid, int(seed))
                    if path.exists() and metadata_path.exists() and not args.force:
                        print(f"[prepare:mask-skip] {method} {dname} {cid} seed={seed}", flush=True)
                        continue
                    mask, details = prepare_mask(
                        method,
                        corrupted_data,
                        random_order,
                        protocol,
                        device,
                    )
                    atomic_save_npz(path, mask=mask.numpy())
                    quality = mask_quality(mask, corrupt_mask)
                    atomic_write_json(metadata_path, {
                        "stage": "controlled_corruption_fixed_mask",
                        "created_at": timestamp(),
                        "protocol_hash": p_hash,
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "trial_seed": int(seed),
                        "method": method,
                        **cell,
                        "num_selected": int(mask.sum()),
                        "num_mask_units": int(mask.numel()),
                        **details,
                        **quality,
                    })
                    print(f"[prepare:mask] {method} {dname} {cid} seed={seed}", flush=True)

            concentrated_masks.sort(key=lambda value: value[0])
            for (_, lower), (_, upper) in zip(concentrated_masks, concentrated_masks[1:]):
                if np.any(lower & ~upper):
                    raise RuntimeError(f"Non-nested concentrated traces for {dname}, seed {seed}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


def mask_quality(selected: torch.Tensor, corrupted: torch.Tensor) -> Dict[str, float]:
    selected = selected.detach().cpu().bool()
    corrupted = corrupted.detach().cpu().bool()
    corrupt_count = int(corrupted.sum())
    selected_count = int(selected.sum())
    selected_corrupt = int((selected & corrupted).sum())
    return {
        "corrupted_incidence_retention_rate": (
            selected_corrupt / corrupt_count if corrupt_count else float("nan")
        ),
        "selected_corruption_fraction": selected_corrupt / max(1, selected_count),
        "selected_corrupted_count": selected_corrupt,
    }


def evaluate(args: argparse.Namespace, protocol: Dict[str, object], device: torch.device) -> None:
    outdir = Path(args.outdir)
    datasets = list(args.datasets or protocol["datasets"])
    seeds = list(args.seed_values or protocol["evaluation"]["seeds"])
    cells = resolve_cells(protocol, args.cell_ids)
    methods = list(args.methods or protocol["evaluation"]["methods"])
    unknown = set(methods) - set(METHOD_TAGS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    train_args = training_kwargs(protocol)
    keep_ratio = float(protocol["evaluation"]["keep_ratio"])
    ehgnn = protocol["ehgnn_f"]
    p_hash = protocol_hash(Path(args.protocol))

    for dname in datasets:
        base_data, num_features, num_classes, split_idx = dataset_context(dname, protocol)
        total = int(base_data.edge_index.size(1))
        budget = max(1, int(keep_ratio * total))
        for seed in seeds:
            for cell in cells:
                cid = str(cell["cell_id"])
                t_path = trace_path(outdir, dname, cid, int(seed))
                metadata_path = trace_metadata_path(outdir, dname, cid, int(seed))
                if not t_path.exists() or not metadata_path.exists():
                    raise FileNotFoundError(f"Prepare corruption trace first: {t_path}")
                data, corrupt_mask = load_corrupted_data(base_data, t_path)
                corruption_metadata = read_json(metadata_path)

                for method in methods:
                    output = result_path(outdir, method, dname, cid, int(seed))
                    if output.exists() and not args.force:
                        print(f"[evaluate:skip] {method} {dname} {cid} seed={seed}", flush=True)
                        continue
                    print(f"[evaluate] {method} {dname} {cid} seed={seed}", flush=True)
                    stored_mask_path: Optional[Path] = None
                    if method == "EHGNN-F":
                        model, metrics, _, _, selected, _, _ = train_learned_with_trajectory(
                            data=data,
                            split_idx=split_idx,
                            mode=str(ehgnn["mode"]),
                            num_features=num_features,
                            num_classes=num_classes,
                            keep_ratio=keep_ratio,
                            seed=int(seed),
                            device=device,
                            trajectory_every=50,
                            mask_init_std=float(ehgnn["mask_init_std"]),
                            mask_lr_multiplier=float(ehgnn["mask_lr_multiplier"]),
                            mask_gradient_estimator=str(ehgnn["mask_gradient_estimator"]),
                            include_test_metrics=True,
                            **train_args,
                        )
                    elif method == "Full":
                        selected = torch.ones(total, dtype=torch.bool)
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
                    else:
                        stored_mask_path = mask_path(outdir, method, dname, cid, int(seed))
                        if not stored_mask_path.exists():
                            raise FileNotFoundError(f"Prepare fixed mask first: {stored_mask_path}")
                        selected = torch.from_numpy(np.load(stored_mask_path)["mask"]).bool()
                        sparse_data = copy.deepcopy(data)
                        sparse_data.edge_index = data.edge_index[:, selected]
                        model, metrics = train_model(
                            data=sparse_data,
                            split_idx=split_idx,
                            mode="random",
                            num_features=num_features,
                            num_classes=num_classes,
                            keep_ratio=keep_ratio,
                            seed=int(seed),
                            device=device,
                            **train_args,
                        )

                    expected = total if method == "Full" else budget
                    if int(selected.sum()) != expected or int(selected.numel()) != total:
                        raise RuntimeError(
                            f"Budget failure for {method}/{dname}/{cid}/s{seed}: "
                            f"{int(selected.sum())}/{int(selected.numel())}, expected {expected}/{total}"
                        )
                    diagnostics = forward_diagnostics(model, data, int(selected.sum()))
                    expected_self_loops = int(data.n_x)
                    if int(diagnostics["num_fixed_self_loops"]) != expected_self_loops:
                        raise RuntimeError(
                            f"Fixed self-loop failure for {method}/{dname}/{cid}/s{seed}: "
                            f"{diagnostics['num_fixed_self_loops']} != {expected_self_loops}"
                        )
                    row: Dict[str, object] = {
                        "stage": "controlled_corruption_test_evaluation",
                        "created_at": timestamp(),
                        "protocol_hash": p_hash,
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "trial_seed": int(seed),
                        "method": method,
                        **cell,
                        "test_acc": 100.0 * float(metrics["test_acc"]),
                        "val_acc": 100.0 * float(metrics["val_acc"]),
                        "train_acc": 100.0 * float(metrics["train_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_run": int(metrics["epochs_run"]),
                        "num_selected": int(selected.sum()),
                        "num_mask_units": int(selected.numel()),
                        "structural_density": float(selected.float().mean()),
                        "num_corrupted": int(corruption_metadata["num_corrupted"]),
                        "carrier_hyperedges": int(corruption_metadata["carrier_hyperedges"]),
                        "mask_path": str(stored_mask_path.resolve()) if stored_mask_path else None,
                        "trace_path": str(t_path.resolve()),
                        **structural_diagnostics(data, selected),
                        **mask_quality(selected, corrupt_mask),
                        **diagnostics,
                    }
                    atomic_write_json(output, row)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def paired_ci(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size < 2 or np.allclose(array, array[0]):
        return mean, mean
    radius = float(stats.t.ppf(0.975, array.size - 1) * stats.sem(array))
    return mean - radius, mean + radius


def wilcoxon_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.allclose(array, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(array, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


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


def summarize(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    outdir = Path(args.outdir)
    datasets = list(protocol["datasets"])
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    methods = list(protocol["evaluation"]["methods"])
    cells = protocol_cells(protocol)
    records = load_records((outdir / "evaluation_runs").glob("eval_*.json"))
    lookup = {
        (str(row["dataset_id"]), str(row["cell_id"]), int(row["trial_seed"]), str(row["method"])): row
        for row in records
    }
    expected = {
        (dname, str(cell["cell_id"]), seed, method)
        for dname in datasets
        for cell in cells
        for seed in seeds
        for method in methods
    }
    missing = sorted(expected - set(lookup))
    if missing:
        raise RuntimeError(f"Incomplete result matrix: missing={len(missing)}, examples={missing[:5]}")

    summary_rows: List[Dict[str, object]] = []
    mechanism_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    comparison_baseline = "Random-Fixed"

    for dname in datasets:
        for cell in cells:
            cid = str(cell["cell_id"])
            summary: Dict[str, object] = {
                "dataset_id": dname,
                "dataset": PRETTY_DATASET.get(dname, dname),
                **cell,
            }
            for method in methods:
                group = [lookup[(dname, cid, seed, method)] for seed in seeds]
                accuracy = [float(row["test_acc"]) for row in group]
                mean, std = mean_std(accuracy)
                summary[f"{method}_mean"] = mean
                summary[f"{method}_std"] = std
                if method in {"EHGNN-F", "Random-Fixed"}:
                    for metric in (
                        "corrupted_incidence_retention_rate",
                        "selected_corruption_fraction",
                    ):
                        values = np.asarray([float(row[metric]) for row in group], dtype=float)
                        finite = values[np.isfinite(values)]
                        if finite.size:
                            mechanism_rows.append({
                                "dataset_id": dname,
                                "dataset": PRETTY_DATASET.get(dname, dname),
                                **cell,
                                "method": method,
                                "metric": metric,
                                "mean": float(finite.mean()),
                                "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                                "n": int(finite.size),
                            })
            summary_rows.append(summary)

            learned = np.asarray([
                float(lookup[(dname, cid, seed, "EHGNN-F")]["test_acc"]) for seed in seeds
            ])
            base = np.asarray([
                float(lookup[(dname, cid, seed, comparison_baseline)]["test_acc"])
                for seed in seeds
            ])
            differences = learned - base
            low, high = paired_ci(differences)
            comparison_rows.append({
                "dataset_id": dname,
                "dataset": PRETTY_DATASET.get(dname, dname),
                **cell,
                "baseline": comparison_baseline,
                "paired_delta_mean": float(differences.mean()),
                "paired_delta_ci95_low": low,
                "paired_delta_ci95_high": high,
                "wins": int((differences > 0).sum()),
                "ties": int(np.isclose(differences, 0.0).sum()),
                "losses": int((differences < 0).sum()),
                "wilcoxon_p_raw": wilcoxon_p(differences),
            })

    adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in comparison_rows])
    for row, value in zip(comparison_rows, adjusted):
        row["wilcoxon_p_holm_24"] = value

    write_csv(outdir / "all_method_cell_summary.csv", summary_rows)
    write_csv(outdir / "paired_comparisons.csv", comparison_rows)
    write_csv(outdir / "mask_mechanism_summary.csv", mechanism_rows)
    plot_rate_curves(outdir, protocol, summary_rows)
    write_summary_markdown(outdir, summary_rows, comparison_rows)
    atomic_write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_hash": protocol_hash(Path(args.protocol)),
        "number_of_evaluation_records": len(records),
        "expected_evaluation_records": len(expected),
        "primary_comparison_family_size": 24,
        "status": "complete",
    })
    print(f"[summarize] complete results written to {outdir}", flush=True)


def plot_rate_curves(
    outdir: Path,
    protocol: Dict[str, object],
    summaries: Sequence[Dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        "Full",
        "EHGNN-F",
        "Random-Fixed",
        "Degree-prefix",
        "Spectral-prefix",
    ]
    labels = {
        "Full": "Full",
        "EHGNN-F": "EHGNN-F",
        "Random-Fixed": "Random-Fixed",
        "Degree-prefix": "Cardinality",
        "Spectral-prefix": "Laplacian-proxy",
    }
    colors = {
        "Full": "#4D4D4D",
        "EHGNN-F": "#D62728",
        "Random-Fixed": "#1F77B4",
        "Degree-prefix": "#2CA02C",
        "Spectral-prefix": "#9467BD",
    }
    markers = {
        "Full": "o",
        "EHGNN-F": "s",
        "Random-Fixed": "^",
        "Degree-prefix": "D",
        "Spectral-prefix": "v",
    }
    etas = [float(value) for value in protocol["corruption"]["rate_curve"]["etas"]]
    seeds = protocol["evaluation"]["seeds"]
    ci_multiplier = float(stats.t.ppf(0.975, len(seeds) - 1) / math.sqrt(len(seeds)))
    lookup = {
        (str(row["dataset_id"]), float(row["eta"]), float(row["alpha"])): row
        for row in summaries
    }

    def draw(ax, dname: str, *, title: bool) -> None:
        for method in methods:
            means = [float(lookup[(dname, eta, 1.0)][f"{method}_mean"]) for eta in etas]
            errors = [
                ci_multiplier * float(lookup[(dname, eta, 1.0)][f"{method}_std"])
                for eta in etas
            ]
            ax.errorbar(
                etas,
                means,
                yerr=errors,
                label=labels[method],
                color=colors[method],
                marker=markers[method],
                linewidth=1.6,
                markersize=4.5,
                capsize=2.5,
            )
        if title:
            ax.set_title(PRETTY_DATASET.get(dname, dname))
        ax.set_xlabel("Rewired incidence fraction")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(etas)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    figure, axis = plt.subplots(figsize=(5.8, 3.6))
    draw(axis, "cora", title=False)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(outdir / "cora_corruption_accuracy_curve.pdf", bbox_inches="tight")
    figure.savefig(outdir / "cora_corruption_accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    for axis, dname in zip(axes, protocol["datasets"]):
        draw(axis, str(dname), title=True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(outdir / "all_dataset_corruption_accuracy_curves.pdf", bbox_inches="tight")
    figure.savefig(outdir / "all_dataset_corruption_accuracy_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary_markdown(
    outdir: Path,
    summaries: Sequence[Dict[str, object]],
    comparisons: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Controlled corruption results",
        "",
        f"Generated: {timestamp()}",
        "",
        "All values below come from the complete ten-seed result matrix. Accuracy is in percentage points.",
        "",
        "## EHGNN-F versus Random-Fixed",
        "",
        "| Dataset | eta | alpha | EHGNN-F | Random-Fixed | Delta [95% CI] | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary_lookup = {(row["dataset_id"], row["cell_id"]): row for row in summaries}
    for row in comparisons:
        summary = summary_lookup[(row["dataset_id"], row["cell_id"])]
        lines.append(
            f"| {row['dataset']} | {float(row['eta']):.1f} | {float(row['alpha']):.1f} | "
            f"{float(summary['EHGNN-F_mean']):.2f} +/- {float(summary['EHGNN-F_std']):.2f} | "
            f"{float(summary['Random-Fixed_mean']):.2f} +/- {float(summary['Random-Fixed_std']):.2f} | "
            f"{float(row['paired_delta_mean']):+.2f} "
            f"[{float(row['paired_delta_ci95_low']):+.2f}, {float(row['paired_delta_ci95_high']):+.2f}] | "
            f"{float(row['wilcoxon_p_holm_24']):.4f} |"
        )
    lines.extend([
        "",
        "The five method curves and retained-mask composition values are in the companion CSV files.",
        "",
        "This controlled intervention is a mechanism study and does not establish universal superiority on natural datasets.",
        "",
    ])
    (outdir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "evaluate", "summarize"), required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seed-values", nargs="+", type=int)
    parser.add_argument("--cell-ids", nargs="+")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
