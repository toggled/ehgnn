#!/usr/bin/env python3
"""Analyze sampler stability and exact equivalence at full retention."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from node_classification_utils import fixed_split
from training_utils import clone_state_dict, make_args, setup_seed
from models_sparse import HCHA
from experiment_controls.core_controls import (
    load_dataset,
    split_hash,
    tensor_hash,
)


DEFAULT_STABILITY_SPEC = Path("experiment_specs/sampler_stability.json")
DEFAULT_FULL_RETENTION_SPEC = Path("experiment_specs/full_retention.json")


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


def verify_protocol(protocol_path: Path, expected_prefix: str) -> Dict[str, object]:
    protocol = read_json(protocol_path)
    if not str(protocol.get("protocol_version", "")).startswith(expected_prefix):
        raise RuntimeError(f"Unexpected protocol version in {protocol_path}")
    if protocol.get("status") != "fixed":
        raise RuntimeError(f"Experiment specification is not fixed: {protocol_path}")
    for item in protocol["dependencies"]:
        path = Path(str(item["path"]))
        observed = file_hash(path)
        if observed != item["sha256"]:
            raise RuntimeError(
                f"Dependency changed: {path}: {observed} != {item['sha256']}"
            )
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


def build_model(
    *,
    mode: str,
    data,
    num_features: int,
    num_classes: int,
    keep_ratio: float,
    training: Mapping[str, object],
    device: torch.device,
) -> HCHA:
    args = make_args(
        mode=mode,
        data=data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=keep_ratio,
        hidden=int(training["hidden"]),
        dropout=float(training["dropout"]),
        sampling=str(training["sampling"]),
    )
    args.mask_init_std = float(training.get("mask_init_std", 0.01))
    args.mask_gradient_estimator = "selected_only"
    args.sparse_self_loop_policy = str(training["sparse_self_loop_policy"])
    model = HCHA(args).to(device)
    model.reset_parameters()
    return model


def deterministic_topk_state(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, float]:
    if not 0 < k <= logits.numel():
        raise ValueError(f"Invalid k={k} for t={logits.numel()}")
    values, indices = torch.topk(logits.detach(), k=min(k + 1, logits.numel()))
    mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    mask[indices[:k]] = True
    gap = math.inf if k == logits.numel() else float((values[k - 1] - values[k]).item())
    return mask, gap


def sampler_step_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    before_mask: torch.Tensor,
    before_gap: float,
    k: int,
) -> Tuple[Dict[str, object], torch.Tensor, float]:
    if before.shape != after.shape:
        raise ValueError("Logit vectors must have the same shape")
    t = int(before.numel())
    after_mask, after_gap = deterministic_topk_state(after, k)
    update = after - before
    l1_update = float(update.abs().sum().item())
    linf_update = float(update.abs().max().item())
    alpha_before = float(torch.sigmoid(before.min()).item())
    alpha_after = float(torch.sigmoid(after.min()).item())
    alpha_pair = min(alpha_before, alpha_after)
    b0 = float(2 * k)
    b1 = (
        float(k)
        / (2.0 * alpha_pair)
        * math.log(float(t) / float(t - k))
        * l1_update
    )
    intersection = int((before_mask & after_mask).sum().item())
    symmetric_difference = int(2 * (k - intersection))
    union = int(2 * k - intersection)
    changed = symmetric_difference > 0
    premise = linf_update < before_gap / 2.0
    if premise and changed:
        raise AssertionError("Deterministic top-k margin proposition was violated")
    return {
        "t": t,
        "k": k,
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
        "alpha_pair": alpha_pair,
        "l1_logit_update": l1_update,
        "linf_logit_update": linf_update,
        "b0": b0,
        "b1": b1,
        "b1_over_b0": b1 / b0,
        "topk_boundary_gap_before": before_gap,
        "topk_boundary_gap_after": after_gap,
        "margin_premise_holds": premise,
        "topk_changed": changed,
        "topk_symmetric_difference": symmetric_difference,
        "topk_jaccard": intersection / union if union else 1.0,
    }, after_mask, after_gap


def train_stability_run(
    *,
    dataset: str,
    budget: float,
    seed: int,
    protocol: Mapping[str, object],
    protocol_hash: str,
    device: torch.device,
) -> Dict[str, object]:
    data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: value.to(device) for name, value in split.items()}
    training = protocol["training"]
    setup_seed(seed)
    model = build_model(
        mode="learnmask",
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=budget,
        training=training,
        device=device,
    )
    mask_parameters = list(model.mask_module.parameters())
    mask_ids = {id(parameter) for parameter in mask_parameters}
    classifier_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in mask_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": classifier_parameters, "lr": float(training["learning_rate"])},
            {
                "params": mask_parameters,
                "lr": float(training["learning_rate"])
                * float(training["mask_lr_multiplier"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    t = int(local_data.edge_index.size(1))
    k = max(1, int(budget * t))
    before_mask, before_gap = deterministic_topk_state(model.mask_module.logits, k)
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    trajectory: List[Dict[str, object]] = []
    start = time.perf_counter()
    maximum_epochs = int(training["maximum_epochs"])
    patience = int(training["patience"])

    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = model.mask_module.logits.detach().clone()
        logits, _ = model(local_data, is_test=False)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            logp[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {dataset}, rho={budget}, seed={seed}")
        loss.backward()
        scorer_grad = model.mask_module.logits.grad
        scorer_grad_l1 = (
            float(scorer_grad.detach().abs().sum().item())
            if scorer_grad is not None
            else 0.0
        )
        optimizer.step()
        after = model.mask_module.logits.detach()
        step, before_mask, before_gap = sampler_step_metrics(
            before, after, before_mask, before_gap, k
        )
        with torch.no_grad():
            model.eval()
            val_logits = model(local_data, is_test=True)
            val_loss = float(
                F.cross_entropy(
                    val_logits[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                ).item()
            )
        trajectory.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "scorer_grad_l1": scorer_grad_l1,
                **step,
            }
        )
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
        raise RuntimeError("No finite validation checkpoint")
    ratio = np.asarray([float(row["b1_over_b0"]) for row in trajectory])
    margin = np.asarray([bool(row["margin_premise_holds"]) for row in trajectory])
    changed = np.asarray([bool(row["topk_changed"]) for row in trajectory])
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": "sampler_stability",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "budget": budget,
        "seed": seed,
        **diagnostics,
        "k": k,
        "epochs_run": len(trajectory),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "end_to_end_seconds": elapsed,
        "fraction_b1_lt_b0": float(np.mean(ratio < 1.0)),
        "fraction_margin_premise": float(np.mean(margin)),
        "fraction_topk_changed": float(np.mean(changed)),
        "median_b1_over_b0": float(np.median(ratio)),
        "minimum_b1_over_b0": float(np.min(ratio)),
        "maximum_b1_over_b0": float(np.max(ratio)),
        "trajectory": trajectory,
    }


def stability_run_path(outdir: Path, dataset: str, budget: float, seed: int) -> Path:
    return outdir / "run_records" / f"{dataset}_r{int(round(100 * budget)):03d}_s{seed}.json"


def run_stability_analysis(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "sampler-stability-")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    budgets = selected(args.budgets, protocol["evaluation"]["budgets"], "budgets")
    seeds = selected(args.seeds, protocol["evaluation"]["model_seeds"], "seeds")
    outdir = Path(str(protocol["output_root"]))
    protocol_hash = file_hash(protocol_path)
    device = torch.device(args.device)
    for dataset in datasets:
        for budget in budgets:
            for seed in seeds:
                path = stability_run_path(outdir, dataset, float(budget), int(seed))
                if path.exists() and not args.force:
                    print(f"[sampler-stability:skip] {dataset} rho={budget} seed={seed}", flush=True)
                    continue
                print(f"[sampler-stability] {dataset} rho={budget} seed={seed}", flush=True)
                result = train_stability_run(
                    dataset=dataset,
                    budget=float(budget),
                    seed=int(seed),
                    protocol=protocol,
                    protocol_hash=protocol_hash,
                    device=device,
                )
                write_json(path, result)
                print(
                    f"[sampler-stability:done] {dataset} rho={budget} seed={seed} "
                    f"epochs={result['epochs_run']} frac_nontrivial={result['fraction_b1_lt_b0']:.3f}",
                    flush=True,
                )


def copy_classifier_state(source: HCHA, target: HCHA) -> None:
    state = source.state_dict()
    target_state = target.state_dict()
    for name, value in state.items():
        if name.startswith("convs."):
            target_state[name] = value.detach().clone()
    target.load_state_dict(target_state)


def classifier_gradients(model: HCHA) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("convs.") and parameter.grad is not None
    }


def compare_tensors(
    left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float
) -> Dict[str, object]:
    difference = (left - right).abs()
    return {
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
    }


def full_retention_equivalence_cell(
    *,
    dataset: str,
    protocol: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: value.to(device) for name, value in split.items()}
    training = dict(protocol["training"])
    training["dropout"] = 0.0
    seed = int(protocol["deterministic_gate"]["seed"])
    setup_seed(seed)
    full = build_model(
        mode="full",
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=1.0,
        training=training,
        device=device,
    )
    learned = build_model(
        mode="learnmask",
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=1.0,
        training=training,
        device=device,
    )
    copy_classifier_state(full, learned)
    atol = float(protocol["deterministic_gate"]["atol"])
    rtol = float(protocol["deterministic_gate"]["rtol"])

    full.eval()
    learned.eval()
    with torch.no_grad():
        full_logits = full(local_data, is_test=True)
        learned_logits = learned(local_data, is_test=True)
        _, _, hard_mask, _ = learned.mask_module(
            local_data,
            keep_ratio=1.0,
            is_test=True,
            return_mask=True,
        )
    forward = compare_tensors(full_logits, learned_logits, atol, rtol)

    full.zero_grad(set_to_none=True)
    learned.zero_grad(set_to_none=True)
    full.train()
    learned.train()
    full_loss = F.cross_entropy(
        full(local_data, is_test=True)[local_split["train"]],
        local_data.y[local_split["train"]],
    )
    learned_train_logits = learned(local_data, is_test=True)
    learned_loss = F.cross_entropy(
        learned_train_logits[local_split["train"]],
        local_data.y[local_split["train"]],
    )
    full_loss.backward()
    learned_loss.backward()
    full_grad = classifier_gradients(full)
    learned_grad = classifier_gradients(learned)
    if set(full_grad) != set(learned_grad):
        raise RuntimeError("Classifier-gradient parameter sets differ")
    gradient_cells = {
        name: compare_tensors(full_grad[name], learned_grad[name], atol, rtol)
        for name in full_grad
    }
    gradients_allclose = all(value["allclose"] for value in gradient_cells.values())
    gradient_max_abs = max(value["max_abs"] for value in gradient_cells.values())
    t = int(local_data.edge_index.size(1))
    n = int(local_data.n_x)
    selected_count = int(hard_mask.sum().item())
    counts_pass = (
        selected_count == t
        and int(learned.last_forward_incidence_count) == t + n
        and int(full.last_forward_incidence_count) == t + n
        and int(learned.last_fixed_self_loop_count) == n
        and int(full.last_fixed_self_loop_count) == n
    )
    passed = counts_pass and bool(forward["allclose"]) and gradients_allclose
    return {
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        **diagnostics,
        "selected_original_incidences": selected_count,
        "full_forward_incidences": int(full.last_forward_incidence_count),
        "learned_forward_incidences": int(learned.last_forward_incidence_count),
        "full_fixed_self_loops": int(full.last_fixed_self_loop_count),
        "learned_fixed_self_loops": int(learned.last_fixed_self_loop_count),
        "counts_pass": counts_pass,
        "forward_max_abs": forward["max_abs"],
        "forward_mean_abs": forward["mean_abs"],
        "forward_allclose": forward["allclose"],
        "gradient_max_abs": gradient_max_abs,
        "gradients_allclose": gradients_allclose,
        "full_loss": float(full_loss.item()),
        "learned_loss": float(learned_loss.item()),
        "passed": passed,
        "gradient_details": gradient_cells,
    }


def run_full_retention_gate(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "full-retention-")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    outdir = Path(str(protocol["output_root"]))
    path = outdir / "deterministic_equivalence_gate.json"
    if path.exists() and not args.force:
        raise RuntimeError(f"Gate output exists; refusing to overwrite {path}")
    device = torch.device(args.device)
    rows = []
    for dataset in datasets:
        print(f"[full-retention:gate] {dataset}", flush=True)
        rows.append(full_retention_equivalence_cell(dataset=dataset, protocol=protocol, device=device))
        print(f"[full-retention:gate:done] {dataset} passed={rows[-1]['passed']}", flush=True)
    result = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "stage": "full_retention_equivalence_gate",
        "all_datasets_requested": set(datasets) == set(protocol["datasets"]),
        "passed": set(datasets) == set(protocol["datasets"]) and all(
            row["passed"] for row in rows
        ),
        "rows": rows,
    }
    write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("Full-retention equivalence check failed")


def summarize_stability(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "sampler-stability-")
    outdir = Path(str(protocol["output_root"]))
    expected = [
        stability_run_path(outdir, dataset, float(budget), int(seed))
        for dataset in protocol["datasets"]
        for budget in protocol["evaluation"]["budgets"]
        for seed in protocol["evaluation"]["model_seeds"]
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Sampler-stability matrix is incomplete; missing {len(missing)} runs")
    runs = [read_json(path) for path in expected]
    summaries = []
    trajectories = []
    for run in runs:
        summary = {key: value for key, value in run.items() if key != "trajectory"}
        summaries.append(summary)
        for row in run["trajectory"]:
            trajectories.append(
                {
                    "dataset_id": run["dataset_id"],
                    "dataset": run["dataset"],
                    "budget": run["budget"],
                    "seed": run["seed"],
                    **row,
                }
            )
    write_csv(outdir / "run_summary.csv", summaries)
    write_csv(outdir / "epoch_trajectory.csv", trajectories)
    groups: List[Dict[str, object]] = []
    for dataset in protocol["datasets"]:
        for budget in protocol["evaluation"]["budgets"]:
            group = [
                row
                for row in trajectories
                if row["dataset_id"] == dataset and float(row["budget"]) == float(budget)
            ]
            ratios = np.asarray([float(row["b1_over_b0"]) for row in group])
            margins = np.asarray([bool(row["margin_premise_holds"]) for row in group])
            changes = np.asarray([bool(row["topk_changed"]) for row in group])
            groups.append(
                {
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "num_epoch_steps": len(group),
                    "fraction_b1_lt_b0": float(np.mean(ratios < 1.0)),
                    "median_b1_over_b0": float(np.median(ratios)),
                    "minimum_b1_over_b0": float(np.min(ratios)),
                    "fraction_margin_premise": float(np.mean(margins)),
                    "fraction_topk_changed": float(np.mean(changes)),
                }
            )
    write_csv(outdir / "cell_summary.csv", groups)
    metadata = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "complete": True,
        "run_count": len(runs),
        "epoch_row_count": len(trajectories),
        "output_hashes": {
            name: file_hash(outdir / name)
            for name in ("run_summary.csv", "epoch_trajectory.csv", "cell_summary.csv")
        },
    }
    write_json(outdir / "summary_metadata.json", metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the sampler-stability analysis")
    run.add_argument("--protocol", default=str(DEFAULT_STABILITY_SPEC))
    run.add_argument("--datasets", nargs="+")
    run.add_argument("--budgets", nargs="+", type=float)
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=run_stability_analysis)
    aggregate = subparsers.add_parser("summarize", help="Summarize the sampler-stability runs")
    aggregate.add_argument("--protocol", default=str(DEFAULT_STABILITY_SPEC))
    aggregate.set_defaults(func=summarize_stability)
    gate = subparsers.add_parser("full-retention-gate", help="Run the full-retention equivalence check")
    gate.add_argument("--protocol", default=str(DEFAULT_FULL_RETENTION_SPEC))
    gate.add_argument("--datasets", nargs="+")
    gate.add_argument("--device", default="cuda:0")
    gate.add_argument("--force", action="store_true")
    gate.set_defaults(func=run_full_retention_gate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
