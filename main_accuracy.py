#!/usr/bin/env python3
"""Run and summarize the main EHGNN-F and Random-Fixed comparisons."""

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

from node_classification_utils import (
    PRETTY_DATASET,
    fixed_split,
    jaccard,
    prepare_dataset,
    train_learned_with_trajectory,
)
from training_utils import train_model, unit_scores_and_mask_for_baseline


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "experiment_specs/main_accuracy.json"
DEFAULT_CONFIG = ROOT / "experiment_specs/main_selected_hyperparameters.json"
DEFAULT_OUTDIR = ROOT / ".work/main_accuracy"


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
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


def protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_values(cli_values: Sequence[int] | None, protocol_values: Sequence[int]) -> List[int]:
    return list(cli_values) if cli_values is not None else list(protocol_values)


def training_kwargs(protocol: Dict[str, object]) -> Dict[str, object]:
    training = protocol["training"]
    return {
        "hidden": int(training["hidden"]),
        "dropout": float(training["dropout"]),
        "lr": float(training["lr"]),
        "wd": float(training["weight_decay"]),
        "epochs": int(training["epochs"]),
        "patience": int(training["patience"]),
        "sampling": str(training["sampling"]),
        "sparse_self_loop_policy": str(training.get("sparse_self_loop_policy", "augment")),
    }


def dataset_context(dname: str, protocol: Dict[str, object]):
    data_settings = protocol["data"]
    data, num_features, num_classes = prepare_dataset(dname)
    split_idx = fixed_split(
        data,
        int(data_settings["split_seed"]),
        float(data_settings["train_prop"]),
        float(data_settings["valid_prop"]),
    )
    return data, num_features, num_classes, split_idx


def forward_diagnostics(model, data, selected_count: int) -> Dict[str, object]:
    self_loop_count = int(getattr(model, "last_fixed_self_loop_count", 0))
    forward_count = int(getattr(model, "last_forward_incidence_count", selected_count))
    expected = selected_count + self_loop_count
    if forward_count != expected:
        raise RuntimeError(
            f"Forward incidence mismatch: observed {forward_count}, expected {expected}"
        )
    original_count = int(data.edge_index.size(1))
    return {
        "num_fixed_self_loops": self_loop_count,
        "forward_incidences": forward_count,
        "effective_forward_density": forward_count / (original_count + self_loop_count),
    }


def collect_json(parts: Sequence[str], pattern: str) -> List[Tuple[Path, Dict[str, object]]]:
    collected: List[Tuple[Path, Dict[str, object]]] = []
    for part in parts:
        for path in sorted(Path(part).rglob(pattern)):
            collected.append((path, read_json(path)))
    return collected


def unique_records(
    records: Iterable[Tuple[Path, Dict[str, object]]],
    key_fields: Sequence[str],
) -> List[Tuple[Path, Dict[str, object]]]:
    unique: Dict[Tuple[object, ...], Tuple[Path, Dict[str, object]]] = {}
    for path, record in records:
        key = tuple(record[field] for field in key_fields)
        if key in unique and unique[key][1] != record:
            raise RuntimeError(f"Conflicting duplicate records for {key}: {unique[key][0]} and {path}")
        unique[key] = (path, record)
    return list(unique.values())


def atomic_save_mask(path: Path, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = mask.detach().cpu().bool().numpy().astype(np.uint8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, packed=np.packbits(array), length=array.size)
    temporary.replace(path)


def load_mask(path: Path) -> torch.Tensor:
    with np.load(path) as archive:
        packed = archive["packed"]
        length = int(archive["length"])
    return torch.from_numpy(np.unpackbits(packed)[:length].astype(bool))


def verify_frozen(protocol_path: Path, frozen_path: Path) -> Dict[str, object]:
    frozen = read_json(frozen_path)
    if frozen.get("status") != "fixed":
        raise RuntimeError("Evaluation requires the fixed paper configuration")
    if frozen.get("protocol_hash") != protocol_hash(protocol_path):
        raise RuntimeError("Frozen configuration does not match the protocol hash")
    return frozen


def evaluate(args, protocol: Dict[str, object], device: torch.device) -> None:
    frozen = verify_frozen(Path(args.protocol), Path(args.frozen_config))
    selected = frozen["selected"]
    datasets = list(args.datasets or protocol["datasets"])
    seeds = resolve_values(args.seed_values, protocol["evaluation"]["seeds"])
    budgets = [float(value) for value in protocol["evaluation"]["budgets"]]
    train_args = training_kwargs(protocol)
    outdir = Path(args.outdir)
    result_dir = outdir / "evaluation_runs"
    mask_dir = outdir / "masks"
    result_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for dname in datasets:
        data, num_features, num_classes, split_idx = dataset_context(dname, protocol)
        for seed in seeds:
            for keep_ratio in budgets:
                ratio_tag = f"{int(round(100 * keep_ratio)):03d}"
                learned_path = result_dir / f"eval_ehgnnf_{dname}_r{ratio_tag}_s{seed}.json"
                learned_mask_path = mask_dir / f"mask_ehgnnf_{dname}_r{ratio_tag}_s{seed}.npz"
                if not learned_path.exists() or args.force:
                    print(f"[evaluate] EHGNN-F dataset={dname} rho={keep_ratio} seed={seed}", flush=True)
                    model, metrics, init_mask, init_probs, final_mask, final_probs, _ = train_learned_with_trajectory(
                        data=data,
                        split_idx=split_idx,
                        mode="learnmask",
                        num_features=num_features,
                        num_classes=num_classes,
                        keep_ratio=keep_ratio,
                        seed=int(seed),
                        device=device,
                        trajectory_every=int(train_args["epochs"]) + 1,
                        mask_init_std=float(selected["mask_init_std"]),
                        mask_lr_multiplier=float(selected["mask_lr_multiplier"]),
                        include_test_metrics=True,
                        **train_args,
                    )
                    atomic_save_mask(learned_mask_path, final_mask)
                    atomic_write_json(learned_path, {
                        "stage": "frozen_test_evaluation",
                        "created_at": timestamp(),
                        "protocol_hash": protocol_hash(Path(args.protocol)),
                        "frozen_config_hash": hashlib.sha256(Path(args.frozen_config).read_bytes()).hexdigest(),
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "seed": int(seed),
                        "keep_ratio": keep_ratio,
                        "method": "EHGNN-F",
                        "unit": "incidence",
                        "sparse_self_loop_policy": train_args["sparse_self_loop_policy"],
                        "mask_init_std": float(selected["mask_init_std"]),
                        "mask_lr_multiplier": float(selected["mask_lr_multiplier"]),
                        "val_acc": 100.0 * float(metrics["val_acc"]),
                        "test_acc": 100.0 * float(metrics["test_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_run": int(metrics["epochs_run"]),
                        "initial_final_jaccard": jaccard(init_mask, final_mask),
                        "mean_abs_probability_change": float((final_probs - init_probs).abs().mean()),
                        "num_selected": int(final_mask.sum()),
                        "num_mask_units": int(final_mask.numel()),
                        "mask_path": str(learned_mask_path.resolve()),
                        **forward_diagnostics(model, data, int(final_mask.sum())),
                    })
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"[evaluate:skip] EHGNN-F dataset={dname} rho={keep_ratio} seed={seed}", flush=True)

                random_path = result_dir / f"eval_random_{dname}_r{ratio_tag}_s{seed}.json"
                random_mask_path = mask_dir / f"mask_random_{dname}_r{ratio_tag}_s{seed}.npz"
                if not random_path.exists() or args.force:
                    print(f"[evaluate] Random-Fixed dataset={dname} rho={keep_ratio} seed={seed}", flush=True)
                    random_edge_index, random_mask, _ = unit_scores_and_mask_for_baseline(
                        data,
                        method="Random-Fixed",
                        unit="incidence",
                        keep_ratio=keep_ratio,
                        seed=int(protocol["evaluation"]["random_mask_seed_offset"]) + int(seed),
                    )
                    random_data = copy.deepcopy(data)
                    random_data.edge_index = random_edge_index
                    model, metrics = train_model(
                        data=random_data,
                        split_idx=split_idx,
                        mode="random",
                        num_features=num_features,
                        num_classes=num_classes,
                        keep_ratio=keep_ratio,
                        seed=int(seed),
                        device=device,
                        **train_args,
                    )
                    atomic_save_mask(random_mask_path, random_mask)
                    atomic_write_json(random_path, {
                        "stage": "frozen_test_evaluation",
                        "created_at": timestamp(),
                        "protocol_hash": protocol_hash(Path(args.protocol)),
                        "frozen_config_hash": hashlib.sha256(Path(args.frozen_config).read_bytes()).hexdigest(),
                        "dataset_id": dname,
                        "dataset": PRETTY_DATASET.get(dname, dname),
                        "seed": int(seed),
                        "keep_ratio": keep_ratio,
                        "method": "Random-Fixed",
                        "unit": "incidence",
                        "sparse_self_loop_policy": train_args["sparse_self_loop_policy"],
                        "val_acc": 100.0 * float(metrics["val_acc"]),
                        "test_acc": 100.0 * float(metrics["test_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "epochs_run": int(metrics["epochs_run"]),
                        "num_selected": int(random_mask.sum()),
                        "num_mask_units": int(random_mask.numel()),
                        "mask_path": str(random_mask_path.resolve()),
                        **forward_diagnostics(model, data, int(random_mask.sum())),
                    })
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    print(f"[evaluate:skip] Random-Fixed dataset={dname} rho={keep_ratio} seed={seed}", flush=True)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean
    sem = float(stats.sem(array))
    radius = float(stats.t.ppf((1.0 + confidence) / 2.0, array.size - 1) * sem)
    return mean - radius, mean + radius


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(args, protocol: Dict[str, object]) -> None:
    verify_frozen(Path(args.protocol), Path(args.frozen_config))
    parts = args.parts or [args.outdir]
    records = unique_records(
        collect_json(parts, "eval_*.json"),
        ("dataset_id", "keep_ratio", "seed", "method"),
    )
    rows_with_paths = records
    rows = [record for _, record in rows_with_paths]
    methods = ("EHGNN-F", "Random-Fixed")
    expected = {
        (dname, float(ratio), int(seed), method)
        for dname in protocol["datasets"]
        for ratio in protocol["evaluation"]["budgets"]
        for seed in protocol["evaluation"]["seeds"]
        for method in methods
    }
    observed = {
        (str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"Incomplete frozen evaluation: missing={len(missing)} extra={len(extra)}")
    expected_protocol_hash = protocol_hash(Path(args.protocol))
    if any(row["protocol_hash"] != expected_protocol_hash for row in rows):
        raise RuntimeError("At least one evaluation record was generated from a different protocol")
    if protocol["training"].get("sparse_self_loop_policy") == "fixed_self_loops_implicit":
        dataset_sizes = {}
        for dname in protocol["datasets"]:
            data, _, _, _ = dataset_context(dname, protocol)
            dataset_sizes[dname] = (int(data.n_x), int(data.edge_index.size(1)))
        for row in rows:
            node_count, original_count = dataset_sizes[str(row["dataset_id"])]
            selected_count = int(row["num_selected"])
            if int(row["num_fixed_self_loops"]) != node_count:
                raise RuntimeError("Fixed self-loop count does not equal the node count")
            if int(row["forward_incidences"]) != selected_count + node_count:
                raise RuntimeError("Fixed self-loop forward count is incorrect")
            expected_density = (selected_count + node_count) / (original_count + node_count)
            if not math.isclose(
                float(row["effective_forward_density"]), expected_density, abs_tol=1e-12
            ):
                raise RuntimeError("Effective forward density is incorrect")

    lookup = {
        (row["dataset_id"], float(row["keep_ratio"]), int(row["seed"]), row["method"]): row
        for row in rows
    }
    paired_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    stability_rows: List[Dict[str, object]] = []
    for dname in protocol["datasets"]:
        for ratio_value in protocol["evaluation"]["budgets"]:
            ratio = float(ratio_value)
            deltas: List[float] = []
            learned_scores: List[float] = []
            random_scores: List[float] = []
            learned_masks: List[Tuple[int, torch.Tensor]] = []
            init_jaccards: List[float] = []
            probability_changes: List[float] = []
            for seed in protocol["evaluation"]["seeds"]:
                learned = lookup[(dname, ratio, int(seed), "EHGNN-F")]
                random = lookup[(dname, ratio, int(seed), "Random-Fixed")]
                learned_score = float(learned["test_acc"])
                random_score = float(random["test_acc"])
                delta = learned_score - random_score
                learned_scores.append(learned_score)
                random_scores.append(random_score)
                deltas.append(delta)
                init_jaccards.append(float(learned["initial_final_jaccard"]))
                probability_changes.append(float(learned["mean_abs_probability_change"]))
                learned_masks.append((int(seed), load_mask(Path(learned["mask_path"]))))
                paired_rows.append({
                    "dataset": PRETTY_DATASET.get(dname, dname),
                    "dataset_id": dname,
                    "keep_ratio": ratio,
                    "seed": int(seed),
                    "ehgnn_test_acc": learned_score,
                    "random_test_acc": random_score,
                    "test_delta": delta,
                })

            delta_mean, delta_std = mean_std(deltas)
            ci_low, ci_high = mean_ci(deltas)
            learned_mean, learned_std = mean_std(learned_scores)
            random_mean, random_std = mean_std(random_scores)
            if np.allclose(deltas, 0.0):
                wilcoxon_p = 1.0
            else:
                wilcoxon_p = float(stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided").pvalue)
            ttest_p = float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
            summary_rows.append({
                "dataset": PRETTY_DATASET.get(dname, dname),
                "dataset_id": dname,
                "keep_ratio": ratio,
                "n": len(deltas),
                "ehgnn_test_acc_mean": learned_mean,
                "ehgnn_test_acc_std": learned_std,
                "random_test_acc_mean": random_mean,
                "random_test_acc_std": random_std,
                "paired_delta_mean": delta_mean,
                "paired_delta_std": delta_std,
                "structural_density": ratio,
                "effective_forward_density": float(
                    learned["effective_forward_density"]
                ),
                "num_fixed_self_loops": int(learned["num_fixed_self_loops"]),
                "paired_delta_ci95_low": ci_low,
                "paired_delta_ci95_high": ci_high,
                "wins": int(np.sum(np.asarray(deltas) > 0.0)),
                "ties": int(np.sum(np.isclose(deltas, 0.0))),
                "wilcoxon_p": wilcoxon_p,
                "paired_t_p": ttest_p,
                "initial_final_jaccard_mean": float(np.mean(init_jaccards)),
                "mean_abs_probability_change_mean": float(np.mean(probability_changes)),
            })

            pairwise = [
                jaccard(left_mask, right_mask)
                for left_index, (_, left_mask) in enumerate(learned_masks)
                for _, right_mask in learned_masks[left_index + 1:]
            ]
            mask_length = int(learned_masks[0][1].numel())
            selected_count = int(learned_masks[0][1].sum())
            actual_ratio = selected_count / mask_length
            chance = actual_ratio / (2.0 - actual_ratio)
            pairwise_mean, pairwise_std = mean_std(pairwise)
            stability_rows.append({
                "dataset": PRETTY_DATASET.get(dname, dname),
                "dataset_id": dname,
                "keep_ratio": ratio,
                "num_seed_pairs": len(pairwise),
                "learned_pairwise_jaccard_mean": pairwise_mean,
                "learned_pairwise_jaccard_std": pairwise_std,
                "learned_pairwise_jaccard_min": float(np.min(pairwise)),
                "learned_pairwise_jaccard_max": float(np.max(pairwise)),
                "chance_jaccard_approx": chance,
                "jaccard_above_chance": pairwise_mean - chance,
            })

    adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in summary_rows])
    for row, adjusted_p in zip(summary_rows, adjusted):
        row["wilcoxon_p_holm_20"] = adjusted_p

    across_dataset_rows: List[Dict[str, object]] = []
    for ratio_value in protocol["evaluation"]["budgets"]:
        ratio = float(ratio_value)
        dataset_means = [
            float(row["paired_delta_mean"])
            for row in summary_rows
            if float(row["keep_ratio"]) == ratio
        ]
        macro_mean, macro_std = mean_std(dataset_means)
        ci_low, ci_high = mean_ci(dataset_means)
        across_dataset_rows.append({
            "keep_ratio": ratio,
            "num_datasets": len(dataset_means),
            "macro_paired_delta_mean": macro_mean,
            "dataset_delta_std": macro_std,
            "macro_delta_ci95_low": ci_low,
            "macro_delta_ci95_high": ci_high,
            "wilcoxon_across_datasets_p": (
                1.0 if np.allclose(dataset_means, 0.0)
                else float(stats.wilcoxon(dataset_means, alternative="two-sided").pvalue)
            ),
        })

    outdir = Path(args.outdir)
    write_csv(outdir / "frozen_evaluation_runs.csv", sorted(
        rows,
        key=lambda row: (str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), str(row["method"])),
    ))
    write_csv(outdir / "paired_seed_results.csv", paired_rows)
    write_csv(outdir / "paired_budget_summary.csv", summary_rows)
    write_csv(outdir / "learned_mask_stability.csv", stability_rows)
    write_csv(outdir / "across_dataset_budget_summary.csv", across_dataset_rows)
    atomic_write_json(outdir / "summary_metadata.json", {
        "created_at": timestamp(),
        "protocol_hash": expected_protocol_hash,
        "number_of_evaluation_records": len(rows),
        "number_of_paired_comparisons": len(paired_rows),
        "holm_family_size": len(summary_rows),
    })
    print(f"[summarize] wrote complete results to {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("evaluate", "summarize"), required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--frozen-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--parts", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seed-values", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol)
    protocol = read_json(protocol_path)
    device = torch.device(
        args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    )
    if args.stage == "evaluate":
        evaluate(args, protocol, device)
    else:
        summarize(args, protocol)


if __name__ == "__main__":
    main()
