#!/usr/bin/env python3
"""Aggregate HSL and HERALD into one audited task-aware-baseline table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HSL_ROOT = ROOT / ".work/hsl"
HERALD_ROOT = ROOT / ".work/herald"
HERALD_PROTOCOL = ROOT / "experiment_specs/herald.json"
OUTPUT_ROOT = ROOT / ".work/task_aware_baselines"


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_std(values: Sequence[float]):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def aggregate_herald():
    protocol = read_json(HERALD_PROTOCOL)
    selection_path = HERALD_ROOT / "frozen_selection.json"
    selection = read_json(selection_path)
    if selection["protocol_hash"] != file_hash(HERALD_PROTOCOL):
        raise RuntimeError("HERALD selection/protocol hash mismatch")
    summaries = []
    evaluation_rows = []
    metrics = (
        "test_acc",
        "test_macro_f1",
        "test_balanced_accuracy",
        "end_to_end_seconds",
        "peak_memory_mb",
        "best_epoch",
        "epochs_run",
        "residual_layer2_distance",
        "residual_layer3_distance",
    )
    for dataset in protocol["datasets"]:
        display = protocol["display_names"][dataset]
        choice = selection["choices"][dataset]
        if choice["status"] != "selected":
            validation_paths = sorted(
                (HERALD_ROOT / "validation/run_records").glob(f"{dataset}_*.json")
            )
            if len(validation_paths) != len(protocol["validation_grid"]):
                raise RuntimeError(f"Incomplete HERALD OOM grid for {dataset}")
            statuses = [read_json(path)["status"] for path in validation_paths]
            if statuses.count("oom") != 1 or statuses.count(
                "oom_inherited_from_minimum_configuration"
            ) != len(protocol["validation_grid"]) - 1:
                raise RuntimeError(f"Unexpected HERALD OOM evidence for {dataset}: {statuses}")
            summaries.append({
                "dataset_id": dataset,
                "dataset": display,
                "status": "oom_no_feasible_configuration",
                "complete_runs": 0,
                "failed_validation_cells": len(statuses),
            })
            continue

        config_id = choice["config_id"]
        rows = []
        for seed in protocol["evaluation"]["seeds"]:
            path = (
                HERALD_ROOT
                / "evaluation/run_records"
                / f"{dataset}_{config_id}_s{seed}.json"
            )
            if not path.exists():
                raise RuntimeError(f"Missing HERALD evaluation: {path}")
            row = read_json(path)
            if row["status"] != "complete":
                raise RuntimeError(f"Incomplete HERALD evaluation: {path}")
            rows.append(row)
            evaluation_rows.append(row)
        summary: Dict[str, object] = {
            "dataset_id": dataset,
            "dataset": display,
            "status": "complete",
            "config_id": config_id,
            "classifier_hidden": choice["classifier_hidden"],
            "dropout": choice["dropout"],
            "learning_rate": choice["learning_rate"],
            "herald_hidden": rows[0]["herald_hidden"],
            "complete_runs": len(rows),
            "failed_validation_cells": 0,
            "num_nodes": rows[0]["num_nodes"],
            "num_hyperedges": rows[0]["num_hyperedges"],
            "learned_incidence_candidate_count": rows[0][
                "learned_incidence_candidate_count"
            ],
            "learned_structure": rows[0]["learned_structure"],
        }
        for metric in metrics:
            mean, std = mean_std([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)
    return protocol, selection_path, evaluation_rows, summaries


def formatted(mean: str, std: str) -> str:
    return f"{float(mean):.2f} +/- {float(std):.2f}"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    protocol, selection_path, herald_runs, herald_summaries = aggregate_herald()
    hsl_rows = read_csv(HSL_ROOT / "method_summary.csv")
    hsl_by_dataset = {row["dataset_id"]: row for row in hsl_rows}
    herald_by_dataset = {row["dataset_id"]: row for row in herald_summaries}
    if set(hsl_by_dataset) != set(protocol["datasets"]):
        raise RuntimeError("HSL and HERALD dataset sets differ")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(HERALD_ROOT / "evaluation_runs.csv", herald_runs)
    write_csv(HERALD_ROOT / "method_summary.csv", herald_summaries)

    lines = [
        "# HSL and HERALD results",
        "",
        "Values are ten-seed mean +/- sample standard deviation. Both methods are descriptive task-aware structure learners outside EHGNN-F's exact-budget Holm family.",
        "",
        "| Dataset | Method | Accuracy | Macro-F1 | Balanced acc. | Original incidences retained | Added incidences | Resulting incidence representation |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in protocol["datasets"]:
        hsl = hsl_by_dataset[dataset]
        herald = herald_by_dataset[dataset]
        if hsl["status"] == "complete":
            hsl_retained = (
                f"{100 * float(hsl['realized_original_retention_mean']):.2f} "
                f"+/- {100 * float(hsl['realized_original_retention_std']):.2f}%"
            )
            hsl_added = (
                f"{float(hsl['new_support_added_mean']):.1f} +/- "
                f"{float(hsl['new_support_added_std']):.1f}"
            )
            hsl_structure = (
                f"{100 * float(hsl['realized_final_density_vs_original_mean']):.2f} "
                f"+/- {100 * float(hsl['realized_final_density_vs_original_std']):.2f}% "
                "of original count"
            )
            lines.append(
                f"| {hsl['dataset']} | HSL | "
                f"{formatted(hsl['test_acc_mean'], hsl['test_acc_std'])} | "
                f"{formatted(hsl['test_macro_f1_mean'], hsl['test_macro_f1_std'])} | "
                f"{formatted(hsl['test_balanced_accuracy_mean'], hsl['test_balanced_accuracy_std'])} | "
                f"{hsl_retained} | {hsl_added} | {hsl_structure} |"
            )
        else:
            lines.append(
                f"| {hsl['dataset']} | HSL | OOM | -- | -- | -- | -- | -- |"
            )

        if herald["status"] == "complete":
            lines.append(
                f"| {herald['dataset']} | HERALD | "
                f"{herald['test_acc_mean']:.2f} +/- {herald['test_acc_std']:.2f} | "
                f"{herald['test_macro_f1_mean']:.2f} +/- {herald['test_macro_f1_std']:.2f} | "
                f"{herald['test_balanced_accuracy_mean']:.2f} +/- {herald['test_balanced_accuracy_std']:.2f} | "
                "n/a | n/a | Dense weighted incidence matrix |"
            )
        else:
            lines.append(
                f"| {herald['dataset']} | HERALD | OOM | -- | -- | -- | -- | -- |"
            )

    lines.extend([
        "",
        "HSL may remove original support and add new binary incidences; its percentage is final non-self-loop support divided by original support. HERALD does not produce a binary subhypergraph: every feasible block constructs a dense weighted N-by-E incidence representation, so an exact-budget density is not applicable.",
        "",
        "## HERALD Compute",
        "",
        "| Dataset | End-to-end seconds | Peak GPU memory (MiB) |",
        "|---|---:|---:|",
    ])
    for dataset in protocol["datasets"]:
        row = herald_by_dataset[dataset]
        if row["status"] != "complete":
            lines.append(f"| {row['dataset']} | OOM | OOM |")
        else:
            lines.append(
                f"| {row['dataset']} | {row['end_to_end_seconds_mean']:.2f} +/- "
                f"{row['end_to_end_seconds_std']:.2f} | {row['peak_memory_mb_mean']:.1f} "
                f"+/- {row['peak_memory_mb_std']:.1f} |"
            )
    summary_path = OUTPUT_ROOT / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    write_csv(OUTPUT_ROOT / "unified_method_summary.csv", [
        {"method": "HSL", **row} for row in hsl_rows
    ] + [
        {"method": "HERALD", **row} for row in herald_summaries
    ])
    outputs = (
        summary_path,
        OUTPUT_ROOT / "unified_method_summary.csv",
        HERALD_ROOT / "evaluation_runs.csv",
        HERALD_ROOT / "method_summary.csv",
    )
    write_json(OUTPUT_ROOT / "summary_metadata.json", {
        "complete": True,
        "hsl_summary_hash": file_hash(HSL_ROOT / "method_summary.csv"),
        "herald_protocol_hash": file_hash(HERALD_PROTOCOL),
        "herald_selection_hash": file_hash(selection_path),
        "herald_complete_runs": len(herald_runs),
        "herald_oom_datasets": sum(
            row["status"] != "complete" for row in herald_summaries
        ),
        "output_hashes": {path.name: file_hash(path) for path in outputs},
    })
    print(summary_path.read_text())


if __name__ == "__main__":
    main()
