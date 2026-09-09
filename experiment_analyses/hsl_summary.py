#!/usr/bin/env python3
"""Summarize all HSL evaluation outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


PROTOCOL_PATH = Path("experiment_specs/hsl.json")


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def normalize_exported_metrics(row):
    """Undo the evaluation writer's duplicated percentage conversion."""
    normalized = dict(row)
    for metric in ("test_macro_f1", "test_balanced_accuracy"):
        normalized[metric] = float(normalized[metric]) / 100.0
    return normalized


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    protocol = read_json(PROTOCOL_PATH)
    outdir = Path(protocol["output_root"])
    selection = read_json(outdir / "frozen_selection.json")
    rows = []
    summaries = []
    metrics = [
        "test_acc",
        "test_macro_f1",
        "test_balanced_accuracy",
        "realized_original_retention",
        "realized_final_density_vs_original",
        "effective_forward_density_vs_full",
        "original_support_retained",
        "original_support_removed",
        "new_support_added",
        "final_unique_non_self_loop_support",
        "retained_hyperedges",
        "hyperedge_retention",
        "end_to_end_seconds",
        "peak_memory_mb",
    ]
    for dataset in protocol["datasets"]:
        choice = selection["choices"][dataset]
        if choice["status"] != "selected":
            summaries.append({
                "dataset_id": dataset,
                "dataset": protocol["display_names"][dataset],
                "status": "oom_no_feasible_configuration",
                "complete_runs": 0,
                "failed_runs": len(protocol["validation_grid"]),
            })
            continue
        config_id = choice["config_id"]
        group = []
        for seed in protocol["evaluation"]["seeds"]:
            path = outdir / "evaluation" / "run_records" / f"{dataset}_{config_id}_s{seed}.json"
            if not path.exists():
                raise RuntimeError(f"Missing frozen HSL evaluation: {path}")
            row = normalize_exported_metrics(read_json(path))
            rows.append(row)
            group.append(row)
        complete = [row for row in group if row["status"] == "complete"]
        if len(complete) != len(protocol["evaluation"]["seeds"]):
            summaries.append({
                "dataset_id": dataset,
                "dataset": protocol["display_names"][dataset],
                "status": "incomplete_test_evaluation",
                "config_id": config_id,
                "complete_runs": len(complete),
                "failed_runs": len(group) - len(complete),
            })
            continue
        summary: Dict[str, object] = {
            "dataset_id": dataset,
            "dataset": protocol["display_names"][dataset],
            "status": "complete",
            "config_id": config_id,
            "hidden": choice["hidden"],
            "learning_rate": choice["learning_rate"],
            "p_add": choice["p_add"],
            "complete_runs": len(complete),
            "failed_runs": 0,
            "num_original_incidences": complete[0]["num_original_incidences"],
            "num_hyperedges": complete[0]["num_hyperedges"],
            "fixed_self_loop_count": complete[0]["fixed_self_loop_count"],
        }
        for metric in metrics:
            mean, std = mean_std([row[metric] for row in complete])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summaries.append(summary)
    if rows:
        write_csv(outdir / "evaluation_runs.csv", rows)
    write_csv(outdir / "method_summary.csv", summaries)

    lines = [
        "# HSL realized-density result",
        "",
        "HSL is a descriptive task-aware structure-learning baseline. It is not part of the exact-budget Holm family.",
        "",
        "## Predictive Performance",
        "",
        "| Dataset | Status | Accuracy | Macro-F1 | Balanced acc. |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        if row["status"] != "complete":
            lines.append(
                f"| {row['dataset']} | OOM: no feasible configuration | -- | -- | -- |"
            )
            continue
        lines.append(
            f"| {row['dataset']} | complete | "
            f"{row['test_acc_mean']:.2f} +/- {row['test_acc_std']:.2f} | "
            f"{row['test_macro_f1_mean']:.2f} +/- {row['test_macro_f1_std']:.2f} | "
            f"{row['test_balanced_accuracy_mean']:.2f} +/- {row['test_balanced_accuracy_std']:.2f} |"
        )
    lines.extend([
        "",
        "## Realized Structure",
        "",
        "Counts and densities are mean +/- sample standard deviation over ten seeds. Fixed self-loops are excluded from original/final support counts and included only in effective forward density.",
        "",
        "| Dataset | Original retained | Original removed | New support | Final non-self-loop support | Final/original density | Retained hyperedges | Effective forward density |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summaries:
        if row["status"] != "complete":
            lines.append(f"| {row['dataset']} | OOM | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {row['dataset']} | "
            f"{row['original_support_retained_mean']:.1f} +/- {row['original_support_retained_std']:.1f} "
            f"({100 * row['realized_original_retention_mean']:.2f} +/- {100 * row['realized_original_retention_std']:.2f}%) | "
            f"{row['original_support_removed_mean']:.1f} +/- {row['original_support_removed_std']:.1f} | "
            f"{row['new_support_added_mean']:.1f} +/- {row['new_support_added_std']:.1f} | "
            f"{row['final_unique_non_self_loop_support_mean']:.1f} +/- {row['final_unique_non_self_loop_support_std']:.1f} | "
            f"{100 * row['realized_final_density_vs_original_mean']:.2f} +/- {100 * row['realized_final_density_vs_original_std']:.2f}% | "
            f"{row['retained_hyperedges_mean']:.1f} +/- {row['retained_hyperedges_std']:.1f} "
            f"({100 * row['hyperedge_retention_mean']:.2f} +/- {100 * row['hyperedge_retention_std']:.2f}%) | "
            f"{100 * row['effective_forward_density_vs_full_mean']:.2f} +/- {100 * row['effective_forward_density_vs_full_std']:.2f}% |"
        )
    lines.extend([
        "",
        "## Compute",
        "",
        "| Dataset | End-to-end seconds | Peak GPU memory (MiB) |",
        "|---|---:|---:|",
    ])
    for row in summaries:
        if row["status"] != "complete":
            lines.append(f"| {row['dataset']} | OOM | OOM |")
            continue
        lines.append(
            f"| {row['dataset']} | {row['end_to_end_seconds_mean']:.2f} +/- {row['end_to_end_seconds_std']:.2f} | "
            f"{row['peak_memory_mb_mean']:.1f} +/- {row['peak_memory_mb_std']:.1f} |"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    outputs = [outdir / "method_summary.csv", outdir / "summary.md"]
    if rows:
        outputs.append(outdir / "evaluation_runs.csv")
    write_json(outdir / "summary_metadata.json", {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(PROTOCOL_PATH),
        "selection_hash": file_hash(outdir / "frozen_selection.json"),
        "complete": True,
        "reporting_correction": (
            "Raw run records multiplied macro-F1 and balanced accuracy by 100 "
            "after classification_metrics had already returned percentages. "
            "The aggregate divides those two fields by 100; accuracy, training, "
            "selection, and structure fields are unchanged."
        ),
        "output_hashes": {path.name: file_hash(path) for path in outputs},
    })


if __name__ == "__main__":
    main()
