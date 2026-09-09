#!/usr/bin/env python3
"""Build the six-dataset Table 2 summary and prespecified test families."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from main_accuracy import holm_adjust


WORK_ROOT = Path(".work")
PARENT_SUMMARY = WORK_ROOT / "structural_baselines/all_method_budget_summary.csv"
NEW_SUMMARY = WORK_ROOT / "large_dataset_accuracy/new_dataset_method_summary.csv"
OUTDIR = WORK_ROOT / "large_dataset_accuracy"
DATASETS = ("actor", "twitch", "pokec", "yelp", "coauthor_dblp", "walmart-trips")
BUDGETS = (0.1, 0.2, 0.3, 0.5)
DISPLAY = {
    "actor": "Actor", "twitch": "Twitch", "pokec": "Pokec", "yelp": "Yelp",
    "coauthor_dblp": "DBLP-CA", "walmart-trips": "Walmart",
}
METHODS = ("Full", "EHGNN-F", "Random-Fixed", "Degree-prefix", "Spectral-prefix")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_jsons(path: Path) -> List[Dict[str, object]]:
    return [json.load(item.open()) for item in sorted(path.glob("eval_*.json"))]


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    radius = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return float(array.mean() - radius), float(array.mean() + radius)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    summaries = read_csv(PARENT_SUMMARY) + read_csv(NEW_SUMMARY)
    table = {(row["dataset_id"], float(row["keep_ratio"])): row for row in summaries}
    expected = {(dataset, ratio) for dataset in DATASETS for ratio in BUDGETS}
    if set(table) != expected:
        raise RuntimeError(f"Summary cells differ from six-dataset design: {expected - set(table)}")

    core_records = (
        read_jsons(WORK_ROOT / "main_accuracy/evaluation_runs")
        + read_jsons(WORK_ROOT / "large_dataset_accuracy/evaluation_runs")
    )
    structural_records = (
        read_jsons(WORK_ROOT / "structural_baselines/evaluation_runs")
        + read_jsons(WORK_ROOT / "large_dataset_accuracy/evaluation_runs")
    )
    core = {
        (str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), str(row["method"])): float(row["test_acc"])
        for row in core_records
        if row.get("method") in {"EHGNN-F", "Random-Fixed"}
        and row.get("dataset_id") in DATASETS
    }
    structural = {
        (str(row["dataset_id"]), float(row["keep_ratio"]), int(row["seed"]), str(row["method"])): float(row["test_acc"])
        for row in structural_records
        if row.get("method") in {"Degree-prefix", "Spectral-prefix"}
        and row.get("dataset_id") in DATASETS
    }

    comparisons: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for ratio in BUDGETS:
            learned = np.asarray([core[(dataset, ratio, seed, "EHGNN-F")] for seed in range(10)])
            baselines = {
                "Random-Fixed": np.asarray([
                    core[(dataset, ratio, seed, "Random-Fixed")] for seed in range(10)
                ]),
                "Degree-prefix": np.asarray([
                    structural[(dataset, ratio, seed, "Degree-prefix")] for seed in range(10)
                ]),
                "Spectral-prefix": np.asarray([
                    structural[(dataset, ratio, seed, "Spectral-prefix")] for seed in range(10)
                ]),
            }
            for baseline, values in baselines.items():
                delta = learned - values
                low, high = paired_ci(delta)
                raw = 1.0 if np.allclose(delta, 0.0) else float(
                    stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue
                )
                comparisons.append({
                    "dataset_id": dataset,
                    "dataset": DISPLAY[dataset],
                    "keep_ratio": ratio,
                    "baseline": baseline,
                    "paired_delta_mean": float(delta.mean()),
                    "paired_delta_ci95_low": low,
                    "paired_delta_ci95_high": high,
                    "wins": int((delta > 0).sum()),
                    "ties": int(np.isclose(delta, 0.0).sum()),
                    "wilcoxon_p_raw": raw,
                })
    random_family = [row for row in comparisons if row["baseline"] == "Random-Fixed"]
    structural_family = [row for row in comparisons if row["baseline"] != "Random-Fixed"]
    for family, field in (
        (random_family, "wilcoxon_p_holm_24"),
        (structural_family, "wilcoxon_p_holm_48"),
    ):
        for row, adjusted in zip(family, holm_adjust([float(item["wilcoxon_p_raw"]) for item in family])):
            row[field] = adjusted
    write_csv(OUTDIR / "six_dataset_paired_comparisons.csv", comparisons)

    random_lookup = {
        (str(row["dataset_id"]), float(row["keep_ratio"])): row for row in random_family
    }
    aggregate_rows = []
    for ratio in BUDGETS:
        deltas = np.asarray([
            float(random_lookup[(dataset, ratio)]["paired_delta_mean"]) for dataset in DATASETS
        ])
        low, high = paired_ci(deltas)
        aggregate_rows.append({
            "keep_ratio": ratio,
            "dataset_count": len(DATASETS),
            "equal_dataset_delta_mean": float(deltas.mean()),
            "equal_dataset_delta_ci95_low": low,
            "equal_dataset_delta_ci95_high": high,
            "across_dataset_wilcoxon_p": (
                1.0 if np.allclose(deltas, 0.0) else float(stats.wilcoxon(deltas).pvalue)
            ),
        })
    write_csv(OUTDIR / "six_dataset_aggregate_random_comparison.csv", aggregate_rows)

    lines = [
        "# Six-dataset Table 2 summary", "",
        "Mean +/- sample standard deviation over ten paired model seeds. Bold is the best sparse mean. "
        "`*` marks EHGNN-F significantly above Random-Fixed and `dagger` significantly below Random-Fixed after Holm correction over 24 cells.", "",
        "| Dataset | Budget | Full | EHGNN-F | Random-Fixed | Cardinality | Laplacian-proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for ratio in BUDGETS:
            row = table[(dataset, ratio)]
            sparse = {
                method: float(row[f"{method}_mean"])
                for method in METHODS if method != "Full"
            }
            best = max(sparse.values())
            comparison = random_lookup[(dataset, ratio)]
            adjusted = float(comparison["wilcoxon_p_holm_24"])
            delta = float(comparison["paired_delta_mean"])
            marker = "*" if adjusted < 0.05 and delta > 0 else ("dagger" if adjusted < 0.05 and delta < 0 else "")

            def cell(method: str) -> str:
                value = float(row[f"{method}_mean"])
                text = f"{value:.2f} +/- {float(row[f'{method}_std']):.2f}"
                if method != "Full" and math.isclose(value, best, abs_tol=5e-3):
                    text = f"**{text}**"
                if method == "EHGNN-F" and marker:
                    text += f" {marker}"
                return text

            lines.append(
                f"| {DISPLAY[dataset]} | {100 * ratio:.0f}% | {cell('Full')} | {cell('EHGNN-F')} | "
                f"{cell('Random-Fixed')} | {cell('Degree-prefix')} | {cell('Spectral-prefix')} |"
            )
    lines.extend(["", "## Equal-dataset EHGNN-F minus Random-Fixed", "", "| Budget | Mean delta | 95% CI | Wilcoxon p |", "|---:|---:|---:|---:|"])
    for row in aggregate_rows:
        lines.append(
            f"| {100 * float(row['keep_ratio']):.0f}% | {float(row['equal_dataset_delta_mean']):+.2f} | "
            f"[{float(row['equal_dataset_delta_ci95_low']):+.2f}, {float(row['equal_dataset_delta_ci95_high']):+.2f}] | "
            f"{float(row['across_dataset_wilcoxon_p']):.4f} |"
        )
    (OUTDIR / "table2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
