#!/usr/bin/env python3
"""Aggregate the score and final-mask diagnostics reported in Appendix Table 12."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np

from main_accuracy import load_mask


ROOT = Path(__file__).resolve().parent
DATASETS = (
    ("actor", "Actor"),
    ("twitch", "Twitch"),
    ("pokec", "Pokec"),
    ("yelp", "Yelp"),
    ("coauthor_dblp", "DBLP-CA"),
    ("walmart-trips", "Walmart"),
)
BUDGETS = (0.1, 0.2, 0.3, 0.5)


def source_root(dataset: str) -> Path:
    if dataset in {"coauthor_dblp", "walmart-trips"}:
        return ROOT / ".work/large_dataset_accuracy"
    return ROOT / ".work/main_accuracy"


def mean_pairwise_jaccard(masks) -> float:
    values = []
    for left, right in itertools.combinations(masks, 2):
        intersection = int((left & right).sum().item())
        union = int((left | right).sum().item())
        values.append(intersection / max(1, union))
    return float(np.mean(values))


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.chdir(ROOT)
    movement_rows = []
    agreement_rows = []
    for dataset, display in DATASETS:
        root = source_root(dataset)
        for budget in BUDGETS:
            ratio = int(round(100 * budget))
            records = []
            learned_masks = []
            random_masks = []
            for seed in range(10):
                record_path = root / "evaluation_runs" / f"eval_ehgnnf_{dataset}_r{ratio:03d}_s{seed}.json"
                with record_path.open() as handle:
                    records.append(json.load(handle))
                learned_masks.append(
                    load_mask(root / "masks" / f"mask_ehgnnf_{dataset}_r{ratio:03d}_s{seed}.npz")
                )
                random_masks.append(
                    load_mask(root / "masks" / f"mask_random_{dataset}_r{ratio:03d}_s{seed}.npz")
                )
            if budget in {0.1, 0.5}:
                movement_rows.append(
                    {
                        "dataset": display,
                        "budget": budget,
                        "mean_absolute_weight_change": float(
                            np.mean([row["mean_abs_probability_change"] for row in records])
                        ),
                        "mean_initial_final_jaccard": float(
                            np.mean([row["initial_final_jaccard"] for row in records])
                        ),
                    }
                )
            agreement_rows.append(
                {
                    "dataset": display,
                    "budget": budget,
                    "ehgnn_f_pairwise_jaccard": mean_pairwise_jaccard(learned_masks),
                    "random_fixed_pairwise_jaccard": mean_pairwise_jaccard(random_masks),
                }
            )

    output = ROOT / "outputs"
    write_csv(output / "mask_movement.csv", movement_rows)
    write_csv(output / "final_mask_agreement.csv", agreement_rows)
    print((output / "mask_movement.csv").read_text(), end="")
    print((output / "final_mask_agreement.csv").read_text(), end="")


if __name__ == "__main__":
    main()
