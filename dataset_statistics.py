#!/usr/bin/env python3
"""Generate the node-classification dataset statistics reported in Table 1."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from experiment_controls.core_controls import dataset_context


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "experiment_specs/random_resampled.json"
DATASETS = ("actor", "twitch", "pokec", "yelp", "coauthor_dblp", "walmart-trips")


def homophily(data, num_classes: int) -> float:
    node_ids = data.edge_index[0].long().cpu()
    edge_ids = data.edge_index[1].long().cpu()
    labels = data.y.long().cpu()
    num_edges = int(data.num_hyperedges)
    degrees = torch.bincount(edge_ids, minlength=num_edges).to(torch.float64)
    encoded = edge_ids * num_classes + labels[node_ids]
    class_counts = torch.bincount(
        encoded, minlength=num_edges * num_classes
    ).reshape(num_edges, num_classes).to(torch.float64)
    same_class_pairs = (class_counts * (class_counts - 1.0) / 2.0).sum(dim=1)
    all_pairs = degrees * (degrees - 1.0) / 2.0
    valid = degrees >= 2
    return float((same_class_pairs[valid] / all_pairs[valid]).mean().item())


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.chdir(ROOT)
    with PROTOCOL.open() as handle:
        protocol = json.load(handle)
    rows = []
    for dataset in DATASETS:
        data, num_features, num_classes, _, _ = dataset_context(dataset, protocol)
        rows.append(
            {
                "dataset": protocol["display_names"][dataset],
                "nodes": int(data.n_x),
                "hyperedges": int(data.num_hyperedges),
                "incidences": int(data.edge_index.size(1)),
                "features": int(num_features),
                "classes": int(num_classes),
                "hyperedge_homophily": homophily(data, num_classes),
            }
        )

    output = ROOT / "outputs" / "dataset_statistics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output.read_text(), end="")


if __name__ == "__main__":
    main()
