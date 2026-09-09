#!/usr/bin/env python3
"""Collect manuscript-facing summaries under a clean output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Artifact:
    label: str
    source: str
    destination: str
    main_table: bool = False


ARTIFACTS = (
    Artifact(
        "Table 1 dataset statistics",
        "outputs/dataset_statistics.csv",
        "tables/table_1/dataset_statistics.csv",
    ),
    Artifact(
        "Table 2 Actor--Yelp summary",
        ".work/structural_baselines/all_method_budget_summary.csv",
        "tables/table_2/actor_through_yelp.csv",
        True,
    ),
    Artifact(
        "Table 2 Actor--Yelp paired comparisons",
        ".work/structural_baselines/new_baseline_paired_comparisons.csv",
        "tables/table_2/actor_through_yelp_paired_comparisons.csv",
        True,
    ),
    Artifact(
        "Table 2 DBLP-CA and Walmart summary",
        ".work/large_dataset_accuracy/new_dataset_method_summary.csv",
        "tables/table_2/dblp_ca_and_walmart.csv",
        True,
    ),
    Artifact(
        "Table 2 DBLP-CA and Walmart paired comparisons",
        ".work/large_dataset_accuracy/new_dataset_paired_comparisons.csv",
        "tables/table_2/dblp_ca_and_walmart_paired_comparisons.csv",
        True,
    ),
    Artifact(
        "Table 2 six-dataset tests",
        ".work/large_dataset_accuracy/six_dataset_paired_comparisons.csv",
        "tables/table_2/six_dataset_paired_comparisons.csv",
        True,
    ),
    Artifact(
        "Table 3 method summary",
        ".work/random_resampled/method_summary.csv",
        "tables/table_3/method_summary.csv",
        True,
    ),
    Artifact(
        "Table 3 paired comparisons",
        ".work/random_resampled/paired_comparisons.csv",
        "tables/table_3/paired_comparisons.csv",
        True,
    ),
    Artifact(
        "Table 4 method summary",
        ".work/mlp_baseline/method_summary.csv",
        "tables/table_4/method_summary.csv",
        True,
    ),
    Artifact(
        "Table 4 paired comparisons",
        ".work/mlp_baseline/comparison_summary.csv",
        "tables/table_4/paired_comparisons.csv",
        True,
    ),
    Artifact(
        "Table 5 runtime summary",
        ".work/runtime/runtime_summary.csv",
        "tables/table_5/runtime_summary.csv",
    ),
    Artifact(
        "Figure 2 PDF",
        ".work/runtime/figure2_three_panel.pdf",
        "figures/figure_2.pdf",
    ),
    Artifact(
        "Figure 2 PNG",
        ".work/runtime/figure2_three_panel.png",
        "figures/figure_2.png",
    ),
    Artifact(
        "Table 6 sampler-stability summary",
        ".work/sampler_stability/cell_summary.csv",
        "tables/table_6/sampler_stability.csv",
    ),
    Artifact(
        "Table 7 task-aware baselines",
        ".work/task_aware_baselines/unified_method_summary.csv",
        "tables/table_7/task_aware_baselines.csv",
    ),
    Artifact(
        "Table 8 split-robustness summary",
        ".work/split_robustness/method_summary.csv",
        "tables/table_8/method_summary.csv",
    ),
    Artifact(
        "Table 8 paired comparisons",
        ".work/split_robustness/paired_comparisons.csv",
        "tables/table_8/paired_comparisons.csv",
    ),
    Artifact(
        "Table 8 split-level differences",
        ".work/split_robustness/split_level_differences.csv",
        "tables/table_8/split_level_differences.csv",
    ),
    Artifact(
        "Table 9 connected-node strata",
        ".work/connected_nodes/subgroup_summary.csv",
        "tables/table_9/connected_node_strata.csv",
    ),
    Artifact(
        "Table 10 effective-density inputs",
        ".work/retained_structure/seed_results.csv",
        "tables/table_10/effective_density_inputs.csv",
    ),
    Artifact(
        "Table 11 retained relational hyperedges",
        ".work/retained_structure/method_summary.csv",
        "tables/table_11/retained_relational_hyperedges.csv",
    ),
    Artifact(
        "Table 12 score and ranking movement",
        "outputs/mask_movement.csv",
        "tables/table_12/score_and_ranking_movement.csv",
    ),
    Artifact(
        "Table 12 final-mask agreement",
        "outputs/final_mask_agreement.csv",
        "tables/table_12/final_mask_agreement.csv",
    ),
    Artifact(
        "Table 13 retained-mask composition",
        ".work/controlled_corruption/mask_mechanism_summary.csv",
        "tables/table_13/retained_mask_composition.csv",
    ),
    Artifact(
        "Table 14 controlled-corruption summary",
        ".work/controlled_corruption/all_method_cell_summary.csv",
        "tables/table_14/method_summary.csv",
    ),
    Artifact(
        "Table 14 paired comparisons",
        ".work/controlled_corruption/paired_comparisons.csv",
        "tables/table_14/paired_comparisons.csv",
    ),
    Artifact(
        "Table 15 conditioned-scorer summary",
        ".work/conditioned_scorers/method_summary.csv",
        "tables/table_15/method_summary.csv",
    ),
    Artifact(
        "Table 15 paired comparisons",
        ".work/conditioned_scorers/paired_comparisons.csv",
        "tables/table_15/paired_comparisons.csv",
    ),
    Artifact(
        "Table 15 feasibility summary",
        ".work/conditioned_scorers/feasibility_summary.csv",
        "tables/table_15/feasibility_summary.csv",
    ),
    Artifact(
        "Table 16 backbone study",
        ".work/backbone_study/summary.csv",
        "tables/table_16/backbone_study.csv",
    ),
    Artifact(
        "Table 17 hyperedge prediction",
        ".work/hyperedge_prediction/summary.csv",
        "tables/table_17/hyperedge_prediction.csv",
    ),
    Artifact(
        "Selected-gradient analysis",
        ".work/selected_gradients/run_summary.csv",
        "analyses/selected_gradients/run_summary.csv",
    ),
    Artifact(
        "Rank-movement analysis",
        ".work/rank_movement/cell_summary.csv",
        "analyses/rank_movement/cell_summary.csv",
    ),
    Artifact(
        "Late-mask analysis",
        ".work/late_mask_analysis/cell_summary.csv",
        "analyses/late_masks/cell_summary.csv",
    ),
    Artifact(
        "Full-retention equivalence check",
        ".work/full_retention/deterministic_equivalence_gate.json",
        "analyses/full_retention/equivalence_check.json",
    ),
    Artifact(
        "Walmart feature check",
        ".work/walmart_feature_check/summary.json",
        "analyses/walmart_features/summary.json",
    ),
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_artifacts(scope: str) -> Iterable[Artifact]:
    if scope == "main":
        return (artifact for artifact in ARTIFACTS if artifact.main_table)
    return iter(ARTIFACTS)


def display_path(path: Path, root: Path) -> str:
    try:
        return f"./{path.relative_to(root).as_posix()}"
    except ValueError:
        return str(path)


def collect(
    scope: str,
    require_complete: bool,
    source_root: Path = ROOT,
    output_root: Optional[Path] = None,
) -> dict[str, object]:
    output_root = output_root or source_root / "outputs"
    copied = []
    missing = []

    for artifact in selected_artifacts(scope):
        source = source_root / artifact.source
        destination = output_root / artifact.destination
        if not source.is_file():
            missing.append(artifact.label)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        copied.append({
            "label": artifact.label,
            "path": display_path(destination, source_root),
            "sha256": file_hash(destination),
        })

    manifest = {
        "scope": scope,
        "complete": not missing,
        "copied": copied,
        "missing": missing,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)

    for item in copied:
        print(f"Collected {item['label']}: {item['path']}")
    if missing:
        print(f"Not yet available: {len(missing)} expected outputs")
    print(f"Manifest: {display_path(manifest_path, source_root)}")

    if not copied:
        raise SystemExit("No completed experiment summaries were found.")
    if require_complete and missing:
        raise SystemExit(
            "Output collection is incomplete: " + ", ".join(missing)
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("main", "all"), default="all")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(args.scope, args.require_complete)


if __name__ == "__main__":
    main()
