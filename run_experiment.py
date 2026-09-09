#!/usr/bin/env python3
"""Run a supplementary experiment using a descriptive public name."""

from __future__ import annotations

import runpy
import sys


EXPERIMENTS = {
    "connected-nodes": (
        "experiment_analyses.connected_stratification",
        (),
        "Connected- and isolated-node analysis (Table 9)",
    ),
    "controlled-corruption": (
        "controlled_corruption",
        (
            "--protocol",
            "experiment_specs/controlled_corruption.json",
            "--outdir",
            ".work/controlled_corruption",
        ),
        "Controlled incidence-corruption study (Tables 13--14)",
    ),
    "core-controls": (
        "experiment_controls.core_controls",
        (),
        "Random-Resampled and feature-only controls (Tables 3--4)",
    ),
    "full-retention-check": (
        "experiment_analyses.full_collapse",
        (),
        "Exact-equivalence check at full retention",
    ),
    "herald": (
        "experiment_analyses.herald_classification",
        (),
        "HERALD validation selection and evaluation (Table 7)",
    ),
    "hsl": (
        "experiment_analyses.hsl_realized",
        (),
        "HSL validation selection and evaluation (Table 7)",
    ),
    "hsl-summary": (
        "experiment_analyses.hsl_summary",
        (),
        "HSL result summary",
    ),
    "late-mask-analysis": (
        "experiment_analyses.mask_equivalence",
        (),
        "Late-training mask analysis (Table 12)",
    ),
    "rank-movement": (
        "experiment_analyses.multirank_turnover",
        (),
        "Score and ranking movement analysis",
    ),
    "retained-structure": (
        "experiment_controls.retained_structure",
        (),
        "Effective density and retained multi-node structure (Tables 10--11)",
    ),
    "sampler-stability": (
        "experiment_analyses.theory_invariants",
        (),
        "Sampler-stability analysis (Table 6)",
    ),
    "selected-gradients": (
        "experiment_analyses.selected_gradient_dynamics",
        (),
        "Selected-gradient exposure analysis (Table 12)",
    ),
    "split-robustness": (
        "experiment_controls.split_robustness",
        (),
        "Robustness over independently generated splits (Table 8)",
    ),
    "task-aware-summary": (
        "experiment_analyses.task_aware_baseline_summary",
        (),
        "Combined HSL and HERALD result summary (Table 7)",
    ),
}


def print_usage(stream=sys.stdout) -> None:
    print("Usage: python run_experiment.py <experiment> [arguments]", file=stream)
    print("\nAvailable experiments:", file=stream)
    width = max(len(name) for name in EXPERIMENTS)
    for name, (_, _, description) in EXPERIMENTS.items():
        print(f"  {name:<{width}}  {description}", file=stream)


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print_usage()
        return

    experiment = sys.argv[1]
    if experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment}", file=sys.stderr)
        print_usage(sys.stderr)
        raise SystemExit(2)

    module, fixed_arguments, _ = EXPERIMENTS[experiment]
    sys.argv = [f"{sys.argv[0]} {experiment}", *fixed_arguments, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
