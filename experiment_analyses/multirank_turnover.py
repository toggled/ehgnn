#!/usr/bin/env python3
"""Isolated audit of multi-rank deterministic top-k turnover certificates."""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from training_utils import clone_state_dict, setup_seed
from experiment_analyses.theory_invariants import (
    build_model,
    dataset_context,
    file_hash,
    read_json,
    selected,
    timestamp,
    verify_protocol,
    write_csv,
    write_json,
)


DEFAULT_PROTOCOL = Path("experiment_specs/rank_movement.json")


def topk_multirank_state(
    logits: torch.Tensor,
    k: int,
    maximum_rank_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Return the top-k mask and Gamma_r for r=1,...,r_cap."""
    t = int(logits.numel())
    if not 0 < k < t:
        raise ValueError(f"Multi-rank audit requires 0 < k < t; got k={k}, t={t}")
    maximum_possible = min(k, t - k)
    r_cap = min(maximum_possible, int(maximum_rank_radius))
    if r_cap < 1:
        raise ValueError("maximum_rank_radius must be positive")

    values, indices = torch.topk(
        logits.detach(),
        k=k + r_cap,
        largest=True,
        sorted=True,
    )
    mask = torch.zeros(t, dtype=torch.bool, device=logits.device)
    mask[indices[:k]] = True
    selected_boundary_outward = values[k - r_cap : k].flip(0)
    excluded_boundary_outward = values[k : k + r_cap]
    gammas = selected_boundary_outward - excluded_boundary_outward
    return mask, gammas, r_cap, r_cap == maximum_possible


def multirank_step_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    before_mask: torch.Tensor,
    before_gammas: torch.Tensor,
    k: int,
    r_cap: int,
    complete_search: bool,
    maximum_rank_radius: int,
) -> Tuple[Dict[str, object], torch.Tensor, torch.Tensor, int, bool]:
    """Measure observed turnover and the smallest available certified bound."""
    if before.shape != after.shape:
        raise ValueError("Logit vectors must have the same shape")
    t = int(before.numel())
    if int(before_gammas.numel()) != r_cap:
        raise ValueError("The stored margin profile does not match r_cap")

    update_linf = float((after - before).abs().max().item())
    threshold = 2.0 * update_linf
    qualifying = torch.nonzero(before_gammas > threshold, as_tuple=False).flatten()
    minimal_r = int(qualifying[0].item()) + 1 if qualifying.numel() else None

    maximum_turnover = 2 * min(k, t - k)
    certified_bound = 2 * (minimal_r - 1) if minimal_r is not None else maximum_turnover
    certificate_found = minimal_r is not None
    certifying_gamma = (
        float(before_gammas[minimal_r - 1].item())
        if minimal_r is not None
        else None
    )

    after_mask, after_gammas, after_r_cap, after_complete = topk_multirank_state(
        after, k, maximum_rank_radius
    )
    intersection = int((before_mask & after_mask).sum().item())
    symmetric_difference = int(2 * (k - intersection))
    union = int(2 * k - intersection)
    jaccard = intersection / union if union else 1.0

    if certificate_found and symmetric_difference > certified_bound:
        raise AssertionError(
            "Multi-rank top-k certificate was violated: "
            f"observed={symmetric_difference}, bound={certified_bound}, r={minimal_r}"
        )
    tightness_ratio = (
        certified_bound / symmetric_difference if symmetric_difference > 0 else None
    )
    jaccard_lower_bound = (
        (k - certified_bound / 2.0) / (k + certified_bound / 2.0)
        if certified_bound < 2 * k
        else 0.0
    )
    return {
        "t": t,
        "k": k,
        "linf_logit_update": update_linf,
        "twice_linf_logit_update": threshold,
        "rank_radius_cap": r_cap,
        "rank_search_complete": complete_search,
        "certificate_found_within_cap": certificate_found,
        "minimal_certifying_r": minimal_r,
        "certifying_gamma": certifying_gamma,
        "certificate_slack": (
            certifying_gamma - threshold if certifying_gamma is not None else None
        ),
        "certified_symmetric_difference_bound": certified_bound,
        "maximum_possible_symmetric_difference": maximum_turnover,
        "normalized_certificate_bound": certified_bound / maximum_turnover,
        "observed_symmetric_difference": symmetric_difference,
        "observed_turnover_count": symmetric_difference // 2,
        "observed_jaccard": jaccard,
        "certified_jaccard_lower_bound": jaccard_lower_bound,
        "bound_to_observed_ratio": tightness_ratio,
        "exact_stability_certificate": minimal_r == 1,
        "topk_changed": symmetric_difference > 0,
    }, after_mask, after_gammas, after_r_cap, after_complete


def run_path(outdir: Path, dataset: str, budget: float, seed: int) -> Path:
    return (
        outdir
        / "run_records"
        / f"{dataset}_r{int(round(100 * budget)):03d}_s{seed}.json"
    )


def train_run(
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
    maximum_rank_radius = int(protocol["evaluation"]["maximum_rank_radius"])
    before_mask, before_gammas, r_cap, complete_search = topk_multirank_state(
        model.mask_module.logits, k, maximum_rank_radius
    )
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
        optimizer.step()
        after = model.mask_module.logits.detach()
        step, before_mask, before_gammas, r_cap, complete_search = (
            multirank_step_metrics(
                before,
                after,
                before_mask,
                before_gammas,
                k,
                r_cap,
                complete_search,
                maximum_rank_radius,
            )
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
    bounds = np.asarray(
        [float(row["normalized_certificate_bound"]) for row in trajectory]
    )
    observed = np.asarray(
        [int(row["observed_symmetric_difference"]) for row in trajectory]
    )
    found = np.asarray(
        [bool(row["certificate_found_within_cap"]) for row in trajectory]
    )
    exact = np.asarray([bool(row["exact_stability_certificate"]) for row in trajectory])
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": "rank_movement",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "budget": budget,
        "seed": seed,
        **diagnostics,
        "k": k,
        "rank_radius_cap": r_cap,
        "rank_search_complete": complete_search,
        "epochs_run": len(trajectory),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "end_to_end_seconds": elapsed,
        "fraction_certificate_found_within_cap": float(np.mean(found)),
        "fraction_exact_stability_certificate": float(np.mean(exact)),
        "median_normalized_certificate_bound": float(np.median(bounds)),
        "median_observed_symmetric_difference": float(np.median(observed)),
        "trajectory": trajectory,
    }


def run(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "rank-movement-")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    budgets = selected(args.budgets, protocol["evaluation"]["budgets"], "budgets")
    seeds = selected(args.seeds, protocol["evaluation"]["model_seeds"], "seeds")
    outdir = Path(str(protocol["output_root"]))
    protocol_hash = file_hash(protocol_path)
    device = torch.device(args.device)
    for dataset in datasets:
        for budget in budgets:
            for seed in seeds:
                path = run_path(outdir, dataset, float(budget), int(seed))
                if path.exists() and not args.force:
                    print(
                        f"[rank-movement:skip] {dataset} rho={budget} seed={seed}",
                        flush=True,
                    )
                    continue
                print(f"[rank-movement] {dataset} rho={budget} seed={seed}", flush=True)
                result = train_run(
                    dataset=dataset,
                    budget=float(budget),
                    seed=int(seed),
                    protocol=protocol,
                    protocol_hash=protocol_hash,
                    device=device,
                )
                write_json(path, result)
                print(
                    f"[rank-movement:done] {dataset} rho={budget} seed={seed} "
                    f"epochs={result['epochs_run']} "
                    f"median_bound={result['median_normalized_certificate_bound']:.4f}",
                    flush=True,
                )


def finite_median(values: Sequence[object]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def aggregate(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "rank-movement-")
    outdir = Path(str(protocol["output_root"]))
    expected = [
        run_path(outdir, dataset, float(budget), int(seed))
        for dataset in protocol["datasets"]
        for budget in protocol["evaluation"]["budgets"]
        for seed in protocol["evaluation"]["model_seeds"]
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Rank-movement matrix is incomplete; missing {len(missing)} runs")
    runs = [read_json(path) for path in expected]
    run_rows = []
    trajectories = []
    for item in runs:
        run_rows.append({key: value for key, value in item.items() if key != "trajectory"})
        for row in item["trajectory"]:
            trajectories.append(
                {
                    "dataset_id": item["dataset_id"],
                    "dataset": item["dataset"],
                    "budget": item["budget"],
                    "seed": item["seed"],
                    **row,
                }
            )
    write_csv(outdir / "run_summary.csv", run_rows)
    write_csv(outdir / "step_metrics.csv", trajectories)

    cell_rows: List[Dict[str, object]] = []
    for dataset in protocol["datasets"]:
        for budget in protocol["evaluation"]["budgets"]:
            rows = [
                row
                for row in trajectories
                if row["dataset_id"] == dataset
                and float(row["budget"]) == float(budget)
            ]
            bounds = np.asarray(
                [float(row["normalized_certificate_bound"]) for row in rows]
            )
            actual = np.asarray(
                [int(row["observed_symmetric_difference"]) for row in rows]
            )
            found = np.asarray(
                [bool(row["certificate_found_within_cap"]) for row in rows]
            )
            exact = np.asarray(
                [bool(row["exact_stability_certificate"]) for row in rows]
            )
            violations = np.asarray(
                [
                    int(row["observed_symmetric_difference"])
                    > int(row["certified_symmetric_difference_bound"])
                    for row in rows
                ]
            )
            cell_rows.append(
                {
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "num_optimizer_steps": len(rows),
                    "fraction_certificate_found_within_cap": float(np.mean(found)),
                    "fraction_exact_stability_certificate": float(np.mean(exact)),
                    "fraction_bound_at_most_1pct_of_maximum": float(np.mean(bounds <= 0.01)),
                    "fraction_bound_at_most_10pct_of_maximum": float(np.mean(bounds <= 0.10)),
                    "median_normalized_certificate_bound": float(np.median(bounds)),
                    "median_certified_symmetric_difference_bound": float(
                        np.median(
                            [int(row["certified_symmetric_difference_bound"]) for row in rows]
                        )
                    ),
                    "median_observed_symmetric_difference": float(np.median(actual)),
                    "median_observed_jaccard": float(
                        np.median([float(row["observed_jaccard"]) for row in rows])
                    ),
                    "median_bound_to_observed_ratio_changed_steps": finite_median(
                        [row["bound_to_observed_ratio"] for row in rows]
                    ),
                    "theorem_violations": int(np.sum(violations)),
                }
            )
    write_csv(outdir / "cell_summary.csv", cell_rows)

    pooled_bounds = np.asarray(
        [float(row["normalized_certificate_bound"]) for row in trajectories]
    )
    pooled = {
        "num_optimizer_steps": len(trajectories),
        "fraction_certificate_found_within_cap": float(
            np.mean([bool(row["certificate_found_within_cap"]) for row in trajectories])
        ),
        "fraction_exact_stability_certificate": float(
            np.mean([bool(row["exact_stability_certificate"]) for row in trajectories])
        ),
        "fraction_bound_at_most_1pct_of_maximum": float(np.mean(pooled_bounds <= 0.01)),
        "fraction_bound_at_most_10pct_of_maximum": float(np.mean(pooled_bounds <= 0.10)),
        "median_normalized_certificate_bound": float(np.median(pooled_bounds)),
        "median_observed_symmetric_difference": float(
            np.median(
                [int(row["observed_symmetric_difference"]) for row in trajectories]
            )
        ),
        "median_observed_jaccard": float(
            np.median([float(row["observed_jaccard"]) for row in trajectories])
        ),
        "median_bound_to_observed_ratio_changed_steps": finite_median(
            [row["bound_to_observed_ratio"] for row in trajectories]
        ),
        "theorem_violations": int(
            sum(
                int(row["observed_symmetric_difference"])
                > int(row["certified_symmetric_difference_bound"])
                for row in trajectories
            )
        ),
    }
    interpretation = protocol["interpretation_rule"]
    moderate_cells = sum(
        row["median_normalized_certificate_bound"]
        <= float(interpretation["moderate_maximum_fraction"])
        and (
            row["median_bound_to_observed_ratio_changed_steps"] is not None
            and row["median_bound_to_observed_ratio_changed_steps"]
            <= float(interpretation["moderate_tightness_ratio"])
        )
        for row in cell_rows
    )
    strong_cells = sum(
        row["median_normalized_certificate_bound"]
        <= float(interpretation["strong_maximum_fraction"])
        and (
            row["median_bound_to_observed_ratio_changed_steps"] is not None
            and row["median_bound_to_observed_ratio_changed_steps"]
            <= float(interpretation["strong_tightness_ratio"])
        )
        for row in cell_rows
    )
    required = int(interpretation["required_cells"])
    category = "strong" if strong_cells >= required else "moderate" if moderate_cells >= required else "weak"
    summary = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "complete": True,
        "run_count": len(runs),
        "cell_count": len(cell_rows),
        "pooled": pooled,
        "strong_cells": strong_cells,
        "moderate_cells": moderate_cells,
        "interpretation": category,
        "cell_rows": cell_rows,
    }
    write_json(outdir / "summary.json", summary)
    write_markdown_summary(outdir / "summary.md", protocol, summary)


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_markdown_summary(
    path: Path,
    protocol: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    pooled = summary["pooled"]
    lines = [
        "# Score and ranking movement",
        "",
        f"Protocol: `{protocol['protocol_version']}`",
        "",
        "This analysis measures the smallest rank radius $r$ found within the specified",
        "maximum for which",
        "$\\Gamma_r > 2\\|\\mathbf u\\|_\\infty$, and compares the resulting",
        "$2(r-1)$ turnover certificate with the observed consecutive top-$K$ masks.",
        "",
        "| Dataset | $\\rho$ | Steps | Cert. found | Exact cert. | Median bound / max | Median bound | Median observed | Median Jaccard | Median bound / observed | Violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cell_rows"]:
        lines.append(
            f"| {row['dataset']} | {float(row['budget']):.1f} | "
            f"{row['num_optimizer_steps']} | "
            f"{100 * float(row['fraction_certificate_found_within_cap']):.2f}% | "
            f"{100 * float(row['fraction_exact_stability_certificate']):.2f}% | "
            f"{format_number(row['median_normalized_certificate_bound'], 4)} | "
            f"{format_number(row['median_certified_symmetric_difference_bound'], 1)} | "
            f"{format_number(row['median_observed_symmetric_difference'], 1)} | "
            f"{format_number(row['median_observed_jaccard'], 4)} | "
            f"{format_number(row['median_bound_to_observed_ratio_changed_steps'], 2)} | "
            f"{row['theorem_violations']} |"
        )
    lines.extend(
        [
            "",
            "## Pooled Result",
            "",
            f"- Optimizer transitions: {pooled['num_optimizer_steps']}",
            f"- Certificate found within cap: {100 * float(pooled['fraction_certificate_found_within_cap']):.2f}%",
            f"- Exact-stability certificate: {100 * float(pooled['fraction_exact_stability_certificate']):.4f}%",
            f"- Median certified bound / maximum turnover: {format_number(pooled['median_normalized_certificate_bound'], 4)}",
            f"- Median observed symmetric difference: {format_number(pooled['median_observed_symmetric_difference'], 1)}",
            f"- Median observed Jaccard: {format_number(pooled['median_observed_jaccard'], 4)}",
            f"- Median bound / observed turnover on changed steps: {format_number(pooled['median_bound_to_observed_ratio_changed_steps'], 2)}",
            f"- Theorem violations: {pooled['theorem_violations']}",
            f"- Predeclared interpretation: **{summary['interpretation']}** ({summary['strong_cells']} strong and {summary['moderate_cells']} moderate cells of {summary['cell_count']})",
            "",
            "A certificate is mathematically valid whenever found. The interpretation label",
            "assesses empirical tightness; it is not a theorem-validity criterion.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("run", help="Run the rank-movement analysis")
    execute.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    execute.add_argument("--datasets", nargs="+")
    execute.add_argument("--budgets", nargs="+", type=float)
    execute.add_argument("--seeds", nargs="+", type=int)
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--force", action="store_true")
    execute.set_defaults(func=run)
    summarize = subparsers.add_parser("summarize", help="Summarize the rank-movement runs")
    summarize.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    summarize.set_defaults(func=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
