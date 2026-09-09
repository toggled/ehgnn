#!/usr/bin/env python3
"""Measure score-tail exposure and movement for selected-only EHGNN-F training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from training_utils import clone_state_dict, setup_seed
from experiment_analyses.theory_invariants import (
    build_model,
    dataset_context,
    file_hash,
    read_json,
    timestamp,
    verify_protocol,
    write_json,
)


DEFAULT_PROTOCOL = Path("experiment_specs/selected_gradients.json")


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


def rank_deciles(values: torch.Tensor, count: int) -> torch.Tensor:
    """Assign rank deciles 0 (lowest) through count-1 (highest)."""
    order = torch.argsort(values.detach())
    groups = torch.empty_like(order)
    ranks = torch.arange(order.numel(), device=order.device)
    groups[order] = torch.clamp((ranks * count) // order.numel(), max=count - 1)
    return groups


def topk_mask(values: torch.Tensor, k: int) -> torch.Tensor:
    mask = torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    mask[torch.topk(values.detach(), k=k).indices] = True
    return mask


def spearman_no_ties(left: np.ndarray, right: np.ndarray) -> float:
    left_order = np.argsort(left, kind="mergesort")
    right_order = np.argsort(right, kind="mergesort")
    left_rank = np.empty(left.size, dtype=np.float64)
    right_rank = np.empty(right.size, dtype=np.float64)
    left_rank[left_order] = np.arange(left.size, dtype=np.float64)
    right_rank[right_order] = np.arange(right.size, dtype=np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def redact_test_labels(data, split: Mapping[str, torch.Tensor]) -> None:
    allowed = torch.zeros(data.y.numel(), dtype=torch.bool, device=data.y.device)
    allowed[split["train"]] = True
    allowed[split["valid"]] = True
    data.y[~allowed] = -1


def summarize_fixed_groups(
    *,
    label: str,
    groups: np.ndarray,
    group_count: int,
    selection_counts: np.ndarray,
    gradient_counts: np.ndarray,
    epochs: int,
    initial_logits: np.ndarray,
    final_logits: np.ndarray,
    final_topk: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    movement = np.abs(final_logits - initial_logits)
    for group in range(group_count):
        member = groups == group
        if not np.any(member):
            continue
        rows.append(
            {
                "grouping": label,
                "decile": group,
                "incidence_count": int(member.sum()),
                "mean_selection_rate": float(selection_counts[member].mean() / epochs),
                "minimum_selection_count": int(selection_counts[member].min()),
                "fraction_ever_selected": float(np.mean(selection_counts[member] > 0)),
                "mean_nonzero_gradient_rate": float(
                    gradient_counts[member].mean() / epochs
                ),
                "mean_absolute_logit_change": float(movement[member].mean()),
                "median_absolute_logit_change": float(np.median(movement[member])),
                "fraction_in_final_topk": float(np.mean(final_topk[member])),
            }
        )
    return rows


def run_path(outdir: Path, seed: int) -> Path:
    return outdir / "run_records" / f"actor_r050_s{seed}.json"


def array_path(outdir: Path, seed: int) -> Path:
    return outdir / "arrays" / f"actor_r050_s{seed}.npz"


def train_run(
    *,
    seed: int,
    protocol: Mapping[str, object],
    protocol_hash: str,
    device: torch.device,
) -> Dict[str, object]:
    dataset = str(protocol["evaluation"]["dataset"])
    budget = float(protocol["evaluation"]["budget"])
    data, num_features, num_classes, split, diagnostics = dataset_context(
        dataset, protocol
    )
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: value.to(device) for name, value in split.items()}
    redact_test_labels(local_data, local_split)
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

    logits_parameter = model.mask_module.logits
    initial_logits_tensor = logits_parameter.detach().clone()
    initial_logits = initial_logits_tensor.cpu().numpy().astype(np.float32)
    t = int(initial_logits.size)
    k = max(1, int(budget * t))
    decile_count = int(protocol["evaluation"]["rank_group_count"])
    initial_groups_tensor = rank_deciles(initial_logits_tensor, decile_count)
    initial_groups = initial_groups_tensor.cpu().numpy().astype(np.int8)

    selected_history: List[np.ndarray] = []
    gradient_history: List[np.ndarray] = []
    dynamic_rows: List[Dict[str, object]] = []
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    wait = 0

    for epoch in range(int(training["maximum_epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = logits_parameter.detach().clone()
        current_groups = rank_deciles(before, decile_count)
        output, _, hard_mask = model(local_data, is_test=False, return_mask=True)
        selected = hard_mask.detach().bool()
        loss = F.cross_entropy(
            output[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at seed={seed}, epoch={epoch}")
        loss.backward()
        gradient = logits_parameter.grad
        if gradient is None:
            raise RuntimeError("The incidence logits did not receive a gradient tensor")
        gradient_nonzero = gradient.detach().ne(0)
        optimizer.step()
        update = logits_parameter.detach() - before

        selected_history.append(selected.cpu().numpy())
        gradient_history.append(gradient_nonzero.cpu().numpy())
        for group in range(decile_count):
            member = current_groups == group
            selected_member = member & selected
            unselected_member = member & ~selected
            selected_count = int(selected_member.sum().item())
            nonzero_count = int((member & gradient_nonzero).sum().item())
            positive_count = int((selected_member & (update > 0)).sum().item())
            negative_count = int((selected_member & (update < 0)).sum().item())
            dynamic_rows.append(
                {
                    "epoch": epoch,
                    "decile": group,
                    "incidence_count": int(member.sum().item()),
                    "selected_count": selected_count,
                    "nonzero_gradient_count": nonzero_count,
                    "selected_positive_update_count": positive_count,
                    "selected_negative_update_count": negative_count,
                    "selected_zero_update_count": selected_count
                    - positive_count
                    - negative_count,
                    "unselected_nonzero_update_count": int(
                        (unselected_member & update.ne(0)).sum().item()
                    ),
                    "mean_absolute_update": float(update[member].abs().mean().item()),
                }
            )

        with torch.no_grad():
            model.eval()
            validation_output = model(local_data, is_test=True)
            validation_loss = float(
                F.cross_entropy(
                    validation_output[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                ).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if wait >= int(training["patience"]):
                break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("No validation-selected checkpoint was produced")

    checkpoint_epochs = best_epoch + 1
    selected_matrix = np.stack(selected_history[:checkpoint_epochs]).astype(bool)
    gradient_matrix = np.stack(gradient_history[:checkpoint_epochs]).astype(bool)
    selection_counts = selected_matrix.sum(axis=0).astype(np.int32)
    gradient_counts = gradient_matrix.sum(axis=0).astype(np.int32)
    final_logits = (
        best_state["mask_module.logits"].detach().cpu().numpy().astype(np.float32)
    )
    final_groups_tensor = rank_deciles(
        torch.from_numpy(final_logits), decile_count
    )
    final_groups = final_groups_tensor.numpy().astype(np.int8)
    final_topk = np.zeros(t, dtype=bool)
    final_topk[np.argpartition(final_logits, -k)[-k:]] = True
    initial_topk = np.zeros(t, dtype=bool)
    initial_topk[np.argpartition(initial_logits, -k)[-k:]] = True
    intersection = int(np.logical_and(initial_topk, final_topk).sum())
    union = int(np.logical_or(initial_topk, final_topk).sum())

    fixed_group_rows = summarize_fixed_groups(
        label="initial_rank",
        groups=initial_groups,
        group_count=decile_count,
        selection_counts=selection_counts,
        gradient_counts=gradient_counts,
        epochs=checkpoint_epochs,
        initial_logits=initial_logits,
        final_logits=final_logits,
        final_topk=final_topk,
    )
    fixed_group_rows.extend(
        summarize_fixed_groups(
            label="final_rank",
            groups=final_groups,
            group_count=decile_count,
            selection_counts=selection_counts,
            gradient_counts=gradient_counts,
            epochs=checkpoint_epochs,
            initial_logits=initial_logits,
            final_logits=final_logits,
            final_topk=final_topk,
        )
    )

    dynamic_checkpoint_rows = [
        row for row in dynamic_rows if int(row["epoch"]) <= best_epoch
    ]
    dynamic_summary: List[Dict[str, object]] = []
    for group in range(decile_count):
        rows = [row for row in dynamic_checkpoint_rows if int(row["decile"]) == group]
        available = sum(int(row["incidence_count"]) for row in rows)
        selected_count = sum(int(row["selected_count"]) for row in rows)
        nonzero_count = sum(int(row["nonzero_gradient_count"]) for row in rows)
        positive_count = sum(
            int(row["selected_positive_update_count"]) for row in rows
        )
        negative_count = sum(
            int(row["selected_negative_update_count"]) for row in rows
        )
        dynamic_summary.append(
            {
                "decile": group,
                "incidence_epoch_count": available,
                "selection_rate": selected_count / available,
                "nonzero_gradient_rate": nonzero_count / available,
                "nonzero_gradient_given_selected": (
                    nonzero_count / selected_count if selected_count else None
                ),
                "positive_update_given_selected": (
                    positive_count / selected_count if selected_count else None
                ),
                "negative_update_given_selected": (
                    negative_count / selected_count if selected_count else None
                ),
            }
        )

    arrays = array_path(Path(str(protocol["output_root"])), seed)
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays,
        initial_logits=initial_logits,
        checkpoint_logits=final_logits,
        selection_counts=selection_counts,
        nonzero_gradient_counts=gradient_counts,
        initial_rank_deciles=initial_groups,
        checkpoint_rank_deciles=final_groups,
        initial_topk=initial_topk,
        checkpoint_topk=final_topk,
    )

    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "budget": budget,
        "seed": seed,
        **diagnostics,
        "t": t,
        "k": k,
        "epochs_run": len(selected_history),
        "checkpoint_epoch": best_epoch,
        "checkpoint_epochs_inclusive": checkpoint_epochs,
        "checkpoint_validation_loss": best_loss,
        "fraction_ever_selected": float(np.mean(selection_counts > 0)),
        "minimum_selection_count": int(selection_counts.min()),
        "median_selection_count": float(np.median(selection_counts)),
        "maximum_selection_count": int(selection_counts.max()),
        "fraction_ever_nonzero_gradient": float(np.mean(gradient_counts > 0)),
        "initial_checkpoint_spearman": spearman_no_ties(
            initial_logits, final_logits
        ),
        "initial_topk_survival_fraction": intersection / k,
        "initial_checkpoint_topk_jaccard": intersection / union,
        "chance_topk_survival_fraction": k / t,
        "fixed_group_summary": fixed_group_rows,
        "dynamic_rank_summary": dynamic_summary,
        "array_path": str(arrays),
    }


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "selected-gradients-")
    allowed = [int(seed) for seed in protocol["evaluation"]["model_seeds"]]
    seeds: Sequence[int] = allowed if args.seeds is None else args.seeds
    if not set(seeds) <= set(allowed):
        raise ValueError(f"Requested seeds outside the experiment specification: {seeds}")
    outdir = Path(str(protocol["output_root"]))
    protocol_hash = file_hash(protocol_path)
    for seed in seeds:
        path = run_path(outdir, int(seed))
        if path.exists() and not args.force:
            print(f"[selected-gradients:skip] seed={seed}", flush=True)
            continue
        print(f"[selected-gradients] seed={seed} device={args.device}", flush=True)
        result = train_run(
            seed=int(seed),
            protocol=protocol,
            protocol_hash=protocol_hash,
            device=torch.device(args.device),
        )
        write_json(path, result)
        print(
            f"[selected-gradients:done] seed={seed} checkpoint={result['checkpoint_epoch']} "
            f"coverage={result['fraction_ever_selected']:.4f}",
            flush=True,
        )


def aggregate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "selected-gradients-")
    outdir = Path(str(protocol["output_root"]))
    records = [
        read_json(run_path(outdir, int(seed)))
        for seed in protocol["evaluation"]["model_seeds"]
    ]
    run_rows: List[Dict[str, object]] = []
    dynamic_rows: List[Dict[str, object]] = []
    fixed_rows: List[Dict[str, object]] = []
    for record in records:
        run_rows.append(
            {
                key: value
                for key, value in record.items()
                if key
                in {
                    "seed",
                    "epochs_run",
                    "checkpoint_epoch",
                    "fraction_ever_selected",
                    "minimum_selection_count",
                    "median_selection_count",
                    "maximum_selection_count",
                    "fraction_ever_nonzero_gradient",
                    "initial_checkpoint_spearman",
                    "initial_topk_survival_fraction",
                    "initial_checkpoint_topk_jaccard",
                    "chance_topk_survival_fraction",
                }
            }
        )
        for row in record["dynamic_rank_summary"]:
            dynamic_rows.append({"seed": record["seed"], **row})
        for row in record["fixed_group_summary"]:
            fixed_rows.append({"seed": record["seed"], **row})

    write_csv(outdir / "run_summary.csv", run_rows)
    write_csv(outdir / "dynamic_rank_summary.csv", dynamic_rows)
    write_csv(outdir / "fixed_rank_summary.csv", fixed_rows)

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in run_rows], dtype=np.float64)

    def dynamic_values(decile: int, key: str) -> np.ndarray:
        return np.asarray(
            [
                float(row[key])
                for row in dynamic_rows
                if int(row["decile"]) == decile
            ],
            dtype=np.float64,
        )

    def fixed_values(grouping: str, decile: int, key: str) -> np.ndarray:
        return np.asarray(
            [
                float(row[key])
                for row in fixed_rows
                if row["grouping"] == grouping and int(row["decile"]) == decile
            ],
            dtype=np.float64,
        )

    aggregate_result = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "run_count": len(records),
        "all_incidences_ever_selected_in_every_run": bool(
            all(float(row["fraction_ever_selected"]) == 1.0 for row in run_rows)
        ),
        "fraction_ever_selected_range": [
            float(values("fraction_ever_selected").min()),
            float(values("fraction_ever_selected").max()),
        ],
        "minimum_selection_count_range": [
            int(values("minimum_selection_count").min()),
            int(values("minimum_selection_count").max()),
        ],
        "dynamic_bottom_decile_selection_rate_range": [
            float(dynamic_values(0, "selection_rate").min()),
            float(dynamic_values(0, "selection_rate").max()),
        ],
        "dynamic_top_decile_selection_rate_range": [
            float(dynamic_values(9, "selection_rate").min()),
            float(dynamic_values(9, "selection_rate").max()),
        ],
        "dynamic_bottom_decile_nonzero_gradient_rate_range": [
            float(dynamic_values(0, "nonzero_gradient_rate").min()),
            float(dynamic_values(0, "nonzero_gradient_rate").max()),
        ],
        "dynamic_bottom_decile_positive_update_given_selected_range": [
            float(dynamic_values(0, "positive_update_given_selected").min()),
            float(dynamic_values(0, "positive_update_given_selected").max()),
        ],
        "dynamic_bottom_decile_negative_update_given_selected_range": [
            float(dynamic_values(0, "negative_update_given_selected").min()),
            float(dynamic_values(0, "negative_update_given_selected").max()),
        ],
        "initial_bottom_decile_logit_change_range": [
            float(fixed_values("initial_rank", 0, "mean_absolute_logit_change").min()),
            float(fixed_values("initial_rank", 0, "mean_absolute_logit_change").max()),
        ],
        "initial_top_decile_logit_change_range": [
            float(fixed_values("initial_rank", 9, "mean_absolute_logit_change").min()),
            float(fixed_values("initial_rank", 9, "mean_absolute_logit_change").max()),
        ],
        "final_bottom_decile_selection_rate_range": [
            float(fixed_values("final_rank", 0, "mean_selection_rate").min()),
            float(fixed_values("final_rank", 0, "mean_selection_rate").max()),
        ],
        "initial_checkpoint_spearman_range": [
            float(values("initial_checkpoint_spearman").min()),
            float(values("initial_checkpoint_spearman").max()),
        ],
        "initial_topk_survival_fraction_range": [
            float(values("initial_topk_survival_fraction").min()),
            float(values("initial_topk_survival_fraction").max()),
        ],
        "chance_topk_survival_fraction": float(
            values("chance_topk_survival_fraction")[0]
        ),
    }
    write_json(outdir / "summary.json", aggregate_result)

    markdown = [
        "# Selected-gradient exposure",
        "",
        "Five validation-only Actor runs at 50% retention. Values are descriptive.",
        "",
        "| Quantity | Across-seed range |",
        "|---|---:|",
        "| Fraction of incidences sampled at least once | "
        f"{aggregate_result['fraction_ever_selected_range'][0]:.4f}--"
        f"{aggregate_result['fraction_ever_selected_range'][1]:.4f} |",
        "| Minimum selections received by any incidence | "
        f"{aggregate_result['minimum_selection_count_range'][0]}--"
        f"{aggregate_result['minimum_selection_count_range'][1]} |",
        "| Current bottom-decile selection rate | "
        f"{aggregate_result['dynamic_bottom_decile_selection_rate_range'][0]:.4f}--"
        f"{aggregate_result['dynamic_bottom_decile_selection_rate_range'][1]:.4f} |",
        "| Current top-decile selection rate | "
        f"{aggregate_result['dynamic_top_decile_selection_rate_range'][0]:.4f}--"
        f"{aggregate_result['dynamic_top_decile_selection_rate_range'][1]:.4f} |",
        "| Initial-bottom-decile mean absolute logit change | "
        f"{aggregate_result['initial_bottom_decile_logit_change_range'][0]:.5f}--"
        f"{aggregate_result['initial_bottom_decile_logit_change_range'][1]:.5f} |",
        "| Initial-top-decile mean absolute logit change | "
        f"{aggregate_result['initial_top_decile_logit_change_range'][0]:.5f}--"
        f"{aggregate_result['initial_top_decile_logit_change_range'][1]:.5f} |",
        "| Initial--checkpoint Spearman correlation | "
        f"{aggregate_result['initial_checkpoint_spearman_range'][0]:.4f}--"
        f"{aggregate_result['initial_checkpoint_spearman_range'][1]:.4f} |",
        "| Initial top-K survival fraction | "
        f"{aggregate_result['initial_topk_survival_fraction_range'][0]:.4f}--"
        f"{aggregate_result['initial_topk_survival_fraction_range'][1]:.4f} |",
        "| Chance top-K survival fraction | "
        f"{aggregate_result['chance_topk_survival_fraction']:.4f} |",
        "",
        "Complete per-seed summaries are in `run_summary.csv`; rank-stratified "
        "results are in `dynamic_rank_summary.csv` and `fixed_rank_summary.csv`.",
    ]
    (outdir / "summary.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(aggregate_result, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "aggregate"])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "run":
        run(arguments)
    else:
        aggregate(arguments)
