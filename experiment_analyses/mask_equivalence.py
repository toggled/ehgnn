#!/usr/bin/env python3
"""Compare late-training masks and boundary-randomized alternatives."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
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


DEFAULT_PROTOCOL = Path("experiment_specs/late_mask_analysis.json")
FAMILIES = ("late", "boundary")


def checkpoint_epochs(total_epochs: int, fractions: Sequence[float]) -> List[int]:
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    epochs = [
        int(math.floor(float(fraction) * (total_epochs - 1) + 0.5))
        for fraction in fractions
    ]
    if epochs != sorted(set(epochs)):
        raise ValueError(f"Checkpoint fractions do not produce distinct epochs: {epochs}")
    return epochs


def core_boundary_masks(
    late_masks: torch.Tensor,
    seeds: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if late_masks.ndim != 2 or late_masks.dtype != torch.bool:
        raise ValueError("late_masks must be a two-dimensional boolean tensor")
    if late_masks.size(0) != len(seeds):
        raise ValueError("One boundary-randomization seed is required per output mask")
    counts = late_masks.sum(dim=1)
    if not bool(torch.all(counts == counts[0])):
        raise ValueError("All late masks must have the same exact budget")
    k = int(counts[0].item())
    core = late_masks.all(dim=0)
    union = late_masks.any(dim=0)
    candidates = torch.where(union & ~core)[0]
    slots = k - int(core.sum().item())
    if candidates.numel() < slots:
        raise RuntimeError("Unstable boundary does not contain enough candidates")

    randomized = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        chosen = candidates[
            torch.randperm(candidates.numel(), generator=generator)[:slots]
        ]
        mask = core.clone()
        mask[chosen] = True
        if int(mask.sum()) != k or not bool(torch.all(mask[core])):
            raise AssertionError("Boundary randomization violated core or budget")
        if bool((mask & ~union).any()):
            raise AssertionError("Boundary randomization selected outside the late union")
        randomized.append(mask)
    return torch.stack(randomized), core, union


def mask_jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = int((left & right).sum().item())
    union = int((left | right).sum().item())
    return intersection / union if union else 1.0


def mask_hash(mask: torch.Tensor) -> str:
    value = mask.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def save_bundle(
    path: Path,
    *,
    late_masks: torch.Tensor,
    boundary_masks: torch.Tensor,
    core: torch.Tensor,
    union: torch.Tensor,
    epochs: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        late=np.packbits(late_masks.numpy(), axis=1, bitorder="little"),
        boundary=np.packbits(boundary_masks.numpy(), axis=1, bitorder="little"),
        core=np.packbits(core.numpy(), bitorder="little"),
        union=np.packbits(union.numpy(), bitorder="little"),
        size=np.asarray([late_masks.size(1)], dtype=np.int64),
        epochs=np.asarray(epochs, dtype=np.int64),
    )
    temporary.replace(path)


def load_bundle(path: Path) -> Dict[str, object]:
    with np.load(path) as bundle:
        size = int(bundle["size"][0])
        result: Dict[str, object] = {
            "epochs": [int(value) for value in bundle["epochs"]],
            "size": size,
        }
        for family in FAMILIES:
            unpacked = np.unpackbits(
                bundle[family], axis=1, count=size, bitorder="little"
            )
            result[family] = torch.from_numpy(unpacked.astype(np.bool_))
        for name in ("core", "union"):
            unpacked = np.unpackbits(
                bundle[name], count=size, bitorder="little"
            )
            result[name] = torch.from_numpy(unpacked.astype(np.bool_))
    return result


def bundle_path(outdir: Path, dataset: str, budget: float, source_seed: int) -> Path:
    return (
        outdir
        / "masks"
        / f"{dataset}_r{int(round(100 * budget)):03d}_ss{source_seed}.npz"
    )


def collection_path(outdir: Path, dataset: str, budget: float, source_seed: int) -> Path:
    return (
        outdir
        / "collection"
        / f"{dataset}_r{int(round(100 * budget)):03d}_ss{source_seed}.json"
    )


def evaluation_path(
    outdir: Path,
    dataset: str,
    budget: float,
    source_seed: int,
    family: str,
    variant: int,
) -> Path:
    return (
        outdir
        / "evaluation"
        / (
            f"{dataset}_r{int(round(100 * budget)):03d}_ss{source_seed}_"
            f"{family}_m{variant}.json"
        )
    )


def source_epoch_map(protocol: Mapping[str, object]) -> Dict[Tuple[str, float, int], int]:
    path = Path(str(protocol["source_runs"]["run_summary_path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            result[(row["dataset_id"], float(row["budget"]), int(row["seed"]))] = int(
                row["epochs_run"]
            )
    return result


def boundary_seed(
    protocol: Mapping[str, object],
    dataset: str,
    budget: float,
    source_seed: int,
    variant: int,
) -> int:
    offset = list(protocol["datasets"]).index(dataset)
    base = int(protocol["mask_generation"]["boundary_seed_base"])
    return base + offset * 100000 + int(round(100 * budget)) * 100 + source_seed * 10 + variant


def redacted_context(dataset: str, protocol: Mapping[str, object]):
    data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
    visible = split["train"] | split["valid"]
    data.y = data.y.clone()
    data.y[~visible] = -1
    if bool((data.y[~visible] != -1).any()):
        raise AssertionError("Test labels were not redacted")
    diagnostics["redacted_label_count"] = int((~visible).sum().item())
    diagnostics["test_labels_redacted"] = True
    return data, num_features, num_classes, split, diagnostics


def collect_one(
    *,
    dataset: str,
    budget: float,
    source_seed: int,
    total_epochs: int,
    protocol: Mapping[str, object],
    protocol_hash: str,
    device: torch.device,
) -> Dict[str, object]:
    data, num_features, num_classes, split, diagnostics = redacted_context(dataset, protocol)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: mask.to(device) for name, mask in split.items()}
    training = protocol["scorer_replay"]
    setup_seed(source_seed)
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
    epochs = checkpoint_epochs(
        total_epochs, protocol["mask_generation"]["late_epoch_fractions"]
    )
    targets = set(epochs)
    t = int(local_data.edge_index.size(1))
    k = max(1, int(budget * t))
    masks: List[torch.Tensor] = []
    val_losses: List[float] = []
    start = time.perf_counter()

    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(local_data, is_test=False)
        loss = F.cross_entropy(
            logits[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite scorer-replay loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            val_logits = model(local_data, is_test=True)
            val_loss = float(
                F.cross_entropy(
                    val_logits[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                ).item()
            )
        if epoch in targets:
            indices = torch.topk(model.mask_module.logits.detach(), k=k).indices.cpu()
            mask = torch.zeros(t, dtype=torch.bool)
            mask[indices] = True
            masks.append(mask)
            val_losses.append(val_loss)

    if len(masks) != len(epochs):
        raise RuntimeError("Failed to capture every specified late checkpoint")
    late_masks = torch.stack(masks)
    boundary_seeds = [
        boundary_seed(protocol, dataset, budget, source_seed, variant)
        for variant in range(len(epochs))
    ]
    boundary_masks, core, union = core_boundary_masks(late_masks, boundary_seeds)
    outdir = Path(str(protocol["output_root"]))
    masks_path = bundle_path(outdir, dataset, budget, source_seed)
    save_bundle(
        masks_path,
        late_masks=late_masks,
        boundary_masks=boundary_masks,
        core=core,
        union=union,
        epochs=epochs,
    )
    pairwise_late = [
        mask_jaccard(late_masks[left], late_masks[right])
        for left, right in itertools.combinations(range(len(epochs)), 2)
    ]
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": "late_mask_collection",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "budget": budget,
        "source_seed": source_seed,
        "source_total_epochs": total_epochs,
        "captured_epochs": epochs,
        "captured_validation_losses": val_losses,
        "num_masks_per_family": len(epochs),
        "t": t,
        "k": k,
        "core_count": int(core.sum().item()),
        "core_fraction_of_k": float(core.sum().item() / k),
        "late_union_count": int(union.sum().item()),
        "late_union_fraction_of_t": float(union.sum().item() / t),
        "median_within_run_late_jaccard": float(np.median(pairwise_late)),
        "minimum_within_run_late_jaccard": float(np.min(pairwise_late)),
        "bundle_path": str(masks_path),
        "bundle_sha256": file_hash(masks_path),
        "end_to_end_seconds": time.perf_counter() - start,
        **diagnostics,
    }


def collect(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "late-mask-analysis-")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    budgets = selected(args.budgets, protocol["evaluation"]["budgets"], "budgets")
    seeds = selected(args.seeds, protocol["evaluation"]["source_seeds"], "seeds")
    epoch_map = source_epoch_map(protocol)
    outdir = Path(str(protocol["output_root"]))
    digest = file_hash(protocol_path)
    device = torch.device(args.device)
    for dataset in datasets:
        for budget in budgets:
            for seed in seeds:
                result_path = collection_path(outdir, dataset, float(budget), int(seed))
                masks_path = bundle_path(outdir, dataset, float(budget), int(seed))
                if result_path.exists() and masks_path.exists() and not args.force:
                    print(f"[late-mask:collect:skip] {dataset} rho={budget} seed={seed}", flush=True)
                    continue
                print(f"[late-mask:collect] {dataset} rho={budget} seed={seed}", flush=True)
                result = collect_one(
                    dataset=dataset,
                    budget=float(budget),
                    source_seed=int(seed),
                    total_epochs=epoch_map[(dataset, float(budget), int(seed))],
                    protocol=protocol,
                    protocol_hash=digest,
                    device=device,
                )
                write_json(result_path, result)
                print(
                    f"[late-mask:collect:done] {dataset} rho={budget} seed={seed} "
                    f"core={result['core_fraction_of_k']:.4f}",
                    flush=True,
                )


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def train_fixed_classifier(
    *,
    data,
    split: Mapping[str, torch.Tensor],
    structural_mask: torch.Tensor,
    num_features: int,
    num_classes: int,
    budget: float,
    protocol: Mapping[str, object],
    device: torch.device,
) -> Dict[str, object]:
    training = protocol["classifier_retraining"]
    sparse_data = copy.deepcopy(data)
    sparse_data.edge_index = data.edge_index[:, structural_mask]
    local_data = sparse_data.to(device)
    local_split = {name: mask.to(device) for name, mask in split.items()}
    setup_seed(int(training["classifier_seed"]))
    model = build_model(
        mode="random",
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=budget,
        training=training,
        device=device,
    )
    initialization_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    epochs_run = 0
    start = time.perf_counter()
    for epoch in range(int(training["maximum_epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(local_data, is_test=False)
        loss = F.cross_entropy(
            logits[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite fixed-mask classifier loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(local_data, is_test=True)
            val_loss = float(
                F.cross_entropy(
                    val_logits[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                ).item()
            )
        epochs_run = epoch + 1
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Fixed-mask classifier produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(local_data, is_test=True)
        train_accuracy = 100.0 * float(
            (logits[local_split["train"]].argmax(dim=-1) == local_data.y[local_split["train"]])
            .float()
            .mean()
            .item()
        )
        valid_accuracy = 100.0 * float(
            (logits[local_split["valid"]].argmax(dim=-1) == local_data.y[local_split["valid"]])
            .float()
            .mean()
            .item()
        )
        final_valid_loss = float(
            F.cross_entropy(
                logits[local_split["valid"]], local_data.y[local_split["valid"]]
            ).item()
        )
    expected_forward = int(structural_mask.sum().item()) + int(data.n_x)
    if int(model.last_forward_incidence_count) != expected_forward:
        raise RuntimeError("Fixed-mask evaluator forward count is incorrect")
    return {
        "classifier_initialization_sha256": initialization_hash,
        "train_accuracy": train_accuracy,
        "validation_accuracy": valid_accuracy,
        "validation_loss": final_valid_loss,
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "logical_forward_count": expected_forward,
        "end_to_end_seconds": time.perf_counter() - start,
    }


def evaluate(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "late-mask-analysis-")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    budgets = selected(args.budgets, protocol["evaluation"]["budgets"], "budgets")
    seeds = selected(args.seeds, protocol["evaluation"]["source_seeds"], "seeds")
    families = selected(args.families, FAMILIES, "families")
    variants = selected(
        args.variants,
        list(range(int(protocol["evaluation"]["masks_per_family_per_source_run"]))),
        "variants",
    )
    outdir = Path(str(protocol["output_root"]))
    digest = file_hash(protocol_path)
    device = torch.device(args.device)
    for dataset in datasets:
        data, num_features, num_classes, split, diagnostics = redacted_context(dataset, protocol)
        t = int(data.edge_index.size(1))
        for budget in budgets:
            k = max(1, int(float(budget) * t))
            for source_seed in seeds:
                masks_path = bundle_path(outdir, dataset, float(budget), int(source_seed))
                if not masks_path.exists():
                    raise RuntimeError(f"Missing collected masks: {masks_path}")
                bundle = load_bundle(masks_path)
                for family in families:
                    family_masks = bundle[family]
                    for variant in variants:
                        result_path = evaluation_path(
                            outdir,
                            dataset,
                            float(budget),
                            int(source_seed),
                            family,
                            int(variant),
                        )
                        if result_path.exists() and not args.force:
                            print(
                                f"[late-mask:evaluate:skip] {dataset} rho={budget} "
                                f"source={source_seed} {family} m={variant}",
                                flush=True,
                            )
                            continue
                        mask = family_masks[int(variant)]
                        if int(mask.sum().item()) != k:
                            raise RuntimeError("Evaluation mask violates exact budget")
                        print(
                            f"[late-mask:evaluate] {dataset} rho={budget} "
                            f"source={source_seed} {family} m={variant}",
                            flush=True,
                        )
                        metrics = train_fixed_classifier(
                            data=data,
                            split=split,
                            structural_mask=mask,
                            num_features=num_features,
                            num_classes=num_classes,
                            budget=float(budget),
                            protocol=protocol,
                            device=device,
                        )
                        write_json(
                            result_path,
                            {
                                "protocol_version": protocol["protocol_version"],
                                "protocol_hash": digest,
                                "created_at": timestamp(),
                                "stage": "late_mask_fixed_mask_retraining",
                                "dataset_id": dataset,
                                "dataset": protocol["display_names"][dataset],
                                "budget": float(budget),
                                "source_seed": int(source_seed),
                                "family": family,
                                "variant": int(variant),
                                "source_epoch": (
                                    int(bundle["epochs"][int(variant)])
                                    if family == "late"
                                    else None
                                ),
                                "t": t,
                                "k": k,
                                "selected_count": int(mask.sum().item()),
                                "exact_budget": int(mask.sum().item()) == k,
                                "fixed_self_loop_count": int(data.n_x),
                                "mask_sha256": mask_hash(mask),
                                "test_labels_redacted": True,
                                **diagnostics,
                                **metrics,
                            },
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, mean, mean
    half = float(1.96 * np.std(array, ddof=1) / math.sqrt(array.size))
    return mean, mean - half, mean + half


def aggregate(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "late-mask-analysis-")
    outdir = Path(str(protocol["output_root"]))
    expected = [
        evaluation_path(outdir, dataset, float(budget), int(seed), family, variant)
        for dataset in protocol["datasets"]
        for budget in protocol["evaluation"]["budgets"]
        for seed in protocol["evaluation"]["source_seeds"]
        for family in FAMILIES
        for variant in range(int(protocol["evaluation"]["masks_per_family_per_source_run"]))
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Late-mask evaluation is incomplete; missing {len(missing)} records")
    records = [read_json(path) for path in expected]
    write_csv(outdir / "evaluation_records.csv", records)
    margin = float(protocol["interpretation"]["accuracy_equivalence_margin_points"])
    jaccard_threshold = float(protocol["interpretation"]["material_jaccard_maximum"])
    cells = []
    pair_rows = []

    for dataset in protocol["datasets"]:
        for budget in protocol["evaluation"]["budgets"]:
            cell = [
                row
                for row in records
                if row["dataset_id"] == dataset and float(row["budget"]) == float(budget)
            ]
            by_key = {
                (int(row["source_seed"]), row["family"], int(row["variant"])): row
                for row in cell
            }
            late_entries = [row for row in cell if row["family"] == "late"]
            late_masks = {}
            core_fractions = []
            for source_seed in protocol["evaluation"]["source_seeds"]:
                bundle = load_bundle(bundle_path(outdir, dataset, float(budget), int(source_seed)))
                core_fractions.append(float(bundle["core"].sum().item() / by_key[(int(source_seed), "late", 0)]["k"]))
                for variant in range(int(protocol["evaluation"]["masks_per_family_per_source_run"])):
                    late_masks[(int(source_seed), variant)] = bundle["late"][variant]
            for left, right in itertools.combinations(sorted(late_masks), 2):
                left_row = by_key[(left[0], "late", left[1])]
                right_row = by_key[(right[0], "late", right[1])]
                jac = mask_jaccard(late_masks[left], late_masks[right])
                accuracy_difference = abs(
                    float(left_row["validation_accuracy"])
                    - float(right_row["validation_accuracy"])
                )
                pair_rows.append(
                    {
                        "dataset_id": dataset,
                        "dataset": protocol["display_names"][dataset],
                        "budget": budget,
                        "left_source_seed": left[0],
                        "left_variant": left[1],
                        "right_source_seed": right[0],
                        "right_variant": right[1],
                        "same_source_trajectory": left[0] == right[0],
                        "jaccard": jac,
                        "absolute_validation_accuracy_difference": accuracy_difference,
                        "materially_different_but_equivalent": (
                            jac <= jaccard_threshold and accuracy_difference <= margin
                        ),
                    }
                )
            current_pairs = [
                row
                for row in pair_rows
                if row["dataset_id"] == dataset and float(row["budget"]) == float(budget)
            ]
            source_differences = []
            for source_seed in protocol["evaluation"]["source_seeds"]:
                late_acc = [
                    float(by_key[(int(source_seed), "late", variant)]["validation_accuracy"])
                    for variant in range(int(protocol["evaluation"]["masks_per_family_per_source_run"]))
                ]
                boundary_acc = [
                    float(by_key[(int(source_seed), "boundary", variant)]["validation_accuracy"])
                    for variant in range(int(protocol["evaluation"]["masks_per_family_per_source_run"]))
                ]
                source_differences.append(float(np.mean(boundary_acc) - np.mean(late_acc)))

            family_stats = {}
            initialization_hashes = set()
            for family in FAMILIES:
                values = [float(row["validation_accuracy"]) for row in cell if row["family"] == family]
                family_stats[f"{family}_validation_accuracy_mean"], family_stats[f"{family}_validation_accuracy_std"] = mean_std(values)
                initialization_hashes.update(
                    row["classifier_initialization_sha256"]
                    for row in cell
                    if row["family"] == family
                )
            boundary_difference, boundary_low, boundary_high = mean_ci(source_differences)
            jaccards = [float(row["jaccard"]) for row in current_pairs]
            cells.append(
                {
                    "dataset_id": dataset,
                    "dataset": protocol["display_names"][dataset],
                    "budget": budget,
                    "num_late_masks": len(late_entries),
                    "mean_temporal_core_fraction": float(np.mean(core_fractions)),
                    "median_late_pairwise_jaccard": float(np.median(jaccards)),
                    "fraction_materially_different_but_equivalent_pairs": float(
                        np.mean([bool(row["materially_different_but_equivalent"]) for row in current_pairs])
                    ),
                    "boundary_minus_late_mean": boundary_difference,
                    "boundary_minus_late_ci_low": boundary_low,
                    "boundary_minus_late_ci_high": boundary_high,
                    "identical_classifier_initialization": len(initialization_hashes) == 1,
                    **family_stats,
                }
            )
    write_csv(outdir / "late_mask_pairs.csv", pair_rows)
    write_csv(outdir / "cell_summary.csv", cells)
    summary = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "complete": True,
        "evaluation_record_count": len(records),
        "late_mask_pair_count": len(pair_rows),
        "all_exact_budget": all(bool(row["exact_budget"]) for row in records),
        "all_test_labels_redacted": all(bool(row["test_labels_redacted"]) for row in records),
        "all_cells_identical_classifier_initialization": all(
            bool(row["identical_classifier_initialization"]) for row in cells
        ),
        "cells": cells,
    }
    write_json(outdir / "summary.json", summary)
    write_summary_markdown(outdir / "summary.md", protocol, summary)


def write_summary_markdown(
    path: Path,
    protocol: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    lines = [
        "# Late-mask and boundary-randomization analysis",
        "",
        f"Protocol: `{protocol['protocol_version']}`",
        "",
        "All values are validation-only and descriptive. `Late` uses five masks from",
        "the final 20% of each of three scorer trajectories. `Boundary` preserves each",
        "trajectory's five-mask intersection and resamples the remaining slots from its",
        "five-mask union.",
        "",
        "| Dataset | $\\rho$ | Late | Boundary | Boundary-Late | Core/$K$ | Pair Jaccard | Equivalent low-J pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cells"]:
        lines.append(
            f"| {row['dataset']} | {float(row['budget']):.1f} | "
            f"{float(row['late_validation_accuracy_mean']):.2f} | "
            f"{float(row['boundary_validation_accuracy_mean']):.2f} | "
            f"{float(row['boundary_minus_late_mean']):+.2f} | "
            f"{float(row['mean_temporal_core_fraction']):.3f} | "
            f"{float(row['median_late_pairwise_jaccard']):.3f} | "
            f"{100 * float(row['fraction_materially_different_but_equivalent_pairs']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "An equivalent low-Jaccard pair has Jaccard at most "
            f"{protocol['interpretation']['material_jaccard_maximum']} and absolute "
            "validation-accuracy difference at most "
            f"{protocol['interpretation']['accuracy_equivalence_margin_points']} percentage points.",
            "Source-seed intervals use the three trajectory-level means and are descriptive.",
            "",
            f"- Complete evaluation records: {summary['evaluation_record_count']}",
            f"- Exact budgets in every record: {summary['all_exact_budget']}",
            f"- Test labels redacted in every record: {summary['all_test_labels_redacted']}",
            f"- Identical classifier initialization within every cell: {summary['all_cells_identical_classifier_initialization']}",
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
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    collect_parser.add_argument("--datasets", nargs="+")
    collect_parser.add_argument("--budgets", nargs="+", type=float)
    collect_parser.add_argument("--seeds", nargs="+", type=int)
    collect_parser.add_argument("--device", default="cuda:0")
    collect_parser.add_argument("--force", action="store_true")
    collect_parser.set_defaults(func=collect)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    evaluate_parser.add_argument("--datasets", nargs="+")
    evaluate_parser.add_argument("--budgets", nargs="+", type=float)
    evaluate_parser.add_argument("--seeds", nargs="+", type=int)
    evaluate_parser.add_argument("--families", nargs="+")
    evaluate_parser.add_argument("--variants", nargs="+", type=int)
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)
    aggregate_parser = subparsers.add_parser("summarize")
    aggregate_parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    aggregate_parser.set_defaults(func=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
