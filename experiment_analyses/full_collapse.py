#!/usr/bin/env python3
"""Check exact equivalence between EHGNN-F and Full at full retention."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from training_utils import setup_seed
from experiment_analyses.theory_invariants import (
    build_model,
    classifier_gradients,
    compare_tensors,
    dataset_context,
    file_hash,
    read_json,
    selected,
    timestamp,
    verify_protocol,
    write_json,
)


DEFAULT_PROTOCOL = Path("experiment_specs/full_retention.json")


def classifier_state(model) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("convs.")
    }


def load_classifier_state(model, state: Mapping[str, torch.Tensor]) -> None:
    target = model.state_dict()
    for name, value in state.items():
        target[name] = value.to(target[name].device)
    model.load_state_dict(target)


def cpu_gradients(model) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in classifier_gradients(model).items()
    }


def evaluate_cell(dataset: str, protocol, device: torch.device) -> Dict[str, object]:
    data, num_features, num_classes, split, diagnostics = dataset_context(dataset, protocol)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: value.to(device) for name, value in split.items()}
    gate = protocol["deterministic_gate"]
    training = dict(protocol["training"])
    training["dropout"] = 0.0
    training["hidden"] = int(gate["hidden"])
    seed = int(gate["seed"])
    atol = float(gate["atol"])
    rtol = float(gate["rtol"])

    setup_seed(seed)
    full = build_model(
        mode="full",
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=1.0,
        training=training,
        device=device,
    )
    frozen_classifier = classifier_state(full)
    full.eval()
    with torch.no_grad():
        full_logits = full(local_data, is_test=True).detach().cpu()
    full.eval()
    full.zero_grad(set_to_none=True)
    full_loss = F.cross_entropy(
        full(local_data, is_test=True)[local_split["train"]],
        local_data.y[local_split["train"]],
    )
    full_loss.backward()
    full_grad = cpu_gradients(full)
    full_loss_value = float(full_loss.item())
    full_forward_count = int(full.last_forward_incidence_count)
    full_self_loop_count = int(full.last_fixed_self_loop_count)
    del full_loss, full
    torch.cuda.empty_cache()

    setup_seed(seed)
    learned = build_model(
        mode="learnmask",
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=1.0,
        training=training,
        device=device,
    )
    load_classifier_state(learned, frozen_classifier)
    learned.eval()
    with torch.no_grad():
        learned_logits = learned(local_data, is_test=True).detach().cpu()
        _, _, hard_mask, _ = learned.mask_module(
            local_data,
            keep_ratio=1.0,
            is_test=True,
            return_mask=True,
        )
    forward = compare_tensors(full_logits, learned_logits, atol, rtol)
    learned.zero_grad(set_to_none=True)
    learned.eval()
    learned_logits_train = learned(local_data, is_test=True)
    learned_loss = F.cross_entropy(
        learned_logits_train[local_split["train"]],
        local_data.y[local_split["train"]],
    )
    learned_loss.backward()
    learned_grad = cpu_gradients(learned)
    if set(full_grad) != set(learned_grad):
        raise RuntimeError("Classifier-gradient parameter sets differ")
    gradient_details = {
        name: compare_tensors(full_grad[name], learned_grad[name], atol, rtol)
        for name in full_grad
    }
    gradients_allclose = all(value["allclose"] for value in gradient_details.values())
    gradient_max_abs = max(value["max_abs"] for value in gradient_details.values())
    t = int(local_data.edge_index.size(1))
    n = int(local_data.n_x)
    selected_count = int(hard_mask.sum().item())
    counts_pass = (
        selected_count == t
        and full_forward_count == t + n
        and int(learned.last_forward_incidence_count) == t + n
        and full_self_loop_count == n
        and int(learned.last_fixed_self_loop_count) == n
    )
    passed = counts_pass and bool(forward["allclose"]) and gradients_allclose
    return {
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        **diagnostics,
        "gate_hidden": int(gate["hidden"]),
        "selected_original_incidences": selected_count,
        "full_forward_incidences": full_forward_count,
        "learned_forward_incidences": int(learned.last_forward_incidence_count),
        "full_fixed_self_loops": full_self_loop_count,
        "learned_fixed_self_loops": int(learned.last_fixed_self_loop_count),
        "counts_pass": counts_pass,
        "forward_max_abs": forward["max_abs"],
        "forward_mean_abs": forward["mean_abs"],
        "forward_allclose": forward["allclose"],
        "gradient_max_abs": gradient_max_abs,
        "gradients_allclose": gradients_allclose,
        "full_loss": full_loss_value,
        "learned_loss": float(learned_loss.item()),
        "passed": passed,
        "gradient_details": gradient_details,
    }


def cell_path(outdir: Path, dataset: str) -> Path:
    return outdir / "gate_cells" / f"{dataset}.json"


def run_gate(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "full-retention-")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    outdir = Path(str(protocol["output_root"]))
    device = torch.device(args.device)
    for dataset in datasets:
        path = cell_path(outdir, dataset)
        if path.exists() and not args.force:
            print(f"[full-retention:skip] {dataset}", flush=True)
            continue
        print(f"[full-retention] {dataset}", flush=True)
        row = evaluate_cell(dataset, protocol, device)
        write_json(path, row)
        print(f"[full-retention:done] {dataset} passed={row['passed']}", flush=True)


def aggregate_gate(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path, "full-retention-")
    outdir = Path(str(protocol["output_root"]))
    paths = [cell_path(outdir, dataset) for dataset in protocol["datasets"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Full-retention check is incomplete; missing {len(missing)} cells")
    rows = [read_json(path) for path in paths]
    result = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": file_hash(protocol_path),
        "created_at": timestamp(),
        "stage": "full_retention_equivalence_check",
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }
    write_json(outdir / "deterministic_equivalence_gate.json", result)
    if not result["passed"]:
        raise RuntimeError("Full-retention equivalence check failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gate", "aggregate"])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "gate":
        run_gate(args)
    else:
        aggregate_gate(args)


if __name__ == "__main__":
    main()
