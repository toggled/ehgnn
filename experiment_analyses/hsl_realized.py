#!/usr/bin/env python3
"""Evaluate HSL with validation selection and realized-structure accounting."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from training_utils import clone_state_dict, setup_seed  # noqa: E402
from experiment_controls.core_controls import (  # noqa: E402
    classification_metrics,
    load_dataset,
    split_hash,
    tensor_hash,
)
from node_classification_utils import fixed_split  # noqa: E402


VENDOR_ROOT = Path("third_party/HSL").resolve()
DEFAULT_PROTOCOL = Path("experiment_specs/hsl.json")


def load_hsl_class():
    if not VENDOR_ROOT.exists():
        raise RuntimeError("HSL is not installed; run scripts/fetch_third_party.sh first")
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    from models.models import HSL as VendorHSL

    return VendorHSL


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    temporary.replace(path)


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


def verify_protocol(path: Path) -> Dict[str, object]:
    protocol = read_json(path)
    if protocol.get("protocol_version") != "hsl-v1":
        raise RuntimeError("Unexpected HSL specification version")
    if protocol.get("status") != "fixed":
        raise RuntimeError("HSL experiment specification is not fixed")
    for item in protocol["dependencies"]:
        source = Path(str(item["path"]))
        observed = file_hash(source)
        if observed != item["sha256"]:
            raise RuntimeError(f"Dependency changed: {source}")
    return protocol


def selected(values: Optional[Sequence], allowed: Sequence, label: str) -> List:
    result = list(allowed if values is None else values)
    if not set(result) <= set(allowed):
        raise ValueError(f"Requested {label} outside the experiment specification: {result}")
    return result


def dataset_context(dataset: str, protocol: Mapping[str, object]):
    data, num_features, num_classes = load_dataset(dataset, protocol)
    cfg = protocol["data"]
    split = fixed_split(
        data,
        int(cfg["split_seed"]),
        float(cfg["train_prop"]),
        float(cfg["valid_prop"]),
    )
    original = data.edge_index.clone()
    nodes = torch.arange(int(data.n_x), dtype=torch.long)
    edges = torch.arange(
        int(data.num_hyperedges),
        int(data.num_hyperedges) + int(data.n_x),
        dtype=torch.long,
    )
    data.edge_index = torch.cat([original, torch.stack([nodes, edges])], dim=1)
    diagnostics = {
        "feature_hash": tensor_hash(data.x),
        "label_hash": tensor_hash(data.y),
        "original_edge_index_hash": tensor_hash(original),
        "split_hash": split_hash(split),
        "num_nodes": int(data.n_x),
        "num_hyperedges": int(data.num_hyperedges),
        "num_original_incidences": int(original.size(1)),
        "num_features": int(num_features),
        "num_classes": int(num_classes),
    }
    return data, original, int(num_features), int(num_classes), split, diagnostics


def make_hsl_args(config, protocol, num_features: int, num_classes: int):
    fixed = protocol["model"]
    return SimpleNamespace(
        num_features=num_features,
        num_classes=num_classes,
        All_num_layers=int(fixed["layers"]),
        dropout=float(fixed["dropout"]),
        aggregate="mean",
        normalization=str(fixed["normalization"]),
        GPR=False,
        p1sample=True,
        p1temperature=float(fixed["p1_temperature"]),
        p1useMLP=False,
        p1init_rate=float(fixed["p1_init_rate"]),
        p2sample=True,
        p2temperature=float(fixed["p2_temperature"]),
        p2sample_add_p=float(config["p_add"]),
        p2sample_type="topk_add",
        p2init_rate=float(fixed["p2_init_rate"]),
        p2MLP_hidden=int(config["hidden"]),
        p2MLP_num_layers=int(fixed["p2_mlp_layers"]),
        hc_beta=float(fixed["hard_concrete_beta"]),
        discrete_sample="gumbel",
        MLP_hidden=int(config["hidden"]),
        MLP_num_layers=int(fixed["mlp_layers"]),
        heads=int(fixed["heads"]),
        cos_head=int(fixed["cosine_heads"]),
        contrast=True,
        lambda_contrast=float(fixed["contrastive_weight"]),
        contrast_type="unsup",
        Classifier_hidden=int(fixed["classifier_hidden"]),
        Classifier_num_layers=int(fixed["classifier_layers"]),
        add_self_loop=True,
    )


@contextmanager
def fixed_eval_rng(seed: int, device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


@torch.no_grad()
def evaluate(model, data, split, eval_seed: int, device: torch.device):
    model.eval()
    with fixed_eval_rng(eval_seed, device):
        logits, masks = model(data, return_mask=True)
    logp = F.log_softmax(logits, dim=1)
    metrics = {}
    for name in ("train", "valid", "test"):
        index = split[name]
        metrics[f"{name}_loss"] = float(F.nll_loss(logp[index], data.y[index]).item())
        metrics[f"{name}_acc"] = float(
            (logp[index].argmax(dim=1) == data.y[index]).float().mean().item()
        )
    return metrics, logp, masks


def structure_metrics(model, original, masks, num_nodes: int, num_hyperedges: int):
    edge_mask = masks[0].detach().cpu().view(-1) >= 0.5
    incidence_mask = masks[1].detach().cpu().view(-1) >= 0.5
    candidate = torch.as_tensor(model.final_edge_index, dtype=torch.long)
    if candidate.size(1) != incidence_mask.numel():
        raise RuntimeError("HSL candidate incidence and mask lengths differ")
    active = incidence_mask & edge_mask[candidate[1]]
    original_code = (
        original[0].cpu().numpy().astype(np.int64) * int(num_hyperedges)
        + original[1].cpu().numpy().astype(np.int64)
    )
    candidate_code = (
        candidate[0].numpy().astype(np.int64) * int(num_hyperedges)
        + candidate[1].numpy().astype(np.int64)
    )
    original_unique = np.unique(original_code)
    active_unique = np.unique(candidate_code[active.numpy()])
    retained_original = int(np.isin(original_unique, active_unique).sum())
    new_support = int((~np.isin(active_unique, original_unique)).sum())
    final_unique = retained_original + new_support
    t = int(original.size(1))
    return {
        "candidate_incidence_columns": int(candidate.size(1)),
        "proposed_addition_columns": int(candidate.size(1) - t),
        "active_candidate_columns": int(active.sum().item()),
        "original_support_retained": retained_original,
        "original_support_removed": int(original_unique.size - retained_original),
        "new_support_added": new_support,
        "final_unique_non_self_loop_support": final_unique,
        "realized_original_retention": retained_original / max(1, original_unique.size),
        "realized_final_density_vs_original": final_unique / max(1, original_unique.size),
        "fixed_self_loop_count": int(num_nodes),
        "effective_forward_incidence_count": final_unique + int(num_nodes),
        "effective_forward_density_vs_full": (final_unique + int(num_nodes))
        / max(1, original_unique.size + int(num_nodes)),
        "retained_hyperedges": int(edge_mask.sum().item()),
        "hyperedge_retention": float(edge_mask.float().mean().item()),
    }


def train_run(
    *,
    dataset: str,
    config: Mapping[str, object],
    seed: int,
    stage: str,
    protocol: Mapping[str, object],
    protocol_hash: str,
    device: torch.device,
) -> Dict[str, object]:
    data, original, num_features, num_classes, split, diagnostics = dataset_context(
        dataset, protocol
    )
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: value.to(device) for name, value in split.items()}
    setup_seed(seed)
    model = load_hsl_class()(
        make_hsl_args(config, protocol, num_features, num_classes), local_data
    ).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    maximum_epochs = int(protocol["training"]["maximum_epochs"])
    patience = int(protocol["training"]["patience"])
    eval_seed = int(protocol["random_streams"]["evaluation_seed_base"]) + seed
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(local_data)
        logits, features = output
        classification_loss = F.cross_entropy(
            logits[local_split["train"]], local_data.y[local_split["train"]]
        )
        contrastive_loss = model.supcon(features[local_split["train"]])
        loss = classification_loss + float(model.lambda_contrast) * contrastive_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite HSL loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        validation, _, _ = evaluate(model, local_data, local_split, eval_seed, device)
        val_loss = float(validation["valid_loss"])
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
        raise RuntimeError("No finite HSL validation checkpoint")
    model.load_state_dict(best_state)
    metrics, logp, masks = evaluate(model, local_data, local_split, eval_seed, device)
    structure = structure_metrics(
        model,
        original,
        masks,
        int(data.n_x),
        int(data.num_hyperedges),
    )
    result = {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": stage,
        "status": "complete",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "config_id": config["config_id"],
        "hidden": config["hidden"],
        "learning_rate": config["learning_rate"],
        "p_add": config["p_add"],
        "seed": seed,
        **diagnostics,
        **structure,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "end_to_end_seconds": elapsed,
        "peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "train_acc": 100.0 * metrics["train_acc"],
        "val_acc": 100.0 * metrics["valid_acc"],
        "val_loss": metrics["valid_loss"],
    }
    if stage == "test_evaluation":
        index = local_split["test"]
        predictions = logp[index].argmax(dim=1).detach().cpu().numpy()
        labels = local_data.y[index].detach().cpu().numpy()
        detail = classification_metrics(labels, predictions, num_classes)
        result.update({
            "test_acc": 100.0 * metrics["test_acc"],
            "test_macro_f1": 100.0 * float(detail["macro_f1"]),
            "test_balanced_accuracy": 100.0 * float(detail["balanced_accuracy"]),
            "test_per_class_recall": detail["per_class_recall"],
        })
    return result


def run_path(outdir: Path, stage: str, dataset: str, config_id: str, seed: int) -> Path:
    return outdir / stage / "run_records" / f"{dataset}_{config_id}_s{seed}.json"


def failure_result(protocol, protocol_hash, dataset, config, seed, stage, error):
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_hash": protocol_hash,
        "created_at": timestamp(),
        "stage": stage,
        "status": "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
        "dataset_id": dataset,
        "dataset": protocol["display_names"][dataset],
        "config_id": config["config_id"],
        "hidden": config["hidden"],
        "learning_rate": config["learning_rate"],
        "p_add": config["p_add"],
        "seed": seed,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run_tuning(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    config_ids = selected(
        args.configs,
        [config["config_id"] for config in protocol["validation_grid"]],
        "configs",
    )
    seeds = selected(args.seeds, protocol["tuning"]["seeds"], "seeds")
    configs = {
        config["config_id"]: config for config in protocol["validation_grid"]
    }
    outdir = Path(str(protocol["output_root"]))
    protocol_hash = file_hash(protocol_path)
    device = torch.device(args.device)
    for dataset in datasets:
        for config_id in config_ids:
            for seed in seeds:
                path = run_path(outdir, "validation", dataset, config_id, int(seed))
                if path.exists() and not args.force:
                    print(f"[hsl:tune:skip] {dataset} {config_id} s={seed}", flush=True)
                    continue
                print(f"[hsl:tune] {dataset} {config_id} s={seed}", flush=True)
                try:
                    result = train_run(
                        dataset=dataset,
                        config=configs[config_id],
                        seed=int(seed),
                        stage="validation_tuning",
                        protocol=protocol,
                        protocol_hash=protocol_hash,
                        device=device,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    result = failure_result(
                        protocol, protocol_hash, dataset, configs[config_id], int(seed),
                        "validation_tuning", error,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                write_json(path, result)
                print(f"[hsl:tune:done] status={result['status']}", flush=True)


def freeze_selection(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    outdir = Path(str(protocol["output_root"]))
    manifest_path = outdir / "frozen_selection.json"
    if manifest_path.exists() and not args.force:
        raise RuntimeError("HSL selection is already frozen")
    choices = {}
    rows = []
    for dataset in protocol["datasets"]:
        candidates = []
        for config in protocol["validation_grid"]:
            run_rows = []
            for seed in protocol["tuning"]["seeds"]:
                path = run_path(outdir, "validation", dataset, config["config_id"], int(seed))
                if not path.exists():
                    raise RuntimeError(f"Missing HSL validation run: {path}")
                run_rows.append(read_json(path))
            complete = [row for row in run_rows if row["status"] == "complete"]
            mean_val = float(np.mean([row["val_acc"] for row in complete])) if complete else -math.inf
            row = {
                "dataset_id": dataset,
                "config_id": config["config_id"],
                "complete_seeds": len(complete),
                "mean_val_acc": mean_val,
                "hidden": config["hidden"],
                "learning_rate": config["learning_rate"],
                "p_add": config["p_add"],
            }
            rows.append(row)
            if len(complete) == len(protocol["tuning"]["seeds"]):
                candidates.append(row)
        if not candidates:
            choices[dataset] = {"status": "no_feasible_configuration"}
            continue
        winner = sorted(
            candidates,
            key=lambda row: (
                -row["mean_val_acc"], row["p_add"], row["hidden"],
                row["learning_rate"], row["config_id"],
            ),
        )[0]
        choices[dataset] = {"status": "selected", **winner}
    write_csv(outdir / "validation_grid_summary.csv", rows)
    write_json(
        manifest_path,
        {
            "status": "frozen_before_test_evaluation",
            "created_at": timestamp(),
            "protocol_version": protocol["protocol_version"],
            "protocol_hash": file_hash(protocol_path),
            "selection_rule": protocol["tuning"]["selection_rule"],
            "choices": choices,
        },
    )


def run_evaluation(args) -> None:
    protocol_path = Path(args.protocol)
    protocol = verify_protocol(protocol_path)
    outdir = Path(str(protocol["output_root"]))
    manifest = read_json(outdir / "frozen_selection.json")
    if manifest.get("status") != "frozen_before_test_evaluation":
        raise RuntimeError("HSL selection manifest is not frozen")
    if manifest.get("protocol_hash") != file_hash(protocol_path):
        raise RuntimeError("HSL selection manifest protocol hash mismatch")
    datasets = selected(args.datasets, protocol["datasets"], "datasets")
    seeds = selected(args.seeds, protocol["evaluation"]["seeds"], "seeds")
    configs = {config["config_id"]: config for config in protocol["validation_grid"]}
    device = torch.device(args.device)
    for dataset in datasets:
        choice = manifest["choices"][dataset]
        if choice["status"] != "selected":
            print(f"[hsl:evaluate:infeasible] {dataset}", flush=True)
            continue
        config = configs[choice["config_id"]]
        for seed in seeds:
            path = run_path(outdir, "evaluation", dataset, config["config_id"], int(seed))
            if path.exists() and not args.force:
                print(f"[hsl:evaluate:skip] {dataset} s={seed}", flush=True)
                continue
            print(f"[hsl:evaluate] {dataset} {config['config_id']} s={seed}", flush=True)
            try:
                result = train_run(
                    dataset=dataset,
                    config=config,
                    seed=int(seed),
                    stage="test_evaluation",
                    protocol=protocol,
                    protocol_hash=file_hash(protocol_path),
                    device=device,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                result = failure_result(
                    protocol, file_hash(protocol_path), dataset, config, int(seed),
                    "test_evaluation", error,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            write_json(path, result)
            print(f"[hsl:evaluate:done] status={result['status']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "freeze-selection", "evaluate"])
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "tune":
        run_tuning(args)
    elif args.command == "freeze-selection":
        freeze_selection(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
