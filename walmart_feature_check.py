#!/usr/bin/env python3
"""Validation-only audit for the non-degenerate Walmart feature construction."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from convert_datasets_to_pygDataset import dataset_Hypergraph
from node_classification_utils import fixed_split
from training_utils import clone_state_dict, make_args, setup_seed
from models_sparse import HCHA, MLP_model
from preprocessing import ExtractV2E
from main_accuracy import atomic_write_json, protocol_hash, timestamp


DEFAULT_PROTOCOL = Path("experiment_specs/large_dataset_accuracy.json")
DEFAULT_OUTDIR = Path(".work/walmart_feature_check")
METHODS = ("mlp", "full_hgnn")


def read_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_walmart(feature_noise: str, feature_seed: int, feature_dimension: int):
    dataset = dataset_Hypergraph(
        name="walmart-trips",
        root="./data/pyg_data/hypergraph_dataset_updated/",
        p2raw="./data/AllSet_all_raw_data/",
        feature_noise=str(feature_noise),
    )
    data = copy.deepcopy(dataset.data)
    data.y = data.y - data.y.min()
    if not hasattr(data, "n_x"):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, "num_hyperedges"):
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1]
        )
    data = ExtractV2E(data)
    data.edge_index[1] -= data.edge_index[1].min()
    data.num_hyperedges = int(data.edge_index[1].max().item() + 1)
    data.n_x = int(data.x.shape[0])
    num_classes = int(data.y.max().item() + 1)
    if feature_dimension < num_classes:
        raise ValueError("Feature dimension cannot be smaller than the class count")
    means = torch.zeros((data.n_x, feature_dimension), dtype=torch.float32)
    means[torch.arange(data.n_x), data.y] = 1.0
    generator = torch.Generator(device="cpu").manual_seed(int(feature_seed))
    epsilon = torch.randn(means.shape, generator=generator, dtype=torch.float32)
    data.x = means + float(feature_noise) * epsilon
    return data, int(feature_dimension), num_classes


def feature_diagnostics(data) -> Dict[str, object]:
    prediction = data.x.argmax(dim=1)
    return {
        "feature_hash": tensor_hash(data.x),
        "label_hash": tensor_hash(data.y),
        "feature_argmax_label_agreement": float(
            (prediction == data.y).float().mean().item()
        ),
        "num_nodes": int(data.n_x),
        "num_hyperedges": int(data.num_hyperedges),
        "num_original_incidences": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
        "num_classes": int(data.y.max().item() + 1),
        "feature_min": float(data.x.min().item()),
        "feature_max": float(data.x.max().item()),
        "feature_mean": float(data.x.mean().item()),
        "feature_std": float(data.x.std(unbiased=True).item()),
        "all_features_finite": bool(torch.isfinite(data.x).all()),
        "unique_feature_rows": int(data.x.unique(dim=0).size(0)),
    }


def make_mlp(num_features: int, num_classes: int, cfg: Mapping[str, object]):
    args = SimpleNamespace(
        num_features=int(num_features),
        num_classes=int(num_classes),
        MLP_hidden=int(cfg["hidden"]),
        All_num_layers=2,
        dropout=float(cfg["dropout"]),
        normalization="None",
    )
    return MLP_model(args)


def make_full_hgnn(
    data,
    num_features: int,
    num_classes: int,
    cfg: Mapping[str, object],
):
    args = make_args(
        mode="full",
        data=data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=1.0,
        hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        sampling="multinomial",
    )
    args.All_num_layers = 1
    args.HCHA_symdegnorm = False
    args.sparse_self_loop_policy = str(cfg["sparse_self_loop_policy"])
    return HCHA(args)


def forward(model, data, method: str, *, training: bool):
    if method == "full_hgnn":
        return model(data, is_test=not training)
    return model(data)


@torch.no_grad()
def evaluate_train_validation(model, data, split_idx, method: str):
    model.eval()
    logits = forward(model, data, method, training=False)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite final logits")
    logp = F.log_softmax(logits, dim=1)
    result = {}
    for split_name, output_name in (("train", "train"), ("valid", "val")):
        indices = split_idx[split_name]
        result[f"{output_name}_loss"] = float(
            F.nll_loss(logp[indices], data.y[indices]).item()
        )
        result[f"{output_name}_acc"] = 100.0 * float(
            (logp[indices].argmax(dim=1) == data.y[indices]).float().mean().item()
        )
    return result


def train_one(protocol, data, split_idx, method: str, seed: int, device):
    audit = protocol["walmart_validation_only_audit"]
    cfg = audit["mlp"] if method == "mlp" else audit["full_hgnn"]
    setup_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    local_split = {
        "train": split_idx["train"].to(device),
        "valid": split_idx["valid"].to(device),
    }
    num_features = int(local_data.x.size(1))
    num_classes = int(local_data.y.max().item() + 1)
    if method == "mlp":
        model = make_mlp(num_features, num_classes, cfg).to(device)
    else:
        model = make_full_hgnn(local_data, num_features, num_classes, cfg).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, local_data, method, training=True)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            logp[local_split["train"]], local_data.y[local_split["train"]]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            val_logits = forward(model, local_data, method, training=False)
            val_logp = F.log_softmax(val_logits, dim=1)
            val_loss = float(
                F.nll_loss(
                    val_logp[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                ).item()
            )
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    metrics = evaluate_train_validation(model, local_data, local_split, method)
    metrics["best_epoch"] = int(best_epoch)
    metrics["epochs_run"] = int(epoch + 1)
    return model, metrics


def run(args, protocol, device):
    audit = protocol["walmart_validation_only_audit"]
    feature_noise = str(audit["feature_noise"])
    feature_seed = int(audit["feature_seed"])
    feature_dimension = int(audit["feature_dimension"])
    data_a, num_features, num_classes = load_walmart(
        feature_noise, feature_seed, feature_dimension
    )
    data_b, _, _ = load_walmart(feature_noise, feature_seed, feature_dimension)
    diagnostics = feature_diagnostics(data_a)
    diagnostics["reproducible_feature_reconstruction"] = bool(
        torch.equal(data_a.x, data_b.x)
        and torch.equal(data_a.y, data_b.y)
        and torch.equal(data_a.edge_index, data_b.edge_index)
    )
    if not diagnostics["reproducible_feature_reconstruction"]:
        raise RuntimeError("Repeated deterministic Walmart reconstructions differ")
    split_idx = fixed_split(
        data_a,
        int(audit["split_seed"]),
        float(audit["train_prop"]),
        float(audit["valid_prop"]),
    )
    seeds = list(args.seeds or audit["model_seeds"])
    methods = list(args.methods or METHODS)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    outdir = Path(args.outdir)
    run_dir = outdir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for seed in seeds:
            result_path = run_dir / f"walmart_sigma1_seed{feature_seed}_{method}_s{seed}.json"
            if result_path.exists() and not args.force:
                print(f"[skip] method={method} seed={seed}", flush=True)
                continue
            print(f"[run] method={method} seed={seed}", flush=True)
            model, metrics = train_one(
                protocol, data_a, split_idx, method, int(seed), device
            )
            result = {
                "stage": "walmart_validation_only_feature_audit",
                "created_at": timestamp(),
                "protocol_hash": protocol_hash(Path(args.protocol)),
                "dataset_id": "walmart-trips",
                "feature_noise": feature_noise,
                "method": method,
                "seed": int(seed),
                "test_metrics_computed": False,
                **diagnostics,
                **metrics,
            }
            if method == "full_hgnn":
                expected = int(data_a.edge_index.size(1)) + int(data_a.n_x)
                result.update(
                    {
                        "num_fixed_self_loops": int(
                            getattr(model, "last_fixed_self_loop_count", -1)
                        ),
                        "forward_incidences": int(
                            getattr(model, "last_forward_incidence_count", -1)
                        ),
                        "expected_forward_incidences": expected,
                    }
                )
                if (
                    result["num_fixed_self_loops"] != int(data_a.n_x)
                    or result["forward_incidences"] != expected
                ):
                    raise RuntimeError("Full-HGNN fixed self-loop check failed")
            if any("test" in key and key != "test_metrics_computed" for key in result):
                raise RuntimeError("Validation-only record contains a test field")
            atomic_write_json(result_path, result)
            print(
                f"[done] method={method} seed={seed} "
                f"val={metrics['val_acc']:.2f} epoch={metrics['best_epoch']}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def summarize(args, protocol):
    outdir = Path(args.outdir)
    rows = []
    for path in sorted((outdir / "runs").glob("walmart_sigma1_seed*_*.json")):
        record = read_json(path)
        if record.get("test_metrics_computed") is not False:
            raise RuntimeError(f"Test-access audit failed: {path}")
        if any("test_acc" in key or "test_loss" in key for key in record):
            raise RuntimeError(f"Test metric found in {path}")
        rows.append(record)
    expected_seeds = set(protocol["walmart_validation_only_audit"]["model_seeds"])
    expected = {(method, seed) for method in METHODS for seed in expected_seeds}
    observed = {(str(row["method"]), int(row["seed"])) for row in rows}
    if observed != expected:
        missing = sorted(expected - observed)
        raise RuntimeError(f"Incomplete audit matrix; missing={missing}")
    method_summary: List[Dict[str, object]] = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        values = np.asarray([float(row["val_acc"]) for row in group])
        method_summary.append(
            {
                "method": method,
                "n": len(group),
                "val_acc_mean": float(values.mean()),
                "val_acc_std": float(values.std(ddof=1)),
                "val_acc_min": float(values.min()),
                "val_acc_max": float(values.max()),
            }
        )
    mlp = next(row for row in method_summary if row["method"] == "mlp")
    first = rows[0]
    conditions = protocol["walmart_validation_only_audit"]["pass_conditions"]
    checks = {
        "feature_argmax_agreement": float(first["feature_argmax_label_agreement"])
        < float(conditions["feature_argmax_label_agreement_below"]),
        "mean_mlp_validation_accuracy": float(mlp["val_acc_mean"])
        < float(conditions["mean_mlp_validation_accuracy_below_percent"]),
        "maximum_mlp_validation_accuracy": float(mlp["val_acc_max"])
        < float(conditions["maximum_seed_mlp_validation_accuracy_below_percent"]),
        "finite_metrics": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("train_acc", "val_acc", "train_loss", "val_loss")
        ),
        "reproducible_feature_reconstruction": all(
            bool(row["reproducible_feature_reconstruction"]) for row in rows
        ),
        "no_test_metrics": all(row["test_metrics_computed"] is False for row in rows),
        "fixed_self_loop_counts": all(
            row["method"] != "full_hgnn"
            or (
                int(row["num_fixed_self_loops"]) == int(row["num_nodes"])
                and int(row["forward_incidences"])
                == int(row["expected_forward_incidences"])
            )
            for row in rows
        ),
    }
    summary = {
        "stage": "walmart_validation_only_feature_audit_summary",
        "created_at": timestamp(),
        "protocol_hash": protocol_hash(Path(args.protocol)),
        "dataset_id": "walmart-trips",
        "feature_noise": str(protocol["walmart_validation_only_audit"]["feature_noise"]),
        "feature_seed": int(protocol["walmart_validation_only_audit"]["feature_seed"]),
        "feature_hash": first["feature_hash"],
        "feature_argmax_label_agreement": first["feature_argmax_label_agreement"],
        "num_nodes": first["num_nodes"],
        "num_original_incidences": first["num_original_incidences"],
        "checks": checks,
        "audit_pass": all(checks.values()),
        "methods": method_summary,
    }
    atomic_write_json(outdir / "summary.json", summary)
    write_csv(outdir / "method_summary.csv", method_summary)
    lines = [
        "# Walmart feature-noise audit",
        "",
        f"Protocol SHA-256: `{summary['protocol_hash']}`.",
        "",
        "This audit used `feature_noise=1` and did not compute or print test metrics.",
        "",
        f"Feature argmax/label agreement: {100.0 * float(summary['feature_argmax_label_agreement']):.2f}%.",
        "",
        "| Method | Validation accuracy | Range | Seeds |",
        "|---|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {float(row['val_acc_mean']):.2f} +/- "
            f"{float(row['val_acc_std']):.2f} | {float(row['val_acc_min']):.2f}--"
            f"{float(row['val_acc_max']):.2f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            f"Audit decision: **{'PASS' if summary['audit_pass'] else 'FAIL'}**.",
            "",
            "The decision concerns benchmark construction only and is independent of future sparsifier test accuracy.",
        ]
    )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "summarize"))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = read_json(Path(args.protocol))
    if args.stage == "run":
        run(args, protocol, torch.device(args.device))
    else:
        summarize(args, protocol)


if __name__ == "__main__":
    main()
