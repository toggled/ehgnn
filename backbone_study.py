#!/usr/bin/env python3
"""Evaluate EHGNN-F with AllSetTransformer and ED-HNN backbones."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgnn import EquivSetGNN
from node_classification_utils import fixed_split
from training_utils import load_v2e_dataset
from models_sparse import SetGNN, add_fixed_self_loops_incidence
from walmart_feature_check import load_walmart


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "experiment_specs/backbone_study.json"
DEFAULT_OUTDIR = ROOT / ".work/backbone_study"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def load_dataset(dataset: str, protocol: Mapping[str, object]):
    if dataset == "walmart-trips":
        data, num_features, num_classes = load_walmart("1", 20260807, 11)
    else:
        data, num_features, num_classes = load_v2e_dataset(dataset)
    cfg = protocol["data"]
    split = fixed_split(
        data,
        int(cfg["split_seed"]),
        float(cfg["train_prop"]),
        float(cfg["valid_prop"]),
    )
    data.x = data.x.float()
    data.y = data.y.long()
    return data, int(num_features), int(num_classes), split


def backbone_args(
    *,
    data,
    num_features: int,
    num_classes: int,
    protocol: Mapping[str, object],
) -> SimpleNamespace:
    cfg = protocol["backbone_configuration"]
    return SimpleNamespace(
        mode="full",
        num_incidences=int(data.edge_index.size(1)),
        num_hyperedges=int(data.num_hyperedges),
        num_features=int(num_features),
        num_classes=int(num_classes),
        F=int(num_features),
        n_x=int(data.n_x),
        All_num_layers=int(cfg["layers"]),
        MLP_hidden=int(cfg["hidden"]),
        MLP_num_layers=2,
        MLP2_num_layers=-1,
        MLP3_num_layers=-1,
        Classifier_hidden=int(cfg["hidden"]),
        Classifier_num_layers=2,
        dropout=float(cfg["dropout"]),
        aggregate="mean",
        normalization="ln",
        deepset_input_norm=True,
        GPR=False,
        LearnMask=False,
        PMA=True,
        heads=1,
        activation="relu",
        edconv_type="EquivSet",
        restart_alpha=0.5,
        keep_ratio=float(protocol["sparsification"]["keep_ratio"]),
        reg="none",
        sampling="multinomial",
        coarse_MLP=32,
        verbose=False,
    )


class ExactIncidenceSelector(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        num_incidences: int,
        keep_ratio: float,
        seed: int,
        mask_init_std: float,
    ) -> None:
        super().__init__()
        self.method = method
        self.num_incidences = int(num_incidences)
        self.k = max(1, int(math.floor(keep_ratio * self.num_incidences)))
        self.sampling_seed = 700000 + int(seed)
        self._sampling_generator: Optional[torch.Generator] = None
        if method == "EHGNN-F":
            generator = torch.Generator(device="cpu").manual_seed(600000 + int(seed))
            logits = torch.normal(
                mean=0.0,
                std=float(mask_init_std),
                size=(self.num_incidences,),
                generator=generator,
            )
            self.logits = nn.Parameter(logits)
        else:
            self.register_parameter("logits", None)
        if method == "Random-Fixed":
            generator = torch.Generator(device="cpu").manual_seed(800000 + int(seed))
            selected = torch.randperm(self.num_incidences, generator=generator)[: self.k]
            mask = torch.zeros(self.num_incidences, dtype=torch.bool)
            mask[selected] = True
            self.register_buffer("random_mask", mask)
        else:
            self.register_buffer("random_mask", torch.empty(0, dtype=torch.bool))

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._sampling_generator is None:
            self._sampling_generator = torch.Generator(device=device)
            self._sampling_generator.manual_seed(self.sampling_seed)
        return self._sampling_generator

    def forward(
        self,
        edge_index: torch.Tensor,
        *,
        is_test: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index.size(1) != self.num_incidences:
            raise ValueError("Selector must receive the complete original incidence array")
        if self.method == "Full":
            mask = torch.ones(self.num_incidences, dtype=torch.bool, device=edge_index.device)
            weights = torch.ones(self.num_incidences, dtype=torch.float32, device=edge_index.device)
            return edge_index, weights, mask
        if self.method == "Random-Fixed":
            mask = self.random_mask
            weights = torch.ones(self.k, dtype=torch.float32, device=edge_index.device)
            return edge_index[:, mask], weights, mask
        probabilities = torch.sigmoid(self.logits)
        if is_test:
            selected = torch.topk(probabilities, k=self.k, largest=True, sorted=False).indices
        else:
            selected = torch.multinomial(
                probabilities,
                self.k,
                replacement=False,
                generator=self._generator(probabilities.device),
            )
        mask = torch.zeros(self.num_incidences, dtype=torch.bool, device=edge_index.device)
        mask[selected] = True
        hard = torch.ones(self.k, dtype=probabilities.dtype, device=probabilities.device)
        weights = hard + probabilities[selected] - probabilities[selected].detach()
        return edge_index[:, selected], weights, mask


class CorrectedAllSetTransformer(nn.Module):
    def __init__(self, args: SimpleNamespace, selector: ExactIncidenceSelector) -> None:
        super().__init__()
        self.backbone = SetGNN(args)
        self.selector = selector

    def forward(self, data, *, is_test: bool) -> torch.Tensor:
        edge_index, weights, _ = self.selector(data.edge_index, is_test=is_test)
        edge_index, weights = add_fixed_self_loops_incidence(
            edge_index,
            int(data.n_x),
            int(data.num_hyperedges),
            weights,
        )
        x = F.dropout(data.x, p=0.2, training=self.training)
        reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
        for v2e, e2v in zip(self.backbone.V2EConvs, self.backbone.E2VConvs):
            x = F.relu(v2e(x, edge_index, weights, self.backbone.aggr))
            x = F.dropout(x, p=self.backbone.dropout, training=self.training)
            x = F.relu(e2v(x, reverse, weights, self.backbone.aggr))
            x = F.dropout(x, p=self.backbone.dropout, training=self.training)
        return self.backbone.classifier(x)


class CorrectedEDHNN(nn.Module):
    def __init__(self, args: SimpleNamespace, selector: ExactIncidenceSelector) -> None:
        super().__init__()
        self.backbone = EquivSetGNN(args.num_features, args.num_classes, args)
        self.selector = selector

    def forward(self, data, *, is_test: bool) -> torch.Tensor:
        edge_index, weights, _ = self.selector(data.edge_index, is_test=is_test)
        edge_index, weights = add_fixed_self_loops_incidence(
            edge_index,
            int(data.n_x),
            int(data.num_hyperedges),
            weights,
        )
        vertex, edges = edge_index
        x = self.backbone.dropout(data.x)
        x = F.relu(self.backbone.lin_in(x))
        x0 = x
        for _ in range(self.backbone.nlayer):
            x = self.backbone.dropout(x)
            x = self.backbone.conv(x, vertex, edges, x0, incidence_weight=weights)
            x = self.backbone.act(x)
        x = self.backbone.dropout(x)
        return self.backbone.classifier(x)


def make_model(
    *,
    backbone: str,
    method: str,
    seed: int,
    data,
    num_features: int,
    num_classes: int,
    protocol: Mapping[str, object],
) -> nn.Module:
    args = backbone_args(
        data=data,
        num_features=num_features,
        num_classes=num_classes,
        protocol=protocol,
    )
    selector = ExactIncidenceSelector(
        method=method,
        num_incidences=int(data.edge_index.size(1)),
        keep_ratio=float(protocol["sparsification"]["keep_ratio"]),
        seed=seed,
        mask_init_std=float(protocol["sparsification"]["mask_init_std"]),
    )
    if backbone == "AllSetTransformer":
        return CorrectedAllSetTransformer(args, selector)
    if backbone == "ED-HNN":
        return CorrectedEDHNN(args, selector)
    raise ValueError(backbone)


@torch.no_grad()
def evaluate(model: nn.Module, data, split: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    model.eval()
    logits = model(data, is_test=True)
    logp = F.log_softmax(logits, dim=-1)
    result = {}
    for name in ("train", "valid", "test"):
        idx = split[name]
        result[f"{name}_loss"] = float(F.nll_loss(logp[idx], data.y[idx]).item())
        result[f"{name}_accuracy"] = float((logp[idx].argmax(dim=-1) == data.y[idx]).float().mean().item())
    return result


def train_run(
    *,
    backbone: str,
    dataset: str,
    method: str,
    seed: int,
    protocol: Mapping[str, object],
    protocol_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    data, num_features, num_classes, split = load_dataset(dataset, protocol)
    set_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    local_split = {key: value.to(device) for key, value in split.items()}
    model = make_model(
        backbone=backbone,
        method=method,
        seed=seed,
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        protocol=protocol,
    ).to(device)
    cfg = protocol["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0
    observed_selector_grad = False
    start_time = time.perf_counter()
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(local_data, is_test=False)
        logp = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(logp[local_split["train"]], local_data.y[local_split["train"]])
        loss.backward()
        selector_grad = model.selector.logits.grad if model.selector.logits is not None else None
        if selector_grad is not None:
            if not torch.isfinite(selector_grad).all():
                raise RuntimeError("Non-finite selector gradient")
            observed_selector_grad |= int(torch.count_nonzero(selector_grad).item()) > 0
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_logits = model(local_data, is_test=True)
            val_logp = F.log_softmax(val_logits, dim=-1)
            val_loss = float(
                F.nll_loss(
                    val_logp[local_split["valid"]], local_data.y[local_split["valid"]]
                ).item()
            )
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_state = clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    final = evaluate(model, local_data, local_split)
    elapsed = time.perf_counter() - start_time
    m = int(local_data.edge_index.size(1))
    selected = m if method == "Full" else model.selector.k
    probabilities = (
        torch.sigmoid(model.selector.logits.detach()).cpu()
        if model.selector.logits is not None
        else None
    )
    return {
        "protocol_hash": protocol_hash(protocol_path),
        "backbone": backbone,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "num_nodes": int(local_data.n_x),
        "num_hyperedges": int(local_data.num_hyperedges),
        "num_original_incidences": m,
        "num_selected_original_incidences": selected,
        "num_fixed_self_loops": int(local_data.n_x),
        "structural_density": selected / m,
        "effective_forward_density": (selected + int(local_data.n_x)) / (m + int(local_data.n_x)),
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        **final,
        "selector_nonzero_finite_gradient_observed": observed_selector_grad if method == "EHGNN-F" else None,
        "mask_probability_mean": float(probabilities.mean()) if probabilities is not None else None,
        "mask_probability_std": float(probabilities.std(unbiased=False)) if probabilities is not None else None,
        "feature_hash": tensor_hash(data.x),
        "edge_index_hash": tensor_hash(data.edge_index),
        "wall_time_seconds": elapsed,
        "test_access": "once_after_best_validation_checkpoint_restoration",
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_path(outdir: Path, backbone: str, dataset: str, method: str, seed: int) -> Path:
    return outdir / "runs" / f"{backbone.lower().replace('-', '')}_{dataset}_{method.lower().replace('-', '')}_s{seed}.json"


def summarize(protocol: Mapping[str, object], protocol_path: Path, outdir: Path) -> None:
    rows = []
    missing = []
    expected_hash = protocol_hash(protocol_path)
    for backbone in protocol["backbones"]:
        for dataset in protocol["datasets"]:
            for method in protocol["methods"]:
                for seed in protocol["model_seeds"]:
                    path = run_path(outdir, backbone, dataset, method, seed)
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    with path.open() as handle:
                        row = json.load(handle)
                    if row["protocol_hash"] != expected_hash:
                        raise RuntimeError(f"Protocol hash mismatch: {path}")
                    rows.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} runs; first missing: {missing[0]}")

    summary = []
    for backbone in protocol["backbones"]:
        for dataset in protocol["datasets"]:
            for method in protocol["methods"]:
                group = [
                    row for row in rows
                    if row["backbone"] == backbone and row["dataset"] == dataset and row["method"] == method
                ]
                item: Dict[str, object] = {
                    "backbone": backbone,
                    "dataset": dataset,
                    "method": method,
                    "n": len(group),
                }
                for metric in ("test_accuracy", "structural_density", "effective_forward_density", "wall_time_seconds"):
                    values = np.asarray([float(row[metric]) for row in group], dtype=float)
                    item[f"{metric}_mean"] = float(values.mean())
                    item[f"{metric}_std"] = float(values.std(ddof=1))
                summary.append(item)
    write_csv(outdir / "runs.csv", rows)
    write_csv(outdir / "summary.csv", summary)
    sparse_rows = [row for row in rows if row["method"] != "Full"]
    atomic_json(
        outdir / "audit.json",
        {
            "protocol_hash": expected_hash,
            "complete": True,
            "num_runs": len(rows),
            "expected_runs": len(protocol["backbones"]) * len(protocol["datasets"]) * len(protocol["methods"]) * len(protocol["model_seeds"]),
            "no_statistical_tests": True,
            "all_sparse_budgets_exact": all(
                row["num_selected_original_incidences"]
                == max(1, int(math.floor(0.5 * row["num_original_incidences"])))
                for row in sparse_rows
            ),
            "all_methods_use_same_self_loop_count": all(
                row["num_fixed_self_loops"] == row["num_nodes"] for row in rows
            ),
            "all_ehgnnf_runs_observed_gradient": all(
                row["selector_nonzero_finite_gradient_observed"]
                for row in rows if row["method"] == "EHGNN-F"
            ),
        },
    )
    for item in summary:
        print(
            f"{item['backbone']:17s} {item['dataset']:15s} {item['method']:7s} "
            f"{100*item['test_accuracy_mean']:.2f}+/-{100*item['test_accuracy_std']:.2f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.protocol.open() as handle:
        protocol = json.load(handle)
    if args.summarize:
        summarize(protocol, args.protocol, args.outdir)
        return
    backbones: Sequence[str] = args.backbones or protocol["backbones"]
    datasets: Sequence[str] = args.datasets or protocol["datasets"]
    methods: Sequence[str] = args.methods or protocol["methods"]
    seeds: Sequence[int] = args.seeds or protocol["model_seeds"]
    if not set(backbones) <= set(protocol["backbones"]):
        raise ValueError("Requested backbone outside frozen protocol")
    if not set(datasets) <= set(protocol["datasets"]):
        raise ValueError("Requested dataset outside frozen protocol")
    if not set(methods) <= set(protocol["methods"]):
        raise ValueError("Requested method outside frozen protocol")
    if not set(seeds) <= set(protocol["model_seeds"]):
        raise ValueError("Requested seed outside frozen protocol")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for backbone in backbones:
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    path = run_path(args.outdir, backbone, dataset, method, seed)
                    if path.exists() and not args.force:
                        print(f"[skip] {backbone} {dataset} {method} seed={seed}", flush=True)
                        continue
                    print(f"[run] {backbone} {dataset} {method} seed={seed}", flush=True)
                    result = train_run(
                        backbone=backbone,
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        protocol=protocol,
                        protocol_path=args.protocol,
                        device=device,
                    )
                    atomic_json(path, result)
                    print(
                        f"[done] {backbone} {dataset} {method} seed={seed} "
                        f"test={100*result['test_accuracy']:.2f} epoch={result['best_epoch']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
