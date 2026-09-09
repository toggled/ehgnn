#!/usr/bin/env python3
"""LP-C1: leakage-free five-seed Hyper-SAGNN context sparsification study."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
HYPERSAGNN_ROOT = ROOT / "Hyper-SAGNN-master"
if str(HYPERSAGNN_ROOT) not in sys.path:
    sys.path.insert(0, str(HYPERSAGNN_ROOT))

from hypersagnn_ehgnn import HyperSAGNNEdgeClassifier  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiment_specs/hyperedge_prediction.json"
DEFAULT_OUTDIR = ROOT / ".work/hyperedge_prediction"


@dataclass
class LinkData:
    core: np.ndarray
    valid_pos: np.ndarray
    test_pos: np.ndarray
    nums_type: np.ndarray
    type_starts: np.ndarray
    type_ends: np.ndarray
    num_nodes: int
    edge_size: int
    positive_hash: set
    context_edge_ids_by_node: List[np.ndarray]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def protocol_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def to_global_padded(edges: np.ndarray, nums_type: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64).copy()
    offsets = np.concatenate(([0], np.cumsum(nums_type)[:-1])).astype(np.int64)
    if edges.shape[1] != len(nums_type):
        raise ValueError(f"Expected one typed position per node type, got {edges.shape}")
    edges += offsets[None, :]
    edges += 1  # zero is reserved for Hyper-SAGNN padding
    return edges


def stable_unique_rows(edges: np.ndarray) -> np.ndarray:
    _, first = np.unique(edges, axis=0, return_index=True)
    return edges[np.sort(first)]


def load_dataset(dataset: str, protocol: Mapping[str, object]) -> LinkData:
    data_root = HYPERSAGNN_ROOT / "data" / dataset
    train_npz = np.load(data_root / "train_data.npz", allow_pickle=True)
    test_npz = np.load(data_root / "test_data.npz", allow_pickle=True)
    train = np.asarray(train_npz["train_data"], dtype=np.int64)
    test = np.asarray(test_npz["test_data"], dtype=np.int64)
    nums_type = np.asarray(train_npz["nums_type"], dtype=np.int64)
    train = stable_unique_rows(to_global_padded(train, nums_type))
    test = stable_unique_rows(to_global_padded(test, nums_type))
    train_hash = {tuple(row.tolist()) for row in train}
    test = np.asarray(
        [row for row in test if tuple(row.tolist()) not in train_hash],
        dtype=np.int64,
    )
    if len(test) == 0:
        raise RuntimeError(f"{dataset}: no test positives remain after overlap removal")

    data_cfg = protocol["data"]
    rng = np.random.default_rng(int(data_cfg["split_seed"]))
    order = rng.permutation(len(train))
    development_fraction = float(
        data_cfg.get(
            "development_fraction_of_sanitized_train",
            data_cfg.get("development_fraction_of_original_train"),
        )
    )
    n_valid = max(1, int(round(development_fraction * len(train))))
    valid_pos = train[order[:n_valid]]
    core = train[order[n_valid:]]

    type_starts = np.concatenate(([1], 1 + np.cumsum(nums_type)[:-1])).astype(np.int64)
    type_ends = (1 + np.cumsum(nums_type)).astype(np.int64)
    all_pos = np.concatenate([core, valid_pos, test], axis=0)
    positive_hash = {tuple(row.tolist()) for row in all_pos}
    if len(positive_hash) != len(all_pos):
        raise RuntimeError(f"{dataset}: positive partitions are not disjoint after canonicalization")
    edge_ids_by_node: List[List[int]] = [[] for _ in range(int(nums_type.sum()) + 1)]
    for edge_id, edge in enumerate(core):
        for node in edge:
            edge_ids_by_node[int(node)].append(edge_id)
    context_edge_ids_by_node = [np.asarray(ids, dtype=np.int64) for ids in edge_ids_by_node]
    return LinkData(
        core=core,
        valid_pos=valid_pos,
        test_pos=test,
        nums_type=nums_type,
        type_starts=type_starts,
        type_ends=type_ends,
        num_nodes=int(nums_type.sum()),
        edge_size=int(train.shape[1]),
        positive_hash=positive_hash,
        context_edge_ids_by_node=context_edge_ids_by_node,
    )


def corrupt_one(
    edge: np.ndarray,
    *,
    data: LinkData,
    rng: np.random.Generator,
) -> np.ndarray:
    for _ in range(1024):
        out = edge.copy()
        position = int(rng.integers(0, data.edge_size))
        out[position] = int(rng.integers(data.type_starts[position], data.type_ends[position]))
        candidate = tuple(out.tolist())
        if candidate not in data.positive_hash and candidate != tuple(edge.tolist()):
            return out
    raise RuntimeError("Could not construct a negative hyperedge after 1024 attempts")


def candidate_set(
    positives: np.ndarray,
    *,
    data: LinkData,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    negatives = np.stack([corrupt_one(edge, data=data, rng=rng) for edge in positives])
    x = np.concatenate([positives, negatives], axis=0)
    y = np.concatenate(
        [np.ones(len(positives), dtype=np.float32), np.zeros(len(negatives), dtype=np.float32)]
    )
    order = rng.permutation(len(x))
    return x[order], y[order]


class ContextHyperSAGNN(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        num_nodes: int,
        num_context_edges: int,
        edge_size: int,
        dimensions: int,
        heads: int,
        d_k: int,
        d_v: int,
        keep_ratio: float,
        mask_init_std: float,
        random_seed: int,
    ) -> None:
        super().__init__()
        self.method = method
        self.num_context_edges = int(num_context_edges)
        self.edge_size = int(edge_size)
        self.num_incidences = self.num_context_edges * self.edge_size
        self.keep_ratio = float(keep_ratio)
        self.k = max(1, int(math.floor(self.keep_ratio * self.num_incidences)))
        embedding = nn.Embedding(num_nodes + 1, dimensions, padding_idx=0)
        self.backbone = HyperSAGNNEdgeClassifier(
            node_embedding=embedding,
            d_model=dimensions,
            num_classes=1,
            n_head=heads,
            d_k=d_k,
            d_v=d_v,
            diag_mask="True",
            bottle_neck=dimensions,
        )
        self.context_projection = nn.Linear(dimensions, dimensions, bias=False)
        self.context_norm = nn.LayerNorm(dimensions)
        if method == "EHGNN-F":
            self.mask_logits = nn.Parameter(torch.empty(self.num_incidences))
            nn.init.normal_(self.mask_logits, mean=0.0, std=mask_init_std)
        else:
            self.register_parameter("mask_logits", None)
        if method == "Random-Fixed":
            generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
            selected = torch.randperm(self.num_incidences, generator=generator)[: self.k]
            random_mask = torch.zeros(self.num_incidences, dtype=torch.bool)
            random_mask[selected] = True
            self.register_buffer("random_mask", random_mask)
        else:
            self.register_buffer("random_mask", torch.empty(0, dtype=torch.bool))

        self.last_selector_grad_count = 0
        self.last_selected_count = self.num_incidences if method in {"Hyper-SAGNN", "FullContext"} else self.k

    def _global_incidence_weight(self, is_test: bool) -> Optional[torch.Tensor]:
        if self.method in {"Hyper-SAGNN", "FullContext"}:
            return None
        if self.method == "Random-Fixed":
            return self.random_mask.to(dtype=self.backbone.node_embedding.weight.dtype)
        probabilities = torch.sigmoid(self.mask_logits)
        if is_test:
            selected = torch.topk(probabilities, k=self.k, largest=True, sorted=False).indices
        else:
            selected = torch.multinomial(probabilities, self.k, replacement=False)
        hard = torch.zeros_like(probabilities)
        hard[selected] = 1.0
        return hard + (probabilities - probabilities.detach()) * hard

    def build_node_context(
        self,
        context_edges: torch.Tensor,
        context_edge_ids: torch.Tensor,
        *,
        is_test: bool,
    ) -> torch.Tensor:
        device = context_edges.device
        dimensions = self.backbone.node_embedding.embedding_dim
        node_count = self.backbone.node_embedding.num_embeddings
        if self.method == "Hyper-SAGNN" or context_edges.numel() == 0:
            return torch.zeros((node_count, dimensions), device=device)

        flat_nodes = context_edges.reshape(-1)
        local_edges = torch.arange(context_edges.size(0), device=device).repeat_interleave(self.edge_size)
        positions = torch.arange(self.edge_size, device=device).repeat(context_edges.size(0))
        global_incidence_ids = context_edge_ids.repeat_interleave(self.edge_size) * self.edge_size + positions
        global_weight = self._global_incidence_weight(is_test)
        if global_weight is None:
            weights = torch.ones(flat_nodes.numel(), device=device)
        else:
            weights = global_weight[global_incidence_ids]
            keep = weights.detach() > 0
            flat_nodes = flat_nodes[keep]
            local_edges = local_edges[keep]
            weights = weights[keep]
        if flat_nodes.numel() == 0:
            return torch.zeros((node_count, dimensions), device=device)

        token_features = self.backbone.node_embedding(flat_nodes)
        edge_sum = torch.zeros((context_edges.size(0), dimensions), device=device)
        edge_degree = torch.zeros((context_edges.size(0), 1), device=device)
        edge_sum.index_add_(0, local_edges, token_features * weights[:, None])
        edge_degree.index_add_(0, local_edges, weights[:, None])
        edge_features = edge_sum / edge_degree.clamp_min(1e-12)

        messages = edge_features[local_edges] * weights[:, None]
        node_sum = torch.zeros((node_count, dimensions), device=device)
        node_degree = torch.zeros((node_count, 1), device=device)
        node_sum.index_add_(0, flat_nodes, messages)
        node_degree.index_add_(0, flat_nodes, weights[:, None])
        node_context = node_sum / node_degree.clamp_min(1e-12)
        return self.context_norm(self.context_projection(node_context))

    def score(self, candidates: torch.Tensor, node_context: Optional[torch.Tensor]) -> torch.Tensor:
        token_embeddings, _ = self.backbone.get_token_embeddings(candidates)
        if node_context is not None:
            token_embeddings = token_embeddings + node_context[candidates]
        probabilities, _ = self.backbone.encode_from_tokens(token_embeddings, candidates)
        return probabilities.view(-1).clamp(1e-6, 1.0 - 1e-6)


def metrics(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    prediction = (probability >= 0.5).astype(np.float32)
    return {
        "accuracy": float((prediction == y).mean()),
        "auc": float(roc_auc_score(y, probability)),
        "aupr": float(average_precision_score(y, probability)),
    }


@torch.no_grad()
def evaluate(
    model: ContextHyperSAGNN,
    candidates: np.ndarray,
    labels: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> Dict[str, float]:
    model.eval()
    context_tensor = torch.as_tensor(context, dtype=torch.long, device=device)
    context_ids = torch.arange(len(context), dtype=torch.long, device=device)
    node_context = None
    if model.method != "Hyper-SAGNN":
        node_context = model.build_node_context(context_tensor, context_ids, is_test=True)
    outputs: List[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        batch = torch.as_tensor(candidates[start : start + batch_size], dtype=torch.long, device=device)
        outputs.append(model.score(batch, node_context).cpu().numpy())
    return metrics(labels, np.concatenate(outputs))


def sample_context_ids(
    data: LinkData,
    count: int,
    excluded: np.ndarray,
    candidate_nodes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_context = len(data.core)
    available = n_context - len(np.unique(excluded))
    if count >= available:
        return np.setdiff1d(np.arange(n_context), excluded, assume_unique=False)
    excluded_set = set(int(value) for value in excluded)
    adjacent_chunks = [
        data.context_edge_ids_by_node[int(node)]
        for node in np.unique(candidate_nodes)
        if data.context_edge_ids_by_node[int(node)].size > 0
    ]
    if adjacent_chunks:
        adjacent = np.unique(np.concatenate(adjacent_chunks))
        adjacent = adjacent[~np.isin(adjacent, excluded)]
    else:
        adjacent = np.empty(0, dtype=np.int64)
    if len(adjacent) > count:
        adjacent = rng.choice(adjacent, size=count, replace=False)
    selected = set(int(value) for value in adjacent)
    selected.update(excluded_set)
    while len(selected) - len(excluded_set) < count:
        draw = rng.integers(0, n_context, size=max(32, 2 * (count - len(selected) + len(excluded_set))))
        selected.update(int(value) for value in draw if int(value) not in excluded_set)
    result = np.fromiter((value for value in selected if value not in excluded_set), dtype=np.int64)
    if len(result) > count:
        adjacent_set = set(int(value) for value in adjacent)
        fill = np.asarray([value for value in result if int(value) not in adjacent_set], dtype=np.int64)
        need = count - len(adjacent)
        fill = rng.choice(fill, size=need, replace=False) if len(fill) > need else fill
        result = np.concatenate([adjacent, fill])
    return result


def train_run(
    *,
    dataset: str,
    method: str,
    seed: int,
    protocol: Mapping[str, object],
    protocol_path: Path,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    data = load_dataset(dataset, protocol)
    cfg = protocol["model"]
    sparse_cfg = protocol["sparsification"]
    train_cfg = protocol["training"]
    model = ContextHyperSAGNN(
        method=method,
        num_nodes=data.num_nodes,
        num_context_edges=len(data.core),
        edge_size=data.edge_size,
        dimensions=int(cfg["dimensions"]),
        heads=int(cfg["heads"]),
        d_k=int(cfg["d_k"]),
        d_v=int(cfg["d_v"]),
        keep_ratio=float(sparse_cfg["keep_ratio"]),
        mask_init_std=float(sparse_cfg["mask_init_std"]),
        random_seed=100000 + seed,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    valid_x, valid_y = candidate_set(data.valid_pos, data=data, seed=300000 + seed)
    test_x, test_y = candidate_set(data.test_pos, data=data, seed=400000 + seed)
    batch_size = int(cfg["training_candidate_batch_size"])
    context_multiplier = float(cfg["training_context_sample_edges_per_positive"])
    max_epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])

    best_auc = -float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0
    start_time = time.perf_counter()
    observed_selector_grad = False

    for epoch in range(max_epochs):
        model.train()
        rng = np.random.default_rng(500000 + 1000 * seed + epoch)
        positive_order = rng.permutation(len(data.core))
        for start in range(0, len(positive_order), batch_size):
            positive_ids = positive_order[start : start + batch_size]
            positives = data.core[positive_ids]
            negatives = np.stack([corrupt_one(edge, data=data, rng=rng) for edge in positives])
            candidates = np.concatenate([positives, negatives], axis=0)
            labels = np.concatenate(
                [np.ones(len(positives), dtype=np.float32), np.zeros(len(negatives), dtype=np.float32)]
            )
            order = rng.permutation(len(candidates))
            candidates = candidates[order]
            labels = labels[order]

            node_context = None
            if method != "Hyper-SAGNN":
                context_count = max(1, int(round(context_multiplier * len(positives))))
                context_ids = sample_context_ids(
                    data, context_count, positive_ids, candidates.reshape(-1), rng
                )
                context_tensor = torch.as_tensor(data.core[context_ids], dtype=torch.long, device=device)
                context_id_tensor = torch.as_tensor(context_ids, dtype=torch.long, device=device)
                node_context = model.build_node_context(
                    context_tensor, context_id_tensor, is_test=False
                )

            candidate_tensor = torch.as_tensor(candidates, dtype=torch.long, device=device)
            label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            probability = model.score(candidate_tensor, node_context)
            loss = F.binary_cross_entropy(probability, label_tensor)
            loss.backward()
            if model.mask_logits is not None and model.mask_logits.grad is not None:
                grad = model.mask_logits.grad
                if torch.isfinite(grad).all() and int(torch.count_nonzero(grad).item()) > 0:
                    observed_selector_grad = True
            optimizer.step()

        valid_metrics = evaluate(model, valid_x, valid_y, data.core, device)
        if valid_metrics["auc"] > best_auc + 1e-12:
            best_auc = valid_metrics["auc"]
            best_epoch = epoch
            best_state = clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("No finite development checkpoint was produced")
    model.load_state_dict(best_state)
    valid_metrics = evaluate(model, valid_x, valid_y, data.core, device)
    test_metrics = evaluate(model, test_x, test_y, data.core, device)
    elapsed = time.perf_counter() - start_time

    probabilities = None
    if model.mask_logits is not None:
        probabilities = torch.sigmoid(model.mask_logits.detach()).cpu()
    return {
        "protocol_hash": protocol_hash(protocol_path),
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "num_core_context_edges": len(data.core),
        "num_development_positive_edges": len(data.valid_pos),
        "num_test_positive_edges": len(data.test_pos),
        "num_context_incidences": model.num_incidences,
        "selected_context_incidences": model.last_selected_count,
        "structural_density": model.last_selected_count / model.num_incidences,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "development_accuracy": valid_metrics["accuracy"],
        "development_auc": valid_metrics["auc"],
        "development_aupr": valid_metrics["aupr"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_aupr": test_metrics["aupr"],
        "selector_nonzero_finite_gradient_observed": observed_selector_grad if method == "EHGNN-F" else None,
        "mask_probability_mean": float(probabilities.mean()) if probabilities is not None else None,
        "mask_probability_std": float(probabilities.std(unbiased=False)) if probabilities is not None else None,
        "wall_time_seconds": elapsed,
        "test_access": "once_after_best_development_checkpoint_restoration"
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(protocol: Mapping[str, object], protocol_path: Path, outdir: Path) -> None:
    rows = []
    missing = []
    for dataset in protocol["datasets"]:
        for method in protocol["methods"]:
            for seed in protocol["model_seeds"]:
                path = outdir / "runs" / f"{dataset}_{method.replace('-', '').lower()}_s{seed}.json"
                if not path.exists():
                    missing.append(str(path))
                else:
                    with path.open() as handle:
                        row = json.load(handle)
                    if row["protocol_hash"] != protocol_hash(protocol_path):
                        raise RuntimeError(f"Protocol hash mismatch: {path}")
                    rows.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} runs; first missing: {missing[0]}")

    summary = []
    for dataset in protocol["datasets"]:
        for method in protocol["methods"]:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            item: Dict[str, object] = {"dataset": dataset, "method": method, "n": len(group)}
            for metric in ("test_accuracy", "test_auc", "test_aupr", "structural_density", "wall_time_seconds"):
                values = np.asarray([float(row[metric]) for row in group], dtype=float)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1))
            summary.append(item)
    write_csv(outdir / "runs.csv", rows)
    write_csv(outdir / "summary.csv", summary)
    atomic_json(
        outdir / "audit.json",
        {
            "protocol_hash": protocol_hash(protocol_path),
            "complete": True,
            "num_runs": len(rows),
            "expected_runs": len(protocol["datasets"]) * len(protocol["methods"]) * len(protocol["model_seeds"]),
            "no_statistical_tests": True,
            "all_sparse_budgets_exact": all(
                row["selected_context_incidences"] == max(1, int(math.floor(0.5 * row["num_context_incidences"])))
                for row in rows if row["method"] in {"Random-Fixed", "EHGNN-F"}
            ),
            "all_ehgnnf_runs_observed_gradient": all(
                row["selector_nonzero_finite_gradient_observed"] for row in rows if row["method"] == "EHGNN-F"
            ),
        },
    )
    for item in summary:
        print(
            f"{item['dataset']:10s} {item['method']:12s} "
            f"AUC {100*item['test_auc_mean']:.2f}+/-{100*item['test_auc_std']:.2f} "
            f"AUPR {100*item['test_aupr_mean']:.2f}+/-{100*item['test_aupr_std']:.2f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
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
    datasets: Sequence[str] = args.datasets or protocol["datasets"]
    methods: Sequence[str] = args.methods or protocol["methods"]
    seeds: Sequence[int] = args.seeds or protocol["model_seeds"]
    if not set(datasets) <= set(protocol["datasets"]):
        raise ValueError("Requested dataset outside frozen protocol")
    if not set(methods) <= set(protocol["methods"]):
        raise ValueError("Requested method outside frozen protocol")
    if not set(seeds) <= set(protocol["model_seeds"]):
        raise ValueError("Requested seed outside frozen protocol")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                tag = method.replace("-", "").lower()
                path = args.outdir / "runs" / f"{dataset}_{tag}_s{seed}.json"
                if path.exists() and not args.force:
                    print(f"[skip] {dataset} {method} seed={seed}", flush=True)
                    continue
                print(f"[run] {dataset} {method} seed={seed}", flush=True)
                result = train_run(
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    protocol=protocol,
                    protocol_path=args.protocol,
                    device=device,
                )
                atomic_json(path, result)
                print(
                    f"[done] {dataset} {method} seed={seed} "
                    f"AUC={100*result['test_auc']:.2f} AUPR={100*result['test_aupr']:.2f} "
                    f"epoch={result['best_epoch']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
