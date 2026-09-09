#!/usr/bin/env python3
"""Shared model-training and exact-budget baseline utilities."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from convert_datasets_to_pygDataset import dataset_Hypergraph
from models_sparse import HCHA
from preprocessing import ExtractV2E


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False


def eval_acc(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    prediction = logits.argmax(dim=-1)
    return float((prediction == y_true).float().mean().item())


def data_paths(dname: str) -> Tuple[str, str, Optional[str]]:
    if dname == "walmart-trips":
        return (
            "./data/AllSet_all_raw_data/",
            "./data/pyg_data/hypergraph_dataset_updated/",
            "1",
        )
    if dname == "cora":
        return (
            "./data/AllSet_all_raw_data/cocitation/",
            "./data/pyg_data/hypergraph_dataset_updated/",
            None,
        )
    if dname == "coauthor_dblp":
        return (
            "./data/AllSet_all_raw_data/coauthorship/",
            "./data/pyg_data/hypergraph_dataset_updated/",
            None,
        )
    if dname == "yelp":
        return (
            "./data/AllSet_all_raw_data/yelp/",
            "./data/pyg_data/hypergraph_dataset_updated/",
            None,
        )
    if dname in {"actor", "pokec", "twitch"}:
        return (
            "./data/hetero/",
            "./data/pyg_data/hypergraph_dataset_updated/",
            None,
        )
    raise ValueError(f"Dataset is not part of the paper: {dname}")


def load_v2e_dataset(dname: str) -> Tuple[Data, int, int]:
    raw_path, cache_path, feature_noise = data_paths(dname)
    if feature_noise is None:
        dataset = dataset_Hypergraph(name=dname, root=cache_path, p2raw=raw_path)
    else:
        dataset = dataset_Hypergraph(
            name=dname,
            root=cache_path,
            p2raw=raw_path,
            feature_noise=feature_noise,
        )

    data = copy.deepcopy(dataset.data)
    if dname in {"yelp", "walmart-trips"}:
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
    if dname == "walmart-trips":
        num_classes = int(data.y.max().item() + 1)
        means = torch.zeros((data.n_x, 11), dtype=torch.float32)
        means[torch.arange(data.n_x), data.y] = 1.0
        generator = torch.Generator(device="cpu").manual_seed(20260807)
        data.x = means + torch.randn(means.shape, generator=generator)
    return data, int(dataset.num_features), int(data.y.max().item() + 1)


def make_args(
    *,
    mode: str,
    data: Data,
    num_features: int,
    num_classes: int,
    keep_ratio: float,
    hidden: int,
    dropout: float,
    sampling: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        num_incidences=int(data.edge_index.shape[1]),
        num_hyperedges=int(data.num_hyperedges),
        num_features=int(num_features),
        num_classes=int(num_classes),
        F=int(num_features),
        n_x=data.n_x,
        All_num_layers=1,
        MLP_hidden=int(hidden),
        dropout=float(dropout),
        HCHA_symdegnorm=False,
        keep_ratio=float(keep_ratio),
        sampling=sampling,
        coarse_MLP=32,
    )


def clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def evaluate_model(
    model: HCHA,
    data: Data,
    split_idx: Dict[str, torch.Tensor],
    *,
    return_outputs: bool = False,
) -> Dict[str, float]:
    model.eval()
    log_probabilities = F.log_softmax(model(data, is_test=True), dim=1)
    output = {
        "train_acc": eval_acc(
            data.y[split_idx["train"]], log_probabilities[split_idx["train"]]
        ),
        "val_acc": eval_acc(
            data.y[split_idx["valid"]], log_probabilities[split_idx["valid"]]
        ),
        "test_acc": eval_acc(
            data.y[split_idx["test"]], log_probabilities[split_idx["test"]]
        ),
        "train_loss": float(
            F.nll_loss(
                log_probabilities[split_idx["train"]], data.y[split_idx["train"]]
            ).item()
        ),
        "val_loss": float(
            F.nll_loss(
                log_probabilities[split_idx["valid"]], data.y[split_idx["valid"]]
            ).item()
        ),
        "test_loss": float(
            F.nll_loss(
                log_probabilities[split_idx["test"]], data.y[split_idx["test"]]
            ).item()
        ),
    }
    output["acc_gap"] = 100.0 * (output["train_acc"] - output["test_acc"])
    output["loss_gap"] = output["test_loss"] - output["train_loss"]
    test_probabilities = log_probabilities[split_idx["test"]].exp()
    one_hot_labels = F.one_hot(
        data.y[split_idx["test"]].long(), num_classes=log_probabilities.size(1)
    ).float()
    output["test_brier"] = float(
        ((test_probabilities - one_hot_labels) ** 2).sum(dim=-1).mean().item()
    )
    if return_outputs:
        output["_test_probs"] = test_probabilities.detach().cpu()
        output["_test_labels"] = data.y[split_idx["test"]].detach().cpu()
    return output


def train_model(
    *,
    data: Data,
    split_idx: Dict[str, torch.Tensor],
    mode: str,
    num_features: int,
    num_classes: int,
    keep_ratio: float,
    hidden: int,
    dropout: float,
    lr: float,
    wd: float,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    sampling: str = "multinomial",
    return_outputs: bool = False,
    sparse_self_loop_policy: str = "augment",
) -> Tuple[HCHA, Dict[str, float]]:
    setup_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    local_split = {name: indices.to(device) for name, indices in split_idx.items()}
    args = make_args(
        mode=mode,
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=keep_ratio,
        hidden=hidden,
        dropout=dropout,
        sampling=sampling,
    )
    args.sparse_self_loop_policy = sparse_self_loop_policy
    model = HCHA(args).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    wait = 0
    train_indices = local_split["train"]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        model_output = model(local_data, is_test=False)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        log_probabilities = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            log_probabilities[train_indices], local_data.y[train_indices]
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            validation_log_probabilities = F.log_softmax(
                model(local_data, is_test=True), dim=1
            )
            validation_loss = float(
                F.nll_loss(
                    validation_log_probabilities[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                )
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_model(
        model, local_data, local_split, return_outputs=return_outputs
    )
    metrics["best_epoch"] = float(best_epoch)
    metrics["epochs_run"] = float(epoch + 1)
    return model, metrics


def unit_scores_and_mask_for_baseline(
    data: Data,
    *,
    method: str,
    unit: str,
    keep_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    setup_seed(seed)
    node_ids, edge_ids = data.edge_index
    if unit == "incidence":
        count = int(node_ids.numel())
        budget = max(1, int(keep_ratio * count))
        if method == "Random-Fixed":
            scores = torch.rand(count)
        elif method == "Degree":
            edge_degree = torch.bincount(
                edge_ids, minlength=int(data.num_hyperedges)
            ).float()
            scores = edge_degree[edge_ids]
        else:
            raise ValueError(method)
        keep = torch.zeros(count, dtype=torch.bool)
        keep[torch.topk(scores, k=budget).indices] = True
        return data.edge_index[:, keep], keep, scores

    if unit != "edge":
        raise ValueError(unit)
    edge_count = int(data.num_hyperedges)
    budget = max(1, int(keep_ratio * edge_count))
    if method == "Random-Fixed":
        scores = torch.rand(edge_count)
    elif method == "Degree":
        scores = torch.bincount(edge_ids, minlength=edge_count).float()
    else:
        raise ValueError(method)
    keep_edges = torch.zeros(edge_count, dtype=torch.bool)
    keep_edges[torch.topk(scores, k=budget).indices] = True
    keep_incidences = torch.isin(
        edge_ids, keep_edges.nonzero(as_tuple=False).view(-1)
    )
    return data.edge_index[:, keep_incidences], keep_edges, scores
