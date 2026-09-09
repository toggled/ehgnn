#!/usr/bin/env python3
"""Shared data, split, and EHGNN-F training utilities."""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models_sparse import HCHA
from preprocessing import rand_train_test_idx
from training_utils import (
    clone_state_dict,
    evaluate_model,
    load_v2e_dataset,
    make_args,
    setup_seed,
)


PRETTY_DATASET = {
    "actor": "Actor",
    "twitch": "Twitch",
    "pokec": "Pokec",
    "yelp": "Yelp",
    "coauthor_dblp": "DBLP-CA",
    "walmart-trips": "Walmart",
    "cora": "Cora",
}

def prepare_dataset(dname: str) -> Tuple[Data, int, int]:
    """Load one of the node-classification datasets used in the paper."""
    if dname not in PRETTY_DATASET:
        raise ValueError(f"Dataset is not part of the paper: {dname}")
    return load_v2e_dataset(dname)


def fixed_split(data, split_seed: int, train_prop: float, valid_prop: float):
    setup_seed(split_seed)
    return rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop)


@torch.no_grad()
def current_mask(model: HCHA, data, keep_ratio: float, device: torch.device):
    local_data = data if data.edge_index.device == device else copy.deepcopy(data).to(device)
    _, _, selected, probabilities = model.mask_module(
        local_data,
        keep_ratio=keep_ratio,
        is_test=True,
        return_mask=True,
    )
    return selected.detach().cpu().bool(), probabilities.detach().cpu().float()


def compact_probability_metrics(probabilities: torch.Tensor) -> Dict[str, float]:
    probabilities = probabilities.detach().float().clamp(1e-7, 1.0 - 1e-7)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    ) / math.log(2.0)
    return {
        "prob_mean": float(probabilities.mean()),
        "prob_std": float(probabilities.std(unbiased=False)),
        "normalized_entropy": float(entropy.mean()),
        "polarization": float((2.0 * (probabilities - 0.5).abs()).mean()),
        "diffuse_040_060": float(
            ((probabilities >= 0.4) & (probabilities <= 0.6)).float().mean()
        ),
        "confident_010": float(
            ((probabilities <= 0.1) | (probabilities >= 0.9)).float().mean()
        ),
    }


def jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    union = (left | right).sum().item()
    return float((left & right).sum().item() / union) if union else 1.0


@torch.no_grad()
def evaluate_train_validation_model(
    model: HCHA,
    data: Data,
    split_idx: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Evaluate tuning runs without accessing the test split."""
    model.eval()
    log_probabilities = F.log_softmax(model(data, is_test=True), dim=1)
    metrics: Dict[str, float] = {}
    for split_name in ("train", "valid"):
        indices = split_idx[split_name]
        predictions = log_probabilities[indices].argmax(dim=-1)
        prefix = "train" if split_name == "train" else "val"
        metrics[f"{prefix}_acc"] = float(
            (predictions == data.y[indices]).float().mean()
        )
        metrics[f"{prefix}_loss"] = float(
            F.nll_loss(log_probabilities[indices], data.y[indices]).item()
        )
    return metrics


def train_learned_with_trajectory(
    *,
    data,
    split_idx,
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
    sampling: str,
    trajectory_every: int,
    mask_init_std: float,
    mask_lr_multiplier: float,
    scorer_hidden_dim: int = 32,
    low_rank: int = 4,
    mask_gradient_estimator: str = "selected_only",
    include_test_metrics: bool = True,
    sparse_self_loop_policy: str = "fixed_self_loops_implicit",
):
    if mask_gradient_estimator != "selected_only":
        raise ValueError("The released LoG experiments use selected-only gradients")
    setup_seed(seed)
    local_data = copy.deepcopy(data).to(device)
    split_names = tuple(split_idx) if include_test_metrics else ("train", "valid")
    local_split = {key: split_idx[key].to(device) for key in split_names}
    model_args = make_args(
        mode=mode,
        data=local_data,
        num_features=num_features,
        num_classes=num_classes,
        keep_ratio=keep_ratio,
        hidden=hidden,
        dropout=dropout,
        sampling=sampling,
    )
    model_args.low_rank = int(low_rank)
    model_args.coarse_MLP = int(scorer_hidden_dim)
    model_args.mask_init_std = mask_init_std
    model_args.mask_gradient_estimator = mask_gradient_estimator
    model_args.sparse_self_loop_policy = sparse_self_loop_policy
    model = HCHA(model_args).to(device)
    model.reset_parameters()
    initial_selected, initial_probabilities = current_mask(
        model, local_data, keep_ratio, device
    )

    mask_parameters = list(model.mask_module.parameters())
    mask_parameter_ids = {id(parameter) for parameter in mask_parameters}
    classifier_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in mask_parameter_ids
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": classifier_parameters, "lr": lr},
            {"params": mask_parameters, "lr": lr * mask_lr_multiplier},
        ],
        weight_decay=wd,
    )

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    wait = 0
    trajectory: List[Dict[str, float]] = []
    train_indices = local_split["train"]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, probabilities = model(local_data, is_test=False)
        log_probabilities = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            log_probabilities[train_indices], local_data.y[train_indices]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_log_probabilities = F.log_softmax(
                model(local_data, is_test=True), dim=1
            )
            validation_loss = float(
                F.nll_loss(
                    validation_log_probabilities[local_split["valid"]],
                    local_data.y[local_split["valid"]],
                )
            )
            if epoch == 0 or (epoch + 1) % trajectory_every == 0:
                _, epoch_probabilities = current_mask(
                    model, local_data, keep_ratio, device
                )
                trajectory.append(
                    {"epoch": epoch, **compact_probability_metrics(epoch_probabilities)}
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
    final_selected, final_probabilities = current_mask(
        model, local_data, keep_ratio, device
    )
    metrics = (
        evaluate_model(model, local_data, local_split)
        if include_test_metrics
        else evaluate_train_validation_model(model, local_data, local_split)
    )
    metrics["best_epoch"] = float(best_epoch)
    metrics["epochs_run"] = float(epoch + 1)
    trajectory.append(
        {
            "epoch": best_epoch,
            "is_best": 1.0,
            **compact_probability_metrics(final_probabilities),
        }
    )
    return (
        model,
        metrics,
        initial_selected,
        initial_probabilities,
        final_selected,
        final_probabilities,
        trajectory,
    )
