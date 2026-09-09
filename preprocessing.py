"""Incidence conversion and split helpers used by the LoG experiments."""

import numpy as np
import torch


def ExtractV2E(data):
    """Keep the vertex-to-hyperedge half of a bidirectional incidence index."""
    edge_index = data.edge_index
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    if not (
        data.n_x + data.num_hyperedges - 1 == data.edge_index[0].max().item()
    ):
        raise ValueError("num_hyperedges does not match the incidence index")
    boundary = torch.where(edge_index[0] == num_nodes)[0].min()
    data.edge_index = edge_index[:, :boundary].type(torch.LongTensor)
    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def rand_train_test_idx(
    label,
    train_prop=0.5,
    valid_prop=0.25,
    ignore_negative=True,
    balance=False,
):
    """Create the fixed random train/validation/test split used in the paper."""
    if balance:
        raise ValueError("Class-balanced splitting is not part of the LoG protocol")

    labeled_nodes = torch.where(label != -1)[0] if ignore_negative else label
    num_labeled = labeled_nodes.shape[0]
    train_num = int(num_labeled * train_prop)
    valid_num = int(num_labeled * valid_prop)
    permutation = torch.as_tensor(np.random.permutation(num_labeled))

    train_indices = permutation[:train_num]
    valid_indices = permutation[train_num : train_num + valid_num]
    test_indices = permutation[train_num + valid_num :]
    if not ignore_negative:
        return train_indices, valid_indices, test_indices

    return {
        "train": index_to_mask(train_indices, label.size(0)),
        "valid": index_to_mask(valid_indices, label.size(0)),
        "test": index_to_mask(test_indices, label.size(0)),
    }
