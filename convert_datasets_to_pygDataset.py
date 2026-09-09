#!/usr/bin/env python3
"""Convert the node-classification datasets used in the paper to PyG data."""

from __future__ import annotations

import os
import os.path as osp
import pickle

import torch
from torch_geometric.data import InMemoryDataset

from load_other_datasets import (
    load_citation_dataset,
    load_cornell_dataset,
    load_yelp_dataset,
)


PAPER_DATASETS = {
    "actor",
    "twitch",
    "pokec",
    "yelp",
    "coauthor_dblp",
    "walmart-trips",
    "cora",
}


def save_data_to_pickle(data, p2root="./data/", file_name=None):
    file_name = file_name or "Hypergraph_star_expansion_dataset"
    os.makedirs(p2root, exist_ok=True)
    path = osp.join(p2root, file_name)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
    return path


class dataset_Hypergraph(InMemoryDataset):
    def __init__(
        self,
        root="./data/pyg_data/hypergraph_dataset_updated/",
        name=None,
        p2raw=None,
        train_percent=0.01,
        feature_noise=None,
        transform=None,
        pre_transform=None,
    ):
        if name not in PAPER_DATASETS:
            raise ValueError(
                "Dataset must be one of: " + ", ".join(sorted(PAPER_DATASETS))
            )
        if p2raw is not None and not osp.isdir(p2raw):
            raise ValueError(f'Raw-data path does not exist: "{p2raw}"')

        self.name = name
        self.feature_noise = feature_noise
        self._train_percent = train_percent
        self.p2raw = p2raw
        self.myraw_dir = osp.join(root, name, "raw")
        super().__init__(osp.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.train_percent = self.data.train_percent

    @property
    def raw_file_names(self):
        if self.feature_noise is None:
            return [self.name]
        return [f"{self.name}_noise_{self.feature_noise}"]

    @property
    def processed_file_names(self):
        if self.feature_noise is None:
            return ["data.pt"]
        return [f"data_noise_{self.feature_noise}.pt"]

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        destination = osp.join(self.myraw_dir, self.raw_file_names[0])
        if osp.isfile(destination):
            return

        if self.name in {"actor", "twitch", "pokec", "cora"}:
            data = load_citation_dataset(
                path=self.p2raw,
                dataset=self.name,
                train_percent=self._train_percent,
            )
        elif self.name == "coauthor_dblp":
            data = load_citation_dataset(
                path=self.p2raw,
                dataset="dblp",
                train_percent=self._train_percent,
            )
        elif self.name == "walmart-trips":
            if self.feature_noise is None:
                raise ValueError("Walmart requires an explicit feature-noise value")
            data = load_cornell_dataset(
                path=self.p2raw,
                dataset=self.name,
                feature_noise=self.feature_noise,
                train_percent=self._train_percent,
            )
        elif self.name == "yelp":
            data = load_yelp_dataset(
                path=self.p2raw,
                train_percent=self._train_percent,
            )
        else:  # Guarded by PAPER_DATASETS.
            raise ValueError(self.name)

        save_data_to_pickle(
            data,
            p2root=self.myraw_dir,
            file_name=self.raw_file_names[0],
        )

    def process(self):
        path = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f"{self.name}()"
