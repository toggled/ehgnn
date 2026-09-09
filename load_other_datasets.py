#!/usr/bin/env python3
"""Load the raw node-classification datasets used in the paper."""

from __future__ import annotations

import os.path as osp
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import Data
from torch_sparse import coalesce


def _coalesced_data(features, labels, edge_index, num_hyperedges, train_percent):
    data = Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        y=torch.as_tensor(labels, dtype=torch.long),
    )
    total_ids = int(data.edge_index.max().item() + 1)
    data.edge_index, data.edge_attr = coalesce(
        data.edge_index, None, total_ids, total_ids
    )
    data.n_x = int(data.x.shape[0])
    data.train_percent = train_percent
    data.num_hyperedges = int(num_hyperedges)
    return data


def load_citation_dataset(path, dataset, train_percent=0.025):
    with open(osp.join(path, dataset, "features.pickle"), "rb") as handle:
        features = pickle.load(handle)
    if dataset not in {"actor", "pokec", "twitch"} and hasattr(features, "todense"):
        features = features.todense()
    with open(osp.join(path, dataset, "labels.pickle"), "rb") as handle:
        labels = pickle.load(handle)
    with open(osp.join(path, dataset, "hypergraph.pickle"), "rb") as handle:
        hypergraph = pickle.load(handle)
    if isinstance(hypergraph, list):
        hypergraph = dict(enumerate(hypergraph))

    num_nodes = int(features.shape[0])
    node_ids = []
    hyperedge_ids = []
    for offset, nodes in enumerate(hypergraph.values()):
        node_ids.extend(nodes)
        hyperedge_ids.extend([num_nodes + offset] * len(nodes))
    edge_index = np.asarray(
        [node_ids + hyperedge_ids, hyperedge_ids + node_ids], dtype=np.int64
    )
    return _coalesced_data(
        features,
        np.asarray(labels, dtype=np.int64),
        edge_index,
        len(hypergraph),
        train_percent,
    )


def load_yelp_dataset(path, train_percent=0.025, name_dictionary_size=1000):
    latlong = pd.read_csv(osp.join(path, "yelp_restaurant_latlong.csv")).values
    locations = pd.read_csv(osp.join(path, "yelp_restaurant_locations.csv"))
    state_ids = locations.state_int.to_numpy()
    city_ids = locations.city_int.to_numpy()
    num_nodes = int(locations.shape[0])

    states = np.zeros((num_nodes, int(state_ids.max())))
    states[np.arange(num_nodes), state_ids - 1] = 1
    cities = np.zeros((num_nodes, int(city_ids.max())))
    cities[np.arange(num_nodes), city_ids - 1] = 1
    names = pd.read_csv(osp.join(path, "yelp_restaurant_name.csv")).values.ravel()
    name_features = CountVectorizer(
        max_features=name_dictionary_size,
        stop_words="english",
        strip_accents="ascii",
    ).fit_transform(names).toarray()
    features = np.hstack([latlong, states, cities, name_features])
    labels = pd.read_csv(
        osp.join(path, "yelp_restaurant_business_stars.csv")
    ).values.ravel()

    incidence = pd.read_csv(osp.join(path, "yelp_restaurant_incidence_H.csv"))
    node_ids = incidence.node.to_numpy() - 1
    hyperedge_ids = incidence.he.to_numpy() - 1 + num_nodes
    forward = np.vstack([node_ids, hyperedge_ids])
    edge_index = np.hstack([forward, forward[::-1]])
    return _coalesced_data(
        features,
        labels,
        edge_index,
        int(incidence.he.max()),
        train_percent,
    )


def load_cornell_dataset(
    path,
    dataset="walmart-trips",
    feature_noise=1.0,
    train_percent=0.025,
):
    labels = pd.read_csv(
        osp.join(path, dataset, f"node-labels-{dataset}.txt"),
        names=["node_label"],
    ).values.ravel()
    num_nodes = int(labels.shape[0])
    num_classes = int(labels.max())
    means = np.zeros((num_nodes, num_classes))
    means[np.arange(num_nodes), labels - 1] = 1
    features = np.random.normal(means, float(feature_noise), means.shape)

    node_ids = []
    hyperedge_ids = []
    next_hyperedge = num_nodes
    with open(osp.join(path, dataset, f"hyperedges-{dataset}.txt")) as handle:
        for line in handle:
            nodes = [int(value) for value in line.strip().split(",")]
            node_ids.extend(nodes)
            hyperedge_ids.extend([next_hyperedge] * len(nodes))
            next_hyperedge += 1
    minimum_node_id = min(node_ids)
    node_ids = [value - minimum_node_id for value in node_ids]
    edge_index = np.asarray(
        [node_ids + hyperedge_ids, hyperedge_ids + node_ids], dtype=np.int64
    )
    return _coalesced_data(
        features,
        labels,
        edge_index,
        next_hyperedge - num_nodes,
        train_percent,
    )
