#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""
from collections import Counter

import torch
import utils

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from layers import *

import math 

from torch_scatter import scatter
from torch_geometric.utils import softmax

def pooled_edge_embeddings(X, V_idx, E_idx, num_edges, reduce='mean'):
    # X: [N, F], V_idx/E_idx: [nnz], num_edges: int (m)
    # returns [m, F]
    if reduce == 'mean':
        return scatter(X[V_idx], E_idx, dim=0, dim_size=num_edges, reduce='mean')
    elif reduce == 'sum':
        return scatter(X[V_idx], E_idx, dim=0, dim_size=num_edges, reduce='sum')
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")

class FeatureConditionedIncidenceMask(nn.Module):
    """
    EHGNN-F (feature-conditioned): s_{v,e} = g_phi([x_v || x_e])
    Top-k is over incidences (nnz entries in edge_index).
    """
    def __init__(self, node_feat_dim, hidden_dim=32, agg='mean', args = None):
        super().__init__()
        self.agg = agg
        in_dim = node_feat_dim * 2
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.reg = args.reg
    @torch.no_grad()
    def _num_edges(self, E_idx):
        return int(E_idx.max().item() + 1)

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask = False):
        V_idx, E_idx = data.edge_index              # [2, nnz]
        X = data.x                                   # [N, F]
        assert E_idx.dtype == torch.long
        assert X.device == E_idx.device

        m = data.num_hyperedges if hasattr(data, 'num_hyperedges') else self._num_edges(E_idx)
        # 1) pooled hyperedge embeddings
        X_e = pooled_edge_embeddings(X, V_idx, E_idx, m, reduce=self.agg)   # [m, F]

        # 2) build per-incidence features [x_v || x_e]
        xv = X[V_idx]                         # [nnz, F]
        xe = X_e[E_idx]                       # [nnz, F]
        pair = torch.cat([xv, xe], dim=1)     # [nnz, 2F]

        # 3) score each incidence
        logits = self.scorer(pair).squeeze(-1)          # [nnz]
        probs  = torch.sigmoid(logits)                  # [nnz]

        # 4) select top-k incidences
        nnz = V_idx.numel()
        k = max(1, int(keep_ratio * nnz))
        if is_test:
            keep_ids = torch.topk(probs, k=k, largest=True).indices
        else:
            if self.reg == 'none':
                # print(scores.shape)
                keep_ids = torch.multinomial(probs, k, replacement = False)
            # TODO: Ablation => Use Bernouli sampling
            elif self.reg == 'l2':
                try:
                    keep_ids = torch.bernoulli(probs).bool()  # Sampled edges based on scores # Need to use regularization in Loss
                except:
                    print(logits.shape)
                    raise ValueError('Scores must be non-negative for L2 regularization.')
            else:
                raise ValueError('Unknown regularizer type: {}'.format(self.reg))
        

        hard = torch.zeros_like(probs)
        hard[keep_ids] = 1.0
        soft = (hard - probs).detach() + probs          # STE

        # 5) mask edge_index rows by kept incidences
        keep_mask = soft > 0.0                          # hard in forward
        V_pruned = V_idx[keep_mask]
        E_pruned = E_idx[keep_mask]
        if return_mask:
            return torch.stack([V_pruned, E_pruned], dim=0), probs, keep_mask
        return torch.stack([V_pruned, E_pruned], dim=0), probs

class FeatureAgnosticEdgeMask(nn.Module):
    """
    EHGNN-C (feature-agnostic): one free logit per hyperedge.
    Top-k is over hyperedges, then we keep all incidences whose E_idx is selected.
    """
    def __init__(self, num_edges, args = None):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_edges))  # s_e
        self.reg = args.reg

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask = False):
        V_idx, E_idx = data.edge_index                          # [2, nnz]
        m = int(self.logits.numel())
        probs = torch.sigmoid(self.logits)                 # [m]
        k = max(1, int(keep_ratio * m))

        if is_test:
            keep_e = torch.topk(probs, k=k, largest=True).indices
        else:
            if self.reg == 'none':
                keep_e = torch.multinomial(probs, k)
            elif self.reg == 'l2':
                keep_e = torch.bernoulli(probs).bool().nonzero()
            else:
                raise ValueError('Unknown regularizer type: {}'.format(self.reg))

        # STE mask on edges
        hard_e = torch.zeros_like(probs)
        hard_e[keep_e] = 1.0
        soft_e = (hard_e - probs).detach() + probs         # STE over edges

        # lift to incidences
        edge_keep_mask = soft_e[E_idx] > 0.0               # [nnz] hard in forward
        V_pruned = V_idx[edge_keep_mask]
        E_pruned = E_idx[edge_keep_mask]
        if return_mask:
            return torch.stack([V_pruned, E_pruned], dim=0),probs, edge_keep_mask
        return torch.stack([V_pruned, E_pruned], dim=0), probs

def compute_sparsification_ratio(original_edge_index, pruned_edge_index):
    original_edges = original_edge_index.size(1)
    pruned_edges = pruned_edge_index.size(1)
    ratio = 1.0 - (pruned_edges / original_edges)
    return ratio
def log_sparsification_info(original_edge_index, pruned_edge_index, label=None, verbose = False):
    ratio = compute_sparsification_ratio(original_edge_index, pruned_edge_index)
    # zerodegnodes = count_zero_degree_nodes(pruned_edge_index,data.n_x)
    # zerodegedges = count_zero_degree_hyperedges(pruned_edge_index,data.num_hyperedges)
    tag = f"[{label}] " if label else ""
    # print(f"{tag}Sparsification Ratio: {ratio:.4f} ({pruned_edge_index.size(1)} / {original_edge_index.size(1)} edges retained): {zerodegnodes}-{zerodegedges}")
    if verbose:
        print(f"{tag}Sparsification Ratio: {ratio:.4f} ({pruned_edge_index.size(1)} / {original_edge_index.size(1)} edges retained)")
    return ratio 

# ---- Learnable Mask Baseline Sparsifier ----
class LearnableEdgeMask_wreg(nn.Module):
    def __init__(self, args, num_edges):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_edges))
        # self.threhold = args.KNN_PRUNE_K
        self.args = args

    def forward(self, data, is_training=True):
        V_idx, E_idx = data.edge_index
        probs = torch.sigmoid(self.logits[E_idx])

        if is_training:
            straight_through = (torch.bernoulli(probs) - probs).detach() + probs
            sampled = straight_through > 0.5
        else:
            k = min(self.args.KNN_PRUNE_K, self.logits.numel())
            topk_indices = torch.topk(torch.sigmoid(self.logits), k=k).indices
            sampled = torch.isin(E_idx, topk_indices)

        V_pruned = V_idx[sampled]
        E_pruned = E_idx[sampled]
        return torch.stack([V_pruned, E_pruned], dim=0)

    def regularization_loss(self):
        probs = torch.sigmoid(self.logits)
        l1 = probs.mean()  # Encourage sparsity
        # entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
        return self.args.LAMBDA_L1 * l1  # + LAMBDA_ENT * entropy.mean()
    
# # ---- Learnable Mask Baseline Sparsifier ----
# class LearnableEdgeMask(nn.Module):
#     def __init__(self, num_edges, args):
#         super().__init__()
#         self.args = args
#         self.logits = nn.Parameter(torch.randn(num_edges))

#     def forward(self, edge_index, keep_ratio = 0.5,is_test = False, return_mask = False):
#         V_idx, E_idx = edge_index
#         # probs = torch.sigmoid(self.logits[E_idx])
#         # mask = (probs > 0.5)
#         # V_pruned = V_idx[mask]
#         # E_pruned = E_idx[mask]
#         probs = torch.sigmoid(self.logits)
#         # probs = (self.logits - self.logits.min())/(self.logits.max()-self.logits.min())
#         # sampled = torch.bernoulli(probs).bool()
#         # sampled = (torch.bernoulli(probs) - probs).detach() + probs  # Gradient flows through probs
#         # # sampled = sampled > 0.5
#         # # V_pruned = V_idx[sampled]
#         # # E_pruned = E_idx[sampled]
#         topk = int(keep_ratio * len(V_idx))
#         if is_test:
#             _, keep_ids = torch.topk(probs, k=topk)
#         else:
#             keep_ids = torch.multinomial(probs, topk)


#         # sampled = (torch.bernoulli(probs) - probs).detach() + probs  # Gra
#         # topk = int(keep_ratio * len(V_idx))
#         # _, keep_ids = torch.topk(sampled, k=topk)

#         mask = torch.isin(E_idx, keep_ids)
#         V_pruned = V_idx[mask]
#         E_pruned = E_idx[mask]
#         if return_mask:
#             return torch.stack([V_pruned, E_pruned], dim=0), probs, mask
#         return torch.stack([V_pruned, E_pruned], dim=0),probs

# ---- Learnable Mask Baseline Sparsifier with straight-through estimator ----
class LearnableEdgeMask(nn.Module):
    def __init__(self, num_edges, args):
        """
        Here num_edges is actually # node, hyperedge connections not # hyperedges. (see train_sparse.py)
        """
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_edges))
        self.reg = args.reg
    def inference_soft(H, p_matrix):
        # H: incidence matrix, shape [n, m]
        # p_matrix: learned probabilities, shape [n, m]
        return H * p_matrix  # elementwise expected sparsified structure

    def forward(self, data, keep_ratio = 0.5,is_test = False, return_mask = False):
        V_idx, E_idx = data.edge_index
        # probs = torch.sigmoid(self.logits[E_idx])
        # mask = (probs > 0.5)
        # V_pruned = V_idx[mask]
        # E_pruned = E_idx[mask]
        probs = torch.sigmoid(self.logits)
        # probs = probs/probs.sum() # Normalize
        # probs = (self.logits - self.logits.min())/(self.logits.max()-self.logits.min())
        # sampled = torch.bernoulli(probs).bool()
        # sampled = (torch.bernoulli(probs) - probs).detach() + probs  # Gradient flows through probs
        # # sampled = sampled > 0.5
        # # V_pruned = V_idx[sampled]
        # # E_pruned = E_idx[sampled]
        scores = probs[E_idx]
        # scores = scores.clamp(0,1)
        # assert (scores>=0 and scores<=1)
        # scores = (scores - scores.min())/(scores.max()-scores.min())  # Ensure non-negative scores
        hard_mask = torch.zeros_like(scores)
        topk = int(keep_ratio * len(V_idx))
        
        if is_test:
            _, keep_ids = torch.topk(scores, k=topk)
            # TODO: Ablation 
            # self.inference_soft(scores, )
        else:
            if self.reg == 'none':
                # print(scores.shape)
                keep_ids = torch.multinomial(scores, topk, replacement = False)
            # TODO: Ablation => Use Bernouli sampling
            elif self.reg == 'l2':
                try:
                    keep_ids = torch.bernoulli(scores).bool()  # Sampled edges based on scores # Need to use regularization in Loss
                except:
                    print(scores.shape)
                    raise ValueError('Scores must be non-negative for L2 regularization.')
            else:
                raise ValueError('Unknown regularizer type: {}'.format(self.reg))
        hard_mask[keep_ids] = 1.0
        soft_mask = (hard_mask - scores).detach() + scores  # straight-through trick

        # Apply mask to incidence
        keep_mask = soft_mask > 0  # still binary at forward time
        # print(probs[keep_mask].min())
        # sampled = (torch.bernoulli(probs) - probs).detach() + probs  # Gra
        # topk = int(keep_ratio * len(V_idx))
        # _, keep_ids = torch.topk(sampled, k=topk)

        V_pruned = V_idx[keep_mask]
        E_pruned = E_idx[keep_mask]
        if return_mask:
            return torch.stack([V_pruned, E_pruned], dim=0),scores, keep_mask

        return torch.stack([V_pruned, E_pruned], dim=0),scores

# Learnable edge mask with hyperedge scoring
class LearnableEdgeMaskplus(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, agg='mean', args = None):
        super().__init__()
        self.scoring_fn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.agg = agg  # aggregation method: 'mean' or 'max'
        self.reg = args.reg
    def forward(self, data, keep_ratio = 0.5, is_test = False, return_mask = False):
        V_idx, E_idx = data.edge_index
        X = data.x
        num_edges = data.num_hyperedges
        # num_edges = int(E_idx.max().item()) + 1
        # Aggregate node features for each hyperedge
        # edge_feat_sum = torch.zeros(num_edges, X.shape[1], device=X.device)
        # edge_feat_count = torch.zeros(num_edges, 1, device=X.device)
        # print('X[V_idx].shape = ',X[V_idx].shape,' x.shape: ',X.shape)
        # edge_feat_sum.index_add_(0, E_idx, X[V_idx])
        # edge_feat_count.index_add_(0, E_idx, torch.ones_like(E_idx, dtype=torch.float).unsqueeze(-1))
        assert E_idx.dtype == torch.long, "E_idx must be torch.long"
        assert E_idx.min() >= 0, "E_idx has negative indices!"
        # print(E_idx.max(), num_edges,E_idx.min())
        assert E_idx.max() < num_edges, "E_idx contains out-of-bounds indices!"
        assert E_idx.device == X.device, "E_idx and X must be on the same device"
        if type(num_edges) is not int:
            num_edges = int(num_edges.item())
        # print(V_idx.shape, X.shape, E_idx.shape, X[V_idx].shape)
        edge_feat_sum = scatter(X[V_idx], E_idx, dim=0, dim_size=num_edges, reduce='mean')
        edge_feats = edge_feat_sum 
        # edge_feat_count = scatter(torch.ones_like(E_idx, dtype=torch.float), E_idx, dim=0, dim_size=num_edges,
        #                           reduce='sum').unsqueeze(-1)

        # edge_feat_count = edge_feat_count.to(edge_feat_sum.device)
        # # print(edge_feat_sum.shape, edge_feat_count.shape)
        # if self.agg == 'mean':
        #     edge_feats = edge_feat_sum/ edge_feat_count.clamp(min=1.0)
        # elif self.agg == 'max':
        #     raise NotImplementedError("Max aggregation is not implemented yet.")
        # else:
        #     raise ValueError("Unsupported aggregation: choose from 'mean'.")

        # Score each hyperedge
        # print('edge_feats.shape: ',edge_feats.shape)
        edge_logits = self.scoring_fn(edge_feats).squeeze(-1)  # shape: [num_edges]
        probs = torch.sigmoid(edge_logits)
        # print(probs.shape)
        
        # Sample top-k edges based on scores
        # topk = int(keep_ratio * len(V_idx))
        # topk = 3614
        topk = int(keep_ratio * num_edges)
        if is_test:
            _, keep_ids = torch.topk(probs, k=topk)
        else:
            if self.reg == 'none':
                keep_ids = torch.multinomial(probs, topk)
            elif self.reg == 'l2':
                keep_ids = torch.bernoulli(probs).bool().nonzero()
            else:
                raise ValueError('Unknown regularizer type: {}'.format(self.reg))
            # TODO: Ablation => Use Bernouli sampling


        # sampled = (torch.bernoulli(probs) - probs).detach() + probs  # Gra
        # topk = int(keep_ratio * len(V_idx))
        # _, keep_ids = torch.topk(sampled, k=topk)

        # Mask the incidence matrix
        mask = torch.isin(E_idx, keep_ids)
        V_pruned = V_idx[mask]
        E_pruned = E_idx[mask]
        if return_mask:
            return torch.stack([V_pruned, E_pruned], dim=0),probs, mask

        return torch.stack([V_pruned, E_pruned], dim=0),probs

# ---- Effective Resistance-Based Sparsifier ----
def effective_resistance_sparsify(edge_index, num_nodes, keep_ratio=0.5):
    V_idx, E_idx = edge_index
    num_edges = E_idx.max().item() + 1
    device = edge_index.device

    # Build Laplacian approximation from incidence
    H = torch.zeros((num_nodes, num_edges), device=device)
    H[V_idx, E_idx] = 1.0
    Dv = torch.diag(H.sum(dim=1))  # node degree matrix
    De_inv = torch.diag(1.0 / (H.sum(dim=0) + 1e-8))  # hyperedge degree inverse
    L = Dv - H @ De_inv @ H.T  # Zhou's Laplacian

    try:
        L_inv_diag = torch.linalg.pinv(L).diag()  # pseudoinverse diagonal
    except RuntimeError:
        L_inv_diag = torch.ones(num_nodes, device=device)  # fallback

    edge_scores = torch.zeros(num_edges, device=device)
    for e_id in range(num_edges):
        nodes_in_e = V_idx[E_idx == e_id]
        if len(nodes_in_e) > 1:
            edge_scores[e_id] = L_inv_diag[nodes_in_e].sum()

    topk = int(keep_ratio * num_edges)
    _, keep_ids = torch.topk(edge_scores, k=topk)
    mask = torch.isin(E_idx, keep_ids)
    return edge_index[:, mask]

# ---- Edge Centrality-Based Sparsifier ----
def edge_centrality_sparsify(edge_index, num_nodes, num_edges, keep_ratio=0.5):
    V_idx, E_idx = edge_index
    device = edge_index.device

    # Bipartite graph: create (num_nodes + num_edges) x (num_nodes + num_edges) matrix
    total_nodes = num_nodes + num_edges
    bip_adj = torch.zeros((total_nodes, total_nodes), device=device)
    for v, e in zip(V_idx, E_idx):
        bip_adj[v, num_nodes + e] = 1
        bip_adj[num_nodes + e, v] = 1

    # Normalize adjacency
    deg = bip_adj.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1  # avoid division by zero
    bip_adj = bip_adj / deg

    # Power iteration for node centrality (PageRank-style diffusion)
    centrality = torch.ones(total_nodes, device=device) / total_nodes
    for _ in range(10):
        centrality = bip_adj @ centrality
        centrality = centrality / centrality.sum()

    edge_centrality = centrality[num_nodes:]  # only hyperedges
    topk = int(keep_ratio * num_edges)
    _, keep_ids = torch.topk(edge_centrality, k=topk)
    mask = torch.isin(E_idx, keep_ids)
    return edge_index[:, mask]

# # ---- Random Sparsifier ----
# def random_sparsify(edge_index, num_edges, keep_ratio=0.5):
#     _, E_idx = edge_index
#     # num_edges = E_idx.max().item() + 1
#     perm = torch.randperm(num_edges, device=edge_index.device)
#     k = int(keep_ratio * num_edges)
#     selected = perm[:k]
#     mask = torch.isin(E_idx, selected)
#     return edge_index[:, mask]

# ---- Random Sparsifier (Straight-Through Version) ----
def random_sparsify(edge_index, num_edges, keep_ratio=0.5):
    V_idx, E_idx = edge_index
    # num_edges = E_idx.max().item() + 1
    device = edge_index.device

    rand_probs = torch.rand(edge_index.shape[1], device=device)
    mask_prob = (rand_probs < keep_ratio).float()
    mask_prob = (mask_prob - mask_prob.detach()) + mask_prob  # Straight-through estimator

    sampled = mask_prob > 0.5
    V_pruned = V_idx[sampled]
    E_pruned = E_idx[sampled]
    return torch.stack([V_pruned, E_pruned], dim=0)

# ---- Degree distribution based Sparsifier -----
def degree_distribution_sparsify(edge_index,keep_ratio=0.5):
    V_idx, E_idx = edge_index
    node_deg = torch.bincount(V_idx, minlength=V_idx.max().item() + 1).float()
    # print('max degree: ',node_deg.max(),' min degree: ',node_deg.min())
    probs = node_deg / node_deg.sum()
    num_samples = int(keep_ratio * node_deg.shape[0])
    sampled_nodes = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    keep_mask = torch.isin(V_idx, sampled_nodes)
    return edge_index[:, keep_mask]
#  This part is for HyperGCN

def scatter_topk(values, group_idx, k_vals, largest=True):
    # values: [N], group_idx: [N], k_vals: [N]
    # Assumes group_idx is sorted or uses torch.argsort(group_idx) beforehand
    # Return: top-k values (flat indices into original array)


    # Sort by group_idx then by value
    sorted_group, perm = torch.sort(group_idx * values.numel() + (values if largest else -values), descending=True)
    group_idx_sorted = group_idx[perm]


    # Count how many to keep per group (based on k_vals)
    _, counts = torch.unique_consecutive(group_idx_sorted, return_counts=True)
    max_k = k_vals.max().item()
    mask = torch.zeros_like(values, dtype=torch.bool)


    start = 0
    for count in counts:
        end = start + count
        group_k = k_vals[group_idx_sorted[start]]
        top_k = min(group_k, count)
        mask[perm[start:start+top_k]] = True
        start = end
    return values[mask], mask.nonzero(as_tuple=True)[0]


class NeuralSparseHG(nn.Module):
    def __init__(self, num_nodes, num_edges, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.logits = nn.Parameter(torch.randn(num_nodes, num_edges))  # Full (v,e) space


    # def forward(self, edge_index, keep_ratio=0.5, is_test=False):
    #     V_idx, E_idx = edge_index  # shape [t]
    #     scores = torch.sigmoid(self.logits[V_idx, E_idx])  # shape [t]


    #     # Compute node degree (# of incidences per node)
    #     deg = scatter(torch.ones_like(V_idx), V_idx, dim=0, dim_size=self.logits.shape[0], reduce='add')
    #     deg = deg[V_idx]  # shape [t]


    #     # Compute dynamic top-k per node
    #     # k_vals = (deg.float() * keep_ratio).clamp(min=1).long()  # shape [t]
        
    #     # Sort scores per node
    #     # sorted_scores, perm = scatter_topk(scores, V_idx, k_vals, largest=True)


    #     sorted_scores, perm = scatter_topk(scores, V_idx, torch.ones_like(deg)*self.top_k, largest=True)


    #     hard_mask = torch.zeros_like(scores)
    #     hard_mask[perm] = 1.0


    #     soft_mask = (hard_mask - scores).detach() + scores  # straight-through estimator
    #     keep_mask = soft_mask > 0


    #     V_pruned = V_idx[keep_mask]
        # E_pruned = E_idx[keep_mask]
        # return torch.stack([V_pruned, E_pruned], dim=0), scores
        
    def forward(self, data, keep_ratio=0.5, is_test=False):
        V_idx, E_idx = data.edge_index  # shape [t]
        scores = torch.sigmoid(self.logits[V_idx, E_idx])  # shape [t]


        # Build list of retained incidence indices (static top-k per node)
        num_nodes = self.logits.shape[0]
        hard_mask = torch.zeros_like(scores)
        node_to_indices = [[] for _ in range(num_nodes)]


        for i, v in enumerate(V_idx):
            node_to_indices[v.item()].append(i)


        keep_ids = []
        for v, idx_list in enumerate(node_to_indices):
            if not idx_list:
                continue
            idx_tensor = torch.tensor(idx_list, device=scores.device)
            node_scores = scores[idx_tensor]
            k = min(self.top_k, len(idx_list))
            if is_test:
                _, topk_indices = torch.topk(node_scores, k=k)
            else:
                topk_indices = torch.multinomial(node_scores, k, replacement=False)
            keep_ids.extend(idx_tensor[topk_indices].tolist())


        keep_ids = torch.tensor(keep_ids, device=scores.device)
        hard_mask[keep_ids] = 1.0
        soft_mask = (hard_mask - scores).detach() + scores  # straight-through estimator
        keep_mask = soft_mask > 0


        V_pruned = V_idx[keep_mask]
        E_pruned = E_idx[keep_mask]
        return torch.stack([V_pruned, E_pruned], dim=0), scores


class HyperGCN(nn.Module):
    def __init__(self, V, E, E_p, X, num_features, num_layers, num_classses, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = num_features, num_layers, num_classses
        cuda = args.cuda  # and torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dname == 'citeseer':
                power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.HyperGCN_fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.HyperGCN_mediators)
            # structure_p = utils.Laplacian(V, E_p, X, args.HyperGCN_mediators)
        else:
            reapproximate = True
            structure = E
            # structure_p = E_p

        self.layers = nn.ModuleList([utils.HyperGraphConvolution(
            h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, num_layers
        self.structure, self.m = structure, args.HyperGCN_mediators

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data_input):
        """
        an l-layer GCN
        """
        data, test_flag = data_input
        do, l, m = self.do, self.l, self.m
        H = data.x

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)

        return H


# class CEGCN(MessagePassing):
#     def __init__(self,
#                  in_dim,
#                  hid_dim,
#                  out_dim,
#                  num_layers,
#                  dropout,
#                  Normalization='bn'
#                  ):
#         super(CEGCN, self).__init__()
#         self.convs = nn.ModuleList()
#         self.normalizations = nn.ModuleList()

#         if Normalization == 'bn':
#             self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
#             self.normalizations.append(nn.BatchNorm1d(hid_dim))
#             for _ in range(num_layers-2):
#                 self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
#                 self.normalizations.append(nn.BatchNorm1d(hid_dim))

#             self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))
#         else:  # default no normalizations
#             self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
#             self.normalizations.append(nn.Identity())
#             for _ in range(num_layers-2):
#                 self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
#                 self.normalizations.append(nn.Identity())

#             self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for layer in self.convs:
#             layer.reset_parameters()
#         for normalization in self.normalizations:
#             if not (normalization.__class__.__name__ == 'Identity'):
#                 normalization.reset_parameters()

#     def forward(self, data):
#         #         Assume edge_index is already V2V
#         x, edge_index, norm = data.x, data.edge_index, data.norm
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index, norm)
#             x = F.relu(x, inplace=True)
#             x = self.normalizations[i](x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index, norm)
#         return x

class CEGCN(MessagePassing):
    def __init__(self, 
                 args,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGCN, self).__init__()
        self.args = args 
        if args.mode == 'learnmask':
            self.mask_module = LearnableEdgeMask(args.num_hyperedges,args=args)
        if args.mode == 'learnmask+':
            self.mask_module = LearnableEdgeMaskplus(args.F,hidden_dim=args.coarse_MLP, args=args)
        self.ratio = 1.0
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()


        if Normalization == 'bn':
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))


            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))
        else:  # default no normalizations
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.Identity())


            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))


        self.dropout = dropout


    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()


    def forward(self, data, is_test = False):
        #         Assume edge_index is already V2V
        if self.args.mode == 'learnmask':
            # edge_index_knn,probs = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio)
            edge_index_knn,probs,keepmask = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio,is_test=is_test,return_mask = True)
        if self.args.mode == 'learnmask+':
            # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
            edge_index_knn, probs,keepmask = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
        if self.args.mode.startswith('learnmask'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
        else:
            edge_index_knn = data.edge_index


        x, edge_index, norm = data.x, edge_index_knn, data.norm
        if self.args.mode.startswith('learnmask'):
            norm = norm[keepmask]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, norm)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, norm)
        if self.args.mode.startswith('learnmask') and is_test == False:
            return x, probs
        return x



# class CEGAT(MessagePassing):
#     def __init__(self,
#                  in_dim,
#                  hid_dim,
#                  out_dim,
#                  num_layers,
#                  heads,
#                  output_heads,
#                  dropout,
#                  Normalization='bn'
#                  ):
#         super(CEGAT, self).__init__()
#         self.convs = nn.ModuleList()
#         self.normalizations = nn.ModuleList()

#         if Normalization == 'bn':
#             self.convs.append(GATConv(in_dim, hid_dim, heads))
#             self.normalizations.append(nn.BatchNorm1d(hid_dim))
#             for _ in range(num_layers-2):
#                 self.convs.append(GATConv(heads*hid_dim, hid_dim))
#                 self.normalizations.append(nn.BatchNorm1d(hid_dim))

#             self.convs.append(GATConv(heads*hid_dim, out_dim,
#                                       heads=output_heads, concat=False))
#         else:  # default no normalizations
#             self.convs.append(GATConv(in_dim, hid_dim, heads))
#             self.normalizations.append(nn.Identity())
#             for _ in range(num_layers-2):
#                 self.convs.append(GATConv(hid_dim*heads, hid_dim))
#                 self.normalizations.append(nn.Identity())

#             self.convs.append(GATConv(hid_dim*heads, out_dim,
#                                       heads=output_heads, concat=False))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for layer in self.convs:
#             layer.reset_parameters()
#         for normalization in self.normalizations:
#             if not (normalization.__class__.__name__ == 'Identity'):
#                 normalization.reset_parameters()

#     def forward(self, data):
#         #         Assume edge_index is already V2V
#         x, edge_index, norm = data.x, data.edge_index, data.norm
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             x = F.relu(x, inplace=True)
#             x = self.normalizations[i](x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x

class CEGAT(MessagePassing):
    def __init__(self, 
                 args,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 heads,
                 output_heads,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGAT, self).__init__()
        self.args = args 
        if args.mode == 'learnmask':
            self.mask_module = LearnableEdgeMask(args.num_hyperedges,args=args)
        if args.mode == 'learnmask+':
            self.mask_module = LearnableEdgeMaskplus(args.F,hidden_dim=args.coarse_MLP, args=args)
        self.ratio = 1.0


        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()


        if Normalization == 'bn':
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(heads*hid_dim, hid_dim))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))


            self.convs.append(GATConv(heads*hid_dim, out_dim,
                                      heads=output_heads, concat=False))
        else:  # default no normalizations
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hid_dim*heads, hid_dim))
                self.normalizations.append(nn.Identity())


            self.convs.append(GATConv(hid_dim*heads, out_dim,
                                      heads=output_heads, concat=False))


        self.dropout = dropout


    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()


    def forward(self, data, is_test=False):
        #         Assume edge_index is already V2V
        if self.args.mode == 'learnmask':
            # edge_index_knn,probs = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio)
            edge_index_knn,probs,keepmask = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio,is_test=is_test,return_mask = True)
        if self.args.mode == 'learnmask+':
            # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
            edge_index_knn, probs,keepmask = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
        if self.args.mode.startswith('learnmask'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
        else:
            edge_index_knn = data.edge_index
        x, edge_index, norm = data.x, edge_index_knn, data.norm
        if self.args.mode.startswith('learnmask'):
            norm = norm[keepmask]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if self.args.mode.startswith('learnmask') and is_test == False:
            return x, probs
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        x = data.x
        G = data.edge_index

        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class HNHN(nn.Module):
    """
    """

    def __init__(self, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        
        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
        else:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.MLP_hidden,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden,
                                           nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x
        
        if self.num_layers == 1:
            conv = self.convs[0]
            x = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data)

        return x


class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer 
    is implemented in pyg.
    """

    def __init__(self, args):
        super(HCHA, self).__init__()
        self.args = args 
        if args.mode == 'learnmask':
            self.mask_module = LearnableEdgeMask(args.num_hyperedges,args=args)
        if args.mode == 'learnmask+':
            self.mask_module = LearnableEdgeMaskplus(args.F,hidden_dim=args.coarse_MLP, args=args)
        if args.mode == 'learnmask_cond':    # NEW: feature-conditioned *incidence* mask (EHGNN-F)
            self.mask_module = FeatureConditionedIncidenceMask(args.F, hidden_dim=32, agg='mean',args=args)
        if args.mode == 'learnmask+_agn':     # NEW: feature-agnostic *edge* mask (EHGNN-C, explicit)
            self.mask_module = FeatureAgnosticEdgeMask(int(args.num_hyperedges),args=args)
        if args.mode == 'Neural':
            self.mask_module = NeuralSparseHG(args.n_x, args.num_hyperedges, top_k=5)
        print('All_num_layers: ',args.All_num_layers,' MLP_hidden: ',args.MLP_hidden)
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.ratio = 1.0
#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(args.num_features,
                                         args.MLP_hidden, self.symdegnorm))
        for _ in range(self.num_layers-2):
            self.convs.append(HypergraphConv(
                args.MLP_hidden, args.MLP_hidden, self.symdegnorm))
        # Output heads is set to 1 as default
        self.convs.append(HypergraphConv(
            args.MLP_hidden, args.num_classes, self.symdegnorm))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, is_test=False,return_mask = False):
        if self.args.mode == 'learnmask' or self.args.mode == 'learnmask_cond' or self.args.mode =='Neural':
            if return_mask:
                edge_index_knn,probs,mask = self.mask_module(data, keep_ratio= self.args.keep_ratio, is_test=is_test, return_mask = True)
            else:
                # edge_index_knn,probs = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio)
                edge_index_knn,probs = self.mask_module(data,keep_ratio= self.args.keep_ratio,is_test=is_test)

        if self.args.mode == 'learnmask+' or self.args.mode == 'learnmask+_agn':
            if return_mask:
                edge_index_knn,probs,mask = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
            else:
                # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
                edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test)
        # print(edge_index_knn.shape)
        if self.args.mode.startswith('learnmask') or self.args.mode.startswith('Neural'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
            # from matplotlib import pyplot as plt
            # # plt.scatter(range(len(probs)),probs)
            # #
            # plt.figure(figsize=(6, 4))
            # plt.hist(probs, bins=20, range=(0, 1), edgecolor='black')
            # plt.xlabel('Value (binned)')
            # plt.ylabel('Frequency')
            # plt.title('Frequency Distribution of Rational Numbers (0 to 1)')
            # plt.grid(True, linestyle='--', alpha=0.5)
            # plt.show()
            # plt.clf()
        elif self.args.mode.startswith('Neural'):
            edge_index_knn, probs = self.mask_module(data.edge_index, keep_ratio=self.args.keep_ratio, is_test=is_test)
        else:
            edge_index_knn = data.edge_index
        x = data.x
        # edge_index = data.edge_index
        edge_index = edge_index_knn
        if self.args.mode.startswith('learnmask') or self.args.mode.startswith('Neural'):
            num_nodes = data.n_x
            num_hyperedges = data.num_hyperedges

            nodes, hyperedges = edge_index[0], edge_index[1]

            # Step 1: Identify hyperedges that are already self-loops (appear only once)
            _, counts = torch.unique(hyperedges, return_counts=True)
            singleton_mask = counts == 1
            singleton_hyperedges = torch.unique(hyperedges)[singleton_mask]

            # Step 2: Identify the nodes that already have self-loops
            # Create a mask for entries with singleton hyperedges
            self_loop_mask = torch.isin(hyperedges, singleton_hyperedges)
            existing_self_loop_nodes = nodes[self_loop_mask].unique()

            # Step 3: Find nodes that need self-loops
            all_nodes = torch.arange(num_nodes, dtype=torch.long).to(edge_index.device)
            needs_self_loop_mask = ~torch.isin(all_nodes, existing_self_loop_nodes)
            nodes_to_add = all_nodes[needs_self_loop_mask]

            # Step 4: Construct new self-loop hyperedges
            num_new_edges = nodes_to_add.size(0)
            if num_new_edges > 0:
                start_hyperedge_id = hyperedges.max().item() + 1
                new_hyperedge_ids = torch.arange(start_hyperedge_id,
                                                 start_hyperedge_id + num_new_edges,
                                                 dtype=torch.long).to(edge_index.device)

                new_edges = torch.stack([nodes_to_add, new_hyperedge_ids], dim=0)
                updated_edge_index = torch.cat([edge_index, new_edges], dim=1)
            else:
                updated_edge_index = edge_index

            # Step 5: Sort by node index for consistency
            sorted_idx = torch.argsort(updated_edge_index[0])
            edge_index = updated_edge_index[:, sorted_idx].long()

            # num_nodes = data.n_x
            # num_hyperedges = data.num_hyperedges
            #
            # # if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
            # #    print('num_hyperedges does not match! 2')
            # #    return
            #
            # hyperedge_appear_fre = Counter(edge_index[1].clone().cpu().numpy())
            # # store the nodes that already have self-loops
            # skip_node_lst = []
            # for edge in hyperedge_appear_fre:
            #     if hyperedge_appear_fre[edge] == 1:
            #         skip_node = edge_index[0][torch.where(
            #             edge_index[1] == edge)[0].item()]
            #         skip_node_lst.append(skip_node.item())
            #
            # new_edge_idx = edge_index[1].max() + 1
            # # print(num_nodes, len(skip_node_lst), num_nodes - len(skip_node_lst))
            # new_edges = torch.zeros(
            #     (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
            # tmp_count = 0
            # for i in range(num_nodes):
            #     if i not in skip_node_lst:
            #         new_edges[0][tmp_count] = i
            #         new_edges[1][tmp_count] = new_edge_idx
            #         new_edge_idx += 1
            #         tmp_count += 1
            # new_edges = new_edges.to(x.device)
            # # data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
            # edge_index = torch.cat((edge_index, new_edges), dim=1)
            # # Sort along w.r.t. nodes
            # _, sorted_idx = torch.sort(edge_index[0])
            # edge_index = edge_index[:, sorted_idx].type(torch.LongTensor).to(x.device)
            # # print(x.device,edge_index.device)
        for i, conv in enumerate(self.convs[:-1]):
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

#         x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if (self.args.mode.startswith('learnmask') or self.args.mode.startswith('Neural')) and is_test == False:
            if return_mask:
                return x, probs, mask
            return x, probs
            
        return x


# class SetGNN(nn.Module):
#     def __init__(self, args, norm_size=None):
#         super(SetGNN, self).__init__()
#         """
#         args should contain the following:
#         V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
#         E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
#         All_num_layers,dropout
#         !!! V_in_dim should be the dimension of node features
#         !!! E_out_dim should be the number of classes (for classification)
#         """
# #         V_in_dim = V_dict['in_dim']
# #         V_enc_hid_dim = V_dict['enc_hid_dim']
# #         V_dec_hid_dim = V_dict['dec_hid_dim']
# #         V_out_dim = V_dict['out_dim']
# #         V_enc_num_layers = V_dict['enc_num_layers']
# #         V_dec_num_layers = V_dict['dec_num_layers']

# #         E_in_dim = E_dict['in_dim']
# #         E_enc_hid_dim = E_dict['enc_hid_dim']
# #         E_dec_hid_dim = E_dict['dec_hid_dim']
# #         E_out_dim = E_dict['out_dim']
# #         E_enc_num_layers = E_dict['enc_num_layers']
# #         E_dec_num_layers = E_dict['dec_num_layers']

# #         Now set all dropout the same, but can be different
#         self.All_num_layers = args.All_num_layers
#         self.dropout = args.dropout
#         self.aggr = args.aggregate
#         self.NormLayer = args.normalization
#         self.InputNorm = args.deepset_input_norm
#         self.GPR = args.GPR
#         self.LearnMask = args.LearnMask
# #         Now define V2EConvs[i], V2EConvs[i] for ith layers
# #         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
# #         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
#         self.V2EConvs = nn.ModuleList()
#         self.E2VConvs = nn.ModuleList()
#         self.bnV2Es = nn.ModuleList()
#         self.bnE2Vs = nn.ModuleList()

#         if self.LearnMask:
#             self.Importance = Parameter(torch.ones(norm_size))

#         if self.All_num_layers == 0:
#             self.classifier = MLP(in_channels=args.num_features,
#                                   hidden_channels=args.Classifier_hidden,
#                                   out_channels=args.num_classes,
#                                   num_layers=args.Classifier_num_layers,
#                                   dropout=self.dropout,
#                                   Normalization=self.NormLayer,
#                                   InputNorm=False)
#         else:
#             self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
#                                              hid_dim=args.MLP_hidden,
#                                              out_dim=args.MLP_hidden,
#                                              num_layers=args.MLP_num_layers,
#                                              dropout=self.dropout,
#                                              Normalization=self.NormLayer,
#                                              InputNorm=self.InputNorm,
#                                              heads=args.heads,
#                                              attention=args.PMA))
#             self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
#             self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
#                                              hid_dim=args.MLP_hidden,
#                                              out_dim=args.MLP_hidden,
#                                              num_layers=args.MLP_num_layers,
#                                              dropout=self.dropout,
#                                              Normalization=self.NormLayer,
#                                              InputNorm=self.InputNorm,
#                                              heads=args.heads,
#                                              attention=args.PMA))
#             self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
#             for _ in range(self.All_num_layers-1):
#                 self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
#                                                  hid_dim=args.MLP_hidden,
#                                                  out_dim=args.MLP_hidden,
#                                                  num_layers=args.MLP_num_layers,
#                                                  dropout=self.dropout,
#                                                  Normalization=self.NormLayer,
#                                                  InputNorm=self.InputNorm,
#                                                  heads=args.heads,
#                                                  attention=args.PMA))
#                 self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden)) 
#                 self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
#                                                  hid_dim=args.MLP_hidden,
#                                                  out_dim=args.MLP_hidden,
#                                                  num_layers=args.MLP_num_layers,
#                                                  dropout=self.dropout,
#                                                  Normalization=self.NormLayer,
#                                                  InputNorm=self.InputNorm,
#                                                  heads=args.heads,
#                                                  attention=args.PMA))
#                 self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
#             if self.GPR:
#                 self.MLP = MLP(in_channels=args.num_features,
#                                hidden_channels=args.MLP_hidden,
#                                out_channels=args.MLP_hidden,
#                                num_layers=args.MLP_num_layers,
#                                dropout=self.dropout,
#                                Normalization=self.NormLayer,
#                                InputNorm=False)
#                 self.GPRweights = Linear(self.All_num_layers+1, 1, bias=False)
#                 self.classifier = MLP(in_channels=args.MLP_hidden,
#                                       hidden_channels=args.Classifier_hidden,
#                                       out_channels=args.num_classes,
#                                       num_layers=args.Classifier_num_layers,
#                                       dropout=self.dropout,
#                                       Normalization=self.NormLayer,
#                                       InputNorm=False)
#             else:
#                 self.classifier = MLP(in_channels=args.MLP_hidden,
#                                       hidden_channels=args.Classifier_hidden,
#                                       out_channels=args.num_classes,
#                                       num_layers=args.Classifier_num_layers,
#                                       dropout=self.dropout,
#                                       Normalization=self.NormLayer,
#                                       InputNorm=False)


# #         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
# #         However, in general this can be arbitrary.


#     def reset_parameters(self):
#         for layer in self.V2EConvs:
#             layer.reset_parameters()
#         for layer in self.E2VConvs:
#             layer.reset_parameters()
#         for layer in self.bnV2Es:
#             layer.reset_parameters()
#         for layer in self.bnE2Vs:
#             layer.reset_parameters()
#         self.classifier.reset_parameters()
#         if self.GPR:
#             self.MLP.reset_parameters()
#             self.GPRweights.reset_parameters()
#         if self.LearnMask:
#             nn.init.ones_(self.Importance)

#     def forward(self, data):
#         """
#         The data should contain the follows
#         data.x: node features
#         data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
#         !!! Note that self loop should be assigned to a new (hyper)edge id!!!
#         !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
#         data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
#         !!! Note that we output final node representation. Loss should be defined outside.
#         """
# #             The data should contain the follows
# #             data.x: node features
# #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
# #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

#         x, edge_index, norm = data.x, data.edge_index, data.norm
#         if self.LearnMask:
#             norm = self.Importance*norm
#         cidx = edge_index[1].min()
#         edge_index[1] -= cidx  # make sure we do not waste memory
#         reversed_edge_index = torch.stack(
#             [edge_index[1], edge_index[0]], dim=0)
#         if self.GPR:
#             xs = []
#             xs.append(F.relu(self.MLP(x)))
#             for i, _ in enumerate(self.V2EConvs):
#                 x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
# #                 x = self.bnV2Es[i](x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#                 x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
#                 x = F.relu(x)
#                 xs.append(x)
# #                 x = self.bnE2Vs[i](x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#             x = torch.stack(xs, dim=-1)
#             x = self.GPRweights(x).squeeze()
#             x = self.classifier(x)
#         else:
#             x = F.dropout(x, p=0.2, training=self.training) # Input dropout
#             for i, _ in enumerate(self.V2EConvs):
#                 assert self.aggr is not None
#                 x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
# #                 x = self.bnV2Es[i](x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#                 x = F.relu(self.E2VConvs[i](
#                     x, reversed_edge_index, norm, self.aggr))
# #                 x = self.bnE2Vs[i](x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#             x = self.classifier(x)

#         return x

class SetGNN(nn.Module):
    def __init__(self, args, norm_size=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
#         V_in_dim = V_dict['in_dim']
#         V_enc_hid_dim = V_dict['enc_hid_dim']
#         V_dec_hid_dim = V_dict['dec_hid_dim']
#         V_out_dim = V_dict['out_dim']
#         V_enc_num_layers = V_dict['enc_num_layers']
#         V_dec_num_layers = V_dict['dec_num_layers']


#         E_in_dim = E_dict['in_dim']
#         E_enc_hid_dim = E_dict['enc_hid_dim']
#         E_dec_hid_dim = E_dict['dec_hid_dim']
#         E_out_dim = E_dict['out_dim']
#         E_enc_num_layers = E_dict['enc_num_layers']
#         E_dec_num_layers = E_dict['dec_num_layers']


        self.args = args 
        if args.mode == 'learnmask':
            self.mask_module = LearnableEdgeMask(args.num_hyperedges,args=args)
        if args.mode == 'learnmask+':
            self.mask_module = LearnableEdgeMaskplus(args.F,hidden_dim=args.coarse_MLP, args=args)
        self.ratio = 1.0
#         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()


        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm_size))


        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden)) 
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            if self.GPR:
                self.MLP = MLP(in_channels=args.num_features,
                               hidden_channels=args.MLP_hidden,
                               out_channels=args.MLP_hidden,
                               num_layers=args.MLP_num_layers,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = Linear(self.All_num_layers+1, 1, bias=False)
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)
            else:
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)




#         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
#         However, in general this can be arbitrary.




    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)


    def forward(self, data, is_test=False):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
#             The data should contain the follows
#             data.x: node features
#             data.V2Eedge_index:  edge list (of size (2,|E|)) where
#             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges
        if self.args.mode == 'learnmask':
            # edge_index_knn,probs = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio)
            edge_index_knn,probs,keepmask = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio,is_test=is_test,return_mask = True)
        if self.args.mode == 'learnmask+':
            # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
            edge_index_knn, probs,keepmask = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
        if self.args.mode.startswith('learnmask'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
        else:
            edge_index_knn = data.edge_index
        # 
        x, edge_index, norm = data.x, edge_index_knn, data.norm
        if self.args.mode.startswith('learnmask'):
            norm = norm[keepmask]
        # Adding self-loop
        if self.args.mode.startswith('learnmask'):
            num_nodes = data.n_x
            num_hyperedges = data.num_hyperedges


            nodes, hyperedges = edge_index[0], edge_index[1]


            # Step 1: Identify hyperedges that are already self-loops (appear only once)
            _, counts = torch.unique(hyperedges, return_counts=True)
            singleton_mask = counts == 1
            singleton_hyperedges = torch.unique(hyperedges)[singleton_mask]


            # Step 2: Identify the nodes that already have self-loops
            # Create a mask for entries with singleton hyperedges
            self_loop_mask = torch.isin(hyperedges, singleton_hyperedges)
            existing_self_loop_nodes = nodes[self_loop_mask].unique()


            # Step 3: Find nodes that need self-loops
            all_nodes = torch.arange(num_nodes, dtype=torch.long).to(edge_index.device)
            needs_self_loop_mask = ~torch.isin(all_nodes, existing_self_loop_nodes)
            nodes_to_add = all_nodes[needs_self_loop_mask]


            # Step 4: Construct new self-loop hyperedges
            num_new_edges = nodes_to_add.size(0)
            if num_new_edges > 0:
                start_hyperedge_id = hyperedges.max().item() + 1
                new_hyperedge_ids = torch.arange(start_hyperedge_id,
                                                 start_hyperedge_id + num_new_edges,
                                                 dtype=torch.long).to(edge_index.device)


                new_edges = torch.stack([nodes_to_add, new_hyperedge_ids], dim=0)
                updated_edge_index = torch.cat([edge_index, new_edges], dim=1)
            else:
                updated_edge_index = edge_index


            # Step 5: Sort by node index for consistency
            sorted_idx = torch.argsort(updated_edge_index[0])
            edge_index = updated_edge_index[:, sorted_idx].long()


        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                assert self.aggr is not None
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)
        if self.args.mode.startswith('learnmask') and is_test == False:
            return x, probs
        return x



class MLP_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, args, InputNorm=False):
        super(MLP_model, self).__init__()
        in_channels = args.num_features
        hidden_channels = args.MLP_hidden
        out_channels = args.num_classes
        num_layers = args.All_num_layers
        dropout = args.dropout
        Normalization = args.normalization

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


"""
The code below is directly adapt from the official implementation of UniGNN.
"""
# NOTE: can not tell which implementation is better statistically 

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X



# v1: X -> XW -> AXW -> norm
class UniSAGEConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce=self.args.second_aggregate, dim_size=N) # [N, C]
        X = X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.]))
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


    def forward(self, X, vertex, edges):
        N = X.shape[0]
        # X0 = X # NOTE: reserved for skip connection
        
        # v1: X -> XW -> AXW -> norm
        X = self.W(X) 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X = (1 + self.eps) * X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        
        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGCNConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV
        
        # v1: X -> XW -> AXW -> norm
        
        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 
        
        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: skip concat here?

        return X



# v2: X -> AX -> norm -> AXW 
class UniGCNConv2(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v3: X -> AX -> norm -> AXW 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        X = self.W(X)


        # NOTE: result might be slighly unstable
        # NOTE: skip concat here?

        return X



class UniGATConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex] # [nnz, H, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, H, C]


        alpha_e = (Xe * self.att_e).sum(-1) # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop( alpha )
        alpha = alpha.unsqueeze(-1)


        Xev = Xe[edges] # [nnz, H, C]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, H, C]
        X = Xv 
        X = X.view(N, H * C)

        if self.args.use_norm:
            X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0 

        # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return X




__all_convs__ = {
    'UniGAT': UniGATConv,
    'UniGCN': UniGCNConv,
    'UniGCN2': UniGCNConv2,
    'UniGIN': UniGINConv,
    'UniSAGE': UniSAGEConv,
}



class UniGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )
        self.V = V 
        self.E = E 
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        V, E = self.V, self.E 
        
        X = self.input_drop(X)
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X = self.conv_out(X, V, E)      
        return F.log_softmax(X, dim=1)



class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0, test_flag):
        N = X.shape[0]
        if(test_flag):
            degE = self.args.UniGNN_degE_p
            degV = self.args.UniGNN_degV_p
        else:
            degE = self.args.UniGNN_degE
            degV = self.args.UniGNN_degV
        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='mean') # [E, C], reduce is 'mean' here as default
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV
        
        X = Xv 

        if self.args.UniGNN_use_norm:
            X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)


        return X



class UniGCNII(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E, V_p, E_p):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V 
        self.E = E 
        self.V_p = V_p 
        self.E_p = E_p 
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu
        self.input_drop = nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data_input):
        data, test_flag = data_input
        x = data.x
        if(test_flag):
            V, E = self.V_p, self.E_p
        else:
            V, E = self.V, self.E
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0, test_flag))
        x = self.dropout(x)
        x = self.convs[-1](x)
        return x
