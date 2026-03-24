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
from torch_geometric.data import Data
from preprocessing import Add_Self_Loops

def pooled_edge_embeddings(X, V_idx, E_idx, num_edges, reduce='mean'):
    # X: [N, F], V_idx/E_idx: [nnz], num_edges: int (m)
    # returns [m, F]
    if reduce == 'mean':
        return scatter(X[V_idx], E_idx, dim=0, dim_size=num_edges, reduce='mean')
    elif reduce == 'sum':
        return scatter(X[V_idx], E_idx, dim=0, dim_size=num_edges, reduce='sum')
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")


def _hyperufg_to_int(value):
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _hyperufg_prepare_structure(edge_index, num_nodes, dtype, device, eps=1e-12):
    if edge_index.numel() == 0:
        node_idx = torch.empty(0, dtype=torch.long, device=device)
        edge_idx = torch.empty(0, dtype=torch.long, device=device)
        deg_v_inv_sqrt = torch.ones(num_nodes, dtype=dtype, device=device)
        edge_scale = torch.empty(0, dtype=dtype, device=device)
        return node_idx, edge_idx, 0, deg_v_inv_sqrt, edge_scale

    node_idx = edge_index[0].long()
    edge_idx = edge_index[1].long()
    edge_idx = edge_idx - edge_idx.min()
    num_edges = int(edge_idx.max().item()) + 1

    ones = torch.ones(edge_idx.size(0), dtype=dtype, device=device)
    deg_e = scatter(ones, edge_idx, dim=0, dim_size=num_edges, reduce='sum').clamp_min(eps)
    deg_v = scatter(ones, node_idx, dim=0, dim_size=num_nodes, reduce='sum').clamp_min(eps)

    deg_v_inv_sqrt = deg_v.pow(-0.5)
    edge_scale = deg_e.pow(-1.0)
    return node_idx, edge_idx, num_edges, deg_v_inv_sqrt, edge_scale


def _hyperufg_apply_propagation(x, structure):
    node_idx, edge_idx, num_edges, deg_v_inv_sqrt, edge_scale = structure
    if edge_idx.numel() == 0:
        return torch.zeros_like(x)

    x_scaled = deg_v_inv_sqrt.unsqueeze(-1) * x
    edge_messages = scatter(
        x_scaled[node_idx],
        edge_idx,
        dim=0,
        dim_size=num_edges,
        reduce='sum',
    )
    edge_messages = edge_messages * edge_scale.unsqueeze(-1)
    node_messages = scatter(
        edge_messages[edge_idx],
        node_idx,
        dim=0,
        dim_size=x.size(0),
        reduce='sum',
    )
    return deg_v_inv_sqrt.unsqueeze(-1) * node_messages


def _hyperufg_chebyshev_coefficients(filter_fn, order, dtype=torch.float32):
    num_samples = max(64, 4 * (order + 1))
    sample_ids = torch.arange(num_samples, dtype=dtype)
    theta = math.pi * (sample_ids + 0.5) / num_samples
    x = torch.cos(theta)
    lam = x + 1.0
    values = filter_fn(lam)

    coeffs = []
    coeffs.append(values.mean())
    for k in range(1, order + 1):
        coeff_k = (2.0 / num_samples) * torch.sum(values * torch.cos(k * theta))
        coeffs.append(coeff_k)
    return torch.stack(coeffs)


def _hyperufg_apply_chebyshev_filter(x, coeffs, structure):
    if coeffs.numel() == 0:
        return torch.zeros_like(x)

    out = coeffs[0] * x
    if coeffs.size(0) == 1:
        return out

    def apply_rescaled_laplacian(tensor):
        # For Zhou's normalized Laplacian L = I - G, with spectrum in [0, 2],
        # the Chebyshev-rescaled operator is L - I = -G.
        return -_hyperufg_apply_propagation(tensor, structure)

    t_prev = x
    t_curr = apply_rescaled_laplacian(x)
    out = out + coeffs[1] * t_curr

    for k in range(2, coeffs.size(0)):
        t_next = 2.0 * apply_rescaled_laplacian(t_curr) - t_prev
        out = out + coeffs[k] * t_next
        t_prev, t_curr = t_curr, t_next

    return out


class HyperUFGLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes, alpha, beta, activation=F.relu, bias=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.activation = activation

        self.theta_node = nn.ParameterList([
            nn.Parameter(torch.ones(num_nodes)),
            nn.Parameter(torch.ones(num_nodes)),
            nn.Parameter(torch.ones(num_nodes)),
        ])
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, x0, filter_coeffs, structure):
        agg = 0.0
        for coeffs, theta in zip(filter_coeffs, self.theta_node):
            response = _hyperufg_apply_chebyshev_filter(x, coeffs, structure)
            response = theta.unsqueeze(-1) * response
            agg = agg + _hyperufg_apply_chebyshev_filter(response, coeffs, structure)

        support = (1.0 - self.alpha) * agg + self.alpha * x0
        out = (1.0 - self.beta) * support + self.beta * self.linear(support)

        if self.activation is not None:
            out = self.activation(out)
        return out


class HyperUFG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.alpha = args.hyperufg_alpha
        self.lambda_gcnii = args.hyperufg_lambda
        self.cheb_order = args.hyperufg_cheb_order
        self.ratio = 1.0
        if args.mode == 'learnmask':
            self.mask_module = LearnableEdgeMask(args.num_incidences, args=args)
        self.input_proj = nn.Linear(args.num_features, args.MLP_hidden)

        num_nodes = _hyperufg_to_int(args.n_x)
        layers = []
        for layer_idx in range(1, args.All_num_layers + 1):
            beta_l = math.log(self.lambda_gcnii / layer_idx + 1.0)
            activation = F.relu if layer_idx < args.All_num_layers else None
            layers.append(
                HyperUFGLayer(
                    in_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_nodes=num_nodes,
                    alpha=self.alpha,
                    beta=beta_l,
                    activation=activation,
                    bias=True,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(args.MLP_hidden, args.num_classes)
        self.register_buffer(
            'low_pass_coeffs',
            _hyperufg_chebyshev_coefficients(
                lambda lam: torch.cos(lam / 8.0) * torch.cos(lam / 16.0),
                self.cheb_order,
            ),
        )
        self.register_buffer(
            'high_pass_1_coeffs',
            _hyperufg_chebyshev_coefficients(
                lambda lam: torch.sin(lam / 8.0) * torch.cos(lam / 16.0),
                self.cheb_order,
            ),
        )
        self.register_buffer(
            'high_pass_2_coeffs',
            _hyperufg_chebyshev_coefficients(
                lambda lam: torch.sin(lam / 16.0),
                self.cheb_order,
            ),
        )

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.classifier.reset_parameters()
        if hasattr(self, 'mask_module'):
            if hasattr(self.mask_module, 'logits'):
                nn.init.normal_(self.mask_module.logits)
        for layer in self.layers:
            for theta in layer.theta_node:
                nn.init.ones_(theta)
            layer.linear.reset_parameters()

    def _forward_logits(self, data, edge_index):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.relu(self.input_proj(x))
        x0 = x
        structure = _hyperufg_prepare_structure(
            edge_index,
            num_nodes=data.x.size(0),
            dtype=data.x.dtype,
            device=data.x.device,
        )
        filter_coeffs = (
            self.low_pass_coeffs.to(device=x.device, dtype=x.dtype),
            self.high_pass_1_coeffs.to(device=x.device, dtype=x.dtype),
            self.high_pass_2_coeffs.to(device=x.device, dtype=x.dtype),
        )

        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, x0, filter_coeffs, structure)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def forward(self, data, epoch=None, return_mask=True, is_test=False):
        if self.args.mode == 'learnmask':
            edge_index_knn, _, keepmask, probs = self.mask_module(
                data,
                keep_ratio=self.args.keep_ratio,
                is_test=is_test,
                return_mask=True,
            )
            keepmask = keepmask.bool()
            self.ratio = log_sparsification_info(
                data.edge_index,
                edge_index_knn,
                label=None,
                verbose=False,
            )
            logits = self._forward_logits(data, edge_index_knn)
            if not is_test:
                if return_mask:
                    return logits, probs, keepmask
                return logits, probs
            return logits

        self.ratio = 1.0
        return self._forward_logits(data, data.edge_index)


class EHGNN_CcondLR(nn.Module):
    """
    EHGNN-C (Cond, LR): Low-rank feature-conditioned hyperedge scoring.
    - Pool node features per hyperedge -> x_e
    - Project x_e and/or learn low-rank edge factors
    - Score each edge and top-k over edges.
    """

    def __init__(self, args, rank=16, agg='mean'):
        super().__init__()
        in_dim = args.num_features
        num_edges = int(args.num_hyperedges)
        self.args = args
        self.rank = rank
        self.agg = agg
        self.reg = args.reg

        # Project hyperedge features -> rank-r
        self.node_proj = nn.Linear(in_dim, rank, bias=False)
        # Optional additional low-rank free factors
        self.edge_factors = nn.Parameter(torch.randn(num_edges, rank))


    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask=False):
        V_idx, E_idx = data.edge_index # [nnz], [nnz]
        X = data.x # [n, F]
        num_edges = int(data.num_hyperedges)

        # 1) project node features: W_x x_v
        Z = self.node_proj(X)                   # [n, r]
        
        # 2) aggregate per hyperedge: u_e = phi({W_x x_v | v in e})
        if self.agg == 'mean':
            u_e = pooled_edge_embeddings(Z, V_idx, E_idx, num_edges, reduce='mean')
        else:
            raise ValueError("Only 'mean' aggregation is supported for now.")

        # 3) bilinear low-rank score: l_e = <u_e, w_e>,  p_e = sigmoid(l_e)
        w_e = self.edge_factors                # [m, r]
        logits = (u_e * w_e).sum(dim=-1)       # [m], l_e
        probs = torch.sigmoid(logits)          # [m], p_e
        # print(logits.shape, data.edge_index.shape)
        m = num_edges
        k = max(1, int(keep_ratio * m))

        if is_test:
            keep_e = torch.topk(probs, k=k, largest=True).indices
        else:
            if self.reg == 'none':
                if self.args.sampling == 'multinomial':
                    keep_e = torch.multinomial(probs, k, replacement=False)
                else:
                    keep_e = torch.topk(probs, k).indices
            elif self.reg == 'l2':
                keep_mask = torch.bernoulli(probs).bool()
                if not keep_mask.any():
                    keep_mask[probs.argmax()] = True
                keep_e = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f'Unknown regularizer type: {self.reg}')

        # Hard edge mask
        hard_e = torch.zeros_like(probs)
        hard_e[keep_e] = 1.0

        # STE
        soft_mask = hard_e + (probs - probs.detach()) * hard_e  # [m]
        soft_vals = soft_mask[E_idx]
        # --- Lift to incidences, prune structure ---
        inc_keep_mask = hard_e[E_idx] > 0          # which incidences survive
        V_pruned = V_idx[inc_keep_mask]
        E_pruned = E_idx[inc_keep_mask]

        # incidence-aligned weights: soft value of the corresponding edge
        # weights = soft_mask        
        soft_pruned = soft_vals[inc_keep_mask]   
        # ---- Hyperedge weights ----
        weights = scatter(
            soft_pruned, E_pruned, dim=0,
            dim_size=m, reduce="mean"
        )
        edge_index_pruned = torch.stack([V_pruned, E_pruned], dim=0)

        if return_mask:
            return edge_index_pruned, weights, hard_e, probs

        return edge_index_pruned, weights, probs

# EHGNN-F(cond,LR)
class EHGNN_FcondLR(nn.Module):
    def __init__(self, args, rank=16, top_k=5):
        super().__init__()
        self.rank = rank
        self.reg = args.reg
        self.top_k = top_k
        self.args = args

        self.node_proj = nn.Linear(args.num_features, rank, bias=False)
        self.edge_factors = nn.Parameter(torch.randn(int(args.num_hyperedges), rank))

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask = False):
        V_idx, E_idx = data.edge_index
        X = data.x
        
        # node_emb = self.node_proj(X)[V_idx] # nnz, r
        # edge_emb = self.edge_factors

        logits = (self.node_proj(X)[V_idx] * self.edge_factors[E_idx]).sum(dim=-1)
        # print(logits.shape, data.edge_index.shape)
        scores = torch.sigmoid(logits)
        nnz = len(scores)

        k = max(1, int(keep_ratio * nnz))

        if is_test:
            keep_ids = torch.topk(scores, k).indices
        else:
            if self.reg=="none":
                if self.args.sampling == 'multinomial':
                    keep_ids = torch.multinomial(scores, k, replacement=False)
                else:
                    keep_ids = torch.topk(scores, k).indices
            elif self.reg == 'l2':
                keep_mask = torch.bernoulli(scores).bool()
                if not keep_mask.any():
                    keep_mask[scores.argmax()] = True
                keep_ids = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f'Unknown regularizer type: {self.reg}')

        hard = torch.zeros_like(scores)
        hard[keep_ids] = 1.0
        soft = hard + (scores - scores.detach()) * hard

        V_pruned = V_idx[keep_ids]
        E_pruned = E_idx[keep_ids]
        soft_vals = soft[keep_ids]

        # hyperedge_weight = scatter(soft_vals, E_pruned, 0,
        #                            dim_size=int(data.num_hyperedges),
        #                            reduce='mean')
        hyperedge_weight = soft_vals
        edge_index_pruned = torch.stack([V_pruned, E_pruned])
        if return_mask:
            return edge_index_pruned, hyperedge_weight, hard, scores
        return edge_index_pruned, hyperedge_weight, scores


# EHGNN-F (cond)
class FeatureConditionedIncidenceMask(nn.Module):
    def __init__(self, F, hidden_dim=32, agg='mean', args=None):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2*F, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.reg = args.reg
        self.args = args
        self.agg = agg

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask = False):
        V_idx, E_idx = data.edge_index
        X = data.x
        m = data.num_hyperedges

        # pooled edge feats
        X_e = scatter(X[V_idx], E_idx, dim=0,
                      dim_size=m, reduce="mean")

        # incidence features
        pair = torch.cat([X[V_idx], X_e[E_idx]], dim=1)

        logits = self.scorer(pair).squeeze(-1)
        probs = torch.sigmoid(logits)
        nnz = len(V_idx)

        k = max(1, int(nnz * keep_ratio))

        # ---- Hard incidence selection ----
        if is_test:
            keep_ids = torch.topk(probs, k).indices
        else:
            if self.reg == "none":
                if self.args.sampling == 'multinomial':
                    keep_ids = torch.multinomial(probs, k, replacement=False)
                else:
                    keep_ids = torch.topk(probs, k).indices
            elif self.reg == 'l2':
                keep_mask = torch.bernoulli(probs).bool()
                if not keep_mask.any():
                    keep_mask[probs.argmax()] = True
                keep_ids = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f'Unknown regularizer type: {self.reg}')

        hard = torch.zeros_like(probs)
        hard[keep_ids] = 1.0
        soft = hard + (probs - probs.detach()) * hard

        V_pruned = V_idx[keep_ids]
        E_pruned = E_idx[keep_ids]
        soft_vals = soft[keep_ids]

        # per-hyperedge weight
        # hyperedge_weight = scatter(
        #     soft_vals, E_pruned, dim=0, dim_size=m,
        #     reduce="mean"
        # )
        hyperedge_weight = soft_vals

        edge_index_pruned = torch.stack([V_pruned, E_pruned])
        if return_mask:
            return edge_index_pruned, hyperedge_weight, hard, probs
        return edge_index_pruned, hyperedge_weight, probs

# EGHNN-C (cond)
class LearnableEdgeMaskplus(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, agg='mean', args=None):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.agg = agg
        self.reg = args.reg
        self.args = args

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask = False):
        V_idx, E_idx = data.edge_index
        X = data.x
        num_edges = data.num_hyperedges

        # ---- pooled hyperedge features ----
        edge_feats = scatter(
            X[V_idx], E_idx, dim=0,
            dim_size=num_edges, reduce='mean'
        )

        logits = self.scorer(edge_feats).squeeze(-1)
        probs = torch.sigmoid(logits)
        m = num_edges

        k = max(1, int(keep_ratio * m))

        # ---- Hard selection ----
        if is_test:
            keep_e = torch.topk(probs, k).indices
        else:
            if self.reg == "none":
                if self.args.sampling == 'multinomial':
                    keep_e = torch.multinomial(probs, k, replacement=False)
                else:
                    keep_e = torch.topk(probs, k).indices
            elif self.reg == 'l2':
                keep_mask = torch.bernoulli(probs).bool()
                if not keep_mask.any():
                    keep_mask[probs.argmax()] = True
                keep_e = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f'Unknown regularizer type: {self.reg}')

        # ---- Hard edge mask ----
        hard_e = torch.zeros_like(probs)
        hard_e[keep_e] = 1.0

        # ---- STE soft mask over edges ----
        soft_e = hard_e + (probs - probs.detach()) * hard_e

        # ---- Lift to incidences ----
        soft_vals = soft_e[E_idx]          # per-incidence weight
        hard_mask = (soft_vals > 0)

        V_pruned = V_idx[hard_mask]
        E_pruned = E_idx[hard_mask]
        soft_pruned = soft_vals[hard_mask]

        # ---- Per-hyperedge final weight ----
        hyperedge_weight = scatter(
            soft_pruned, E_pruned, dim=0,
            dim_size=num_edges, reduce='mean'
        )

        edge_index_pruned = torch.stack([V_pruned, E_pruned])
        if return_mask:
            return edge_index_pruned, hyperedge_weight, hard_e, probs
        return edge_index_pruned, hyperedge_weight, probs


# ---- EHGNN-F => Learnable Mask Baseline Sparsifier with straight-through estimator ----
# --- soft mask is returned to be used as weights during hypergraph conv.
class LearnableEdgeMask(nn.Module): 
    def __init__(self, num_incidences, args):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_incidences), requires_grad=True)
        self.reg = args.reg
        self.args = args 
        
    def inference_soft(H, p_matrix):
        return H * p_matrix  # elementwise expected sparsified structure

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask=False):
        V_idx, E_idx = data.edge_index
        probs = torch.sigmoid(self.logits)        # [num_incidences]
        scores = probs                 # [nnz]

        topk = int(keep_ratio * len(V_idx))

        if is_test:
            keep_ids = torch.topk(scores, k=topk).indices
            # keep_ids = torch.multinomial(scores, topk, replacement=False)
        else:
            if self.reg == 'none':
                if self.args.sampling == 'multinomial':
                    keep_ids = torch.multinomial(scores, topk, replacement=False)
                else:
                    # keep_ids = gumbel_topk(scores, k=topk)
                    keep_ids = torch.topk(scores, k=topk).indices
            elif self.reg == 'l2':
                keep_mask = torch.bernoulli(scores).bool()
                if not keep_mask.any():
                    keep_mask[scores.argmax()] = True
                keep_ids = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f'Unknown regularizer type: {self.reg}')

        
        hard_mask = torch.zeros_like(scores)
        hard_mask[keep_ids] = 1.0

        # --- STE: gradient only through kept edges ---

        soft_mask = hard_mask + (scores - scores.detach()) * hard_mask

        V_pruned = V_idx[keep_ids]
        E_pruned = E_idx[keep_ids]
        soft_vals = soft_mask[keep_ids]            # [#kept incidences]

        # ---- Aggregate to per-hyperedge weights ----
        # num_hyperedges = data.num_hyperedges
        # hyperedge_weight = scatter(
        #     soft_vals,
        #     E_pruned,
        #     dim=0,
        #     dim_size=num_hyperedges,
        #     reduce='mean'
        # )  # shape [num_hyperedges]
        hyperedge_weight = soft_vals
        edge_index_pruned = torch.stack([V_pruned, E_pruned], dim=0)
        if return_mask:
            return edge_index_pruned, hyperedge_weight, hard_mask, probs
        return edge_index_pruned, hyperedge_weight, probs


# EHGNN-C
class FeatureAgnosticEdgeMask_nob(nn.Module):
    def __init__(self, num_edges, args):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_edges), requires_grad=True)
        self.reg = args.reg
        self.args = args

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask = False):
        V_idx, E_idx = data.edge_index
        m = self.logits.numel()

        probs = torch.sigmoid(self.logits)
        k = max(1, int(keep_ratio * m))

        # ---- Hard edge selection ----
        if is_test:
            keep_e = torch.topk(probs, k).indices
        else:
            if self.reg == 'none':
                if self.args.sampling == 'multinomial':
                    keep_e = torch.multinomial(probs, k, replacement=False)
                else:
                    keep_e = torch.topk(probs, k).indices
            elif self.reg == 'l2':
                keep_mask = torch.bernoulli(probs).bool()
                if not keep_mask.any():
                    keep_mask[probs.argmax()] = True
                keep_e = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f'Unknown regularizer type: {self.reg}')

        # ---- STE edge mask ----
        hard_e = torch.zeros_like(probs)
        hard_e[keep_e] = 1.0
        soft_e = hard_e + (probs - probs.detach()) * hard_e

        # ---- Lift to incidences ----
        soft_vals = soft_e[E_idx]
        hard_mask = soft_vals > 0

        V_pruned = V_idx[hard_mask]
        E_pruned = E_idx[hard_mask]
        soft_pruned = soft_vals[hard_mask]

        # ---- Hyperedge weights ----
        hyperedge_weight = scatter(
            soft_pruned, E_pruned, dim=0,
            dim_size=m, reduce="mean"
        )
        edge_index_pruned = torch.stack([V_pruned, E_pruned])
        if return_mask:
            return edge_index_pruned, hyperedge_weight, hard_e, probs
        
        return edge_index_pruned, hyperedge_weight, probs


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


def gumbel_topk(logits, k, tau=1.0):
    g = -torch.empty_like(logits).exponential_().log()
    y = (logits + g) / tau
    return y.topk(k).indices


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

# ------------------------------
# Conjugate Gradient Solver
# ------------------------------
def cg_solve(matvec, b, x0=None, tol=1e-5, max_iter=200):
    """
    Solve A x = b for SPD A given implicitly by matvec(x) = A x.
    Returns approximate solution x.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    return x

# ------------------------------
# Approximate diag(L^+) via Hutchinson
# ------------------------------
def approximate_L_inv_diag_hutchinson(H, Dv_diag, De_inv_diag,
                                      num_probes=16,
                                      cg_tol=1e-5,
                                      cg_max_iter=200,
                                      reg=1e-3):
    """
    Approximate diag(L^+) where
        L = Dv - H De^{-1} H^T   (Zhou Laplacian for hypergraphs),
    using Hutchinson estimator + CG to solve (L + reg I) z = g.
    H: [n, m] incidence (0/1)
    Dv_diag: [n] node degrees
    De_inv_diag: [m] 1 / hyperedge degrees
    """
    device = H.device
    n, m = H.shape

    # Precompute diagonal of L for cheap Jacobi-like preconditioning if desired
    # (optional; here we just keep it for possible extensions).
    # L_diag = Dv_diag - (H * De_inv_diag) @ H.T ... but we don't strictly need it

    De_inv = De_inv_diag  # [m]

    def matvec_L(x):
        # x: [n]
        # compute L x = Dv x - H De^{-1} H^T x
        # 1: H^T x  -> [m]
        Ht_x = H.T @ x                      # [m]
        # 2: De_inv * (H^T x)  -> [m]
        tmp = De_inv * Ht_x
        # 3: H @ tmp  -> [n]
        H_tmp = H @ tmp
        # final: Dv x - H @ tmp (+ reg * x)
        return Dv_diag * x - H_tmp + reg * x

    diag_est = torch.zeros(n, device=device)

    for _ in range(num_probes):
        # Rademacher random vector: entries in {-1, +1}
        g = torch.empty(n, device=device).bernoulli_(0.5).mul_(2).sub_(1)
        # Solve (L + reg I) z = g
        z = cg_solve(matvec_L, g, tol=cg_tol, max_iter=cg_max_iter)
        # Hutchinson contribution to diag(L^+): elementwise product
        diag_est += z * g

    diag_est = diag_est / float(num_probes)
    return diag_est  # approx diag(L^+)

# ------------------------------
# Approximate Effective Resistance-Based Sparsifier
# ------------------------------
def effective_resistance_sparsify_approx(edge_index,
                                         num_nodes,
                                         num_edges,
                                         keep_ratio=0.5,
                                         num_probes=16,
                                         cg_tol=1e-5,
                                         cg_max_iter=200,
                                         reg=1e-3):
    """
    Approximate spectral sparsifier using effective-resistance-like scores.
    Uses Hutchinson + CG instead of explicit pseudoinverse.

    edge_index: [2, nnz], where
        edge_index[0] = V_idx (node indices)
        edge_index[1] = E_idx (hyperedge indices)
    """
    V_idx, E_idx = edge_index
    device = edge_index.device

    # num_edges = int(E_idx.max().item()) + 1

    # Build (dense) incidence H ∈ R^{n×m}
    # If needed, switch to sparse here.
    H = torch.zeros((num_nodes, num_edges), device=device)
    H[V_idx, E_idx] = 1.0

    # Node and edge degrees
    Dv_diag = H.sum(dim=1)                 # [n]
    De_diag = H.sum(dim=0)                 # [m]
    De_inv_diag = 1.0 / (De_diag + 1e-8)   # [m]

    # Approximate diag(L^+)
    L_inv_diag_approx = approximate_L_inv_diag_hutchinson(
        H, Dv_diag, De_inv_diag,
        num_probes=num_probes,
        cg_tol=cg_tol,
        cg_max_iter=cg_max_iter,
        reg=reg
    )  # [n]

    # Hyperedge scores: sum of diag(L^+) over nodes in each hyperedge
    edge_scores = torch.zeros(num_edges, device=device)
    for e_id in range(num_edges):
        nodes_in_e = V_idx[E_idx == e_id]
        if nodes_in_e.numel() > 1:
            edge_scores[e_id] = L_inv_diag_approx[nodes_in_e].sum()

    # Keep top-k edges
    topk = int(keep_ratio * num_edges)
    topk = max(topk, 1)  # at least one edge
    _, keep_ids = torch.topk(edge_scores, k=topk)
    mask = torch.isin(E_idx, keep_ids)
    V_pruned = V_idx[mask]
    E_pruned = E_idx[mask]
    return torch.stack([V_pruned, E_pruned], dim=0)

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

def make_L_matvec(V_idx, E_idx, num_nodes, num_edges, Dv_diag, De_inv_diag, reg=1e-3):
    """
    Returns a function matvec(x) that computes (L + reg*I) x
    using only edge_index and degree vectors.
    """
    def matvec(x):
        # x: [num_nodes]

        # ---- H^T x: sum over nodes in each hyperedge ----
        # H^T x (e) = sum_{v in e} x_v
        Ht_x = torch.zeros(num_edges, device=x.device)
        Ht_x.index_add_(0, E_idx, x[V_idx])   # scatter: e ← e + x[v]

        # ---- De^{-1} H^T x ----
        tmp = De_inv_diag * Ht_x             # [num_edges]

        # ---- H tmp: sum over hyperedges incident to each node ----
        # (H tmp)(v) = sum_{e: v in e} tmp[e]
        H_tmp = torch.zeros(num_nodes, device=x.device)
        H_tmp.index_add_(0, V_idx, tmp[E_idx])  # scatter: v ← v + tmp[e]

        # ---- L x = Dv x - H De^{-1} H^T x + reg * x ----
        return Dv_diag * x - H_tmp + reg * x

    return matvec

def approximate_L_inv_diag_hutchinson_edgeindex(
    V_idx, E_idx, num_nodes, num_edges,
    num_probes=16, cg_tol=1e-5, cg_max_iter=200, reg=1e-3
):
    device = V_idx.device

    # Node & edge degrees from incidence
    Dv_diag = torch.bincount(V_idx, minlength=num_nodes).float().to(device)  # [n]
    De_diag = torch.bincount(E_idx, minlength=num_edges).float().to(device)  # [m]
    De_inv_diag = 1.0 / (De_diag + 1e-8)                                     # [m]

    # Matvec for L + reg I using only edge_index
    matvec_L = make_L_matvec(
        V_idx, E_idx, num_nodes, num_edges, Dv_diag, De_inv_diag, reg=reg
    )

    diag_est = torch.zeros(num_nodes, device=device)

    for _ in range(num_probes):
        # Rademacher random vector {-1, +1}
        g = torch.empty(num_nodes, device=device).bernoulli_(0.5).mul_(2).sub_(1)
        z = cg_solve(matvec_L, g, tol=cg_tol, max_iter=cg_max_iter)
        diag_est += z * g   # Hutchinson contribution

    diag_est = diag_est / float(num_probes)
    return diag_est, Dv_diag, De_inv_diag

def effective_resistance_sparsify_approx_edgeindex(
    edge_index,
    num_nodes,
    num_edges,
    keep_ratio=0.5,
    num_probes=16,
    cg_tol=1e-5,
    cg_max_iter=200,
    reg=1e-3,
):
    V_idx, E_idx = edge_index
    device = edge_index.device

    # num_edges = int(E_idx.max().item()) + 1

    # Approximate diag(L^+) using only edge_index
    L_inv_diag_approx, Dv_diag, De_inv_diag = approximate_L_inv_diag_hutchinson_edgeindex(
        V_idx, E_idx, num_nodes, num_edges,
        num_probes=num_probes,
        cg_tol=cg_tol,
        cg_max_iter=cg_max_iter,
        reg=reg
    )  # [num_nodes]

    # Hyperedge scores: sum of node diagonals in each hyperedge
    edge_scores = torch.zeros(num_edges, device=device)
    for e_id in range(num_edges):
        nodes_in_e = V_idx[E_idx == e_id]
        if nodes_in_e.numel() > 1:
            edge_scores[e_id] = L_inv_diag_approx[nodes_in_e].sum()

    topk = max(int(keep_ratio * num_edges), 1)
    _, keep_ids = torch.topk(edge_scores, k=topk)
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

# def degree_distribution_sparsify(edge_index, keep_ratio=0.5):
#     V_idx, E_idx = edge_index
#     nnz = V_idx.numel()  # number of incidences

#     # degree of each node
#     node_deg = torch.bincount(V_idx, minlength=V_idx.max().item() + 1).float()
#     probs = node_deg[V_idx]                    # probability per-incidence
#     probs = probs / probs.sum()                # normalize

#     # number of incidences to keep
#     k = int(keep_ratio * nnz)

#     # sample incidences directly
#     keep_ids = torch.multinomial(probs, num_samples=k, replacement=False)

#     # build new edge_index
#     V_new = V_idx[keep_ids]
#     E_new = E_idx[keep_ids]
#     return torch.stack([V_new, E_new], dim=0)

# Edge cardinality distribution sparsifier: sample hyperedges with probability proportional to their degree (number of incident nodes) 
def degree_distribution_sparsify(edge_index,keep_ratio=0.5):
    V_idx, E_idx = edge_index
    num_hyperedges = int(E_idx.max().item() + 1)
    edge_deg = torch.bincount(E_idx, minlength=num_hyperedges).float()
    probs = edge_deg / edge_deg.sum()
    num_samples = min(num_hyperedges, int(keep_ratio * num_hyperedges))
    sampled_edges = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    keep_mask = torch.isin(E_idx, sampled_edges)
    return edge_index[:, keep_mask]

def min_degree_sparsify(edge_index, num_hyperedges, keep_ratio=0.5):
    V_idx, E_idx = edge_index
    num_hyperedges = int(num_hyperedges)
    
    node_deg = torch.bincount(V_idx, minlength=V_idx.max().item() + 1).float()
    incidence_degrees = node_deg[V_idx]
    min_degrees = scatter(
        incidence_degrees, E_idx, dim=0, dim_size=num_hyperedges, reduce='min'
    )

    num_samples = min(num_hyperedges, int(keep_ratio * num_hyperedges))
    if num_samples <= 0:
        return edge_index[:, :0]

    probs = min_degrees / min_degrees.sum()
    sampled_hyperedges = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    keep_mask = torch.isin(E_idx, sampled_hyperedges)
    return edge_index[:, keep_mask]

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

def per_node_topk(V_idx, scores, k):
    """
    V_idx: [nnz] node id for each incidence
    scores: [nnz] score for each (v,e)
    k: scalar top-k per node
    Returns:
        keep_ids: indices of kept incidences
    """

    # 1. Sort by node id, then by score descending
    #    Produces grouping + ranking in one shot
    sort_key = V_idx * (scores.numel()) + (-scores)  # ensure descending sort
    _, perm = torch.sort(sort_key)                   # [nnz]
    
    V_sorted = V_idx[perm]

    # 2. Find segment boundaries
    # Mark edges where V changes
    change = torch.ones_like(V_sorted, dtype=torch.bool)
    change[1:] = V_sorted[1:] != V_sorted[:-1]

    # segment_starts: positions where each node's incidences begin
    segment_starts = torch.nonzero(change, as_tuple=True)[0]
    
    # segment_ends: next start or end of array
    segment_ends = torch.empty_like(segment_starts)
    segment_ends[:-1] = segment_starts[1:]
    segment_ends[-1] = V_sorted.numel()
    
    # 3. For each node segment, keep top-k (fast: at most |V| iterations)
    keep = []
    for start, end in zip(segment_starts.tolist(), segment_ends.tolist()):
        count = end - start
        t = min(count, k)
        keep.append(perm[start:start+t])

    return torch.cat(keep, dim=0)

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
            self.mask_module = LearnableEdgeMask(args.num_incidences,args=args)
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


    def forward(self, data, return_mask = True, is_test = False):
        #         Assume edge_index is already V2V
        if self.args.mode == 'learnmask':
            # edge_index_knn,probs = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio)
            edge_index_knn, _, keepmask, probs = self.mask_module(data,keep_ratio= self.args.keep_ratio,is_test=is_test,return_mask = True)
            keepmask = keepmask.bool()
        if self.args.mode == 'learnmask+':
            # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
            edge_index_knn, _, keepmask, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
            keepmask = keepmask.bool()
        if self.args.mode.startswith('learnmask'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
        else:
            edge_index_knn = data.edge_index


        x, edge_index, norm = data.x, edge_index_knn, data.norm
        if self.args.mode.startswith('learnmask'):
            with torch.no_grad():
                norm = norm[keepmask]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, norm)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, norm)
        if self.args.mode.startswith('learnmask') and is_test == False:
            if return_mask:
                return x, probs, keepmask
            return x, probs
        return x


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
            self.mask_module = LearnableEdgeMask(args.num_incidences,args=args)
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


    def forward(self, data, return_mask = True, is_test=False):
        #         Assume edge_index is already V2V
        if self.args.mode == 'learnmask':
            # edge_index_knn,probs = self.mask_module(data.edge_index,keep_ratio= self.args.keep_ratio)
            edge_index_knn, _, keepmask, probs = self.mask_module(data,keep_ratio= self.args.keep_ratio,is_test=is_test,return_mask = True)
            keepmask = keepmask.bool()
        if self.args.mode == 'learnmask+':
            # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
            edge_index_knn, _, keepmask, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
            keepmask = keepmask.bool()
        if self.args.mode.startswith('learnmask'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
        else:
            edge_index_knn = data.edge_index
        x, edge_index, norm = data.x, edge_index_knn, data.norm
        if self.args.mode.startswith('learnmask'):
            with torch.no_grad():
                norm = norm[keepmask]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if self.args.mode.startswith('learnmask') and is_test == False:
            if return_mask:
                return x, probs, keepmask
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


# For HCHA
def add_self_loops_hypergraph(edge_index, num_nodes, hyperedge_weight=None):
    """
    edge_index: [2, nnz], (node, hyperedge) pairs
    num_nodes: total number of nodes
    hyperedge_weight: [num_hyperedges] or None

    Returns:
      edge_index_new, hyperedge_weight_new
    """
    device = edge_index.device
    nodes, hyperedges = edge_index

    if hyperedge_weight is None:
        # Assume hyperedges are 0..max, set all weights = 1
        num_hyperedges = hyperedges.max().item() + 1
        hyperedge_weight = torch.ones(num_hyperedges, device=device)
    else:
        num_hyperedges = hyperedge_weight.size(0)

    # 1) Hyperedges that currently have degree 1 (self-loop candidates)
    uniq_edges, counts = torch.unique(hyperedges, return_counts=True)
    singleton_hyperedges = uniq_edges[counts == 1]

    # 2) Nodes that already have a singleton hyperedge
    self_loop_mask = torch.isin(hyperedges, singleton_hyperedges)
    existing_self_loop_nodes = nodes[self_loop_mask].unique()

    # 3) Nodes that still need self-loops
    all_nodes = torch.arange(num_nodes, device=device)
    needs_self_loop_mask = ~torch.isin(all_nodes, existing_self_loop_nodes)
    nodes_to_add = all_nodes[needs_self_loop_mask]

    num_new_edges = nodes_to_add.numel()
    if num_new_edges == 0:
        return edge_index.long(), hyperedge_weight  # nothing to add

    # 4) Assign new hyperedge IDs at the END of hyperedge_weight
    start_hyperedge_id = num_hyperedges
    new_hyperedge_ids = torch.arange(
        start_hyperedge_id,
        start_hyperedge_id + num_new_edges,
        device=device
    )

    new_edges = torch.stack([nodes_to_add, new_hyperedge_ids], dim=0)
    edge_index_new = torch.cat([edge_index, new_edges], dim=1)

    # 5) Extend hyperedge_weight with ones for these new self-loop edges
    new_weights = torch.ones(num_new_edges, device=device)
    hyperedge_weight_new = torch.cat([hyperedge_weight, new_weights], dim=0)

    # (Optional) you *don't* need to sort; if you do, keep indices & weights aligned.
    # _, sorted_idx = torch.sort(edge_index_new[0])
    # edge_index_new = edge_index_new[:, sorted_idx]

    return edge_index_new.long(), hyperedge_weight_new


# gradient passes to learnmasks logit in this version.
# We used softmasks as weights in the HGNN conv layer
class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer 
    is implemented in pyg.
    """

    def __init__(self, args):
        super(HCHA, self).__init__()
        self.args = args 
        if args.mode == 'learnmask':
            self.mask_module = LearnableEdgeMask(args.num_incidences,args=args)
        if args.mode == 'learnmask+':
            self.mask_module = LearnableEdgeMaskplus(args.F,hidden_dim=args.coarse_MLP, args=args)
        if args.mode == 'learnmask_cond':    # NEW: feature-conditioned *incidence* mask (EHGNN-F)
            self.mask_module = FeatureConditionedIncidenceMask(args.F, hidden_dim=args.coarse_MLP, agg='mean',args=args)
            # self.mask_module = SegmentwiseFeatureIncidenceMask(args.F, hidden_dim=args.coarse_MLP,args=args)
        if args.mode == 'learnmask+_agn':     # NEW: feature-agnostic *edge* mask (EHGNN-C, explicit)
            self.mask_module = FeatureAgnosticEdgeMask_nob(int(args.num_hyperedges),args=args)
        if args.mode == 'Neural': # EHGNN-C (cond,LR)
            self.mask_module = EHGNN_CcondLR(args, rank=4)
        if args.mode == 'NeuralF':
            self.mask_module = EHGNN_FcondLR(args, rank=4, top_k=5) # EHGNN-F (cond,LR)
        print('All_num_layers: ',args.All_num_layers,' MLP_hidden: ',args.MLP_hidden)
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.ratio = 1.0
#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConvOld(args.num_features,
                                         args.MLP_hidden, self.symdegnorm))
        for _ in range(self.num_layers-2):
            self.convs.append(HypergraphConvOld(
                args.MLP_hidden, args.MLP_hidden, self.symdegnorm))
        # Output heads is set to 1 as default
        self.convs.append(HypergraphConvOld(
            args.MLP_hidden, args.num_classes, self.symdegnorm))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, is_test=False,return_mask = False):
        if (self.args.mode.startswith('learnmask') or self.args.mode.startswith('Neural')):
            if return_mask:
                edge_index_knn, weights, hard_masks, scores = self.mask_module(data,keep_ratio= self.args.keep_ratio,is_test=is_test, return_mask = True)
            else:
                edge_index_knn,weights, scores = self.mask_module(data,keep_ratio= self.args.keep_ratio,is_test=is_test)
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = self.args.verbose)
        else: # Full, Random, Degdist, Effresist
            edge_index_knn = data.edge_index
            weights = None 
        x = data.x
        # edge_index = data.edge_index
        # if epoch == 0:
        #     print('beforing self-loop: weights shape: ',weights.shape,' ', data.num_hyperedges, ' #incidences: ',self.args.num_incidences)
        edge_index = edge_index_knn
        if self.args.mode != 'full':
            edge_index,weights = add_self_loops_hypergraph(
                                                    edge_index, 
                                                    num_nodes=data.n_x,            
                                                    hyperedge_weight=weights,   # may be None or [num_hyperedges]
                                                    )
        # if epoch == 0:
        #     print('weights shape: ',weights.shape,' ', data.num_hyperedges, ' #incidences: ',self.args.num_incidences)
        usewt = True 
        for i, conv in enumerate(self.convs[:-1]):
            if usewt: 
                x = F.elu(conv(x, edge_index, hyperedge_weight = weights))
            else:
                x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

#         x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if (self.args.mode.startswith('learnmask') or self.args.mode.startswith('Neural')) and is_test == False:
            if return_mask:
                return x, scores, hard_masks
            return x, scores
            
        return x

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
            self.mask_module = LearnableEdgeMask(args.num_incidences,args=args)
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


    def forward(self, data, return_mask=True, is_test=False):
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
            edge_index_knn, _, keepmask, probs = self.mask_module(data,keep_ratio= self.args.keep_ratio,is_test=is_test,return_mask = True)
        if self.args.mode == 'learnmask+':
            # edge_index_knn, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio)
            edge_index_knn, _, keepmask, probs = self.mask_module(data, keep_ratio=self.args.keep_ratio,is_test = is_test, return_mask = True)
        if self.args.mode.startswith('learnmask'):
            self.ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = False)
            probs = probs.cpu().detach()
        else:
            edge_index_knn = data.edge_index
        # 
        x, edge_index, norm = data.x, edge_index_knn, data.norm
        if self.args.mode.startswith('learnmask'):
            with torch.no_grad():
                keepmask = keepmask.bool()
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
            if return_mask:
                return x, probs, keepmask
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

# class HSL(nn.Module):
#     """
#     Hyperedge Sampling Layer (HSL)-style sparsifier.

#     - Encoder: node MLP -> h0
#     - Two HGNN-style views:
#         * Full view: HGNN on full hypergraph with self-loops (all hyperedges kept).
#         * Sampled view: HGNN on same incidence graph but with learnable
#           hyperedge weights (soft sample) and binary masks for empirical r_v.
#     - Sampling (Sec. 3.3):
#         * Hyperedge-wise scores from aggregated node embeddings.
#         * Gumbel-sigmoid reparameterization for m_e in (0, 1).
#     - Contrastive loss:
#         * Alignment between node embeddings of full and sampled views.

#     Outputs:
#         - logits: [N, C] (masked view, used for classification)
#         - probs_e: [M]   (hyperedge probabilities)
#         - contrastive_loss: scalar
#         - (optionally) hard_mask_e: [M] in {0,1} and current_rv.
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         in_dim = args.num_features
#         hid_dim = args.MLP_hidden
#         num_classes = args.num_classes
#         self.num_layers = args.All_num_layers  # #HGNN layers
#         self.add_self_loop = getattr(args, "add_self_loop", True)

#         # HSL hyperparameters
#         self.temperature = getattr(args, "hsl_tau", 0.5)
#         self.contrastive_weight = getattr(args, "hsl_contrastive_weight", 1.0)

#         # ----- Node encoder -----
#         self.encoder = MLP(
#             in_channels=in_dim,
#             hidden_channels=hid_dim,
#             out_channels=hid_dim,
#             num_layers=args.MLP_num_layers,
#             dropout=args.dropout,
#             Normalization=args.normalization,
#             InputNorm=args.deepset_input_norm,
#         )

#         # ----- Hyperedge scoring MLP (for sampling) -----
#         # Takes aggregated hyperedge embeddings (mean over nodes) as input.
#         self.mask_mlp = MLP(
#             in_channels=hid_dim,
#             hidden_channels=hid_dim,
#             out_channels=1,
#             num_layers=args.MLP_num_layers,
#             dropout=args.dropout,
#             Normalization=args.normalization,
#             InputNorm=False,
#         )

#         # ----- Classifier on node embeddings (sampled view) -----
#         self.classifier = MLP(
#             in_channels=hid_dim,
#             hidden_channels=args.Classifier_hidden,
#             out_channels=num_classes,
#             num_layers=args.Classifier_num_layers,
#             dropout=args.dropout,
#             Normalization=args.normalization,
#             InputNorm=False,
#         )


#     def reset_parameters(self):
#         self.encoder.reset_parameters()
#         self.mask_mlp.reset_parameters()
#         self.classifier.reset_parameters()
#         self.current_rv = None

#     # ------------------------------------------------------
#     # HGNN-like propagation with hyperedge weights m_e
#     # H is given as incidence list "edge_index": [2, nnz]
#     # v_idx in [0, N-1], e_idx in [0, M_total-1]
#     # m_e: [M_total] in (0,1] (includes original + self-loop hyperedges)
#     # ------------------------------------------------------
#     @staticmethod
#     def _hgnn_propagate(h_in, edge_index, num_nodes, num_edges, m_e):
#         """
#         Simple two-step HGNN propagation with hyperedge weights m_e:

#             node -> edge (weighted mean)
#             edge -> node (weighted mean)

#         This corresponds to applying H * diag(m_e) in both directions.
#         """
#         v_idx = edge_index[0]  # [nnz]
#         e_idx = edge_index[1]  # [nnz]
#         w = m_e[e_idx]         # [nnz]

#         # node -> edge
#         msg_e = h_in[v_idx] * w.unsqueeze(-1)  # [nnz, F]
#         sum_e = scatter(msg_e, e_idx, dim=0, dim_size=num_edges, reduce='sum')
#         deg_e = scatter(w, e_idx, dim=0, dim_size=num_edges, reduce='sum')
#         h_e = sum_e / deg_e.clamp_min(1e-12).unsqueeze(-1)

#         # edge -> node
#         msg_v = h_e[e_idx] * w.unsqueeze(-1)   # [nnz, F]
#         sum_v = scatter(msg_v, v_idx, dim=0, dim_size=num_nodes, reduce='sum')
#         deg_v = scatter(w, v_idx, dim=0, dim_size=num_nodes, reduce='sum')
#         h_v = sum_v / deg_v.clamp_min(1e-12).unsqueeze(-1)

#         return h_v

#     @staticmethod
#     def contrastive_alignment(h_full, h_masked):
#         """
#         Alignment-style contrastive loss between views:

#             L = E_i || normalize(h_full_i) - normalize(h_masked_i) ||^2
#         """
#         z1 = F.normalize(h_full, dim=-1)
#         z2 = F.normalize(h_masked, dim=-1)
#         return ((z1 - z2) ** 2).sum(dim=-1).mean()

#     @staticmethod
#     def compute_empirical_rv(edge_index_no_self, hard_mask_e, num_nodes):
#         """
#         Compute empirical vertex retention r_v based on original H and sampled \hat{H}
#         BEFORE adding self-loops.

#         r_v = (# vertices incident to at least one sampled hyperedge) /
#               (# vertices incident to at least one original hyperedge)
#         """
#         v_idx = edge_index_no_self[0]  # [nnz]
#         e_idx = edge_index_no_self[1]  # [nnz], 0..M-1

#         # original degrees
#         deg_orig = torch.bincount(v_idx, minlength=num_nodes)

#         # incidences kept after sampling
#         keep_inc = hard_mask_e[e_idx] > 0.5
#         if keep_inc.sum() == 0:
#             # trivial case: everything pruned -> r_v = 0 (or 1 by convention)
#             return torch.tensor(0.0, device=edge_index_no_self.device)

#         v_idx_kept = v_idx[keep_inc]
#         deg_mask = torch.bincount(v_idx_kept, minlength=num_nodes)

#         active_orig = deg_orig > 0
#         if active_orig.sum() == 0:
#             return torch.tensor(1.0, device=edge_index_no_self.device)

#         active_mask = (deg_mask > 0) & active_orig

#         rv = active_mask.float().sum() / active_orig.float().sum()
#         return rv

#     def _build_self_looped_edge_index(self, data):
#         """
#         Build incidence with self-loops added on top of original H.
#         We never sample self-loop hyperedges; they stay always active.
#         """
#         device = data.x.device

#         # Original incidence (no self-loops), with hyperedge ids 0..M-1
#         edge_index_no_self = data.edge_index.to(device)

#         # Number of original hyperedges (M_orig)
#         if isinstance(data.num_hyperedges, torch.Tensor):
#             num_he_orig = int(data.num_hyperedges)
#         else:
#             num_he_orig = int(data.num_hyperedges)

#         if not self.add_self_loop:
#             # No additional hyperedges
#             num_edges_total = num_he_orig
#             return edge_index_no_self, edge_index_no_self, num_he_orig, num_edges_total

#         # Use your existing Add_Self_Loops helper
#         tmp = Data()
#         tmp.edge_index = edge_index_no_self.detach().cpu()

#         if isinstance(data.n_x, torch.Tensor):
#             tmp.n_x = data.n_x.detach().cpu()
#             num_nodes = int(data.n_x[0].item())
#         else:
#             num_nodes = int(data.n_x)
#             tmp.n_x = torch.tensor([num_nodes], dtype=torch.long)

#         # num_hyperedges used internally by Add_Self_Loops
#         if isinstance(data.num_hyperedges, torch.Tensor):
#             tmp.num_hyperedges = data.num_hyperedges.detach().cpu()
#         else:
#             tmp.num_hyperedges = torch.tensor([num_he_orig], dtype=torch.long)

#         tmp = Add_Self_Loops(tmp)
#         edge_index_with_self = tmp.edge_index.to(device)

#         # total hyperedges after adding self-loops
#         num_edges_total = int(edge_index_with_self[1].max().item()) + 1

#         return edge_index_no_self, edge_index_with_self, num_he_orig, num_edges_total

#     # def _sample_hyperedges(self, he_feat, is_test=False):
#     #     """
#     #     Hyperedge-wise sampling:

#     #     he_feat : [M_orig, H]
#     #     Returns:
#     #         probs_e   : [M_orig] sigmoid scores
#     #         m_soft_e  : [M_orig] Gumbel-sigmoid (training) or probs (eval)
#     #         m_hard_e  : [M_orig] binary mask for r_v computation (and logging)
#     #     """
#     #     logits = self.mask_mlp(he_feat).view(-1)   # [M_orig]
#     #     probs = torch.sigmoid(logits)              # [M_orig]

#     #     if self.training and not is_test:
#     #         # Gumbel-sigmoid reparameterization (continuous sample)
#     #         eps = 1e-10
#     #         u = torch.rand_like(probs)
#     #         g = -torch.log(-torch.log(u + eps) + eps)
#     #         logit = torch.log(probs + eps) - torch.log(1.0 - probs + eps)
#     #         y = (logit + g) / self.temperature
#     #         m_soft = torch.sigmoid(y)
#     #     else:
#     #         # deterministic at test time
#     #         m_soft = probs

#     #     # Binary mask used only for measuring r_v (and optionally logging).
#     #     m_hard = (m_soft >= 0.5).float()
#     #     # m_hard = torch.nn.functional.gumbel_softmax(probs,tau=self.temperature,hard = True)
#     #     # m_soft = None 
#     #     return probs, m_soft, m_hard

#     def _sample_hyperedges(self, he_feat, is_test=False):
#         """
#         he_feat: [M, H]
#         returns:
#             probs_e : [M]   = sigmoid(logits)
#             m_st_e  : [M]   = straight-through hard mask used in propagation
#             m_hard  : [M]   = detached hard mask used for statistics (r, r_v)
#         """
#         logits = self.mask_mlp(he_feat).view(-1)   # [M]
#         probs  = torch.sigmoid(logits)             # [M]

#         if self.training and not is_test:
#             # Concrete / Gumbel–sigmoid (binary analogue of gumbel_softmax)
#             eps = 1e-10
#             u   = torch.rand_like(logits)
#             g   = -torch.log(-torch.log(u + eps) + eps)
#             y   = (logits + g) / self.temperature
#             m_soft = torch.sigmoid(y)              # (0,1)
#         else:
#             # deterministic at eval time
#             m_soft = probs

#         # HARD mask (0/1) as in the paper
#         m_hard = (m_soft >= 0.5).float()

#         # Straight-through version used in the graph:
#         # forward == m_hard, backward gradients == m_soft
#         m_st = (m_hard - m_soft).detach() + m_soft

#         return probs, m_st, m_hard

#     def forward(self, data, epoch=None, is_test=False, return_mask=False):
#         """
#         Forward pass:

#         - Build full incidence with self-loops (H_full).
#         - Encode nodes: x -> h0.
#         - Aggregate hyperedge features from h0.
#         - Hyperedge-wise sampling -> m_soft_e, m_hard_e, probs_e.
#         - Compute empirical r_v using H and m_hard_e (no self-loops).
#         - Define hyperedge weights for:
#             * full view: w_full = 1 on all hyperedges (orig + self-loops)
#             * sampled view: w_mask = m_soft_e on orig hyperedges, 1 on self-loops
#         - Run HGNN-style propagation on both views for num_layers steps.
#         - Compute contrastive alignment loss between h_full and h_masked.
#         - Classify using sampled view.
#         """
#         x = data.x
#         device = x.device

#         # Node count
#         if isinstance(data.n_x, torch.Tensor):
#             num_nodes = int(data.n_x[0].item())
#         else:
#             num_nodes = x.size(0)

#         # Build incidence with and without self-loops
#         edge_index_no_self, edge_index_full, num_he_orig, num_edges_total = \
#             self._build_self_looped_edge_index(data)

#         # ----- Node encoder -----
#         h0 = self.encoder(x)  # [N, H]

#         # ----- Hyperedge features (mean pooling over encoded nodes) -----
#         # We use the ORIGINAL incidence (no self-loops) here, since sampling is only on true hyperedges.
#         v_idx = edge_index_no_self[0]
#         e_idx = edge_index_no_self[1]  # 0..M_orig-1

#         he_feat = scatter(
#             h0[v_idx], e_idx, dim=0, dim_size=num_he_orig, reduce='mean'
#         )  # [M_orig, H]

#         # ----- Sample hyperedges -----
#         probs_e, m_soft_e, m_hard_e = self._sample_hyperedges(he_feat, is_test=is_test)
#         self.ratio = float(m_hard_e.mean().item())

#         # ----- Empirical r_v (before adding self-loops) -----
#         # with torch.no_grad():
#         #     rv = self._compute_empirical_rv(edge_index_no_self, m_hard_e, num_nodes)
#         #     self.current_rv = float(rv.item())

#         # ----- Hyperedge weights for full and sampled views -----
#         # full view: all hyperedges (orig + self-loops) have weight 1
#         w_full = h0.new_ones(num_edges_total)

#         # sampled view: original hyperedges use m_soft_e, self-loops weight = 1
#         w_mask = h0.new_ones(num_edges_total)
#         w_mask[:num_he_orig] = m_soft_e

#         # ----- HGNN-style propagation (num_layers) -----
#         h_full = h0
#         h_masked = h0
#         for _ in range(self.num_layers):
#             h_full = self._hgnn_propagate(h_full, edge_index_full,
#                                           num_nodes, num_edges_total, w_full)
#             h_full = F.relu(h_full)

#             h_masked = self._hgnn_propagate(h_masked, edge_index_full,
#                                             num_nodes, num_edges_total, w_mask)
#             h_masked = F.relu(h_masked)

#         # ----- Contrastive alignment loss -----
#         con_loss = self.contrastive_alignment(h_full, h_masked)

#         # ----- Classifier on masked view -----
#         logits = self.classifier(h_masked)  # [N, C]

#         if return_mask:
#             # logits, hyperedge_probs, hard_mask, contrastive_loss
#             return logits, probs_e, m_hard_e, con_loss
#         else:
#             # logits, hyperedge_probs, contrastive_loss
#             return logits, probs_e, con_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data


class HSL(nn.Module):
    """
    Hyperedge Sampling Layer (HSL)-style sparsifier with
    (i) hyperedge sampling and
    (ii) incident-node sampling, following Eq. (6) of the paper.

    Pipeline (Option A):

    1) Encoder HGNN:
       - Input: original incidence H (without self-loops).
       - We build an H_with_self that has additional self-loop hyperedges.
       - Run a parameter-free HGNN-style propagation on H_with_self
         starting from an MLP-encoded X, to get:
           - node embeddings X_tilde (h_enc_nodes)
           - hyperedge embeddings H_e (h_e_orig) via mean pooling.

    2) Mask generation:
       - hyperedge scores:
             z^e_i = MLP_e(H_e[i])
       - incident-node scores (for each incidence (v,e)):
             Z_{v,e} = MLP_v([X_tilde[v] || H_e[e]])
         (this corresponds to Eq. (6) in the paper.)

       - Both are turned into Bernoulli masks via straight-through
         Gumbel-sigmoid:
             m_e      : [M]   hyperedge mask (0/1, ST)
             m_ve     : [nnz] incidence mask (0/1, ST)

    3) Masked incidence:
       - For each incidence index k with (v_k, e_k):
             w_mask_inc[k] = m_e[e_k] * m_ve[k]
       - For self-loop hyperedges (added hyperedges), we do NOT sample:
             w_mask_inc[k] = 1 when e_k >= M_orig

    4) Downstream HGNN:
       - Two HGNN-style stacks (parameter-free) starting from X_tilde:
            full view   : weights all ones → h_full
            masked view : weights w_mask_inc → h_masked

       - Contrastive loss:
            L_con = E_v || normalize(h_full_v) - normalize(h_masked_v) ||^2

       - Classification MLP on h_masked.

    Outputs:
        forward(..., return_mask=False):
            logits         : [N, C] (masked view)
            probs_e        : [M]  (sigmoid(hyperedge logits))
            contrastive_loss

        forward(..., return_mask=True):
            logits, probs_e, m_hard_e, contrastive_loss

    Bookkeeping:
        - self.ratio           : mean(m_hard_e) = fraction of hyperedges kept
        - self._last_m_ve_hard : [nnz] hard incidence mask from last forward
        - compute_empirical_rv uses both m_hard_e and self._last_m_ve_hard
          to compute r_v.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        in_dim = args.num_features
        hid_dim = args.MLP_hidden
        num_classes = args.num_classes

        # Number of HGNN layers for downstream propagation
        self.num_layers = args.All_num_layers
        self.add_self_loop = getattr(args, "add_self_loop", True)

        # HSL hyperparameters
        self.temperature = getattr(args, "hsl_tau", 0.5)
        self.contrastive_weight = getattr(args, "hsl_contrastive_weight", 1.0)

        # ----- Node encoder (pre-HGNN) -----
        self.encoder = MLP(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            out_channels=hid_dim,
            num_layers=args.MLP_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=args.deepset_input_norm,
        )

        # ----- Hyperedge scoring MLP (for hyperedge-wise sampling) -----
        self.edge_mask_mlp = MLP(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            out_channels=1,
            num_layers=args.MLP_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

        # ----- Incident-node scoring MLP (Eq. (6)) -----
        # Input: concat([X_tilde[v], H_e[e]])  → scalar
        self.incident_mask_mlp = MLP(
            in_channels=2 * hid_dim,
            hidden_channels=hid_dim,
            out_channels=1,
            num_layers=args.MLP_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

        # ----- Classifier on node embeddings (sampled view) -----
        self.classifier = MLP(
            in_channels=hid_dim,
            hidden_channels=args.Classifier_hidden,
            out_channels=num_classes,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

        # Bookkeeping for r and r_v
        self.current_rv = None
        self.ratio = 1.0            # hyperedge retention fraction
        self._last_m_ve_hard = None # last hard incidence mask

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_mask_mlp.reset_parameters()
        self.incident_mask_mlp.reset_parameters()
        self.classifier.reset_parameters()
        self.current_rv = None
        self.ratio = 1.0
        self._last_m_ve_hard = None

    # ------------------------------------------------------------------
    # HGNN-style propagation with per-incidence weights
    # H is given as edge_index: [2, nnz]
    # w_inc: [nnz] weights on incidences (node, hyperedge).
    # ------------------------------------------------------------------
    @staticmethod
    def _hgnn_propagate(h_in, edge_index, num_nodes, num_edges, w_inc):
        """
        Two-step HGNN propagation with per-incidence weights w_inc:

            node -> edge (weighted mean)
            edge -> node (weighted mean)
        """
        v_idx = edge_index[0]  # [nnz]
        e_idx = edge_index[1]  # [nnz]
        w = w_inc              # [nnz]

        # node -> edge
        msg_e = h_in[v_idx] * w.unsqueeze(-1)  # [nnz, F]
        sum_e = scatter(msg_e, e_idx, dim=0, dim_size=num_edges, reduce='sum')
        deg_e = scatter(w, e_idx, dim=0, dim_size=num_edges, reduce='sum')
        h_e = sum_e / deg_e.clamp_min(1e-12).unsqueeze(-1)

        # edge -> node
        msg_v = h_e[e_idx] * w.unsqueeze(-1)   # [nnz, F]
        sum_v = scatter(msg_v, v_idx, dim=0, dim_size=num_nodes, reduce='sum')
        deg_v = scatter(w, v_idx, dim=0, dim_size=num_nodes, reduce='sum')
        h_v = sum_v / deg_v.clamp_min(1e-12).unsqueeze(-1)

        return h_v

    # ------------------------------------------------------------------
    # Contrastive alignment between full-view and masked-view node reps
    # ------------------------------------------------------------------
    @staticmethod
    def contrastive_alignment(h_full, h_masked):
        z1 = F.normalize(h_full, dim=-1)
        z2 = F.normalize(h_masked, dim=-1)
        return ((z1 - z2) ** 2).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Build incidence with and without self-loops
    # ------------------------------------------------------------------
    def _build_self_looped_edge_index(self, data):
        """
        Build incidence with self-loops added on top of original H.
        We never sample self-loop hyperedges; they stay always active.

        Returns:
            edge_index_no_self  : [2, nnz_orig]   original incidences
            edge_index_full     : [2, nnz_full]   with self-loops
            num_he_orig         : M_orig
            num_edges_total     : M_total (orig + self-loops)
        """
        device = data.x.device

        # Original incidence (no self-loops), with hyperedge ids 0..M-1
        edge_index_no_self = data.edge_index.to(device)

        # Number of original hyperedges (M_orig)
        if isinstance(data.num_hyperedges, torch.Tensor):
            num_he_orig = int(data.num_hyperedges)
        else:
            num_he_orig = int(data.num_hyperedges)

        if not self.add_self_loop:
            num_edges_total = num_he_orig
            return edge_index_no_self, edge_index_no_self, num_he_orig, num_edges_total

        # Use existing Add_Self_Loops helper (assumed to preserve
        # original incidence order and append self-loops at the end).
        tmp = Data()
        tmp.edge_index = edge_index_no_self.detach().cpu()

        if isinstance(data.n_x, torch.Tensor):
            tmp.n_x = data.n_x.detach().cpu()
            num_nodes = int(data.n_x[0].item())
        else:
            num_nodes = int(data.n_x)
            tmp.n_x = torch.tensor([num_nodes], dtype=torch.long)

        # num_hyperedges used internally by Add_Self_Loops
        if isinstance(data.num_hyperedges, torch.Tensor):
            tmp.num_hyperedges = data.num_hyperedges.detach().cpu()
        else:
            tmp.num_hyperedges = torch.tensor([num_he_orig], dtype=torch.long)

        tmp = Add_Self_Loops(tmp)
        edge_index_with_self = tmp.edge_index.to(device)

        # Total hyperedges after adding self-loops
        num_edges_total = int(edge_index_with_self[1].max().item()) + 1

        return edge_index_no_self, edge_index_with_self, num_he_orig, num_edges_total

    # ------------------------------------------------------------------
    # Straight-through Bernoulli for hyperedges and incidences
    # ------------------------------------------------------------------
    def _sample_masks(self, h_nodes, h_e_orig, edge_index_no_self, is_test=False):
        """
        Mask generation:

        Inputs:
            h_nodes           : [N, H] encoded node embeddings (X_tilde)
            h_e_orig          : [M, H] hyperedge embeddings
            edge_index_no_self: [2, nnz_orig] original incidence

        Returns:
            probs_e    : [M]       sigmoid edge probs
            m_e_st     : [M]       ST hard hyperedge mask for propagation
            m_e_hard   : [M]       detached hard hyperedge mask (0/1)
            m_ve_st    : [nnz]     ST hard incidence mask for propagation
            m_ve_hard  : [nnz]     detached hard incidence mask (0/1)
        """
        v_idx = edge_index_no_self[0]  # [nnz]
        e_idx = edge_index_no_self[1]  # [nnz], 0..M-1
        num_he_orig = h_e_orig.size(0)

        # ----- Hyperedge scores -----
        edge_logits = self.edge_mask_mlp(h_e_orig).view(-1)  # [M]
        probs_e = torch.sigmoid(edge_logits)                 # [M]

        # ----- Incident-node scores (Eq. (6)) -----
        # concat([X_tilde[v], H_e[e]]) for each incidence
        node_part = h_nodes[v_idx]            # [nnz, H]
        edge_part = h_e_orig[e_idx]          # [nnz, H]
        inc_feat = torch.cat([node_part, edge_part], dim=-1)  # [nnz, 2H]

        inc_logits = self.incident_mask_mlp(inc_feat).view(-1)  # [nnz]
        probs_ve = torch.sigmoid(inc_logits)                    # [nnz]

        # ----- ST Gumbel-sigmoid sampling for both -----
        def st_bernoulli(logits, probs, training, is_test):
            if training and not is_test:
                eps = 1e-10
                u = torch.rand_like(logits)
                g = -torch.log(-torch.log(u + eps) + eps)
                y = (logits + g) / self.temperature
                m_soft = torch.sigmoid(y)           # (0,1)
            else:
                m_soft = probs                      # deterministic at eval

            m_hard = (m_soft >= 0.5).float()        # 0/1 hard mask
            m_st = (m_hard - m_soft).detach() + m_soft  # straight-through
            return m_st, m_hard

        m_e_st, m_e_hard = st_bernoulli(edge_logits, probs_e,
                                        self.training, is_test)
        m_ve_st, m_ve_hard = st_bernoulli(inc_logits, probs_ve,
                                          self.training, is_test)

        # Bookkeeping:
        self.ratio = float(m_e_hard.mean().item())     # r: hyperedge retention
        self._last_m_ve_hard = m_ve_hard.detach()      # store for r_v

        return probs_e, m_e_st, m_e_hard, m_ve_st, m_ve_hard

    # ------------------------------------------------------------------
    # Empirical r_v using hyperedge + incidence masks
    # ------------------------------------------------------------------
    def compute_empirical_rv(self, edge_index_no_self, hard_mask_e, num_nodes):
        """
        Compute empirical vertex retention r_v based on the combined
        effect of hyperedge mask (hard_mask_e) and the last incidence
        mask self._last_m_ve_hard.

        r_v = (# vertices incident to at least one sampled incidence) /
              (# vertices incident to at least one original incidence)
        """
        device = edge_index_no_self.device

        if isinstance(num_nodes, torch.Tensor):
            num_nodes_int = int(num_nodes[0].item())
        else:
            num_nodes_int = int(num_nodes)

        v_idx = edge_index_no_self[0]  # [nnz]
        e_idx = edge_index_no_self[1]  # [nnz], 0..M-1

        # original degrees
        deg_orig = torch.bincount(v_idx, minlength=num_nodes_int)

        # incidence mask: combine hyperedge mask and incidence mask
        if (self._last_m_ve_hard is not None and
            self._last_m_ve_hard.size(0) == v_idx.size(0)):
            keep_inc = (hard_mask_e[e_idx] > 0.5) & (self._last_m_ve_hard > 0.5)
        else:
            # fallback: only hyperedge-level mask
            keep_inc = (hard_mask_e[e_idx] > 0.5)

        if keep_inc.sum() == 0:
            # trivial case: everything pruned
            rv = torch.tensor(0.0, device=device)
            self.current_rv = float(rv.item())
            return rv

        v_idx_kept = v_idx[keep_inc]
        deg_mask = torch.bincount(v_idx_kept, minlength=num_nodes_int)

        active_orig = deg_orig > 0
        if active_orig.sum() == 0:
            rv = torch.tensor(1.0, device=device)
            self.current_rv = float(rv.item())
            return rv

        active_mask = (deg_mask > 0) & active_orig
        rv = active_mask.float().sum() / active_orig.float().sum()
        self.current_rv = float(rv.item())
        return rv

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data, epoch=None, is_test=False, return_mask=False):
        """
        Forward pass:

        1) Build H_no_self and H_full (with self-loops).
        2) Encode nodes via encoder MLP, then one HGNN stack on H_full
           to get X_tilde and hyperedge embeddings H_e for scoring.
        3) Sample hyperedges and incidences → masks.
        4) Build per-incidence weights for full and masked views.
        5) Run downstream HGNN for both views and compute contrastive loss.
        6) Classify using masked-view node embeddings.

        Returns:
            if return_mask == False:
                logits, probs_e, contrastive_loss
            else:
                logits, probs_e, m_hard_e, contrastive_loss
        """
        x = data.x
        device = x.device

        # Node count
        if isinstance(data.n_x, torch.Tensor):
            num_nodes = int(data.n_x[0].item())
        else:
            num_nodes = x.size(0)

        # Build incidence with and without self-loops
        edge_index_no_self, edge_index_full, num_he_orig, num_edges_total = \
            self._build_self_looped_edge_index(data)

        nnz_orig = edge_index_no_self.size(1)
        nnz_full = edge_index_full.size(1)

        # ----- Step 1: encoder MLP + HGNN to get X_tilde and H_e -----
        # MLP encoder
        h0 = self.encoder(x)  # [N, H]

        # Encoder HGNN (parameter-free) on full graph with self-loops
        w_full_inc_enc = h0.new_ones(nnz_full)
        h_enc = h0
        for _ in range(self.num_layers):
            h_enc = self._hgnn_propagate(
                h_enc, edge_index_full,
                num_nodes=num_nodes,
                num_edges=num_edges_total,
                w_inc=w_full_inc_enc,
            )
            h_enc = F.relu(h_enc)

        # Hyperedge embeddings for ORIGINAL hyperedges only
        v_ns = edge_index_no_self[0]
        e_ns = edge_index_no_self[1]
        h_e_orig = scatter(
            h_enc[v_ns], e_ns, dim=0, dim_size=num_he_orig, reduce='mean'
        )  # [M_orig, H]

        # ----- Step 2: sample hyperedges + incidences -----
        probs_e, m_e_st, m_e_hard, m_ve_st, m_ve_hard = \
            self._sample_masks(h_enc, h_e_orig, edge_index_no_self,
                               is_test=is_test)

        # ----- Step 3: per-incidence weights for full & masked views -----
        # full view: all incidences weight = 1
        w_full_inc = h_enc.new_ones(nnz_full)

        # masked view:
        #   - for original incidences (first nnz_orig entries), use:
        #       w_mask_inc_orig[k] = m_e_st[e_ns[k]] * m_ve_st[k]
        #   - for self-loop incidences (rest), weight = 1
        w_mask_inc = h_enc.new_ones(nnz_full)
        w_mask_inc_orig = m_e_st[e_ns] * m_ve_st       # [nnz_orig]
        w_mask_inc[:nnz_orig] = w_mask_inc_orig        # assume Add_Self_Loops
                                                       # preserves order

        # ----- Step 4: downstream HGNN on full vs masked views -----
        h_full = h_enc
        h_masked = h_enc
        for _ in range(self.num_layers):
            h_full = self._hgnn_propagate(
                h_full, edge_index_full,
                num_nodes=num_nodes,
                num_edges=num_edges_total,
                w_inc=w_full_inc,
            )
            h_full = F.relu(h_full)

            h_masked = self._hgnn_propagate(
                h_masked, edge_index_full,
                num_nodes=num_nodes,
                num_edges=num_edges_total,
                w_inc=w_mask_inc,
            )
            h_masked = F.relu(h_masked)

        # ----- Step 5: contrastive alignment loss -----
        con_loss = self.contrastive_alignment(h_full, h_masked)

        # ----- Step 6: classifier on masked view -----
        logits = self.classifier(h_masked)  # [N, C]

        if return_mask:
            # (logits, hyperedge_probs, hard_hyperedge_mask, contrastive_loss)
            return logits, probs_e, m_e_hard, con_loss
        else:
            return logits, probs_e, con_loss
