"""Models and structural helpers used by the LoG experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch_scatter import scatter

from layers import HalfNLHconv, HypergraphConv, HypergraphConvOld, MLP

import math

class EHGNN_FcondLR(nn.Module):
    """Low-rank feature-conditioned incidence scorer."""

    def __init__(self, args, rank=16):
        super().__init__()
        self.rank = rank
        self.args = args

        self.node_proj = nn.Linear(args.num_features, rank, bias=False)
        self.edge_factors = nn.Parameter(torch.randn(int(args.num_hyperedges), rank))

    def reset_parameters(self):
        self.node_proj.reset_parameters()
        nn.init.normal_(
            self.edge_factors,
            mean=0.0,
            std=float(getattr(self.args, 'mask_init_std', 0.01))
            / math.sqrt(max(1, self.rank)),
        )

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask=False):
        node_ids, edge_ids = data.edge_index
        logits = (
            self.node_proj(data.x)[node_ids] * self.edge_factors[edge_ids]
        ).sum(dim=-1)
        scores = torch.sigmoid(logits)
        k = max(1, int(keep_ratio * scores.numel()))

        if is_test:
            keep_ids = torch.topk(scores, k).indices
        elif self.args.sampling == "multinomial":
            keep_ids = torch.multinomial(scores, k, replacement=False)
        else:
            keep_ids = torch.topk(scores, k).indices

        hard = torch.zeros_like(scores)
        hard[keep_ids] = 1.0
        straight_through = hard + (scores - scores.detach()) * hard
        edge_index_pruned = torch.stack(
            [node_ids[keep_ids], edge_ids[keep_ids]]
        )
        incidence_weights = straight_through[keep_ids]
        if return_mask:
            return edge_index_pruned, incidence_weights, hard, scores
        return edge_index_pruned, incidence_weights, scores


# EHGNN-F (cond)
class FeatureConditionedIncidenceMask(nn.Module):
    """Direct feature-conditioned incidence scorer."""

    def __init__(self, feature_dim, hidden_dim=32, args=None):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.args = args

    def reset_parameters(self):
        self.scorer[0].reset_parameters()
        output = self.scorer[2]
        nn.init.normal_(
            output.weight,
            mean=0.0,
            std=float(getattr(self.args, 'mask_init_std', 0.01))
            / math.sqrt(max(1, output.in_features)),
        )
        nn.init.zeros_(output.bias)

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask=False):
        node_ids, edge_ids = data.edge_index
        edge_features = scatter(
            data.x[node_ids],
            edge_ids,
            dim=0,
            dim_size=int(data.num_hyperedges),
            reduce="mean",
        )
        pair = torch.cat([data.x[node_ids], edge_features[edge_ids]], dim=1)
        logits = self.scorer(pair).squeeze(-1)
        scores = torch.sigmoid(logits)
        k = max(1, int(scores.numel() * keep_ratio))

        if is_test:
            keep_ids = torch.topk(scores, k).indices
        elif self.args.sampling == "multinomial":
            keep_ids = torch.multinomial(scores, k, replacement=False)
        else:
            keep_ids = torch.topk(scores, k).indices

        hard = torch.zeros_like(scores)
        hard[keep_ids] = 1.0
        straight_through = hard + (scores - scores.detach()) * hard
        edge_index_pruned = torch.stack(
            [node_ids[keep_ids], edge_ids[keep_ids]]
        )
        incidence_weights = straight_through[keep_ids]
        if return_mask:
            return edge_index_pruned, incidence_weights, hard, scores
        return edge_index_pruned, incidence_weights, scores

class LearnableEdgeMask(nn.Module):
    """EHGNN-F incidence scorer and selected-only straight-through sampler."""

    def __init__(self, num_incidences, args):
        super().__init__()
        self.logits = nn.Parameter(torch.empty(num_incidences), requires_grad=True)
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.logits,
            mean=float(getattr(self.args, "mask_init_mean", 0.0)),
            std=float(getattr(self.args, "mask_init_std", 1.0)),
        )

    def forward(self, data, keep_ratio=0.5, is_test=False, return_mask=False):
        vertex, hyperedge = data.edge_index
        weights = torch.sigmoid(self.logits)
        budget = max(1, int(keep_ratio * vertex.numel()))

        if is_test:
            selected = torch.topk(weights, k=budget).indices
        elif getattr(self.args, "sampling", "multinomial") == "multinomial":
            selected = torch.multinomial(weights, budget, replacement=False)
        else:
            selected = torch.topk(weights, k=budget).indices

        hard_mask = torch.zeros_like(weights)
        hard_mask[selected] = 1.0
        straight_through = hard_mask + (weights - weights.detach()) * hard_mask
        selected_index = torch.stack([vertex[selected], hyperedge[selected]], dim=0)
        selected_weights = straight_through[selected]
        if return_mask:
            return selected_index, selected_weights, hard_mask, weights
        return selected_index, selected_weights, weights


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

def fixed_self_loops(edge_index, num_nodes, num_original_hyperedges):
    """Append one immutable singleton hyperedge per node using a fixed ID range."""
    num_nodes = int(num_nodes.item()) if torch.is_tensor(num_nodes) else int(num_nodes)
    num_original_hyperedges = (
        int(num_original_hyperedges.item())
        if torch.is_tensor(num_original_hyperedges)
        else int(num_original_hyperedges)
    )
    if edge_index.numel() and int(edge_index[1].max()) >= num_original_hyperedges:
        raise ValueError(
            "Original hyperedge IDs overlap the fixed self-loop range"
        )
    device = edge_index.device
    self_loop_nodes = torch.arange(num_nodes, device=device)
    self_loop_edges = torch.arange(
        num_original_hyperedges,
        num_original_hyperedges + num_nodes,
        device=device,
    )
    self_loops = torch.stack([self_loop_nodes, self_loop_edges], dim=0)
    return torch.cat([edge_index, self_loops], dim=1).long()


def add_fixed_self_loops_incidence(
    edge_index,
    num_nodes,
    num_original_hyperedges,
    incidence_weight,
):
    """Append fixed self-loops and preserve incidence-aligned learned weights."""
    if incidence_weight.numel() != edge_index.size(1):
        raise ValueError(
            "incidence_weight must align with edge_index columns: "
            f"got {incidence_weight.numel()} weights for {edge_index.size(1)} incidences"
        )
    edge_index_new = fixed_self_loops(
        edge_index, num_nodes, num_original_hyperedges
    )
    self_loop_count = edge_index_new.size(1) - edge_index.size(1)
    self_loop_weights = torch.ones(
        self_loop_count,
        dtype=incidence_weight.dtype,
        device=incidence_weight.device,
    )
    return edge_index_new, torch.cat([incidence_weight, self_loop_weights])


class HCHA(nn.Module):
    """HGNN backbone with the incidence scorers evaluated in the paper."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.mode == "learnmask":
            self.mask_module = LearnableEdgeMask(args.num_incidences, args=args)
        elif args.mode == "learnmask_cond":
            self.mask_module = FeatureConditionedIncidenceMask(
                args.F,
                hidden_dim=args.coarse_MLP,
                args=args,
            )
        elif args.mode == "NeuralF":
            self.mask_module = EHGNN_FcondLR(args, rank=args.low_rank)

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.symdegnorm = args.HCHA_symdegnorm
        self.incidence_mask_mode = args.mode in {
            "learnmask",
            "learnmask_cond",
            "NeuralF",
        }
        self.sparse_self_loop_policy = getattr(
            args,
            "sparse_self_loop_policy",
            "fixed_self_loops_implicit",
        )
        if self.sparse_self_loop_policy != "fixed_self_loops_implicit":
            raise ValueError(
                "The released LoG experiments use fixed implicit self-loops"
            )

        conv_cls = HypergraphConv if self.incidence_mask_mode else HypergraphConvOld
        self.convs = nn.ModuleList(
            [conv_cls(args.num_features, args.MLP_hidden, self.symdegnorm)]
        )
        for _ in range(self.num_layers - 2):
            self.convs.append(
                conv_cls(args.MLP_hidden, args.MLP_hidden, self.symdegnorm)
            )
        self.convs.append(
            conv_cls(args.MLP_hidden, args.num_classes, self.symdegnorm)
        )

    def reset_parameters(self):
        if hasattr(self, "mask_module"):
            self.mask_module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, is_test=False, return_mask=False):
        if self.incidence_mask_mode:
            selected = self.mask_module(
                data,
                keep_ratio=self.args.keep_ratio,
                is_test=is_test,
                return_mask=return_mask,
            )
            if return_mask:
                edge_index, weights, hard_mask, scores = selected
            else:
                edge_index, weights, scores = selected
        else:
            edge_index = data.edge_index
            weights = None

        self.last_forward_incidence_count = int(edge_index.size(1)) + int(data.n_x)
        self.last_fixed_self_loop_count = int(data.n_x)
        x = data.x
        for conv in self.convs[:-1]:
            if self.incidence_mask_mode:
                if self.training:
                    x = F.elu(
                        checkpoint(
                            lambda features, incidence_weights, layer=conv: layer(
                                features,
                                edge_index,
                                inc_weight=incidence_weights,
                                fixed_self_loops=True,
                            ),
                            x,
                            weights,
                            use_reentrant=False,
                        )
                    )
                else:
                    x = F.elu(
                        conv(
                            x,
                            edge_index,
                            inc_weight=weights,
                            fixed_self_loops=True,
                        )
                    )
            else:
                x = F.elu(conv(x, edge_index, fixed_self_loops=True))
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.incidence_mask_mode and self.training and not is_test:
            final_conv = self.convs[-1]
            x = checkpoint(
                lambda features: final_conv(
                    features,
                    edge_index,
                    fixed_self_loops=True,
                ),
                x,
                use_reentrant=False,
            )
        else:
            x = self.convs[-1](x, edge_index, fixed_self_loops=True)

        if self.incidence_mask_mode and not is_test:
            if return_mask:
                return x, scores, hard_mask
            return x, scores
        return x


class SetGNN(nn.Module):
    """AllSetTransformer components used by ``backbone_study.py``."""

    def __init__(self, args, norm_size=None):
        super().__init__()
        del norm_size
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()

        for layer_index in range(args.All_num_layers):
            input_dim = args.num_features if layer_index == 0 else args.MLP_hidden
            self.V2EConvs.append(
                HalfNLHconv(
                    in_dim=input_dim,
                    hid_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=args.dropout,
                    Normalization=args.normalization,
                    InputNorm=args.deepset_input_norm,
                    heads=args.heads,
                    attention=args.PMA,
                )
            )
            self.E2VConvs.append(
                HalfNLHconv(
                    in_dim=args.MLP_hidden,
                    hid_dim=args.MLP_hidden,
                    out_dim=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=args.dropout,
                    Normalization=args.normalization,
                    InputNorm=args.deepset_input_norm,
                    heads=args.heads,
                    attention=args.PMA,
                )
            )

        classifier_input = (
            args.num_features if args.All_num_layers == 0 else args.MLP_hidden
        )
        self.classifier = MLP(
            in_channels=classifier_input,
            hidden_channels=args.Classifier_hidden,
            out_channels=args.num_classes,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        self.classifier.reset_parameters()


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
