from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypersagnn_ehgnn import (
    BatchHypergraphData,
    DegreeDistributionEdgeMask,
    EffectiveResistanceEdgeMask,
    EHGNNCHardEdgePruner,
    FeatureAgnosticEdgeMask,
    HyperSAGNNEdgeClassifier,
    LearnableEdgeMaskplus,
    RandomEdgeMask,
    build_batch_hypergraph,
    index_mean,
    multiclass_accuracy,
)


def reduce_incidence_to_edge(data: BatchHypergraphData, inc_values: torch.Tensor) -> torch.Tensor:
    edge_values = torch.zeros(data.num_hyperedges, device=inc_values.device, dtype=inc_values.dtype)
    edge_count = torch.zeros(data.num_hyperedges, device=inc_values.device, dtype=inc_values.dtype)
    edge_values.index_add_(0, data.E_idx, inc_values)
    edge_count.index_add_(0, data.E_idx, torch.ones_like(inc_values))
    return edge_values / edge_count.clamp_min(1.0)


class LearnableIncidenceMask(nn.Module):
    """
    Incidence-level learnable sparsifier.
    Unlike v1, incidence masks remain incidence-level all the way into
    message passing instead of being collapsed before contextualization.
    """

    def __init__(self, num_edges_total: int, max_edge_size: int, reg: str = "none"):
        super().__init__()
        self.logits = nn.Embedding(num_edges_total, max_edge_size)
        nn.init.normal_(self.logits.weight, mean=0.0, std=0.02)
        self.reg = reg

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        if data.edge_ids is None:
            raise ValueError("learnmask requires edge_ids.")
        valid = data.token_valid
        logits_full = self.logits(data.edge_ids)
        logits = logits_full[valid]
        probs = torch.sigmoid(logits)
        nnz = probs.numel()
        k = max(1, int(keep_ratio * nnz))
        keep_ids = torch.topk(probs, k=k).indices if is_test else torch.multinomial(probs, k, replacement=False)
        hard = torch.zeros_like(probs)
        hard[keep_ids] = 1.0
        soft = (hard - probs).detach() + probs
        edge_probs = reduce_incidence_to_edge(data, probs.detach())
        edge_soft = reduce_incidence_to_edge(data, soft)
        edge_hard = (reduce_incidence_to_edge(data, hard) > 0).float()
        return {
            "mask_level": "incidence",
            "inc_probs": probs,
            "inc_soft": soft,
            "inc_hard": hard,
            "edge_probs": edge_probs,
            "edge_soft": edge_soft,
            "edge_hard": edge_hard,
        }


class FeatureConditionedIncidenceMask(nn.Module):
    """
    Feature-conditioned incidence scorer that remains incidence-level
    through the contextualization stage.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, reg: str = "none"):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(2 * input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.reg = reg

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        edge_feats = edge_feat_override if edge_feat_override is not None else index_mean(data.x[data.V_idx], data.E_idx, data.num_hyperedges)
        pair = torch.cat([data.x[data.V_idx], edge_feats[data.E_idx]], dim=-1)
        probs = torch.sigmoid(self.scorer(pair).squeeze(-1))
        nnz = probs.numel()
        k = max(1, int(keep_ratio * nnz))
        keep_ids = torch.topk(probs, k=k).indices if is_test else torch.multinomial(probs, k, replacement=False)
        hard = torch.zeros_like(probs)
        hard[keep_ids] = 1.0
        soft = (hard - probs).detach() + probs
        edge_probs = reduce_incidence_to_edge(data, probs.detach())
        edge_soft = reduce_incidence_to_edge(data, soft)
        edge_hard = (reduce_incidence_to_edge(data, hard) > 0).float()
        return {
            "mask_level": "incidence",
            "inc_probs": probs,
            "inc_soft": soft,
            "inc_hard": hard,
            "edge_probs": edge_probs,
            "edge_soft": edge_soft,
            "edge_hard": edge_hard,
        }


class NeuralEdgeMask(nn.Module):
    def __init__(self, edge_feat_dim: int, rank: int = 16, num_edges_total: Optional[int] = None, reg: str = "none"):
        super().__init__()
        self.pruner = EHGNNCHardEdgePruner(edge_feat_dim=edge_feat_dim, rank=rank, num_edges=num_edges_total, reg=reg)

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        edge_feats = edge_feat_override if edge_feat_override is not None else index_mean(data.x[data.V_idx], data.E_idx, data.num_hyperedges)
        edge_ids = data.edge_ids if self.pruner.edge_factors is not None else None
        out = self.pruner(edge_feat=edge_feats, keep_ratio=keep_ratio, edge_ids=edge_ids, is_test=is_test)
        return {
            "mask_level": "edge",
            "edge_probs": out["probs"],
            "edge_soft": out["soft_mask"],
            "edge_hard": out["hard_mask"],
        }


class NeuralIncidenceMask(nn.Module):
    """
    EHGNN-F(cond,LR): low-rank feature-conditioned incidence scorer that
    stays incidence-level through the v2 message-passing path.
    """

    def __init__(self, edge_feat_dim: int, rank: int = 16, num_edges_total: Optional[int] = None, reg: str = "none"):
        super().__init__()
        if num_edges_total is None:
            raise ValueError("NeuralF requires num_edges_total for edge-factor lookup.")
        self.rank = rank
        self.reg = reg
        self.node_proj = nn.Linear(edge_feat_dim, rank, bias=False)
        self.edge_factors = nn.Parameter(torch.randn(num_edges_total, rank))

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        if data.edge_ids is None:
            raise ValueError("NeuralF requires edge_ids.")

        edge_ids_per_inc = data.edge_ids[data.E_idx]
        logits = (self.node_proj(data.x)[data.V_idx] * self.edge_factors[edge_ids_per_inc]).sum(dim=-1)
        scores = torch.sigmoid(logits)
        nnz = scores.numel()
        k = max(1, int(keep_ratio * nnz))

        if is_test:
            keep_ids = torch.topk(scores, k=k).indices
        else:
            if self.reg == "none":
                keep_ids = torch.multinomial(scores, k, replacement=False)
            elif self.reg == "l2":
                keep_mask = torch.bernoulli(scores).bool()
                if not keep_mask.any():
                    keep_mask[scores.argmax()] = True
                keep_ids = keep_mask.nonzero(as_tuple=False).view(-1)
            else:
                raise ValueError(f"Unknown regularizer type: {self.reg}")

        hard = torch.zeros_like(scores)
        hard[keep_ids] = 1.0
        soft = hard + (scores - scores.detach()) * hard
        edge_probs = reduce_incidence_to_edge(data, scores.detach())
        edge_soft = reduce_incidence_to_edge(data, soft)
        edge_hard = (reduce_incidence_to_edge(data, hard) > 0).float()
        return {
            "mask_level": "incidence",
            "inc_probs": scores,
            "inc_soft": soft,
            "inc_hard": hard,
            "edge_probs": edge_probs,
            "edge_soft": edge_soft,
            "edge_hard": edge_hard,
        }


class HypergraphBatchSparsifier(nn.Module):
    def __init__(
        self,
        mode: str,
        edge_feat_dim: int,
        reg: str = "none",
        rank: int = 16,
        num_edges_total: Optional[int] = None,
        max_edge_size: Optional[int] = None,
        hidden_dim: int = 32,
    ):
        super().__init__()
        alias = {"learnmask_agn": "learnmask+_agn"}
        self.mode = alias.get(mode, mode)
        if self.mode == "random":
            self.impl = RandomEdgeMask()
        elif self.mode == "degdist":
            self.impl = DegreeDistributionEdgeMask()
        elif self.mode == "effresist":
            self.impl = EffectiveResistanceEdgeMask()
        elif self.mode == "Neural":
            self.impl = NeuralEdgeMask(edge_feat_dim=edge_feat_dim, rank=rank, num_edges_total=num_edges_total, reg=reg)
        elif self.mode == "NeuralF":
            self.impl = NeuralIncidenceMask(edge_feat_dim=edge_feat_dim, rank=rank, num_edges_total=num_edges_total, reg=reg)
        elif self.mode == "learnmask":
            self.impl = LearnableIncidenceMask(num_edges_total=num_edges_total, max_edge_size=max_edge_size, reg=reg)
        elif self.mode == "learnmask_cond":
            self.impl = FeatureConditionedIncidenceMask(input_dim=edge_feat_dim, hidden_dim=hidden_dim, reg=reg)
        elif self.mode == "learnmask+":
            self.impl = LearnableEdgeMaskplus(input_dim=edge_feat_dim, hidden_dim=hidden_dim, reg=reg)
        elif self.mode == "learnmask+_agn":
            self.impl = FeatureAgnosticEdgeMask(num_edges_total=num_edges_total, reg=reg)
        else:
            raise ValueError(f"Unsupported sparsifier mode: {mode}")

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        if edge_feat_override is None:
            out = self.impl(data, keep_ratio=keep_ratio, is_test=is_test)
        else:
            try:
                out = self.impl(data, keep_ratio=keep_ratio, is_test=is_test, edge_feat_override=edge_feat_override)
            except TypeError:
                out = self.impl(data, keep_ratio=keep_ratio, is_test=is_test)
        if "mask_level" not in out:
            out["mask_level"] = "edge"
        return out


class HyperSAGNNSparse(nn.Module):
    """
    V2 sparse Hyper-SAGNN that preserves the distinction between
    edge-level and incidence-level sparsifiers during message passing.
    """

    def __init__(
        self,
        backbone: HyperSAGNNEdgeClassifier,
        sparsifier: HypergraphBatchSparsifier,
        keep_ratio: float = 0.5,
        recon_weight: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.sparsifier = sparsifier
        self.keep_ratio = keep_ratio
        self.recon_weight = recon_weight
        self.context_proj = nn.Linear(backbone.bottle_neck, backbone.bottle_neck, bias=False)

    def _masked_hypergraph_context_edge(self, x: torch.Tensor, token_emb: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        B, _, D = token_emb.shape
        device = token_emb.device
        valid = x.ne(0)
        if not valid.any():
            return token_emb
        flat_x = x[valid]
        flat_h = token_emb[valid]
        flat_e = torch.arange(B, device=device).unsqueeze(1).expand_as(x)[valid]
        uniq_nodes, inv_node = torch.unique(flat_x, sorted=True, return_inverse=True)
        node_h0 = index_mean(flat_h, inv_node, uniq_nodes.numel())
        msg_in = self.context_proj(node_h0)[inv_node]
        edge_mean = index_mean(msg_in, flat_e, B)
        node_ctx = torch.zeros((uniq_nodes.numel(), D), device=device, dtype=flat_h.dtype)
        node_ctx.index_add_(0, inv_node, edge_mean[flat_e] * edge_mask[flat_e].unsqueeze(-1))
        node_h1 = node_h0 + node_ctx
        token_h1 = torch.zeros_like(token_emb)
        token_h1[valid] = node_h1[inv_node]
        return token_h1

    def _masked_hypergraph_context_incidence(
        self,
        x: torch.Tensor,
        token_emb: torch.Tensor,
        batch_data: BatchHypergraphData,
        inc_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = x.ne(0)
        if not valid.any():
            return token_emb

        node_h0 = batch_data.x
        msg_source = self.context_proj(node_h0)[batch_data.V_idx] * inc_mask.unsqueeze(-1)
        edge_sum = torch.zeros((batch_data.num_hyperedges, node_h0.size(-1)), device=token_emb.device, dtype=token_emb.dtype)
        edge_den = torch.zeros((batch_data.num_hyperedges, 1), device=token_emb.device, dtype=token_emb.dtype)
        edge_sum.index_add_(0, batch_data.E_idx, msg_source)
        edge_den.index_add_(0, batch_data.E_idx, inc_mask.unsqueeze(-1))
        edge_mean = edge_sum / edge_den.clamp_min(1.0)

        node_ctx = torch.zeros_like(node_h0)
        recv_msg = edge_mean[batch_data.E_idx] * inc_mask.unsqueeze(-1)
        node_ctx.index_add_(0, batch_data.V_idx, recv_msg)
        node_h1 = node_h0 + node_ctx

        token_h1 = torch.zeros_like(token_emb)
        token_h1[valid] = node_h1[batch_data.inv_node]
        return token_h1

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, edge_ids: Optional[torch.Tensor] = None, is_test: bool = False):
        token_h0, recon_loss = self.backbone.get_token_embeddings(x)
        batch_data = build_batch_hypergraph(x.long(), token_h0, edge_ids=edge_ids)
        sparse_out = self.sparsifier(batch_data, keep_ratio=self.keep_ratio, is_test=is_test)

        if sparse_out["mask_level"] == "incidence":
            token_h1 = self._masked_hypergraph_context_incidence(x.long(), token_h0, batch_data, sparse_out["inc_soft"])
        else:
            token_h1 = self._masked_hypergraph_context_edge(x.long(), token_h0, sparse_out["edge_soft"])

        logits_all, _ = self.backbone.encode_from_tokens(token_h1, x)
        out = {
            "logits_all": logits_all,
            "logits_kept": logits_all,
            "keep_mask": sparse_out["edge_hard"] > 0,
            "probs": sparse_out["edge_probs"],
            "edge_mask_soft": sparse_out["edge_soft"],
            "recon_loss": recon_loss,
        }
        if "inc_probs" in sparse_out:
            out["inc_probs"] = sparse_out["inc_probs"]
            out["inc_soft"] = sparse_out["inc_soft"]

        if y is not None:
            if logits_all.dim() == 2 and logits_all.size(-1) == 1:
                cls_loss = F.binary_cross_entropy(logits_all, y.float().view(-1, 1))
            else:
                cls_loss = F.cross_entropy(logits_all, y)
            out["y_kept"] = y
            out["loss_cls"] = cls_loss
            out["loss"] = cls_loss + self.recon_weight * recon_loss
        return out


class HyperSAGNNSparsePreEncMask(HyperSAGNNSparse):
    """
    V2 sparse Hyper-SAGNN variant that scores masks from a richer pre-context
    Hyper-SAGNN edge representation before masked message passing.
    """

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, edge_ids: Optional[torch.Tensor] = None, is_test: bool = False):
        token_h0, recon_loss = self.backbone.get_token_embeddings(x)
        _, edge_feat_pre = self.backbone.encode_from_tokens(token_h0, x)
        batch_data = build_batch_hypergraph(x.long(), token_h0, edge_ids=edge_ids)
        sparse_out = self.sparsifier(
            batch_data,
            keep_ratio=self.keep_ratio,
            is_test=is_test,
            edge_feat_override=edge_feat_pre,
        )

        if sparse_out["mask_level"] == "incidence":
            token_h1 = self._masked_hypergraph_context_incidence(x.long(), token_h0, batch_data, sparse_out["inc_soft"])
        else:
            token_h1 = self._masked_hypergraph_context_edge(x.long(), token_h0, sparse_out["edge_soft"])

        logits_all, _ = self.backbone.encode_from_tokens(token_h1, x)
        out = {
            "logits_all": logits_all,
            "logits_kept": logits_all,
            "keep_mask": sparse_out["edge_hard"] > 0,
            "probs": sparse_out["edge_probs"],
            "edge_mask_soft": sparse_out["edge_soft"],
            "recon_loss": recon_loss,
            "edge_feat_pre": edge_feat_pre,
        }
        if "inc_probs" in sparse_out:
            out["inc_probs"] = sparse_out["inc_probs"]
            out["inc_soft"] = sparse_out["inc_soft"]

        if y is not None:
            if logits_all.dim() == 2 and logits_all.size(-1) == 1:
                cls_loss = F.binary_cross_entropy(logits_all, y.float().view(-1, 1))
            else:
                cls_loss = F.cross_entropy(logits_all, y)
            out["y_kept"] = y
            out["loss_cls"] = cls_loss
            out["loss"] = cls_loss + self.recon_weight * recon_loss
        return out


__all__ = [
    "HyperSAGNNEdgeClassifier",
    "HypergraphBatchSparsifier",
    "HyperSAGNNSparse",
    "HyperSAGNNSparsePreEncMask",
    "multiclass_accuracy",
]
