import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_non_pad_mask(seq: torch.Tensor) -> torch.Tensor:
    if seq.dim() != 2:
        raise ValueError(f"Expected [B, L], got {tuple(seq.shape)}")
    return seq.ne(0).float().unsqueeze(-1)


def get_attn_key_pad_mask(seq_k: torch.Tensor, seq_q: torch.Tensor) -> torch.Tensor:
    if seq_k.dim() != 2 or seq_q.dim() != 2:
        raise ValueError("seq_k and seq_q must both be rank-2 tensors.")
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp(min=1e-12)
    return (x * mask).sum(dim=dim) / denom


def index_mean(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, values.size(-1)), device=values.device, dtype=values.dtype)
    cnt = torch.zeros((dim_size, 1), device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    cnt.index_add_(0, index, torch.ones((values.size(0), 1), device=values.device, dtype=values.dtype))
    return out / cnt.clamp_min(1.0)


@dataclass
class BatchHypergraphData:
    x: torch.Tensor
    V_idx: torch.Tensor
    E_idx: torch.Tensor
    num_nodes: int
    num_hyperedges: int
    edge_ids: Optional[torch.Tensor]
    token_valid: torch.Tensor
    inv_node: torch.Tensor


def build_batch_hypergraph(x: torch.Tensor, token_emb: torch.Tensor, edge_ids: Optional[torch.Tensor] = None) -> BatchHypergraphData:
    B, L, _ = token_emb.shape
    device = token_emb.device
    valid = x.ne(0)
    flat_x = x[valid]
    flat_e = torch.arange(B, device=device).unsqueeze(1).expand(B, L)[valid]
    uniq_nodes, inv_node = torch.unique(flat_x, sorted=True, return_inverse=True)
    node_feats = index_mean(token_emb[valid], inv_node, uniq_nodes.numel())
    return BatchHypergraphData(
        x=node_feats,
        V_idx=inv_node,
        E_idx=flat_e,
        num_nodes=uniq_nodes.numel(),
        num_hyperedges=B,
        edge_ids=edge_ids,
        token_valid=valid,
        inv_node=inv_node,
    )


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        reshape=False,
        use_bias=True,
        residual=False,
        layer_norm=False,
    ):
        super().__init__()
        self.w_stack = []
        self.dims = dims
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=use_bias))
            self.add_module(f"PWF_Conv{i}", self.w_stack[-1])
        self.reshape = reshape
        self.layer_norm = nn.LayerNorm(dims[-1])
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.residual = residual
        self.layer_norm_flag = layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.transpose(1, 2)
        for i in range(len(self.w_stack) - 1):
            output = torch.tanh(self.w_stack[i](output))
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output).transpose(1, 2)

        if self.reshape:
            output = output.view(output.shape[0], -1, 1)

        if self.dims[0] == self.dims[-1]:
            if self.residual:
                output = output + x
            if self.layer_norm_flag:
                output = self.layer_norm(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
        super().__init__()
        self.w_stack = []
        for i in range(len(dims) - 1):
            self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
            self.add_module(f"FF_Linear{i}", self.w_stack[-1])
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.reshape = reshape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        for i in range(len(self.w_stack) - 1):
            output = torch.tanh(self.w_stack[i](output))
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, diag_mask, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -float("inf"))

        masked = attn.masked_fill((diag_mask == 0), -1e32)
        attn = torch.softmax(masked, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout, diag_mask, input_dim):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.diag_mask_flag = diag_mask

        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))

        self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
        self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self.layer_norm1(q)
        k = self.layer_norm2(k)
        v = self.layer_norm3(v)

        sz_b, len_q, _ = q.shape
        _, len_k, _ = k.shape
        _, len_v, _ = v.shape

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        n = sz_b * n_head
        diag = torch.ones((len_v, len_v), device=v.device)
        if self.diag_mask_flag == "True":
            diag = diag - torch.eye(len_v, len_v, device=v.device)
        diag = diag.repeat(n, 1, 1)
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        dynamic, attn = self.attention(q, k, v, diag, mask=mask)
        dynamic = dynamic.view(n_head, sz_b, len_q, d_v).permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        static = v.view(n_head, sz_b, len_q, d_v).permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        dynamic = self.fc1(dynamic)
        static = self.fc2(static)
        if self.dropout is not None:
            dynamic = self.dropout(dynamic)
            static = self.dropout(static)
        return dynamic, static, attn


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck):
        super().__init__()
        self.mul_head_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout_mul,
            diag_mask=diag_mask,
            input_dim=bottle_neck,
        )
        self.pff_n1 = PositionwiseFeedForward([d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
        self.pff_n2 = PositionwiseFeedForward([bottle_neck, d_model, d_model], dropout=dropout_pff, residual=False, layer_norm=True)

    def forward(self, dynamic, static, slf_attn_mask, non_pad_mask):
        dynamic, static1, attn = self.mul_head_attn(dynamic, dynamic, static, mask=slf_attn_mask)
        dynamic = self.pff_n1(dynamic * non_pad_mask) * non_pad_mask
        static1 = self.pff_n2(static * non_pad_mask) * non_pad_mask
        return dynamic, static1, attn


class HyperSAGNNEdgeClassifier(nn.Module):
    """
    Hyper-SAGNN-style edge encoder + multiclass edge head.
    """

    def __init__(
        self,
        node_embedding: nn.Module,
        d_model: int,
        num_classes: int,
        n_head: int = 8,
        d_k: int = 16,
        d_v: int = 16,
        diag_mask: str = "True",
        bottle_neck: Optional[int] = None,
    ):
        super().__init__()
        self.node_embedding = node_embedding
        self.diag_mask_flag = diag_mask
        self.bottle_neck = d_model if bottle_neck is None else bottle_neck

        self.encode1 = EncoderLayer(
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout_mul=0.3,
            dropout_pff=0.4,
            diag_mask=diag_mask,
            bottle_neck=self.bottle_neck,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        # Match Modules.py Classifier head semantics.
        self.pff_classifier = PositionwiseFeedForward([d_model, 1], reshape=True, use_bias=True)

    def _node_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.node_embedding(x.reshape(-1))
        if isinstance(out, tuple):
            node_emb, recon = out
        else:
            node_emb, recon = out, torch.zeros((), device=x.device)
        return node_emb, recon

    def get_token_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.long()
        B, L = x.shape
        node_emb, recon_loss = self._node_embed(x)
        return node_emb.view(B, L, -1), recon_loss

    def encode_from_tokens(self, token_emb: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.long()
        slf_attn_mask = get_attn_key_pad_mask(x, x)
        non_pad_mask = get_non_pad_mask(x)

        dynamic, static, _ = self.encode1(token_emb, token_emb, slf_attn_mask, non_pad_mask)
        dynamic = self.layer_norm1(dynamic)
        static = self.layer_norm2(static)

        if self.diag_mask_flag == "True":
            token_feat = (dynamic - static) ** 2
        else:
            token_feat = dynamic

        # Keep this pooled feature for pruner scoring.
        edge_feat = masked_mean(token_feat, non_pad_mask, dim=1)

        # Match Modules.py "mode = sum" branch:
        # token score -> sigmoid -> masked mean over tokens.
        output = self.pff_classifier(token_feat)
        output = torch.sigmoid(output)
        output = torch.sum(output * non_pad_mask, dim=-2, keepdim=False)
        mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
        output = output / mask_sum
        return output, edge_feat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_emb, recon_loss = self.get_token_embeddings(x)
        logits, edge_feat = self.encode_from_tokens(token_emb, x)
        return logits, edge_feat, recon_loss


class EHGNNCHardEdgePruner(nn.Module):
    """
    EHGNN-C style hard pruning over hyperedges.
    - Scores edges.
    - Selects top-k edges (or multinomial sample during train).
    - Uses STE mask: soft=(hard-prob).detach()+prob.
    """

    def __init__(self, edge_feat_dim: int, rank: int = 16, num_edges: Optional[int] = None, reg: str = "none"):
        super().__init__()
        self.reg = reg
        self.node_proj = nn.Linear(edge_feat_dim, rank, bias=False)
        self.edge_factors = nn.Parameter(torch.randn(num_edges, rank)) if num_edges is not None else None
        self.fallback_scorer = nn.Linear(rank, 1) if num_edges is None else None

    def score_edges(self, edge_feat: torch.Tensor, edge_ids: Optional[torch.Tensor]) -> torch.Tensor:
        u = self.node_proj(edge_feat)
        if self.edge_factors is not None:
            if edge_ids is None:
                raise ValueError("edge_ids must be provided when num_edges is fixed.")
            w = self.edge_factors[edge_ids]
            logits = (u * w).sum(dim=-1)
        else:
            logits = self.fallback_scorer(u).squeeze(-1)
        return torch.sigmoid(logits)

    def forward(
        self,
        edge_feat: torch.Tensor,
        keep_ratio: float = 0.5,
        edge_ids: Optional[torch.Tensor] = None,
        is_test: bool = False,
    ) -> Dict[str, torch.Tensor]:
        probs = self.score_edges(edge_feat, edge_ids)
        m = probs.numel()
        k = max(1, int(keep_ratio * m))
        k = min(k, m)

        if is_test:
            keep_ids = torch.topk(probs, k=k, largest=True).indices
        else:
            if self.reg != "none":
                raise ValueError(f"Unknown regularizer type: {self.reg}")
            # sample_w = probs.clamp_min(1e-8)
            # sample_w = sample_w / sample_w.sum()
            keep_ids = torch.multinomial(probs, k, replacement=False)

        hard = torch.zeros_like(probs)
        hard[keep_ids] = 1.0
        soft = (hard - probs).detach() + probs
        keep_mask = soft > 0.0

        return {
            "probs": probs,
            "hard_mask": hard,
            "soft_mask": soft,
            "keep_mask": keep_mask,
            "keep_ids": keep_ids,
        }


class RandomEdgeMask(nn.Module):
    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        m = data.num_hyperedges
        k = max(1, int(keep_ratio * m))
        probs = torch.ones(m, device=data.x.device) * keep_ratio
        keep_e = torch.randperm(m, device=data.x.device)[:k]
        hard = torch.zeros_like(probs)
        hard[keep_e] = 1.0
        soft = (hard - probs).detach() + probs
        return {"edge_probs": probs, "edge_soft": soft, "edge_hard": hard}


class DegreeDistributionEdgeMask(nn.Module):
    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        m = data.num_hyperedges
        edge_deg = torch.bincount(data.E_idx, minlength=m).float()
        probs = edge_deg / edge_deg.sum().clamp_min(1e-12)
        k = max(1, int(keep_ratio * m))
        keep_e = torch.topk(probs, k=k).indices if is_test else torch.multinomial(probs, k, replacement=False)
        hard = torch.zeros(m, device=data.x.device)
        hard[keep_e] = 1.0
        soft = (hard - probs).detach() + probs
        return {"edge_probs": probs, "edge_soft": soft, "edge_hard": hard}


class EffectiveResistanceEdgeMask(nn.Module):
    def __init__(self):
        super().__init__()
        print('Spectral')
    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        device = data.x.device
        n, m = data.num_nodes, data.num_hyperedges
        H = torch.zeros((n, m), device=device, dtype=data.x.dtype)
        H[data.V_idx, data.E_idx] = 1.0
        Dv = torch.diag(H.sum(dim=1))
        De_inv = torch.diag(1.0 / (H.sum(dim=0) + 1e-8))
        L = Dv - H @ De_inv @ H.T
        try:
            diag = torch.linalg.pinv(L + 1e-3 * torch.eye(n, device=device)).diag()
        except RuntimeError:
            diag = torch.ones(n, device=device)
        edge_scores = torch.zeros(m, device=device)
        for e_id in range(m):
            nodes_in_e = data.V_idx[data.E_idx == e_id]
            if nodes_in_e.numel() > 1:
                edge_scores[e_id] = diag[nodes_in_e].sum()
        probs = edge_scores
        k = max(1, int(keep_ratio * m))
        keep_e = torch.topk(probs, k=k).indices if is_test else torch.multinomial(probs, k, replacement=False)
        hard = torch.zeros_like(probs)
        hard[keep_e] = 1.0
        soft = (hard - probs).detach() + probs
        return {"edge_probs": probs, "edge_soft": soft, "edge_hard": hard}


class LearnableIncidenceMask(nn.Module):
    def __init__(self, num_edges_total: int, max_edge_size: int, reg: str = "none"):
        super().__init__()
        self.logits = nn.Embedding(num_edges_total, max_edge_size)
        nn.init.normal_(self.logits.weight, mean=0.0, std=0.02)
        self.reg = reg

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        if data.edge_ids is None:
            raise ValueError("learnmask requires edge_ids.")
        B = data.num_hyperedges
        max_len = self.logits.embedding_dim
        local_pos = torch.arange(max_len, device=data.x.device).unsqueeze(0).expand(B, max_len)
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
        edge_soft = torch.zeros(B, device=data.x.device)
        edge_cnt = torch.zeros(B, device=data.x.device)
        edge_soft.index_add_(0, data.E_idx, soft)
        edge_cnt.index_add_(0, data.E_idx, torch.ones_like(soft))
        edge_soft = edge_soft / edge_cnt.clamp_min(1.0)
        edge_hard = (edge_soft > 0).float()
        return {
            "edge_probs": edge_soft.detach(),
            "edge_soft": edge_soft,
            "edge_hard": edge_hard,
            "inc_probs": probs,
            "inc_soft": soft,
        }


class FeatureConditionedIncidenceMask(nn.Module):
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
        edge_soft = torch.zeros(data.num_hyperedges, device=data.x.device)
        edge_cnt = torch.zeros(data.num_hyperedges, device=data.x.device)
        edge_soft.index_add_(0, data.E_idx, soft)
        edge_cnt.index_add_(0, data.E_idx, torch.ones_like(soft))
        edge_soft = edge_soft / edge_cnt.clamp_min(1.0)
        edge_hard = (edge_soft > 0).float()
        return {
            "edge_probs": edge_soft.detach(),
            "edge_soft": edge_soft,
            "edge_hard": edge_hard,
            "inc_probs": probs,
            "inc_soft": soft,
        }


class LearnableEdgeMaskplus(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, reg: str = "none"):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.reg = reg

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        edge_feats = edge_feat_override if edge_feat_override is not None else index_mean(data.x[data.V_idx], data.E_idx, data.num_hyperedges)
        probs = torch.sigmoid(self.scorer(edge_feats).squeeze(-1))
        m = probs.numel()
        k = max(1, int(keep_ratio * m))
        keep_e = torch.topk(probs, k=k).indices if is_test else torch.multinomial(probs, k, replacement=False)
        hard = torch.zeros_like(probs)
        hard[keep_e] = 1.0
        soft = (hard - probs).detach() + probs
        return {"edge_probs": probs, "edge_soft": soft, "edge_hard": hard}


class FeatureAgnosticEdgeMask(nn.Module):
    def __init__(self, num_edges_total: int, reg: str = "none"):
        super().__init__()
        self.logits = nn.Embedding(num_edges_total, 1)
        nn.init.zeros_(self.logits.weight)
        self.reg = reg

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        if data.edge_ids is None:
            raise ValueError("learnmask+_agn requires edge_ids.")
        probs = torch.sigmoid(self.logits(data.edge_ids).view(-1))
        m = probs.numel()
        k = max(1, int(keep_ratio * m))
        keep_e = torch.topk(probs, k=k).indices if is_test else torch.multinomial(probs, k, replacement=False)
        hard = torch.zeros_like(probs)
        hard[keep_e] = 1.0
        soft = (hard - probs).detach() + probs
        return {"edge_probs": probs, "edge_soft": soft, "edge_hard": hard}


class NeuralEdgeMask(nn.Module):
    """
    Adapter that exposes EHGNNCHardEdgePruner through the common sparsifier API.
    """

    def __init__(self, edge_feat_dim: int, rank: int = 16, num_edges_total: Optional[int] = None, reg: str = "none"):
        super().__init__()
        self.pruner = EHGNNCHardEdgePruner(edge_feat_dim=edge_feat_dim, rank=rank, num_edges=num_edges_total, reg=reg)

    def forward(self, data: BatchHypergraphData, keep_ratio=0.5, is_test=False, edge_feat_override: Optional[torch.Tensor] = None):
        edge_feats = edge_feat_override if edge_feat_override is not None else index_mean(data.x[data.V_idx], data.E_idx, data.num_hyperedges)
        edge_ids = data.edge_ids if self.pruner.edge_factors is not None else None
        out = self.pruner(edge_feat=edge_feats, keep_ratio=keep_ratio, edge_ids=edge_ids, is_test=is_test)
        return {
            "edge_probs": out["probs"],
            "edge_soft": out["soft_mask"],
            "edge_hard": out["hard_mask"],
        }


class NeuralIncidenceMask(nn.Module):
    """
    EHGNN-F(cond,LR): low-rank feature-conditioned incidence scorer.
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

        edge_probs = torch.zeros(data.num_hyperedges, device=data.x.device)
        edge_soft = torch.zeros(data.num_hyperedges, device=data.x.device)
        edge_cnt = torch.zeros(data.num_hyperedges, device=data.x.device)
        edge_probs.index_add_(0, data.E_idx, scores.detach())
        edge_soft.index_add_(0, data.E_idx, soft)
        edge_cnt.index_add_(0, data.E_idx, torch.ones_like(soft))
        edge_probs = edge_probs / edge_cnt.clamp_min(1.0)
        edge_soft = edge_soft / edge_cnt.clamp_min(1.0)
        edge_hard = (edge_soft > 0).float()

        return {
            "edge_probs": edge_probs,
            "edge_soft": edge_soft,
            "edge_hard": edge_hard,
            "inc_probs": scores,
            "inc_soft": soft,
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
        self.edge_level = self.mode in {"random", "degdist", "effresist", "learnmask+", "learnmask+_agn", "Neural"}
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
        return self.impl(data, keep_ratio=keep_ratio, is_test=is_test, edge_feat_override=edge_feat_override)


class HyperSAGNNSparse(nn.Module):
    """
    Hyper-SAGNN with pluggable sparsifier baselines used to build masked context.
    Loss/metrics are computed on all candidate edges.
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

    def _masked_hypergraph_context(self, x: torch.Tensor, token_emb: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = token_emb.shape
        device = token_emb.device
        valid = x.ne(0)
        if not valid.any():
            return token_emb
        flat_x = x[valid]
        flat_h = token_emb[valid]
        flat_e = torch.arange(B, device=device).unsqueeze(1).expand(B, L)[valid]
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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, edge_ids: Optional[torch.Tensor] = None, is_test: bool = False):
        token_h0, recon_loss = self.backbone.get_token_embeddings(x)
        batch_data = build_batch_hypergraph(x.long(), token_h0, edge_ids=edge_ids)
        sparse_out = self.sparsifier(batch_data, keep_ratio=self.keep_ratio, is_test=is_test)
        edge_soft = sparse_out["edge_soft"]
        edge_hard = sparse_out["edge_hard"]
        token_h1 = self._masked_hypergraph_context(x.long(), token_h0, edge_soft)
        logits_all, edge_feat = self.backbone.encode_from_tokens(token_h1, x)
        keep_mask = edge_hard > 0
        out = {
            "logits_all": logits_all,
            "logits_kept": logits_all,
            "keep_mask": keep_mask,
            "probs": sparse_out["edge_probs"],
            "edge_mask_soft": edge_soft,
            "recon_loss": recon_loss,
        }
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
    Sparse Hyper-SAGNN variant that scores masks from a richer pre-context
    Hyper-SAGNN edge representation instead of a pooled base embedding.
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
        edge_soft = sparse_out["edge_soft"]
        edge_hard = sparse_out["edge_hard"]
        token_h1 = self._masked_hypergraph_context(x.long(), token_h0, edge_soft)
        logits_all, _ = self.backbone.encode_from_tokens(token_h1, x)
        keep_mask = edge_hard > 0
        out = {
            "logits_all": logits_all,
            "logits_kept": logits_all,
            "keep_mask": keep_mask,
            "probs": sparse_out["edge_probs"],
            "edge_mask_soft": edge_soft,
            "recon_loss": recon_loss,
            "edge_feat_pre": edge_feat_pre,
        }
        if y is not None:
            if logits_all.dim() == 2 and logits_all.size(-1) == 1:
                cls_loss = F.binary_cross_entropy(logits_all, y.float().view(-1, 1))
            else:
                cls_loss = F.cross_entropy(logits_all, y)
            out["y_kept"] = y
            out["loss_cls"] = cls_loss
            out["loss"] = cls_loss + self.recon_weight * recon_loss
        return out


@torch.no_grad()
def multiclass_accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean()
