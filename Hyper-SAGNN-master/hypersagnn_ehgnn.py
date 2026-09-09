import math
from typing import Optional, Tuple

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

