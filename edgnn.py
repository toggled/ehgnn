"""ED-HNN components used by the backbone adaptability experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from layers import MLP


class EquivSetConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        mlp1_layers=1,
        mlp2_layers=1,
        mlp3_layers=1,
        aggr="add",
        alpha=0.5,
        dropout=0.0,
        normalization="None",
        input_norm=False,
    ):
        super().__init__()
        self.W1 = (
            MLP(
                in_features,
                out_features,
                out_features,
                mlp1_layers,
                dropout=dropout,
                Normalization=normalization,
                InputNorm=input_norm,
            )
            if mlp1_layers > 0
            else nn.Identity()
        )
        self.W2 = (
            MLP(
                in_features + out_features,
                out_features,
                out_features,
                mlp2_layers,
                dropout=dropout,
                Normalization=normalization,
                InputNorm=input_norm,
            )
            if mlp2_layers > 0
            else lambda features: features[..., in_features:]
        )
        self.W = (
            MLP(
                out_features,
                out_features,
                out_features,
                mlp3_layers,
                dropout=dropout,
                Normalization=normalization,
                InputNorm=input_norm,
            )
            if mlp3_layers > 0
            else nn.Identity()
        )
        self.aggr = aggr
        self.alpha = alpha

    def reset_parameters(self):
        for module in (self.W1, self.W2, self.W):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x, vertex, edges, x0, incidence_weight=None):
        num_nodes = x.shape[-2]
        node_messages = self.W1(x)[..., vertex, :]

        if incidence_weight is None:
            edge_features = torch_scatter.scatter(
                node_messages, edges, dim=-2, reduce=self.aggr
            )
        else:
            weight = incidence_weight.to(dtype=node_messages.dtype)
            edge_features = torch_scatter.scatter(
                node_messages * weight[..., None], edges, dim=-2, reduce="sum"
            )
            if self.aggr == "mean":
                edge_degree = torch_scatter.scatter(
                    weight, edges, dim=-1, reduce="sum"
                ).clamp_min(1e-12)
                edge_features = edge_features / edge_degree[..., None]

        edge_messages = self.W2(
            torch.cat([x[..., vertex, :], edge_features[..., edges, :]], dim=-1)
        )
        if incidence_weight is None:
            updated = torch_scatter.scatter(
                edge_messages,
                vertex,
                dim=-2,
                reduce=self.aggr,
                dim_size=num_nodes,
            )
        else:
            updated = torch_scatter.scatter(
                edge_messages * weight[..., None],
                vertex,
                dim=-2,
                reduce="sum",
                dim_size=num_nodes,
            )
            if self.aggr == "mean":
                node_degree = torch_scatter.scatter(
                    weight,
                    vertex,
                    dim=-1,
                    reduce="sum",
                    dim_size=num_nodes,
                ).clamp_min(1e-12)
                updated = updated / node_degree[..., None]

        return self.W((1.0 - self.alpha) * updated + self.alpha * x0)


class EquivSetGNN(nn.Module):
    """Minimal ED-HNN backbone used by ``backbone_study.py``."""

    def __init__(self, num_features, num_classes, args):
        super().__init__()
        activations = {"Id": nn.Identity(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        self.act = activations[args.activation]
        self.dropout = nn.Dropout(args.dropout)
        self.nlayer = args.All_num_layers
        mlp2_layers = (
            args.MLP_num_layers
            if args.MLP2_num_layers < 0
            else args.MLP2_num_layers
        )
        mlp3_layers = (
            args.MLP_num_layers
            if args.MLP3_num_layers < 0
            else args.MLP3_num_layers
        )
        self.lin_in = nn.Linear(num_features, args.MLP_hidden)
        self.conv = EquivSetConv(
            args.MLP_hidden,
            args.MLP_hidden,
            mlp1_layers=args.MLP_num_layers,
            mlp2_layers=mlp2_layers,
            mlp3_layers=mlp3_layers,
            alpha=args.restart_alpha,
            aggr=args.aggregate,
            dropout=args.dropout,
            normalization=args.normalization,
            input_norm=True,
        )
        self.classifier = MLP(
            in_channels=args.MLP_hidden,
            hidden_channels=args.Classifier_hidden,
            out_channels=num_classes,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False,
        )

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data):
        vertex, edges = data.edge_index
        x = F.relu(self.lin_in(self.dropout(data.x)))
        x0 = x
        for _ in range(self.nlayer):
            x = self.act(self.conv(self.dropout(x), vertex, edges, x0))
        return self.classifier(self.dropout(x))
