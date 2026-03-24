import argparse
import csv
import importlib
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
variant_map = {
    "full": "Full",
    "random": "Random",
    "degdist": "EdgeDeg",
    "effresist": "Spectral",
    "learnmask": "EHGNN-F",       # EHGNN-F
    "learnmask_cond": "EHGNN-F(cond)",      # EHGNN-C
    "learnmask+": "EHGNN-C(cond)",       # EHGNN-F
    "learnmask+_agn": "EHGNN-C",      # EHGNN-C
    "Neural": "EHGNN-C(cond,LR)",
    "NeuralF": "EHGNN-F(cond,LR)"
}
from hypersagnn_ehgnn import (
    HyperSAGNNEdgeClassifier,
    HyperSAGNNSparse,
    HypergraphBatchSparsifier,
    multiclass_accuracy,
)


CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from Modules import MultipleEmbedding  # noqa: E402
from torchsummary import summary  # noqa: E402


def load_hypersagnn_adapter(impl: str):
    """ 
    hypersagnn_ehgnnv2.py:
    - Keeps edge-level sparsifiers edge-level, and 
    - Keeps incidence-level sparsifiers incidence-level through message passing.
    hypersagnn_ehgnn.py:
    - Incidence-level sparsifiers aggregate incidence scores to obtain edge-level scores. 
    """
    module_name = "hypersagnn_ehgnnv2" if impl == "v2" else "hypersagnn_ehgnn"
    return importlib.import_module(module_name)


def parse_args():
    parser = argparse.ArgumentParser("Trainer for Hyper-SAGNN baseline / Hyper-SAGNN+EHGNN-C")
    # Dataset / output paths.
    parser.add_argument("--data", type=str, default="ramani", help="Dataset name under --data-root to train/evaluate on.")
    parser.add_argument("--data-root", type=str, default="data", help="Root directory containing dataset subfolders.")
    parser.add_argument("--save-path", type=str, default="checkpoints/hypersagnn_ehgnn", help="Directory used to save model checkpoints.")
    parser.add_argument("--results-csv", type=str, default="checkpoints/hypersagnn_ehgnn/results_final.csv", help="CSV file where final run metrics are appended.")

    # Model family and sparsifier selection.
    parser.add_argument("--model", type=str, default="ehgnn", choices=["baseline", "ehgnn", "ehgnn_preenc"], help="Choose plain Hyper-SAGNN baseline, the current sparse EHGNN wrapper, or the pre-encoder mask-scoring sparse variant.")
    parser.add_argument(
        "--mode",
        type=str,
        default="learnmask+_agn",
        choices=["random", "degdist", "effresist", "learnmask", "learnmask+", "learnmask_cond", "learnmask+_agn", "learnmask_agn", "Neural", "NeuralF"],
        help="Sparsifier variant to use when --model ehgnn is selected.",
    )
    parser.add_argument("--feature", type=str, default="adj", choices=["adj", "learned"], help="Node feature source: adjacency-derived MultipleEmbedding or a plain learned embedding table.")
    parser.add_argument("--supervised", action="store_true", help="Use explicit class labels instead of generating negative edges for link prediction.")
    parser.add_argument("--hypersagnn-impl", type=str, default="v1", choices=["v1", "v2"], help="Select the Hyper-SAGNN sparsification implementation module to import.")
    parser.add_argument("--kept-only", action="store_true", help="For sparse models only, compute loss and metrics using kept edges only instead of all candidate edges.")

    # Optimization / runtime settings.
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size; use -1 to process the full split as one batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay applied by the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy, Python, and PyTorch.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g. cpu, cuda:0, cuda:1.")

    # Hyper-SAGNN backbone dimensions.
    parser.add_argument("--dimensions", type=int, default=64, help="Hidden dimension used by embeddings and the Hyper-SAGNN encoder.")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads in the Hyper-SAGNN encoder.")
    parser.add_argument("--d-k", type=int, default=16, help="Per-head key/query dimension.")
    parser.add_argument("--d-v", type=int, default=16, help="Per-head value dimension.")
    parser.add_argument("--diag", type=str, default="True", help="Whether to mask self-attention diagonal terms inside Hyper-SAGNN.")

    # Sparsifier and auxiliary-loss settings.
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Target fraction of edges/incidences kept by the sparsifier.")
    parser.add_argument("--rank", type=int, default=16, help="Low-rank size used by Neural / EHGNN-C-style mask scorers.")
    parser.add_argument("--recon-weight", type=float, default=0.01, help="Weight applied to the adjacency reconstruction loss from MultipleEmbedding.")
    parser.add_argument("--reg", type=str, default="none", help="Optional regularizer mode passed through to sparsifier implementations.")
    parser.add_argument("--coarse-mlp", type=int, default=32, help="Hidden width for feature-conditioned sparsifier MLPs.")

    # Supervised labels or unsupervised negative-sampling setup.
    parser.add_argument("--train-label-file", type=str, default="", help="Optional path to training labels for supervised mode.")
    parser.add_argument("--test-label-file", type=str, default="", help="Optional path to test labels for supervised mode.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes; updated automatically when supervised labels are loaded.")
    parser.add_argument("--neg-num", type=int, default=5, help="Number of negative edges generated per positive edge in unsupervised mode.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu":
        return torch.device("cpu")

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA unavailable; falling back to CPU.")
            return torch.device("cpu")
        try:
            dev = torch.device(device_arg)
            if dev.index is not None and dev.index >= torch.cuda.device_count():
                print(
                    f"Requested {device_arg}, but only {torch.cuda.device_count()} CUDA device(s) found. "
                    "Falling back to cuda:0."
                )
                return torch.device("cuda:0")
            return dev
        except Exception:
            print(f"Invalid CUDA device '{device_arg}'. Falling back to cuda:0.")
            return torch.device("cuda:0")

    print(f"Unknown device '{device_arg}'. Falling back to CPU.")
    return torch.device("cpu")


def add_padding_idx(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec) + 1
    vec = np.sort(vec, axis=-1)
    return vec.astype("int64")


def build_hash(data: np.ndarray) -> set:
    out = set()
    for datum in data:
        temp = np.sort(np.asarray(datum).astype(np.int64))
        out.add(tuple(temp.tolist()))
    return out


def maybe_load_labels(npz_obj, key_candidates: List[str]) -> Optional[np.ndarray]:
    for k in key_candidates:
        if k in npz_obj:
            arr = npz_obj[k]
            if arr.dtype == object:
                return None
            return np.asarray(arr)
    return None


def get_adjacency(data: np.ndarray, num: np.ndarray, num_list: np.ndarray, norm: bool = True):
    A = np.zeros((num_list[-1], num_list[-1]), dtype=np.float32)
    for datum in tqdm(data, desc="adj-build", leave=False):
        for i in range(datum.shape[-1]):
            for j in range(datum.shape[-1]):
                if i != j:
                    A[datum[i], datum[j]] += 1.0

    if norm:
        temp = np.concatenate((np.zeros((1), dtype=int), num), axis=0)
        temp = np.cumsum(temp)
        for i in range(len(temp) - 1):
            denom = np.max(A[temp[i] : temp[i + 1], :], axis=0, keepdims=True) + 1e-10
            A[temp[i] : temp[i + 1], :] /= denom
    return csr_matrix(A).astype("float32")


def generate_H(edge: np.ndarray, nums_type: np.ndarray, node_type_mapping: List[int], weight: np.ndarray):
    nums_examples = len(edge)
    H = [0 for _ in range(len(nums_type))]
    for i in range(edge.shape[-1]):
        t = node_type_mapping[i]
        H[t] += csr_matrix(
            (np.sqrt(weight), (edge[:, i], range(nums_examples))),
            shape=(nums_type[t], nums_examples),
        )
    return H


def generate_embeddings(
    edge: np.ndarray,
    nums_type: np.ndarray,
    num: np.ndarray,
    num_list: np.ndarray,
    node_type_mapping: List[int],
    weight: np.ndarray,
):
    if len(num) == 1:
        return [get_adjacency(edge, num, num_list, True)]

    H = generate_H(edge, nums_type, node_type_mapping, weight)
    embeddings = [
        H[i].dot(s_vstack([H[j] for j in range(len(num))]).T).astype("float32")
        for i in range(len(nums_type))
    ]

    new_embeddings = []
    zero_num_list = [0] + list(num_list)
    for i, e in enumerate(embeddings):
        for j, k in enumerate(range(zero_num_list[i], zero_num_list[i + 1])):
            e[j, k] = 0
        col_sum = np.array(e.sum(0)).reshape((-1))
        new_e = e[:, col_sum > 0]
        new_e.eliminate_zeros()
        new_embeddings.append(new_e)

    for i in range(len(nums_type)):
        col_max = np.array(new_embeddings[i].max(0).todense()).flatten()
        _, col_index = new_embeddings[i].nonzero()
        new_embeddings[i].data /= (col_max[col_index] + 1e-10)
    return new_embeddings


def load_data(args):
    train_npz = np.load(os.path.join(args.data_root, args.data, "train_data.npz"), allow_pickle=True)
    test_npz = np.load(os.path.join(args.data_root, args.data, "test_data.npz"), allow_pickle=True)

    train_data = np.asarray(train_npz["train_data"]).astype("int64")
    test_data = np.asarray(test_npz["test_data"]).astype("int64")
    # Keep local (per-type) ids for the adj/MultipleEmbedding branch.
    train_data_local = train_data.copy()
    test_data_local = test_data.copy()
    nums_type = np.asarray(train_npz["nums_type"]).astype("int64")
    num_list = np.cumsum(nums_type).astype("int64")

    try:
        train_weight = train_npz["train_weight"].astype("float32")
        test_weight = test_npz["test_weight"].astype("float32")
    except Exception:
        test_weight = np.ones(len(test_data), dtype="float32")
        train_weight = np.ones(len(train_data), dtype="float32") * args.neg_num

    node_type_mapping = None
    if len(nums_type) > 1:
        node_type_mapping = list(range(train_data.shape[1]))
        for i in range(len(node_type_mapping) - 1):
            train_data[:, i + 1] += num_list[node_type_mapping[i + 1] - 1]
            test_data[:, i + 1] += num_list[node_type_mapping[i + 1] - 1]

    train_weight_mean = float(np.mean(train_weight))
    train_weight = train_weight / train_weight_mean * args.neg_num
    test_weight = test_weight / train_weight_mean * args.neg_num

    train_labels = None
    test_labels = None
    if args.supervised:
        if args.train_label_file:
            train_labels = np.load(args.train_label_file)
        else:
            train_labels = maybe_load_labels(train_npz, ["train_labels", "labels", "y", "train_y"])
        if args.test_label_file:
            test_labels = np.load(args.test_label_file)
        else:
            test_labels = maybe_load_labels(test_npz, ["test_labels", "labels", "y", "test_y"])

        if train_labels is None:
            raise ValueError("Supervised mode requires explicit numeric train labels.")
        train_labels = np.asarray(train_labels).astype("int64").reshape(-1)
        if len(train_labels) != len(train_data):
            raise ValueError(f"train_labels length {len(train_labels)} != len(train_data) {len(train_data)}")
        if test_labels is not None:
            test_labels = np.asarray(test_labels).astype("int64").reshape(-1)
            if len(test_labels) != len(test_data):
                raise ValueError(f"test_labels length {len(test_labels)} != len(test_data) {len(test_data)}")
        args.num_classes = max(args.num_classes, int(train_labels.max()) + 1)
    else:
        args.num_classes = 2

    # Global-id version (after offsets) is used by the sequence model path.
    train_data_raw = train_data.copy()
    test_data_raw = test_data.copy()
    train_data = add_padding_idx(train_data)
    test_data = add_padding_idx(test_data)

    embeddings_initial = None
    if args.feature == "adj":
        if node_type_mapping is None:
            node_type_mapping = list(range(train_data_raw.shape[1]))
        embeddings_initial = generate_embeddings(
            # Must use local ids (before type offsets), matching main_torch.py.
            edge=train_data_local,
            nums_type=nums_type,
            num=nums_type,
            num_list=num_list,
            node_type_mapping=node_type_mapping,
            weight=train_weight,
        )

    return {
        "train_data": train_data,
        "test_data": test_data,
        "train_data_raw": train_data_raw,
        "test_data_raw": test_data_raw,
        "nums_type": nums_type,
        "num_list": num_list,
        "node_type_mapping": node_type_mapping,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_weight": train_weight,
        "test_weight": test_weight,
        "embeddings_initial": embeddings_initial,
    }


def sample_node_for_position(pos: int, num_list: np.ndarray, node_type_mapping: Optional[List[int]]) -> int:
    if node_type_mapping is None:
        return np.random.randint(1, int(num_list[-1]) + 1)
    type_id = node_type_mapping[pos]
    start = 0 if type_id == 0 else int(num_list[type_id - 1])
    end = int(num_list[type_id])
    return np.random.randint(start + 1, end + 1)


def generate_negative_batch(
    batch_x: np.ndarray,
    batch_ids: np.ndarray,
    edge_dict: set,
    num_list: np.ndarray,
    node_type_mapping: Optional[List[int]],
    neg_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positives = batch_x.copy()
    negatives = []
    for sample in positives:
        sample = sample.copy()
        L = sample.shape[0]
        for _ in range(neg_num):
            trial = 0
            temp = sample.copy()
            while trial < 128:
                trial += 1
                pos = np.random.randint(0, L)
                temp[pos] = sample_node_for_position(pos, num_list, node_type_mapping)
                temp = np.sort(temp)
                if tuple(temp.tolist()) not in edge_dict:
                    negatives.append(temp.copy())
                    break
    if len(negatives) > 0:
        neg_ids = np.repeat(batch_ids, neg_num)[: len(negatives)]
    else:
        neg_ids = np.asarray([], dtype=np.int64)

    x = np.concatenate([positives, np.asarray(negatives, dtype=np.int64)], axis=0)
    y = np.concatenate(
        [np.ones((len(positives),), dtype=np.int64), np.zeros((len(negatives),), dtype=np.int64)],
        axis=0,
    )
    edge_ids = np.concatenate([batch_ids, neg_ids], axis=0)
    idx = np.random.permutation(len(x))
    return (
        torch.as_tensor(x[idx], dtype=torch.long),
        torch.as_tensor(y[idx], dtype=torch.long),
        torch.as_tensor(edge_ids[idx], dtype=torch.long),
    )


def compute_epoch_metrics(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    y_np = y.numpy()
    c = logits.shape[1]
    try:
        if c == 1:
            p = logits.view(-1).numpy()
            pred = (p >= 0.5).astype(np.int64)
            acc = float((pred == y_np).mean())
            auc = float(roc_auc_score(y_np, p))
            aupr = float(average_precision_score(y_np, p))
        elif c == 2:
            p = torch.softmax(logits, dim=-1)[:, 1].numpy()
            pred = (p >= 0.5).astype(np.int64)
            acc = float((pred == y_np).mean())
            auc = float(roc_auc_score(y_np, p))
            aupr = float(average_precision_score(y_np, p))
        else:
            p = torch.softmax(logits, dim=-1).numpy()
            y_bin = label_binarize(y_np, classes=np.arange(c))
            acc = float(multiclass_accuracy(logits, y).item())
            auc = float(roc_auc_score(y_bin, p, average="macro", multi_class="ovr"))
            aupr = float(average_precision_score(y_bin, p, average="macro"))
    except Exception:
        acc = float(multiclass_accuracy(logits, y).item()) if c > 1 else 0.0
        auc, aupr = 0.0, 0.0
    return acc, auc, aupr


def run_epoch(
    model,
    backbone: HyperSAGNNEdgeClassifier,
    optimizer: Optional[torch.optim.Optimizer],
    data_x: np.ndarray,
    labels: Optional[np.ndarray],
    edge_dict: set,
    num_list: np.ndarray,
    node_type_mapping: Optional[List[int]],
    args,
    device: torch.device,
    is_test: bool,
    edge_id_offset: int = 0,
):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_recon = 0.0
    steps = 0
    logits_all = []
    labels_all = []

    effective_batch_size = len(data_x) if args.batch_size == -1 else args.batch_size
    if effective_batch_size <= 0:
        raise ValueError(f"batch_size must be positive or -1, got {args.batch_size}")

    batch_num = int(math.ceil(len(data_x) / effective_batch_size))
    iterator = tqdm(range(batch_num), mininterval=0.1, desc="  - (Validation)", leave=False) if is_test else tqdm(
        range(batch_num), mininterval=0.1, desc="  - (Training)", leave=False
    )

    for i in iterator:
        start = i * effective_batch_size
        end = min((i + 1) * effective_batch_size, len(data_x))
        batch_np = data_x[start:end]
        batch_ids_np = edge_id_offset + np.arange(start, end, dtype=np.int64)
        if args.supervised:
            batch_x = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            batch_y = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)
            batch_edge_ids = torch.as_tensor(batch_ids_np, dtype=torch.long, device=device)
        else:
            batch_x, batch_y, batch_edge_ids = generate_negative_batch(
                batch_np,
                batch_ids=batch_ids_np,
                edge_dict=edge_dict,
                num_list=num_list,
                node_type_mapping=node_type_mapping,
                neg_num=args.neg_num,
            )
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_edge_ids = batch_edge_ids.to(device)

        with torch.set_grad_enabled(optimizer is not None):
            if args.model == "baseline":
                logits, _, recon_loss = backbone(batch_x)
                logits_kept, y_kept = logits, batch_y
                if logits_kept.dim() == 2 and logits_kept.size(-1) == 1:
                    loss_bce = F.binary_cross_entropy(logits_kept, y_kept.float().view(-1, 1))
                else:
                    loss_bce = F.cross_entropy(logits_kept, y_kept)
                loss = loss_bce + args.recon_weight * recon_loss
            else:
                out = model(batch_x, y=batch_y, edge_ids=batch_edge_ids, is_test=is_test)
                if args.kept_only:
                    keep_mask = out["keep_mask"].view(-1).bool()
                    logits_kept = out["logits_all"][keep_mask]
                    y_kept = batch_y[keep_mask]
                    if logits_kept.dim() == 2 and logits_kept.size(-1) == 1:
                        loss_bce = F.binary_cross_entropy(logits_kept, y_kept.float().view(-1, 1))
                    else:
                        loss_bce = F.cross_entropy(logits_kept, y_kept)
                else:
                    logits_kept = out["logits_all"]
                    y_kept = batch_y
                    loss_bce = out["loss_cls"]
                recon_loss = out["recon_loss"]
                loss = loss_bce + args.recon_weight * recon_loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(loss_bce.item())
        total_recon += float(recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss)
        logits_all.append(logits_kept.detach().cpu())
        labels_all.append(y_kept.detach().cpu())
        steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    logits_cat = torch.cat(logits_all, dim=0)
    y_cat = torch.cat(labels_all, dim=0)
    acc, auc, aupr = compute_epoch_metrics(logits_cat, y_cat)
    return total_loss / steps, total_bce / steps, total_recon / steps, acc, auc, aupr


class SummaryWrapper(nn.Module):
    def __init__(self, backbone: HyperSAGNNEdgeClassifier):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits, _, _ = self.backbone(x.long())
        return logits


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    hypersagnn_adapter = load_hypersagnn_adapter(args.hypersagnn_impl)
    if args.model == "ehgnn_preenc" and not hasattr(hypersagnn_adapter, "HyperSAGNNSparsePreEncMask"):
        raise ValueError(f"--model ehgnn_preenc is not available in {hypersagnn_adapter.__name__}. Use --hypersagnn-impl v1.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_data(args)
    train_x = data["train_data"]
    test_x = data["test_data"]
    train_y = data["train_labels"]
    test_y = data["test_labels"]
    num_list = data["num_list"]
    node_type_mapping = data["node_type_mapping"]
    train_dict = build_hash(train_x)
    test_dict = build_hash(test_x).union(train_dict)

    if args.feature == "adj":
        node_embedding = MultipleEmbedding(
            embedding_weights=data["embeddings_initial"],
            dim=args.dimensions,
            sparse=False,
            num_list=torch.as_tensor(num_list),
            node_type_mapping=node_type_mapping,
        ).to(device)
        # Modules.py keeps num_list as a plain tensor attr (not a buffer),
        # so .to(device) does not move it automatically.
        if hasattr(node_embedding, "num_list"):
            node_embedding.num_list = node_embedding.num_list.to(device)
    else:
        node_embedding = nn.Embedding(int(num_list[-1]) + 1, args.dimensions, padding_idx=0).to(device)

    backbone = hypersagnn_adapter.HyperSAGNNEdgeClassifier(
        node_embedding=node_embedding,
        d_model=args.dimensions,
        num_classes=args.num_classes,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        diag_mask=args.diag,
        bottle_neck=args.dimensions,
    ).to(device)

    if args.model == "baseline":
        model = backbone
    else:
        num_edges_total = len(train_x) + len(test_x)
        sparsifier = hypersagnn_adapter.HypergraphBatchSparsifier(
            mode=args.mode,
            edge_feat_dim=args.dimensions,
            reg=args.reg,
            rank=args.rank,
            num_edges_total=num_edges_total,
            max_edge_size=train_x.shape[1],
            hidden_dim=args.coarse_mlp,
        ).to(device)
        if args.model == "ehgnn_preenc":
            model = hypersagnn_adapter.HyperSAGNNSparsePreEncMask(
                backbone=backbone, sparsifier=sparsifier, keep_ratio=args.keep_ratio, recon_weight=args.recon_weight
            ).to(device)
        else:
            model = hypersagnn_adapter.HyperSAGNNSparse(
                backbone=backbone, sparsifier=sparsifier, keep_ratio=args.keep_ratio, recon_weight=args.recon_weight
            ).to(device)

    try:
        summary(SummaryWrapper(backbone).to(device), (train_x.shape[1],))
    except Exception as e:
        print(f"summary() hook skipped: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_path = os.path.join(
        args.save_path,
        f"{args.data}_{args.model}_{args.mode}_{args.feature}_{args.hypersagnn_impl}.pt",
    )
    best_val_acc = -1.0
    best_epoch = -1
    best_val = {
        "loss": 0.0,
        "bce": 0.0,
        "recon": 0.0,
        "acc": 0.0,
        "auc": 0.0,
        "aupr": 0.0,
    }

    for epoch in range(args.epochs):
        print(f"[ Epoch {epoch} / {args.epochs} ]")
        tr_loss, tr_bce, tr_recon, tr_acc, tr_auc, tr_aupr = run_epoch(
            model=model,
            backbone=backbone,
            optimizer=optimizer,
            data_x=train_x,
            labels=train_y,
            edge_dict=train_dict,
            num_list=num_list,
            node_type_mapping=node_type_mapping,
            args=args,
            device=device,
            is_test=False,
            edge_id_offset=0,
        )
        print(
            f"  - (Training)   loss: {tr_loss:7.4f}, bce: {tr_bce:7.4f}, recon: {tr_recon:7.4f}, "
            f"acc: {100.0 * tr_acc:6.2f}%, auc: {tr_auc:6.4f}, aupr: {tr_aupr:6.4f}"
        )

        eval_labels = test_y if (args.supervised and test_y is not None) else None
        va_loss, va_bce, va_recon, va_acc, va_auc, va_aupr = run_epoch(
            model=model,
            backbone=backbone,
            optimizer=None,
            data_x=test_x,
            labels=eval_labels,
            edge_dict=test_dict,
            num_list=num_list,
            node_type_mapping=node_type_mapping,
            args=args,
            device=device,
            is_test=True,
            edge_id_offset=len(train_x),
        )
        print(
            f"  - (Validation) loss: {va_loss:7.4f}, bce: {va_bce:7.4f}, recon: {va_recon:7.4f}, "
            f"acc: {100.0 * va_acc:6.2f}%, auc: {va_auc:6.4f}, aupr: {va_aupr:6.4f}"
        )

        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            best_val = {
                "loss": va_loss,
                "bce": va_bce,
                "recon": va_recon,
                "acc": va_acc,
                "auc": va_auc,
                "aupr": va_aupr,
            }
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_acc": va_acc}, ckpt_path)

    print(f"Best validation accuracy: {100.0 * best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {ckpt_path}")
    write_results_csv(args, ckpt_path, best_epoch, best_val)


def write_results_csv(args, ckpt_path: str, best_epoch: int, best_val: Dict[str, float]):
    csv_path = args.results_csv
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    header = [
        "timestamp",
        "data",
        "model",
        "mode",
        "variant",
        "feature",
        "hypersagnn_impl",
        "kept_only",
        "supervised",
        "device",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dimensions",
        "keep_ratio",
        "rank",
        "recon_weight",
        "seed",
        "best_epoch",
        "best_val_loss",
        "best_val_bce",
        "best_val_recon",
        "best_val_acc",
        "best_val_auc",
        "best_val_aupr",
        "checkpoint",
    ]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": args.data,
        "model": args.model,
        "mode": args.mode,
        "variant": variant_map.get(args.mode,'None') if args.model in ["ehgnn", "ehgnn_preenc"] else "Hyper-SAGNN",
        "feature": args.feature,
        "hypersagnn_impl": args.hypersagnn_impl,
        "kept_only": args.kept_only,
        "supervised": args.supervised,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dimensions": args.dimensions,
        "keep_ratio": args.keep_ratio,
        "rank": args.rank,
        "recon_weight": args.recon_weight,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val["loss"],
        "best_val_bce": best_val["bce"],
        "best_val_recon": best_val["recon"],
        "best_val_acc": best_val["acc"],
        "best_val_auc": best_val["auc"],
        "best_val_aupr": best_val["aupr"],
        "checkpoint": ckpt_path,
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()

# Keep this unless major changes are made to the code structure: 
# Baseline + adj: 
# python trainer.py --data wordnet --model baseline --feature adj
# EHGNN hard-prune + adj: 
# python trainer.py --data wordnet --model ehgnn --feature adj --keep-ratio 0.5

# python trainer.py --data wordnet --model baseline --feature adj
# python trainer.py --data wordnet --model ehgnn --mode random --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode degdist --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode effresist --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode learnmask --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode learnmask+ --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode learnmask_cond --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode learnmask+_agn --feature adj --keep-ratio 0.5
# python trainer.py --data wordnet --model ehgnn --mode Neural --feature adj --keep-ratio 0.5
