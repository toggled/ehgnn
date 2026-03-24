#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse
import random
import copy
import json
import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from layers import *
from preprocessing import *
from torch_geometric.data import Data

from convert_datasets_to_pygDataset import dataset_Hypergraph
import sys
#sys.path.append('../../')
from mla_utils import *

import pandas as pd

import torch
from contextlib import contextmanager
import subprocess
import re
from edgnn import EquivSetGNN
from diagnostic import * 

# def get_gpu_memory_usage(cuda_id):
#     result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
#     memory_usage = [int(x) for x in result.decode('utf-8').strip().split('\n')]
#     return memory_usage[int(cuda_id)]
 
variant_map = {
    "full": "Full",
    "random": "Random",
    "degdist": "Degdist",
    "effresist": "Spectral",
    "learnmask": "EHGNN-F",       # EHGNN-F
    "learnmask_cond": "EHGNN-F(cond)",      # EHGNN-C
    "learnmask+": "EHGNN-C(cond)",       # EHGNN-F
    "learnmask+_agn": "EHGNN-C",      # EHGNN-C
    "Neural": "EHGNN-C(cond,LR)",
    "NeuralF": "EHGNN-F(cond,LR)"
}
dname_map = {
    'coauthor_cora': 'Cora-CA', 
    '20newsW100': '20news', 
    'coauthor_dblp': 'DBLP-CA',
    'NTU2012': 'NTU2012', 
    'yelp': 'Yelp', 
    'walmart-trips': 'Walmart', 
    'house-committees': 'House',
    'cora': 'Cora', 
    'citeseer': 'Citeseer', 
    'pubmed': 'PubMed',
    'actor': 'Actor',
    'pokec': 'Pokec',
    'twitch': 'Twitch', 
    'ModelNet40': 'ModelNet40',
    'Mushroom': 'Mushroom', 
    'trivago': 'Trivago',
    'syn_coherent': 'SynCoherent',
    'syn_core_decoy': 'SynCore+Decoy',
}
def is_syn_family_name(dname: str) -> bool:
    return isinstance(dname, str) and (dname.startswith("syn_family_") or dname.startswith("synfamily_"))

def parse_method(args, data, data_p):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        # args.aggregate = 'add'
        args.__setattr__('aggregate','add')
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    # elif args.method == 'CEGCN':
    #     model = CEGCN(in_dim=args.num_features,
    #                   hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
    #                   out_dim=args.num_classes,
    #                   num_layers=args.All_num_layers,
    #                   dropout=args.dropout,
    #                   Normalization=args.normalization)

    # elif args.method == 'CEGAT':
    #     model = CEGAT(in_dim=args.num_features,
    #                   hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
    #                   out_dim=args.num_classes,
    #                   num_layers=args.All_num_layers,
    #                   heads=args.heads,
    #                   output_heads=args.output_heads,
    #                   dropout=args.dropout,
    #                   Normalization=args.normalization)
    elif args.method == 'CEGCN':
        model = CEGCN(args = args,
                      in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)


    elif args.method == 'CEGAT':
        model = CEGAT(args = args, 
                     in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)



    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        
        He_dict = get_HyperGCN_He_dict(data.cpu())
        He_dict_p = get_HyperGCN_He_dict(data_p.cpu())
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         E_p=He_dict_p,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1,2,3,4,5,6,7]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            
            (row_p, col_p), value_p = torch_sparse.from_scipy(data_p.edge_index)
            V_p, E_p = row_p, col_p
            V_p, E_p = V_p.to(device), E_p.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E, V_p=V_p, E_p=E_p)
    #     Below we can add different model, such as HyperGCN and so on
    elif args.method == 'EDGNN':
        model = EquivSetGNN(args.num_features, args.num_classes, args)
    elif args.method == 'HSL':
        model = HSL(args)
    return model


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

#     def plot_result(self, run=None):
#         plt.style.use('seaborn')
#         if run is not None:
#             result = 100 * torch.tensor(self.results[run])
#             x = torch.arange(result.shape[0])
#             plt.figure()
#             print(f'Run {run + 1:02d}:')
#             plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
#             plt.legend(['Train', 'Valid', 'Test'])
#         else:
#             result = 100 * torch.tensor(self.results[0])
#             x = torch.arange(result.shape[0])
#             plt.figure()
# #             print(f'Run {run + 1:02d}:')
#             plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
#             plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(args, model, data, split_idx, eval_func, result=None,edge_index_knn = None):
    """ If result is not None, make sure log softmax has been applied before passing in """
    test_flag = True
    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
        data_input = [data, test_flag]
    else:
        data_input = data
    if args.mode in ['random', 'degdist', 'effresist']:
        data_input.edge_index = edge_index_knn

    # if args.mode == 'edgecent':
    #     data_input.edge_index  = edge_centrality_sparsify(data.edge_index, data.n_x, data.num_hyperedges,\
    #                                                       keep_ratio=args.keep_ratio)
    #     # data_input = Add_Self_Loops(data_input)
    # if args.mode == 'random':
    #     data_input.edge_index = random_sparsify(data.edge_index, data.num_hyperedges, keep_ratio=args.keep_ratio)
    #     # data_input = Add_Self_Loops(data_input)
    # if args.mode == 'degdist':
    #     data_input.edge_index = degree_distribution_sparsify(data.edge_index,keep_ratio=args.keep_ratio)
        # data_input = Add_Self_Loops(data_input)
    if result is not None:
        out = result
    else:
        model.eval()
        if args.method == 'HSL':
            logits, edge_probs, contrastive_loss = model(data_input, is_test=True)
            out = F.log_softmax(logits, dim=1)
        elif args.mode.startswith('learnmask') or args.mode.startswith('Neural'):
            out = model(data_input,is_test= True)
            out = F.log_softmax(out, dim=1)
        else:
            out = model(data_input)
            out = F.log_softmax(out, dim=1)
    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

#     ipdb.set_trace()
#     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _choose_adv_class(c: int, num_classes: int, rng: np.random.Generator, mode: str) -> int:
    if num_classes <= 1:
        return c
    if mode == "next":
        return (c + 1) % num_classes
    # random != c
    others = [cc for cc in range(num_classes) if cc != c]
    return int(rng.choice(others))

def corrupt_family(
    data: Data,
    *,
    p_edge: float,
    q_inc: float,
    adversarial_mode: str,
    seed: int,
) -> Data:
    """
    Dynamic corruption for real data in V2E format.

    - With prob p_edge, an entire hyperedge's incidences are replaced by nodes from an adversarial class.
      (coarse corruption)
    - Otherwise, within the hyperedge, replace ~q_inc fraction of incidences with adversarial-class nodes.
      (fine corruption)

    Assumes data.edge_index = [V_idx; E_idx] with E_idx in [0, m-1] (or at least integer ids).
    """
    assert 0.0 <= p_edge <= 1.0
    assert 0.0 <= q_inc <= 1.0

    out = copy.deepcopy(data)

    V_idx = out.edge_index[0].clone()
    E_idx = out.edge_index[1].clone()

    # Ensure hyperedge ids start at 0 (you do this later for some methods already). :contentReference[oaicite:3]{index=3}
    E_idx = E_idx - E_idx.min()

    N = out.x.size(0)
    y = out.y
    num_classes = int(y.max().item() + 1)

    m = int(E_idx.max().item() + 1)
    rng = np.random.default_rng(seed)

    # Precompute nodes per class for fast sampling
    class_nodes = []
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1).cpu().numpy()
        class_nodes.append(idx)

    # Group incidence positions per hyperedge
    # (looping over m is ok for Cora-scale; optimize later if needed)
    new_V_list = []
    new_E_list = []

    for e in range(m):
        pos = (E_idx == e).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0:
            continue

        nodes = V_idx[pos]
        edge_sz = int(nodes.numel())

        # true class = majority label among nodes in this hyperedge
        labels = y[nodes].detach().cpu().numpy()
        # majority vote (ties broken arbitrarily by argmax)
        binc = np.bincount(labels, minlength=num_classes)
        c = int(binc.argmax())
        c_bad = _choose_adv_class(c, num_classes, rng, adversarial_mode)

        if rng.random() < p_edge:
            # ---- coarse corruption: replace entire edge by adversarial-class nodes ----
            pool = class_nodes[c_bad]
            if pool.size == 0:
                # fallback: keep as-is
                new_nodes = nodes.detach().cpu().numpy()
            else:
                replace = pool.size < edge_sz
                new_nodes = rng.choice(pool, size=edge_sz, replace=replace)
        else:
            # ---- fine corruption: replace a fraction of incidences ----
            k = int(round(q_inc * edge_sz))
            k = max(0, min(k, edge_sz))

            nodes_np = nodes.detach().cpu().numpy().copy()
            if k > 0:
                rep_pos = rng.choice(edge_sz, size=k, replace=False)
                pool = class_nodes[c_bad]
                if pool.size > 0:
                    replace = pool.size < k
                    nodes_np[rep_pos] = rng.choice(pool, size=k, replace=replace)

            new_nodes = nodes_np

        new_V_list.append(torch.from_numpy(np.asarray(new_nodes, dtype=np.int64)))
        new_E_list.append(torch.full((edge_sz,), e, dtype=torch.long))

    new_V = torch.cat(new_V_list, dim=0).to(out.edge_index.device)
    new_E = torch.cat(new_E_list, dim=0).to(out.edge_index.device)
    out.edge_index = torch.stack([new_V, new_E], dim=0)

    # keep num_hyperedges consistent if present
    out.num_hyperedges = torch.tensor([m], device=out.edge_index.device)
    # keep n_x consistent if present
    if not hasattr(out, "n_x"):
        out.n_x = torch.tensor([N], device=out.edge_index.device)

    return out

def perturb_hyperedges(data, prop, perturb_type='delete'):
    data_p = copy.deepcopy(data)
    edge_index = data_p.edge_index
    num_node = data.x.shape[0]
    e_idxs = edge_index[1,:] - num_node
    num_edge = (edge_index[1,:].max()) - (edge_index[1,:].min())
    if((perturb_type == 'delete') or (perturb_type == 'replace')):
        p_num = num_edge * prop
        p_num = int(p_num)
        if p_num == 0:
            return data
        chosen_edges = torch.as_tensor(np.random.permutation(int(num_edge.numpy()))).to(edge_index.device)
        chosen_edges = chosen_edges[:p_num]
        if(perturb_type == 'delete'):
            data_p.edge_index = delete_edges(edge_index, chosen_edges, e_idxs)
        else: # replace = add + delete
            data_p.edge_index = replace_edges(edge_index, chosen_edges, e_idxs, num_node)
    elif(perturb_type == 'add'):
        # p_num = num_edge * prop / (1 - prop)
        p_num = num_edge * prop
        p_num = int(p_num)
        if p_num == 0:
            return data
        data_p.edge_index = add_edges(edge_index, p_num, num_node)
    return data_p

def delete_edges(edge_index, chosen_edges, e_idxs):
    for i in range(chosen_edges.shape[0]):
        chosen_edge = chosen_edges[i]
        edge_index = edge_index[:, (e_idxs != chosen_edge)]
        e_idxs = e_idxs[(e_idxs != chosen_edge)]
    return edge_index

def replace_edges(edge_index, chosen_edges, e_idxs, num_node):
    edge_index = delete_edges(edge_index, chosen_edges, e_idxs)
    edge_index = add_edges_r(edge_index, chosen_edges, num_node)
    return edge_index

def add_edges_r(edge_index, chosen_edges, num_node):
    edge_idxs = [edge_index]
    for i in range(chosen_edges.shape[0]):
        new_edge = torch.as_tensor(np.random.choice(int(num_node), 16, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = chosen_edges[i] + num_node
            edge_idxs.append(edge_idx_i)
    edge_idxs = torch.cat(edge_idxs, dim=1)
    return torch.tensor(edge_idxs, dtype=torch.int64)
    
def add_edges(edge_index, p_num, num_node):
    start_e_idx = edge_index[1,:].max() + 1
    edge_idxs = [edge_index]
    for i in range(p_num):
        new_edge = torch.as_tensor(np.random.choice(int(num_node.cpu().numpy()), 5, replace=False)).to(edge_index.device)
        for j in range(new_edge.shape[0]):
            edge_idx_i = torch.zeros([2,1]).to(edge_index.device)
            edge_idx_i[0,0] = new_edge[j]
            edge_idx_i[1,0] = start_e_idx
            edge_idxs.append(edge_idx_i)
        start_e_idx = start_e_idx + 1
    edge_idxs = torch.cat(edge_idxs, dim=1)
    return torch.tensor(edge_idxs, dtype=torch.int64)

def unignn_ini_ve(data, device):
    data = ConstructH(data)
    data.edge_index = sp.csr_matrix(data.edge_index)
    # Compute degV and degE
    (row, col), value = torch_sparse.from_scipy(data.edge_index)
    V, E = row, col
    return V, E

def unignn_get_deg(V, E):
    degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
    from torch_scatter import scatter
    degE = scatter(degV[V], E, dim=0, reduce='mean')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[torch.isinf(degV)] = 1
    return degV, degE

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True 
    if torch.cuda.is_available():
        torch.cuda.current_device()
        torch.cuda._initialized = True

# --- Main part of the training ---
# # Part 0: Parse arguments


def sparsity_budget_loss(mask_probs, num_edges, args):
    # mask_probs: tensor of shape [num_edges] with values in [0,1]
    # target_budget = num_edges * args.keep_ratio
    return (mask_probs.mean() - args.keep_ratio) ** 2

def kl_divergence_to_bernoulli(mask_probs, p):
    # p is the desired sparsity rate
    eps = 1e-10  # for numerical stability
    kl = mask_probs * (mask_probs + eps).log() - mask_probs * (p + eps).log() + \
         (1 - mask_probs) * (1 - mask_probs + eps).log() - (1 - mask_probs) * (1 - p + eps).log()
    return kl.sum()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='walmart-trips-100')
    # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--seed', default=1, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1,0,1,2,3,4,5,6,7], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=64,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=-1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    parser.add_argument('--perturb_type', default='delete', type=str)
    parser.add_argument('--perturb_prop', default=0.0, type=float)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    # Args for ED-GNN
    parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
    parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
    parser.add_argument('--restart_alpha', default=0.5, type=float)

    # Args for sparsification methods
    parser.add_argument('--patience', type=int, default=200, help='Patience for training with early stopping.')
    parser.add_argument('--mode', type=str, default='learnmask', choices=['static','adaptive','budgeted','random','learnmask','learnmask+','degree','degdist','effresist','edgecent','full','Neural','learnmask_cond','learnmask+_agn','NeuralF'])
    parser.add_argument('--keep_ratio', type=float, default=0.3, help='keep ratio for sparsification')
    parser.add_argument('--reg',type=str,default='none', choices=['none','l2','kl'], help='regularization for the model')
    parser.add_argument('--coarse_MLP', type=int, default = 32, help='hidden dimension for the coarse MLP')
    parser.add_argument('--withbucket', action='store_true')
    parser.add_argument('--num_buckets', type=int, default=4)
    parser.add_argument('--withcluster', action='store_true')
    parser.add_argument('--cluster_size', type=int, default=8)
    parser.add_argument('--withchunk', action='store_true')
    parser.add_argument('--chunk_size', type=float, default=1)
    parser.add_argument('--fname',type=str,default='none', help='save filename (for testing)')
    parser.add_argument('--chunk_scatter', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    # HSL-specific CLI arguments
    parser.add_argument('--hsl_contrastive_weight', type=float, default=1.0,
                    help='weight for HSL contrastive alignment loss')
    parser.add_argument('--hsl_tau', type=float, default=0.5,
                        help='temperature for HSL Gumbel-sigmoid sampling')
    parser.add_argument('--hsl_lambda', type=float, default=0.5, help='l2 regularizer')
    # Spectral method make better: Use approximate pseudoinverse of laplacian using a Hutchinson-type random projection estimator for diag(L^+)
    parser.add_argument('--approxLinv', action='store_true') # If set uses laplacian pseudoinverse approximately
    parser.add_argument('--theory',action='store_true')
    parser.add_argument('--sampling',type=str,default='multinomial',choices=['multinomial','gumbel'])
    parser.add_argument("--diagnostics", type=int, default=0)
    # ---- Dynamic corruption (in-memory) ----
    parser.add_argument('--corrupt', action='store_true',
                        help="Apply in-memory corruption family to dataset.data.")
    parser.add_argument('--p_edge', type=float, default=0.0,
                        help="Probability an entire hyperedge is corrupted (coarse-type).")
    parser.add_argument('--q_inc', type=float, default=0.0,
                        help="Fraction of incidences corrupted inside non-globally-bad edges (fine-type).")
    parser.add_argument('--adversarial_mode', type=str, default='next',
                        choices=['next', 'random'])
    parser.add_argument('--corrupt_seed', type=int, default=0)

    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    # args = parser.parse_args([])
    # args, _ = parser.parse_known_args()
    
    ### NOTE: If we activate theory, the revised EHGNN-F in models_sparse_fixgradprop.py correctly propagates gradient to logits.
    if args.theory:
        from theoryutils import *
        # from models_sparse_fixgradprop import *
        from models_sparse_tmlr import *
        print('Corrected the bugs in backproping gradients to logits.')
    else:
        from models_sparse import * # Version whose result was reported in the rebuttal. Has issues with propagating grad from loss -> logits.

    
    # # Part 1: Load data
    root='./base_newsplit'
    # root = './base_newsplit_'+args.mode
    save_probs = False
    # root='./'+args.attack+'_hypergraphMLP_final2'
    os.makedirs(root, exist_ok=True)
    save = False
    print('------------ ',args.dname,args.mode,'-------------')
    # AllSetTransformer co-citeseer mla_fgsm 1
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed','actor','pokec','twitch','amazon','syn', 'syn_coherent','syn_core_decoy','syn_incidence_favoured_v3',
                        'ufg_n0.3','ufg_n0.4','ufg_n0.8','ufg_n0.9']
    
    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100','ufg_n0.3','ufg_n0.4','ufg_n0.8','ufg_n0.9']
    if args.dname in ['actor']:
        args.__setattr__('add_self_loop', False)
    if args.dname in existing_dataset or is_syn_family_name(args.dname):
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = './data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = './data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = './data/AllSet_all_raw_data/yelp/'
            elif dname in ['actor','pokec','twitch','amazon','syn']:
                p2raw = './data/hetero/'
                print('p2raw: ',p2raw)
            elif dname in ['syn_coherent','syn_core_decoy','syn_incidence_favoured_v3'] or is_syn_family_name(dname):
                p2raw = None
            else:
                p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
        if args.corrupt:
            args.dname = (
                f"corrupt_family_{args.dname}"
                f"_pe{args.p_edge:.2f}"
                f"_qi{args.q_inc:.2f}"
                f"_{args.adversarial_mode}"
            )
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100','ufg_n0.3','ufg_n0.4','ufg_n0.8','ufg_n0.9']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])
    
    # ipdb.set_trace()
    #     Preprocessing
    # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
    setup_seed(args.seed)
    if args.cuda in [0, 1, 2, 3, 4, 5, 6, 7]:
        device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
            data_p = Add_Self_Loops(data_p)
        if args.exclude_self:
            data = expand_edge_index(data)
            data_p = expand_edge_index(data_p)
    
        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
        data_p = norm_contruction(data_p, option=args.normtype)
        
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')
        data_p = ConstructV2V(data_p)
        data_p = norm_contruction(data_p, TYPE='V2V')
    
    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
    #     ipdb.set_trace()
    #   Feature normalization, default option in HyperGCN
        # X = data.x
        # X = sp.csr_matrix(utils.normalise(np.array(X)), dtype=np.float32)
        # X = torch.FloatTensor(np.array(X.todense()))
        # data.x = X
    
    # elif args.method in ['HGNN']:
    #     data = ExtractV2E(data)
    #     if args.add_self_loop:
    #         data = Add_Self_Loops(data)
    #     data = ConstructH(data)
    #     data = generate_G_from_H(data)
    
    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data_p = Add_Self_Loops(data_p)
            data = Add_Self_Loops(data)
            
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
        
        H_p = ConstructH_HNHN(data_p)
        data_p = generate_norm_HNHN(H_p, data_p, args)
        data_p.edge_index[1] -= data_p.edge_index[1].min()
    
    elif args.method in ['HCHA', 'HGNN', 'EDGNN', 'HSL']:
        # rep = find_repeated_hyperedges(data)
        # # print('repeated hyperedges: ',len(rep[0]))
        # import sys 
        # sys.exit(1)
        # print('if: ',data)
        # print(data.edge_index[1].min(),data.edge_index[1].max())
        # print(data.edge_index[0].min(),data.edge_index[0].max())
        data = ExtractV2E(data)
        print('after extract: ',data)
        # ---- In-memory corruption ----
        if args.corrupt:
            data = corrupt_family(
                data,
                p_edge=args.p_edge,
                q_inc=args.q_inc,
                adversarial_mode=args.adversarial_mode,
                seed=args.corrupt_seed,
            )
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.mode == 'full' and args.method != "HSL":
            if args.add_self_loop:
                data = Add_Self_Loops(data)
                # print('after add self loop: ',data)
                data_p = Add_Self_Loops(data_p)
    #    Make the first he_id to be 0
        # data_p.edge_index[1] -= data_p.edge_index[1].min()
        # print('data.edge_index[1].min(): ',data.edge_index[1].min())
        data.edge_index[1] -= data.edge_index[1].min()
        # print('after min: ',data)
    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
            data_p = Add_Self_Loops(data_p)

            
        V, E = unignn_ini_ve(data, args)
        V, E = V.to(device), E.to(device)
        
        V_p, E_p = unignn_ini_ve(data_p, args)
        V_p, E_p = V_p.to(device), E_p.to(device)

        args.UniGNN_degV, args.UniGNN_degE = unignn_get_deg(V, E)
        args.UniGNN_degV_p, args.UniGNN_degE_p = unignn_get_deg(V_p, E_p)
    
        V, E = V.cpu(), E.cpu()
        del V
        del E
        V_p, E_p = V_p.cpu(), E_p.cpu()
        del V_p
        del E_p
    
    #     Get splits
    # split_idx_lst = []
    # for run in range(args.runs):
    #     split_idx = rand_train_test_idx(
    #         data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    #     split_idx_lst.append(split_idx)
    if args.dname == 'cora':
        dataset = 'co-cora'
    elif args.dname == 'citeseer':
        dataset = 'co-citeseer'
    elif args.dname == 'coauthor_cora':
        dataset = 'coauth_cora'
    elif args.dname == '20newsW100':
        dataset = "news20"
    elif args.dname == 'house-committees-100':
        dataset = "house"
    else:
        dataset = args.dname
        # raise ValueError('dataset not supported')
    if args.withchunk:
        assert args.chunk_size > 0
        if args.mode =='learnmask+' or args.mode == 'learnmask+_agn':
            args.__setattr__('chunk_size', math.ceil(args.chunk_size * data.num_hyperedges))
        if args.mode == 'learnmask' or args.mode == 'learnmask_cond':
            args.__setattr__('chunk_size', math.ceil(args.chunk_size * data.edge_index.shape[1]))
    
    args.__setattr__('dataset',dataset)
    args.__setattr__('num_hyperedges',data.num_hyperedges)
    args.__setattr__('num_incidences',data.edge_index.shape[1])
    
    # args.__setattr__('num_hyperedges',data.edge_index.shape[1])
    args.__setattr__('n_x',data.n_x)
    args.__setattr__('F',data.x.shape[1])
    # if args.dname.startswith('learnmask'):
    #     hsl_sparsity = {
    #     "cora": 0.9758251905441284,
    #     "walmart-trips": 0.9999662439028422,
    #     "coauthor_dblp": 0.9999677340189616,
    #     "yelp": 0.4427742051581542,
    #     "ModelNet40": 0.9414345026016236,
    #     "NTU2012": 1.0,
    #     "coauthor_cora": 0.9970686634381613,
    #     "house-committees": 0.2658914625644684,
    #     "citeseer": 0.3010973930358886,
    #     "pubmed": 0.7629340092341105,
    #     "actor": 0.9939090013504028,
    #     "pokec": 0.1583333338300387,
    #     "twitch": 0.7323714047670364,
    #     "20newsW100": 1.0,
    #     "trivago": 0.0,
    #     "Mushroom": 1.0
    #     }
    #     args.__setattr__('keep_ratio',1-hsl_sparsity[dname])
    #     if args.dname in ['NTU2012','20newsW100','Mushroom','walmart-trips','coauthor_dblp','coauthor_cora','actor']:
    #         import sys 
    #         sys.exit(1)

    # if dataset not in ['cora','citeseer','coauth_cora']:
    #     print('here')
    #     split_idx = rand_train_test_idx(data.y)
    #     train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    # else:
    #     _, _, _, train_mask, val_mask, test_mask = get_dataset(args, device=device,\
    #                                                            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    # split_idx = rand_train_test_idx(data.y)
    split_idx = rand_train_test_idx(data.y,train_prop = 0.1,valid_prop = 0.1)
    train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    print('% Train: ',sum(train_mask)*100/len(train_mask))
    print(data)
    import time

    # print("Printed immediately.")
    # time.sleep(40) # Pause for 2.5 seconds
    # split_idx_lst = kfold_train_test_idx(data.y, args.runs)
    # # Part 2: Load model
    
    model = parse_method(args, data, data_p)
    # put things to device
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    if args.mode in ['random','degdist','effresist']:
        edge_index_copy = deepcopy(data.edge_index)
    model, data_p = model.to(device), data_p.to(device)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
        args.UniGNN_degV_p = args.UniGNN_degV_p.to(device)
        args.UniGNN_degE_p = args.UniGNN_degE_p.to(device)
    
    num_params = count_parameters(model)
    _,_,static_model_bytes = get_param_and_buffer_memory_bytes(model)
    # print('static model size (MB): ', static_model_bytes/(1024**2))
    # # Part 3: Main. Training + Evaluation
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model.train()
    # print('MODEL:', model)
    
    ### Training loop ###
    runtime_list = []
    eval_runtime_list = []
    sparsification_time_list = []
    spars_ratios = []
    max_memory_list = []
    end_to_end_time_list = []
    stop_epoch_list = []
    test_accuracy_list = []
    mask_flipfraction_list = []
    diag_list = []

    # memories_epochs = []
    data_copy = deepcopy(data)

    for run in range(args.runs):
        if args.mode in ['random','degdist','effresist']:
            data.edge_index = edge_index_copy
        data = data.to(device)
        # split_idx = split_idx_lst[run]
        split_idx = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
        train_idx = split_idx['train'].to(device)
        setup_seed(run)
        model.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #     This is for HNHN only
    #     if args.method == 'HNHN':
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100, gamma=0.51)
        best_val = float('-inf')
        best_val_loss = float('inf')
        patience = args.patience 
        patience_counter = 0
        best_model_state = None
        Z_orig = None
        start_time = time.time()
        if args.mode == 'effresist':
            
            if args.approxLinv:
                # edge_index_knn=  effective_resistance_sparsify_approx(
                #                     data.edge_index,
                #                     data.n_x,
                #                     data.num_hyperedges,
                #                     keep_ratio=args.keep_ratio,
                #                     num_probes=16,     # try 8, 16, 32 to trade accuracy vs. cost
                #                     cg_tol=1e-5,
                #                     cg_max_iter=200,
                #                     reg=1e-3,
                #                 )
                edge_index_knn = effective_resistance_sparsify_approx_edgeindex(
                    data.edge_index,
                    data.num_nodes,
                    int(data.num_hyperedges),
                    keep_ratio=args.keep_ratio,
                    num_probes=16,
                    cg_tol=1e-5,
                    cg_max_iter=200,
                    reg=1e-3,
                )

            else:
                edge_index_knn = effective_resistance_sparsify(data.edge_index, data.n_x, keep_ratio=args.keep_ratio)
            ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = args.verbose)
            data.edge_index = edge_index_knn
        if args.mode == 'edgecent':
            edge_index_knn = edge_centrality_sparsify(data.edge_index, data.n_x, data.num_hyperedges, keep_ratio=args.keep_ratio)
            ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = args.verbose)
            data.edge_index = edge_index_knn
        if args.mode == 'random':
            edge_index_knn = random_sparsify(data.edge_index, data.num_hyperedges, keep_ratio=args.keep_ratio)
            ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = args.verbose)
            data.edge_index = edge_index_knn
        if args.mode == 'degdist':
            edge_index_knn = degree_distribution_sparsify(data.edge_index,keep_ratio=args.keep_ratio)
            ratio = log_sparsification_info(data.edge_index, edge_index_knn, label=None, verbose = args.verbose)
            data.edge_index = edge_index_knn
        total_epochs = 0
        end_time = time.time()
        sp_time = end_time - start_time
        sparsification_time_list.append(sp_time)
        start_time = time.time()
        memories = []
        probabilities = []
        masks = []
        mask_flips = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        run_peak_mem = 0
        # if args.theory:
        #     grad_tracker = GradSignTracker()
        for epoch in tqdm(range(args.epochs)):
            #         Training part
            total_epochs+=1
            model.train()
            optimizer.zero_grad()
            test_flag = False
            if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
                data_input = [data, test_flag]
            else:
                data_input = data
            # data_input = Add_Self_Loops(data_input)
            # print(epoch,' ',data_input)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)  # keep if you want per-epoch peaks
            # mem_before = torch.cuda.memory_allocated(device)
            if args.method == 'HSL':
                # HSL returns (logits, edge_probs, contrastive_loss)
                logits, probs, contrastive_loss = model(data_input, epoch)

                out = F.log_softmax(logits, dim=1)

                cls_loss = criterion(out[train_idx], data.y[train_idx])
                loss = cls_loss + args.hsl_contrastive_weight * contrastive_loss
            elif args.mode.startswith('learnmask') or args.mode.startswith('Neural'):
                if save_probs:
                    out, probs,mask = model(data_input,epoch, return_mask = True)
                    out = F.log_softmax(out, dim=1)
                    probabilities.append(probs.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())
                else:
                    if args.theory:
                        out, mask, probs = model(data_input,epoch, return_mask = True)
                        # with torch.no_grad():
                        #     V_idx, E_idx = data.edge_index
                        #     # probs = mask # torch.sigmoid(model.mask_module.logits)
                        #     k = int(args.keep_ratio * mask.numel())
                        #     keep_e = torch.topk(mask, k).indices
                        #     mask_e = torch.zeros_like(mask, dtype=torch.bool)
                        #     mask_e[keep_e] = True
                        # if epoch == 0:
                        #     prev_mask_e = None
                        # if prev_mask_e is not None:
                        #     # fraction of edges whose keep/drop decision changed
                        #     flip_frac = (mask_e != prev_mask_e).float().mean().item()
                        #     # Jaccard similarity of kept sets
                        #     inter = (mask_e & prev_mask_e).float().sum()
                        #     union = (mask_e | prev_mask_e).float().sum().clamp(min=1)
                        #     jacc = (inter / union).item()
                        #     mask_flips.append(flip_frac*100)
                        #     print(f"[epoch {epoch}] mask flip frac = {flip_frac:.4f}, Jaccard = {jacc:.4f}")

                        # prev_mask_e = mask_e.clone()
                    else:
                        out, probs = model(data_input,epoch)
                    out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], data.y[train_idx])

            else:
                out = model(data_input)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], data.y[train_idx])
            # if args.theory:
            #     if epoch % 10 == 0:
            #         norms, L = estimate_model_lipschitz(model)
            #         print(f"[epoch {epoch}] Lipschitz constant upper bound ≈ {L:.2e}")
            #         for name, sn in norms:
            #             print(f"  {name}: spectral norm ≈ {sn:.3f}")

            
            # mem_after_fwd = torch.cuda.memory_allocated(device)
            if args.reg == 'none':
                pass 
            elif args.reg == 'l2':
                if args.method == 'HSL':
                    _lambda = args.hsl_lambda
                else:
                    _lambda = 1
                loss += _lambda * sparsity_budget_loss(mask_probs = probs, num_edges = data.edge_index.shape[1], args = args)
            loss.backward()
            # if args.theory:
            #     # pick the correct logits tensor
            #     # mask_logits = None
            #     # for name, p in model.mask_module.named_parameters():
            #     #     if "logits" in name:
            #     #         mask_logits = p
            #     #         break

            #     # if mask_logits is not None:
            #     #     frac = grad_tracker.track(mask_logits, epoch)
            #     #     print(f"[epoch {epoch}] sign flip fraction = {frac:.4f}")
            #     mask_logits = getattr(model.mask_module, "logits", None)
            #     assert mask_logits is not None 
            #     if mask_logits is not None:
            #         frac = grad_tracker.track(mask_logits, epoch)
            #         print(f"[epoch {epoch}] grad sign flip fraction = {frac}")

            # mem_after_bwd = torch.cuda.memory_allocated(device)

            # if torch.cuda.is_available():
            #     memory_i = get_gpu_memory_usage(args.cuda)
            #     # print('cuda: ',args.cuda,' memory (MB): ',memory_i)
            # else:
            #     memory_i = 0
            # memories.append(memory_i)
            optimizer.step()
            peak_mem = torch.cuda.max_memory_allocated(device)
            run_peak_mem = max(run_peak_mem, peak_mem)
            # Approximations
            # fwd_dynamic = (mem_after_fwd - mem_before)/(1024**2) # in MB
            # bwd_dynamic = (mem_after_bwd - mem_after_fwd)/(1024**2)
            # peak_dynamic = (peak_mem - static_model_bytes)/(1024**2)
            # memories_epochs.append((fwd_dynamic, bwd_dynamic, peak_dynamic))
            valid_loss = F.nll_loss(
                out[split_idx['valid']], data.y[split_idx['valid']])
    #         if args.method == 'HNHN':
    #             scheduler.step()
    #         Evaluation part
    #         if epoch%10 == 0:
    #             if args.mode == 'effresist':
    #                 result =evaluate(args, model, data, split_idx, eval_func, edge_index_knn=edge_index_knn)
    #             else:
    #                 result = evaluate(args, model, data, split_idx, eval_func,edge_index_knn=None)
    #             train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
    #             logger.add_result(run, result[:3])
            if valid_loss.item() < best_val_loss:
                best_val_loss = valid_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
                Z_orig = out.detach()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f'Early stopping at epoch {epoch}.')
                    stop_epoch = epoch
                    break
            # if epoch % args.display_step == 0 and args.display_step > 0:
            #     result = evaluate(args, model, data, split_idx, eval_func,edge_index_knn=None)
            #     print(f'Epoch: {epoch:02d}, '
            #           f'Train Loss: {loss:.4f}, '
            #           f'Valid Loss: {result[4]:.4f}, '
            #           f'Test  Loss: {result[5]:.4f}, '
            #           f'Train Acc: {100 * result[0]:.2f}%, '
            #           f'Valid Acc: {100 * result[1]:.2f}%, '
            #           f'Test  Acc: {100 * result[2]:.2f}%')
                if args.method == 'HSL':
                    print('sparsification ratio = ',1-model.ratio)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        end_time = time.time()
        total_train_time = (end_time - start_time)
        runtime_list.append(total_train_time/total_epochs)
        stop_epoch_list.append(total_epochs)
        if getattr(args, "diagnostics", 0) == 1:
            diag_run = run_incidence_diagnostics(
                    model=model,
                    data=data,
                    args=args,
                    evaluate_fn=evaluate,
                    split_idx=split_idx,
                    eval_func=eval_func,
                )
            if isinstance(diag_run, dict):
                diag_list.append(diag_run)

        start_time = time.time()
        if args.mode in ['random', 'degdist', 'effresist']:
            result = evaluate(args, model, data, split_idx, eval_func, edge_index_knn=edge_index_knn)
        else:
            result = evaluate(args, model, data, split_idx, eval_func, edge_index_knn=None)
        end_time = time.time()
        eval_time = end_time - start_time
        eval_runtime_list.append(eval_time)
        if args.mode.startswith('learnmask') or args.mode.startswith('Neural'):
            ratio = model.ratio 
        if args.mode == 'full':
            ratio = 0.0
        if args.method == 'HSL':
            # model.eval()
            # with torch.no_grad():
            #     _, _, m_hard_e, _ = model(data_input, is_test=True,return_mask=True)
            # rv_emp = model.compute_empirical_rv(data.edge_index, m_hard_e, data.n_x)
            # ratio = rv_emp.detach().cpu().item()
            ratio = 1-model.ratio
            # print('keepratio/model.ratio = ',model.ratio)

        spars_ratios.append(ratio)
        # max_memory_list.append(max(memories))
        if torch.cuda.is_available():
            run_peak_mem = max(run_peak_mem, torch.cuda.max_memory_allocated(device))
        max_memory_list.append(run_peak_mem / (1024**2))
        end_to_end_time_list.append(sp_time + total_train_time + eval_time)
        train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
        out[0],out[1],out[2] = out[0].cpu().detach(),out[1].cpu().detach(),out[2].cpu().detach()
        test_accuracy_list.append(test_acc*100)
        # if args.theory:
        #     mask_flipfraction_list.append(mask_flips)
        logger.add_result(run, result[:3])
        print(f'Train accuracy: {train_acc*100:.2f}' )
        print(f'Valid accuracy: {valid_acc*100:.2f}')
        print(f'Test accuracy: {test_acc*100:.2f}')
        if save_probs:
            if args.dname == 'syn':
                root = 'results/output/'
                root = os.path.join(root, args.mode)
                os.system('mkdir -p '+root)
            np.save(os.path.join(root, f'{args.method}_{args.dname}_run{run}_probs.npy'), np.array(probabilities))
            np.save(os.path.join(root, f'{args.method}_{args.dname}_run{run}_masks.npy'), np.array(masks))
            np.save(os.path.join(root, f'{args.method}_{args.dname}_run{run}_output.npy'), out.detach().cpu().numpy())
        # logger.print_statistics(run)
        del optimizer
        torch.cuda.empty_cache()
    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
    print(f'Runtime/epoch: mean = {avg_time:.3f}, std = {std_time:.3f}')
    print(f'Eval Runtime: {np.mean(eval_runtime_list):.3f} ± {np.std(eval_runtime_list):.3f}')
    print(f'Sparsification ratio: {np.mean(spars_ratios):.3f} ± {np.std(spars_ratios):.3f}')
    print(f'Max memory usage (MB): {np.mean(max_memory_list):.3f} ± {np.std(max_memory_list):.3f}')
    print(f'Max memory usage (GB): {np.mean(max_memory_list)/1024:.3f}')
    best_val, best_test = logger.print_statistics()
    print(f'Avg Test acc: {best_test.mean():.2f} ± {best_test.std():.2f}')
    diag_avg = {}
    diag_std = {}
    if len(diag_list) > 0:
        keys = sorted(set().union(*[d.keys() for d in diag_list]))
        for k in keys:
            vals = [d.get(k, np.nan) for d in diag_list]
            vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]

            if len(vals) == 0:
                # all runs were NaN → explicitly store NaN
                diag_avg[k] = np.nan
                diag_std[k] = np.nan
            else:
                diag_avg[k] = float(np.mean(vals))
                diag_std[k] = float(np.std(vals))

    # if args.method == 'HSL':
    #     print('Empirical RVs over runs: ', spars_ratios)
    # stats = {
    #     "method": args.method,
    #     "dname": args.dname,
    #     "mode": args.mode,
    #     "withbucket": args.withbucket,
    #     "lr": args.lr,
    #     "weight_decay": args.wd,
    #     "patience": args.patience, 
    #     "epochs": args.epochs,
    #     "runs": args.runs,
    #     "keep_ratio": args.keep_ratio,
    #     "reg": args.reg,
    #     "sparsification_time_avg": np.mean(sparsification_time_list),
    #     "sparsification_time_std": np.std(sparsification_time_list),
    #     "runtime_per_epoch_avg": np.mean(runtime_list),
    #     "runtime_per_epoch_std": np.std(runtime_list),
    #     "eval_runtime_avg": np.mean(eval_runtime_list),
    #     "eval_runtime_std": np.std(eval_runtime_list),
    #     "sparsification_ratio_avg": np.mean(spars_ratios),
    #     "sparsification_ratio_std": np.std(spars_ratios),
    #     "max_memory_avg": np.mean(max_memory_list),
    #     "max_memory_std": np.std(max_memory_list),
    #     "best_val_acc_avg": best_val.mean().item(),
    #     "best_val_acc_std": best_val.std().item(),
    #     "best_test_acc_avg": best_test.mean().item(),
    #     "best_test_acc_std": best_test.std().item(),
    #     "end_to_end_time_avg": np.mean(end_to_end_time_list),
    #     "end_to_end_time_std": np.std(end_to_end_time_list),
    #     "stop_epoch_avg": np.mean(stop_epoch_list),
    #     "stop_epochs": str(" ".join([str(i) for i in stop_epoch_list])),
    #     "test_accs": str(" ".join([str(i) for i in test_accuracy_list])),
    #     "max_memorys": str(" ".join([str(i) for i in max_memory_list])),
    #     "num_params": num_params,
    #     "MLP_hidden": args.MLP_hidden,
    #     "num_buckets": args.num_buckets if args.withbucket else None,
    #     "cluster_size": args.cluster_size if args.withcluster else None,
    #     "chunk_size": args.chunk_size if args.withchunk else None
    # }
    # if args.theory:
    #     from matplotlib import pyplot as plt
    #     plt.figure(figsize=(6,4))
    #     plt.plot(range(1,len(mask_flipfraction_list[0])+1), mask_flipfraction_list[0])
    #     plt.xlabel('Epoch',fontsize=17)
    #     plt.ylabel('% of masks flipped',fontsize=17)
    #     # if args.dname == 'cora':
    #     #     plt.title("Cora",fontsize=18)
    #     # if args.dname == 'yelp':
    #     #     plt.title("Yelp",fontsize=18)
    #     # if args.dname == 'actor':
    #     #     plt.title("Actor",fontsize=18)
    #     plt.xticks(fontsize=15)
    #     plt.yticks(fontsize=15)
    #     # plt.show()
    #     plt.savefig(f'results/mask_flip_fraction_{args.method}_{args.dname}_{args.mode}.pdf', bbox_inches='tight')
    stats = {
        "method": args.method,
        "dname": dname_map.get(args.dname, args.dname),
        "mode": args.mode,
        "withbucket": args.withbucket,
        "lr": args.lr,
        "weight_decay": args.wd,
        "patience": args.patience, 
        "epochs": args.epochs,
        "runs": args.runs,
        "keep_ratio": args.keep_ratio,
        "reg": args.reg,
        "sparsification_time_avg": np.mean(sparsification_time_list),
        "sparsification_time_std": np.std(sparsification_time_list),
        "runtime_per_epoch_avg": np.mean(runtime_list),
        "runtime_per_epoch_std": np.std(runtime_list),
        "eval_runtime_avg": np.mean(eval_runtime_list),
        "eval_runtime_std": np.std(eval_runtime_list),
        "sparsification_ratio_avg": np.mean(spars_ratios),
        "sparsification_ratio_std": np.std(spars_ratios),
        "max_memory_avg": np.mean(max_memory_list),
        "max_memory_std": np.std(max_memory_list),
        "best_val_acc_avg": best_val.mean().item(),
        "best_val_acc_std": best_val.std().item(),
        "best_test_acc_avg": best_test.mean().item(),
        "best_test_acc_std": best_test.std().item(),
        "end_to_end_time_avg": np.mean(end_to_end_time_list),
        "end_to_end_time_std": np.std(end_to_end_time_list),
        "stop_epoch_avg": np.mean(stop_epoch_list),
        "stop_epochs": str(" ".join([str(i) for i in stop_epoch_list])),
        "test_accs": str(" ".join([str(i) for i in test_accuracy_list])),
        "max_memorys": str(" ".join([str(i) for i in max_memory_list])),
        "num_params": num_params,
        "MLP_hidden": args.MLP_hidden,
        "num_buckets": args.num_buckets if args.withbucket else None,
        "cluster_size": args.cluster_size if args.withcluster else None,
        "chunk_size": args.chunk_size if args.withchunk else None,
         "static_model_MB": static_model_bytes / (1024**2),
         "activation_peak_MB": np.mean(max_memory_list) - static_model_bytes / (1024**2),
         "peak_memory_MB": np.mean(max_memory_list),
         "variant": variant_map[args.mode]
    }

    if args.fname != 'none':
        if len(diag_avg) > 0:
            stats.update({f"diag_{k}_avg": v for k, v in diag_avg.items()})
            stats.update({f"diag_{k}_std": v for k, v in diag_std.items()})

        save_to_csv(stats, filename =args.method+'_'+args.fname)
    else:
        save_to_csv(stats, filename = args.method+"_hgnncCorrect.csv")

    # save_to_csv(stats, filename = args.method+"_resultsICLR.csv")
    # save_to_csv(stats, filename = args.method+"_results.csv")

    # save_to_csv(stats, filename = args.method+"_results_ablation2.csv")
    # save_to_csv(stats, filename = args.method+"_results_ablation_meanAgg.csv")
    # save_to_csv(stats, filename = args.method+"_results_ablation_probnorm.csv")
    # if args.mode.startswith('learnmask+') and (args.coarse_MLP != 32):
    #     stats['coarse_MLP'] = args.coarse_MLP
    #     save_to_csv(stats, filename = args.method+"_results_ablation_coarse_MLPdim.csv")
