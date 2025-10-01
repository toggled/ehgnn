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
from models_sparse import *
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
 
def get_gpu_memory_usage():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_usage = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return memory_usage
 
 
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
    test_flag = True
    if ((args.method == 'UniGCNII') or (args.method == 'HyperGCN')):
        data_input = [data, test_flag]
    else:
        data_input = data
    if args.mode == 'effresist':
        data_input.edge_index = edge_index_knn

    if args.mode == 'edgecent':
        data_input.edge_index  = edge_centrality_sparsify(data.edge_index, data.n_x, data.num_hyperedges,\
                                                          keep_ratio=args.keep_ratio)
        # data_input = Add_Self_Loops(data_input)
    if args.mode == 'random':
        data_input.edge_index = random_sparsify(data.edge_index, data.num_hyperedges, keep_ratio=args.keep_ratio)
        # data_input = Add_Self_Loops(data_input)
    if args.mode == 'degdist':
        data_input.edge_index = degree_distribution_sparsify(data.edge_index,keep_ratio=args.keep_ratio)
        # data_input = Add_Self_Loops(data_input)
    if result is not None:
        out = result
    else:
        model.eval()
        if args.mode.startswith('learnmask') or args.mode.startswith('Neural'):
            out = model(data_input,is_test= True)
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
    ## Attack args
    parser.add_argument('--attack', type=str, default='mla', \
                    choices=['mla','Rand-flip', 'Rand-feat','gradargmax','mla_fgsm'], help='model variant')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Node Feature perturbation bound')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--patience', type=int, default=200,
                    help='Patience for training with early stopping.')
    parser.add_argument('--T', type=int, default=80, help='Number of iterations for the attack.')
    parser.add_argument('--mla_alpha', type=float, default=4.0, help='weight for classification loss')
    parser.add_argument('--eta_H', type=float, default=1e-2, help='Learning rate for H perturbation')
    parser.add_argument('--eta_X', type=float, default=1e-2, help='Learning rate for X perturbation')
    parser.add_argument('--num_epochs_sur', type=int, default=80, help='#epochs for the surrogate training.')
    parser.add_argument('--mode', type=str, default='learnmask', choices=['static','adaptive','budgeted','random','learnmask','learnmask+','degree','degdist','effresist','edgecent','full','Neural','learnmask_cond','learnmask+_agn'])
    parser.add_argument('--keep_ratio', type=float, default=0.5, help='keep ratio for sparsification')
    parser.add_argument('--reg',type=str,default='none', choices=['none','l2','kl'], help='regularization for the model')
    parser.add_argument('--coarse_MLP', type=int, default = 32, help='hidden dimension for the coarse MLP')
    # parser.add_argument('--delxdelh',default='Both',choices=['Both','delx','delh'])
    parser.add_argument('--verbose', action='store_true')

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
    
    
    # # Part 1: Load data
    root='./base_newsplit'
    # root = './base_newsplit_'+args.mode
    save_probs = True
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
                        'cora', 'citeseer', 'pubmed','actor','pokec','twitch','amazon','syn',
                        'ufg_n0.3','ufg_n0.4','ufg_n0.8','ufg_n0.9']
    
    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100','ufg_n0.3','ufg_n0.4','ufg_n0.8','ufg_n0.9']
    if args.dname in ['actor']:
        args.__setattr__('add_self_loop', False)
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = '../data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = '../data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = '../data/AllSet_all_raw_data/yelp/'
            elif dname in ['actor','pokec','twitch','amazon','syn']:
                p2raw = '../data/hetero/'
                print('p2raw: ',p2raw)
            else:
                p2raw = '../data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = '../data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        data = dataset.data
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
    
    elif args.method in ['HCHA', 'HGNN']:
        # print('if: ',data)
        # print(data.edge_index[1].min(),data.edge_index[1].max())
        # print(data.edge_index[0].min(),data.edge_index[0].max())
        data = ExtractV2E(data)
        print('after extract: ',data)
        data_p = perturb_hyperedges(data, args.perturb_prop, args.perturb_type)
        if args.mode == 'full':
            if args.add_self_loop:
                data = Add_Self_Loops(data)
                print('after add self loop: ',data)
                data_p = Add_Self_Loops(data_p)
    #    Make the first he_id to be 0
        data_p.edge_index[1] -= data_p.edge_index[1].min()
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
    
    args.__setattr__('dataset',dataset)
    args.__setattr__('num_hyperedges',data.edge_index.shape[1])
    args.__setattr__('n_x',data.n_x)
    args.__setattr__('F',data.x.shape[1])
    # if dataset not in ['cora','citeseer','coauth_cora']:
    #     print('here')
    #     split_idx = rand_train_test_idx(data.y)
    #     train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    # else:
    #     _, _, _, train_mask, val_mask, test_mask = get_dataset(args, device=device,\
    #                                                            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    # split_idx = rand_train_test_idx(data.y)
    split_idx = rand_train_test_idx(data.y,train_prop = 0.2,valid_prop = 0.2)
    train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
    print('% Train: ',sum(train_mask)*100/len(train_mask))

    
    # split_idx_lst = kfold_train_test_idx(data.y, args.runs)
    # # Part 2: Load model
    
    model = parse_method(args, data, data_p)
    # put things to device
    if args.cuda in [0,1,2,3,4,5,6,7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    edge_index_copy = deepcopy(data.edge_index)
    model, data_p = model.to(device), data_p.to(device)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
        args.UniGNN_degV_p = args.UniGNN_degV_p.to(device)
        args.UniGNN_degE_p = args.UniGNN_degE_p.to(device)
    
    num_params = count_parameters(model)
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
    for run in range(args.runs):
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
            if args.mode.startswith('learnmask') or args.mode.startswith('Neural'):
                if save_probs:
                    out, probs,mask = model(data_input,return_mask = True)
                    probabilities.append(probs.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())
                else:
                    out, probs = model(data_input)
            else:
                out = model(data_input)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            if args.reg == 'none':
                pass 
            elif args.reg == 'l2':
                _lambda = 1.0 
                loss += _lambda * sparsity_budget_loss(mask_probs = probs, num_edges = data.edge_index.shape[1], args = args)
            loss.backward()
            if torch.cuda.is_available():
                memory_i = get_gpu_memory_usage()
            else:
                memory_i = 0
            memories.append(memory_i)
            optimizer.step()
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
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        end_time = time.time()
        total_train_time = (end_time - start_time)
        runtime_list.append(total_train_time/total_epochs)
        stop_epoch_list.append(total_epochs)
        start_time = time.time()
        if args.mode == 'effresist':
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
        spars_ratios.append(ratio)
        max_memory_list.append(max(memories))
        end_to_end_time_list.append(sp_time + total_train_time + eval_time)
        train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out = result
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
    
    ### Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
    print(f'Runtime/epoch: mean = {avg_time:.3f}, std = {std_time:.3f}')
    print(f'Eval Runtime: {np.mean(eval_runtime_list):.3f} ± {np.std(eval_runtime_list):.3f}')
    print(f'Sparsification ratio: {np.mean(spars_ratios):.3f} ± {np.std(spars_ratios):.3f}')
    print(f'Max memory usage (MB): {np.mean(max_memory_list):.3f} ± {np.std(max_memory_list):.3f}')
    print(f'Max memory usage (GB): {np.mean(max_memory_list)/1024:.3f}')
    best_val, best_test = logger.print_statistics()
    print(f'Avg Test acc: {best_test.mean():.2f} ± {best_test.std():.2f}')
    stats = {
        "method": args.method,
        "dname": args.dname,
        "mode": args.mode,
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
        "num_params": num_params
    }
    # save_to_csv(stats, filename = args.method+"_resultsICLR.csv")
    # save_to_csv(stats, filename = args.method+"_results.csv")

    # save_to_csv(stats, filename = args.method+"_results_ablation2.csv")
    # save_to_csv(stats, filename = args.method+"_results_ablation_meanAgg.csv")
    # save_to_csv(stats, filename = args.method+"_results_ablation_probnorm.csv")
    # if args.mode.startswith('learnmask+') and (args.coarse_MLP != 32):
    #     stats['coarse_MLP'] = args.coarse_MLP
    #     save_to_csv(stats, filename = args.method+"_results_ablation_coarse_MLPdim.csv")
