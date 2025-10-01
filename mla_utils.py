import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm 
import os
# from dhg.data import Cooking200, CoauthorshipCora,CocitationCora,CocitationCiteseer,CoauthorshipDBLP, CocitationPubmed,\
#                      Tencent2k, News20,YelpRestaurant, WalmartTrips,HouseCommittees, Yelp3k
# from dhg.utils import split_by_ratio
# from dhg import Hypergraph
import pandas as pd 

def plot_results(args,results,root,plot_acc_traj=True):
    loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, \
        cls_loss_trajectory, deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory,\
             target_test_trajectory = results 
    os.makedirs(os.path.join(root,str(args.seed)), exist_ok=True)
    prefix = os.path.join(root,str(args.seed), args.dataset+'_'+args.model+'_'+str(args.ptb_rate))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(loss_meta_trajectory)
    plt.title('Meta Loss over Iterations')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 2)
    plt.plot(acc_drop_trajectory)
    plt.title('Accuracy Drop over Iterations')
    plt.xlabel('Iteration')

    plt.subplot(2, 2, 3)
    plt.plot(lap_shift_trajectory)
    plt.title('Laplacian Shift over Iterations')
    plt.xlabel('Iteration')
    plt.subplot(2, 2, 4)
    plt.plot(feature_shift_trajectory)
    plt.title('Feature Shift (L2) over Iterations')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(prefix+'_laplacian_feature_shift.png')

    # Plot individual loss components
    plt.figure(figsize=(10, 4))
    plt.yscale('log')
    plt.plot(lap_dist_trajectory, label='Laplacian Loss')
    plt.plot(cls_loss_trajectory, label='Classification Loss')
    plt.plot(deg_penalty_trajectory, label='Degree Penalty')
    plt.title('Meta Loss Components over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(prefix+'_loss_components.png')
    
    # Plot the accuracy trajectories for both models: surrogate and target during attackers iteration
    if plot_acc_traj:
        # Plot the accuracy trajectories for both models: surrogate and target during attackers iteration
        epochs = range(1, args.T + 1)
        # Create the plot
        plt.figure(figsize=(10, 4))
        # Plot the accuracy trajectories for both models
        print(len(surrogate_test_trajectory),len(target_test_trajectory),len(epochs))
        plt.plot(epochs, surrogate_test_trajectory, label='Surrogate Model', color='blue', marker='o')
        plt.plot(epochs, target_test_trajectory, label='Target Model', color='red', marker='x')
        # Add labels and title
        plt.xlabel('Attack Epoch (t)')
        plt.ylabel('Accuracy (Test Set)')
        # plt.title('Epoch-by-Epoch Test Set Accuracy Comparison')
        plt.legend()
        plt.savefig(prefix+'_accuracy_trajectory.png')


# Synthetic hypergraph generator
# def generate_synthetic_hypergraph(n_nodes=200, n_edges=100, d=10):
#     H = torch.zeros((n_nodes, n_edges))
#     for e in range(n_edges):
#         nodes = np.random.choice(n_nodes, size=np.random.randint(2, min(6, n_nodes)), replace=False)
#         H[nodes, e] = 1
#     X = torch.randn(n_nodes, d)
#     labels = torch.randint(0, 3, (n_nodes,))
#     return H.float(), X.float(), labels

# Evaluation helpers
# Compute the Frobenius norm of the embedding shift
# This measures how much the learned node representations change under the attack
def embedding_shift(Z1, Z2):
    return torch.norm(Z1 - Z2).item()

def lap(H):
    de = H.sum(dim=0).clamp(min=1e-6)
    De_inv = torch.diag(1.0 / de)
    dv = H @ torch.ones(H.shape[1], device=H.device)
    Dv_inv_sqrt = torch.diag(1.0 / dv.clamp(min=1e-6).sqrt())
    return torch.eye(H.shape[0], device=H.device) - Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
# Measure the Frobenius norm difference between Laplacians of original and perturbed incidence matrices
# This evaluates the impact of perturbations on the hypergraph structure
def laplacian_diff(H1, H2):
    return torch.norm(lap(H1) - lap(H2)).item()

# Visualize original and adversarial embeddings using t-SNE for interpretability
# Helps detect how much the attack shifted the embedding geometry
def visualize_tsne(args, root, Z1, Z2, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    Z = torch.cat([Z1, Z2], dim=0).detach().cpu().numpy()
    tsne = TSNE(n_components=2)
    Z_2d = tsne.fit_transform(Z)
    plt.scatter(Z_2d[:Z1.shape[0], 0], Z_2d[:Z1.shape[0], 1], c='k', label='original', alpha=0.6, s = 2)
    plt.scatter(Z_2d[Z1.shape[0]:, 0], Z_2d[Z1.shape[0]:, 1], c='red', label='adversarial', alpha=0.6, s = 2)
    plt.legend()
    # plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(root,args.dataset+'_'+args.model+'_'+str(args.ptb_rate)+'_tsne.png'))

# Quantify how stealthy the attack is based on L0 structural change and L∞ feature deviation
def measure_stealthiness(H, H_adv, X, X_adv):
    h_l0 = torch.sum((H - H_adv).abs() > 1e-6).item()
    x_delta = torch.norm((X - X_adv), p=float('inf')).item()

    degree_orig = H.sum(dim=1)
    degree_adv = H_adv.sum(dim=1)
    deg_shift_inf = torch.norm(degree_orig - degree_adv, p=float('inf')).item()
    deg_shift_l1 = torch.norm(degree_orig - degree_adv, p=1).item()
    deg_shift_l2 = torch.norm(degree_orig - degree_adv, p=2).item()

    edge_card_orig = H.sum(dim=0)
    edge_card_adv = H_adv.sum(dim=0)
    edge_card_shift_inf = torch.norm(edge_card_orig - edge_card_adv, p=float('inf')).item()
    edge_card_shift_l1 = torch.norm(edge_card_orig - edge_card_adv, p=1).item()
    edge_card_shift_l2 = torch.norm(edge_card_orig - edge_card_adv, p=2).item()

    return h_l0, x_delta, deg_shift_l1, edge_card_shift_l1,deg_shift_l2, edge_card_shift_l2, deg_shift_inf, edge_card_shift_inf

# Evaluate how well the attack generalizes across different models
# Returns the output logits of each model when run on the adversarially perturbed inputs
def evaluate_transferability(H_adv, X_adv, model_list):
    return [model(X_adv,H_adv).detach() for model in model_list]

# Check how much semantic meaning of node features has changed
# Measured via average cosine similarity between original and perturbed features
def semantic_feature_change(X, X_adv):
    cosine = F.cosine_similarity(X, X_adv, dim=1)
    return 1.0 - cosine.mean().item()

# Evaluate correlation between node degrees and embedding drift
# A negative Pearson correlation supports theory that low-degree nodes are more vulnerable
def degree_sensitivity(H, Z_orig, Z_adv):
    degrees = H.sum(dim=1)
    per_node_shift = torch.norm(Z_orig - Z_adv, dim=1)
    return torch.corrcoef(torch.stack([degrees, per_node_shift]))[0, 1].item()

# Measure drop in classification accuracy before and after the attack
# This is the ultimate indicator of the attack's effectiveness
@torch.no_grad
def classification_drop(args, model, H, HG, X, H_adv, X_adv, labels,W_e = None):
    assert W_e is None
    # print(args.model)
    if args.model in ['hypergcn'] and (HG is not None):
        logits_orig = model(X, HG)
        H_adv = Hypergraph(H_adv.shape[0], incidence_matrix_to_edge_list(H_adv),device=H.device)
        logits_adv = model(X_adv,H_adv)
    else:
        logits_orig = model(X, H)
        logits_adv = model(X_adv,H_adv)
    # else:
    #     logits_orig = model(X, H, W_e)
    #     logits_adv = model(X_adv,H_adv, W_e)
    acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
    acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
    return acc_orig, acc_adv, (acc_orig - acc_adv)/acc_orig

@torch.no_grad
def classification_drop_pois(model, model_pois, H, X, H_adv, X_adv, labels, W_e = None):
    if W_e is None:
        logits_orig = model(X, H)
        logits_adv = model_pois(X_adv,H_adv)
    else:
        logits_orig = model(X, H, W_e)
        logits_adv = model_pois(X_adv,H_adv,W_e)
    acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
    acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
    return acc_orig, acc_adv, (acc_orig - acc_adv)/acc_orig

# @torch.no_grad
# def classification_drop_pois_separate(model, model_pois, H, X, H_adv, X_adv, labels, train_mask, val_mask, test_mask, W_e = None):
#     if W_e is None:
#         logits_orig = model(X, H)
#         logits_adv = model_pois(X_adv,H_adv)
#     else:
#         logits_orig = model(X, H, W_e)
#         logits_adv = model_pois(X_adv,H_adv,W_e)
#     # acc_orig = (logits_orig.argmax(dim=1) == labels).float().mean().item()
#     # acc_adv = (logits_adv.argmax(dim=1) == labels).float().mean().item()
#     acc_orig_test = accuracy(logits_orig[test_mask],labels[test_mask])
#     acc_adv_test = accuracy(logits_adv[test_mask],labels[test_mask])
#     acc_orig_val = accuracy(logits_orig[val_mask],labels[val_mask])
#     acc_adv_val = accuracy(logits_adv[val_mask],labels[val_mask])
#     acc_orig_train = accuracy(logits_orig[train_mask],labels[train_mask])
#     acc_adv_train = accuracy(logits_adv[train_mask],labels[train_mask])
#     accuracydict = {'clean':{'train':acc_orig_train,'test':acc_orig_test,'val':acc_orig_val},\
#                     'adv':{'train':acc_adv_train,'test':acc_adv_test,'val':acc_adv_val}}
#     return accuracydict, (accuracydict['clean']['test'] - accuracydict['adv']['test'])/accuracydict['clean']['test']
def save_npz(root, seed, results):
    root = os.path.join(root, str(seed))
    loss_meta_trajectory, acc_drop_trajectory, lap_shift_trajectory, lap_dist_trajectory, cls_loss_trajectory, \
               deg_penalty_trajectory,feature_shift_trajectory,surrogate_test_trajectory, target_test_trajectory = results
    np.savez(os.path.join(root, 'loss_meta_trajectory.npz'),loss_meta_trajectory)
    np.savez(os.path.join(root, 'acc_drop_trajectory.npz'), acc_drop_trajectory)
    np.savez(os.path.join(root, 'lap_shift_trajectory.npz'), lap_shift_trajectory)
    np.savez(os.path.join(root, 'lap_dist_trajectory.npz'), lap_dist_trajectory)
    np.savez(os.path.join(root, 'cls_loss_trajectory.npz'), cls_loss_trajectory)
    np.savez(os.path.join(root, 'deg_penalty_trajectory.npz'), deg_penalty_trajectory)
    np.savez(os.path.join(root, 'feature_shift_trajectory.npz'), feature_shift_trajectory)
    np.savez(os.path.join(root, 'surrogate_test_trajectory.npz'), surrogate_test_trajectory)
    np.savez(os.path.join(root, 'target_test_trajectory.npz'), target_test_trajectory)
def save_to_csv(results, filename='results.csv'):
    # Convert results into a DataFrame for better handling
    df = pd.DataFrame(results, columns=sorted(list(results.keys())), index=[0])
    
    # Append to the CSV file, creating a new one if it doesn't exist
    if os.path.isfile(filename):
        existing_df = pd.read_csv(filename)
        if not existing_df.columns.equals(df.columns):
            raise ValueError("Column mismatch between results and existing CSV file.")
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)

def train_model(args, model, H, HG, X, y, W_e = None):
    print('---- Model Training -----')
    if args.model in ['hypergcn','unisage','unigin','unigat','unigcn']:
        H = HG
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    num_epochs = args.num_epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        if W_e is None:
            logits = model(X, H)
        else:
            logits = model(X,H,W_e)
        loss = criterion(logits, y)  # assuming y is your target labels
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"Epoch {epoch}: Loss = {loss.item()}, Accuracy = {acc * 100}%")

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_model_separate(args, model, H, HG, X, y, train_mask, val_mask, test_mask, W_e = None):
    print('---- Model Training -----')
    if args.model in ['hypergcn','unisage','unigin','unigat','unigcn']:
        H = HG
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs
    num_epochs = args.num_epochs
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        if W_e is None:
            logits = model(X, H)
        else:
            logits = model(X,H,W_e)
        loss = criterion(logits[train_mask], y[train_mask])  # assuming y is your target labels
        train_loss= loss.item()
        test_loss = F.cross_entropy(logits[test_mask], y[test_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # acc = (logits[test_mask].argmax(dim=1) == y[test_mask]).float().mean().item()
            # print(acc)
            acc = accuracy(logits[test_mask],y[test_mask])
            print(f"Epoch {epoch}: Train_Loss = {train_loss}, Test_loss = {test_loss}, Test accuracy = {acc * 100}%")

def train_with_early_stopping(args, model, H, HG, X, y, train_mask, val_mask, test_mask, W_e=None):
    print('---- Model Training with Early Stopping -----')
    
    if args.model in ['hypergcn']:
        H = HG

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    num_epochs = args.num_epochs
    patience = args.patience if hasattr(args, 'patience') else 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        if W_e is None:
            logits = model(X, H)
        else:
            logits = model(X, H, W_e)

        loss = criterion(logits[train_mask], y[train_mask])
        train_loss = loss.item()
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            if W_e is None:
                logits = model(X, H)
            else:
                logits = model(X, H, W_e)
            val_loss = criterion(logits[val_mask], y[val_mask])
            test_loss = criterion(logits[test_mask], y[test_mask])
            acc = accuracy(logits[test_mask], y[test_mask])

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train_Loss = {train_loss:.4f}, Val_Loss = {val_loss:.4f}, Test_Loss = {test_loss:.4f}, Test Accuracy = {acc * 100:.2f}%")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Load best model state (optional but common)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


# def get_dataset(args, device):
#     if args.dataset == 'co-citeseer': #cocitation-citeseer
#         data = CocitationCiteseer()
#     if args.dataset == 'coauth_cora':
#         data = CoauthorshipCora()
#     if args.dataset == 'coauth_dblp':
#         data = CoauthorshipDBLP()
#     if args.dataset == 'co-cora': # cocitation_cora
#         data = CocitationCora()
#     if args.dataset == 'co-pubmed': #cocitation-pubmed
#         data = CocitationPubmed()
#     if args.dataset == 'yelp':
#         data = YelpRestaurant()
#     if args.dataset == 'yelp3k':
#         data = Yelp3k()
#         train_ratio, val_ratio, test_ratio = 0.1, 0.1, 0.8
#         num_v = data["labels"].shape[0]
#         train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
#         # train_mask = train_mask.to(device)
#         # val_mask = val_mask.to(device)
#         # test_mask = test_mask.to(device)
#         val_mask = test_mask | val_mask
#         val_mask = val_mask.to(device)
#         test_mask = val_mask.to(device)
#         data["train_mask"] = train_mask
#         data["val_mask"] = val_mask
#         data["test_mask"] = test_mask

#     if args.dataset == "cooking":
#         data = Cooking200()
#         train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
#         num_v = data["labels"].shape[0]
#         train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
#         # data["train_mask"] = train_mask
#         # data["val_mask"] = val_mask
#         # data["test_mask"] = test_mask
#         val_mask = test_mask | val_mask
#         val_mask = val_mask.to(device)
#         test_mask = val_mask.to(device)
#         data["train_mask"] = train_mask
#         data["val_mask"] = val_mask
#         data["test_mask"] = test_mask
#     if args.dataset == 'tencent2k':
#         data = Tencent2k()
#         # random.seed(1)
#         # train_ratio, val_ratio, test_ratio = 0.5, 0.1, 0.4
#         # num_v = data["labels"].shape[0]
#         # train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
#         # train_mask = train_mask.to(device)
#         # val_mask = test_mask | val_mask
#         # val_mask = val_mask.to(device)
#         # test_mask = val_mask.to(device)
#         # data["train_mask"] = train_mask
#         # data["val_mask"] = val_mask
#         # data["test_mask"] = test_mask
#     if args.dataset == 'walmart':
#         data = WalmartTrips()
#         train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
#         num_v = data["labels"].shape[0]
#         train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
#         train_mask = train_mask.to(device)
#         val_mask = val_mask.to(device)
#         test_mask = test_mask.to(device)
#     if args.dataset == 'house':
#         data = HouseCommittees()
#         train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
#         num_v = data["labels"].shape[0]
#         train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
#         train_mask = train_mask.to(device)
#         val_mask = val_mask.to(device)
#         test_mask = test_mask.to(device)
#     if args.dataset == 'news20':
#         data = News20()
#         num_v = data["labels"].shape[0]
#         train_ratio, val_ratio, test_ratio = 0.1, 0.1, 0.8
#         train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
#         # train_mask = train_mask.to(device)
#         # val_mask = val_mask.to(device)
#         # test_mask = test_mask.to(device)
#         val_mask = test_mask | val_mask
#         val_mask = val_mask.to(device)
#         test_mask = val_mask.to(device)
#         data["train_mask"] = train_mask
#         data["val_mask"] = val_mask
#         data["test_mask"] = test_mask

#     labels = data["labels"].to(device)

#     if 'features' not in data.content:
#         print(args.dataset,' does not have features.')
#         features = torch.ones(data["num_vertices"],1).to(device)
#     else:
#         features = data["features"].to(device)
#     if "train_mask" in data.content:
#         train_mask = data["train_mask"].to(device)
#     if "val_mask" in data.content:
#         val_mask = data["val_mask"].to(device)
#     if "test_mask" in data.content:
#         test_mask = data["test_mask"].to(device)
#     return data, features, labels, train_mask, val_mask, test_mask

def get_dataset(args, device, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
    if args.dataset == 'co-citeseer':  # cocitation-citeseer
        data = CocitationCiteseer()
    elif args.dataset == 'coauth_cora':
        data = CoauthorshipCora()
    elif args.dataset == 'coauth_dblp':
        data = CoauthorshipDBLP()
    elif args.dataset == 'co-cora':  # cocitation_cora
        data = CocitationCora()
    elif args.dataset == 'co-pubmed':  # cocitation-pubmed
        data = CocitationPubmed()
    elif args.dataset == 'yelp':
        data = YelpRestaurant()
    elif args.dataset == 'yelp3k':
        data = Yelp3k()
    elif args.dataset == "cooking":
        data = Cooking200()
    elif args.dataset == 'tencent2k':
        data = Tencent2k()
    elif args.dataset == 'walmart':
        data = WalmartTrips()
    elif args.dataset == 'house':
        data = HouseCommittees()
    elif args.dataset == 'news20':
        data = News20()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_v = data["labels"].shape[0]
    train_mask, val_mask, test_mask = split_by_ratio(num_v, data["labels"], train_ratio, val_ratio, test_ratio)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    data["train_mask"] = train_mask
    data["val_mask"] = val_mask
    data["test_mask"] = test_mask

    labels = data["labels"].to(device)

    if 'features' not in data.content:
        print(args.dataset, 'does not have features.')
        features = torch.ones(data["num_vertices"], 1).to(device)
    else:
        features = data["features"].to(device)

    return data, features, labels, train_mask, val_mask, test_mask

def incidence_matrix_to_edge_list(incidence_matrix):
    """
    Convert a |V| x |E| incidence matrix to an edge list.

    Parameters:
    incidence_matrix (torch.Tensor): A |V| x |E| tensor representing the incidence matrix of a hypergraph.

    Returns:
    List[Tuple]: A list of tuples, where each tuple contains the nodes that form a hyperedge.
    """
    num_nodes, num_hyperedges = incidence_matrix.shape
    edge_list = []

    for e in range(num_hyperedges):
        # Find the nodes that are part of hyperedge e
        nodes_in_hyperedge = torch.where(incidence_matrix[:, e] == 1)[0].tolist()
        # Create a tuple of nodes and add it to the edge list
        edge_list.append(tuple(nodes_in_hyperedge))

    return edge_list

# def filter_potential_singletons(modified_H):
#         """
#         Computes a mask for entries potentially leading to singleton nodes, i.e., nodes whose degree is 1.
#         Prevents modifications that could lead to singleton nodes.
        
#         :param H: Hypergraph incidence matrix (|V| x |E|)
#         :return: A mask with the same shape as H, where entries that would lead to singletons are set to 0.
#         """

#         degrees = modified_H.sum(1)
#         # degree_one = ((degrees == 1) | (degrees == 0))
#         degree_one = (degrees<=1)
#         # We need to create a mask of shape (|V|, |E|) where nodes with degree 1 have their entries set to 0
#         resh = degree_one.unsqueeze(1).repeat(1, modified_H.shape[1])  # Shape (|V|, |E|)
        
#         degree_one = (modified_H.sum(0)<=2)
#         resh2 = degree_one.repeat(1,modified_H.shape[0]).reshape(modified_H.shape[0],-1)  # Shape (|V|, |E|)

#         l_and = (resh | resh2).float() * modified_H 
#         flat_mask = 1 - l_and
#         return flat_mask

def compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y):
    results = {}
     # Laplacian Frobenius norm change
    results['laplacian_norm'] = laplacian_diff(H, H_adv)

    # Embedding shift (ΔZ Fro norm)
    results['embedding_shift'] = embedding_shift(Z_orig, Z_adv)

    # Stealthiness measures
    h_l0, x_linf, deg_shift_l1, edge_card_shift_l1, deg_shift_l2, edge_card_shift_l2, deg_shift_linf, edge_card_shift_linf = measure_stealthiness(H, H_adv, X, X_adv)
    results.update({
        'h_l0': h_l0,
        'x_linf': x_linf,
        'deg_shift_l1': deg_shift_l1,
        'edge_card_shift_l1': edge_card_shift_l1,
        'deg_shift_l2': deg_shift_l2,
        'edge_card_shift_l2': edge_card_shift_l2,
        'deg_shift_linf': deg_shift_linf,
        'edge_card_shift_linf': edge_card_shift_linf
    })
    # Semantic change in features
    results['semantic_change'] = semantic_feature_change(X, X_adv)

    # Embedding sensitivity vs node degree (Pearson r)
    results['degree_sensitivity'] = degree_sensitivity(H, Z_orig, Z_adv)

    # Classification accuracy before and after attack
    labels = y
    logits_orig = Z_orig
    logits_adv = Z_adv

    acc_orig_test = accuracy(logits_orig[test_mask], labels[test_mask])
    acc_adv_test = accuracy(logits_adv[test_mask], labels[test_mask])
    acc_orig_val = accuracy(logits_orig[val_mask], labels[val_mask])
    acc_adv_val = accuracy(logits_adv[val_mask], labels[val_mask])
    acc_orig_train = accuracy(logits_orig[train_mask], labels[train_mask])
    acc_adv_train = accuracy(logits_adv[train_mask], labels[train_mask])

    acc_dict = {
        'clean_train': acc_orig_train.item(), 'clean_test': acc_orig_test.item(), 'clean_val': acc_orig_val.item(),
        'adv_train': acc_adv_train.item(), 'adv_test': acc_adv_test.item(), 'adv_val': acc_adv_val.item()}
    results.update(acc_dict)

    acc_drop = (acc_dict['clean_test'] - acc_dict['adv_test']) / acc_dict['clean_test']
    results['acc_drop%'] = acc_drop * 100
    return results 

def evasion_setting_evaluate(args, H, X, y, Z_orig, H_adv, X_adv, H_adv_HG, model, poisoned_model,train_mask,val_mask,test_mask, draw=False):
    print('================ Evasion setting =================')
    if args.model in ['hypergcn']:
        Z_adv = poisoned_model(X_adv, H_adv_HG).detach()
    else:
        Z_adv = poisoned_model(X_adv, H_adv).detach()
    
    results = compute_statistics(H,H_adv,Z_orig,Z_adv,X,X_adv,train_mask,val_mask,test_mask,y)
    if args.verbose: 
        print("Laplacian Frobenius norm change:", results['laplacian_norm'])
        print("Embedding shift (ΔZ Fro norm):", results['embedding_shift'])
        print("Structural L0 perturbation:", results['h_l0'])
        print("Feature L-infinity perturbation:", results["x_linf"])
        print("Total shift in degree distribution (Linf):", results["deg_shift_linf"])
        print("Total shift in degree distribution (L1):", results["deg_shift_l1"])
        print("Total shift in degree distribution (L2):",results["deg_shift_l2"])
        print("Total shift in edge-cardinality distribution (Linf):", results["edge_card_shift_linf"])
        print("Total shift in edge-cardinality distribution (L1):", results["edge_card_shift_l1"])
        print("Total shift in edge-cardinality distribution (L2):", results["edge_card_shift_l2"])

        print("Semantic change in features (1 - avg. cosine):", results['semantic_change'])
        print("Embedding sensitivity vs node degree (Pearson r):", results["degree_sensitivity"])
    
    # Return the results dictionary
    return results

def canonical_boundary_matrix(H_bin):
    B1 = torch.zeros_like(H_bin)  # Initialize B1 with zeros (same shape as H_bin)
    
    for j in range(H_bin.shape[1]):  # Iterate over each hyperedge (columns of H_bin)
        nodes = torch.where(H_bin[:, j] > 0)[0]  # Get indices of nodes in the j-th hyperedge
        
        if len(nodes) >= 2:  # Only consider hyperedges with more than one node
            min_idx = nodes[0]  # The first node in the sorted list (smallest index)
            for i in nodes[1:]:  # Iterate over the remaining nodes
                B1[i, j] = 1  # Assign positive orientation to nodes after the first one
            B1[min_idx, j] = -1  # Assign negative orientation to the first node in sorted order
            
    return B1

def hodge_laplacian_L0(B1):
    return B1 @ B1.T
