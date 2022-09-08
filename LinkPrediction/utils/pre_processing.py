import scipy.io as sio
import pickle as pkl
import itertools
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import preprocessing
import scipy.io as sio
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.decomposition import PCA


def normalize(mx): #GCN util
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_graph(adj): #GCN util
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def mask_test_edges(adj): #GCN util
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)

    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    num_test = int(np.floor(edges.shape[0] / 20.))
    num_val = int(np.floor(edges.shape[0] / 40.))


    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)


    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    print("started generating " + str(num_test) + " negative test edges...")

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])


    print("started generating " + str(num_val) + " negative validation edges...")
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
        # print(len(val_edges_false) / num_val)

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def get_roc_score(emb, adj_orig, edges_pos, edges_neg): #GCN util
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_roc_score_GCN(adj, adj_orig, edges_pos, edges_neg): #GCN util

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = adj
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])


    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def sparse_to_tuple(sparse_mx): #GCN util
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def build_edge_subgraph(full_g, split_list): #GS util
    modes = ['train', 'valid', 'test']
    subgraphs = [] 
    for i in range(len(split_list)): 
        edge_sample = split_list[i]
        us, vs, eids = full_g.edges('all')
        edge_format = np.array(list(zip(us.numpy(), vs.numpy())), dtype='int32')
        nrows, ncols = edge_format.shape
        dtype = (', '.join([str(edge_format.dtype)]*ncols))
        overlap = np.intersect1d(edge_format.view(dtype), edge_sample.view(dtype), return_indices=True)
        eids = overlap[1]
        subgraph = full_g.edge_subgraph(eids, relabel_nodes=False, store_ids=True)
        subgraphs.append(subgraph)
        print("***Generated {} subgraph of size: {} (edges)***".format(modes[i], subgraph.num_edges()))
    return subgraphs

def build_edge_subgraph(full_g, split_list): #GS util
    modes = ['train', 'valid', 'test']
    subgraphs = [] 
    for i in range(len(split_list)): 
        edge_sample = split_list[i]
        us, vs, eids = full_g.edges('all')
        edge_format = np.array(list(zip(us.numpy(), vs.numpy())), dtype='int32')
        nrows, ncols = edge_format.shape
        dtype = (', '.join([str(edge_format.dtype)]*ncols))
        overlap = np.intersect1d(edge_format.view(dtype), edge_sample.view(dtype), return_indices=True)
        eids = overlap[1]
        subgraph = full_g.edge_subgraph(eids, relabel_nodes=False, store_ids=True)
        subgraphs.append(subgraph)
        print("***Generated {} subgraph of size: {} (edges)***".format(modes[i], subgraph.num_edges()))
    return subgraphs
    
def build_node_subgraph(full_g, split_list): #GS util
    subgraphs = [] 
    modes = ['train', 'valid', 'test']
    for i in range(len(split_list)): 
        subgraph = full_g.subgraph(split_list[i])
        subgraphs.append(subgraph)
        print("***Generated {} subgraph of size: {} (nodes) {} (edges)***".format(modes[i], subgraph.num_nodes(), subgraph.num_edges()))
    return subgraphs 

def sparse_mx_to_torch_sparse_tensor(sparse_mx): #GCN util
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx): #GCN util
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize2(mx): #GCN util
    """Row-normalize sparse matrix"""
    rowmode = np.sqrt(np.array(mx.power(2).sum(1)))
    r_inv = np.power(rowmode, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def process_GCN_dataset_NC(adj, features, labels, idx_train, idx_val, idx_test, n_components): 
    
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    pca = PCA(n_components)
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)
    adj = sp.csc_matrix(adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    a = pca.fit_transform(np.array(features))
    simi = normalize2(sp.csr_matrix(features[idx_train, :], dtype=np.float32)).dot(normalize2(sp.csr_matrix(features[idx_train, :], dtype=np.float32)).T)
    d = np.array(simi.sum(axis=1).flatten())[0]
    l_s = np.diag(d) - simi

    features = torch.FloatTensor(a).to('cuda')
    labels = torch.LongTensor(np.where(labels)[1]).to('cuda')
    
    adj = sparse_mx_to_torch_sparse_tensor(adj).to('cuda')

    idx_train = torch.LongTensor(idx_train).to('cuda')
    idx_val = torch.LongTensor(idx_val).to('cuda')
    idx_test = torch.LongTensor(idx_test).to('cuda')

    return adj, features, labels, idx_train, idx_val, idx_test 

def process_GCN_dataset_LP(adj, features, adj_train): 
    
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    return adj_orig, adj_norm, adj_label, pos_weight

def row_normalize(mx): #LADIES util
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx

def package_mxl(mxl, device='cuda'): #LADIES util
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]

def sparse_mx_to_torch_sparse_tensor_LADIES(sparse_mx): #LADIES_tuil
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape 

def LADIES_prepare_data(sampler, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth, args): #LADIES_util
    train_batches = [] 
    for b in range(args.batch_num):  
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        subgraph = sampler(np.random.randint(2**32 - 1), batch_nodes, num_nodes, lap_matrix, depth,  args.samp_num*20)
        train_batches.append(subgraph)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    valid_batches = sampler(np.random.randint(2**32 - 1), batch_nodes, num_nodes, lap_matrix, depth, args.samp_num*20)
    return train_batches, valid_batches

def LADIES_all_data(sampler, nodes, samp_num, num_nodes, lap_matrix, depth, args, tensor=True): 
    entire_subgraph, output_nodes, input_nodes = sampler(np.random.randint(2**32 - 1), nodes, num_nodes, lap_matrix, depth, args.samp_num*20)
    if not tensor: 
        return entire_subgraph
    return package_mxl(entire_subgraph, 'cuda')

def ladies_sampler2(seed, batch_nodes, num_nodes, lap_matrix, depth, samp_num=64): #LADIES util 
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor_LADIES(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

