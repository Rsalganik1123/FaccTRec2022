from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np 
import torch 
import dgl 
import pickle as pkl 
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset

data_paths = {
    'BC': './data/BlogCatalog.mat',
    'F': './data//data/Flickr.mat',
    }
split_paths = {
    'BC': './data/BlogCatalog_split.pkl', 
    'F': './data/Flickr_split.pkl'
}

def load_LP_REDRESS_dataset(dataset_name, n_components): 
    print("***Loading {} dataset***".format(dataset_name))
    data_path = data_paths[dataset_name]
    data = sio.loadmat(data_path)
    features = data["Attributes"]
    adj = data["Network"] 
    if n_components != 0: 
        print("*** applying PCA to get {} components".format(n_components))
        pca = PCA(n_components)
        a = pca.fit_transform(np.array(features.todense()))
        return adj, torch.FloatTensor(a)
    else: 
        features = torch.FloatTensor(features.todense())
        return adj, features 

def load_LP_presplit(dataset_name):
    data = pkl.load(open(split_paths[dataset_name], "rb"))
    return data['adj_train'], data['train_edges'], data['val_edges'], data['val_edges_false'], data["test_edges"], data["test_edges_false"]

def adj_to_graph(adj, features, labels=None, idx=None,device=None):
    modes = ['train', 'val', 'test']
    g = dgl.from_scipy(adj)
    if device: 
        features = features.to(device)
        labels = labels.to(device)
        g = g.to(device)
    if idx: 
        for i in range(len(idx)):
            nl = idx[i]
            mask = np.zeros(g.num_nodes(), dtype=bool)
            mask[nl] = True
            name = '{}_mask'.format(modes[i])
            g.ndata[name] = torch.Tensor(mask)
    g.ndata['feat'] = features 
    print('loaded graph of size:{} (nodes), {} (edges)'.format(g.num_nodes(), g.num_edges()))
    return g 

def make_dataloaders(args, graph, sampler, idx_train, idx_val, idx_test, device='cuda'): 

    train_dataloader = dgl.dataloading.DataLoader(
            graph, torch.tensor(idx_train, dtype = torch.int64), sampler,
            device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=False)
    val_dataloader = dgl.dataloading.DataLoader(
            graph,  torch.tensor(idx_train, dtype = torch.int64), sampler,
            device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=False)
    test_dataloader = dgl.dataloading.DataLoader(
            graph,  torch.tensor(idx_train, dtype = torch.int64), sampler,
            device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=False)
    return train_dataloader, val_dataloader, test_dataloader 