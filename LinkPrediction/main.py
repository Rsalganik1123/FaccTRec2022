import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import os 
import time
import numpy as np
import pickle 
import sys 
import wandb
from ogb.linkproppred import DglLinkPropPredDataset
import scipy.io as sio
import scipy.sparse as sp
import numpy as np 
from typing import NamedTuple
from torch import optim
import tqdm 
from sklearn.metrics import roc_auc_score, average_precision_score
from dgllife.utils import EarlyStopping
# from torch.nn import Sigmoid 

from utils.data_loading import * 
from utils.pre_processing import *
from utils.fairness import * 
from utils.misc import * 
# from launchers.args_LP import * 
# from launchers.gridsearch_LP import * 
# from model.GCN_LP import *  
# from model.SAGE_LP import * 

all_ndcg_list_train = []
auc = []
ap = []

def GCN_train(mode, epoch, model, optimizer, features, adj_norm, adj_label, pos_weight, adj_orig, val_edges, val_edges_false, top_k): 
    lambdas_para = 1 
    model.train()
    optimizer.zero_grad() 
    recovered, embedding = model(features, adj_norm)
    loss = loss_function_gcn(preds=recovered, labels=adj_label,
                        pos_weight=pos_weight)
    cur_loss = loss.item()
    if logging: 
        wandb.log({'epoch': epoch, "train_loss": cur_loss, })
    if mode == 'fairness': 
        loss.backward(retain_graph=True)
        avg_ndcg, pred_sim, lambdas = train_fair_2(top_k, model, optimizer, features, embedding)
        pred_sim.backward(1 * lambdas)
        optimizer.step()
        fairness_loss = torch.sum(pred_sim*lambdas)
        print("Epoch: {} , Batch loss:{}, Fairness loss:{}".format(epoch, cur_loss, fairness_loss.item())) 
        all_ndcg_list_train.append(avg_ndcg)
    else: 
        loss.backward()
        print("Epoch: {} , Batch loss:{}".format(epoch, cur_loss))
    optimizer.step()

    roc_curr, ap_curr = get_roc_score_GCN(recovered.cpu().detach().numpy(), adj_orig, val_edges, val_edges_false)
    if logging: 
        wandb.log({'epoch': epoch, 'valid_auc': roc_curr, 'valid_ap': ap_curr})
    print("Epoch: {} , Valid loss:{}, Valid AUC:{}, Valid AP:{}".format(epoch, cur_loss, roc_curr, ap_curr))
    
    ap.append(ap_curr)
    auc.append(roc_curr)

def GCN_test(model, features, adj_norm, adj_label, pos_weight, adj_orig, test_edges, test_edges_false): 
    model.eval()
    recovered, embedding = model(features, adj_norm)
    loss = loss_function_gcn(preds=recovered, labels=adj_label,pos_weight=pos_weight)
    roc_score, ap_score = get_roc_score_GCN(recovered.cpu().detach().numpy(), adj_orig, test_edges, test_edges_false)
    return roc_score, ap_score 

def GCN_loop(args, wandb_configs=None): 
    print("***GCN TRAIN -- EXP: {}***".format(args.exp_name))
    adj, features = load_LP_REDRESS_dataset(args.dataset, args.n_components) 
    n_nodes, feat_dim = features.shape
    print("***Loaded features for dataset:{} of shape:{}".format(args.dataset, features.shape))
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_LP_presplit(args.dataset)
    adj_orig, adj_norm, adj_label, pos_weight = process_GCN_dataset_LP(adj, features, adj_train)
    model = GCN_inform(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("***Loaded model:{}***".format(model))

    model.cuda()
    features = features.cuda()  
    adj_norm = adj_norm.cuda()  
    adj_label = adj_label.cuda()
    pos_weight = pos_weight.cuda() 
    
    exp_folder = '/home/mila/r/rebecca.salganik/scratch/ReDress/GCN/{}/'.format(args.exp_name)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    _, embeddings = model(features, adj_norm)
    init_fairness, _, _ = train_fair_2(args.top_k, model, [], features, embeddings)
    print("init_fairness", init_fairness)

    print("***UTILITY MODE***")
    for e in range(args.pretrain_epochs): 
        GCN_train('utility', e, model, optimizer, features, adj_norm, adj_label,  pos_weight, adj_orig, val_edges, val_edges_false, args.top_k)
        save_state = {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if saving: 
            backup_fpath = os.path.join(  exp_folder, 'u_model_bak_%06d.pt' % (e,)) 
            print("saving checkpoint for epoch:{} to:{}".format(e, backup_fpath))
            torch.save(save_state, backup_fpath)
 
    _, utility_embeddings = model(features, adj_norm)
    fairness_before, _, _ = train_fair_2(args.top_k, model, [], features, utility_embeddings)
    
    print("fairness before", fairness_before)
    if saving: 
        pickle.dump(utility_embeddings, open(os.path.join(exp_folder,'utility_emb.pkl'), "wb"))
    utility_test_auc, utility_test_ap = GCN_test(model, features, adj_norm, adj_label, pos_weight, adj_orig, test_edges, test_edges_false)
    print("***FAIRNESS MODE***")
    for e in range(args.epochs): 
        GCN_train('fairness', e, model, optimizer, features, adj_norm, adj_label, pos_weight, adj_orig, val_edges, val_edges_false, args.top_k)
        save_state = {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if saving: 
            backup_fpath = os.path.join( exp_folder, 'u+f_model_bak_%06d.pt' % (e,)) 
            print("saving checkpoint for epoch:{} to:{}".format(e, backup_fpath))
            torch.save(save_state, backup_fpath) 

    _, fairness_embeddings = model(features, adj_norm)
    fairness_after, _, _ = train_fair_2(args.top_k, model, [], features, fairness_embeddings)
    
    if saving: 
        pickle.dump(fairness_embeddings, open(os.path.join(exp_folder,'fair_emb.pkl'), "wb"))

    fair_test_auc, fair_test_ap = GCN_test(model, features, adj_norm, adj_label, pos_weight, adj_orig, test_edges, test_edges_false)

    print("****RESULTS****")
    print('GCN_auc' + str(utility_test_auc))
    print('GCN+ReDress_auc' + str(fair_test_auc))
    print('GCN_ap' + str(utility_test_ap))
    print('GCN+ReDress_ap' + str(fair_test_ap))
    print('init_fairness' + str(init_fairness))
    print('GCN_fairness' + str(fairness_before))
    print('GCN+ReDress_fairness' + str(fairness_after))

    results = {
        'auc_before': utility_test_auc, 
        'auc_after' : fair_test_auc, 
        'GCN_ap': utility_test_ap, 
        'GCN+ReDress_ap': fair_test_ap, 
        'init_fairness': init_fairness, 
        'GCN_fairness' : fairness_before, 
        'GCN+ReDress_fairness': fairness_after}

    save_results(args.exp_name, args, results, wandb_configs)

def SAGE_train(mode, epoch, model, opt, train_dataloader, val_dataloader, top_k, sigma_1, similarity, graph):
    model.train()
    batch_losses, fairness_losses =  [], []  #train_auc, train_ap = [], []
    if mode == 'fairness':
        for i in tqdm.tqdm(range(5)): 
            opt.zero_grad() 
            input_nodes, pair_graph, neg_pair_graph, blocks = next(iter(train_dataloader))
            x = blocks[0].srcdata['feat']
            embedding, pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            pos_label, neg_label = torch.ones_like(pos_score), torch.zeros_like(neg_score)
            score, labels = torch.cat([pos_score, neg_score]), torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            loss.backward(retain_graph=True)
            features, embeddings = blocks[-1].dstdata['feat'], model.return_embeddings(blocks, 'cuda')
            # if 'jaccard' in similarity:
            #     sg = dgl.merge(blocks)
            #     features = sg.ndata['feat']
            avg_ndcg, pred_sim, lambdas = train_fair_2(top_k, model, opt, features, embeddings, sigma_1, similarity, blocks[-1].dstnodes(), graph)
            pred_sim.backward(1 * lambdas)
            fairness_loss = torch.sum(pred_sim*lambdas) 
            fairness_losses.append(avg_ndcg)
            opt.step() 
    if mode == 'utility':
        for input_nodes, pair_graph, neg_pair_graph, blocks in tqdm.tqdm(train_dataloader):
            x = blocks[0].srcdata['feat']
            embedding, pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            pos_label, neg_label = torch.ones_like(pos_score), torch.zeros_like(neg_score)
            score, labels = torch.cat([pos_score, neg_score]), torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            batch_losses.append(loss.item())
            opt.step()    
            fairness_losses = [0]
    print("Epoch:{}, Batch Loss:{}, Fairness Loss:{} ".format(epoch, np.mean(batch_losses), np.mean(fairness_losses)))
    if logging: 
        wandb.log({'epoch': epoch, "train_loss": np.mean(batch_losses), 'fairness_loss': np.mean(fairness_losses)})
    
    valid_loss, valid_ap, valid_auc = [] , [], [] 
    model.eval()  
    for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(val_dataloader):
        x = blocks[0].srcdata['feat']
        embedding, pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
        pos_label, neg_label = torch.ones_like(pos_score), torch.zeros_like(neg_score)
        score, labels = torch.cat([pos_score, neg_score]), torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(score, labels)
        roc_score = roc_auc_score(labels.cpu(), torch.sigmoid(score).cpu().detach().numpy())
        ap_score = average_precision_score(labels.cpu(), torch.sigmoid(score).cpu().detach().numpy())
        valid_loss.append(loss.item())
        valid_auc.append(roc_score)
        valid_ap.append(ap_score)
    print("Epoch:{}, Valid Loss:{}, Valid AUC:{}, Valid AP:{}".format(epoch, np.mean(valid_loss), np.mean(valid_auc), np.mean(valid_ap)))
    if logging: 
        wandb.log({'epoch': epoch, "valid_loss": np.mean(valid_loss), 'valid_auc': np.mean(valid_auc), 'valid_ap': np.mean(valid_ap)})

    ap.append(np.mean(valid_ap))
    auc.append(np.mean(valid_auc))
    return np.mean(valid_auc) , np.mean(valid_ap)

def SAGE_test(model, opt, test_dataloader):
    model.eval() 
    test_loss, test_auc, test_ap = [] , [] , []  
    for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(test_dataloader):
        x = blocks[0].srcdata['feat']
        embedding, pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
        pos_label, neg_label = torch.ones_like(pos_score), torch.zeros_like(neg_score)
        score, labels = torch.cat([pos_score, neg_score]), torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(score, labels)
        roc_score = roc_auc_score(labels.cpu(), torch.sigmoid(score).cpu().detach().numpy())
        ap_score = average_precision_score(labels.cpu(), torch.sigmoid(score).cpu().detach().numpy())
        test_loss.append(loss.item())
        test_auc.append(roc_score)
        test_ap.append(ap_score)
    return np.mean(test_auc), np.mean(test_ap)

def SAGE_loop(args, wandb_configs=None): 
    print("***GS TRAIN***")
    adj, features = load_LP_REDRESS_dataset(args.dataset, args.n_components)
    g = adj_to_graph(adj, features) 
    print("***Loaded graph for dataset:{} of size: {} (nodes) {} (edges)".format(args.dataset, g.num_nodes(), g.num_edges()))
    sigma_tuned = 1
    if args.fine_tune: 
        sigma_tuned = get_sigma(args) 
    # similarity = simi if 'cosine' in args.similarity else jaccard_simi
    # print("similarity metric:{}, sigma_tuned:{}".format(similarity, sigma_tuned))
    
    device = 'cuda'
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_LP_presplit(args.dataset)
    train_g, val_g, test_g = build_edge_subgraph(g, [train_edges, val_edges, test_edges])
    model = SAGE(g.ndata['feat'].shape[1], args.hidden).to(device) 
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    sampler = dgl.dataloading.NeighborSampler(args.layer_neighbors, prefetch_node_feats=['feat'])
    sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(args.neg_neighbors))
    
    train_dataloader = dgl.dataloading.DataLoader(
            train_g, torch.arange(train_g.num_edges()), sampler,
            device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=False)
    val_dataloader = dgl.dataloading.DataLoader(
            val_g,  torch.arange(val_g.num_edges()), sampler,
            device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=False)
    test_dataloader = dgl.dataloading.DataLoader(
            test_g, torch.arange(test_g.num_edges()), sampler,
            device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_uva=False)

    exp_folder = '/home/mila/r/rebecca.salganik/scratch/ReDress/GS/{}/'.format(args.exp_name)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    
    
    init_fairness, _ = GS_all_fairness_LP(model, g, args.top_k, sigma_tuned, args.similarity)
    stopper = EarlyStopping(mode='higher', patience = 3)
    print("***UTILITY MODE***")
    best_valid_auc, best_valid_ap = 0, 0  
    for e in range(args.pretrain_epochs): 
        val_auc, val_ap = SAGE_train('utility', e, model, opt, train_dataloader, val_dataloader, args.top_k, sigma_tuned, args.similarity, g)
        save_state = {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
        }
        if saving: 
            backup_fpath = os.path.join(  exp_folder, 'u_model_bak_%06d.pt' % (e,)) 
            print("saving checkpoint for epoch:{} to:{}".format(e, backup_fpath))
            torch.save(save_state, backup_fpath)
        if early_stopping: 
            print("stopping after {} epochs with val auc:{}".format(e, best_valid_auc)) 
            early_stop = stopper.step(val_auc, model)
            if early_stop: 
                args = args._replace(pretrain_epochs=e)
                backup_fpath = os.path.join( exp_folder, 'u_model_bak_%06d.pt' % (e,)) 
                print("saving checkpoint for epoch:{} to:{}".format(e, backup_fpath))
                torch.save(save_state, backup_fpath)
                break 
        best_valid_auc, best_valid_ap = val_auc, val_ap 
    fairness_before, utility_embeddings = GS_all_fairness_LP(model, g, args.top_k, sigma_tuned, args.similarity)
    print("Fairness before", fairness_before)

    if saving_emb: 
        pickle.dump(utility_embeddings, open(os.path.join(exp_folder,'utility_emb.pkl'), "wb"))

    utility_test_auc, utility_test_ap = SAGE_test(model, opt, test_dataloader) 
    print("***FAIRNESS MODE***")
    for e in range(args.epochs): 
        SAGE_train('fairness', e, model, opt, train_dataloader, val_dataloader, args.top_k, sigma_tuned, args.similarity, g)
        save_state = {
            "epoch": e + 1,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
        }
        if saving: 
            backup_fpath = os.path.join( exp_folder, 'u+f_model_bak_%06d.pt' % (e,)) 
            print("saving checkpoint for epoch:{} to:{}".format(e, backup_fpath))
            torch.save(save_state, backup_fpath) 

    fairness_after, fair_embeddings = GS_all_fairness_LP(model, g, args.top_k, sigma_tuned, args.similarity)
    if saving_emb: 
        pickle.dump(fair_embeddings, open(os.path.join(exp_folder,'fair_emb.pkl'), "wb"))
    # fair_test_auc, fair_test_ap = SAGE_test(model, opt, test_dataloader) 

    print("****RESULTS****")
    print('SAGE_auc:' + str(best_valid_auc))
    print('SAGE+ReDress_auc:' + str(auc[-1]))
    print('SAGE_ap:' + str(best_valid_ap))
    print('SAGE_ReDres_ap:' + str(ap[-1]))
    print('init_fairness'+ str(init_fairness)) 
    print('vanillaSAGE_fairness:' + str(fairness_before))
    print('ReDress+SAGE_fairness:' + str(fairness_after))

    results = {
        'SAGE_auc': best_valid_auc, 
        'SAGE+ReDress_auc' : auc[-1], 
        'SAGE_ap': best_valid_ap, 
        'SAGE_ReDres_ap': ap[-1], 
        'init_fairness': init_fairness,
        'vanillaSAGE_fairness' : fairness_before, 
        'ReDress+SAGE_fairness': fairness_after}

    save_results(args.exp_name, args, results, wandb_configs)
    
if __name__ == "__main__": 
    saving = True
    logging = False
    early_stopping = False 
    saving_emb = False
    args = get_args()
    # if 'batch' in sys.argv[-1]:  
    # args = get_args_full()
    
    if 'GCN' in args.exp_name : 
        # args = ArgsForGCN(seed = 42, pretrain_epochs=200, epochs=60, hidden1= 100 , hidden2=32, 
        #         top_k=10, lr= 0.01, dropout= 0.0, exp_name= sys.argv[-1], dataset = dataset)
        if logging: 
            wandb.init(
                project='ReDress', 
                name=args.exp_name,
                dir="/home/mila/r/rebecca.salganik/scratch/ReDress/",
                config = args._asdict()
            )
        GCN_loop(args,  wandb.run)
    elif 'GS' in args.exp_name: 
        # GS_args = ArgsForSAGE(epochs=60, hidden=256, lr= 0.001, weight_decay=0.001, layer_neighbors=[15, 15, 15], 
        #                 exp_name=sys.argv[-1], pretrain_epochs=50, neg_neighbors=5, top_k=10, dataset = dataset)            
        if logging: 
            wandb.init(
                project='ReDress', 
                name=args.exp_name,
                dir="/home/mila/r/rebecca.salganik/scratch/ReDress/",
                config = args._asdict()
            )
        SAGE_loop(args, wandb.run) 
    else: 
        print('ERROR: No model specified')
