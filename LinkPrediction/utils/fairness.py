import torch 
import numpy as np
import dgl 
from scipy.sparse.csgraph import laplacian  
from scipy.sparse import coo_matrix  
import tqdm 
import pickle

def idcg_computation(x_sorted_scores, top_k):
    c = 2 * torch.ones_like(x_sorted_scores)[:top_k]
    numerator = c.pow(x_sorted_scores[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:top_k].shape[0], dtype=torch.float)).cuda()
    final = numerator / denominator

    return torch.sum(final)

def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float))
    final = numerator / denominator

    return torch.sum(final)

def ndcg_exchange_abs(x_corresponding, j, k, idcg, top_k):
    new_score_rank = x_corresponding
    dcg1 = dcg_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    dcg2 = dcg_computation(new_score_rank, top_k)

    return torch.abs((dcg1 - dcg2) / idcg)

def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    # print("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()

def cosine_simi(output):  # new_version
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)
    
    return res

def lambdas_computation(x_similarity, y_similarity, top_k=10, sigma_1=1):
    # print("***lambdas computation***")
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    # ***************************** ranking ******************************
    # print("***ranking***")
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    # ***************************** pairwise delta ******************************
    # print("***pairwise delta***")
    sigma_tuned = sigma_1
    length_of_k = top_k #k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1 :(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    # ***************************** NDCG delta from ranking ******************************
    # print("***NDCG from ranking***")
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        if i >= 0.6 * y_similarity.shape[0]:
            break
        idcg = idcg_computation(x_sorted_scores[i, :], top_k)
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = ndcg_exchange_abs(x_corresponding[i, :], j, k, idcg, top_k)
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])
    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])   # 本来是 -


    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding

def jaccard_simi(adj):
    adj = torch.FloatTensor(adj.A).cuda()
    simi = torch.zeros_like(adj)
   
    for i in range(adj.shape[0]):
        one = torch.sum(adj[i, :].repeat(adj.shape[0], 1).mul(adj), axis=1)
        two = torch.sum(adj[i, :]) * torch.sum(adj, axis=1)
        simi[i,:] = ( one / two ).T
    return simi

def jaccard_simi_LP(adj):
    adj = torch.FloatTensor(adj.A).cuda()
    print(adj.shape)
    simi = torch.zeros((adj.shape[0], adj.shape[0])).cuda()
    ones = torch.ones_like(adj).cuda()
    zeros1 = torch.zeros_like(adj).cuda()
    zeros2 = torch.zeros_like(simi).cuda() 

    for i in range(adj.shape[0]):
        one = torch.sum(adj[i, :].mul(adj), axis=1)
        two = torch.sum(torch.where(adj[i, :].repeat(adj.shape[0], 1) + adj > 0, ones, zeros1), axis=1)
        simi[i,:] = ( one / two )

    simi = 10 * torch.where(torch.isnan(simi), zeros2, simi) 
    return simi.cuda()

def jaccard_simi2(nodes, g):
    simi = torch.zeros((len(nodes), len(nodes))) 
    zeros2 = torch.zeros_like(simi).cuda() 

    for i in nodes.cpu(): 
        for j in nodes.cpu(): 
            intersect = len(np.intersect1d(g.in_edges(i)[0], g.in_edges(j)[0])) 
            union = len(g.in_edges(i)[0]) + len(g.in_edges(j)[0]) - intersect
            simi[i][j] = intersect/union 
    simi = 10 * torch.where(torch.isnan(simi), zeros2, simi)
    return simi 

def train_fair_2(top_k, model, features, embedding, sigma_1, similarity, nodes=None, graph=None, mode ='sub'):
    lambdas_para = 1
    model.train()
    
    pred_sim = cosine_simi(embedding) # embedding: |V| x |hid|, Y_sim: |V| x |V| 
    
    if 'jaccard' in similarity:
        adj = graph.adj(scipy_fmt='coo', transpose=True).tocsc()
        apriori_sim = jaccard_simi(adj) #"coo" .to('cuda') # features: |V| x |feature|, x_sim: |V| x |V| 
    else: 
        apriori_sim = cosine_simi(features) # features: |V| x |feature|, x_sim: |V| x |V| 
    print(pred_sim.shape, apriori_sim.shape)
    lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(apriori_sim, pred_sim, top_k, sigma_1) 
    assert lambdas.shape == pred_sim.shape
    should_return = avg_ndcg(x_corresponding, apriori_sim, x_sorted_scores, y_sorted_idxs, top_k)
    return should_return, pred_sim, lambdas 

def GS_all_fairness_LP(args, model, graph) : 
    if 'cosine' in args.similarity: 
        embeddings = model.inference(graph, 'cuda', 4096, 1, 'cpu')
        pred_sim = cosine_simi(embeddings)
        features = graph.ndata['feat'].to('cuda')
        apriori_sim = cosine_simi(features)
    elif 'jaccard' in args.similarity: 
        print("JACCARD")
        embeddings = model.inference(graph, 'cuda', 4096, 1, 'cpu')
        adj = graph.adj(scipy_fmt='coo').tocsc() 
        pred_sim = cosine_simi(embeddings) 
        apriori_sim =jaccard_simi(adj).cuda() 
    else: 
        print("WRONG SIMILARITY FUNCTION")
        exit() 
    lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(apriori_sim, pred_sim, args.top_k, args.sigma) 
    assert lambdas.shape == pred_sim.shape
    should_return = avg_ndcg(x_corresponding, apriori_sim, x_sorted_scores, y_sorted_idxs, args.top_k)
    return should_return, embeddings 


def train_fair(args, model, subgraph=None, full_apriori=None):
    lambdas_para = 1
    model.train()
    if 'cosine' in args.similarity: 
        features, embeddings = subgraph[-1].dstdata['feat'], model.return_embeddings(subgraph, 'cuda')
        pred_sim = cosine_simi(embeddings)
        apriori_sim = cosine_simi(features)
    elif 'jaccard' in args.similarity:

        #Option 1: 
        embeddings = model.inference(subgraph, 'cuda', 4096, 0, 'cuda') #'cpu'
        adj = subgraph.adj(scipy_fmt='coo').tocsc() #transpose=True
        pred_sim = cosine_simi(embeddings) #jaccard_simi_LP(adj).cuda() #
        apriori_sim = jaccard_simi_LP(adj).cuda() #LP

        #Option 2: 
        # model.return_embeddings(subgraph, 'cuda')
        # nodes = subgraph[-1].dstnodes() 
        # pred_sim = cosine_simi(embeddings) #jaccard_lookup(full_apriori, nodes).cuda() #
        # apriori_sim = jaccard_lookup(full_apriori, nodes).cuda() 

        #Option 3: 
        # embeddings = model.return_embeddings(subgraph, 'cuda')
        # subgraph = dgl.merge([subgraph[-1]])
        # adj = subgraph.adj(scipy_fmt='coo').tocsc() #transpose=True
        # pred_sim = cosine_simi(embeddings) #jaccard_simi_LP(adj).cuda() #
        # apriori_sim = jaccard_simi(adj).cuda() #LP
    else: 
        print("WRONG SIMILARITY FUNC")
        exit() 
    print(pred_sim.shape, apriori_sim.shape)
    lambdas, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(apriori_sim, pred_sim, args.top_k, args.sigma) 
    assert lambdas.shape == pred_sim.shape
    should_return = avg_ndcg(x_corresponding, apriori_sim, x_sorted_scores, y_sorted_idxs, args.top_k)
    return should_return, pred_sim.cuda(), lambdas.cuda() 


def jaccard_lookup(grid, nodes): 
    nodes = nodes.cpu().numpy()
    pos = [] 
    for n1 in nodes: 
        for n2 in nodes: 
            pos.append([n1, n2])
    idx = np.array(pos).reshape(-1, 2).T.tolist() 
    submat = grid[tuple(idx)].reshape(len(nodes), len(nodes)) 
    # print("GRID:{}, NODES:{},  SUBMAT:{}".format(grid.shape, nodes.shape, submat.shape))
    return torch.tensor(submat).cuda() 