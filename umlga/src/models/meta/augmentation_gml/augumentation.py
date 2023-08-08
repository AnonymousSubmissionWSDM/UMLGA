"""
@Filename       : augumentation.py
@Create Time    : 2022/6/3 15:26
@Author         :
@Description    : 

"""
import copy

import torch as th
import random
import numpy as np
import scipy.sparse as sp


def drop_feature(feature, drop_prob):
    drop_mask = th.empty((feature.size(1),), dtype=th.float32, device=feature.device).uniform_(0, 1) < drop_prob
    feature = feature.clone()
    feature[:, drop_mask] = 0

    return feature


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, th.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = th.bernoulli(drop_prob).to(th.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, th.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = th.bernoulli(drop_prob).to(th.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(th.bool).to(th.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, th.ones_like(edge_weights) * threshold)
    sel_mask = th.bernoulli(1. - edge_weights).to(th.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(th.float32)
    s_col = th.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(th.float32)
    pv_col = pv[edge_index[1]].to(th.float32)
    s_row = th.log(pv_row)
    s_col = th.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, th.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


def random_mask(feature, drop_percent=0.2):
    node_num = feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(feature)
    zeros = th.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def random_edge(adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def drop_node(feature, adj, drop_percent=0.2):
    adj = th.tensor(adj.todense().tolist())
    feature = feature.squeeze(0)

    node_num = feature.shape[0]
    drop_num = int(node_num * drop_percent)  # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(feature, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def subgraph(feature, adj, drop_percent=0.2):
    adj = th.tensor(adj.todense().tolist())
    feature = feature.squeeze(0)
    node_num = feature.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):

        all_neighbor_list += th.nonzero(adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(feature, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


def init_seed_nodes(x, class_num):
    c0_idx = int(np.random.uniform(0, len(x)))
    centroid = x[c0_idx].reshape(1, -1)  # select the first cluster centroid
    seed_nodes = [c0_idx]
    k = 1
    n = x.shape[0]
    while k < class_num:
        d2 = []
        for i in range(n):
            subs = centroid - x[i, :]
            dimension2 = np.power(subs, 2)
            dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            d2.append(np.min(dimension_s))
        new_c_idx = np.argmax(d2)
        seed_nodes.append(new_c_idx)
        centroid = np.vstack([centroid, x[new_c_idx]])
        k += 1
    return seed_nodes


def concat_batch_graph(batch_adjs, batch_features):
    features_dim = batch_features[0].shape[1]
    total_length = sum([len(g) for g in batch_adjs])
    adj = np.zeros((total_length, total_length))
    features = np.zeros((total_length, features_dim))
    cursor = 0
    for ba, bf in zip(batch_adjs, batch_features):
        batch_length = ba.shape[0]
        adj[cursor: cursor + batch_length, cursor: cursor + batch_length] = ba
        features[cursor: cursor + batch_length] = bf

        cursor += batch_length

    return adj, features


def sub_graph(adj, features, nid):
    # sample ego graph of from give nodes
    new_nid_dict = {nid: 0}
    max_new_nid = 1
    coo_1 = []
    coo_2 = []
    id_map = dict()
    first_order_neighs = np.nonzero(adj[nid])[0].tolist()
    for f_neigh in first_order_neighs:
        coo_1.append(0)
        if f_neigh in new_nid_dict.keys():
            coo_2.append(new_nid_dict[f_neigh])
        else:
            new_nid_dict[f_neigh] = max_new_nid
            coo_2.append(max_new_nid)
            max_new_nid += 1

        second_order_neighs = np.nonzero(adj[f_neigh])[0].tolist()
        for s_neigh in second_order_neighs:
            coo_1.append(new_nid_dict[f_neigh])
            if s_neigh in new_nid_dict.keys():
                coo_2.append(new_nid_dict[s_neigh])
            else:
                new_nid_dict[s_neigh] = max_new_nid
                coo_2.append(max_new_nid)
                max_new_nid += 1

    values = np.ones(len(coo_1))
    new_adj = sp.coo_matrix((values, (coo_1, coo_2)), shape=(max_new_nid, max_new_nid)).todense()
    new_adj = np.array(np.logical_or(new_adj, new_adj.transpose()), dtype=np.int32)
    new_features = np.zeros((new_adj.shape[0], features.shape[1]))

    for k, v in new_nid_dict.items():
        new_features[v] = features[k]

    return new_adj, new_features, id_map


def run_augmentation(adj, features, node_present=0.1, edge_present=0.2, feature_present=0.1):
    node_num = adj.shape[0]
    aug_features = copy.deepcopy(features)
    candidate_nodes = range(1, adj.shape[0])
    dropped_nodes = random.sample(candidate_nodes, int(node_present * node_num))
    aug_adj = delete_row_col(adj, dropped_nodes)

    return aug_adj, aug_features


def augmentation_task_generator(adj, features, pretrained, n_way, k_shot, n_query):
    # sample n_way nodes as diverse as possible

    seed_nodes = init_seed_nodes(pretrained, n_way)
    support_graphs = []
    query_graphs = []
    for nid in seed_nodes:
        graphs = []
        for _ in range(k_shot + n_query):
            raw_adj, raw_features = sub_graph(adj, features, nid)
            aug_adj, aug_features = run_augmentation(raw_adj, raw_features)
            graphs.append((aug_adj, aug_features))

        support_graphs.append(graphs[:k_shot])
        query_graphs.append(graphs[k_shot:])

    local_labels = np.array(range(n_way))
    return support_graphs, query_graphs, local_labels


if __name__ == '__main__':
    # features = np.random.rand(200, 128)
    # seed_nodes = init_seed_nodes(features, 5)
    # print(seed_nodes)

    # adj1 = np.zeros((3, 3)) + 1
    # adj2 = np.ones((4, 4)) + 2
    #
    # features1 = np.random.rand(3, 2)
    # features2 = np.random.rand(4, 2)
    # print(concat_batch_graph([adj1, adj2], [features1, features2]))

    adj1 = np.array([[0, 1, 0, 0], [1, 1, 1,0 ], [0, 1, 1,0 ], [0,0,0,1]])
    features1 = np.random.rand(4, 5)
    print(adj1)
    print(features1)
    print(sub_graph(adj1, features1, 3))