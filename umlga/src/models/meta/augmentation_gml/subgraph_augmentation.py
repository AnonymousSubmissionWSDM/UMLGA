"""
@Filename       : subgraph_augmentation.py
@Create Time    : 2022/11/7 22:32
@Author         :
@Description    : 

"""
import copy
import csv
import itertools
import random
from math import ceil

import dgl
import scipy.sparse as sp
import numpy as np
import os

import torch
from torch.utils.data import Dataset

import torch as th

from utils.data import load_data

class SubGraph(Dataset):
    def __init__(self, G, adj, class_list, id_by_class, n_way, k_shot, m_query, hop):
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot support set
        self.m_query = m_query  # for query set
        self.support_size = self.n_way * self.k_shot  # num of samples per support set
        self.query_size = self.n_way * self.m_query  # number of samples per set for evaluation
        self.hop = hop  # number of hops
        self.G = G
        self.adj = adj
        self.class_list = class_list
        self.id_by_class = id_by_class

    def generate_subgraph(self, i):
        G = self.G

        if self.hop == 2:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
            h_hops_neighbor = th.tensor(list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
        elif self.hop == 1:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            h_hops_neighbor = th.tensor(list(set(f_hop + [i]))).numpy()
        elif self.hop == 3:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
            n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
            h_hops_neighbor = th.tensor(
                list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
        if h_hops_neighbor.reshape(-1, ).shape[0] > 1000:
            h_hops_neighbor = np.random.choice(h_hops_neighbor, 1000, replace=False)
            h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))

        sub = G.subgraph(h_hops_neighbor).to('cuda')

        return sub

    def create_task(self, seed):
        random.seed(seed)

        n_way = self.n_way
        k_shot = self.k_shot
        m_query = self.m_query
        if len(self.class_list) == n_way:
            class_selected = self.class_list
        else:
            class_selected = random.sample(self.class_list, self.n_way)
        g_support = []
        center_nodes_support = []
        g_query = []
        center_nodes_query = []

        for cla in class_selected:
            temp = random.sample(self.id_by_class[cla], k_shot + m_query)
            for support_id in temp[:k_shot]:
                g = self.generate_subgraph(support_id)
                g_support.append(g)
                relabel_map = {origin_id: new_id for new_id, origin_id in enumerate(g.ndata[dgl.NID].cpu().numpy())}
                center_nodes_support.append(relabel_map[support_id])

            for query_id in temp[k_shot:]:
                g = self.generate_subgraph(query_id)
                g_query.append(g)
                relabel_map = {origin_id: new_id for new_id, origin_id in enumerate(g.ndata[dgl.NID].cpu().numpy())}
                center_nodes_query.append(relabel_map[query_id])

        random.seed()
        return g_support, center_nodes_support, g_query, center_nodes_query

    def __getitem__(self, idx):
        return self.create_task(idx)


if __name__ == '__main__':
    dataset = 'Amazon_clothing'
    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(
        '../../../../data', dataset)
    data = SubGraph(adj, features, class_list_test, id_by_class, n_way=5, k_shot=5, m_query=15, hop=2)
    g: dgl.DGLGraph = data[0][0][0]
    print(g.nodes[0])
    print(g.nodes[1])
    print(g.nodes[2])
    print(g.nodes[3])
    print(g.nodes[4])
    print(g.ndata[dgl.NID])
    print(g.number_of_nodes())
    print(g.ndata['x'])
    print(g.ndata['x'].shape)


class SubGraphAndAugmentation(Dataset):
    def __init__(self, G, adj, n_way, k_shot, m_query, hop, augmentation_method, augmentation_parameter):
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot support set
        self.m_query = m_query  # for query set
        self.support_size = self.n_way * self.k_shot  # num of samples per support set
        self.query_size = self.n_way * self.m_query  # number of samples per set for evaluation
        self.hop = hop  # number of hops
        self.G = G
        self.adj = adj
        self.augmentation_method_and_parameter = [(globals()[m], float(p)) for m, p in
                                                  zip(augmentation_method, augmentation_parameter)]
                                             # (drop_feature, 0.2),
                                             # (random_mask, 0.2),
                                             # (drop_edge, 0.2)]

    def generate_subgraph(self, i):
        G = self.G

        if self.hop == 2:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
            h_hops_neighbor = th.tensor(list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
        elif self.hop == 1:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            h_hops_neighbor = th.tensor(list(set(f_hop + [i]))).numpy()
        elif self.hop == 3:
            f_hop = [n.item() for n in G.in_edges(i)[0]]
            n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
            n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
            h_hops_neighbor = th.tensor(
                list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
        if h_hops_neighbor.reshape(-1, ).shape[0] > 1000:
            h_hops_neighbor = np.random.choice(h_hops_neighbor, 1000, replace=False)
            h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))

        sub = G.subgraph(h_hops_neighbor)

        return sub

    def create_task(self):
        n_way = self.n_way
        k_shot = self.k_shot
        m_query = self.m_query

        g_support = []
        center_nodes_support = []
        g_query = []
        center_nodes_query = []

        for _ in range(n_way):
            center_node = random.choice(range(self.adj.shape[0]))
            sampled_graph = self.generate_subgraph(center_node)
            relabel_map = {origin_id: new_id for new_id, origin_id in
                           enumerate(sampled_graph.ndata[dgl.NID].cpu().numpy())}
            center_nodes_new_id = relabel_map[center_node]
            temp = []
            am_list = []
            for _ in range(k_shot + m_query):
                augmentation_method, augmentation_parameter = random.choice(self.augmentation_method_and_parameter)
                temp.append(augmentation_method(sampled_graph, center_nodes_new_id, augmentation_parameter))
                am_list.append(augmentation_method)

            for augmentation_g, am in zip(temp[:k_shot], am_list[:k_shot]):
                g_support.append(augmentation_g)
                relabel_map = {origin_id: new_id for new_id, origin_id in enumerate(augmentation_g.ndata[dgl.NID].cpu().numpy())}
                center_nodes_support.append(relabel_map[center_nodes_new_id] if am == subgraph else center_nodes_new_id)

            for augmentation_g, am in zip(temp[k_shot:], am_list[k_shot:]):
                g_query.append(augmentation_g)
                relabel_map = {origin_id: new_id for new_id, origin_id in enumerate(augmentation_g.ndata[dgl.NID].cpu().numpy())}
                center_nodes_query.append((relabel_map[center_nodes_new_id] if am == subgraph else center_nodes_new_id))

        return g_support, center_nodes_support, g_query, center_nodes_query

    def __getitem__(self, idx):
        return self.create_task()


def drop_node(g, center, drop_percent=0.2):
    # to avoid the disorder of the node id, we choose to remove the
    # edges from nodes rather than directly remove the nodes
    augmentation_g = copy.deepcopy(g)
    node_num = augmentation_g.number_of_nodes()
    drop_num = int(node_num * drop_percent)  # number of drop nodes
    all_node_list = [i for i in range(node_num)]
    all_node_list.remove(center)

    drop_node_list = sorted(random.sample(all_node_list, drop_num))
    drop_out_edges_from, drop_out_edges_to = augmentation_g.out_edges(drop_node_list)
    drop_in_edges_from, drop_in_edges_to = augmentation_g.in_edges(drop_node_list)
    drop_edges_from = th.cat([drop_out_edges_from, drop_in_edges_from])
    drop_edges_to = th.cat([drop_out_edges_to, drop_in_edges_to])
    drop_edges_id = augmentation_g.edge_ids(drop_edges_from, drop_edges_to)
    augmentation_g.remove_edges(drop_edges_id)
    return augmentation_g

def subgraph(g, center, drop_percent=0.2):
    f_hop = [n.item() for n in g.in_edges(center)[0]]
    f_hop = np.random.choice(f_hop, size=ceil(len(f_hop) * drop_percent), replace=False).tolist()

    n_l = [[n.item() for n in g.in_edges(i)[0]] for i in f_hop]
    h_hops_neighbor = th.tensor(list(set(list(itertools.chain(*n_l)) + f_hop + [center]))).numpy()
    h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [center]))

    sub = g.subgraph(h_hops_neighbor)
    return sub

def drop_feature(g, center, drop_percent=0.2):
    augmentation_g = copy.deepcopy(g)
    feature = augmentation_g.ndata['x']
    # feature_num = feature.shape[1]
    # drop_feature_num = int(feature_num * drop_percent)
    # feature_var = th.var(feature, dim=0)
    # print(feature_var)
    # drop_mask = th.multinomial(feature_var, drop_feature_num)
    drop_mask = th.empty((feature.size(1),), dtype=th.float32, device=feature.device).uniform_(0, 1) < 1 - drop_percent
    feature = feature.clone()
    feature[:, drop_mask] = 0
    augmentation_g.ndata['x'] = feature
    return augmentation_g


def random_mask(g, center, drop_percent=0.2):
    augmentation_g = copy.deepcopy(g)
    feature = augmentation_g.ndata['x']
    node_num = feature.shape[0]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    node_idx.remove(center)
    mask_idx = list(random.sample(node_idx, mask_num))

    zeros = th.zeros_like(feature[0])
    feature[torch.LongTensor(mask_idx).cuda()] = zeros

    return augmentation_g


def drop_edge(g, center, drop_percent=0.2):
    augmentation_g = copy.deepcopy(g)
    percent = drop_percent / 2

    row_idx, col_idx = augmentation_g.edges()
    row_idx = row_idx.cpu().numpy()
    col_idx = col_idx.cpu().numpy()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)
    drop_edges_from = []
    drop_edges_to = []
    for i in drop_idx:
        drop_edges_from.append(single_index_list[i][0])
        drop_edges_from.append(single_index_list[i][1])
        drop_edges_to.append(single_index_list[i][1])
        drop_edges_to.append(single_index_list[i][0])

    drop_edges_id = augmentation_g.edge_ids(torch.LongTensor(drop_edges_from).cuda(), torch.LongTensor(drop_edges_to).cuda())
    augmentation_g.remove_edges(drop_edges_id)

    return augmentation_g


# def pr_drop_weights(g, center, p):
#     pv = compute_pr(edge_index, k=k)
#     pv_row = pv[edge_index[0]].to(th.float32)
#     pv_col = pv[edge_index[1]].to(th.float32)
#     s_row = th.log(pv_row)
#     s_col = th.log(pv_col)
#     if aggr == 'sink':
#         s = s_col
#     elif aggr == 'source':
#         s = s_row
#     elif aggr == 'mean':
#         s = (s_col + s_row) * 0.5
#     else:
#         s = s_col
#     weights = (s.max() - s) / (s.max() - s.mean())
#
#     return weights
#
#
# def evc_drop_weights(g, center, p):
#
#     evc = eigenvector_centrality(data)
#     evc = evc.where(evc > 0, th.zeros_like(evc))
#     evc = evc + 1e-8
#     s = evc.log()
#
#     edge_index = data.edge_index
#     s_row, s_col = s[edge_index[0]], s[edge_index[1]]
#     s = s_col
#
#     return (s.max() - s) / (s.max() - s.mean())

