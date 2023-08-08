"""
@Filename       : geolocation_data.py
@Create Time    : 2022/5/19 11:41
@Author         :
@Description    : 

"""
import csv
import json
import os
import random
import sys
import pickle as pkl
import numpy as np
import networkx as nx
import torch as th
import scipy.sparse as sp
import scipy.io as sio
from networkx.readwrite import json_graph
from sklearn import preprocessing


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data_1(rootpath, dataset):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(rootpath, dataset, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(rootpath, dataset, "ind.{}.test.index".format(dataset)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    degree = np.sum(adj, axis=1)
    degree = th.FloatTensor(degree)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    labels = labels.argmax(axis=1)
    all_class = set(labels.tolist())

    id_by_class = {}
    for i in all_class:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla].append(id)

    class_list_valid = []
    class_list_test = list(random.sample(all_class, 2))
    class_list_train = list(all_class.difference(set(class_list_test)))

    adj = th.FloatTensor(adj.todense())
    features = th.FloatTensor(adj)
    labels = th.LongTensor(labels)

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def load_data_3(rootpath, dataset):
    names = ['x', 'y', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(rootpath, dataset, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, graph = tuple(objects)
    adj = graph
    degree = np.sum(adj, axis=1)
    degree = th.FloatTensor(degree)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    labels = np.array(y)

    # labels = labels.argmax(axis=1)

    all_class = set(labels.tolist())
    id_by_class = {}
    for i in all_class:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla].append(id)

    class_list_test = list(random.sample(all_class, 15))
    class_list_valid = list(random.sample(all_class.difference(set(class_list_test)), 15))
    class_list_train = list(all_class.difference(class_list_test).difference(class_list_valid))

    adj = th.FloatTensor(adj.todense())
    features = th.FloatTensor(adj)
    labels = th.LongTensor(labels)

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_data_2(rootpath, dataset):
    valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}

    n1s = []
    n2s = []
    for line in open(os.path.join(rootpath, dataset, "{}_network".format(dataset))):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s), max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_nodes, num_nodes))

    data_train = sio.loadmat(os.path.join(rootpath, dataset, "{}_train.mat".format(dataset)))
    train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

    data_test = sio.loadmat(os.path.join(rootpath, dataset, "{}_test.mat".format(dataset)))
    class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

    labels = np.zeros((num_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = th.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = th.FloatTensor(features)
    labels = th.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    class_list_valid = random.sample(train_class, valid_num_dic[dataset])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))
    # return adj, features, labels, class_list_train, class_list_valid, class_list_test, id_by_class
    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def load_reddit_data(rootpath, dataset, mode):
    prefix = os.path.join(rootpath, dataset, dataset)

    G_data = json.load(open(prefix + "-G.json"))
    id_map = json.load(open(prefix + "-id_map.json"))

    for node_item in G_data['nodes']:
        node_item['id'] = id_map[node_item['id']]
    G = json_graph.node_link_graph(G_data)
    print(G.number_of_nodes())
    if os.path.exists(prefix + "-feats.npy"):
        features = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        features = None

    class_map = json.load(open(prefix + "-class_map.json"))

    labels = th.LongTensor(len(class_map))
    for nid, id in id_map.items():
        labels[id] = class_map[nid]
    # Remove all nodes that do not have val/test annotations
    # (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    node_list = list(G.nodes)
    for node in node_list:
        if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    # Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")

    if mode == 'training':
        node_list = list(G.nodes)
        for node in node_list:
            if G.nodes[node]['val'] or G.nodes[node]['test']:
                G.remove_node(node)

    existing_node_ids = np.array(list(G.nodes))
    features = features[existing_node_ids]

    adj = nx.adjacency_matrix(G)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    degree = np.sum(adj, axis=1)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    degree = th.FloatTensor(degree)
    features = th.FloatTensor(features)

    id_by_class = {}
    for nid, clazz in class_map.items():
        if clazz in id_by_class.keys():
            id_by_class[clazz].append(id_map[nid])
        else:
            id_by_class[clazz] = [id_map[nid]]

    all_class = set(id_by_class.keys())

    class_list_test = list(random.sample(all_class, 15))
    class_list_valid = list(random.sample(all_class.difference(set(class_list_test)), 10))
    class_list_train = list(all_class.difference(class_list_test).difference(class_list_valid))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def task_generator(id_by_class, class_list, n_way, k_shot, m_query):
    # sample class indices
    if len(class_list) == n_way:
        class_selected = class_list
    else:
        class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


def load_data_4(rootpath, dataset):
    number_of_edges = int(open(os.path.join(rootpath, dataset, 'num-edge-list.csv')).readline().strip())
    number_of_nodes = int(open(os.path.join(rootpath, dataset, 'num-node-list.csv')).readline().strip())

    n1s = []
    n2s = []
    for line in open(os.path.join(rootpath, dataset, 'edge.csv')):
        n1, n2 = line.strip().split(',')
        n1s.append(int(n1))
        n2s.append(int(n2))

    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(number_of_nodes, number_of_nodes))
    features = th.FloatTensor(number_of_nodes, 100)
    for idx, line in enumerate(open(os.path.join(rootpath, dataset, 'node-feat.csv'))):
        features[idx] = th.FloatTensor(list(map(lambda x:float(x), line.strip().split(','))))

    labels = []
    for idx, line in enumerate(open(os.path.join(rootpath, dataset, 'node-label.csv'))):
        labels.append(int(line.strip()))

    class_list = []
    for cla in labels:
        if cla not in class_list:
            class_list.append(cla)  # unsorted

    degree = np.sum(adj, axis=1)
    degree = th.FloatTensor(degree)
    all_class = set(labels)
    id_by_class = {}
    for i in all_class:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla].append(id)

    class_list_test = list(random.sample(all_class, 20))
    class_list_valid = list(random.sample(all_class.difference(set(class_list_test)), 10))
    class_list_train = list(all_class.difference(class_list_test).difference(class_list_valid))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    labels = th.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def load_data_5(rootpath, dataset):
    file = os.path.join(rootpath, 'cora-full', 'cora.npz')
    loader = np.load(file)
    adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                       loader['adj_indptr']), shape=loader['adj_shape'])

    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                       loader['attr_indptr']), shape=loader['attr_shape'])
    features = features.todense()
    labels = loader.get('labels')

    degree = np.sum(adj, axis=1)
    degree = th.FloatTensor(degree)

    features = th.FloatTensor(features)

    all_class = set(labels)
    id_by_class = {}
    for i in all_class:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla].append(id)

    class_list_test = list(random.sample(all_class, 10))
    class_list_valid = list(random.sample(all_class.difference(set(class_list_test)), 10))
    class_list_train = list(all_class.difference(class_list_test).difference(class_list_valid))

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    labels = th.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # return adj, features, labels, class_list_train, class_list_valid, class_list_test, id_by_class
    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def load_data(rootpath, dataset):
    dataloader_method_dict = {
        'cora': load_data_1,
        'citeseer': load_data_1,
        'pubmed': load_data_1,
        'Amazon_clothing': load_data_2,
        'Amazon_eletronics': load_data_2,
        'dblp': load_data_2,
        'cmu': load_data_3,
        'twitter_us': load_data_3,
        'twitter_world': load_data_3,
        'ogbn-products': load_data_4,
        'cora-full': load_data_5
    }
    return dataloader_method_dict[dataset](rootpath, dataset)


if __name__ == '__main__':
    #load_reddit_data('../../data', 'reddit', mode='training')
    load_data('../../data', 'cora-full')
