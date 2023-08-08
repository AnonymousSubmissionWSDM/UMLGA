"""
@Filename       : main_augmentation.py
@Create Time    : 2022/9/11 22:43
@Author         :
@Description    : 

"""

import argparse
import random
import sys
import time
from itertools import chain

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim

from models.meta.augmentation_gml.encoder import GCN, Valuator, SGC
from models.meta.augmentation_gml.subgraph_augmentation import SubGraph, SubGraphAndAugmentation
from utils.data import load_data, load_reddit_data
from utils.metric import euclidean_dist, accuracy, f1

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hop', type=int, default=2,
                    help='Number of hop of the sampled graph')
parser.add_argument('--encoder', type=str, default='gcn', help='way.')

parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=3, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=10)
parser.add_argument('--dataset', default='cora-full', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')
parser.add_argument('--augmentation_method', type=str, nargs='+', default=['subgraph'])
parser.add_argument('--augmentation_parameter', type=float, nargs='+', default=[0.2])

args = parser.parse_args()
args.cuda = args.use_cuda and th.cuda.is_available()

random.seed(args.seed)
th.manual_seed(args.seed)
if args.cuda:
    th.cuda.manual_seed(args.seed)

n_way = args.way
k_shot = args.shot
m_query = args.qry
dataset = args.dataset
hop = args.hop
augmentation_method_list = args.augmentation_method
augmentation_parameter_list = args.augmentation_parameter

if dataset == 'reddit':
    training_adj, training_features, labels, training_degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_reddit_data(
        '../../../../data', dataset, mode='training')
    adj, features, _, degrees, _, _, _, _ = load_reddit_data(
        '../../../../data', dataset, mode='valid')
else:
    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(
        '../../../../data', dataset)

print('Training Class Num: {}'.format(len(class_list_train)))
print('Validation Class Num: {}'.format(len(class_list_valid)))
print('Testing Class Num: {}'.format(len(class_list_test)))

encoder = SGC() if args.encoder == 'sgc' else GCN(nfeat=features.shape[1],
                                                  nhid=args.hidden,
                                                  dropout=args.dropout)

scorer = Valuator(nfeat=features.shape[1],
                  nhid=args.hidden,
                  dropout=args.dropout)

optimizer_encoder = optim.Adam(encoder.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    encoder = encoder.cuda()
    scorer = scorer.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()
    if dataset == 'reddit':
        training_adj = training_adj.cuda()
        training_features = training_features.cuda()


def train(g_support, center_nodes_support, g_query, center_nodes_query, n_way, k_shot):
    encoder.train()
    optimizer_encoder.zero_grad()

    # embedding lookup
    batch_support = dgl.batch(g_support)
    cursor = 0
    for idx, center_node_support in enumerate(center_nodes_support):
        center_nodes_support[idx] = center_node_support + cursor
        cursor += g_support[idx].number_of_nodes()

    batch_query = dgl.batch(g_query)
    cursor = 0
    for idx, center_node_query in enumerate(center_nodes_query):
        center_nodes_query[idx] = center_node_query + cursor
        cursor += g_query[idx].number_of_nodes()

    support_embeddings = encoder(batch_support, batch_support.ndata['x'])[th.LongTensor(center_nodes_support).cuda()]
    z_dim = support_embeddings.shape[-1]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = encoder(batch_query, batch_query.ndata['x'])[th.LongTensor(center_nodes_query).cuda()]

    scores = scorer(batch_support, batch_support.ndata['x'])[th.LongTensor(center_nodes_support).cuda()]
    support_scores = scores.view([n_way, k_shot])
    support_scores = th.sigmoid(support_scores).unsqueeze(-1)
    support_scores = support_scores / th.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)

    output = F.log_softmax(-dists, dim=1)
    pseudo_labels = th.LongTensor(list(chain.from_iterable([[i] * m_query for i in range(n_way)])))
    if args.cuda:
        pseudo_labels = pseudo_labels.cuda()
    loss_train = F.nll_loss(output, pseudo_labels)

    loss_train.backward()
    optimizer_encoder.step()

    if args.cuda:
        output = output.cpu().detach()
        pseudo_labels = pseudo_labels.cpu().detach()
    acc_train = accuracy(output, pseudo_labels)
    f1_train = f1(output, pseudo_labels)

    return acc_train, f1_train


def valid(g_support, center_nodes_support, g_query, center_nodes_query, n_way, k_shot):
    encoder.eval()
    # embedding lookup
    batch_support = dgl.batch(g_support)
    cursor = 0
    for idx, center_node_support in enumerate(center_nodes_support):
        center_nodes_support[idx] = center_node_support + cursor
        cursor += g_support[idx].number_of_nodes()

    batch_query = dgl.batch(g_query)
    cursor = 0
    for idx, center_node_query in enumerate(center_nodes_query):
        center_nodes_query[idx] = center_node_query + cursor
        cursor += g_query[idx].number_of_nodes()

    support_embeddings = encoder(batch_support, batch_support.ndata['x'])[th.LongTensor(center_nodes_support).cuda()]
    z_dim = support_embeddings.shape[-1]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = encoder(batch_query, batch_query.ndata['x'])[th.LongTensor(center_nodes_query).cuda()]
    scores = scorer(batch_support, batch_support.ndata['x'])[th.LongTensor(center_nodes_support).cuda()]
    support_scores = scores.view([n_way, k_shot])
    support_scores = th.sigmoid(support_scores).unsqueeze(-1)
    support_scores = support_scores / th.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)

    output = F.log_softmax(-dists, dim=1)

    pseudo_labels = th.LongTensor(list(chain.from_iterable([[i] * m_query for i in range(n_way)])))

    if args.cuda:
        output = output.cpu().detach()
        pseudo_labels = pseudo_labels.cpu().detach()
    acc_train = accuracy(output, pseudo_labels)
    f1_train = f1(output, pseudo_labels)

    return acc_train, f1_train


def test(g_support, center_nodes_support, g_query, center_nodes_query, labels, n_way, k_shot):
    encoder.eval()
    # embedding lookup
    batch_support = dgl.batch(g_support)
    cursor = 0
    for idx, center_node_support in enumerate(center_nodes_support):
        center_nodes_support[idx] = center_node_support + cursor
        cursor += g_support[idx].number_of_nodes()

    batch_query = dgl.batch(g_query)
    cursor = 0
    for idx, center_node_query in enumerate(center_nodes_query):
        center_nodes_query[idx] = center_node_query + cursor
        cursor += g_query[idx].number_of_nodes()

    support_embeddings = encoder(batch_support, batch_support.ndata['x'])[th.LongTensor(center_nodes_support).cuda()]
    z_dim = support_embeddings.shape[-1]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = encoder(batch_query, batch_query.ndata['x'])[th.LongTensor(center_nodes_query).cuda()]
    scores = scorer(batch_support, batch_support.ndata['x'])[th.LongTensor(center_nodes_support).cuda()]
    support_scores = scores.view([n_way, k_shot])
    support_scores = th.sigmoid(support_scores).unsqueeze(-1)
    support_scores = support_scores / th.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores
    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)

    output = F.log_softmax(-dists, dim=1)
    labels = th.LongTensor(list(chain.from_iterable([[i] * m_query for i in range(n_way)])))

    if args.cuda:
        output = output.cpu().detach()
        labels = labels.cpu().detach()
    acc_train = accuracy(output, labels)
    f1_train = f1(output, labels)

    return acc_train, f1_train


class Logger():
    def __init__(self, file_name, stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'a')

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)

    def close(self):
        self.log.flush()


if __name__ == '__main__':
    meta_test_num = 50
    meta_valid_num = 50
    print("Dataset:{}, N-way Number: {}, K-shot Number: {}, Query Number: {}".format(args.dataset, n_way, k_shot,
                                                                                     m_query))
    print("Augmentation Method: {}\nAugmentation Parameter: {}".format(augmentation_method_list,
                                                                       augmentation_parameter_list))
    logger = Logger(
        '{}-{}way{}shot{}{}.log'.format(dataset, n_way, k_shot, augmentation_method_list, augmentation_parameter_list))
    sys.stdout = logger
    # Sampling a pool of tasks for validation/testing
    # train_pool = SubGraphAndAugmentation(adj, features,  n_way, k_shot, m_query, args.task_num, args.hop)
    # valid_pool = SubGraphAndAugmentation(adj, features, n_way, k_shot, m_query, args.task_num, args.hop, pool_size=meta_valid_num)
    if type(adj) is th.sparse.Tensor:
        indices = adj._indices()
        G = dgl.graph(data=(indices[0], indices[1]))
    else:
        G = dgl.graph(adj)
    G.ndata['x'] = features
    if dataset == 'reddit':
        if type(adj) is th.sparse.Tensor:
            indices = training_adj._indices()
            training_G = dgl.graph(data=(indices[0], indices[1]))
        else:
            training_G = dgl.graph(adj)
        training_G.ndata['x'] = training_features
        train_pool = SubGraphAndAugmentation(training_G, training_adj, n_way, k_shot, m_query, hop,
                                             augmentation_method_list, augmentation_parameter_list)
    else:
        train_pool = SubGraphAndAugmentation(G, adj, n_way, k_shot, m_query, hop, augmentation_method_list,
                                             augmentation_parameter_list)
    valid_dataset = SubGraphAndAugmentation(G, adj, n_way, k_shot, m_query, hop, augmentation_method_list,
                                            augmentation_parameter_list)
    test_dataset = SubGraph(G, adj, class_list_test, id_by_class, n_way, k_shot, m_query, hop)
    t_total = time.time()
    meta_train_acc = []
    max_acc, max_f1 = -1, -1
    patient = 10
    for episode in range(args.episodes):
        g_support, center_nodes_support, g_query, center_nodes_query = train_pool[episode]
        acc_train, f1_train = train(g_support, center_nodes_support, g_query, center_nodes_query, n_way, k_shot)
        meta_train_acc.append(acc_train)
        if episode > 0 and episode % 10 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))
            # validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                g_support, center_nodes_support, g_query, center_nodes_query = valid_dataset[idx]
                acc_test, f1_test = valid(g_support, center_nodes_support, g_query, center_nodes_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                      np.array(meta_test_f1).mean(axis=0)))
            # testing
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_test_num):
                g_support, center_nodes_support, g_query, center_nodes_query = test_dataset[idx]
                acc_test, f1_test = test(g_support, center_nodes_support, g_query, center_nodes_query, labels, n_way,
                                         k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                    np.array(meta_test_f1).mean(axis=0)))
            if np.array(meta_test_acc).mean(axis=0) > max_acc:
                max_acc = np.array(meta_test_acc).mean(axis=0)
                max_f1 = np.array(meta_test_f1).mean(axis=0)
                print("The Best Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(max_acc, max_f1))
                patient = 10
            else:
                patient = patient - 1
                if patient < 0:
                    print("The Best Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(max_acc, max_f1))
                    logger.close()
                    exit()

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
