"""
@Filename       : metric.py
@Create Time    : 2022/5/21 11:40
@Author         :
@Description    : 

"""

import torch as th
from sklearn.metrics import f1_score


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return th.pow(x - y, 2).sum(2)  # N x M


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1