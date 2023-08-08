"""
@Filename       : encoder.py
@Create Time    : 2022/11/10 0:45
@Author         :
@Description    :

"""



import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, 2 * nhid, allow_zero_in_degree=True)
        self.gc2 = GraphConv(2 * nhid, nhid, allow_zero_in_degree=True)
        self.dropout = dropout

    def forward(self, g, feat):
        x = F.relu(self.gc1(g, feat))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)

        return F.log_softmax(x, dim=1)


class Valuator(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.
    """

    def __init__(self, nfeat, nhid, dropout):
        super(Valuator, self).__init__()

        self.gc1 = GraphConv(nfeat, 2 * nhid, allow_zero_in_degree=True)
        self.gc2 = GraphConv(2 * nhid, nhid, allow_zero_in_degree=True)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, g, feat):
        x = F.relu(self.gc1(g, feat))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(g, x))
        x = self.fc3(x)

        return x


class SGC(nn.Module):
    def __init__(self):
        super(SGC, self).__init__()

    def forward(self, features, adj, degree=2):
        for i in range(degree):
            features = th.spmm(adj, features)
        return features
