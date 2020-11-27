import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.W.weight, nonlinearity='relu')

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.sparse.mm(adj, support)  # mat1 is sparse, while mat2 is dense; return is dense tensor
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
