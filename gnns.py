import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv


class GAT(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False, None))
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, middle=False):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            middle_feats.append(h)
            h = self.activation(h)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        if middle:
            return logits, middle_feats
        return logits




class GCN1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN1, self).__init__()
        self.layer = GraphConvolution(in_dim, out_dim)
        self.fc1_a = nn.Linear(out_dim, 32)
        self.fc2_a = nn.Linear(32, 2)
        self.mlp_layer = nn.Linear(721, out_dim)


    def forward(self, inputs, adj):
        logits = self.layer(inputs, adj)
        h_back_a = self.fc1_a(logits)
        h_back_a = h_back_a.relu()
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)

        return pred_org_back_a

class GCN2(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GCN2, self).__init__()
        self.layer1 = GraphConvolution(in_dim, hidden_dim)
        self.layer2 = GraphConvolution(hidden_dim, hidden_dim)
        self.fc1_a = nn.Linear(hidden_dim, 32)
        self.fc2_a = nn.Linear(32, 2)

    def forward(self, inputs, adj):
        h = self.layer1(inputs, adj)
        h = F.relu(h)
        logits = self.layer2(h, adj)
        h_back_a = self.fc1_a(logits)
        h_back_a = h_back_a.relu()
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        return pred_org_back_a

class GCN3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN3, self).__init__()
        self.layer = GraphConvolution(in_dim, out_dim)
        self.fc1_a = nn.Linear(out_dim, 32)
        self.fc2_a = nn.Linear(32, 2)
        self.mlp_layer = nn.Linear(721, out_dim)


    def forward(self, inputs, adj):
        logits = self.layer(inputs, adj)
        # h_0 = self.mlp_layer(feat2).relu()
        # logits = logits + h_0
        h_back_a = self.fc1_a(logits)
        h_back_a = h_back_a.relu()
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)

        return pred_org_back_a

class GCN4(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GCN4, self).__init__()
        self.layer1 = GraphConvolution(in_dim, hidden_dim)
        self.layer2 = GraphConvolution(hidden_dim, hidden_dim)
        self.fc1_a = nn.Linear(hidden_dim, 32)
        self.fc2_a = nn.Linear(32, 2)

    def forward(self, inputs, adj):
        h = self.layer1(inputs, adj)
        h = F.relu(h)
        logits = self.layer2(h, adj)
        h_back_a = self.fc1_a(logits)
        h_back_a = h_back_a.relu()
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)
        return pred_org_back_a


class GCN5(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN5, self).__init__()
        self.layer = GraphConvolution(in_dim, out_dim)
        self.fc1_a = nn.Linear(out_dim, 32)
        self.fc2_a = nn.Linear(32, 2)
        self.mlp_layer = nn.Linear(721, out_dim)

    def forward(self, inputs, adj):
        logits = self.layer(inputs, adj)

        h_back_a = self.fc1_a(logits)
        h_back_a = h_back_a.relu()
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)

        return pred_org_back_a


class GCN6(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN6, self).__init__()
        self.layer = GraphConvolution(in_dim, out_dim)
        self.fc1_a = nn.Linear(out_dim, 32)
        self.fc2_a = nn.Linear(32, 2)
        self.mlp_layer = nn.Linear(721, out_dim)

    def forward(self, inputs, adj):
        logits = self.layer(inputs, adj)

        h_back_a = self.fc1_a(logits)
        h_back_a = h_back_a.relu()
        pred_org_back_a = F.log_softmax(self.fc2_a(h_back_a), dim=1)

        return pred_org_back_a



class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # 矩阵相乘 用户购买的商品 * 商品的属性特征
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
