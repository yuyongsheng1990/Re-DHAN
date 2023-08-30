# -*- coding: utf-8 -*-
# @Time : 2022/11/2 15:01
# @Author : yysgz
# @File : S1_GAT_Model.py
# @Project : data_checking.py
# @Description:

from torch.functional import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout

class GAT(nn.Module):
    '''
    adopt this module when using mini-batch
    '''

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(GAT, self).__init__()
        self.GAT1 = GATConv(in_channels=in_dim, out_channels=hid_dim, heads=heads,
                            add_self_loops=False)  # 这只是__init__函数声明变量
        self.GAT2 = GATConv(in_channels=hid_dim * heads, out_channels=out_dim, add_self_loops=False)  # 隐藏层维度，输出维度64
        self.layers = ModuleList([self.GAT1, self.GAT2])
        self.norm = BatchNorm1d(heads * hid_dim)  # 将num_features那一维进行归一化，防止梯度扩散

    def forward(self, x, adjs, device):  # 这里的x是指batch node feature embedding， adjs是指RL_samplers 采样的batch node 子图 edge
        for i, (edge_index, _, size) in enumerate(adjs):  # adjs list包含了从第L层到第1层采样的结果，adjs中子图是从大到小的。adjs(edge_index,e_id,size), edge_index是子图中的边
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]  # (1418, 302); (100, 512) Target nodes are always placed first; size是子图shape, shape[1]即子图 node number; x[:size[1], : ]即取出前n个node
            x = self.layers[i]((x, x_target), edge_index)  # 这里调用的是forward函数, layers[1] output (1418, 512) out_dim * heads; layers[2] output (100, 64)
            if i == 0:
                x = self.norm(x)  # 归一化操作，防止梯度散射
                x = F.elu(x)  # 非线性激活函数elu
                x = F.dropout(x, training=self.training)
            del edge_index
        return x


# GAT model, intra means inside, which is exactly opposite of extra
class Intra_AGG(nn.Module):  # intra-aggregation
    def __init__(self, GAT_args):
        super(Intra_AGG, self).__init__()
        in_dim, hid_dim, out_dim, heads = GAT_args
        self.gnn = GAT(in_dim, hid_dim, out_dim, heads)  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4

    def forward(self, x, adjs, device):
        x = self.gnn(x, adjs, device)
        return x  # (100, 64)


# mlp model, inter means middle
class Inter_AGG(nn.Module):  # inter-aggregation
    def __init__(self, mlp_args=None):
        super(Inter_AGG, self).__init__()
        if mlp_args is not None:
            hid_dim, out_dim = mlp_args
            self.mlp = nn.Sequential(
                Linear(hid_dim, hid_dim),
                BatchNorm1d(hid_dim),
                ReLU(inplace=True),
                Dropout(),
                Linear(hid_dim, out_dim),
            )

    def forward(self, features, thresholds, inter_opt):  # (3,100,64), [[0.2], [0.2], [0.2]], 'cat_w_avg'
        batch_size = features[0].size(0)  # 100
        features = torch.transpose(features, dim0=0, dim1=1)  # (100, 3, 64)
        if inter_opt == 'cat_wo_avg':
            features = features.reshape(batch_size, -1)
        elif inter_opt == 'cat_w_avg':
            # weighted average and concatenate
            features = torch.mul(features, thresholds).reshape(batch_size, -1)  # (100, 192)
        elif inter_opt == 'cat_w_avg_mlp':
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
            features = self.mlp(features)
        elif inter_opt == 'cat_wo_avg_mlp':
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
            features = self.mlp(features)
        elif inter_opt == 'add_wo_avg':
            features = features.sum(dim=1)
        elif inter_opt == 'add_w_avg':
            features = torch.mul(features, thresholds).sum(dim=1)
        return features
