# -*- coding: utf-8 -*-
# @Time : 2022/11/29 16:58
# @Author : yysgz
# @File : MarGNN.py
# @Project : utils Models
# @Description :

import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout
import time
from layers.S1_GAT_Model import Inter_AGG


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
                # x = F.dropout(x, training=self.training)
            del edge_index
        return x

# MarGNN model 返回node embedding representation
class KPGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads, num_relations):  # inter_opt='cat_w_avg'
        super(KPGNN, self).__init__()

        self.num_relations = num_relations  # 3

        self.intra_aggs = GAT(in_dim, hid_dim, out_dim, heads)
        self.inter_agg = Inter_AGG()
        # residual connection
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, batch_nodes, adjs, n_ids, device):  # adjs是RL_sampler采样的 batch_nodes 的子图edge; n_ids是采样过程中遇到的node list。都是list: 3, 对应entity, userid, word
        # RL_threshold: tensor([[.5], [.5], [.5]])

        # RL_thresholds = torch.FloatTensor([[1.], [1.], [1.]]).to(device)
        batch_feature = x[batch_nodes].to(device)
        residual_features = self.fc(batch_feature)

        features = []
        for i in range(self.num_relations):  # i: 0, 1, 2
            # print('Intra Aggregation of relation %d' % i)
            gat_features = self.intra_aggs(x[n_ids[i]], adjs[i], device)
            features.append(torch.add(gat_features, residual_features))  # x表示batch feature embedding, intra_aggs整合batch node neighbors -> (100, 64)

        features = torch.stack(features, dim=0)  # (3, 100, 64)
        features = torch.transpose(features, dim0=0, dim1=1)  # (100, 3, 64)
        # features = torch.mul(features, RL_thresholds).reshape(len(batch_nodes), -1)  # (100, 192)
        features = features.reshape(len(batch_nodes), -1)  # (100, 192)

        return features   # (100,192)