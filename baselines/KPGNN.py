# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> KPGNN
# @Time: 4/8/23 18:47 
# @Author: Yu Yongsheng
# @Description:

"""
    KPGNN equals to FinEvent without RL
# """
'''
-----------------------------------negative modification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout


class GAT(nn.Module):
    # adopt this module when using mini-batch

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(GAT, self).__init__()
        self.GAT1 = GATConv(in_channels=out_dim, out_channels=hid_dim, heads=heads,
                            add_self_loops=False)  # 这只是__init__函数声明变量
        self.GAT2 = GATConv(in_channels=hid_dim * heads, out_channels=out_dim, add_self_loops=False)  # 隐藏层维度，输出维度64
        self.layers = ModuleList([self.GAT1, self.GAT2])
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adjs, device):  # 这里的x是指batch node feature embedding， adjs是指RL_samplers 采样的batch node 子图 edge
        x = x.to(device)
        x = self.fc(x)
        for i, (edge_index, _, size) in enumerate(adjs):  # adjs list包含了从第L层到第1层采样的结果，adjs中子图是从大到小的。adjs(edge_index,e_id,size), edge_index是子图中的边
            # x: Tensor, edge_index: Tensor
            initial_feat = x
            edge_index = edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]  # (1418, 302); (100, 512) Target nodes are always placed first; size是子图shape, shape[1]即子图 node number; x[:size[1], : ]即取出前n个node
            x = self.layers[i]((x, x_target), edge_index)  # 这里调用的是forward函数, layers[1] output (1418, 512) out_dim * heads; layers[2] output (100, 64)
            del edge_index
        return x


# GAT model, intra means inside, which is exactly opposite of extra
class MultiHeadGATLayer(nn.Module):  # intra-aggregation
    def __init__(self, GAT_args):
        super(MultiHeadGATLayer, self).__init__()
        in_dim, hid_dim, out_dim, heads = GAT_args
        self.gnn = GAT(in_dim, hid_dim, out_dim, heads)  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4

    def forward(self, x, adjs, device):
        x = self.gnn(x, adjs, device)
        return x  # (100, 64)


# MarGNN model 返回node embedding representation
class KPGNN(nn.Module):
    def __init__(self, GNN_args, num_relations, inter_opt, is_shared=False):  # inter_opt='cat_w_avg'
        super(KPGNN, self).__init__()

        self.num_relations = num_relations  # 3
        self.inter_opt = inter_opt

        self.layers = torch.nn.ModuleList([MultiHeadGATLayer(GNN_args) for _ in range(self.num_relations)])


    def forward(self, x, batch_nodes, adjs, n_ids, device):  # adjs是RL_sampler采样的 batch_nodes 的子图edge; n_ids是采样过程中遇到的node list。都是list: 3, 对应entity, userid, word

        features = []
        for i in range(self.num_relations):  # i: 0, 1, 2
            features.append(self.layers[i](x[n_ids[i]], adjs[i], device))  # x表示batch feature embedding, MultiHeadGATLayers整合batch node neighbors -> (100, 64)

        features = torch.stack(features, dim=0)  # (3, 100, 64)
        features = features.reshape(len(batch_nodes), -1)

        return features   # (100,192)
'''



