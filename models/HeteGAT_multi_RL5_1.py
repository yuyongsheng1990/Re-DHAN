# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description :
"""
    GATConv with 3-layer edge_index + residual connection: z0 + z1 -> h2
"""
import gc
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数
from torch_geometric.nn.inits import glorot, ones, zeros
from torch_scatter import scatter_add

from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout, ELU
from models.MyGCN import MyGCN

# from layers.S1_GAT_Model import Intra_AGG
from models.Attn_Head import Attn_Head, Temporal_Attn_Head, SimpleAttnLayer
from models.MLP_model import MLP_model

class GAT(nn.Module):
    '''
    adopt this module when using mini-batch
    '''

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(GAT, self).__init__()
        self.heads = heads
        self.hid_dim = hid_dim
        self.GAT1 = GATConv(in_channels=in_dim, out_channels=hid_dim, heads=heads, add_self_loops=False)  # 这只是__init__函数声明变量
        self.GAT2 = GATConv(in_channels=hid_dim * heads, out_channels=hid_dim, heads=heads, add_self_loops=False)  # 隐藏层维度，输出维度64
        self.GAT3 = GATConv(in_channels=hid_dim * heads, out_channels=out_dim, add_self_loops=False)  # 隐藏层维度，输出维度64
        self.layers = ModuleList([self.GAT1, self.GAT2, self.GAT3])
        self.norm = BatchNorm1d(heads * hid_dim)  # 将num_features那一维进行归一化，防止梯度扩散
        self.elu = nn.ELU()
        self.decay_bias = nn.ParameterList(nn.Parameter(torch.Tensor(1, 1)) for _ in range(3))
        self.k = 0
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # edge attention and hot attention with bias and decay weights
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []
        for i in range(3):
            self.hop_atts.append(nn.Parameter(torch.Tensor(1, self.hid_dim * self.heads * 2)))
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, 1)))
            self.decay_weights.append(torch.log(torch.tensor((1.0 / (i + 1)) + (1 + 1e-9))))
        self.hop_atts[0] = nn.Parameter(torch.Tensor(1, self.hid_dim * self.heads))
    def reset_parameters(self):
        # for line in self.mlp: line.reset_parameters()
        for atts in self.hop_atts: glorot(atts)
        for bias in self.hop_biases: ones(bias)
        for bias in self.decay_bias: ones(bias)

    def forward(self, x, adjs, device):  # 这里的x是指batch node feature embedding， adjs是指RL_samplers 采样的batch node 子图 edge
        z_scale = None
        z_sum = None
        for i, (edge_index, _, size) in enumerate(adjs):  # adjs list包含了从第L层到第1层采样的结果，adjs中子图是从大到小的。adjs(edge_index,e_id,size), edge_index是子图中的边
            # ---------------1-layer h0-------------------------------
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]  # (1418, 302); (100, 512) Target nodes are always placed first; size是子图shape, shape[1]即子图 node number; x[:size[1], : ]即取出前n个node
            x = self.layers[i]((x, x_target), edge_index)  # h0, (2819,256); h1, (1291,256)
            if i < len(adjs)-1:
                self.k = i
                if z_scale is not None:
                    z_scale = z_scale[:size[1]]
                g = self.hop_attns_pred(x, z_scale)  # hop attention weight, g0=(2819,1)
                z = x * g  # z0, (2819,256); z1=(1291,256)
                z_scale = z * self.decay_weights[i]  # z0_scale, (2819,256)
                if z_sum is not None:
                    z_sum = z_sum[:size[1]] + z
                else:
                    z_sum = z
            if i == len(adjs)-2:
                x = z_sum
            if i <= 1:
                x = self.norm(x)  # 归一化操作，防止梯度散射
                x = F.elu(x)  # 非线性激活函数elu
                x = F.dropout(x, training=self.training)
            del edge_index
        return x

    def hop_attns_pred(self, h, z_scale):
        if z_scale is None:
            x = h  # h0, (2819, 256)
        else:
            x = torch.cat((h, z_scale), dim=-1)  # h1, (1291,512)

        g = x.view(-1, int(x.shape[-1]))  # [1], g0, (2819,256); g1, (1291,512)
        g = self.elu(g)
        # hop_atts[0], 1x1x256; [1], 1x1x512; [2], 1x1x512; [3], 1x1x512
        g_atts = (self.hop_atts[self.k] * g).sum(-1)  # (2819,); (1291,)
        g_bias = self.hop_biases[self.k]  # (1,1)
        g = g_atts + g_bias  # (1, 2819); (1,1291)
        return g.transpose(1, 0)  # (2819, 1)

# HAN model with RL
class HeteGAT_multi_RL5_1(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 num_relations, hid_units, n_heads):
        super(HeteGAT_multi_RL5_1, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.nb_classes = nb_classes  # 3
        self.nb_nodes = nb_nodes  # 4762
        self.attn_drop = attn_drop  # 0.5
        self.feat_drop = feat_drop  # 0.0
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.time_lambda = time_lambda  # -0.2
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.hid_units = [4]
        self.n_heads = [2, 1, 4]  # [8,1]
        self.num_layers = 3
        self.elu = nn.ELU()

        # 1-layer MLP
        self.mlp = nn.Sequential(
                    Linear(self.feature_size, self.hid_dim, bias=True),
                    BatchNorm1d(self.hid_dim),
                    ELU(inplace=True),
                    Dropout(),
                    Linear(self.hid_dim, self.hid_dim,  bias=True),)

        # 2层 GAT module - 2 from FinEvent
        # self.GNN_args = hid_dim, int(hid_dim / 2), hid_dim, 2  # 300, hid_dim=128, out_dim=64, heads=4
        self.intra_aggs = nn.ModuleList([GAT(self.feature_size, int(self.hid_dim / 2), self.out_dim, self.n_heads[0])
                                         for _ in range(self.num_relations)])

        # ---------------------Final Linear module-------------------------------
        # self.final_linear = nn.Linear(out_dim, out_dim, bias=True, weight_initializer='glorot')
        self.final_linear = nn.Linear(self.out_dim, self.out_dim, bias=True)
        self.final_linear_list = [self.final_linear for _ in range(self.num_relations)]

        # -------------------meta-path aggregation----------------------------------
        self.simpleAttnLayer = SimpleAttnLayer(self.out_dim, self.hid_dim, time_major=False, return_alphas=True)  # 64, 128

    def forward(self, features, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds):
        embed_list = []
        features = features.to(device)
        batch_nodes = batch_nodes.to(device)
        batch_features = features[batch_nodes].to(device)
        batch_time = features[batch_nodes][:, -2:-1] * 10000  # (100, 1)  # 恢复成days representation
        batch_time = batch_time.to(device)
        # multi-head attention in a hierarchical manner
        for i, (biases) in enumerate(biases_mat_list):
            biases = biases.to(device)
            batch_bias = biases[batch_nodes][:, batch_nodes]  # (100, 100)
            attns = []
            '''
            (n_ids[i])  # (2596,); (137,); (198,)
            edge_index=tensor([[]]), e_id=None, size=(2596,1222); edge_index=tensor([[]]), e_id=None, size=(1222, 100)
            edge_index=tensor([[]]), e_id=None, size=(137,129); edge_index=tensor([[]]), e_id=None, size=(129, 100)
            edge_index=tensor([[]]), e_id=None, size=(198,152); edge_index=tensor([[]]), e_id=None, size=(152, 100)
            '''

            # ---------------2 GAT layers from FinEvent h1-------------------------------------
            # mlp_features = self.mlp(features[n_ids[i]])  # h0
            # feature_embedding = self.intra_aggs[i](mlp_features, adjs[i], device)  # (100,256)

            feature_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,256)

            # final_embedding = self.final_linear_list[i](feature_embedding)

            embed_list.append(torch.unsqueeze(feature_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed, device, RL_thresholds)  # (100, 64)
        del multi_embed
        gc.collect()
        # out = []
        # # 添加一个全连接层做预测(final_embedding, prediction) -> (100, 3)
        # out.append(self.fc(final_embed))

        return final_embed
