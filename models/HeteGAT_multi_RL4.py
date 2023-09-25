# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description : graph transformer嘛 = attention + residual connection.
# edge index在每层上传递，并且一直residual connection against over-smooth. 这不就是妥妥的Transformer吗！！！
"""
    DHAN4-1: Graph Transformer with residual beta connection, mlp and linear
    DHAN4-2: Graph Transformer with only residual beta connection
    DHAN4-3: Graph Transformer with my desigend residual connection
"""

import gc
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, PointTransformerConv  # PyG封装好的GATConv函数
"""
    - HGTConv, Heterogeneous Graph Transformer, focus on Heterogeneous, but I've converted it to homogeneous graphs. 
    - PointTransformer, 不合适。它主要处理点云数据，focus on local attention mechanism，即[batch_adj, batch_adj]
"""
from torch_geometric.nn.inits import glorot, ones, zeros
from torch_scatter import scatter_add

from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout, ELU
from models.MyGCN import MyGCN

# from layers.S1_GAT_Model import Intra_AGG
from models.Attn_Head import Attn_Head, Temporal_Attn_Head, SimpleAttnLayer
from models.MLP_model import MLP_model

class GraphFormer(nn.Module):
    '''
    adopt this module when using mini-batch
    '''

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(GraphFormer, self).__init__()
        # DHAN4-1: residual beta=True
        self.GraphFormer1 = TransformerConv(in_channels=in_dim, out_channels=hid_dim, heads=heads, dropout=0.2, beta=True)  # 这只是__init__函数声明变量
        self.GraphFormer2 = TransformerConv(in_channels=hid_dim * heads, out_channels=hid_dim, dropout=0.2, beta=True)  # 隐藏层维度，输出维度64
        self.GraphFormer3 = TransformerConv(in_channels=hid_dim, out_channels=out_dim, dropout=0.2, beta=True)  # 隐藏层维度，输出维度64
        self.layers = ModuleList([self.GraphFormer1, self.GraphFormer2, self.GraphFormer3])
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

# HAN model with RL
class HeteGAT_multi_RL4(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 num_relations, hid_units, n_heads):
        super(HeteGAT_multi_RL4, self).__init__()
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
        self.k = 0
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout()
        self.norm = BatchNorm1d(self.hid_dim)


        # 1-layer MLP
        self.mlp = nn.Sequential(
                    Linear(self.feature_size, self.hid_dim, bias=True),
                    BatchNorm1d(self.hid_dim),
                    ELU(inplace=True),
                    Dropout(),
                    Linear(self.hid_dim, self.hid_dim,  bias=True),)

        # 2层 GAT module - 2 from FinEvent 300,
        # hid_dim=128, out_dim=64, heads=4
        self.intra_aggs = nn.ModuleList([GraphFormer(self.hid_dim, int(self.hid_dim / 2), self.hid_dim, self.n_heads[0])
                                         for _ in range(self.num_relations)])   # DHAN-4: original version
        # self.intra_aggs = nn.ModuleList([GraphFormer(self.feature_size, self.hid_dim, self.out_dim, self.n_heads[0])
        #                                  for _ in range(self.num_relations)])   # DHAN-4_2: GraphFormer with only residual beta connection

        # ---------------------Final Linear module-------------------------------
        # self.final_linear = nn.Linear(out_dim, out_dim, bias=True, weight_initializer='glorot')
        self.final_linear = nn.Linear(self.hid_dim, self.out_dim, bias=True)
        self.final_linear_list = [self.final_linear for _ in range(self.num_relations)]

        # -------------------meta-path aggregation----------------------------------
        self.simpleAttnLayer = SimpleAttnLayer(self.out_dim, self.hid_dim, time_major=False, return_alphas=True)  # 64, 128

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # edge attention and hot attention with bias and decay weights
        self.hop_biases = nn.ParameterList()
        for i in range(self.num_layers):
            # self.hop_atts.append(nn.Parameter())
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, 1)))
    def reset_parameters(self):
        # for line in self.mlp: line.reset_parameters()
        for bias in self.hop_biases: ones(bias)

    def forward(self, features, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds):
        embed_list = []
        features = features.to(device)
        # multi-head attention in a hierarchical manner
        for i, (bias) in enumerate(biases_mat_list):

            attns = []
            '''
            (n_ids[i])  # (2596,); (137,); (198,)
            edge_index=tensor([[]]), e_id=None, size=(2596,1222); edge_index=tensor([[]]), e_id=None, size=(1222, 100)
            edge_index=tensor([[]]), e_id=None, size=(137,129); edge_index=tensor([[]]), e_id=None, size=(129, 100)
            edge_index=tensor([[]]), e_id=None, size=(198,152); edge_index=tensor([[]]), e_id=None, size=(152, 100)
            '''

            # ---------------2 GAT layers from FinEvent h1-------------------------------------
            # mlp_features = self.mlp(features[n_ids[i]])  # h0
            # feature_embedding = self.intra_aggs[i](mlp_features, adjs[i], device)  # (100,256) for DHAN4-origin; DHAN4-1
            final_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,256) for DHAN4-2

            # ----------------1-layer Final Linear----------------------------------------------
            # final_embedding = self.final_linear_list[i](feature_embedding)

            embed_list.append(torch.unsqueeze(final_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed, device, RL_thresholds)  # (100, 64)
        del multi_embed
        gc.collect()

        return final_embed