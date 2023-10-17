# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description :
"""
2023.10.20
    PyG double GCN + GAT -》实现Deep GNN，因为residual connection是那些狗币伪造的数据，根本没有任何真实性事实性的依据，是虚假的科研结论。
    而真正的counter over-smooth的本质是whole graph convolution operation无损地传递structure info，即edge index
    所以，真正应该做的是 GCN + GAT -》deep GNN，而不是那些狗币伪造数据误导的科研方向residual connection。
2023.10.21
    PyG~double GCN，效果贼好，但嵌入到我的ReDHAN model中，最终结果还是很差，为啥呢？
"""

import gc
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数

from models.Attn_Head import Attn_Head, Temporal_Attn_Head, SimpleAttnLayer

class DoubleGCN(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(DoubleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels=feat_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)

        # 归一化
        self.norm = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)
        # 激活函数，防止梯度爆炸
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, features, edge_index, batch_nodes, device):
        x, edge_index = features.to(device), edge_index.to(device)  # (4796, 302)
        x = self.conv1(x, edge_index)  # (4793, 256)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)  # (4793, 128)
        x = self.norm2(x)
        x = self.relu(x)

        return F.log_softmax(x[batch_nodes], dim=1)
        # features = torch.stack(embed_list, dim=0)  # (3,100,128)
        # features = torch.transpose(features, dim0=0, dim1=1)  # (100, 3, 128)
        # features = torch.reshape(features, (len(batch_nodes),-1))  # (100,384)
        # return features

# HAN model with RL
class HeteGAT_multi_RL6(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)

    '''
    def __init__(self, feature_size, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 num_relations):
        super(HeteGAT_multi_RL6, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
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
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout()

        self.gcn_layers = nn.ModuleList([DoubleGCN(self.feature_size, self.hid_dim, self.out_dim)
                                         for _ in range(self.num_relations)])  # DHAN-4: original version

        # temporal_attention-1 with multi-head time attention and batch nodes
        self.time_atts = self.temporal_attn_head(self.hid_dim, self.out_dim)

        # -------------------meta-path aggregation----------------------------------
        self.simpleAttnLayer = SimpleAttnLayer(self.out_dim, self.hid_dim,  return_alphas=True)  # 64, 128


    def temporal_attn_head(self, attn_input_dim, attn_out_dim):
        layers = []
        for i in range(self.num_relations):  # 3
            attn_list = []
            for j in range(self.n_heads[-1]):  # 8-head
                attn_list.append(Temporal_Attn_Head(in_channel=int(attn_input_dim/self.n_heads[-1]), out_sz=int(attn_out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop))
            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, features, multi_r_data, batch_nodes, device):
        embed_list = []
        features = features.to(device)
        batch_time = features[batch_nodes][:, -2:-1] * 10000  # (100, 1)  # 恢复成days representation
        batch_time = batch_time.to(device)
        batch_bias = torch.full((batch_nodes.shape[0], batch_nodes.shape[0]), torch.log(torch.tensor((1.0/3) + (1+1e-9))))
        # multi-head attention in a hierarchical manner
        for i, (edge_index) in enumerate(multi_r_data):

            # ---------------2 GCN layers from FinEvent h1-------------------------------------
            gcn_embedding = self.gcn_layers[i](features, edge_index, batch_nodes, device)  # (100,256)

            # # ----------------1-layer Final Linear----------------------------------------------
            # attns = []
            # attn_embed_size = int(gcn_embedding.shape[1] / self.n_heads[-1])  # h_1_quz: 128, out_size: 64, heads: 8
            # for n in range(self.n_heads[-1]):
            #     attns.append(self.time_atts[i][n](gcn_embedding[:, n * attn_embed_size: (n + 1) * attn_embed_size], batch_bias,
            #                                       device, self.time_lambda, batch_time))
            # h_2 = torch.cat(attns, dim=-1)  # (1, 100, 64)
            # h2 = torch.squeeze(h_2)

            embed_list.append(torch.unsqueeze(gcn_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed, device)  # (100, 64)
        del multi_embed
        gc.collect()

        return final_embed