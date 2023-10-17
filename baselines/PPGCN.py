# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> PPGCN
# @Time: 4/8/23 19:05 
# @Author: Yu Yongsheng
# @Description: PPGCN，就是2-layer GCN model, 判断pair是否属于同一类

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from torch_geometric.nn import GCNConv

'''
  - PPGCN-1: 利用torch geometric实现GCN on whole graph。2023.10.21，PyG~double GCN，效果贼好，但嵌入到我的ReDHAN model中，最终结果还是很差，为啥呢？
  - PPGCN-2: 自己构建convolution operation
'''
"""
# print("------------PPGNC-1----------------------------")
class PPGCN(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim):
        super(PPGCN, self).__init__()
        self.conv1 = GCNConv(in_channels=feat_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)

        # 归一化
        self.norm = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)
        # 激活函数，防止梯度爆炸
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, features, multi_r_data, batch_nodes, device):
        embed_list = []  # GCN的二维edge_index，注意adj_mx index需要在range [batch_siza, batch_size]，因为是convolution
        for i, (edge_index) in enumerate(multi_r_data):
            x, edge_index = features.to(device), edge_index.to(device)  # (4796, 302)
            x = self.conv1(x, edge_index)  # (4793, 256)
            x = self.norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)  # (4793, 128)
            x = self.norm2(x)
            x = self.relu(x)

            embed_list.append(F.log_softmax(x[batch_nodes], dim=1))  # (100, 128)

        features = torch.stack(embed_list, dim=0)  # (3,100,128)
        features = torch.transpose(features, dim0=0, dim1=1)  # (100, 3, 128)
        features = torch.reshape(features, (len(batch_nodes),-1))  # (100,384)
        return features
# """
# """
# print("------------PPGNC-2----------------------------")
# model = PPGCN(feat_dim, args.hid_dim, args.out_dim)
# pred = model(features_list, multi_r_data, batch_nodes, device)  # PPGCN-1 baseline model
'-------------------------graph convolution layer--------------'
import torch
from torch import nn
from torch.nn import functional as F


# 构建图卷积层
# 搭建graph convolution layer
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.relu = nn.ELU()
        self.w = nn.Parameter(torch.rand(in_channels, out_channels), requires_grad=True)  # 初始化参数矩阵。
        # rand(size)随机抽取[0,1)之间的数据组成size的tensor；nn.Parameter将不可训练tensor变成可训练tensor

    # 定义forward函数
    def forward(self, x, adj, device):
        A = adj + torch.eye(adj.size(0)).to(device)
        D_mx = torch.diag(torch.sum(A, 1))
        D_hat = D_mx.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D_hat, A), D_hat)
        mm_1 = torch.mm(A_hat, x)
        vector = self.relu(torch.mm(mm_1, self.w))
        return vector


class PPGCN(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout):
        super(PPGCN, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.conv1 = GCNConv(self.input_dim, self.hid_dim)
        self.conv2 = GCNConv(self.hid_dim, self.out_dim)

        # 归一化
        self.norm = nn.BatchNorm1d(input_dim)
        self.norm_2 = nn.BatchNorm1d(out_dim)
        # 激活函数，防止梯度爆炸
        self.sig = nn.Sigmoid()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def forward(self, features, adj_mat_list, batch_nodes, device):
        features = features.to(device)
        batch_nodes = batch_nodes.to(device)
        embed_list = []
        for i, (adj_mx) in enumerate(adj_mat_list):
            # batch_nodes = batch_node_list[i]
            adj_mx = adj_mx.to(device)
            batch_feature = features[batch_nodes]  # (100, 302)
            batch_adj = adj_mx[batch_nodes][:, batch_nodes]  # (100, 100)
            batch_feature = self.norm(batch_feature)
            h1 = self.conv1(batch_feature, batch_adj, device)
            h1 = self.sig(h1)
            h2 = self.conv2(h1, batch_adj, device)
            h2 = self.norm_2(h2)
            embed_list.append(h2)
        embed = torch.stack(embed_list, dim=0)
        embed_mean = torch.mean(torch.transpose(embed, 1, 0), dim=1)
        return self.relu(embed_mean)
# """