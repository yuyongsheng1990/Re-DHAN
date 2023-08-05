# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> PPGCN
# @Time: 4/8/23 19:05 
# @Author: Yu Yongsheng
# @Description: PPGCN，就是2-layer GCN model, 判断pair是否属于同一类

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

"""  --> negative, 效果很差
# print("------------PPGNC-1----------------------------")

class PPGCN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(PPGCN, self).__init__()
        self.conv1 = GCNConv(in_channels=inp_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)

        # 归一化
        self.norm = nn.BatchNorm1d(inp_dim)
        self.norm_2 = nn.BatchNorm1d(out_dim)
        # 激活函数，防止梯度爆炸
        self.sig = nn.Sigmoid()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def forward(self, features_list, multi_r_data, batch_nodes, device):
        embed_list = []  # GCN的二维edge_index，注意adj_mx index需要在range [batch_siza, batch_size]，因为是convolution
        for i, (feature, edge_index) in enumerate(zip(features_list, multi_r_data)):
            batch_feature = feature[batch_nodes,:]
            batch_edge_idx = self.extract_batch_edge_idx(batch_nodes, edge_index)  # 抽取batch edge index
            batch_feature = self.norm(batch_feature)
            feature_1 = self.conv1(batch_feature, batch_edge_idx)
            feature_1 = self.sig(feature_1)
            feature_2 = self.conv2(feature_1, batch_edge_idx)
            feature_2 = self.norm_2(feature_2)
            embed_list.append(feature_2)

        features = torch.stack(embed_list, dim=0)
        features = torch.reshape(features, (len(batch_nodes),-1))
        return features

    def extract_batch_edge_idx(self, batch_nodes, edge_index):
        extract_edge_index = torch.Tensor()
        for i in batch_nodes:
            extract_edge_i = torch.Tensor()
            # extract 1-st row index and 2-nd row index
            edge_index_bool_0 = edge_index[0, :]
            edge_index_bool_0 = (edge_index_bool_0 == i)
            if edge_index_bool_0 is None:
                continue
            bool_indices_0 = np.where(edge_index_bool_0)[0]
            # extract data
            edge_index_0 = edge_index[0:, bool_indices_0]
            for j in batch_nodes:
                edge_index_bool_1 = edge_index_0[1, :]
                edge_index_bool_1 = (edge_index_bool_1 == j)
                if edge_index_bool_1 is None:
                    continue
                bool_indices_1 = np.where(edge_index_bool_1)[0]
                edge_index_1 = edge_index_0[0:, bool_indices_1]
                extract_edge_i = torch.cat((extract_edge_i, edge_index_1), dim=1)
            extract_edge_index = torch.cat((extract_edge_index, extract_edge_i), dim=1)
        # reset index value in a specific range
        uni_set = torch.unique(extract_edge_index)
        to_set = torch.tensor(list(range(len(uni_set))))
        labels_reset = extract_edge_index.clone().detach()
        for from_val, to_val in zip(uni_set, to_set):
            labels_reset = torch.where(labels_reset == from_val, to_val, labels_reset)
        return labels_reset.type(torch.long)
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
    def forward(self, x, adj):
        A = adj + torch.eye(adj.size(0))
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

    def forward(self, features_list, adj_mat_list, batch_node_list, device):
        embed_list = []
        for i, (feat, adj_mx) in enumerate(zip(features_list, adj_mat_list)):
            batch_nodes = batch_node_list[i]
            batch_feature = feat[batch_nodes]  # (100, 302)
            batch_adj = adj_mx[batch_nodes][:, batch_nodes]  # (100, 100)
            batch_feature = self.norm(batch_feature)
            h1 = self.conv1(batch_feature, batch_adj)
            h1 = self.sig(h1)
            h2 = self.conv2(h1, batch_adj)
            h2 = self.norm_2(h2)
            embed_list.append(h2)
        embed = torch.stack(embed_list, dim=0)
        embed_mean = torch.mean(torch.transpose(embed, 1, 0), dim=1)
        return self.relu(embed_mean)
# """
