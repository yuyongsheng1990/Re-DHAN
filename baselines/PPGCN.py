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

def extract_batch_edge_idx(batch_nodes, edge_index):
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


class PPGCN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(PPGCN, self).__init__()
        self.conv1 = GCNConv(in_channels=inp_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)
        self.norm = nn.BatchNorm1d(hid_dim)

    def forward(self, features_list, multi_r_data, batch_nodes, device):
        embed_list = []  # GCN的二维edge_index，注意adj_mx index需要在range [batch_siza, batch_size]，因为是convolution
        for i, (feature, edge_index) in enumerate(zip(features_list, multi_r_data)):
            batch_feature = feature[batch_nodes,:]
            batch_edge_idx = extract_batch_edge_idx(batch_nodes, edge_index)  # 抽取batch edge index
            feature_1 = self.conv1(batch_feature, batch_edge_idx)
            feature_1 = self.norm(feature_1)
            feature_2 = self.conv2(feature_1, batch_edge_idx)
            embed_list.append(feature_2)

        features = torch.stack(embed_list, dim=0)
        features = torch.reshape(features, (len(batch_nodes),-1))
        return features

'''
class PPGCN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(PPGCN, self).__init__()
        self.conv1 = GCNConv(in_channels=inp_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)
        self.layers = nn.ModuleList([self.conv1, self.conv2])
        self.intra_aggs = nn.ModuleList([self.layers for _ in range(3)])
        self.norm = nn.BatchNorm1d(hid_dim)

    def forward(self, feat_list, adjs, n_ids, device):
        embed_list = []
        for i in range(len(feat_list)):
            feat = feat_list[i][n_ids[i]]
            for j, (edge_index, _, size) in enumerate(adjs):
                feat, edge_index = feat.to(device), edge_index.to(device)
                feat = self.intra_aggs[i][j](feat, edge_index)
                if j == 0:
                    feat = self.norm(feat)
                    feat = F.elu(feat)
                    feat = F.dropout(feat, training=self.training)
                del edge_index
            embed_list.append(feat)
        features = torch.stack(embed_list, dim=0)
        features = torch.reshape(features, (.shape[0]))
        return features
'''