# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description :

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout
from models.MyGCN import MyGCN

from layers.S1_GAT_Model import Intra_AGG
from models.Attn_Head import Attn_Head, Temporal_Attn_Head, SimpleAttnLayer
from models.MLP_model import MLP_model

# HAN model with RL
class HeteGAT_multi_RL(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                              activation=nn.ELU(), residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 bias_mx_len, hid_units, n_heads, activation=nn.ELU()):
        super(HeteGAT_multi_RL, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.nb_classes = nb_classes  # 3
        self.nb_nodes = nb_nodes  # 4762
        self.attn_drop = attn_drop  # 0.5
        self.feat_drop = feat_drop  # 0.0
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.time_lambda = time_lambda  # -0.2
        self.bias_mx_len = bias_mx_len  # list:3, (4762,4762)
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.activation = activation  # nn.ELU
        # self.residual = residual

        # # 1层 hierarchical attention module - 1 from HAN model
        self.layers_1 = self.first_make_attn_head(attn_input_dim=hid_dim, attn_out_dim=self.out_dim)
        # 1-layer temporal attention model from HAN model
        self.layers_2 = self.second_make_attn_head(attn_input_dim=hid_dim, attn_out_dim=self.out_dim)
        # self.layers_2 = self.second_make_attn_head(attn_input_dim=self.feature_size, attn_out_dim=self.hid_dim)

        # 1-layer nn.conv1d
        self.Conv1d = nn.Conv1d(int(hid_dim/2), hid_dim, 1, bias=False)  # (64,64)

        # 1-layer Linear module
        self.linear = nn.Linear(out_dim, out_dim)
        # self.linear = nn.Linear(self.feature_size, self.feature_size)

        # 2层 GAT module - 2 from FinEvent
        # self.GNN_args = (self.feature_size, int(hid_dim / 2), hid_dim, 4)  # 300, hid_dim=128, out_dim=64, heads=4
        self.GNN_args = (hid_dim, int(hid_dim / 2), hid_dim, 4)  # 300, hid_dim=128, out_dim=64, heads=4
        # self.GNN_args = (hid_dim, int(hid_dim / 2), out_dim, 4)  # 300, hid_dim=128, out_dim=64, heads=4
        self.intra_aggs = nn.ModuleList([Intra_AGG(self.GNN_args) for _ in range(self.bias_mx_len)])

        # 1层GCNconv
        self.GCNConv = GCNConv(out_dim, out_dim)

        # 1层GCN model, <-自己定义的
        self.MyGCN = MyGCN(hid_dim, out_dim)

        # 1层GATConv
        self.GATConv = GATConv(out_dim, out_dim)

        # 1-layer MLP
        self.mlp = MLP_model(self.feature_size, int(self.feature_size/2), hid_dim)
        # self.mlp = MLP_model(hid_dim, int(hid_dim/2), hid_dim)
        # self.mlp = MLP_model(out_dim, int(out_dim/2), out_dim)

        # normalization, 防止梯度扩散
        # self.norm = BatchNorm1d(hid_dim)

        self.simpleAttnLayer = SimpleAttnLayer(out_dim, hid_dim, time_major=False, return_alphas=True)  # 64, 128
        self.fc = nn.Linear(64, nb_classes)  # 64, 3


    # first layter attention. (100, 302) -> (100, 128)
    def first_make_attn_head(self, attn_input_dim, attn_out_dim):
        layers = []
        for i in range(self.bias_mx_len):  # 3
            attn_list = []
            for j in range(self.n_heads[0]):  # 8-head
                attn_list.append(Attn_Head(in_channel=int(attn_input_dim/self.n_heads[0]), out_sz=int(attn_out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=self.activation))
            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    # second layer attention. (100, 128) -> (100, 64)
    def second_make_attn_head(self, attn_input_dim, attn_out_dim):
        layers = []
        for i in range(self.bias_mx_len):  # 3
            attn_list = []
            for j in range(self.n_heads[0]):  # 8-head
                attn_list.append(Temporal_Attn_Head(in_channel=int(attn_input_dim/self.n_heads[0]), out_sz=int(attn_out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=self.activation))
            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, features_list, biases_mat_list, batch_node_list, adjs, n_ids, device, RL_thresholds):
        embed_list = []

        # multi-head attention in a hierarchical manner
        for i, (features, biases) in enumerate(zip(features_list, biases_mat_list)):
            features, biases = features.to(device), biases.to(device)
            attns = []
            '''
            (n_ids[i])  # (2596,); (137,); (198,)
            edge_index=tensor([[]]), e_id=None, size=(2596,1222); edge_index=tensor([[]]), e_id=None, size=(1222, 100)
            edge_index=tensor([[]]), e_id=None, size=(137,129); edge_index=tensor([[]]), e_id=None, size=(129, 100)
            edge_index=tensor([[]]), e_id=None, size=(198,152); edge_index=tensor([[]]), e_id=None, size=(152, 100)
            '''
            batch_nodes = batch_node_list[i].to(device)
            # -----------------1-layer MLP------------------------------------------------
            mlp_features = self.mlp(features[n_ids[i]])

            # -----------------1-layer Linear------------------------------------------------
            # mlp_features = self.linear(features[n_ids[i]])

            # ---------------2 GAT layers from FinEvent-------------------------------------
            feature_embedding = self.intra_aggs[i](mlp_features, adjs[i], device)  # (100,128)
            # feature_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,128)
            batch_time = features[batch_nodes][:, -2:-1] * 10000  # (100, 1)  # 恢复成days representation
            batch_time = batch_time.to(device)
            batch_bias = biases[batch_nodes][:, batch_nodes]  # (100, 100)

            # -----------------normalization--------------------------------
            # feature_embedding = self.norm(feature_embedding)   -> negative

            # -----------------1-layer MLP------------------------------------------------
            # feature_embedding = self.mlp(feature_embedding)

            # 1-layer nn.conv1d -----------------------------------------------------------
            # feature_embedding = self.Conv1d(torch.unsqueeze(feature_embedding, dim=-1))
            # feature_embedding = torch.squeeze(feature_embedding)  # 压缩, (100, 128)

            # ----------------------- 1-layer Hierarchical Attention from HAN -----------------------
            # attn_embed_size = int(feature_embedding.shape[1] / self.n_heads[0])
            # for n in range(self.n_heads[0]):  # [8,1], 8个head
            #     # multi-head attention 计算, integrate meta-path based neighbors for specific node.
            #     attns.append(self.layers_1[i][n](feature_embedding[:, n*attn_embed_size: (n+1)*attn_embed_size], batch_bias, device))
            # h_1 = torch.cat(attns, dim=-1)  # shape=(1, 100, 128)
            # # h_1 = self.layers_1_pass(torch.permute(h_1, (1,2,0))) # Conv1d 传递 (100, 128,1) -> (100, 1, 128)
            # h_1_quz = torch.squeeze(h_1)  # 压缩, (100, 128)

            # -------------------1-layer temporal hierarchical attention module-----------------------------------
            # 2-nd layer. (100, 128) -> (100, 64)
            attn_embed_size = int(feature_embedding.shape[1] / self.n_heads[0])  # h_1_quz: 128, out_size: 64, heads: 8
            for n in range(self.n_heads[0]):
                attns.append(self.layers_2[i][n](feature_embedding[:, n * attn_embed_size: (n + 1) * attn_embed_size], batch_bias, device, self.time_lambda, batch_time))
            h_2 = torch.cat(attns, dim=-1)  # (1, 100, 64)
            feature_embedding = torch.squeeze(h_2)

            # -------------l-layer MLP---------------------
            # feature_embedding = self.mlp(feature_embedding)

            # 3-rd layer: nn.Linear
            # h_2_trans = self.linear(h_2)  # nn.Linear transformation, (1,100,64)
            # h_1_trans = torch.transpose(h_2_trans, 2, 1)
            # embed_list.append(torch.transpose(h_2_trans, 1, 0))  # list:2. 其中每个元素tensor, (100, 1, 64)

            # ---------------l-layer GCNConv --------------------------------------------
            # # edge_index 是2*n matrix, 需要做一步转换
            # all_edge_index = torch.tensor([], dtype=int)
            # for node in range(batch_bias.shape[0]):
            #     neighbor = batch_bias[node].squeeze()  # IntTensor是torch定义的7中cpu tensor类型之一；
            #     # squeeze对数据维度进行压缩，删除所有为1的维度
            #     # del self_loop in advance
            #     neighbor[node] = 0  # 对角线元素置0
            #     neighbor_idx = neighbor.nonzero()  # 返回非零元素的索引, size: (43, 1)
            #     neighbor_sum = neighbor_idx.size(0)  # 表示非零元素数据量,43
            #     loop = torch.tensor(node).repeat(neighbor_sum, 1)  # repeat表示按列重复node的次数
            #     edge_index_i_j = torch.cat((loop, neighbor_idx),
            #                                dim=1).t()  # cat表示按dim=1按列拼接；t表示对二维矩阵进行转置, node -> neighbor
            #     self_loop = torch.tensor([[node], [node]])
            #     all_edge_index = torch.cat((all_edge_index, edge_index_i_j, self_loop), dim=1)
            #     del neighbor, neighbor_idx, loop, self_loop, edge_index_i_j
            # feature_embedding = self.GCNConv(feature_embedding, all_edge_index)  # GCNConv 只支持torch.longTensor 或 torch_sparse.SparseTensor 的edge_idx
            # additional GCNConv无法用到全部的structure info，从batch里面提的edge_idx几乎没有！学不到什么东西，效果肯定不好！不同用GCN_model,因为GCN用的是邻接矩阵A，不是edge_index

            # --------------1-layer GCN model---------------------------------------------
            # feature_embedding = self.MyGCN(feature_embedding, batch_bias)

            # -----------------1-layer MLP------------------------------------------------
            # feature_embedding = self.mlp_f(feature_embedding)

            # ----------------1-layer Linear----------------------------------------------
            feature_embedding = self.linear(feature_embedding)
            # feature_embedding = self.linear(F.elu(feature_embedding))

            embed_list.append(torch.unsqueeze(feature_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed, RL_thresholds)  # (100, 64)
        # final_embed = torch.mul(multi_embed, RL_thresholds).reshape(len(batch_nodes), -1)

        # out = []
        # # 添加一个全连接层做预测(final_embedding, prediction) -> (100, 3)
        # out.append(self.fc(final_embed))

        return final_embed
