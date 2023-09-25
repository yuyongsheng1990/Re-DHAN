# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description : CGAT1: Convolutional GAT. GAT message incorporation, gcn pass

import gc
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # PyG封装好的GATConv函数
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout, ELU
from models.MyGCN import MyGCN

# from layers.S1_GAT_Model import Intra_AGG
from models.Attn_Head import Attn_Head, Temporal_Attn_Head, SimpleAttnLayer
# from models.MLP_model import MLP_model

# # 构建图卷积层
# # 搭建graph convolution layer
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__()
#         self.relu = nn.ELU()
#         self.w = nn.Parameter(torch.rand(in_channels, out_channels), requires_grad=True)  # 初始化参数矩阵。
#         # rand(size)随机抽取[0,1)之间的数据组成size的tensor；nn.Parameter将不可训练tensor变成可训练tensor
#
#     # 定义forward函数
#     def forward(self, x, adj, device):
#         A = adj + torch.eye(adj.size(0)).to(device)
#         D_mx = torch.diag(torch.sum(A, 1))
#         D_hat = D_mx.inverse().sqrt()
#         A_hat = torch.mm(torch.mm(D_hat, A), D_hat)
#         mm_1 = torch.mm(A_hat, x)
#         vector = self.relu(torch.mm(mm_1, self.w))
#         return vector

# new GAT model：attention计算relative importance，convolution pass
class CGAT(nn.Module):
    '''
    adopt this module when using mini-batch
    '''

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(CGAT, self).__init__()
        self.GAT1 = GATConv(in_channels=in_dim, out_channels=in_dim, heads=heads,
                            add_self_loops=False)  # 这只是__init__函数声明变量
        self.GAT2 = GATConv(in_channels=hid_dim, out_channels=hid_dim, add_self_loops=False)  # 隐藏层维度，输出维度64
        self.GAT_layers = ModuleList([self.GAT1, self.GAT2])

        self.CONV_1 = nn.Conv1d(in_channels=heads * in_dim, out_channels=hid_dim, kernel_size=1)
        self.CONV_2 = nn.Conv1d(in_channels=hid_dim, out_channels=out_dim, kernel_size=1)
        self.CONV_layers = ModuleList([self.CONV_1, self.CONV_2])

        self.sig = nn.Sigmoid()
        self.norm = BatchNorm1d(hid_dim)  # 将num_features那一维进行归一化，防止梯度扩散

    def forward(self, x, adjs, device):  # 这里的x是指batch node feature embedding， adjs是指RL_samplers 采样的batch node 子图 edge
        for i, (edge_index, _, size) in enumerate(adjs):  # adjs list包含了从第L层到第1层采样的结果，adjs中子图是从大到小的。adjs(edge_index,e_id,size), edge_index是子图中的边
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]  # (1418, 302); (100, 512) Target nodes are always placed first; size是子图shape, shape[1]即子图 node number; x[:size[1], : ]即取出前n个node

            x = self.GAT_layers[i]((x, x_target), edge_index)  # 这里调用的是forward函数, layers[1] output (1418, 512) out_dim * heads; layers[2] output (100, 64)
            x = torch.unsqueeze(x, -1)
            x = self.CONV_layers[i](x)
            x = torch.squeeze(x)
            del edge_index
        return x
# HAN model with RL
class HeteGAT_multi_RL_CGAT(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                              activation=nn.ELU(), residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 num_relations, hid_units, n_heads, activation=nn.ELU()):
        super(HeteGAT_multi_RL_CGAT, self).__init__()
        self.feature_size = feature_size  # list:3, (4762, 300)
        self.nb_classes = nb_classes  # 3
        self.nb_nodes = nb_nodes  # 4762
        self.attn_drop = attn_drop  # 0.5
        self.feat_drop = feat_drop  # 0.0
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.time_lambda = time_lambda  # -0.2
        self.num_relations = num_relations  # list:3, (4762,4762)
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.activation = activation  # nn.ELU
        # self.residual = residual

        # 1-layer temporal attention model from HAN model
        self.layers_2 = self.temporal_attn_head(attn_input_dim=hid_dim, attn_out_dim=out_dim)

        # 1-layer Linear module
        self.linear = nn.Linear(out_dim, out_dim)
        self.linear_list = nn.ModuleList(self.linear for _ in range(self.num_relations))

        # 2层 GAT module - 2 from FinEvent
        self.intra_aggs = nn.ModuleList([CGAT(hid_dim, int(hid_dim / 2), hid_dim, 4) for _ in range(self.num_relations)])

        # nn.Conv1d
        self.Conv1d = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # 1-layer MLP
        self.mlp = nn.Sequential(
                    Linear(self.feature_size, int(self.feature_size/2)),
                    BatchNorm1d(int(self.feature_size/2)),
                    ELU(inplace=True),
                    Dropout(),
                    Linear(int(self.feature_size/2), hid_dim),)

        # normalization, 防止梯度扩散
        # self.norm = BatchNorm1d(hid_dim)

        self.simpleAttnLayer = SimpleAttnLayer(out_dim, hid_dim, time_major=False, return_alphas=True)  # 64, 128
        self.fc = nn.Linear(64, nb_classes)  # 64, 3


    # first layter attention. (100, 302) -> (100, 128)
    def normal_attn_head(self, attn_input_dim, attn_out_dim):
        layers = []
        for i in range(self.num_relations):  # 3
            attn_list = []
            for j in range(self.n_heads[0]):  # 8-head
                attn_list.append(Attn_Head(in_channel=int(attn_input_dim/self.n_heads[0]), out_sz=int(attn_out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=self.activation))
            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    # second layer attention. (100, 128) -> (100, 64)
    def temporal_attn_head(self, attn_input_dim, attn_out_dim):
        layers = []
        for i in range(self.num_relations):  # 3
            attn_list = []
            for j in range(self.n_heads[0]):  # 8-head
                attn_list.append(Temporal_Attn_Head(in_channel=int(attn_input_dim/self.n_heads[0]), out_sz=int(attn_out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=self.activation))
            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, features, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds):
        embed_list = []
        features = features.to(device)
        batch_nodes = batch_nodes.to(device)
        batch_embeddings = features[batch_nodes].to(device)
        # multi-head attention in a hierarchical manner
        for i, (biases) in enumerate(biases_mat_list):
            biases = biases.to(device)
            batch_features = features[n_ids[i]].to(device)
            attns = []
            '''
            (n_ids[i])  # (2596,); (137,); (198,)
            edge_index=tensor([[]]), e_id=None, size=(2596,1222); edge_index=tensor([[]]), e_id=None, size=(1222, 100)
            edge_index=tensor([[]]), e_id=None, size=(137,129); edge_index=tensor([[]]), e_id=None, size=(129, 100)
            edge_index=tensor([[]]), e_id=None, size=(198,152); edge_index=tensor([[]]), e_id=None, size=(152, 100)
            '''
            # -----------------1-layer MLP------------------------------------------------
            mlp_features = self.mlp(batch_features)

            # -----------------1-layer Linear------------------------------------------------
            # mlp_features = self.linear(features[n_ids[i]])

            # ---------------2 GAT layers from FinEvent-------------------------------------
            feature_embedding = self.intra_aggs[i](mlp_features, adjs[i], device)  # (100,128)
            # feature_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,128)
            batch_time = features[batch_nodes][:, -2:-1] * 10000  # (100, 1)  # 恢复成days representation
            # batch_time = batch_time.to(device)
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
            """
                # convolution operation
                feature_embedding = self.Conv1d(torch.transpose(h_2, 2, 1))
                feature_embedding = torch.squeeze(feature_embedding)
                feature_embedding = torch.transpose(feature_embedding, 1, 0)
            """

            # --------------1-layer Conv model---------------------------------------------
            # feature_embedding = self.MyGCN(feature_embedding, batch_bias)

            # ----------------1-layer Linear----------------------------------------------
            batch_feat_embeddings = self.linear_list[i](feature_embedding)
            # batch_feat_embeddings = self.linear_list[i](batch_embeddings)
            # ------------------residual connection
            # feature_embedding = torch.add(feature_embedding, batch_feat_embeddings)

            embed_list.append(torch.unsqueeze(feature_embedding, dim=1))
            del feature_embedding
            gc.collect()
        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed, device, RL_thresholds)  # (100, 64)
        del multi_embed
        gc.collect()
        # out = []
        # # 添加一个全连接层做预测(final_embedding, prediction) -> (100, 3)
        # out.append(self.fc(final_embed))

        return final_embed
