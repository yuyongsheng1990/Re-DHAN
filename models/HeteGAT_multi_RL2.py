# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description : optimize Re-DHAN with temporal multi-head attention for mlp, bias, decay weight, ELU according to AERO-GNN

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

            del edge_index
        return x
# HAN model with RL
class HeteGAT_multi_RL2(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                             residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 num_relations, hid_units, n_heads):
        super(HeteGAT_multi_RL2, self).__init__()
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

        # 1-layer MLP
        self.mlp = nn.Sequential(
                    Linear(self.feature_size, self.hid_dim, bias=True),
                    BatchNorm1d(self.hid_dim),
                    ELU(inplace=True),
                    Dropout(),
                    Linear(self.hid_dim, self.hid_dim,  bias=True),)

        # 2层 GAT module - 2 from FinEvent
        # self.GNN_args = hid_dim, int(hid_dim / 2), hid_dim, 2  # 300, hid_dim=128, out_dim=64, heads=4
        self.intra_aggs = nn.ModuleList([GAT(self.hid_dim, int(self.hid_dim / 2), self.hid_dim, self.n_heads[0])
                                         for _ in range(self.num_relations)])

        # 1-layer temporal attention model from HAN model
        # temporal_attention-1 with multi-head time attention and batch nodes
        self.time_atts = self.temporal_attn_head(self.hid_dim, self.out_dim)

        # ---------------------Final Linear module-------------------------------
        # self.final_linear = nn.Linear(out_dim, out_dim, bias=True, weight_initializer='glorot')
        self.final_linear = nn.Linear(self.out_dim, self.out_dim, bias=True)
        self.final_linear_list = [self.final_linear for _ in range(self.num_relations)]

        # -------------------meta-path aggregation----------------------------------
        self.simpleAttnLayer = SimpleAttnLayer(self.out_dim, self.hid_dim, return_alphas=True)  # 64, 128

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # edge attention and hot attention with bias and decay weights
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []
        for i in range(self.num_layers):
            # self.hop_atts.append(nn.Parameter())
            self.hop_atts.append(nn.Parameter(torch.Tensor(1, self.hid_dim * self.n_heads[0])))
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, 1)))
            self.decay_weights.append(torch.log(torch.tensor((1.0/(i+1)) + (1+1e-9))))
        self.hop_atts[0] = nn.Parameter(torch.Tensor(1, self.hid_dim))
    def reset_parameters(self):
        # for line in self.mlp: line.reset_parameters()
        for atts in self.time_atts: glorot(atts)
        for atts in self.hop_atts: glorot(atts)
        for bias in self.hop_biases: ones(bias)

    # second layer attention. (100, 128) -> (100, 64)
    def temporal_attn_head(self, attn_input_dim, attn_out_dim):
        layers = []
        for i in range(self.num_relations):  # 3
            attn_list = []
            for j in range(self.n_heads[-1]):  # 8-head
                attn_list.append(Temporal_Attn_Head(in_channel=int(attn_input_dim/self.n_heads[-1]), out_sz=int(attn_out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop))
            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    # def time_edge_attns_pred(self, z_scale, batch_adj_mx, batch_time):  # h, (100,256); z_scale, (100,256); batch_edge_idx, (2,2430)
    #     # edge attention
    #     a_ij = z_scale + z_scale  # (100, 256)
    #     a_ij = self.elu(a_ij)
    #     # atts=10, (0): Parameter 1x1x64; (1): Parameter 1x1x64; ...(9): Parameter 1x1x64
    #     a_ij = (self.time_atts1 * a_ij).sum(dim=-1)  # (100,)
    #     a_ij = self.softplus(a_ij) + 1e-9  # (100,)
    #
    #     # symmetric normalization (alpha_ij)
    #     batch_num = batch_adj_mx.shape[0]
    #     # row, col = batch_edge_index[0], batch_edge_index[1]  # row, (165,); col, (165,)
    #     deg = batch_adj_mx.sum(dim=-1)  # degree matrix
    #     # deg = scatter_add(a_ij, col, dim=0, dim_size=batch_num)  # degree matrix
    #     deg_inv_sqart = deg.pow(-0.5)
    #     deg_inv_sqart.masked_fill_(deg_inv_sqart == float('inf'), 0)
    #     # a_ij = deg_inv_sqart[row] * deg_inv_sqart[col]
    #     # calculate time weight
    #     time_weight_mx = self.time_decay_weight(batch_time, self.time_lambda)  # (100, 100)
    #     a_ij = time_weight_mx * a_ij.unsqueeze(dim=-1)
    #
    #     return a_ij

    def hop_attns_pred(self, h, z_scale):
        if z_scale is None:
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, int(x.shape[-1]))  # [1], (100,512)
        g = self.elu(g)
        # hop_atts[0], 1x1x256; [1], 1x1x512; [2], 1x1x512; [3], 1x1x512
        g_atts = (self.hop_atts[self.k] * g).sum(-1)  # (100,)
        g_bias = self.hop_biases[self.k]  # (1,1)
        g = g_atts + g_bias  # (1, 100)
        return g.transpose(1, 0)  # (100, 1)

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
            self.k = 0
            # -----------------1-layer MLP h0------------------------------------------------
            batch_mlp_features = self.mlp(batch_features)  # (100, 256)
            # -----------------hop attention z0------------------------------------------
            g0 = self.hop_attns_pred(batch_mlp_features, z_scale=None)  # z0 from hop attention, (100, 1)
            z0 = batch_mlp_features * g0  # (100,256)
            z0_scale = z0 * self.decay_weights[self.k]  # (100,256) for later edge attention and hop attention

            # ---------------2 GAT layers from FinEvent h1-------------------------------------
            self.k += 1
            mlp_features = self.mlp(features[n_ids[i]])  # h0
            feature_embedding = self.intra_aggs[i](mlp_features, adjs[i], device)  # (100,256)
            # --------------hop attention z1---------------------------------------------------
            g1 = self.hop_attns_pred(feature_embedding, z_scale=z0_scale)  # (100, 1)
            z1 = feature_embedding * g1  # (100, 256)
            z1 = z0 + z1  # (100, 256)
            z1_scale = z1 * self.decay_weights[self.k]  # (100, 256)

            # -------------------1-layer temporal hierarchical attention module h2-----------------------------------
            # -----------------h2 with multi-head attention------------------------------------
            attn_embed_size = int(z1.shape[1] / self.n_heads[-1])  # h_1_quz: 128, out_size: 64, heads: 8
            for n in range(self.n_heads[-1]):
                attns.append(self.time_atts[i][n](feature_embedding[:, n * attn_embed_size: (n + 1) * attn_embed_size], batch_bias, device, self.time_lambda, batch_time))
            h_2 = torch.cat(attns, dim=-1)  # (1, 100, 64)
            h2 = torch.squeeze(h_2)

            # # --------------hop attention z2--------------------------------------------------
            # g2 = self.hop_attns_pred(h2, z_scale=z1_scale)
            # z2 = h2 * g2
            # z2 = z1 + z2

            # # ----------------1-layer Final Linear----------------------------------------------
            # h2 = self.elu(h2)
            # h2 = self.dropout(h2)
            final_embedding = self.final_linear_list[i](h2)

            embed_list.append(torch.unsqueeze(final_embedding, dim=1))

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        final_embed, att_val = self.simpleAttnLayer(multi_embed, device)  # (100, 64)
        del multi_embed
        gc.collect()
        # out = []
        # # 添加一个全连接层做预测(final_embedding, prediction) -> (100, 3)
        # out.append(self.fc(final_embed))

        return final_embed
