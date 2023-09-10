# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description : CGAT2: MLP + GAT

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

# new GAT model：attention计算relative importance，convolution pass
class CGAT(nn.Module):
    '''
    adopt this module when using mini-batch
    '''

    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:  # in_dim, 302; hid_dim, 128; out_dim, 64, heads, 4
        super(CGAT, self).__init__()
        self.GAT1 = GATConv(in_channels=in_dim, out_channels=hid_dim, heads=heads,
                            add_self_loops=False)  # 这只是__init__函数声明变量
        self.GAT2 = GATConv(in_channels=hid_dim*heads, out_channels=out_dim, add_self_loops=False)  # 隐藏层维度，输出维度64
        self.GAT_layers = ModuleList([self.GAT1, self.GAT2])

        self.norm = BatchNorm1d(hid_dim*heads)  # 将num_features那一维进行归一化，防止梯度扩散

    def forward(self, x, adjs, device):  # 这里的x是指batch node feature embedding， adjs是指RL_samplers 采样的batch node 子图 edge
        for i, (edge_index, _, size) in enumerate(adjs):  # adjs list包含了从第L层到第1层采样的结果，adjs中子图是从大到小的。adjs(edge_index,e_id,size), edge_index是子图中的边
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)  # x: (2703, 302); (2, 53005); -> x: (1418, 512); (2, 2329)
            x_target = x[:size[1]]  # (1418, 302); (100, 512) Target nodes are always placed first; size是子图shape, shape[1]即子图 node number; x[:size[1], : ]即取出前n个node

            x = self.GAT_layers[i]((x, x_target), edge_index)  # 这里调用的是forward函数, layers[1] output (1418, 512) out_dim * heads; layers[2] output (100, 64)
            if i == 0:
                x = self.norm(x)  # 归一化操作，防止梯度散射
                x = F.elu(x)  # 非线性激活函数elu
                x = F.dropout(x, training=self.training)
            del edge_index
        return x
# HAN model with RL
class HeteGAT_multi_RL_CGAT2(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                              activation=nn.ELU(), residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim, time_lambda,
                 num_relations, hid_units, n_heads, activation=nn.ELU()):
        super(HeteGAT_multi_RL_CGAT2, self).__init__()
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
        self.intra_aggs = nn.ModuleList([CGAT(self.feature_size, int(self.hid_dim/2), self.hid_dim, 4) for _ in range(self.num_relations)])

        # nn.Conv1d
        self.Conv1d = nn.Conv1d(hid_dim, out_dim, kernel_size=1)

        # 1-layer MLP for CGAT-2.1
        self.mlp = nn.Sequential(
                    Linear(self.feature_size, self.hid_dim),
                    BatchNorm1d(int(self.hid_dim)),
                    ELU(inplace=True),
                    Dropout(),
                    Linear(self.hid_dim, self.out_dim),)


        self.simpleAttnLayer = SimpleAttnLayer(out_dim, hid_dim, time_major=False, return_alphas=True)  # 64, 128
        self.fc = nn.Linear(64, nb_classes)  # 64, 3

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
        batch_features = features[batch_nodes].to(device)
        # multi-head attention in a hierarchical manner
        for i, (biases) in enumerate(biases_mat_list):
            biases = biases.to(device)
            attns = []
            '''
            (n_ids[i])  # (2596,); (137,); (198,)
            edge_index=tensor([[]]), e_id=None, size=(2596,1222); edge_index=tensor([[]]), e_id=None, size=(1222, 100)
            edge_index=tensor([[]]), e_id=None, size=(137,129); edge_index=tensor([[]]), e_id=None, size=(129, 100)
            edge_index=tensor([[]]), e_id=None, size=(198,152); edge_index=tensor([[]]), e_id=None, size=(152, 100)
            '''
            # -----------------1-layer MLP------------------------------------------------
            mlp_features = self.mlp(batch_features)

            # ---------------2 GAT layers from FinEvent-------------------------------------
            feature_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,128)
            # feature_embedding = self.intra_aggs[i](features[n_ids[i]], adjs[i], device)  # (100,128)
            batch_time = features[batch_nodes][:, -2:-1] * 10000  # (100, 1)  # 恢复成days representation
            # batch_time = batch_time.to(device)
            batch_bias = biases[batch_nodes][:, batch_nodes]  # (100, 100)


            # -------------------1-layer temporal hierarchical attention module-----------------------------------
            # 2-nd layer. (100, 128) -> (100, 64)
            attn_embed_size = int(feature_embedding.shape[1] / self.n_heads[0])  # h_1_quz: 128, out_size: 64, heads: 8
            for n in range(self.n_heads[0]):
                attns.append(self.layers_2[i][n](feature_embedding[:, n * attn_embed_size: (n + 1) * attn_embed_size], batch_bias, device, self.time_lambda, batch_time))
            h_2 = torch.cat(attns, dim=-1)  # (1, 100, 64)
            feature_embedding = torch.squeeze(h_2)

            # ------------------residual connection-------------------------------
            feature_embedding = torch.add(mlp_features, feature_embedding) / 2
            # ----------------1-layer Linear----------------------------------------------
            feature_embedding = self.linear_list[i](feature_embedding)


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
