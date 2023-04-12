# -*- coding: utf-8 -*-
# @Time : 2023/3/21 20:38
# @Author : yysgz
# @File : HeteGAT_multi.py
# @Project : HAN_torch
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Attn_Head import Attn_Head, Self_Attn_Head, SimpleAttnLayer

class HeteGAT_multi(nn.Module):
    '''
    inputs_list=feas_list, nb_classes=nb_classes, nb_nodes=nb_nodes, attn_drop=0.5,
                              ffd_drop=0.0, biases_list=biases_list, hid_units=args.hid_units, n_heads=args.n_heads,
                              activation=nn.ELU(), residual=args.residual)

    '''
    def __init__(self, feature_size, nb_classes, nb_nodes, attn_drop, feat_drop, hid_dim, out_dim,
                 bias_mx_len, hid_units, n_heads, activation=nn.ELU()):
        super(HeteGAT_multi, self).__init__()
        self.feature_size = feature_size  # list:3, (3025, 1864)
        self.nb_classes = nb_classes  # 3
        self.nb_nodes = nb_nodes  # 3025
        self.attn_drop = attn_drop  # 0.5
        self.feat_drop = feat_drop  # 0.0
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.bias_mx_len = bias_mx_len  # list:2, (3025,3025)
        self.hid_units = hid_units  # [8]
        self.n_heads = n_heads  # [8,1]
        self.activation = activation  # nn.ELU
        # self.residual = residual

        self.layers = self._make_attn_head()
        self.w_multi = nn.Linear(out_dim, out_dim)
        # self.w_multi = nn.Conv1d(out_dim, out_dim, 1, bias=False)  # (64,64)

        self.simpleAttnLayer = SimpleAttnLayer(out_dim, hid_dim, time_major=False, return_alphas=True)  # 64, 128
        self.fc = nn.Linear(out_dim, nb_classes)  # 64, 3

    def _make_attn_head(self):
        layers = []
        for i in range(self.bias_mx_len):  # (3025,1864); (3025,3025)
            attn_list = []
            for j in range(self.n_heads[0]):  # 8-head
                attn_list.append(Attn_Head(in_channel=int(self.feature_size/self.n_heads[0]), out_sz=int(self.out_dim/self.hid_units[0]),  # in_channel,233; out_sz,8
                                feat_drop=self.feat_drop, attn_drop=self.attn_drop, activation=self.activation))

            layers.append(nn.Sequential(*list(m for m in attn_list)))
        return nn.Sequential(*list(m for m in layers))

    def forward(self, features_list, biases_mat_list, batch_node_list, device, RL_thresholds):  # (3025, 1864); (100,)
        embed_list = []

        # multi-head attention in a hierarchical manner
        for i, (features, biases) in enumerate(zip(features_list, biases_mat_list)):
            attns = []

            batch_nodes = batch_node_list[i]
            batch_feature = features[batch_nodes]  # (100, 1864)
            batch_bias = biases[batch_nodes][:,batch_nodes]  # (100, 100)
            attn_embed_size = int(batch_feature.shape[1] / self.n_heads[0])
            jhy_embeds = []
            for n in range(self.n_heads[0]):  # [8,1], 8个head
                # multi-head attention 计算
                attns.append(self.layers[i][n](batch_feature[:, n*attn_embed_size: (n+1)*attn_embed_size], batch_bias, device))

            h_1 = torch.cat(attns, dim=-1)  # shape=(1, 100, 64)
            h_1_trans = self.w_multi(h_1)  # with nn.Linear transformation
            # h_1_trans = self.w_multi(torch.transpose(h_1, 2, 1))  # with nn.Conv1d transformation
            # h_1_trans = torch.transpose(h_1_trans, 2, 1)
            embed_list.append(torch.transpose(h_1_trans, 1, 0))  # list:2. 其中每个元素tensor, (100, 1, 64)

        multi_embed = torch.cat(embed_list, dim=1)   # tensor, (100, 3, 64)
        # simple attention 合并多个meta-based homo-graph embedding
        # final_embed, att_val = self.simpleAttnLayer(multi_embed)  # (100, 64)
        final_embed = torch.mul(multi_embed, RL_thresholds).reshape(len(batch_nodes), -1)
        # out = []
        # # 添加一个全连接层做预测(final_embedding, prediction) -> (100, 3)
        # out.append(self.fc(final_embed))

        return final_embed
