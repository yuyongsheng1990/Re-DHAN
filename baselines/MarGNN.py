# -*- coding: utf-8 -*-
# @Time : 2022/11/29 16:58
# @Author : yysgz
# @File : MarGNN.py
# @Project : utils Models
# @Description :

import torch
import torch.nn as nn
from torch.functional import Tensor
import time
from layers.S1_GAT_Model import Inter_AGG, Intra_AGG

# MarGNN model 返回node embedding representation
class MarGNN(nn.Module):
    def __init__(self, GNN_args, num_relations, inter_opt, is_shared=False):  # inter_opt='cat_w_avg'
        super(MarGNN, self).__init__()

        self.num_relations = num_relations  # 3
        self.inter_opt = inter_opt
        self.is_shared = is_shared
        if not self.is_shared:
            self.intra_aggs = torch.nn.ModuleList([Intra_AGG(GNN_args) for _ in range(self.num_relations)])
        else:
            self.intra_aggs = Intra_AGG(GNN_args)  # shared parameters

        if self.inter_opt == 'cat_w_avg_mlp' or 'cat_wo_avg_mlp':
            in_dim, hid_dim, out_dim, heads = GNN_args
            mlp_args = self.num_relations * out_dim, out_dim
            self.inter_agg = Inter_AGG(mlp_args)
        else:
            self.inter_agg = Inter_AGG()

    def forward(self, x, adjs, n_ids, device, RL_thresholds):  # adjs是RL_sampler采样的 batch_nodes 的子图edge; n_ids是采样过程中遇到的node list。都是list: 3, 对应entity, userid, word
        # RL_threshold: tensor([[.5], [.5], [.5]])
        if RL_thresholds is None:
            RL_thresholds = torch.FloatTensor([[1.], [1.], [1.]])
        if not isinstance(RL_thresholds, Tensor):
            RL_thresholds = torch.FloatTensor(RL_thresholds)
        RL_thresholds = RL_thresholds.to(device)

        features = []
        for i in range(self.num_relations):  # i: 0, 1, 2
            if not self.is_shared:
                # print('Intra Aggregation of relation %d' % i)
                features.append(self.intra_aggs[i](x[n_ids[i]], adjs[i], device))  # x表示batch feature embedding, intra_aggs整合batch node neighbors -> (100, 64)
            else:
                # shared parameters
                # print('Shared Intra Aggregation ...')
                features.append(self.intra_aggs(x[n_ids[i]], adjs[i], device))

        features = torch.stack(features, dim=0)  # (3, 100, 64)
        features = self.inter_agg(features, RL_thresholds, self.inter_opt)  # [[0.2], [0.2], [0.2]], 'cat_w_avg', (100, 192)

        return features   # (100,192)