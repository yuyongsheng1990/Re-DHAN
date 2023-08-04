# -*- coding: utf-8 -*-
# @Time : 2023/8/2 21:02
# @Author : yysgz
# @File : MLP_model.py
# @Project : run_offline model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout

class MLP_model(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(MLP_model, self).__init__()
        self.mlp = nn.Sequential(
                Linear(input_dim, hid_dim),
                BatchNorm1d(hid_dim),
                ReLU(inplace=True),
                Dropout(),
                Linear(hid_dim, out_dim),)

    def forward(self, batch_features):  # (100,302)

        vector = self.mlp(batch_features)
        return vector  # (100, 128)