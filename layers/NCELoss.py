# -*- coding: utf-8 -*-
# @Time : 2023/4/8 20:46
# @Author : yysgz
# @File : NCELoss.py
# @Project : Re-HAN_Model
# @Description :
import random

import torch
from torch import nn

class NCECriterion(nn.Module):

    def __init__(self, nce_m, eps):
        super(NCECriterion, self).__init__()
        self.nce_m = nce_m
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, labels):
        batch_size = x.size(0)
        # 噪声均匀分布
        noise_distribution = torch.tensor(1/batch_size).repeat(batch_size, 1).t().squeeze()
        # 计算non-parametric softmax classifier
        prob = torch.matmul(x, x.t())
        prob = torch.div(prob, self.eps)

        # softmax 计算概率
        pred_prob = self.softmax(prob)
        true_prob = self.softmax(labels.float())

        # 随机取两个向量v和v', 计算后验概率 h(i,v) = Pmt / (Pmt + m*Pnt)
        v_1_idx = random.randint(0, batch_size-1)
        v_2_idx = random.randint(0, batch_size-1)

        Pmt_1 = pred_prob[v_1_idx]
        Pnt_1 = Pmt_1.add(self.nce_m / batch_size)
        h_1 = torch.div(Pmt_1, Pnt_1)
        Pmt_2 = pred_prob[v_2_idx]
        Pnt_2 = Pmt_2.add(self.nce_m / batch_size)
        h_2 = torch.div(Pmt_2, Pnt_2)

        # 为了避免出现nan、inf，对后验概率做sigmoid激活，(0,1) -> 去除0影响。
        h_1_ac = self.sigmoid(h_1)
        h_2_ac = self.sigmoid(h_2)

        # 取对数
        h_1_log = torch.log(h_1_ac)
        h_2_log = torch.log(1 - h_2_ac)

        # calculate expectation
        Expection_1 = torch.matmul(true_prob, h_1_log)  # (100,) * (100,) = -4.9336
        Expection_2 = torch.matmul(noise_distribution, h_2_log)  # (1,100) * (100,) =
        # calculate NCE loss function
        nce_loss = -Expection_1 - self.nce_m * Expection_2

        return nce_loss, batch_size