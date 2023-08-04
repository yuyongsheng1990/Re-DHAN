# -*- coding: utf-8 -*-
# @Time : 2023/8/3 16:40
# @Author : yysgz
# @File : MyGCN.py
# @Project : 自己定义一个1-layer GCN model

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyGCN(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(MyGCN, self).__init__()
        self.w = nn.Parameter(torch.rand(inp_dim, out_dim), requires_grad=True)
        self.Linear = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        A = torch.where(adj>0.5, 1, 0)
        A = A + torch.eye(adj.size(0))
        D_mx = torch.diag(torch.sum(A, 1))
        D_hat = D_mx.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D_hat, A), D_hat)
        mm_1 = torch.mm(A_hat, x)
        vector = torch.relu(torch.mm(mm_1, self.w))
        return self.Linear(vector)
