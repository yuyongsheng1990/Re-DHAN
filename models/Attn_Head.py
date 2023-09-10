# -*- coding: utf-8 -*-
# @Time : 2023/3/23 14:45
# @Author : yysgz
# @File : Self_Attn_Head.py
# @Project : HAN_torch
# @Description :
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attn_Head(nn.Module):
    def __init__(self, in_channel, out_sz, feat_drop=0.0, attn_drop=0.0, activation=None, return_attn=False):
        super(Attn_Head, self).__init__()
        # self.bias_mat = bias_mat  # (3025,3025)
        self.feat_drop = feat_drop  # 0.0
        self.attn_drop = attn_drop  # 0.5
        self.return_attn = return_attn
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)  # (233,8)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.activation = activation

    def forward(self, x, bias_mx, device):  # (100, 37); bias_mx, 残差, (100, 100), batch_time (100, 2)
        seq = x.float().to(device)  # (100, 37)
        if self.feat_drop != 0.0:
            seq = self.feat_dropout(x)  # 以rate置0
            seq = seq.float()
        # reshape x and bias_mx for nn.Conv1d
        seq = torch.unsqueeze(seq, dim=0)
        seq = torch.transpose(seq, 2, 1)  # (1, 37, 100)
        bias_mx = torch.unsqueeze(bias_mx, dim=0).to(device)  # (1, 100, 100)
        seq_fts = self.conv1(seq)  # x*Wv=v, 一维卷积操作, out: (1, 16, 100)

        f_1 = self.conv2_1(seq_fts)  # x*Wq=q,(1, 1, 100)
        f_2 = self.conv2_2(seq_fts)  # x*Wk=k, (1, 1, 100)

        logits = f_1 + torch.transpose(f_2, 2, 1)  # 转置 (1, 100, 100)
        logits = self.leakyrelu(logits)

        attns = self.softmax(logits + bias_mx.float())  # add残差, (1, 100, 100)

        if self.attn_drop != 0.0:
            attns = self.attn_dropout(attns)
        if self.feat_drop != 0.0:
            seq_fts = self.feat_dropout(seq_fts)
        ret = torch.matmul(attns, torch.transpose(seq_fts, 2, 1))  # (1, 100, 16)

        if self.return_attn:
            return self.activation(ret), attns
        else:
            return self.activation(ret)  # activation

# # numpy version: time exponential decay formula
# def time_decay_weight(vectors, time_lambda):  # 衰减参数 lambda
#     # torch 转换为numpy array
#     vectors = torch.squeeze(vectors.cpu().t()).numpy()  # (1,100)
#     # 计算每两个元素相减的绝对值并形成矩阵
#     diff_matrix = np.abs(np.subtract.outer(vectors, vectors))
#     time_weight_mx = np.exp(diff_matrix * (-time_lambda))  # 不要用softmax，它会将差异抹平！！！
#     # time_weight_mx = F.softmax(torch.from_numpy(time_matrix), dim=1)  # dim=1, 在行上进行softmax; dim=0, 在列上进行softmax
#     return torch.from_numpy(time_weight_mx)

# tensor version: time exponential decay formula
def time_decay_weight(vectors, time_lambda, device):  # 衰减参数 lambda
    # torch 转换为 array
    vectors = vectors.to(device)
    time_lambda = torch.tensor(time_lambda).to(device)
    vectors = torch.squeeze(vectors.t())  # (1,100) -> (100,)
    # 计算每两个元素相减的绝对值并形成矩阵
    diff_matrix = torch.abs(vectors.unsqueeze(0) - vectors.unsqueeze(1))
    time_weight_mx = torch.exp(diff_matrix * (-time_lambda))  # 不要用softmax，它会将差异抹平！！！
    # time_weight_mx = F.softmax(torch.from_numpy(time_matrix), dim=1)  # dim=1, 在行上进行softmax; dim=0, 在列上进行softmax
    return time_weight_mx

# Temporal node attention
class Temporal_Attn_Head(nn.Module):
    def __init__(self, in_channel, out_sz, feat_drop=0.0, attn_drop=0.0, activation=None, return_attn=False):
        super(Temporal_Attn_Head, self).__init__()
        # self.bias_mat = bias_mat  # (3025,3025)
        self.feat_drop = feat_drop  # 0.0
        self.attn_drop = attn_drop  # 0.5
        self.return_attn = return_attn
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)  # (233,8)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.activation = activation

    def forward(self, x, bias_mx, device, time_lambda, batch_time):  # (100, 16); bias_mx, 残差, (100, 100), batch_time (100, 1)
        seq = x.float().to(device)
        if self.feat_drop != 0.0:
            seq = self.feat_dropout(x)  # 以rate置0
            seq = seq.float()
        # reshape x and bias_mx for nn.Conv1d,
        seq = torch.unsqueeze(seq, dim=0)
        seq = torch.transpose(seq, 2, 1)  # (1, 16, 100)
        bias_mx = torch.unsqueeze(bias_mx, dim=0).to(device)  # (1, 100, 100)
        seq_fts = self.conv1(seq)  # x*Wv=v, 一维卷积操作, out: (1, 8, 100)

        f_1 = self.conv2_1(seq_fts)  # x*Wq=q,(1, 1, 100)
        f_2 = self.conv2_2(seq_fts)  # x*Wk=k, (1, 1, 100)

        logits = f_1 + torch.transpose(f_2, 2, 1)  # 转置 (1, 100, 100)
        logits = self.leakyrelu(logits)

        attns = self.softmax(logits + bias_mx.float())  # add残差, (1, 100, 100)

        if self.attn_drop != 0.0:
            attns = self.attn_dropout(attns)
        if self.feat_drop != 0.0:
            seq_fts = self.feat_dropout(seq_fts)

        # add time_decay_weight
        # if batch_time is not None:
        time_weight_mx = time_decay_weight(batch_time, time_lambda, device)  # (100, 100)
        # 时间衰减权重应该是对应元素相乘，而不是矩阵相乘
        time_weight_mx = torch.unsqueeze(time_weight_mx, 0)  # (1, 100, 100)
        attns_tt = torch.mul(attns, time_weight_mx)  # temporal attention weight, (1, 100, 100)

        ret = torch.matmul(attns_tt, torch.transpose(seq_fts, 2, 1))  # (1, 100, 8)

        if self.return_attn:
            return self.activation(ret), attns
        else:
            return self.activation(ret)  # activation

class Self_Attn_Head(nn.Module):
    def __init__(self, in_channel, out_sz, feat_drop=0.0, attn_drop=0.0, activation=None, return_attn=False):
        super(Self_Attn_Head, self).__init__()
        # self.bias_mat = bias_mat  # (3025,3025)
        self.out_sz = out_sz
        self.feat_drop = feat_drop  # 0.0
        self.attn_drop = attn_drop  # 0.5
        self.return_attn = return_attn
        self.conv1 = nn.Conv1d(in_channel, out_sz, 1, bias=False)  # (233,8)
        self.conv2_1 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.conv2_2 = nn.Conv1d(out_sz, 1, 1, bias=False)  # (8,1)
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.activation = activation

    def forward(self, x, bias_mx, device):  # (100, 233); bias_mx, 残差, (100, 100)
        seq = x.float().to(device)
        if self.feat_drop != 0.0:
            seq = self.feat_dropout(x)  # 以rate置0
            seq = seq.float()
        # reshape x and bias_mx for nn.Conv1d, (1, 233, 100)
        seq = torch.transpose(seq[np.newaxis], 2, 1)  # (1, 233, 100)
        bias_mx = bias_mx[np.newaxis].to(device)
        seq_v = self.conv1(seq)  # x*Wv=v, 一维卷积操作, out: (1, 8, 100)

        seq_q = self.conv2_1(seq)  # x*Wq=q,(1, 1, 100)
        seq_k = self.conv2_2(seq)  # x*Wk=k, (1, 1, 100)

        logits = torch.matmul(torch.transpose(seq_q, 2, 1), seq_k)  # 转置 (1, 100, 100)
        logits = torch.div(logits, math.sqrt(self.out_sz))  # scaled, 放缩除以跟下dk

        attns = self.softmax(logits + bias_mx.float())  # add残差, (1, 100, 100)

        if self.attn_drop != 0.0:
            attns = self.attn_dropout(attns)
        if self.feat_drop != 0.0:
            seq_v = self.feat_dropout(seq_v)

        ret = torch.matmul(attns, torch.transpose(seq_v, 2, 1))  # (1, 100, 8)

        if self.return_attn:
            return self.activation(ret), attns
        else:
            return self.activation(ret)  # activation

class SimpleAttnLayer(nn.Module):
    def __init__(self, inputs, attn_size, time_major=False, return_alphas=False):  # inputs, 64; attention_size,128; return_alphas=True
        super(SimpleAttnLayer, self).__init__()
        self.hidden_size = inputs  # 64
        self.return_alphas = return_alphas  # True
        self.time_major = time_major
        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, attn_size))  # (64, 128)
        self.b_omega = nn.Parameter(torch.Tensor(attn_size))  # (128,)
        self.u_omega = nn.Parameter(torch.Tensor(attn_size, 1))  # (128,)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.zeros_(self.b_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, x, device, RL_thresholds):  # (100,2,64); 将RL sampleing weight 加入到meta-path importance weight
        '''
        这是一个点积dot-product attention
        inputs: tensor, (3025, 64)
        attention_size: 128
        '''
        batch_size = x.shape[0]
        if isinstance(x, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = torch.concat(x, 2)  # 表示在shape第2个维度上拼接
        x = x.to(device)  # v
        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)  # (100,2,128) 作为attention q
        vu = torch.matmul(v, self.u_omega)  # (100,2,1) qk相乘得一维相似度向量
        alphas = self.softmax(vu)
        # 在meta-path aggregation weight上加入meta-path, 需要保持shape一致
        tensor_rl = RL_thresholds.to(device)  # (3, 1)
        alphas = torch.add(alphas, tensor_rl) /2 # mean, (100, 3, 1)

        output = torch.sum(x * alphas.reshape(alphas.shape[0],-1,1), dim=1)  # (100,2,64)*(100,1,2) -> (100,64)
        # # output = torch.mul(x, alphas.reshape(alphas.shape[0],-1,1)).reshape(batch_size, -1)  # (100,2,64)*(100,1,2) -> (100,64)
        # # output = torch.mean(x * alphas.reshape(alphas.shape[0],-1,1), dim=1)  # intra mean

        if not self.return_alphas:
            return output
        else:
            return output, alphas  # attention输出、softmax概率