import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphCL.aug import normalize_adj
class Discriminator2(nn.Module):
    def __init__(self, n_h):
        super(Discriminator2, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)  # input_ft1, input_ft264, out_ft; 64, 1, 64
        self.elu_1 = nn.ELU()
        self.elu_2 = nn.ELU()
        self.norm = nn.BatchNorm1d(n_h)
        self.sft_1 = nn.Softmax()
        self.sft_2 = nn.Softmax()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # aug_1 embedding, (90,64); original embedding(100,64); shuffle original fts,(100,64)
    def forward(self, h_aug, h_pos, h_neg, device, s_bias1=None, s_bias2=None):
        # reference AdaGCL: https://github.com/YL-wang/CIKM_AdaGCL/blob/master/amazon/ns_com.py

        h_aug, h_pos_2, h_neg = h_aug.to(device), h_pos.to(device), h_neg.to(device)
        T = 0.5  # this temperature hyper-parameter can refer to Dual-channel graph contrastive learning for self-supervised graph-level representation learning.
        batch_size, _ = h_aug.size()
        x_aug_abs = h_aug.norm(dim=1)
        x_pos_abs = h_pos.norm(dim=1)
        x_neg_abs = h_neg.norm(dim=1)

        sim_pos_matrix = torch.einsum('ik,jk->ij', h_aug, h_pos_2) / torch.einsum('i,j->ij', x_aug_abs, x_pos_abs)
        sim_pos_matrix = torch.exp(sim_pos_matrix / T)  # Euclidean distance similarity
        pos_sim = sim_pos_matrix[range(batch_size), range(batch_size)]
        pos_sub = sim_pos_matrix.sum(dim=1).unsqueeze(dim=-1) - pos_sim
        loss_pos = pos_sim / pos_sub  # (100,80)
        loss_pos = - torch.log(loss_pos).mean()   # positive samples similarity get the smaller, the better


        loss = loss_pos # + loss_neg

        return loss


