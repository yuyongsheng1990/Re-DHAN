import torch
import torch.nn as nn
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
    def forward(self, aug_embed, h_pos, h_neg, device, s_bias1=None, s_bias2=None):

        T = 0.2
        batch_size, _ = h_pos.size()
        x_pos_abs = h_pos.norm(dim=1)
        x_neg_abs = h_neg.norm(dim=1)
        x_aug_abs = aug_embed.norm(dim=1)

        sim_pos_matrix = torch.einsum('ik,jk->ij', h_pos, aug_embed) / torch.einsum('i,j->ij', x_pos_abs, x_aug_abs)
        sim_pos_matrix = torch.exp(sim_pos_matrix / T)
        pos_sim = sim_pos_matrix[range(batch_size), range(batch_size)]
        loss_pos = pos_sim / (sim_pos_matrix.sum(dim=1) - pos_sim)

        sim_neg_matrix = torch.einsum('ik,jk->ij', h_neg, aug_embed) / torch.einsum('i,j->ij', x_neg_abs, x_aug_abs)
        sim_neg_matrix = torch.exp(sim_neg_matrix / T)
        neg_sim = sim_neg_matrix[range(batch_size), range(batch_size)]
        loss_neg = neg_sim / (sim_neg_matrix.sum(dim=1) - neg_sim)

        loss = - torch.log(loss_pos + loss_neg).mean()

        return loss

