import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, n_h)  # input_ft1, input_ft264, out_ft; 64, 1, 64

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # aug_1 embedding, (90,64); original embedding(100,64); shuffle original fts,(100,64)
    def forward(self, aug_embed, h_pos, h_neg, device, s_bias1=None, s_bias2=None):

        aug_embed, h_pos, h_neg = aug_embed.to(device), h_pos.to(device), h_neg.to(device)
        aug_x = aug_embed.expand_as(h_pos)

        # Bilinear双向线性映射，将subgraph embedding 与pos embedding对齐；将sub embedding 2与neg embedding对齐。
        # pos对齐，相似度为1，neg为0. 表明是graph-level embedding 的一致性
        sc_1 = torch.sum(self.f_k(h_pos, aug_x), dim=1).unsqueeze(dim=1)  # 处理原始特征 tensor, dim=0, (1,64); dim=1, (64,1)
        # sc_1 = torch.sum(self.f_k(h_pos, aug_x), dim=0).unsqueeze(0)  # 处理原始特征 tensor, (1,64)
        sc_2 = torch.sum(self.f_k(h_neg, aug_x), dim=1).unsqueeze(1)  # 处理shuffled原始特征 tensor, (1,64)
        # sc_2 = torch.sum(self.f_k(h_neg, aug_x), dim=0).unsqueeze(0)  # 处理shuffled原始特征 tensor, (1,64)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # logits = torch.cat((sc_1, sc_2), 1)  # (1,128)
        logits = torch.cat((sc_1, sc_2), 0)  # (1,128)

        return logits

