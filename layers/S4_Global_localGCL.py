# -*- coding: utf-8 -*-
# @Time : 2023/7/31 15:46
# @Author : yysgz
# @File : S4_Global_localGCL.py
# @Project : run_offline model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

gl_loss_fn = nn.BCEWithLogitsLoss()
class GlobalLocalGraphContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(GlobalLocalGraphContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, global_embeddings, local_embeddings, labels):  # labels, (100, 1)
        """
        Global-Local Graph Contrastive Loss function.

        Parameters:
            global_embeddings (torch.Tensor): Global embeddings of nodes or graphs (shape: [batch_size, embedding_dim]).
            local_embeddings (torch.Tensor): Local embeddings of nodes or graphs (shape: [batch_size, num_neighbors, embedding_dim]).
            labels (torch.Tensor): The ground truth labels for the nodes or graphs (shape: [batch_size]).

        Returns:
            torch.Tensor: The Global-Local Graph Contrastive Loss value.
        """
        batch_size, embedding_dim = global_embeddings.shape  # 100, 64
        num_neighbors = local_embeddings.shape[0]  # 100

        # Reshape local_embeddings to [batch_size * num_neighbors, embedding_dim]
        local_embeddings = local_embeddings.view(batch_size, embedding_dim)  # (100, 64)

        # Concatenate global_embeddings and local_embeddings
        all_embeddings = torch.cat([global_embeddings, local_embeddings], dim=0)  # 按行拼接, (200, 64)

        # Compute pairwise similarity matrix
        similarity_matrix = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature  # (200, 200), 对称矩阵, 每个original element和aug element -> similarity

        # Construct positive and negative masks
        test_tensor = labels.view(1, -1)  # (1, 100)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1))  # eq function, element-wise comparison and return boolean tensor; view 重构 tensor size -> (100 , 100)，相同label的True
        mask = mask.repeat(2, 2)  # (200, 200) <- 针对对称矩阵，需要做这步repeat

        # Ignore diagonal elements for positive samples
        mask.fill_diagonal_(0)  # 填充对角线元素为0

        # Compute logit scores for positive and negative samples
        positive_scores = similarity_matrix[mask].view(-1,  1)  # 所有相同label元素对应的similarities matrix, 并将其resize为(1576, 1) shape
        negative_scores = similarity_matrix[~mask].view(-1, 1)  # ~ 取反，所有不相同label元素对应的similarities matrix, 并将其resize为(38424, 1) shape

        # Calculate contrastive loss using InfoNCE loss (Noise Contrastive Estimation)
        logits = torch.cat([positive_scores, negative_scores], dim=0)  # 按行拼接, (40000, 1)
        labels = torch.cat([torch.ones(positive_scores.size(0), 1), torch.zeros(negative_scores.size(0), 1)], dim=0)   # (40000, )
        contrastive_loss = gl_loss_fn(logits, labels)

        return contrastive_loss