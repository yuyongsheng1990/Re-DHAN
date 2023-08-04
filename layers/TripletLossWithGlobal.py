# -*- coding: utf-8 -*-
# @Time : 2023/8/3 15:01
# @Author : yysgz
# @File : TripletLossWithGlobal.py
# @Project : Triplet Loss with global embedding as positive sample

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 计算triplet_loss损失函数
class TripletLossGlobal(nn.Module):
    '''
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels
    Triplets are generated using triplet_selector objects that take embeddings and targets and return indices of triplets
    '''

    def __init__(self, margin, triplet_selector):
        super(TripletLossGlobal, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector  # selector选择器对象，含有get_triplets方法

    def forward(self, embeddings, target):  # (100, 192); target, (100, )
        triplets_embeddings = self.triplet_selector.get_triplets(embeddings, target)  # 根据embeddings和labels返回最大loss的triplets list, (179, 3), positive:0,1; negative: 2.
        # if embeddings.is_cuda():
        #     triplets = triplets.cuda()
        # embeddings矩阵索引是单个元素，取行向量，多个行向量又组成矩阵！！
        ap_distances = (triplets_embeddings[:, 0] - triplets_embeddings[:, 1]).pow(2).sum(1)  # 第一列pre embedding 到 第二列pre embedding距离distance, (179,), =20 .pow(.5);
        an_distances = (triplets_embeddings[:, 0] - triplets_embeddings[:, 2]).pow(2).sum(1)  # (179,), =30 ().pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)  # (179, ), margin=3. 这表示positive pair distance - negative pair distance，自然是希望区分度越大越好，即loss越小越好，梯度下降adam

        return losses.mean(), len(triplets_embeddings)

class TripletSelector:
    '''
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets * 3]
    '''
    def __init__(self):
        pass
    def get_triplets(self, embeddings, labels):
        raise NotImplementedError  # 如果这个方法没有被子类重写，但是调用了，就会报错。

# 矩阵计算
def distance_matrix_computation(vectors):
    distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(-1,1)
    return distance_matrix


# 具体实现三元损失函数triplets_loss，返回某标签下ith元素和jth元素，其最大loss对应的其他标签元素索引
class FunctionNegativeTripletSelector(TripletSelector):
    '''
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    '''

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn  # 返回loss_values最大元素值的index的selector

    def get_triplets(self, embeddings, labels):  # (100, 192); target, (100, )
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = distance_matrix_computation(embeddings)  # pre embedding计算distance matrix (100, 100)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()  # 100
        triGlobal_embed_list = []

        # embedding计算的distance matrix与labels计算loss，取最大loss_index
        # 对于每个标签label
        for label in set(labels):
            label_mask = (labels == label)  # numpy array, (100,), ([True, False, True, True])
            label_indices = np.where(label_mask)[0]  # 同一标签索引, label_index, (3, ) array([0, 2, 3], dtype=int64)
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[  # (97, )
                0]  # 其他标签索引, not_label_index, array([1], dtype=int64)
            # anchor_pos_list = list(combinations(label_indices, 2))  # 2个元素的标签索引组合, list: 3, [(23, 66), (23, 79), (66, 79)]
            # anchor_pos_list = np.array(anchor_pos_list)  # 转换成np.array才能进行slice切片操作, (3, 2)
            pos_embeddings = embeddings[label_indices]
            ave_pos_embedding = np.mean(pos_embeddings, dim=1)


            for ap_idx in label_indices:
                ap_embedding = embeddings[ap_idx, :]
                ap_distance = (ap_embedding - ave_pos_embedding).pow(2).sum(1)
                loss_values = ap_distance - distance_matrix[  # loss_values, (97, ); ap_distance是同一label第一个位置到第二个位置的距离
                            torch.LongTensor(ap_idx), torch.LongTensor(negative_indices)] + self.margin  # anchor_positive, (2, ) [23, 66]; label第一个位置到其他label位置的距离
                loss_values = loss_values.data.cpu().numpy()  # (97, )
                hard_neg_max_index = self.negative_selection_fn(loss_values)  # hard返回最大loss的索引, 4
                if hard_neg_max_index is not None:  # if 最大loss值非空
                    hard_negative = negative_indices[hard_neg_max_index]  # 返回最大loss对应的negative label 位置
                    hard_neg_embedding = embeddings[hard_negative]
                    # 对于谋标签下ith元素和jth元素，其最大loss对应的其他标签元素索引
                    triGlobal_embed_list.append([ap_embedding, ave_pos_embedding, hard_neg_embedding])  # positive label位置, negative label位置

        if len(triGlobal_embed_list) == 0:
            triGlobal_embed_list.append([pos_embeddings[0], pos_embeddings[1], embeddings[negative_indices[0]]])

        triGlobal_embed_list = np.array(triGlobal_embed_list)  # (179, 3)
        return torch.LongTensor(triGlobal_embed_list)