# -*- coding: utf-8 -*-
# @Time : 2022/11/29 16:55
# @Author : yysgz
# @File : S3_NeighborRL.py
# @Project : utils Models
# @Description :

from typing import Any, Dict
import numpy as np
import torch
from torch.functional import Tensor
import math
import os

def cal_similarity_node_edge(multi_r_data, features, save_path=None):
    '''
    This is used to culculate the similarity between node and its neighbors in advance
    in order to avoid the repetitive computation.
    Args:
        multi_r_data ([type]): [description]
        features ([type]): [description]
        save_path ([type], optional): [description]. Defaults to None.
    '''
    relation_config: Dict[str, Dict[int, Any]] = {}
    for relation_id, r_data in enumerate(multi_r_data):
        node_config: Dict[int, Any] = {}
        r_data: Tensor  # entity,(2, 487962); (2, 8050); (2, 51498)
        unique_nodes = r_data[1].unique()  # 第二行 neighbor idx: entity, (4762, );
        num_nodes = unique_nodes.size(0)  # 4762
        for node in range(num_nodes):  # neighbor index
            # get neighbors' index
            neighbors_idx = torch.where(r_data[1] == node)[0]  # how many same neighbor, 返回neighbor index
            # get neghbors
            neighbors = r_data[0, neighbors_idx]  # 0 表示 node
            num_neighbors = neighbors.size(0)  # node number
            neighbors_features = features[neighbors, :]  # different node embedding
            target_features = features[node, :]  # neighbor embedding
            # calculate enclidean distance with broadcast
            dist: Tensor = torch.norm(neighbors_features - target_features, p=2, dim=1)  # torch.norm求a列维度(dim指定)的2(p指定)范数(长度)，即相似度
            # smaller is better and we use 'top p' in our paper
            # (threshold * num_neighbors) see RL_neighbor_filter for details
            sorted_neighbors, sorted_index = dist.sort(descending=False)  # 排序后的neighbor相似度，及neighbor index, 注意这里的neighbor也是index形式
            node_config[node] = {'neighbors_idx': neighbors_idx,
                                'sorted_neighbors': sorted_neighbors,
                                'sorted_index': sorted_index,
                                'num_neighbors': num_neighbors}
        relation_config['relation_%d' % relation_id] = node_config  # relation neighbor config
    if save_path is not None:
        print(save_path)
        save_path = os.path.join(save_path, 'relation_config.npy')
        np.save(save_path, relation_config)


# 返回filtered neighbor index
def RL_neighbor_filter(multi_r_data, RL_thtesholds, load_path, device):  # (2, 487962), (2, 8050), (2,51499)
    load_path = load_path + '/relation_config.npy'
    relation_config = np.load(load_path, allow_pickle=True)  # dict: 3. 4762, neighbor similarity
    relation_config = relation_config.tolist()
    relations = list(relation_config.keys())  # ['relation_0', 'relation_1', 'relation_2'], entity, userid, word
    multi_remain_data = []

    for i in range(len(relations)):  # 3, entity, userid, word
        edge_index: Tensor = multi_r_data[i]  # 二维矩阵，(2, 487962); (2, 8050); (2,51499), node -> neighbors
        unique_nodes = edge_index[1].unique()  # neighbor 4762, 这里应该也弄反了，应该是取node而不是neighbor, edge_index[0]
        num_nodes = unique_nodes.size(0)  # 指的是neighbor number
        num_nodes = torch.tensor(num_nodes).to(device)
        remain_node_index = torch.tensor([]).to(device)
        for node in range(num_nodes):
            # extract config，得到sorted neighbor nodes，sorted neighbor idx
            neighbors_idx = relation_config[relations[i]][node]['neighbors_idx'].to(device)
            num_neighbors = relation_config[relations[i]][node]['num_neighbors']
            num_neighbors = torch.tensor(num_neighbors).to(device)
            sorted_neighbors = relation_config[relations[i]][node]['sorted_neighbors'].to(device)  # 指的是相似度排序sorted similarity
            sorted_index = relation_config[relations[i]][node]['sorted_index'].to(device)  # 指的是sorted neighbor index

            if num_neighbors < 5:
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue  # add limitations

            threshold = float(RL_thtesholds[i])

            num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
            filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]
            remain_node_index = torch.cat((remain_node_index, filtered_neighbors_idx))

        remain_node_index = remain_node_index.type('torch.LongTensor')
        edge_index = edge_index[:, remain_node_index]  # 这里对了，取的是neighbor idex
        multi_remain_data.append(edge_index)

    return multi_remain_data