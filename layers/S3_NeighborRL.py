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
            sorted_neighbors, sorted_index = dist.sort(descending=False)  # 注意这里的sort_index指的是最佳neighbor在neighbors_idx中的索引，草了！
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
def RL_neighbor_filter(multi_r_data, RL_thtesholds, load_path, model_name, device):  # (2, 487962), (2, 8050), (2,51499)
    load_path = load_path + '/relation_config.npy'
    relation_config = np.load(load_path, allow_pickle=True)  # dict: 3. dict: 4762, neighbor similarity
    relation_config = relation_config.tolist()  # {0: neighbor_idx: tensor([0]), num_neighbors:1, sorted_index: tensor([0]); 1:{....}}
    relations = list(relation_config.keys())  # ['relation_0', 'relation_1', 'relation_2'], entity, userid, word
    multi_remain_data = []

    for i in range(len(relations)):  # 3, entity, userid, word
        edge_index: Tensor = multi_r_data[i]  # 二维矩阵，(2, 487962); (2, 8050); (2,51499), node -> neighbors
        unique_nodes = edge_index[1].unique()  # target neighbor node 4762
        num_nodes = unique_nodes.size(0)  # 指的是neighbor number, 4762
        num_nodes = torch.tensor(num_nodes).to(device)
        remain_node_index = torch.tensor([]).to(device)
        for node in range(num_nodes):
            # extract config，得到sorted neighbor nodes，sorted neighbor idx
            neighbors_idx = relation_config[relations[i]][node]['neighbors_idx'].to(device)
            num_neighbors = relation_config[relations[i]][node]['num_neighbors']
            num_neighbors = torch.tensor(num_neighbors).to(device)
            sorted_neighbors = relation_config[relations[i]][node]['sorted_neighbors'].to(device)  # 指的是相似度排序sorted similarity
            sorted_index = relation_config[relations[i]][node]['sorted_index'].to(device)  # 指的是sorted neighbor index

            if num_neighbors < 5:  # one agent
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue  # add limitations

            # DHAN model: A3C。
            '''
                DRL，大部分基础算法都是单线程的，也就是一个agent去跟环境交互产生经验。
                    - 那到底是不同num_neighbors在一个agent之下，即environment，不同的num_neighbors对应不同的state，即frame，然后不同的sub-agent给出policy 概率
                A2C和A3C这两种算法，创新的引入了并行架构，即整个agent由一个global network和多个并行独立的worker构成，每个worker都包括一套Actor-Critic网络。
                    - 每个worker都会独立的跟自己的环境，得到独立的采样经验，即相当于一个sub-agent。
                    - 多个workers独立探索一个环境的副本，相当于同时去探索环境的不同部分，并且使用不同的策略也可以极大提升多样性(只限于A3C)。
                    - 我这个code说起来还真不算探索不同的副本，只能算multi-agents对不同policy的探索
                    - A2C和A3C都是在并行架构下，目的是找到best policy:
                        - A3C，用不同的workers控制不同的policy，并行异步梯度更新。
                            - 只要有一个worker完成当前episode，主网络就会根据它的loss进行更新，并不影响其它仍旧在使用就策略的worker。
                            - global network会根据每个worker计算的loss更新参数，再去更新worker参数。
                        - A2C，用不同的workers控制相同的policy，并行同步梯度更新。
                            - A2C是A3C的同步版本，A2C也会构建多个进程，包括多个并行的worker。
                            - 但这些是同步的，即每轮training中，worker被要求平均loss，统一更新梯度，使用同一套策略。
                            - 所以，A2C与AC的最大不同，除了一个worker和多个workers的区别外，就是Advantage function，去减小high variance。
                更好的training方式是采用多线程的并行架构。这样既能实现experience replay，又能高效利用计算资源，提升训练效率。
                其实是multi-agents，每个不同的情况state对应一个agent (<5;<5<10;>10), policy做不同的动作action(threshold=0.1,0.2,0.3...),
                然后不同的action，会返回不同的value Q. Q的期望得到状态价值函数V，表明当前policy下，整体局势好不好。
                实际做了一个multi-agents。
                每个worker i 会做一个policy和critic的工作。action sampling probability。
            '''
            if (model_name == 'ReDHAN') or (model_name == 'ReDHANr'):  # 精确匹配
                if num_neighbors <= 10:  # second agent
                    threshold = float(RL_thtesholds[i]) + 0.3
                else:  # third agent
                    threshold = float(RL_thtesholds[i]) + 0.2
                num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
                filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]  # 注意这里的sort_index指的是最佳neighbor在neighbors_idx中的索引，草了！
            elif model_name == 'FinEvent':
            # # FinEvent model: AC
                threshold = float(RL_thtesholds[i])
                num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
                filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]  # 注意这里的sort_index指的是最佳neighbor在neighbors_idx中的索引，草了！
            else:
                num_kept_neighbors = 5
                filtered_neighbors_idx = neighbors_idx[:num_kept_neighbors]  # 注意这里的sort_index指的是最佳neighbor在neighbors_idx中的索引，草了！
            remain_node_index = torch.cat((remain_node_index, filtered_neighbors_idx))

        remain_node_index = remain_node_index.type('torch.LongTensor')
        edge_index = edge_index[:, remain_node_index]  # 这里对了，取的是neighbor idex
        multi_remain_data.append(edge_index)

    return multi_remain_data
