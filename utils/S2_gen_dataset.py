# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:04
# @Author : yysgz
# @File : S2_gen_dataset.py
# @Project : FinEvent Models
# @Description :

import numpy as np
import os
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from scipy import sparse

from torch_geometric.data import Data
from torch_sparse.tensor import SparseTensor
# from .utils import generateMasks, gen_offline_masks，是指从utils.py文件中导入函数: generatemasks, gen_offline_masks.

# relations_ids = ['entity', 'userid', 'word'],分别读取这三个文件
def sparse_trans(datapath = 'incremental_0808/0/s_m_tid_userid_tid.npz'):
    relation = sparse.load_npz(datapath)
    all_edge_index = torch.tensor([], dtype=int)
    for node in range(relation.shape[0]):
        neighbor = torch.IntTensor(relation[node].toarray()).squeeze()  # IntTensor是torch定义的7中cpu tensor类型之一；
                                                                        # squeeze对数据维度进行压缩，删除所有为1的维度
        # del self_loop in advance
        neighbor[node] = 0
        neighbor_idx = neighbor.nonzero()  # 返回非零元素的索引
        neighbor_sum = neighbor_idx.size(0)  # 表示第0维度的数据量
        loop = torch.tensor(node).repeat(neighbor_sum, 1)  # repeat表示沿着指定的维度重复tensor的次数
        edge_index_i_j = torch.cat((loop, neighbor_idx), dim=1).t()  # cat表示拼接；t表示对二维矩阵进行转置
        self_loop = torch.tensor([[node], [node]])
        all_edge_index = torch.cat((all_edge_index, edge_index_i_j, self_loop), dim=1)
        del neighbor, neighbor_idx, loop, self_loop, edge_index_i_j
    return all_edge_index

def coo_trans(datapath = 'incremental_0808/0/s_m_tid_userid_tid.npz'):
    relation: csr_matrix = sparse.load_npz(datapath)
    relation: coo_matrix = relation.tocoo()
    sparse_edge_index = torch.LongTensor([relation.row, relation.col])  # sparse稀疏矩阵用三元组(row,col,data)来存储非零元素信息
    return sparse_edge_index

def create_dataset(loadpath, relation, mode):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features_embeddings.npy'))
    features = torch.FloatTensor(features)
    print('features laoded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    relation_edge_index = coo_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation))
    print('edge index laoded')
    data = Data(x=features, edge_index=relation_edge_index, y=labels)
    data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
    train_i, i = mode[0], mode[1]
    if train_i == i:
        data.train_mask, data.val_mask = generateMasks(len(labels), data_split, train_i, i)
    else:
        data.test_mask = generateMasks(len(labels), data_split, train_i, i)
    return data


'''
mode: (train_i, i)
message block 0-train_i, as training dataset
random selection from training dataset, as validation dataset
other message blocks between train_i and i, having no labels
message block i, as test dataset
'''


# 返回training, validation, test data
def create_homodataset(loadpath, mode, valid_percent=0.2):
    features = np.load(os.path.join(loadpath, 'features_embeddings.npy'))  # features embeddings
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)

    data = Data(x=features, edge_index=None, y=labels)  # torch_geometric提供的图数据类型Data，x表示tensor矩阵，
    # 形状为[num_nodes, num_node_features];
    # edge_index表示coo格式的图的边关系，形状为[2, num_edge]
    data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
    # load number of message in each blocks
    # e.g. data_split = [  500  ,   100, ...,  100]
    #                    block_0  block_1    block_n
    train_i, i = mode[0], mode[1]
    if train_i == i:
        data.train_mask, data.val_mask = generateMasks(len(labels), data_split, train_i, i, valid_percent)
    else:
        data.test_mask = generateMasks(len(labels), data_split, train_i, i)
    return data


def create_offline_homodataset(loadpath, mode):
    features = np.load(loadpath +'/features.npy')
    features = torch.FloatTensor(features)  # x embedding, (4762, 302)
    print('features loaded')
    labels = np.load(loadpath + '/labels.npy')
    print('labels loaded')
    labels = torch.LongTensor(labels)  # label, 4762
    data = Data(x=features, edge_index=None, y=labels)  # Data类定义数据
    data.train_mask, data.val_mask, data.test_mask = gen_data_slices(len(labels))  # 在随机打乱的序列中选取train_idx, val_idx, test_idx

    return data


# get edge_index_relation data
def create_multi_relational_graph(loadpath, relations, mode):
    # edge index: 二维矩阵。node-> nonzero neighbors; node-> repeated nodes.
    multi_relation_edge_index = [torch.load(loadpath + '/edge_index_%s.pt' % relation) for relation in relations]
    print('sparse trans...')
    print('edge index loaded')

    return multi_relation_edge_index

def save_multi_relational_graph(loadpath, relations, mode):
    for relation in relations:
        relation_edge_index = sparse_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation))
        torch.save(relation_edge_index, loadpath + '/' + str(mode[1]) + '/edge_index_%s.pt' % relation)


# 返回training，validation, test的索引indices
def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, remove_absolete=2):
    '''
    Intro:
    This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
    If remove_obsolete mode 0 or 1:
    For initial/maintenance epochs:
    - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
    - Randomly sample validation_percent of the training indices as validation indices
    For inference(prediction) epochs:
    - The (i + 1)th block (block i) is used as test set.

    Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, only their features and structural info are leveraged)

    :param length: the length of label list
    :param data_split: loaded splited data (generated in custom_message_graph.py)
    :param train_i, i: flag, indicating for initial/maintenance stage if train_i == i and inference stage for others
    :param validation_percent: the percent of validation data occupied in whole dataset
    :param save_path: path to save data
    :param num_indices_to_remove: number of indices ought to be removed
    :returns train indices, validation indices or test indices
    '''
    # step1: verify total number of nodes
    assert length == data_split[i]  # 500

    # step2.0: if is in initial/maintenance epochs, generate train and validation indices
    if train_i == i:
        # step3: randomly shuffle the graph indices
        train_indices = torch.randperm(length)  # 返回一个随机打散的0-n-1 tensor数组
        # step4: get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # step5: sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        train_indices = train_indices[n_validation_samples:]
        # step6: save indices
        if save_path is not None:
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(validation_indices, save_path + '/validation_indices.pt')
        return train_indices, validation_indices
    # step2.1: if is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.arange(0, (data_split[i]), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
        return test_indices


def gen_data_slices(length, validation_percent=0.2, test_percent=0.1):
    test_length = int(length * test_percent)
    valid_length = int(length * validation_percent)
    train_length = length - valid_length - test_length

    samples = torch.randperm(length)  # 返回随机打散的0~n-1的tensor数组
    train_indices = samples[:train_length]
    valid_indices = samples[train_length: train_length + valid_length]
    test_indices = samples[train_length + valid_length:]

    return train_indices, valid_indices, test_indices

import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from scipy import sparse
import random

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.loader import NeighborSampler, RandomNodeSampler


# NeighborSampler返回结果：batch_size, n_id,adjs(edge_index,e_id,size)
class MySampler(object):
    def __init__(self, sampler) -> None:
        super().__init__()
        self.sampler = sampler

    def sample(self, multi_relational_edge_index: List[Tensor], node_idx, sizes, batch_size):
        if self.sampler == 'RL_sampler':
            return self._RL_sample(multi_relational_edge_index, node_idx, sizes, batch_size)
        elif self.sampler == 'random_sampler':
            return self._random_sample(multi_relational_edge_index, node_idx, batch_size)
        elif self.sampler == 'const_sampler':
            return self._const_sample(multi_relational_edge_index, node_idx, batch_size)

    def _RL_sample(self, multi_relational_edge_index: List[Tensor], node_idx, sizes, batch_size):
        outs = []  # filtered 二维矩阵 node -> non-zero neighbors, (2,104479); (2,6401); (2,15072); node_idx = batch_nodes, sizes=[-1,-1]
        all_n_ids = []
        for id, edge_index in enumerate(multi_relational_edge_index):  # 返回数据和数据下标, entity, userid, word, (2,104479); (2,6401); (2,15072)
            loader = NeighborSampler(edge_index=edge_index,  # (2,104479); (2,6401); (2,15072)
                                     sizes=sizes,  # [-1, -1]
                                     node_idx=node_idx,  # batch_nodes
                                     return_e_id=False,
                                     batch_size=batch_size,  # 100
                                     num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):  # NeighborSampler返回结果：batch_size, n_ids, adjs(edge_index,e_id,size)
                outs.append(adjs)  # adjs list包含了从第L层到第1层采样的结果，adjs中子图是从大到小的。包括：edge_index, bipartite二分子图中source节点到target节点的边; e_id是edge_index在原始图中的id; size是子图shape
                all_n_ids.append(n_ids)  # n_ids是包含所有在L层卷积中遇到的节点的list，且target节点在n_ids前几位

            assert id == 0  # 断言，条件为false时触发，中断程序
        return outs, all_n_ids

    def _random_sample(self, multi_relational_edge_index: List[Tensor], node_idx, batch_size):
        outs = []
        all_n_ids = []
        sizes = [random.randint(10, 100), random.randint(10, 50)]
        for edge_index in multi_relational_edge_index:
            loader = NeighborSampler(edge_index=edge_index, sizes=sizes, node_idx=node_idx, return_e_id=False,
                                     batch_size=batch_size, num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                outs.append(adjs)
                all_n_ids.append(n_ids)
            assert id == 0
        return outs, all_n_ids

    def _const_sample(self, multi_relational_edge_index: List[Tensor], node_idx, batch_size):
        outs = []
        all_n_ids = []
        sizes = [25, 15]
        for edge_index in multi_relational_edge_index:
            loader = NeighborSampler(edge_index=edge_index, sizes=sizes, node_idx=node_idx, return_e_id=False,
                                     batch_size=batch_size, num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                outs.append(adjs)
                all_n_ids.append(n_ids)
            assert id == 0
        return outs, all_n_ids

def save_embeddings(extracted_features, save_path, file_name):
    torch.save(extracted_features, save_path + '/' + file_name)
    print('extracted features saved.')