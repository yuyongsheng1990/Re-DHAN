# -*- coding: utf-8 -*-
# @Time : 2022/12/2 20:14
# @Author : yysgz
# @File : S3_save_edge_index.py
# @Project : FinEvent Models
# @Description :

import torch
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy import sparse

# import os
# project_path = os.path.abspath(os.path.join(os.getcwd(), "../."))  # # 获取上上级路径
# offline_dataset_path = project_path + '/data'

# relations_ids = ['entity', 'userid', 'word'],分别读取这三个文件
def sparse_trans(datapath = None):
    relation = sparse.load_npz(datapath)  # (4762, 4762)
    all_edge_index = torch.tensor([], dtype=int)
    for node in range(relation.shape[0]):
        neighbor = torch.IntTensor(relation[node].toarray()).squeeze()  # IntTensor是torch定义的7中cpu tensor类型之一；
                                                                        # squeeze对数据维度进行压缩，删除所有为1的维度
        # del self_loop in advance
        neighbor[node] = 0  # 对角线元素置0
        neighbor_idx = neighbor.nonzero()  # 返回非零元素的索引, size: (43, 1)
        neighbor_sum = neighbor_idx.size(0)  # 表示非零元素数据量,43
        loop = torch.tensor(node).repeat(neighbor_sum, 1)  # repeat表示按列重复node的次数
        edge_index_i_j = torch.cat((loop, neighbor_idx), dim=1).t()  # cat表示按dim=1按列拼接；t表示对二维矩阵进行转置, node -> neighbor
        self_loop = torch.tensor([[node], [node]])
        all_edge_index = torch.cat((all_edge_index, edge_index_i_j, self_loop), dim=1)
        del neighbor, neighbor_idx, loop, self_loop, edge_index_i_j
    return all_edge_index  ## 返回二维矩阵，最后一维是node。 node -> nonzero neighbors

# if __name__=='__main__':
#     # save edge_index_[entity, userid, word].pt 文件
#     # 分别返回entity, userid, word这三个 homogeneous adjacency mx的非零邻居索引 non-zero neighbor index
#     relations = ['entity', 'userid', 'word']
#     for relation in relations:
#         relation_edge_index = sparse_trans(os.path.join(offline_dataset_path, 's_m_tid_%s_tid.npz' % relation))  # entity, (2, 487962); userid, (2, 8050); word, (2, 51498)
#         torch.save(relation_edge_index, offline_dataset_path + '/edge_index_%s.pt' % relation)