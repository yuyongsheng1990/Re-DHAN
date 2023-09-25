import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np


def aug_subgraph(input_fea, input_bias_list, drop_percent=0.2):  # (4286,302); list:3,(4286,4286); (4286,);drop_p,0.1

    aug_input_fea_list = []
    aug_input_bias_list = []
    aug_sub_node_list = []

    for b in range(len(input_bias_list)):

        input_adj = input_bias_list[b]  # tensor, (4286,4286)
        node_num = input_fea.shape[0]  # 4286

        all_node_list = [i for i in range(node_num)]   # tensor,(4286,)
        s_node_num = int(node_num * (1 - drop_percent))  # subgraph 取9成node, int, 3857
        center_node_id = random.randint(0, node_num - 1)  # randint 随机返回一个整数 -> 作为中心节点, 859
        sub_node_id_list = [center_node_id]  # subgraph中心节点id列表
        all_neighbor_list = []

        for i in range(s_node_num - 1):

            all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()  # 第i个中心节点的non-zero neighbors, list:2,[1795,605]; 3,[1795,605,859]

            all_neighbor_list = list(set(all_neighbor_list))
            new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]  # 去除subgraph中心节点node的non-zero neighbors list. list:2,[1795,605];
            if len(new_neighbor_list) != 0:
                new_node = random.sample(new_neighbor_list, 1)[0]  # 若non-zero neighbors list非空，则随机取一个作为中心节点, 605
                sub_node_id_list.append(new_node)  # 加入subgraph 中心节点 list中
            else:
                break

        # sub_node_list, 2120, [859,605,1795,1483...], all_neighbor_list, 2120, [1,5,8,10,12...]; sub_node_id_list,2120,[430,2844,670,924...], all_neighbor_list,2120,[1,5,8,10,12...]
        drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])  # list:1207, 不在subgraph 中心节点node list的 drop node list

        aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)  # (2120,3703), subgraph node对应的features
        aug_input_adj = delete_row_col(input_adj, drop_node_list)  # tensor,（2120, 2120), subgraph node对应的adj_mx
        aug_input_idx = torch.tensor(range(aug_input_fea.shape[0]))

        aug_input_fea_list.append(aug_input_fea)
        aug_input_bias_list.append(aug_input_adj)
        aug_sub_node_list.append(aug_input_idx)

    return aug_input_fea_list, aug_input_bias_list, aug_sub_node_list  # 抽取子图node features和adjs

# edge perturbations for edge index
# single node edge index perturbation
def node_edge_perturbe(edge_index, drop_percent):
    node_list = edge_index[0]
    drop_num = max(int(node_list.shape[0] * drop_percent), 1)
    all_idx_list = [i for i in range(node_list.shape[0])]
    drop_index_list = random.sample(all_idx_list, drop_num)
    kp_index_bool = [i not in drop_index_list for i in all_idx_list]
    kp_edge_index = edge_index[0:, kp_index_bool]
    return kp_edge_index
# all nodes edge index perturbation
def aug_edge_index_perturbation(edge_index, drop_percent=0.2):
    # unique node id
    edge_index_0 = edge_index[0, :]
    uni_nodes = torch.unique(edge_index_0)
    pert_edge_index = torch.Tensor()
    for i in uni_nodes:
        edge_index_0_bool = (edge_index_0 == i)
        edge_index_extrct = edge_index[0:, edge_index_0_bool]
        if edge_index_extrct.shape[1] > 1:
            pert_node_edge_idx = node_edge_perturbe(edge_index_extrct, drop_percent)
            pert_edge_index = torch.cat((pert_edge_index, pert_node_edge_idx), dim=1)
        else:
            pert_edge_index = torch.cat((pert_edge_index, edge_index_extrct), dim=1)
    return pert_edge_index.int()

# edge perturbations for adj
def aug_edge_perturbation(input_adj, drop_percent=0.2):

    # input_edge_bias_list = []
    #
    # for b in range(len(input_adj_list)):
    #     input_adj = input_adj_list[b]  # (100, 100)
    zero_idx = torch.nonzero(input_adj == 0)
    nonzero_idx = torch.nonzero(input_adj)  # tensor(9620, 2)

    # drop edges
    nonzero_edge_num = int(nonzero_idx.size(0) / 2)  # 4864
    drop_num = int(nonzero_edge_num * drop_percent)  # 972
    edge_index_list = [i for i in range(nonzero_edge_num)]
    drop_index_list = random.sample(edge_index_list, drop_num)  # 获取扔掉的索引列表, list: 4864.
    for i in drop_index_list:
        # fetch edge idx
        drop_edge_idx = nonzero_idx[i]  # tensor([9, 83])
        # 重置为0
        input_adj[drop_edge_idx[0], drop_edge_idx[1]] = 0
        input_adj[drop_edge_idx[1], drop_edge_idx[0]] = 0

    # # add edges
    # zero_edge_num = int(zero_idx.size(0) / 2)  # 153
    # add_num = int(zero_edge_num * drop_percent)  # 30
    # edge_index_list = [i for i in range(zero_edge_num)]
    # add_index_list = random.sample(edge_index_list, add_num)
    # for i in add_index_list:
    #     # fetch add edge idx
    #     add_edge_idx = zero_idx[i]  # tensor([1, 68])
    #     # 重置为1
    #     input_adj[add_edge_idx[0], add_edge_idx[1]] = 1
    #     input_adj[add_edge_idx[1], add_edge_idx[0]] = 1

        # input_edge_bias_list.append(input_adj)

    return input_adj


def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out

# # numpy version: 归一化矩阵
# def normalize_adj(adj):  # tensor, (4286,4286)
#     """Symmetrically normalize adjacency matrix."""
#     adj = np.array(adj.cpu())
#     mu = np.mean(adj, axis=1)
#     std = np.std(adj, axis=1)
#     return torch.from_numpy((adj-mu)/std)

# tensor version: 归一化矩阵
def normalize_adj(adj):  # tensor, (4286,4286)
    """Symmetrically normalize adjacency matrix."""
    mu = torch.mean(adj, dim=0)
    std = torch.std(adj, dim=0)
    norm_adj = torch.sub(adj, mu)/std
    return norm_adj
