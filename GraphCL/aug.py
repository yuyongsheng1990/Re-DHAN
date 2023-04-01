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





def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out


def normalize_adj(adj):  # tensor, (4286,4286)
    """Symmetrically normalize adjacency matrix."""
    adj = np.array(adj)
    mu = np.mean(adj, axis=1)
    std = np.std(adj, axis=1)
    return torch.from_numpy((adj-mu)/std)
