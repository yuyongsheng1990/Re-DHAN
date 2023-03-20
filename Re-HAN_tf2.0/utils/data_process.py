import numpy as np
import pandas as pd

import os
project_path = os.getcwd()

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):  # 邻接矩阵adjacency matrix
    nb_graphs = adj.shape[0]  # 行,即num_nodes
    mt = np.empty(adj.shape)  # 根据给定的维度和数值类型，返回一个新的ndarray数组，其元素不进行初始化
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])  # 返回一个二维的ndarray数组
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))  # 相乘
        for i in range(sizes[g]):  # 这个应该可以简化，直接对整个数组元素做操作！！！
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)  # 科学计数法，2.5 x 10^(-27)表示为：2.5e-27

###############################################
# This section of code adapted from tkipf/gcn #
###############################################

# load file
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# 生成掩码bool数组
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)  # 生成全是0的数组
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:  # 此人编码功底实在很烂
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))  # 序列化读出file对象
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))  # 创建一个空的lil_matrix
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))  # 创建一个shape的全是0的数组
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()  # vstack按行拼接
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # 从列表字典中返回一个图，获取邻接矩阵

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])  # 生成掩码bool数组
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)  # 全为0的shape numpy数组
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

'''转换为稀疏矩阵tuple'''
def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()  # 将稀疏矩阵转回numpy矩阵
    mu = f[train_mask == True, :].mean(axis=0)  # 按行求平均
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]  # sigma>0 get bool array
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # power数组元素求n次方，flatten是降到一维
    r_inv[np.isinf(r_inv)] = 0.  # isinf判断是否为无穷
    r_mat_inv = sp.diags(r_inv)  # 从对角线构造一个稀疏矩阵。
    features = r_mat_inv.dot(features)  # dot矩阵乘法
    return features.todense(), sparse_to_tuple(features)  # todense()转换成密集矩阵numpy.matrix

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix. 对称归一化邻接矩阵"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # 对角线为1的矩阵
    return sparse_to_tuple(adj_normalized)
