# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:08
# @Author : yysgz
# @File : S3_clustering.py
# @Project : FinEvent Models
# @Description :

# utility，功能
import numpy as np
import torch

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 交集
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
    # extract the features and labels of the test tweets
    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()  # detach()阻断反向传播，返回值为tensor；numpy()将tensor转换为numpy
        non_isolated_index = list(np.where(temp != 1)[0])  # np.where返回符合条件元素的索引index
        indices = intersection(indices, non_isolated_index)  # 取交集
    # Extract labels
    extract_labels = extract_labels.cpu().numpy()
    labels_true = extract_labels[indices]  # (952,)

    # Extrac features
    X = extract_features.cpu().detach().numpy()  # (952, 192)
    assert labels_true.shape[0] == X.shape[0]  # assert断言，在判断式false时触发异常
    n_test_tweets = X.shape[0]  # 952

    # Get the total number of classes
    n_classes = len(set(labels_true.tolist()))  # 49, unique()和nunique()不香吗？

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_

    nmi = metrics.normalized_mutual_info_score(labels_true, labels)  # 计算归一化互信息
    ami = metrics.adjusted_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)  # 计算兰德系数

    # Return number of test tweets, number of classes covered by the test tweets, and KMeans clustering NMI
    return n_test_tweets, n_classes, nmi, ami, ari

def evaluate_fn(test_y, pred_y):
    # ----------------------------Evaluation----------------------------------------------------
    # NMI, AMI, ARI
    bert_nmi = metrics.normalized_mutual_info_score(test_y, pred_y)
    bert_ami = metrics.adjusted_mutual_info_score(test_y, pred_y)
    bert_ari = metrics.adjusted_rand_score(test_y, pred_y)
    print('NMI: {:.4f}; AMI: {:.4f}; ARI: {:.4f}'.format(bert_nmi, bert_ami, bert_ari))
    return bert_nmi, bert_ami, bert_ari

def run_dbscan(extract_features, extract_labels, indices, save_path, former_save_path, isoPath=None):
    # extract the features and labels of the test tweets
    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()  # detach()阻断反向传播，返回值为tensor；numpy()将tensor转换为numpy
        non_isolated_index = list(np.where(temp != 1)[0])  # np.where返回符合条件元素的索引index
        indices = intersection(indices, non_isolated_index)  # 取交集
    # former embeddings
    former_embeddings = torch.load(former_save_path + '/final_embeddings.pt')
    former_embeddings = former_embeddings.cpu().numpy()
    # former labels
    former_labels = torch.load(former_save_path + '/final_labels.pt')
    former_labels = former_labels.cpu().numpy()
    # split data
    tran_x, test_x, tran_y, test_y = train_test_split(former_embeddings, former_labels, test_size=0.2, random_state=0)

    i, best_eps = 0.5, 0.5
    best_cnum = 2
    best_nmi = 0
    best_ami = 0
    best_ari = 0

    while i <= 20:
        for j in range(1, 4):
            message = 'DBSCAN eps: {:.2f}, min_samples: {:d}'.format(i, j) + '\n'
            print(message)
            # -----------------------------DBSCAN--------------------------------------------------------
            dbscan_model = DBSCAN(eps=i, min_samples=j)
            dbscan_model.fit(tran_x, tran_y)
            pred_y = dbscan_model.fit_predict(test_x)
            # print(dbscan_model.eps)
            # print(dbscan_model.min_samples)
            bert_nmi, bert_ami, bert_ari = evaluate_fn(test_y, pred_y)
            message += 'NMI: {:.4f}; AMI: {:.4f}; ARI: {:.4f}'.format(bert_nmi, bert_ami, bert_ari) + '\n'
            with open(save_path + '/dbscan_log.txt', 'a') as f:
                f.write(message)
            # ---------------------------best paras--------------------------------
            if best_nmi < bert_nmi:
                best_eps = i
                best_cnum = j
                best_nmi = bert_nmi
                best_ami = bert_ami
                best_ari = bert_ari
            else:
                continue
        i += 0.5
    message = '**DBSCAN best eps**: {:.2f}, min_samples: {:d}'.format(best_eps, best_cnum) + '\n'
    message += '**Best NMI: {:.4f}; **Best AMI: {:.4f}; **Best ARI: {:.4f}'.format(best_nmi, best_ami, best_ari) + '\n'
    print(message)
    with open(save_path + 'dbscan_log.txt', 'a') as f:
        f.write(message)

    # Extract labels
    extract_labels = extract_labels.cpu().numpy()
    labels_true = extract_labels[indices]  # (952,)

    # Extrac features
    X = extract_features.cpu().detach().numpy()  # (952, 192)
    assert labels_true.shape[0] == X.shape[0]  # assert断言，在判断式false时触发异常
    n_test_tweets = X.shape[0]  # 952

    # dbscan clustering
    kmeans = DBSCAN(eps=best_eps, min_samples=best_cnum).fit(X)
    labels = kmeans.labels_

    # Get the total number of classes
    n_classes = len(set(labels_true.tolist()))  # 49, unique()和nunique()不香吗？

    nmi, ami, ari = evaluate_fn(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and KMeans clustering NMI
    return n_test_tweets, n_classes, nmi, ami, ari


# 用softmax分类
def run_softmax(extract_features, extract_labels, indices, isoPath=None):
    # extract the features and labels of the test tweets
    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()  # detach()阻断反向传播，返回值为tensor；numpy()将tensor转换为numpy
        non_isolated_index = list(np.where(temp != 1)[0])  # np.where返回符合条件元素的索引index
        indices = intersection(indices, non_isolated_index)  # 取交集
    # Extract labels
    extract_labels = extract_labels.cpu().numpy()
    labels_true = extract_labels[indices]  # (952,)

    # Extrac features
    X = extract_features.cpu().detach().numpy()  # (952, 192)
    assert labels_true.shape[0] == X.shape[0]  # assert断言，在判断式false时触发异常
    n_test_tweets = X.shape[0]  # 952

    # Get the total number of classes
    n_classes = len(set(labels_true.tolist()))  # 49, unique()和nunique()不香吗？

