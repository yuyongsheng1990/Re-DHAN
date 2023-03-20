# -*- coding: utf-8 -*-
# @Time : 2022/12/4 15:27
# @Author : yysgz
# @File : kmeans_model.py
# @Project : process.py
# @Description :
import os
import numpy as np

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

import pickle  # 把训练好的模型存储起来

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import roc_curve, f1_score
from sklearn import manifold  # 一种非线性降维的手段
from sklearn.model_selection import train_test_split

def my_Kmeans(x, y, k=4, time=10, return_NMI=False):

    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    n_classes = len(np.unique(y))
    estimator = KMeans(n_clusters=n_classes)
    total_nmi = []
    total_ami = []
    total_ari = []
    if time:
        for i in range(time):
            estimator.fit(x, y)
            pred_y = estimator.predict(x)
            # 计算clustering evaluation指标
            batch_nmi = normalized_mutual_info_score(y, pred_y)
            batch_ami = adjusted_mutual_info_score(y, pred_y)
            batch_ari = adjusted_rand_score(y, pred_y)

            total_nmi.append(batch_nmi)
            total_ami.append(batch_ami)
            total_ari.append(batch_ari)

        nmi_score = np.mean(total_nmi)
        ami_score = np.mean(total_ami)
        ari_score = np.mean(total_ari)
        print(
            'NMI (10 avg): {:.4f} , AMI (10avg): {:.4f},  ARI (10avg): {:.4f}'.format(nmi_score, ami_score, ari_score))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return nmi_score, ami_score, ari_score