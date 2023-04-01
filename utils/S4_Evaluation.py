# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:09
# @Author : yysgz
# @File : S4_Evaluation.py
# @Project : FinEvent Models
# @Description :

import numpy as np
from utils.S3_clustering import run_kmeans

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError  # 没有重写，就会报错

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


# 累加平均metrics
class AccumulateAccuracy(Metric):
    '''
    works with classification model
    '''

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


# 非零平均
class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path,  # extract_features (952, 192); y, (4762); indices, (952,)
             is_validation=True, cluster_type='kmeans'):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    if cluster_type == 'kmeans':
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices)
    elif cluster_type == 'dbscan':
        pass

    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' NMI: '
    message += str(nmi)
    message += '\n\t' + mode + ' AMi: '
    message += str(ami)
    message += '\n\t' + mode + ' ARI: '
    message += str(ari)
    if cluster_type == 'dbscan':
        message += '\n\t' + mode + ' best_eps: '
        message += '\n\t' + mode + ' best_min_Pts: '

    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices,
                                                        save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + 'tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets'
        message += str(n_classes)
        message += '\n\t' + mode + 'NMI: '
        message += str(nmi)
        message += '\n\t' + mode + 'AMI: '
        message += str(ami)
        message += '\n\t' + mode + 'ARI: '
        message += str(ari)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    np.save(save_path + '/%s_metric.npy' % mode, np.asarray([nmi, ami, ari]))
    if is_validation:
        return nmi
    else:
        pass