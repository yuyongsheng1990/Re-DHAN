# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:12
# @Author : yysgz
# @File : run_offline model.py
# @Project : FinEvent Models
# @Description :
import random

import numpy as np
import scipy.sparse as sp
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import gc
import time
from typing import List
import os
project_path = os.getcwd()

from layers.S2_TripletLoss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector
from layers.NCELoss import NCECriterion
from layers.S3_NeighborRL import cal_similarity_node_edge, RL_neighbor_filter
from baselines.MarGNN import MarGNN

from utils.S2_gen_dataset import create_offline_homodataset, create_multi_relational_graph, MySampler, save_embeddings
from utils.S4_Evaluation import AverageNonzeroTripletsMetric, evaluate

from models.HeteGAT_multi import HeteGAT_multi
from models.HeteGAT_multi_geometric import HeteGAT_multi_geometirc

from GraphCL import aug, discriminator
def args_register():
    parser = argparse.ArgumentParser()  # 创建参数对象
    # 添加参数
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of initial-training/maintenance-training epochs.')
    parser.add_argument('--window_size', default=3, type=int, help='Maintain the model after predicting window_size blocks.')
    parser.add_argument('--patience', default=5, type=int,
                        help='Early stop if perfermance did not improve in the last patience epochs.')
    parser.add_argument('--margin', default=3, type=float, help='Margin for computing triplet losses')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size (number of nodes sampled to compute triplet loss in each batch)')
    parser.add_argument('--hid_dim', default=128, type=int, help='Hidden dimension')
    parser.add_argument('--out_dim', default=64, type=int, help='Output dimension of tweet representation')
    parser.add_argument('--heads', default=4, type=int, help='Number of heads used in GAT')
    parser.add_argument('--validation_percent', default=0.2, type=float, help='Percentage of validation nodes(tweets)')
    parser.add_argument('--attn_drop', default=0.5, type=float, help='masked probability for attention layer')
    parser.add_argument('--feat_drop', default=0.0, type=float, help='dropout probability for feature embedding in attn')
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False, action='store_true',
                        help='If true, use hardest negative messages to form triplets. Otherwise use random ones')
    parser.add_argument('--is_shared', default=False)
    parser.add_argument('--inter_opt', default='cat_w_avg')
    parser.add_argument('--is_initial', default=True)
    parser.add_argument('--sampler', default='RL_sampler')
    parser.add_argument('--cluster_type', default='kmeans', help='Types of clustering algorithms')  # DBSCAN

    # RL-0，第一个强化学习是learn the optimal neighbor weights
    parser.add_argument('--threshold_start0', default=[[0.2], [0.2], [0.2]], type=float,
                        help='The initial value of the filter threshold for state1 or state3')
    parser.add_argument('--RL_step0', default=0.02, type=float, help='The starting epoch of RL for state1 or state3')
    parser.add_argument('--RL_start0', default=0, type=int, help='The starting epoch of RL for state1 or state3')

    # RL-1，第二个强化学习是learn the optimal DBSCAN params.
    parser.add_argument('--eps_start', default=0.001, type=float, help='The initial value of the eps for state2')
    parser.add_argument('--eps_step', default=0.02, type=float, help='The step size of eps for state2')
    parser.add_argument('--min_Pts_start', default=2, type=int, help='The initial value of the min_Pts for state2')
    parser.add_argument('--min_Pts_step', default=1, type=int, help='The step size of min_Pts for state2')

    # NCE loss params
    parser.add_argument('--nce_m', default=2.0, type=float, help='the m value for calculating NCE loss')
    parser.add_argument('--nce_eps', default=2, type=float, help='the temperature param for NCE loss function')
    # other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help='Use cuda')
    parser.add_argument('--data_path', default= project_path + '/data', type=str, help='graph data path')  # 相对路径，.表示当前所在目录
    # parser.add_argument('--result_path', default='./result/offline result', type=str,
    #                     help='Path of features, labels and edges')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None, type=str,
                        help='File path that contains the training, validation and test masks')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval')
    # subgraph contrastive loss和triplet loss加权概率
    parser.add_argument('--loss_p', default=0.8, type=float, help='A percent value used to weighted add triplet loss and subgraph contrastive loss')
    args = parser.parse_args(args=[])  # 解析参数

    return args


# 将二维矩阵list 转换成adj matrix list
def relations_to_adj(filtered_multi_r_data, nb_nodes=None):
    relations_mx_list = []
    for r_data in filtered_multi_r_data:
        data = np.ones(r_data.shape[1])
        relation_mx = sp.coo_matrix((data, (r_data[0], r_data[1])), shape=(nb_nodes, nb_nodes), dtype=int)
        relations_mx_list.append(torch.tensor(relation_mx.todense()))
    return relations_mx_list

# relations_ids = ['entity', 'userid', 'word'],分别读取这三个文件
def sparse_trans(relation_list):  # (4762, 4762)
    all_edge_index_list = []
    for i in range(len(relation_list)):
        relation = sp.csr_matrix(relation_list[i])
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
        all_edge_index_list.append(all_edge_index)
    return all_edge_index_list  ## 返回二维矩阵，最后一维是node。 node -> nonzero neighbors

# 计算偏差矩阵
def adj_to_bias(adj, nb_nodes, nhood=1):  # adj,(3025, 3025); sizes, [3025]
    mt = np.eye(adj.shape[0])
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.eye(adj.shape[1])))  # 相乘
    mt = np.where(mt > 0, 1, mt)
    return torch.from_numpy(-1e9 * (1.0 - mt))  # 科学计数法，2.5 x 10^(-27)表示为：2.5e-27

def offline_FinEvent_model(train_i,  # train_i=0
                           i,  # i=0
                           args,
                           metrics,
                           embedding_save_path,
                           loss_fn,
                           model=None,
                           loss_fn_dgi=None):
    # step1: make dir for graph i
    # ./incremental_0808//embeddings_0403005348/block_xxx
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)

    # step2: load data
    relation_ids: List[str] = ['entity', 'userid', 'word']
    homo_data = create_offline_homodataset(args.data_path, [train_i, i])  # (4762, 302), 包含x: feature embedding和y: label, generate train_slices (3334), val_slices (952), test_slices (476)
    # 返回entity, userid, word的homogeneous adj mx中non-zero neighbor idx。二维矩阵，node -> non-zero neighbor idx, (2,487962), (2,8050), (2, 51498)
    features_list = [homo_data.x, homo_data.x, homo_data.x]  # list:3, (4762, 302)
    multi_r_data = create_multi_relational_graph(args.data_path, relation_ids, [train_i, i])
    num_relations = len(multi_r_data)  # 3

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # input dimension (300 in our paper)
    num_dim = homo_data.x.size(0)  # 4762
    feat_dim = homo_data.x.size(1)  # embedding dimension, 302
    nb_classes = len(np.unique(homo_data.y))
    attn_drop = args.attn_drop
    feat_drop = args.feat_drop

    # prepare graph configs for node filtering
    if args.is_initial:
        print('prepare node configures...')  # 计算neighbor node与node的相似度，并排序sorted
        cal_similarity_node_edge(multi_r_data, homo_data.x, save_path_i)
        filter_path = save_path_i
    else:
        filter_path = args.data_path + str(i)

    # Multi-Agent
    # initialize RL thresholds, RL_threshold: [[.5],[.5],[.5]]
    RL_thresholds = torch.FloatTensor(args.threshold_start0).to(device)  # [[0.2], [0.2], [0.2]]
    # RL_filter means extract limited sorted neighbors based on RL_threshold and neighbor similarity, return filtered node -> neighbor index
    if args.sampler == 'RL_sampler':
        filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds,
                                                   filter_path)  # filtered 二维矩阵, (2,104479); (2,6401); (2,15072)
    else:
        filtered_multi_r_data = multi_r_data

    if model is None:  # pre-training stage in our paper
        # print('Pre-Train Stage...')
        # # HAN_0 model without RL_filter and Neighbor_sampler
        # relations_mx_list = relations_to_adj(multi_r_data, nb_nodes=num_dim)
        # biases_mat_list = [adj_to_bias(adj, num_dim, nhood=1).to(device) for adj in relations_mx_list]  # 偏差矩阵list:3,tensor, (4762,4762)
        # model = HeteGAT_multi(feature_size=feat_dim, nb_classes=nb_classes, nb_nodes=num_dim, attn_drop=attn_drop,
        #                       feat_drop=feat_drop, hid_dim=args.hid_dim, out_dim=args.out_dim,
        #                       bias_mx_len=num_relations, hid_units=[8], n_heads=[8,1], activation=nn.ELU())

        # HAN_1 model with RL_filter and no Neighbor_sampler
        relations_mx_list = relations_to_adj(filtered_multi_r_data, nb_nodes=num_dim)
        biases_mat_list = [adj_to_bias(adj, num_dim, nhood=1).to(device) for adj in relations_mx_list]  # 偏差矩阵list:3,tensor, (4762,4762)
        model = HeteGAT_multi(feature_size=feat_dim, nb_classes=nb_classes, nb_nodes=num_dim, attn_drop=attn_drop,
                              feat_drop=feat_drop, hid_dim=args.hid_dim, out_dim=args.out_dim,
                              bias_mx_len=num_relations, hid_units=[8], n_heads=[8,1], activation=nn.ELU())

        # # HAN_2 model with RL_filter and Neighbor_sampler，这要torch_geometric重写HAN模型，要不然用不上FinEvent中的neighbor_sampler.
        # # 所以，既要用到adjs_list for RL_sampler，也要用到bias_list for HAN algorithm.
        # relations_mx_list = relations_to_adj(filtered_multi_r_data, nb_nodes=num_dim)  # 邻接矩阵list:3,tensor, (4762,4762)
        # biases_mat_list = [adj_to_bias(adj, num_dim, nhood=1).to(device) for adj in relations_mx_list]  # 偏差矩阵list:3,tensor, (4762,4762)
        # model = HeteGAT_multi_geometirc(feature_size=feat_dim, nb_classes=nb_classes, nb_nodes=num_dim, attn_drop=attn_drop,
        #                                 feat_drop=feat_drop, hid_dim=args.hid_dim, out_dim=args.out_dim,
        #                                 bias_mx_len=num_relations, hid_units=[8], n_heads=[8,1], activation=nn.ELU())

        # # baselin 1: feat_dim=302; hidden_dim=128; out_dim=64; heads=4; inter_opt=cat_w_avg; is_shared=False
        # model = MarGNN((feat_dim, args.hid_dim, args.out_dim, args.heads),
        #                num_relations=num_relations, inter_opt=args.inter_opt, is_shared=args.is_shared)
    else:
        biases_mat_list = None

    # define sampler
    sampler = MySampler(args.sampler)  # RL_sampler
    # load model to device
    model.to(device)

    # define NCE Loss
    # loss_fn = NCECriterion(args.nce_m, args.nce_eps)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # record training log
    message = '\n------Start initial training /maintaining using block' + str(i) + '------\n'
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    # step12.0: record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    gcl_loss_fn = nn.BCEWithLogitsLoss()
    gcl_dropout_percent = 0.1
    gcl_disc = discriminator.Discriminator(args.out_dim)
    gcl_loss = None
    loss_p = args.loss_p  # loss加权概率

    # data.train_mask, data.val_mask, data.test_mask = gen_offline_masks(len(labels))
    train_num_samples, valid_num_samples, test_num_samples = homo_data.train_mask.size(0), homo_data.val_mask.size(
        0), homo_data.test_mask.size(0)  # 3354, 952, 476, 这里的train_mask指的是train_idx,不是bool类型
    all_num_samples = train_num_samples + valid_num_samples + test_num_samples
    torch.save(homo_data.train_mask, save_path_i + '/train_mask.pt')
    torch.save(homo_data.val_mask, save_path_i + '/valid_mask.pt')
    torch.save(homo_data.test_mask, save_path_i + '/test_mask.pt')

    # step12.1: record validation nmi of all epochs before early stop
    all_vali_nmi = []
    # step12.2: record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # step12.3: record the time spent in mins on each epoch
    mins_train_epochs = []

    # step13: start training------------------------------------------------------------
    print('----------------------------------training----------------------------')
    for epoch in range(args.n_epochs):
        start_epoch = time.time()
        losses = []
        total_loss = 0.0

        for metric in metrics:
            metric.reset()

        # step13.0: forward
        model.train()

        # mini-batch training
        batch = 0
        num_batches = int(train_num_samples / args.batch_size) + 1  # 34
        for batch in range(num_batches):
            start_batch = time.time()
            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, train_num_samples)
            batch_nodes = homo_data.train_mask[i_start:i_end]  # 100个train_idx
            batch_labels = homo_data.y[batch_nodes]

            # sampling neighbors of batch nodes
            # adjs是RL_sampler采样的子图edge; n_ids是采样过程中遇到的node list。都是list: 3, 对应entity, userid, word
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                         batch_size=args.batch_size)
            optimizer.zero_grad()  # 将参数置0

            batch_node_list = [batch_nodes, batch_nodes, batch_nodes]

            pred = model(features_list, biases_mat_list, batch_node_list, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
            # pred = model(features_list, biases_mat_list, batch_node_list, adjs, n_ids, device, RL_thresholds)  # HAN_2 model
            # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # Fin-Event pred: x表示combined feature embedding, 302; pred, 其实是个embedding (100,192)

            loss_outputs = loss_fn(pred, batch_labels)  # (12.8063), 179
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            """
            '''----------GraphCL 对比学习-------------------------'''
            # subgraph sampling
            batch_features = homo_data.x[batch_nodes]
            batch_biases_mat_list = [biases[batch_nodes][:,batch_nodes] for biases in biases_mat_list]
            aug_fts_list1, aug_bias_list1, aug_sub_node_list1 = aug.aug_subgraph(batch_features, batch_biases_mat_list, drop_percent=gcl_dropout_percent)
            aug_fts_list2, aug_bias_list2, aug_sub_node_list2 = aug.aug_subgraph(batch_features, batch_biases_mat_list, drop_percent=gcl_dropout_percent)
            # 归一化normalization
            gcl_biases_mat_list = [aug.normalize_adj(adj + np.eye(adj.shape[0])) for adj in biases_mat_list]  # 原始adj matrix做归一化normalize, ndarray, (3327,3327)
            aug_bias_list1 = [aug.normalize_adj(aug_bias1 + np.eye(aug_bias1.shape[0])) for aug_bias1 in aug_bias_list1]  # aug_adj1做归一化, tensor, (2120,2120)
            aug_bias_list2 = [aug.normalize_adj(aug_bias2 + np.eye(aug_bias2.shape[0])) for aug_bias2 in aug_bias_list2]  # aug_adj2做归一化, tensor,(2120,2120)
            # negative samples
            features_neg = homo_data.x.clone()  # shuffled tensor
            features_neg = features_neg[torch.randperm(features_neg.shape[0])]
            features_neg_list = [features_neg,features_neg,features_neg]
            # 构建标签label
            lbl_1 = torch.ones(1, args.out_dim)  # labels for aug_1, (1,192)
            lbl_2 = torch.zeros(1, args.out_dim)  # (1,192)
            lbl = torch.cat((lbl_1, lbl_2), 1)  # (1,128)
            # 基于data augmentation生成关于original features和shuffled features的embedding
            h_pos = model(features_list, gcl_biases_mat_list, batch_node_list, device, RL_thresholds)  # HAN_0/HAN_1计算正样本 positive feature embeddings, (100, 64)
            h_neg = model(features_neg_list, gcl_biases_mat_list, batch_node_list, device, RL_thresholds)  # HAN_0/HAN_1构建负样本 negative feature embeddings
            # h_pos = model(features_list, gcl_biases_mat_list, batch_node_list, adjs, n_ids, device, RL_thresholds)  # HAN_2计算正样本 positive feature embeddings, (100, 64)
            # h_neg = model(features_neg_list, gcl_biases_mat_list, batch_node_list, adjs, n_ids, device, RL_thresholds)  # HAN_2构建负样本 negative feature embeddings
            # 构建subgraph augmentation embedding
            h_aug_1 = model(aug_fts_list1, aug_bias_list1, aug_sub_node_list1, device, RL_thresholds)  # HAN_0/HAN_1 构建 subgraph augmentation embeddings, (90, 64)
            h_aug_2 = model(aug_fts_list2, aug_bias_list2, aug_sub_node_list2, device, RL_thresholds)  # HAN_0/HAN_1 计算正样本 subgraph augmentation embeddings
            aug_adjs, aug_n_ids = sampler.sample(sparse_trans(aug_bias_list1),
                                                 node_idx=aug_sub_node_list1[0], sizes=[-1, -1],
                                                 batch_size=aug_fts_list1[0].shape[0])  # RL_sampler from aug_adj for HAN_2
            # h_aug_1 = model(aug_fts_list1, aug_bias_list1, aug_sub_node_list1, aug_adjs, aug_n_ids, device, RL_thresholds)  # HAN_2 构建 subgraph augmentation embeddings, (90, 64)
            # h_aug_2 = model(aug_fts_list2, aug_bias_list2, aug_sub_node_list2, aug_adjs, aug_n_ids, device, RL_thresholds)  # HAN_2 计算正样本 subgraph augmentation embeddings
            # readout
            c_aug_1 = nn.Sigmoid()(torch.mean(h_aug_1, 0))  # (64,)
            c_aug_2 = nn.Sigmoid()(torch.mean(h_aug_2, 0))
            # discriminator
            ret_1 = gcl_disc(c_aug_1, h_pos, h_neg)  # (100, 384)
            ret_2 = gcl_disc(c_aug_2, h_pos, h_neg)  # (100, 384)
            ret = ret_1 + ret_2
                # logits, (1,6654)
            gcl_loss = gcl_loss_fn(ret, lbl)
            """
            '''------------------两个loss加权求和---------------------------'''
            if gcl_loss is not None:
                loss = loss * loss_p + gcl_loss * (1-loss_p)
            losses.append(loss.item())
            total_loss += loss.item()

            # step13.1: metrics
            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)
            if batch % args.log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)] \tloss: {:.6f}'.format(batch * args.batch_size, train_num_samples,
                                                                           100. * batch / ((train_num_samples // args.batch_size) + 1),
                                                                           np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                # print(message)
                with open(save_path_i + '.log.txt', 'a') as f:
                    f.write(message)
                losses = []

            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
            del pred, loss_outputs
            gc.collect()

            # step13.2: backward
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time.time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)

            del loss
            gc.collect()

        # step14: print loss
        total_loss /= (batch + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch, args.n_epochs, total_loss)

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        # step15: validation--------------------------------------------------------
        print('---------------------validation-------------------------------------')
        # inder the representations of all tweets
        model.eval()

        # we recommend to forward all nodes and select the validation indices instead
        extract_features = torch.FloatTensor([])
        num_batches = int(all_num_samples / args.batch_size) + 1  # 这里的all_num_samples,是为了然后epoch n对应的model求feature,而不是用于evaluation

        # all mask are then splited into mini-batch in order
        all_mask = torch.arange(0, num_dim, dtype=torch.long)

        for batch in range(num_batches):
            start_batch = time.time()

            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, all_num_samples)
            batch_nodes = all_mask[i_start:i_end]
            batch_node_list = [batch_nodes, batch_nodes, batch_nodes]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                         batch_size=args.batch_size)

            pred = model(features_list, biases_mat_list, batch_node_list, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
            # pred = model(features_list, biases_mat_list, batch_node_list, adjs, n_ids, device, RL_thresholds)  # HAN_2 model
            # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # baseline-1: MarGNN pred

            extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

            del pred
            gc.collect()

        # evaluate the model: conduct kMeans clustering on the validation and report NMI
        validation_nmi = evaluate(extract_features[homo_data.val_mask],  # val feature embedding
                                  homo_data.y,
                                  indices=homo_data.val_mask,  # 即index, validation时，index不变
                                  epoch=epoch,
                                  num_isolated_nodes=0,
                                  save_path=save_path_i,
                                  is_validation=True,
                                  cluster_type=args.cluster_type)
        all_vali_nmi.append(validation_nmi)

        # step16: early stop
        if validation_nmi > best_vali_nmi:
            best_vali_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Best model was at epoch ', str(best_epoch))
        else:
            wait += 1
        if wait >= args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # step17: save all validation nmi
    np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
    # save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')

    "-----------------loading best model---------------"
    # step18: load the best model of the current block
    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print('Best model loaded.')

    # 合并gcl_model和model参数
    for name, param in model.named_parameters(): pass
    # del homo_data, multi_r_data
    torch.cuda.empty_cache()

    # test--------------------------------------------------------
    print('--------------------test----------------------------')
    model.eval()

    # we recommend to forward all nodes and select the validation indices instead
    extract_features = torch.FloatTensor([])
    num_batches = int(all_num_samples / args.batch_size) + 1  # 这里的all_num_samples,是为了然后epoch n对应的model求feature,而不是用于evaluation

    # all mask are then splited into mini-batch in order
    all_mask = torch.arange(0, num_dim, dtype=torch.long)

    for batch in range(num_batches):
        start_batch = time.time()

        # split batch
        i_start = args.batch_size * batch
        i_end = min((batch + 1) * args.batch_size, all_num_samples)
        batch_nodes = all_mask[i_start:i_end]
        batch_nodes_list = [batch_nodes, batch_nodes, batch_nodes]

        # sampling neighbors of batch nodes
        adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                     batch_size=args.batch_size)

        pred = model(features_list, biases_mat_list, batch_nodes_list, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
        # pred = model(features_list, biases_mat_list, batch_nodes_list, adjs, n_ids, device, RL_thresholds)  # HAN_2 model
        # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # baseline-1: MarGNN pred

        extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)
        del pred
        gc.collect()

    save_embeddings(extract_features, save_path_i)

    test_nmi = evaluate(extract_features[homo_data.test_mask],
                        homo_data.y,
                        indices=homo_data.test_mask,
                        epoch=-1,
                        num_isolated_nodes=0,
                        save_path=save_path_i,
                        is_validation=False,
                        cluster_type=args.cluster_type)


if __name__ == '__main__':
    # define args
    args = args_register()

    # check CUDA
    print('Using CUDA:', torch.cuda.is_available())

    # create working path
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    embedding_path = args.data_path + '/offline_embeddings/'
    if not os.path.exists(embedding_path):
        os.mkdir(embedding_path)
    print('embedding path: ', embedding_path)

    # record hyper-parameters
    with open(embedding_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)  # __dict__将模型参数保存成字典形式；indent缩进打印

    print('Batch Size:', args.batch_size)
    print('Intra Agg Mode:', args.is_shared)
    print('Inter Agg Mode:', args.inter_opt)
    print('Reserve node config?', args.is_initial)

    # load number of message in each blocks
    # e.g. data_split = [  500  ,   100, ...,  100]
    #                    block_0  block_1    block_n
    # define loss function，调用forward(embeddings, labels)方法，最终loss返回单个值
    # contrastive loss in our paper
    if args.use_hardest_neg:
        # HardestNegativeTripletSelector返回某标签下ith元素和jth元素，其最大loss对应的其他标签元素索引
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))  # margin used for computing tripletloss
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

    # define metrics
    BCL_metrics = [AverageNonzeroTripletsMetric()]
    # define detection stage
    # Streaming = FinEvent(args)
    # pre-train stage: train on initial graph
    train_i = 0
    offline_FinEvent_model(train_i=train_i,
                          args=args,
                          i=0,
                          metrics=BCL_metrics,
                          embedding_save_path=embedding_path,
                          loss_fn=loss_fn,
                          model=None)
    print('model finished')