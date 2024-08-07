# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:12
# @Author : yysgz
# @File : run_offline model.py
# @Project : FinEvent Models
# @Description :
import random
import sys

import numpy as np
import scipy.sparse as sp
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import time
from typing import List
import os

project_path = os.getcwd()

from layers.S2_TripletLoss import OnlineTripletLoss, HardestNegativeTripletSelector, RandomNegativeTripletSelector
from layers.NCELoss import NCECriterion
from layers.TripletLossWithGlobal import TripletLossGlobal
from layers.S3_NeighborRL import cal_similarity_node_edge, RL_neighbor_filter
from layers.S4_Global_localGCL import GlobalLocalGraphContrastiveLoss

# from baselines.MarGNN import MarGNN
# from baselines.PPGCN import PPGCN
from baselines.lda2vec import ldaEmbedding_fn
from baselines.bert_model import bertEmbedding_fn
# from baselines.KPGNN import KPGNN

from utils.S2_gen_dataset import create_offline_homodataset, create_multi_relational_graph, MySampler, save_embeddings
from utils.S4_Evaluation import AverageNonzeroTripletsMetric, evaluate

# from models.HeteGAT_multi import HeteGAT_multi
from models.HeteGAT_multi_RL2 import HeteGAT_multi_RL2

from models.MLP_model import MLP_model

from GraphCL import aug, discriminator, discriminator2
def args_register():
    parser = argparse.ArgumentParser()  # 创建参数对象
    # 添加参数
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of initial-training/maintenance-training epochs.')
    parser.add_argument('--window_size', default=3, type=int, help='Maintain the model after predicting window_size blocks.')
    parser.add_argument('--patience', default=5, type=int,
                        help='Early stop if perfermance did not improve in the last patience epochs.')
    parser.add_argument('--margin', default=8, type=float, help='Margin for computing triplet losses')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size (number of nodes sampled to compute triplet loss in each batch)')
    parser.add_argument('--hid_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--out_dim', default=256, type=int, help='Output dimension of tweet representation')
    parser.add_argument('--heads', default=8, type=int, help='Number of heads used in GAT')
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
    parser.add_argument('--time_lambda', default=0.2, type=float, help='The hyperparameter of time exponential decay')  # DHAN 时间衰减参数 lambda
    parser.add_argument('--gl_eps', default=2, type=float, help='the temperature param for global-local GCL function')

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
    parser.add_argument('--data_path', default=project_path + '/data', type=str, help='graph data path')  # 相对路径，.表示当前所在目录
    # parser.add_argument('--result_path', default='./result/offline result', type=str,
    #                     help='Path of features, labels and edges')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None, type=str,
                        help='File path that contains the training, validation and test masks')
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval')
    # subgraph contrastive loss和triplet loss加权概率
    parser.add_argument('--para_t', default=0.8, type=float, help='A percent value of triplet loss used for loss optimization')
    parser.add_argument('--para_s', default=0.2, type=float, help='A percent value of GraphCL loss used for loss optimization')
    parser.add_argument('--para_g', default=0.2, type=float, help='A percent value of global-local GCL loss used for loss optimization')
    args = parser.parse_args(args=[])  # 解析参数

    return args


# # tensor version: 将二维矩阵list 转换成adj matrix list
def relations_to_adj(r_data, nb_nodes=None, device=None):
    data = torch.ones(r_data.shape[1]).to(device)
    relation_mx = torch.sparse_coo_tensor(indices=r_data, values=data, size=[nb_nodes, nb_nodes],
                                          dtype=torch.int32)
    return relation_mx.to_dense()

def relations_to_Weightadj(relation_path):
    relations = ['entity', 'userid', 'word']
    weight_adj_list = []
    for relation in relations:
        sp_relation_mx = sp.load_npz(os.path.join(relation_path, 's_m_tid_%s_tid.npz' % relation))
        indices = torch.from_numpy(
            np.vstack((sp_relation_mx.row, sp_relation_mx.col)).astype(np.int64))
        values = torch.from_numpy(sp_relation_mx.data)
        shape = torch.Size(sp_relation_mx.shape)
        weight_adj_mx = torch.sparse.FloatTensor(indices, values, shape)
        weight_adj_list.append(weight_adj_mx.to_dense())
    return weight_adj_list

# tensor version: 计算偏差矩阵
def adj_to_bias(adj, nhood=1, device=None):  # adj,(3025, 3025); sizes, [3025]
    mt = torch.eye(adj.shape[0]).to(device)
    for _ in range(nhood):
        adj = torch.add(adj, torch.eye(adj.shape[1]).to(device))
        mt = torch.matmul(mt, adj)  # 相乘
    mt = torch.where(mt > 0, 1, mt)
    return (-1e9 * (1.0 - mt))  # 科学计数法，2.5 x 10^(-27)表示为：2.5e-27

def offline_FinEvent_model(train_i,  # train_i=0
                           i,  # i=0
                           model_name,
                           args,
                           metrics,
                           embedding_save_path,
                           loss_fn,
                           model=None,
                           loss_fn_dgi=None):
    # step1: make dir for graph i
    # ./incremental_0808//embeddings_0403005348/block_xxx
    save_path_i = embedding_save_path + '/block_' + str(i) + '/' + model_name  # + '/'
    if not os.path.exists(save_path_i):
        os.makedirs(save_path_i)
    data_name = 'Twitter'
    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # step2: load data
    relation_ids: List[str] = ['entity', 'userid', 'word']  # twitter dataset
    if data_name == 'MAVEN':
        relation_ids: List[str] = ['entity', 'word']  # MAVEN dataset
    homo_data = create_offline_homodataset(embedding_save_path + '/block_' + str(i), [train_i, i])  # (4762, 302), 包含x: feature embedding和y: label, generate train_slices (3334), val_slices (952), test_slices (476)
    # 返回entity, userid, word的homogeneous adj mx中non-zero neighbor idx。二维矩阵，node -> non-zero neighbor idx, (2,487962), (2,8050), (2, 51498)
    # features_list = [homo_data.x.to(device) for _ in range(3)]  # list:3, (4762, 302)
    multi_r_data = create_multi_relational_graph(embedding_save_path + '/block_' + str(i), relation_ids, [train_i, i])
    num_relations = len(multi_r_data)  # 3
    # load data to device
    multi_r_data = [r_data.to(device) for r_data in multi_r_data]

    # input dimension (300 in our paper)
    num_dim = homo_data.x.size(0)  # 4762
    feat_dim = homo_data.x.size(1)  # embedding dimension, 302
    nb_classes = len(np.unique(homo_data.y))
    attn_drop = args.attn_drop
    feat_drop = args.feat_drop
    print('message numbers: ', num_dim)
    print('edge_numbers: ', multi_r_data[0].shape, multi_r_data[1].shape, multi_r_data[2].shape)
    print('event classes: ', nb_classes)

    # prepare graph configs for node filtering
    if args.is_initial:
        print('prepare node configures...')  # 计算neighbor node与node的相似度，并排序sorted
        cal_similarity_node_edge(multi_r_data, homo_data.x, save_path_i)
        filter_path = save_path_i
    else:
        filter_path = args.data_path + str(i)

    # Multi-Agent
    # initialize RL thresholds, RL_threshold: [[.5],[.5],[.5]]
    # if (train_i != 0) and (model_name == 'ReDHAN' or 'FinEvent'):
    #     best_rl_path = embedding_save_path + '/block_' + str(i-1) + '/' + model_name + '/models/best_rl_thres.pt'
    #     RL_thresholds = torch.load(best_rl_path).to(device)
    # else:
    RL_thresholds = torch.FloatTensor(args.threshold_start0).to(device)  # [[0.2], [0.2], [0.2]]
    if data_name == 'MAVEN':
        RL_thresholds = torch.FloatTensor([[0.2], [0.2]]).to(device)  # MAVEN RL_thresholds, [[0.2], [0.2], [0.2]]

    # sampling size
    if model_name == 'ReDHAN':
        sample_size = [-1]
    elif model_name == 'FinEvent':
        sample_size = [-1, -1]

    # RL_filter means extract limited sorted neighbors based on RL_threshold and neighbor similarity, return filtered node -> neighbor index
    if args.sampler == 'RL_sampler':
        filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds,
                                                   filter_path, model_name, device)  # filtered 二维矩阵, (2,104479); (2,6401); (2,15072)
    else:
        filtered_multi_r_data = multi_r_data  # if multi_r_data 爆显存，就取constant neighbor filter

    # load data to device
    filtered_multi_r_data = [filtered_r_data.to(device) for filtered_r_data in filtered_multi_r_data]

    print('filtered_edge_numbers: ', filtered_multi_r_data[0].shape, filtered_multi_r_data[1].shape, filtered_multi_r_data[2].shape)
    message = 'message numbers: ' + str(num_dim) + '\n'
    message += 'edge numbers: ' + str(multi_r_data[0].shape) + str(multi_r_data[1].shape) + str(multi_r_data[2].shape) + '\n'
    message += 'filtered message numbers: ' + str(filtered_multi_r_data[0].shape) + str(filtered_multi_r_data[1].shape) + str(filtered_multi_r_data[2].shape) + '\n'
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    if model is None:  # pre-training stage in our paper
        # print('Pre-Train Stage...')
        # # HAN model without RL_filter and Neighbor_sampler
        # adj_mx_list = [relations_to_adj(r_data, torch.tensor(num_dim).to(device), device) for r_data in multi_r_data]
        # biases_mat_list = [adj_to_bias(adj, torch.tensor(1).to(device), device) for adj in adj_mx_list]  # 偏差矩阵list:3,tensor, (4762,4762)
        # model = HeteGAT_multi(feature_size=feat_dim, nb_classes=nb_classes, nb_nodes=num_dim, attn_drop=attn_drop,
        #                       feat_drop=feat_drop, hid_dim=args.hid_dim, out_dim=args.out_dim,  # 时间衰减参数，默认: -0.2
        #                       num_relations=num_relations, hid_units=[8], n_heads=[8,1], activation=nn.ELU())

        # DHAN model with RL_filter and Neighbor_sampler，这要torch_geometric重写HAN模型，要不然用不上FinEvent中的neighbor_sampler.
        # 所以，既要用到adjs_list for RL_sampler，也要用到bias_list for HAN algorithm.
        adj_mx_list = [relations_to_adj(filtered_r_data, torch.tensor(num_dim).to(device), device) for filtered_r_data in filtered_multi_r_data]  # 邻接矩阵list:3,tensor, (4762,4762)
        biases_mat_list = [adj_to_bias(adj, torch.tensor(1).to(device), device) for adj in adj_mx_list]  # 偏差矩阵list:3,tensor, (4762,4762)
        model = HeteGAT_multi_RL2(feature_size=feat_dim, nb_classes=nb_classes, nb_nodes=num_dim, attn_drop=attn_drop,
                                        feat_drop=feat_drop, hid_dim=args.hid_dim, out_dim=args.out_dim, time_lambda=args.time_lambda,  # 时间衰减参数，默认: -0.2
                                        num_relations=num_relations, hid_units=[8], n_heads=[8,1])

        # baseline 1: feat_dim=302; hidden_dim=128; out_dim=64; heads=4; inter_opt=cat_w_avg; is_shared=False
        # model = MarGNN((feat_dim, args.hid_dim, args.out_dim, args.heads),
        #                num_relations=num_relations, inter_opt=args.inter_opt, is_shared=args.is_shared)

        # baseline 2: MLP, multi-layer perceptron
        # model = MLP_model(input_dim=feat_dim, hid_dim=args.hid_dim, out_dim=args.out_dim)

        # baseline 3: PPGCN-2, 双层GCN
        # model = PPGCN(feat_dim, args.hid_dim, args.out_dim)

        # # baseline 4: KPGNN。方案1：multi_r_data + FinEvent + constant_sampler; 方案2：multi_r_data + KPGNN + constant_sampler.
        # model = KPGNN(feat_dim, args.hid_dim, args.out_dim, args.heads, num_relations=num_relations)

        # baseline 5: GraphFormer.
        # model = HeteGAT_multi_RL4(feat_dim, args.hid_dim, args.out_dim, args.heads, num_relations=num_relations)

    else:
        biases_mat_list = [relations_to_adj(r_data, torch.tensor(num_dim).to(device), device) for r_data in multi_r_data]

    del multi_r_data
    # define sampler
    if model_name == "KPGNN":
        sampler = MySampler('const_sampler')  # dgl.NeighborSampler固定采样
    else:
        sampler = MySampler(args.sampler)  # RL_sampler
    sampler_random = MySampler('random_sampler')
    # load model to device
    model.to(device)

    # define NCE Loss
    # loss_fn = NCECriterion(args.nce_m, args.nce_eps)
    # loss_fn = loss_fn.to(device)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # record training log
    if i==0:
        message = '\n------------------Start initial training ------------------------\n'
    else:
        message = '\n---------------maintaining using block' + str(i) + '------\n'
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)

    start_running_time = time.time()

    # step12.0: record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    gcl_loss_fn = nn.BCEWithLogitsLoss()
    # gl_loss_fn = GlobalLocalGraphContrastiveLoss(args.gl_eps)
    gcl_dropout_percent = 0.1
    gcl_disc = discriminator.Discriminator(args.out_dim)
    gcl_disc.to(device)
    gcl_disc2 = discriminator2.Discriminator2(args.out_dim)
    gcl_disc2.to(device)
    para_t = args.para_t  # triplet loss加权概率
    para_s = args.para_s  # GraphCL loss 加权概率
    para_g = args.para_g  # global-local GCL loss 加权概率

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
    gcl_loss = None
    gl_loss = None  # edge perturbations

    if model_name == 'BERT':
        print('-------------------BERT----------------------------')
        # baseline 3: BERT embedding
        bert_embeddings = bertEmbedding_fn(embedding_save_path, i)
        # # bert_embeddings = np.load(embedding_save_path +'/block_' + str(i) + '/bert_embeddings.npy')
        bert_embeddings = torch.FloatTensor(bert_embeddings)
        print(bert_embeddings.shape)
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
            batch_bert = bert_embeddings[batch_nodes, :]  # Bert embedding
            extract_features = torch.cat((extract_features, batch_bert.cpu().detach()), dim=0)
            del batch_bert
            gc.collect()

        save_embeddings(extract_features, save_path_i, file_name='final_embeddings.pt')
        save_embeddings(homo_data.y, save_path_i, file_name='labels.pt')

        test_nmi = evaluate(extract_features[homo_data.test_mask],
                            homo_data.y,
                            indices=homo_data.test_mask,
                            epoch=-2,
                            num_isolated_nodes=0,
                            save_path=save_path_i,
                            is_validation=False,
                            cluster_type=args.cluster_type)

        mins_spent = (time.time() - start_running_time) / 60
        message += '\nWhole Running Time took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        return test_nmi, model.state_dict()
    elif model_name == 'LDA':
        # baseline 4: LDA embedding
        lda_embeddings = ldaEmbedding_fn(homo_data.train_mask, i)
        lda_embeddings = torch.FloatTensor(lda_embeddings)
        print(lda_embeddings.shape)
        print('-------------------LDA----------------------------')
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
            batch_bert = lda_embeddings[batch_nodes, :]  # Bert embedding
            extract_features = torch.cat((extract_features, batch_bert.cpu().detach()), dim=0)
            del batch_bert
            gc.collect()

        save_embeddings(extract_features, save_path_i, file_name='final_embeddings.pt')
        save_embeddings(homo_data.y, save_path_i, file_name='labels.pt')

        test_nmi = evaluate(extract_features[homo_data.test_mask],
                            homo_data.y,
                            indices=homo_data.test_mask,
                            epoch=-2,
                            num_isolated_nodes=0,
                            save_path=save_path_i,
                            is_validation=False,
                            cluster_type=args.cluster_type)

        mins_spent = (time.time() - start_running_time) / 60
        message += '\nWhole Running Time took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        return test_nmi, model.state_dict()

    if train_i != 0:
        model.eval()
        message = '\n------------incremental detection' + str(i) + '------\n'
        print(message)
        "-----------------loading best model---------------"
        # step18: load the best model of the current block
        best_model_path = embedding_save_path + '/block_' + str(i-1) + '/' + model_name + '/models/best.pt'
        checckpoint = torch.load(best_model_path)
        # if model_name == "HAN":
        #     checckpoint['model'].pop('fc.bias')
        #     checckpoint['model'].pop('fc.weight')
        model.load_state_dict(checckpoint['model'], strict=False)
        print('Last Stream best model loaded.')

        # we recommend to forward all nodes and select the validation indices instead
        extract_features = torch.FloatTensor([])
        num_batches = int(
            all_num_samples / args.batch_size) + 1  # 这里的all_num_samples,是为了然后epoch n对应的model求feature,而不是用于evaluation

        # all mask are then splited into mini-batch in order
        all_mask = torch.arange(0, num_dim, dtype=torch.long)

        for batch in range(num_batches):
            start_batch = time.time()

            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, all_num_samples)
            batch_nodes = all_mask[i_start:i_end]
            # batch_node_list = [batch_nodes for _ in range(3)]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size,  # DHAN-4: -1选取所有的邻居
                                         batch_size=args.batch_size)

            # pred = model(homo_data.x, biases_mat_list, batch_nodes, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
            pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # DHAN model
            # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # baseline-1: MarGNN pred
            # pred = model(homo_data.x[batch_nodes], device)  # MLP baseline, (100, 302) -> (100, 128)
            # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PPGCN-1 baseline model
            # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # KPGNN model
            # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model

            extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)
            del pred
            gc.collect()

        save_embeddings(extract_features, save_path_i, file_name='incre_embeddings.pt')

        former_save_path = embedding_save_path + '/block_' + str(i - 1) + '/' + model_name
        test_nmi = evaluate(extract_features[homo_data.test_mask],
                            homo_data.y,
                            indices=homo_data.test_mask,
                            epoch=-2,
                            num_isolated_nodes=0,
                            save_path=save_path_i,
                            former_save_path=former_save_path,
                            is_validation=False,
                            cluster_type='dbscan')

        mins_spent = (time.time() - start_running_time) / 60
        message += '\nIncremental Detection Time took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

    # step13: start training------------------------------------------------------------
    print('----------------------------------training----------------------------')
    for epoch in range(args.n_epochs):
        start_epoch_time = time.time()
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
            batch_features = homo_data.x[batch_nodes]
            batch_labels = homo_data.y[batch_nodes].to(device)

            # sampling neighbors of batch nodes
            # adjs是RL_sampler采样的子图edge; n_ids是采样过程中遇到的node list。都是list: 3, 对应entity, userid, word
            # sizes=[-1,-1]，list length=2 表明了这是一个两层的卷积；-1表示选取所有的邻居, 2表示在第i层采样2条边
            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size, # DHAN-4: -1选取所有的邻居
                                         batch_size=args.batch_size)
            optimizer.zero_grad()  # 将参数置0

            # batch_node_list = [batch_nodes.to(device) for _ in range(3)]

            # pred = model(homo_data.x, biases_mat_list, batch_nodes, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
            pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # HAN_2 model
            # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # deep HAN_2 model
            # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # Fin-Event pred: x表示combined feature embedding, 302; pred, 其实是个embedding (100,192)
            # pred = model(homo_data.x[batch_nodes], device)  # MLP baseline, (100, 302) -> (100, 128)
            # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PPGCN-2 baseline model
            # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # KPGNN model
            # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model

            loss_outputs = loss_fn(pred, batch_labels)  # (12.8063), 179
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs  # GCN loss, 这不是梯度爆炸吗
            print('triplet_loss: ', loss)

            '''----------new GraphCL v1.0 with disc_cosine: comparison of batch graph with subgraph augmentation ------------------------'''
            # Random sample 80% nodes from batch_features
            random_idx_1 = torch.LongTensor(
                random.sample(range(batch_nodes.shape[0]), int(batch_nodes.shape[0] * 0.8)))  # 0.8; 0.9
            random_idx_2 = torch.LongTensor(
                random.sample(range(batch_nodes.shape[0]), int(batch_nodes.shape[0] * 0.9)))  # 0.8; 0.9
            sub_nodes_1 = torch.index_select(batch_nodes, 0, random_idx_1).to(device)  # 采样维度 dim=0
            sub_nodes_2 = torch.index_select(batch_nodes, 0, random_idx_2).to(device)  # 采样维度 dim=0
            # subgraph feature embeddings
            # 归一化
            gcl_normalized_r_data_list = [aug.normalize_adj(torch.add(adj, torch.eye(adj.shape[0]).to(device))) for adj in biases_mat_list]  # 原始adj matrix做归一化normalize, ndarray, (3327,3327)
            # negative samples
            features_neg = homo_data.x.clone()
            features_neg = features_neg[torch.randperm(features_neg.shape[0])]
            # 构建标签label. Bilinear的值域为[0,1] 或[-1, 1], 值域变化受输入数据影响
            lbl_1 = torch.ones(batch_nodes.shape[0], 1)  # labels for aug_1, (1,192)
            lbl_2 = torch.zeros(batch_nodes.shape[0], 1)  # (1,192)
            lbl = torch.cat((lbl_1, lbl_2), dim=0)  # (1,128)
            # 基于data augmentation生成关于original features和shuffled features的embedding
            h_pos = pred.clone()
            h_neg = model(features_neg, gcl_normalized_r_data_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # HAN_2构建负样本 negative feature embeddings
            # h_neg = model(features_neg, adjs, n_ids, device, RL_thresholds)   # FinEvent
            # 构建subgraph augmentation embedding
            aug_adjs_1, aug_n_ids_1 = sampler.sample(filtered_multi_r_data,
                                                     node_idx=sub_nodes_1, sizes=sample_size,
                                                     batch_size=len(sub_nodes_1))  # RL_sampler from aug_adj for HAN_2
            h_aug_1 = model(homo_data.x, biases_mat_list, sub_nodes_1, aug_adjs_1, aug_n_ids_1, device, RL_thresholds)  # HAN_2 构建 subgraph augmentation embeddings, (90, 64)
            # h_aug_1 = model(homo_data.x, aug_adjs_1, aug_n_ids_1, device, RL_thresholds)  # FinEvent 构建 subgraph augmentation embeddings, (90, 64)
            aug_adjs_2, aug_n_ids_2 = sampler.sample(filtered_multi_r_data,
                                                     node_idx=sub_nodes_2, sizes=sample_size,
                                                     batch_size=len(sub_nodes_2))  # RL_sampler from aug_adj for HAN_2
            h_aug_2 = model(homo_data.x, biases_mat_list, sub_nodes_2, aug_adjs_2, aug_n_ids_2, device, RL_thresholds)  # HAN_2 计算正样本 subgraph augmentation embeddings
            # h_aug_2 = model(homo_data.x, aug_adjs_2, aug_n_ids_2, device, RL_thresholds)  # FinEvent 计算正样本 subgraph augmentation embeddings

            # discriminator. Bilinear双向线性映射，将subgraph embedding 与pos embedding对齐；将sub embedding 2与neg embedding对齐。pos对齐，相似度为1，neg为0.
            # logits, (1,6654)
            # readout
            c_aug_1 = torch.sigmoid(torch.mean(h_aug_1, 0))  # (64,)
            c_aug_2 = torch.sigmoid(torch.mean(h_aug_2, 0))
            ret_1 = gcl_disc(c_aug_1, h_pos, h_neg, device)  # 鉴别器，本质上是一个预估的插值，做平滑smooth用，它可以对输入图像的微小变化具有一定的鲁棒性
            ret_2 = gcl_disc(c_aug_2, h_pos, h_neg, device)  # (100, 384) # BiLinear
            ret = ret_1 + ret_2
            gcl_loss = gcl_loss_fn(ret.cpu(), lbl)  # ret, (1,128); lbl, (1,128)
            print('gcl_loss: ', gcl_loss)
            message = '\n triplet_loss: {:.2f} '.format(loss)
            message += '\n gcl_loss: {:.2f} '.format(gcl_loss)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)

            '''------------------三个loss加权求和---------------------------'''
            if gcl_loss is not None:
                loss = loss + gcl_loss  # 0.823; 0.732; 0.657

            if gl_loss is not None:
                loss = para_t * loss + para_g * gl_loss
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
                with open(save_path_i + '/log.txt', 'a') as f:
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

            # del loss
            gc.collect()

        # step14: print loss
        total_loss /= (batch + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch, args.n_epochs, total_loss)

        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time.time() - start_epoch_time) / 60
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
            # batch_node_list = [batch_nodes.to(device) for _ in range(3)]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size,  # -1选取所有的邻居
                                         batch_size=args.batch_size)

            # pred = model(homo_data.x, biases_mat_list, batch_nodes, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
            pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # DHAN model
            # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # baseline-1: MarGNN pred
            # pred = model(homo_data.x[batch_nodes], device)  # MLP baseline, (100, 302) -> (100, 128)
            # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PPGCN-2 baseline model
            # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # KPGNN model
            # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model

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
                                  former_save_path=None,
                                  is_validation=True,
                                  cluster_type=args.cluster_type)
        all_vali_nmi.append(validation_nmi)

        message = 'Epoch: {}/{}. Validation_nmi : {:.4f}'.format(epoch, args.n_epochs, validation_nmi)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)

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
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, p)
            print('Best model was at epoch ', str(best_epoch))
            best_rl_path = model_path + '/best_rl_thres.pt'
            torch.save(RL_thresholds, best_rl_path)
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
    checkpoint = torch.load(best_model_path)
    # # start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    # # optimizer.load_state_dict(checkpoint['optimizer'])
    print('Best model loaded.')

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
        # batch_node_list = [batch_nodes.to(device) for _ in range(3)]

        # sampling neighbors of batch nodes
        adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=sample_size,  # DHAN-4: -1选取所有的邻居
                                         batch_size=args.batch_size)

        # pred = model(homo_data.x, biases_mat_list, batch_nodes, device, RL_thresholds)  # HAN_0/HAN_1 pred: (100, 192)
        pred = model(homo_data.x, biases_mat_list, batch_nodes, adjs, n_ids, device, RL_thresholds)  # DHAN model
        # pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)  # baseline-1: MarGNN pred
        # pred = model(homo_data.x[batch_nodes], device)  # MLP baseline, (100, 302) -> (100, 128)
        # pred = model(homo_data.x, filtered_multi_r_data, batch_nodes, device)  # PPGCN上-2 baseline model
        # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # KPGNN model
        # pred = model(homo_data.x, batch_nodes, adjs, n_ids, device)  # GraphFormer model

        extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)
        del pred
        gc.collect()

    save_embeddings(extract_features, save_path_i, file_name='final_embeddings.pt')
    save_embeddings(homo_data.y, save_path_i, file_name='final_labels.pt')

    test_nmi = evaluate(extract_features[homo_data.test_mask],
                        homo_data.y,
                        indices=homo_data.test_mask,
                        epoch=-1,
                        num_isolated_nodes=0,
                        save_path=save_path_i,
                        former_save_path=None,
                        is_validation=False,
                        cluster_type=args.cluster_type)

    mins_spent = (time.time() - start_running_time) / 60
    message += '\nWhole Running Time took {:.2f} mins'.format(mins_spent)
    message += '\n'
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    return test_nmi, model.state_dict()

if __name__ == '__main__':
    # define args
    args = args_register()

    # check CUDA
    print('Using CUDA:', torch.cuda.is_available())

    # create working path
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    embedding_path = args.data_path + '/offline_embeddings'
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
        # loss_fn = TripletLossGlobal(args.margin, HardestNegativeTripletSelector(args.margin))  # margin used for computing tripletloss with global embedding -> negative
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    # N_pair_loss
    # loss_fn = FunctionNPairLoss(args.margin)
    # define metrics
    BCL_metrics = [AverageNonzeroTripletsMetric()]
    # define detection stage
    # Streaming = FinEvent(args)
    # pre-train stage: train on initial graph

    model_name = 'ReDHAN'  # 更换detection模型
    train_i, i = 0, 0  # set which incremental detection stream is
    #  重复实验，选取最佳model参数
    best_nmi = 0
    model_path = embedding_path + '/block_' + str(i) + '/' + model_name + '/models'
    for iteration in range(5):
        test_nmi, model_dict = offline_FinEvent_model(train_i=train_i,
                                                      i=i,
                                                      model_name=model_name,
                                                      args=args,
                                                       metrics=BCL_metrics,
                                                      embedding_save_path=embedding_path,
                                                      loss_fn=loss_fn,
                                                      model=None)
        if (model_name == 'BERT') or (model_name == 'LDA'):
            pass
        else:
            print('test_nmi: ', test_nmi)
            print('best_nmi: ', best_nmi)
            if best_nmi < test_nmi:
                print('best_model_dict chang')
                best_nmi = test_nmi
                p = model_path + '/best.pt'
                torch.save({
                            'model':model_dict,
                            }, p)
    print('model finished')