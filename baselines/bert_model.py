# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> KPGNN
# @Time: 4/8/23 18:47
# @Author: Yu Yongsheng
# @Description: baseline models: Bert、GCN
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import time
# --------------------------------laod_tweet_data------------------------------------
print('--------------------------------load_tweet_data------------------------------------')
import os
# project_path = os.getcwd() + '/Re-HAN_Model'
project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # # 获取上上级路径

load_path = project_path + '/data/raw dataset/'
save_path = project_path + '/data/offline_embeddings/block_0'

start_time = time.time()

import datetime

def bertEmbedding_fn(embedding_save_path, i):

    save_path = embedding_save_path +'/block_' + str(i)

    # load data (68841 tweets, multiclasses filtered)
    p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
    p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
    # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组，Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化。
    np_part1 = np.load(p_part1, allow_pickle=True)  # (35000, 16)
    np_part2 = np.load(p_part2, allow_pickle=True)  # (33841, 16)

    np_tweets = np.concatenate((np_part1, np_part2), axis=0)  # (68841, 16)
    print('Data loaded.')

    df = pd.DataFrame(data=np_tweets, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', 'user_loc',
                                                'place_type', 'place_full_name', 'place_country_code', 'hashtags',
                                                'user_mentions', 'image_urls', 'entities', 'words', 'filtered_words',
                                                'sampled_words'])
    print('Data converted to dataframe.')
    # sort date by time
    df = df.sort_values(by='created_at').reset_index(drop=True)

    # append date
    df['date'] = [d.date() for d in df['created_at']]
    # 因为graph太大，爆了内存，所以取4天的twitter data做demo，后面用nci server
    init_day = df.loc[0, 'date']
    df = df[(df['date'] >= init_day + datetime.timedelta(days=i)) & (
                df['date'] <= init_day + datetime.timedelta(days=int(i+1)))].reset_index()  # (11971, 18)
    print(df.shape)
    print(df.event_id.nunique())
    print(df.user_id.nunique())
    # df = df.loc[:5]
    mins_spent = (time.time() - start_time) / 60
    print('loading data took {:.2f} mins'.format(mins_spent))

    # 离线状态下可运行
    TRANSFORMERS_OFFLINE=1
    # -----------------------------------------Bert model------------------------------------------------
    print('----------------bert model------------------------------')
    bert_base_repo = project_path + '/bert_base_repo'
    tokennizer = AutoTokenizer.from_pretrained(bert_base_repo, local_files_only=True)  # bert-base-uncased
    bert_model = AutoModel.from_pretrained(bert_base_repo, local_files_only=True)  # bert-base-uncased
    mins_spent = (time.time() - start_time) / 60
    print('loading bert model took {:.2f} mins'.format(mins_spent))

    # -------------------------------------bert_embeddings-------------------------------------------
    print('-------------------------bert embeddings-----------------------------------')
    bert_embedding_list = []
    for i in range(df.filtered_words.shape[0]):
        feat = df.loc[i, 'filtered_words']
        bert_embeddings = bert_model(**tokennizer(' '.join(feat), return_tensors='pt'))[0][0][0][:128]  # nlp生成300维向量；join函数将列表连接成字符串
        bert_embedding_list.append(bert_embeddings.detach().numpy())
    bert_embeddings = np.stack(bert_embedding_list, axis=0)

    print(bert_embeddings.shape)
    np.save(save_path + '/bert_embeddings.npy', bert_embeddings)
    mins_spent = (time.time() - start_time) / 60
    print('converting bert embeddings took {:.2f} mins'.format(mins_spent))

    return bert_embeddings

# import datetime
# # -----------------------------------------Bert embeddings------------------------------------------------
# TRANSFORMERS_OFFLINE=1
# bert_base_repo = project_path + '/bert_base_repo'
# def documents_to_bert_features(x):  # x, list type
#     tokennizer = AutoTokenizer.from_pretrained(bert_base_repo, local_files_only=True)
#     bert_model = AutoModel.from_pretrained(bert_base_repo, local_files_only=True)
#     bert_vector = bert_model(**tokennizer(' '.join(x), return_tensors='pt'))[0][0][0][:128]
#     return bert_vector.detach().numpy()
#
# # load data (68841 tweets, multiclasses filtered)
# p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
# p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
# # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组，Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化。
# np_part1 = np.load(p_part1, allow_pickle=True)  # (35000, 16)
# np_part2 = np.load(p_part2, allow_pickle=True)  # (33841, 16)
#
# np_tweets = np.concatenate((np_part1, np_part2), axis=0)  # (68841, 16)
# print('Data loaded.')
#
# df = pd.DataFrame(data=np_tweets, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', 'user_loc',
#                                            'place_type', 'place_full_name', 'place_country_code', 'hashtags',
#                                            'user_mentions', 'image_urls', 'entities', 'words', 'filtered_words',
#                                            'sampled_words'])
# print('Data converted to dataframe.')
# # sort date by time
# df = df.sort_values(by='created_at').reset_index(drop=True)
#
# # append date
# df['date'] = [d.date() for d in df['created_at']]
# # 因为graph太大，爆了内存，所以取4天的twitter data做demo，后面用nci server
# init_day = df.loc[0, 'date']
# df = df[(df['date'] >= init_day + datetime.timedelta(days=0)) & (
#             df['date'] <= init_day + datetime.timedelta(days=2))].reset_index()  # (11971, 18)
# print(df.shape)
# print(df.event_id.nunique())
# print(df.user_id.nunique())
# # df = df.loc[:5]
# bert_embeddings = df.filtered_words.apply(lambda x: documents_to_bert_features(x))  # nlp生成300维向量；join函数将列表连接成字符串
# bert_embeddings = np.stack(bert_embeddings, axis=0)
#
# print(bert_embeddings.shape)
# np.save(save_path + '/bert_embeddings.npy', bert_embeddings)

'''
    # ------------------------------------data_split---------------------------------------------------------
    from sklearn.model_selection import train_test_split
    
    tran_x, test_x, tran_y, test_y = train_test_split(x, y, test_size=0.2, random_state=i)
    # -------------------------------------DBSCAN--------------------------------------------------------
    from sklearn.cluster import DBSCAN
    
    dbscan_model = DBSCAN()
    dbscan_model.fit(tran_x, tran_y)
    pred_y = dbscan_model.fit_predict(test_x)
    # --------------------------------------Evaluation----------------------------------------------------
    # NMI, AMI, ARI
    from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score
    
    bert_nmi = normalized_mutual_info_score(test_y, pred_y)
    bert_ami = adjusted_mutual_info_score(test_y, pred_y)
    bert_ari = adjusted_rand_score(test_y, pred_y)
'''

