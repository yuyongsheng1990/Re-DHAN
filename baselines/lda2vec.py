# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> lda2vec
# @Time: 3/8/23 00:42 
# @Author: Yu Yongsheng
# @Description: gensim 调包实现 lda2vec embedding

import numpy as np
import pandas as pd
import os
import datetime
import torch
import gensim
# --------------------------------load_tweet_data------------------------------------
project_path = os.getcwd()

# -----------------------------------------lda2vec embeddings------------------------------------------------
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

train_vec = project_path + '/baselines/proasmdatasetVec.txt.model'
def process_doc(df):
    for line, event_id in zip(list(df.filtered_words), list(df.event_id)):
        # print(line)
        yield TaggedDocument(words=line, tags=[event_id])

def train_lda(ftrain):
    # 实例化Doc2Vec模型
    model = Doc2Vec(vector_size=128, window=3, cbow_mean=1, min_count=1)
    # 更新现有的word2vec模型
    model.build_vocab(ftrain)  # 使用数据建立单词表
    model.train(ftrain, total_examples=model.corpus_count, epochs=50)  # 训练模型，更新模型参数
    model.save(train_vec)
    return model

def ldaEmbedding_fn(tran_idx, i):
    load_path = project_path + '/data/raw dataset/'
    save_path = project_path + '/data/offline_embeddings_3days/block_' + str(i)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
                df['date'] <= init_day + datetime.timedelta(days=int(i+1)))].reset_index(drop=True)  # (11971, 18)
    print(df.shape)
    print(df.event_id.nunique())
    print(df.user_id.nunique())
    # training
    train_corpus = list(process_doc(df.loc[tran_idx, :]))
    model_dm = train_lda(train_corpus)

    # 模型训练完成后，可以用来生成一段文本的paragraph vector。

    lda_embeddings = df.filtered_words.apply(lambda x: model_dm.infer_vector(x))  # nlp生成300维向量；join函数将列表连接成字符串
    lda_embeddings = np.stack(lda_embeddings, axis=0)

    print(lda_embeddings.shape)
    np.save(save_path + '/lda_embeddings.npy', lda_embeddings)
    return lda_embeddings