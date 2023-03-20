import time
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'  # 设置使用GPU1,2,3

from scipy import sparse
import torch

from models import HeteGAT_multi
from utils.data_process import adj_to_bias
from clustering.KNN_model import my_KNN
from clustering.kmeans_model import my_Kmeans

config = tf.compat.v1.ConfigProto()  # 用来对session进行参数配置
config.gpu_options.allow_growth = True  # 允许tf自动选择一个存在并且可用的设备来运行操作。

dataset = 'acm'
featype = 'fea'
checkpt_file = os.path.abspath(os.path.join(os.getcwd(), "../..")) +\
                            '/result/Re_HAN_result/offline_result.ckpt'
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 100
patience = 50
lr = 0.05  # learning rate
l2_coef = 0.005  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

import os
project_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # # 获取上级路径

load_data_path = project_path + '/result/FinEvent result/offline dataset'
load_embeddings_path = project_path + '/result/FinEvent result/offline result/offline_embeddings'


def load_offline_data(load_data_path, load_embeddings_path):
    #     data = sio.loadmat(path)  # load .mat file
    '''
    全是ndarray
    PTP: (3025, 3025)，全是1   <---meta-paths得到的homo 邻接矩阵。
    PLP: (3025, 3025)，有0有1，有些向量时相同的
    PAP: (3025, 3025)，对角线全是1，其他元素基本是0，很少是1.
    feature: (3025, 1870)，由0、1组成。
    label: (3025, 3),就3列，1061、965、999
    train_idx: (1, 600)，0-2225随机抽取的索引
    val_idx: (1, 300)，200-2325之间随机抽取的索引
    test_idx: (1, 2125)，300-3024之间随机抽取的索引
    '''

    labels = np.load(load_data_path + '/sorted_labels.npy', allow_pickle=True)
    # 将单行y转换成label矩阵，一列代表one label
    labels_dict = {}
    count = 0
    for i in range(len(labels)):
    #     print(i)
        element = labels[i]
        if element in labels_dict:
            labels[i] = labels_dict[element]
        else:
            labels_dict[element] = count
            labels[i] = count
            count += 1
    truelabels = np.zeros((labels.shape[0], max(labels) + 1))
    for i, label in enumerate(labels):
        # print(i, label)
        truelabels[i][label] = 1

    truefeatures = np.load(load_data_path + '/sorted_combined_features_embeddings.npy', allow_pickle=True)

    N = truefeatures.shape[0]
    adj_entity = sparse.load_npz(load_data_path + '/s_m_tid_entity_tid_matrix.npz').todense()
    adj_entity = np.asarray(adj_entity) - np.eye(N)
    adj_userid = sparse.load_npz(load_data_path + '/s_m_tid_userid_tid_matrix.npz').todense()
    adj_userid = np.asarray(adj_userid) - np.eye(N)
    rownetworks = [adj_entity, adj_userid]  # , data['PTP'] - np.eye(N)]
    '''
    rownetworks: list: 2。第1个元素，ndarray,(3025, 3025)；第2个元素，ndarray，(3025, 3025)
    '''
    y = truelabels  # shape为(3025, 3)
    train_idx = torch.load(load_embeddings_path + '/block_0/train_mask.pt')  # (1, 600)
    train_idx = np.asarray(torch.unsqueeze(train_idx, 0))
    print(train_idx.shape)
    val_idx = torch.load(load_embeddings_path + '/block_0/valid_mask.pt')  # (1, 300)
    val_idx = np.asarray(torch.unsqueeze(val_idx, 0))
    test_idx = torch.load(load_embeddings_path + '/block_0/test_mask.pt')  # (1, 2125)
    test_idx = np.asarray(torch.unsqueeze(test_idx, 0))

    train_mask = sample_mask(train_idx, y.shape[0])  # 3025长度的bool list，train_idx位置为True
    val_mask = sample_mask(val_idx, y.shape[0])  # 3025长度的boolean list
    test_mask = sample_mask(test_idx, y.shape[0])

    # 提取train、val、test的标签
    # 所以，为什么不直接用train_test_split呢？
    y_train = np.zeros(y.shape)  # shape为(3025, 3)的zero 列表
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]  # 取出train_idx为true的label，放入y_train，y_train其余位置为0
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures]  # truefeatures: (3025, 1870)
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask

# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_offline_data(load_data_path, load_embeddings_path)
if featype == 'adj':
    fea_list = adj_list

import scipy.sparse as sp

# truefeatures: (3025, 1870)
nb_nodes = fea_list[0].shape[0]  # 3025
ft_size = fea_list[0].shape[1]   # 1870
nb_classes = y_train.shape[1]    # 3

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]  # np.newaxis行增加一个新的维度
'''
fea_list:list:3。第一个元素，ndarray, (1, 3025, 1870)
'''
adj_list = [adj[np.newaxis] for adj in adj_list]  # adj_list: 2. 单个元素(1, 3025, 3025)
y_train = y_train[np.newaxis]  # ndarray: (1, 3025, 3)
y_val = y_val[np.newaxis]      # ndarray: (1, 3025, 3)
y_test = y_test[np.newaxis]    # ndarray: (1, 3025, 3)
train_mask = train_mask[np.newaxis]  # ndarray(1, 3025)
val_mask = val_mask[np.newaxis]      # ndarray(1, 3025)
test_mask = test_mask[np.newaxis]    # ndarray(1, 3025)

biases_list = [adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
with tf.Graph().as_default():  # 创建一个新的计算图
    with tf.name_scope('input'):  # 创建一个上下文管理器
        ftr_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,  # 占位符，提前分配必要的内存
                                                shape=(batch_size, nb_nodes, ft_size),
                                                # batch_size:1, nb_nodes:3025, fea_size: 1870
                                                name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]  # fea_list,长度为3，内部单个元素，(1, 3025, 1870)
        bias_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,
                                                 shape=(batch_size, nb_nodes, nb_nodes),
                                                 name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]  # 邻接矩阵转换成的biases_list: 2. 单个元素占位符tensor, (1, 3025, 3025)
        lbl_in = tf.compat.v1.placeholder(dtype=tf.int32,
                                          shape=(batch_size, nb_nodes, nb_classes),  # tensor, nb_classes: 3
                                          name='lbl_in')
        msk_in = tf.compat.v1.placeholder(dtype=tf.int32,
                                          shape=(batch_size, nb_nodes),  # tensor, (1, 3025)
                                          name='msk_in')
        attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name='attn_drop')  # tensor, ()
        ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')  # # tensor, ()
        is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=(), name='is_train')  # # tensor, ()
    # forward；model = HeteGAT_multi
    logits, final_embedding, att_val = model.inference(ftr_in_list,  # list:3, tensor（1， 3025， 1870）
                                                       nb_classes,  # 3
                                                       nb_nodes,  # 3025
                                                       is_train,  # bool
                                                       attn_drop,  # tensor, ()
                                                       ffd_drop,  # tensor, ()
                                                       bias_mat_list=bias_in_list,  # list:2, tensor(1, 3025, 3025)
                                                       hid_units=hid_units,  # hid_units:[8]
                                                       n_heads=n_heads,  # n_heads: [8, 1]
                                                       residual=residual,  # residual: False
                                                       activation=nonlinearity)  # nonlinearity:tf.nn.elu

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])  # （3025， 3）
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])  # （3025， 3）
    msk_resh = tf.reshape(msk_in, [-1])  # mask，（3025， ）
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh,
                                              msk_resh)  # 占位符计算softmax cross_entropy based on (pred, y)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)  # 计算accuracy
    # optimzie
    train_op = model.training(loss, lr, l2_coef)  # lr = 0.005、l2_coef = 0.001

    saver = tf.compat.v1.train.Saver()  # 用于保存模型

    init_op = tf.group(tf.compat.v1.global_variables_initializer(),  # 全局变量初始化；group组合多个operation
                       tf.compat.v1.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.compat.v1.Session(config=config) as sess:  # 创建session
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):  # 200
            tr_step = 0

            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:
                # feature,占位符内存已经分配完毕，fea_list是真实数据，输入进行训练模型
                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]  # dict:3. 每个元素tensor, (1, 3025, 1870)
                       for i, d in zip(ftr_in_list, fea_list)}
                # bias
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]  # dict: 2. 每个元素tensor, (1, 3025, 3025)
                       for i, d in zip(bias_in_list, biases_list)}
                # other params
                fd3 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.6,
                       ffd_drop: 0.6}
                fd = fd1
                fd.update(fd2)  # 字典update方法
                fd.update(fd3)  # 获得字典形式的所有数据、参数
                # training操作：更新权重；计算loss；计算accuracy；attention概率
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:
                # fd1 = {ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size]}
                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                 feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1
            # import pdb; pdb.set_trace()
            print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (train_loss_avg / tr_step, train_acc_avg / tr_step,
                   val_loss_avg / vl_step, val_acc_avg / vl_step))

            # =============   judging  =================
            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        # loading model params
        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        # ============= testing =================
        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],

                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        print('start knn, kmean.....')
        xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]

        from numpy import linalg as LA

        # xx = xx / LA.norm(xx, axis=1)
        yy = y_test[test_mask]

        print('xx: {}, yy: {}'.format(xx.shape, yy.shape))

        # my_KNN(xx, yy)
        my_Kmeans(xx, yy)

        sess.close()