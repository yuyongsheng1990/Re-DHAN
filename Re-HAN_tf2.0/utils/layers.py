import numpy as np
import pandas as pd

import os
project_path = os.getcwd()

import numpy as np
import tensorflow as tf
import tf_slim


def attn_head(features, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    """[summary]
    multi-head attention计算
    [description]
    # forward；model = HeteGAT_multi
    attns.append(layers.attn_head(features,            # list:3, tensor（1， 3025， 1870）
                                bias_mat=bias_mat,     # list:2, tensor(1, 3025, 3025)
                                out_sz=hid_units[0],   # hid_units:[8]，卷积核的个数
                                activation=activation, # nonlinearity:tf.nn.elu
                                in_drop=ffd_drop,      # tensor, ()
                                coef_drop=attn_drop,   # tensor, ()
                                residual=False))
    Arguments:
        features {[type]} -- shape=(batch_size, nb_nodes, fea_size))
    """
    with tf.name_scope('my_attn'):  # 定义一个上下文管理器
        if in_drop != 0.0:
            features = tf.nn.dropout(features, 1.0 - in_drop)  # 以rate置0
        features_fts = tf.compat.v1.layers.conv1d(features, out_sz, 1, use_bias=False)  # 一维卷积操作, out: (1, 3025, 8)

        f_1 = tf.compat.v1.layers.conv1d(features_fts, 1, 1)  # (1, 3025, 1)
        f_2 = tf.compat.v1.layers.conv1d(features_fts, 1, 1)  # (1, 3025, 1)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # 转置         # (1, 3025, 3025)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)  # (1, 3025, 3025)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            features_fts = tf.nn.dropout(features_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, features_fts)  # (1, 3025, 8)
        ret = tf_slim.bias_add(vals)  # 将bias向量加到value矩阵上      # (1. 3025， 8)

        # residual connection 残差连接
        if residual:
            if features.shape[-1] != ret.shape[-1]:
                ret = ret + tf.keras.layers.Conv1D(features, ret.shape[-1], 1)  # activation
            else:
                features_fts = ret + features
        if return_coef:
            return activation(ret), coefs
        else:
            return activation(ret)  # activation


def attn_head_const_1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    """[summary]
    [description]
    """
    adj_mat = 1.0 - bias_mat / -1e9
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.compat.v1.layers.conv1d(seq, out_sz, 1, use_bias=False)

        logits = adj_mat
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf_slim.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.compat.v1.layers.conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.compat.v1.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1)
        logits = tf.sparse_add(adj_mat * f_1, adj_mat *
                               tf.transpose(f_2, [0, 2, 1]))
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)  # 将softmax应用于批量的N维SparseTensor

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(
                                        coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)  # SparseTensor稀疏矩阵乘法
        vals = tf.expand_dims(vals, axis=0)  # 在0处扩展维度1
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf_slim.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.compat.v1.layers.conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
#                                                      time_major=False,
#                                                      return_alphas=True)
def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    '''
    inputs: tensor, (3025, 2, 64)
    attention_size: 128
    '''
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)  # 表示在shape第2个维度上拼接

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])  #

    hidden_size = inputs.shape[2]  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.compat.v1.random_normal([hidden_size, attention_size], stddev=0.1))  # (64, 128)
    b_omega = tf.Variable(tf.compat.v1.random_normal([attention_size], stddev=0.1))               # (128, )
    u_omega = tf.Variable(tf.compat.v1.random_normal([attention_size], stddev=0.1))               # (128, )

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)   # (3025, 2, 128)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape   tensor, (3025, 2)
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape   tensor, (3025, 2)

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)  # (3025, 2, 64) * (3025, 2, 1) = (3025, 2, 64) -> (3025, 2)

    if not return_alphas:
        return output
    else:
        return output, alphas  # attention输出、softmax概率