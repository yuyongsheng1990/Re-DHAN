import numpy as np
import tensorflow as tf

from models.base_gattn import BaseGAttN
from utils.layers import attn_head, SimpleAttLayer

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):  # 残差
        attns = []
        for _ in range(n_heads[0]):
            attns.append(attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        # multi-head attention
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits

class HeteGAT_multi(BaseGAttN):
    '''
    # forward；model = HeteGAT_multi
    logits, final_embedding, att_val = model.inference(ftr_in_list,  # list:3, tensor（1， 3025， 1870）
                                                       nb_classes,   # 3
                                                       nb_nodes,     # 3025
                                                       is_train,     # bool
                                                       attn_drop,    # tensor, ()
                                                       ffd_drop,     # tensor, ()
                                                       bias_mat_list=bias_in_list,  # list:2, tensor()
                                                       hid_units=hid_units,   # hid_units:8
                                                       n_heads=n_heads,       # n_heads: [8, 1]
                                                       residual=residual,     # residual: False
                                                       activation=nonlinearity)  # nonlinearity:tf.nn.elu

    '''
    def inference(ftr_in_list, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        for features, bias_mat in zip(ftr_in_list, bias_mat_list):
            attns = []
            jhy_embeds = []
            for _ in range(n_heads[0]):   # [8,1]
                # multi-head attention 计算
                attns.append(attn_head(features, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
            h_1 = tf.concat(attns, axis=-1)

            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(attn_head(h_1, bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop, residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))  # list:2. 其中每个元素tensor, (3025, 1, 64)

        multi_embed = tf.concat(embed_list, axis=1)   # tensor, (3025, 2, 64)
        # attention输出：tensor(3025, 64)、softmax概率
        final_embed, att_val = SimpleAttLayer(multi_embed,
                                              mp_att_size,
                                              time_major=False,
                                              return_alphas=True)

        out = []
        for i in range(n_heads[-1]):  # 1
            # 用于添加一个全连接层(input, output) -> (3025, 3)
            out.append(tf.compat.v1.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]  # add_n是列表相加。tensor,(3025, 3)
        # logits_list.append(logits)
        print('de')

        logits = tf.expand_dims(logits, axis=0)  # (1, 3025, 3)
        # attention通过全连接层预测(1, 3025, 3)、attention final_embedding tensor(3025, 64)、attention 概率
        return logits, final_embed, att_val

class HeteGAT_no_coef(BaseGAttN):
    def inference(ftr_in_list, nb_classes, nb_nodes, is_train, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        # coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):

                attns.append(attn_head(ftr_in_list, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.compat.v1.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        print('de')
        logits = tf.expand_dims(logits, axis=0)
        # if return_coef:
        #     return logits, final_embed, att_val, coef_list
        # else:
        return logits, final_embed, att_val

class HeteGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128,
                  return_coef=False):
        embed_list = []
        coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):
                if return_coef:
                    a1, a2 = attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                              return_coef=return_coef)
                    attns.append(a1)
                    head_coef_list.append(a2)
                    # attns.append(attn_head(inputs, bias_mat=bias_mat,
                    #                               out_sz=hid_units[0], activation=activation,
                    #                               in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                    #                               return_coef=return_coef)[0])
                    #
                    # head_coef_list.append(attn_head(inputs, bias_mat=bias_mat,
                    #                                        out_sz=hid_units[0], activation=activation,
                    #                                        in_drop=ffd_drop, coef_drop=attn_drop,
                    #                                        residual=False,
                    #                                        return_coef=return_coef)[1])
                else:
                    attns.append(attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            head_coef = tf.concat(head_coef_list, axis=0)
            head_coef = tf.reduce_mean(head_coef, axis=0)
            coef_list.append(head_coef)
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.compat.v1.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        logits = tf.expand_dims(logits, axis=0)
        if return_coef:
            return logits, final_embed, att_val, coef_list
        else:
            return logits, final_embed, att_val