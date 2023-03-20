import tensorflow as tf


class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(
            tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(  # 计算多分类交叉熵
            labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    # 更新梯度权重
    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.compat.v1.trainable_variables()  # 查看可训练变量,list
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not  # add_n实现列表相加；l2_loss是l2范数值得一半
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        # lossL2, tensor(mul_5.0), shape=()
        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)  # adam函数

        # training op
        train_op = opt.minimize(loss + lossL2)  # 计算梯度，然后更新参数

        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)  # 返回行最大值索引
        return tf.confusion_matrix(labels, preds)  # 混淆矩阵

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(  # 返回交叉熵向量
            logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)  # 改变tensor数据类型
        mask /= tf.reduce_mean(mask)  # 通过均值求loss
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(  # sigmoid交叉熵
            logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(  # 判断向量元素是否相等
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))  # 四舍五入函数

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)

        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)  # 非零元素个数
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure