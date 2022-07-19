# coding=utf-8
''' 2021_06_04 lcreg163@163.com

从一个mini-batch内抽取listwise的样本.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def row_not_all_zero(x):
    """矩阵的一行中是否有非零元素.

    Symbols:
        M: 矩阵的行数
        N: 矩阵的列数

    Args:
        x (tf.Tensor): 矩阵，形状为(M, N)

    Returns:
        （tf.Tensor): x的每行中是否有非零元素，形状为(M,)
    """
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = tf.not_equal(x, 0.0)
    x = tf.cast(x, tf.int32)
    r = tf.reduce_sum(x, axis=-1, keepdims=False) > 0
    return r


def row_has_value_greater_than(x, threshold):
    """矩阵的一行中是否含有大于阈值的值.

    Symbols:
        M: 矩阵的行数
        N: 矩阵的列数

    Args:
        x (tf.Tensor): 矩阵，形状为(M, N)
        threshold (float): 阈值

    Returns:
        （tf.Tensor): x的每行中是否有大于threshold的值，形状为(M,)
    """
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = x > threshold
    x = tf.cast(x, tf.int32)
    r = tf.reduce_sum(x, axis=-1, keepdims=False) > 0
    return r


def row_has_value_less_than(x, threshold):
    """矩阵的一行中是否含有小于阈值的值.

    Args:
        x (tf.Tensor): 矩阵，形状为M*N
        threshold (float): 阈值

    Returns:
        （tf.Tensor): x的每行中是否有小于threshold的值，形状为(M,)
    """
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    x = x < threshold
    x = tf.cast(x, tf.int32)
    r = tf.reduce_sum(x, axis=-1, keepdims=False) > 0
    return r


def nan_to_zero(val):
    """将NaN转换为0.

    Args:
        val (tf.Tensor): 标量tensor，形状为()

    Returns:
        (tf.Tensor): 标量tensor，如果val不为NaN，则为val；否则为0.0
    """
    rank = len(val.shape)
    if rank != 0:
        raise ValueError('input muust be a scalar tf.Tensor')
    return tf.cond(tf.math.is_nan(val), lambda: tf.constant(0.0, dtype=val.dtype), lambda: val)


def to_listwise_sample(group_ids, labels, logits, do_mask_logits=True, value_of_masked_logit=-1E9, pos_neg_th=0.5):
    """ 从一个batch中抽取listwise_sample.

        对于同一个group_id, 只有既有正样本，又有负样本的才会认为是有效的，否则会被mask掉

    Args:
        group_ids (tf.Tensor): 样本分组ID，比如用户ID，形状为(batch_size,)
        labels (tf.Tensor): 样本标签，最好是正样本>0, 负样本为0，形状为(batch_size,)
        logits (tf.Tensor): 模型输出(before sigmoid)，形状为(batch_size,)
        do_mask_logits (tf.Tensor): 由于需要对齐样本长度，输出矩阵中会有一些无效值，如果do_mask_logits=True，则会将无效值设为value_of_masked_logit
        value_of_masked_logit (float): 被mask掉的值的logit
        pos_neg_th (float): 区分正负样本的阈值

    Returns:
        dense_mask (tf.Tensor): 有效样本的mask， 形状为 (num_valid_group, batch_size)
        dense_labels (tf.Tensor): 样本标签，按行归一化为有效的分布， 形状为 (num_valid_group, batch_size)
        dense_logits (tf.Tensor): 样本的logtis， 形状为 (num_valid_group, batch_size)
    """
    if len(group_ids.shape) > 1:
        group_ids = tf.reshape(group_ids, [-1])
    y, idx, _ = tf.unique_with_counts(group_ids)

    idx = tf.reshape(idx, [-1, 1])

    n = tf.size(idx)

    idx2 = tf.range(n, dtype=idx.dtype)
    idx2 = tf.reshape(idx2, [-1, 1])

    sp_idx = tf.concat([idx, idx2], axis=-1)
    sp_idx = tf.cast(sp_idx, dtype=tf.int64)
    sp_shape = [tf.size(y), n]
    sp_values_mask = tf.cast(tf.reshape(idx2, [-1]), dtype=tf.float32) + 1 > 0

    def gen_dense(sp_values):
        if len(sp_values.shape) > 1:
            sp_values = tf.reshape(sp_values, [-1])
        sp = tf.SparseTensor(indices=sp_idx, values=sp_values, dense_shape=sp_shape)
        sp = tf.sparse.reorder(sp)
        d = tf.sparse.to_dense(sp)
        return d

    dense_mask = gen_dense(sp_values_mask)
    dense_labels = gen_dense(labels)
    dense_logits = gen_dense(logits)

    has_pos_sample = row_has_value_greater_than(dense_labels, pos_neg_th)
    has_neg_sample = row_has_value_less_than(gen_dense(labels - pos_neg_th), 0.0)
    row_mask = tf.logical_and(has_pos_sample, has_neg_sample)

    if do_mask_logits:
        dense_logits = dense_logits + (1.0 - tf.cast(dense_mask, tf.float32)) * value_of_masked_logit

    dense_mask = tf.boolean_mask(dense_mask, row_mask)
    dense_labels = tf.boolean_mask(dense_labels, row_mask)
    dense_labels = dense_labels / tf.reduce_sum(dense_labels, axis=-1, keepdims=True)
    dense_logits = tf.boolean_mask(dense_logits, row_mask)

    dense_labels = tf.stop_gradient(dense_labels)
    return dense_mask, dense_labels, dense_logits


def listwise_loss_via_softmax_cross_entropy_with_logits(labels_for_softmax,
                                                        logits_for_softmax,
                                                        weights=None,
                                                        do_reduce=True):
    """基于softmax cross entropy的listwise loss.

    Args:
        labels_for_softmax (tf.Tensor): 样本标签，每行之和为1
        logits_for_softmax (tf.Tensor): 模型输出的logits，形状和labels_for_softmax相同
        weights (tf.Tensor): 样本权重
        do_reduce (bool): 是否对多个样本的loss进行求均值操作

    Returns:
        (tf.Tensor): listwise loss
    """
    labels_for_softmax = tf.stop_gradient(labels_for_softmax)
    listwise_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_for_softmax, logits=logits_for_softmax)
    if weights is not None:
        listwise_loss = listwise_loss * weights
    if do_reduce:
        listwise_loss = tf.reduce_mean(listwise_loss)
        listwise_loss = nan_to_zero(listwise_loss)
    return listwise_loss
