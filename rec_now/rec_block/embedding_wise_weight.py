# coding=utf-8
''' 2021_09_27 lcreg163@163.com

将各个embedding的权重变为各个embedding的各个dimension上的权重.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def gather_embedding_element_wise_weight(embedding_weights, pos_idx):
    """根据每个embedding的weight，扩展为各个embedding的各个element的weight.

        对于DNN的输入（记为dnn_input），一般为batch_size * sum(embedding_dim)大小，但是，各个embedding的dim可能是不同的。
        当我们计算得到各个embedding的weight后，需要将其变换为dnn_input相同的大小，该函数即实现该功能

    Args:
        embedding_weights (tf.flota32): 大小为batch_size * num_embedding，表示各个embedding的weights
        pos_idx (List[int]): 长度为sum(embedding_dim)的list，其中元素属于[0, num_embedding-1]，表示该位置上的数属于第几个embedding

    Returns:
        (tf.float32): 大小为batch_size * sum(embedding_dim)，表示每个embedding的各个position上的weight
    """

    if isinstance(pos_idx, list):
        pos_idx = tf.constant([pos_idx], dtype=tf.int32)
    num_embedding = int(embedding_weights.shape[-1])
    batch_size = tf.shape(embedding_weights)[0]
    sample_idx = tf.range(batch_size) * num_embedding
    sample_idx = tf.reshape(sample_idx, [-1, 1])
    pos_idx_mat = sample_idx + pos_idx
    embedding_weights_reshaped = tf.reshape(embedding_weights, [-1])
    element_wise_weight = tf.gather(embedding_weights_reshaped, pos_idx_mat)
    return element_wise_weight
