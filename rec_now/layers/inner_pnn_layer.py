# coding=utf-8
''' 2021_10_15 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class InnerPNNLayer(keras.layers.Layer):
    """Inner Product-based Neural Networks (IPNN) 层.

    Reference:
        [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)

    Symbols:
        B: batch size
        D: 输入embedding的维度
        F: field的数量
        P: F个field两两交叉的组数, 及C(F, 2)，其中C表示组合数
    """

    def call(self, inputs):
        """计算inputs的IPNN.

        Args:
            inputs (List[tf.Tensor]): 长度为F的list，其中各个tesnor的形状为(B, D)

        Returns:
            (tf.Tensor): IPNN的结果，形状为(B, P)
        """
        dim = int(inputs[0].shape[-1])
        num_field = len(inputs)
        embedding = tf.concat(inputs, axis=1)  # (B, F*D)
        embedding = tf.reshape(embedding, [-1, num_field, dim])  # (B, F, D)
        trans_emb = tf.transpose(embedding, [1, 0, 2])  # (F, B, D)

        row = []
        col = []
        for r in range(num_field - 1):
            for c in range(r + 1, num_field):
                row.append(r)
                col.append(c)

        gathered_row = tf.gather(trans_emb, row)  # (P, ?, D)
        gathered_col = tf.gather(trans_emb, col)  # (P, ?, D)
        products = gathered_row * gathered_col

        outputs = tf.reduce_sum(products, -1, keepdims=False)  # (P, B)
        outputs = tf.transpose(outputs, [1, 0])  # (B, P)
        return outputs
