# coding=utf-8
''' 2021_10_15 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class FMLayer(keras.layers.Layer):
    """Factorization Machines (FM) 的二阶交叉层.

    Reference:
        [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

    Symbols:
        B: batch size
        D: 输入embedding的维度
        F: field的数量
    """

    def call(self, inputs):
        """计算inputs的FM.

        Args:
            inputs (List[tf.Tensor]): 输入矩阵list，长度为F，其中每个tensor的形状为(B, D)

        Returns:
            (tf.Tensor): FM的结果，形状为(B, 1)
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        sumed_fm_emb = tf.reduce_sum(inputs, axis=0)  # (B, D)
        sumed_fm_emb_square = tf.square(sumed_fm_emb)  # (B, D)
        squared_fm_emb = tf.square(inputs)  # [(B, D)] * F
        squared_fm_emb_sum = tf.reduce_sum(squared_fm_emb, axis=0)  # (B, D)
        second_order = tf.subtract(sumed_fm_emb_square, squared_fm_emb_sum)  # (B, D)
        second_order_sum = 0.5 * tf.reduce_sum(second_order, axis=1, keepdims=True)  # (B, 1)
        return second_order_sum
