# coding=utf-8
''' 2021_11_29 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class PoolingLayer(keras.layers.Layer):
    """ 对输入进行聚合的层.

    支持的聚合方法有:支持的方法有None, 'mean', 'sum', 'max', 'min', callable object.

    Example:
        inputs = [[1, 2, 3], [10, 11, 12]]
        PoolingLayer(axis=0, keepdims=True, combiner='sum')(inputs) = [[11, 13, 15]]
        PoolingLayer(axis=1, keepdims=False, combiner='sum')(inputs) = [6, 33]
    """
    combiner_to_func = {
        'mean': tf.reduce_mean,
        'sum': tf.reduce_sum,
        'max': tf.reduce_max,
        'min': tf.reduce_min
    }

    def __init__(self, axis=None, keepdims=False, combiner=None, name=None, **kwargs):
        """
        Args:
            axis (int, optional): 对输入的哪一维度进行聚合. Defaults to None.
            keepdims (bool, optional): 是否保留聚合后的维度. Defaults to False.
            combiner (str or callable, optional): 聚合方法. Defaults to None.
                    支持的方法有None, 'mean', 'sum', 'max', 'min', callable object
            name (str, optional): 该层的名称. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims
        self.combiner = combiner

    def call(self, inputs):
        """对输入进行聚合.

        Args:
            inputs (tf.tensor): 任意张量

        Raises:
            ValueError: 不支持的聚合函数

        Returns:
            (tf.tensor): inputs的聚合结果
        """
        combiner = self.combiner
        axis = self.axis
        keepdims = self.keepdims
        if combiner is None:
            return inputs
        if callable(combiner):
            return combiner(inputs)

        if combiner in PoolingLayer.combiner_to_func:
            pooling_func = PoolingLayer.combiner_to_func[combiner]
            return pooling_func(inputs, axis=axis, keepdims=keepdims)

        raise ValueError("combiner must be one of None, "
                         "'mean', 'sum', 'max', 'min' or a callable object")
