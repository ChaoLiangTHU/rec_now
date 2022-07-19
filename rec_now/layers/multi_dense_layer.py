# coding=utf-8
''' 2021_11_29 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import tensor_shape


class MultiDenseLayer(keras.layers.Dense):
    """ 输入输出维度相同的多个Dense层.

    比如，有N个参数形状完全相同的experts，输出维度均为U.
    输入一个形状为(B, D) 或 (N, B, D)的矩阵，则返回一个形状为(N, B, U)的矩阵.
    相当于同时计算了N个keras.layers.Dense层，但只使用了一次tf.matmul，速度快.

    Symbols:
        B: batch size
        D: 输入的维度
        N: DNN的数量
        U: 输出的维度
    """

    def __init__(self, units, num_dnn, **kwargs):
        """
        Args:
            units (int): 单个DNN的输出维度
            num_dnn (int): DNN的数量
        """
        super().__init__(units, **kwargs)
        self.num_dnn = int(num_dnn)

    def _build_kernel(self, last_dim):
        """生成参数矩阵.
        """
        self.kernel = self.add_weight(  # (N, D, U)
            'kernel',
            shape=[self.num_dnn, last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

    def _build_bias(self):
        """生成bias参数.
        """
        if self.use_bias:
            self.bias = self.add_weight(  # (N, 1, U)
                'bias',
                shape=[self.num_dnn, 1, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        dtype = tf.dtypes.as_dtype(self.dtype or keras.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `MultiDenseLayer` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self._build_kernel(last_dim)
        self._build_bias()
        self.built = True

    def call(self, inputs):
        """计算该dense层的输出.

        Args:
            inputs (tf.Tensor): 输入，形状为(B, D) 或 (N, B, D)
        Returns:
            (tf.Tensor): dense层的输出，形状为(N, B, D)
        """
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=0)  # (1, B, D)
        outputs = tf.matmul(inputs, self.kernel)  # (N, B, U)
        if self.use_bias:
            outputs = outputs + self.bias  # (N, B, U)
        outputs = self.activation(outputs)  # (N, B, U)
        return outputs
