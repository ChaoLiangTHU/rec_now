# coding=utf-8
''' 2021_11_29 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class DCNLayer(keras.layers.Dense):
    """Deep & Cross Network (DCN) 中的cross network.

    Reference:
        [Deep & Cross Network for Ad Click Predictions]
        (https://arxiv.org/abs/1708.05123)

    Symbols:
        B: batch size
        D: 输入维度
    """

    def __init__(self, degree_of_cross, **kwargs):
        """
        Args:
            degree_of_cross (int): 交叉的阶数
            其他参数复用自父类keras.layers.Dense
        """
        super().__init__(0, **kwargs)
        self.degree_of_cross = degree_of_cross

    def _build_kernels(self):
        """分配kernels变量.
        """
        self.kernels = []
        for layer_idx in range(self.degree_of_cross):
            kernel = self.add_weight(
                f'kernel_{layer_idx}',
                shape=[self.input_dim, 1],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)
            self.kernels.append(kernel)

    def _build_biases(self):
        """分配biases变量.
        """
        self.biases = []
        for layer_idx in range(self.degree_of_cross):
            bias = self.add_weight(
                f'bias_{layer_idx}',
                shape=[1, self.input_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            self.biases.append(bias)

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        self.input_dim = int(input_shape[-1])

        # 分配kernels
        self._build_kernels()

        # 分配biases
        if self.use_bias:
            self._build_biases()
        else:
            self.biases = None

        self.built = True

    def call(self, inputs):
        """计算MMoE.

        Args:
            inputs (tf.Tensor): 该层的输入，形状为 (B, D)
            merge_output (bool, optional): 是否将所有task的结果合并到一个张量中. Defaults to True.

        Returns:
            MMoE的结果.
                如果merge_output = True: 返回一个形状为(T, B, dim_out)的张量
                如果merge_output = False: 返回num_task个形状为(B, dim_out)的张量
        """
        layer_input = inputs  # (B, D)
        for layer_idx in range(self.degree_of_cross):
            kernel = self.kernels[layer_idx]  # (D, 1)
            cross = tf.matmul(layer_input, kernel)  # (B, 1)
            layer_output = inputs * cross  # (B, D)
            if self.use_bias:
                bias = self.biases[layer_idx]  # (1, D)
                layer_output = layer_output + bias  # (B, D)
            layer_output = self.activation(layer_output)
            layer_input = layer_output  # (B, D)

        outputs = layer_output  # (B, D)
        return outputs
