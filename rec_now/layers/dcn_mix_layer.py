# coding=utf-8
''' 2021_11_11 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class DCNMixLayer(keras.layers.Layer):
    """ DCN mix layer.

    Reference:
        [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems]
         (https://arxiv.org/abs/2008.13535v2)

    Symbols:
        B: batch size
        D: 输入维度
        S: 子空间维度
        N: 每层的expert数量
        L: 层数
    """

    def __init__(self, dim_sub_space, num_layer=1, num_expert=2,
                 activation_inner='tanh', activation_outer='tanh',
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        Args:
            dim_sub_space (int): 子空间维度
            num_layer (int, optional): 层数
            num_expert (int, optional): 每层的expert数
            activation_inner (str, optional): 子空间的内层激活函数
            activation_outer (str, optional): 子空间的外层激活函数
            kernel_initializer (str, optional): 映射矩阵的初始化函数
            bias_initializer (str, optional): bias的初始化函数
            其他参数意义同keras.layers.Layer
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.dim_sub_space = dim_sub_space
        self.num_layer = num_layer
        self.num_expert = num_expert
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.activation_inner = keras.activations.get(activation_inner)
        self.activation_outer = keras.activations.get(activation_outer)

    def _build_dnn_params(self, dim_in):
        """生成原空间和子空间之间相互变换所需的参数.
        """
        # 原空间到子空间的映射參數，L层，每层N个expert，每组参数形状为(D, S)，N个expert的参数形状为(N, D, S)
        self.origin_to_sub_kernels = [
            self.add_weight(
                'origin_to_sub_kernels_of_layer%s' % layer_idx,
                shape=[self.num_expert, dim_in, self.dim_sub_space],
                initializer=self.kernel_initializer,
                dtype=self.dtype,
                trainable=True)
            for layer_idx in range(self.num_layer)
        ]

        # 子空间到子空间的映射參數，L层，每层N个expert，每组参数形状为(S, S)，N个expert的参数形状为(N, S, S)
        self.sub_to_sub_kernels = [
            self.add_weight(
                'sub_to_sub_kernels_of_layer%s' % layer_idx,
                shape=[self.num_expert, self.dim_sub_space, self.dim_sub_space],
                initializer=self.kernel_initializer,
                dtype=self.dtype,
                trainable=True)
            for layer_idx in range(self.num_layer)
        ]

        # 子空间到原空间的映射參數，L层，每层N个expert，每组参数形状为(S, D)，N个expert的参数形状为(N, S, D)
        self.sub_to_origin_kernels = [
            self.add_weight(
                'sub_to_origin_kernels_of_layer%s' % layer_idx,
                shape=[self.num_expert, self.dim_sub_space, dim_in],
                initializer=self.kernel_initializer,
                dtype=self.dtype,
                trainable=True)
            for layer_idx in range(self.num_layer)
        ]

        # 子空间到原空间的映射的bias，L层，每层N个expert，每个bias为D维，N个expert的参数形状为(1, N, D)
        self.biases = [
            self.add_weight(
                'bias_of_layer%s' % layer_idx,
                shape=[1, self.num_expert, dim_in],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)
            for layer_idx in range(self.num_layer)
        ]

    def _build_gates(self):
        # 门控网络，L层，每层N个expert，故输出为N维
        self.gate_layers = [
            tf.keras.layers.Dense(self.num_expert, use_bias=False, name='gate_of_layer%s' % layer_idx)
            for layer_idx in range(self.num_layer)
        ]

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        dim_in = int(input_shape[-1])  # input_shape: B*D
        self._build_dnn_params(dim_in)
        self._build_gates()

        return super().build(input_shape)

    def call(self, inputs):
        """对inputs使用DCN-Mix.

        Args:
            inputs (tf.Tensor): 输入，形状为(B, D)

        Returns:
            tf.Tensor: 输出，形状和输入相同，为(B, D)
        """
        extended_inputs = tf.expand_dims(inputs, axis=1)  # (B, 1, D)

        layer_input = inputs  # (B, D)
        for layer_idx in range(self.num_layer):
            # layer parameters
            origin_to_sub_kernels = self.origin_to_sub_kernels[layer_idx]  # (N, D, S)
            sub_to_sub_kernels = self.sub_to_sub_kernels[layer_idx]  # (N, S, S)
            sub_to_origin_kernels = self.sub_to_origin_kernels[layer_idx]  # (N, S, D)
            biases = self.biases[layer_idx]  # (1, N, D)
            gate_layer = self.gate_layers[layer_idx]  # dense layer, output dim = N

            # sub space operations
            sub_space = tf.tensordot(layer_input, origin_to_sub_kernels, axes=[[1], [1]])  # (B, N, S)
            sub_space = self.activation_inner(sub_space)
            sub_space = tf.einsum('bns,nst->bnt', sub_space, sub_to_sub_kernels)  # (B, N, S)
            sub_space = self.activation_outer(sub_space)

            # origin space operations
            origin_space = tf.einsum('bns,nsd->bnd', sub_space, sub_to_origin_kernels)  # (B, N, D)
            origin_space = origin_space + biases  # (B, N, D)
            origin_space = extended_inputs * origin_space  # (B, N, D)

            # gate operations
            gates = gate_layer(layer_input)  # (B, N)
            gates = tf.nn.softmax(gates, axis=-1)

            layer_output = tf.einsum('bnd,bn->bd', origin_space, gates)  # (B, D)
            layer_input = layer_output
        return layer_output
