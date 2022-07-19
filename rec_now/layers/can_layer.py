# coding=utf-8
''' 2021_11_29 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math

import tensorflow as tf
from tensorflow import keras

from rec_now.layers.pooling_layer import PoolingLayer


class CANLayer(keras.layers.Layer):
    """co-action network.

    对于输入inputs，使用另一个输入(dnn_params)作为DNN的参数，对inputs进行变换.

    Reference:
        [CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction]
        (https://arxiv.org/abs/2011.05625)

    Symbols:
        B: batch size, 在代码中，用-1表示不确定的batch size
        L: 每个样本含有多少个输入(如一个样本中含L个点击历史的embedding)，约定全零embedding为padding
        D0: 输入embedding的维度
        D1, D2, D3, ..., Dn: DNN各层的输出维度
        size_dnn_param: DNN的总参数量
    """
    CAN_EXPANDED_INPUT_DIM = 4  # CAN层的输入的维度

    def __init__(self, dnn_dims=None, activation=tf.tanh,
                 use_bias=True, use_res_net=False,
                 output_layer_use_activation=False,
                 output_combiner='sum',
                 mask_all_zero_embedding=True, **kwargs):
        """
        Args:
            dnn_dims (List[int], optional): DNN的维度，如果为None，则自动根据输入参数量确定. Defaults to None.
            activation (str or callable, optional): 激活函数. Defaults to tf.tanh.
            use_bias (bool, optional): 是否使用bias. Defaults to True.
            use_res_net (bool, optional): 是否使用residual network，如果为True，需要dnn_dims中各个元素均和输入维度相同. Defaults to False.
            output_layer_use_activation (bool, optional): 输出层是否使用activation函数. Defaults to False.
            output_combiner (str, optional): 输出的化简方式，支持None, 'mean', 'sum', 'max', 'min'等. Defaults to 'sum'.
            mask_all_zero_embedding (bool, optional): 是否对全零向量进行mask. Defaults to True.
        """
        super().__init__(**kwargs)
        self.dnn_dims = dnn_dims
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.use_res_net = use_res_net
        self.output_layer_use_activation = output_layer_use_activation
        self.output_combiner = output_combiner
        self.mask_all_zero_embedding = mask_all_zero_embedding

    @classmethod
    def _get_layer_param_size(cls, dim_in, dim_out, use_bias):
        """计算一层DNN的参数数量.

        Args:
            dim_in (int): 网络输入的维度
            dim_out (int): 网络输出的维度
            use_bias (bool): 是否使用bias

        Returns:
            (int): 该层参数的数量
        """
        param_size = dim_in * dim_out
        if use_bias:
            param_size += dim_out

        return param_size

    @classmethod
    def get_dnn_param_size(cls, input_dim, dnn_dims, use_bias=True):
        """计算一个DNN的所有参数的数量.

        Args:
            input_dim (int): 网络输入的维度
            dnn_dims (List[int]): 网络各层的输出维度
            use_bias (bool, optional): 是否使用bias. Defaults to True.

        Returns:
            (int): 该DNN参数的数量
        """
        extended_dims = [input_dim]
        extended_dims.extend(dnn_dims)

        total_param_size = 0
        for layer_idx in range(1, len(extended_dims)):
            dim_in = extended_dims[layer_idx - 1]
            dim_out = extended_dims[layer_idx]
            size_param = cls._get_layer_param_size(dim_in, dim_out, use_bias)
            total_param_size += size_param

        return total_param_size

    @classmethod
    def _has_non_zero(cls, tensor, axis=-1, keepdims=True):
        """tensor的第axis维是否有非零元素.
        """
        not_zero = tf.not_equal(tensor, tf.constant(0.0, dtype=tensor.dtype))
        has_non_zero = tf.reduce_any(not_zero, axis=axis, keepdims=keepdims)
        return has_non_zero

    def _auto_decide_dnn_param_size(self, input_dim, total_param_size):
        """计算DNN的层数.

        假设DNN每层的维度都和输入维度相同的情况下，计算DNN的层数，并给出每层的输出维度.

        Args:
            input_dim (int): 网络输入的维度
            total_param_size (int): 网络的参数量

        Raises:
            ValueError: 网络输入维度和网络参数量无法匹配

        Returns:
            (List[int]): 网络各层的输出维度
        """
        one_layer_param_size = self._get_layer_param_size(input_dim, input_dim, self.use_bias)
        n_layer = float(total_param_size) / one_layer_param_size
        if math.floor(n_layer) != n_layer:
            raise ValueError(
                f'dnn_param_size not match! input_dim: {input_dim}, total_param_size: {total_param_size}, '
                f'use_bias:{self.use_bias}, one_layer_param_size(auto decide): {one_layer_param_size}')

        n_layer = math.floor(n_layer)
        dnn_dims = [input_dim] * n_layer
        return dnn_dims

    def _check_dnn_param_size(self, input_dim, dnn_dims, size_dnn_param):
        """检测DNN的参数数量是否正确.
        """
        _size_dnn_param = self.get_dnn_param_size(input_dim, dnn_dims, self.use_bias)
        if _size_dnn_param != size_dnn_param:
            raise ValueError(
                f'dnn_param_size not match! input_dim: {input_dim}, '
                f'expected total_param_size: {size_dnn_param},\n'
                f'use_bias:{self.use_bias}, dnn_dims: {str(dnn_dims)}, '
                f'calculated total_param_size: {_size_dnn_param}')

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        self.output_combiner_func = PoolingLayer(axis=1, keepdims=False,
                                                 combiner=self.output_combiner)
        return super().build(input_shape)

    def _apply_kernel(self, dim_in, dim_out, param_idx_start, dnn_params, layer_input):
        """计算矩阵相乘.
        """
        size_kernel = dim_in * dim_out
        kernel_idx_start = param_idx_start
        kernel_idx_end = kernel_idx_start + size_kernel
        param_idx_start = kernel_idx_end
        kernel = dnn_params[:, kernel_idx_start:kernel_idx_end]  # (B, dim_in*dim_out)
        kernel = tf.reshape(kernel, [-1, 1, dim_in, dim_out])  # (B, 1, dim_in, dim_out)
        layer_output = tf.matmul(layer_input, kernel)
        return layer_output, param_idx_start

    def _apply_bias(self, dim_out, param_idx_start, dnn_params, layer_output):
        """计算和bias相加.
        """
        if not self.use_bias:
            return layer_output, param_idx_start
        size_bias = dim_out
        bias_idx_start = param_idx_start
        bias_idx_end = bias_idx_start + size_bias
        param_idx_start = bias_idx_end
        bias = dnn_params[:, bias_idx_start:bias_idx_end]  # (B, dim_out)
        bias = tf.reshape(bias, [-1, 1, 1, dim_out])  # (B, 1, 1, dim_out)
        layer_output = layer_output + bias  # (B, L, 1, dim_out)
        return layer_output, param_idx_start

    def _apply_mask(self, inputs, outputs):
        """对输出进行mask.
        """
        mask = self._has_non_zero(inputs, axis=-1, keepdims=True)  # (B, L, 1, 1)
        mask = tf.cast(mask, dtype=outputs.dtype)
        outputs = outputs * mask  # (B, L, 1, dim_out)
        return outputs

    def _expand_input_dim(self, inputs):
        """将inputs的维度扩展到CAN_EXPANDED_INPUT_DIM维.
        """
        while len(inputs.shape) < CANLayer.CAN_EXPANDED_INPUT_DIM:
            axis = len(inputs.shape) - 1
            inputs = tf.expand_dims(inputs, axis=axis)
        return inputs

    def _get_dnn_dims(self, dim_in, size_dnn_param):
        """获取网络各层的输出维度.
        """
        if self.dnn_dims is None:
            dnn_dims = self._auto_decide_dnn_param_size(dim_in, size_dnn_param)
        else:
            dnn_dims = self.dnn_dims
        return dnn_dims

    def _apply_activation(self, layer_output, is_last_layer):
        """根据是否为最后一层，决定是否使用activation.
        """
        if self.output_layer_use_activation or not is_last_layer:
            layer_output = self.activation(layer_output)
        return layer_output

    def _apply_res_net(self, layer_input, layer_output):
        """根据use_res_net参数决定是否使用残差网络.
        """
        if self.use_res_net:
            layer_output = layer_input + layer_output
        return layer_output

    def _reshape_outputs(self, inputs, outputs):
        """根据inputs形状和output_combiner对outputs进行reshape.
        """
        outputs = tf.squeeze(outputs, axis=[-2])  # (B, L, dim_out)
        if len(inputs.shape) == 2:  # 输入为2维矩阵
            outputs = tf.squeeze(outputs, axis=[1])  # (B, dim_out)
        elif self.output_combiner is not None:  # 输入为3维张量
            outputs = self.output_combiner_func(outputs)  # (B, dim_out)
        return outputs

    def call(self, inputs, dnn_params):
        """计算co-action network.

        Args:
            inputs (tf.Tensor): 输入，形状为(B, L, D0) 或 (B, D0)
            dnn_params (tf.Tensor): DNN的参数，要和类参数中的dnn_dims和use_bias相匹配。形状为(B, size_dnn_param)

        Raises:
            ValueError: inputs形状不对

        Returns:
            (tf.Tensor): output_combiner 不为None时，输出形状为(B, Dn)
                         output_combiner = None，且输入为3维时，形状为(B, L, Dn)
                         output_combiner = None，且输入为2维时，形状为(B, Dn)
        """
        dim_in = int(inputs.shape[-1])  # 缩写为D0

        inputs = self._expand_input_dim(inputs)  # (B, L, 1, dim_in)

        size_dnn_param = int(dnn_params.shape[-1])
        dnn_dims = self._get_dnn_dims(dim_in, size_dnn_param)
        self._check_dnn_param_size(dim_in, dnn_dims, size_dnn_param)

        param_idx_start = 0  # 当前网络参数的起始下标，随网络层数改变
        layer_input = inputs  # (B, L, 1, dim_in)
        for layer_idx, dim_out in enumerate(dnn_dims):
            # apply weights and bias
            layer_output, param_idx_start = self._apply_kernel(dim_in, dim_out, param_idx_start,
                                                               dnn_params, layer_input)

            layer_output, param_idx_start = self._apply_bias(dim_out, param_idx_start,
                                                             dnn_params, layer_output)

            # apply activation and res_net
            is_last_layer = layer_idx == (len(dnn_dims) - 1)
            layer_output = self._apply_activation(layer_output, is_last_layer)
            layer_output = self._apply_res_net(layer_input, layer_output)

            # update params
            dim_in = dim_out
            layer_input = layer_output

        outputs = layer_output  # (B, L, 1, dim_out)
        if self.mask_all_zero_embedding:
            outputs = self._apply_mask(inputs, outputs)  # (B, L, 1, dim_out)

        outputs = self._reshape_outputs(inputs, outputs)
        return outputs
