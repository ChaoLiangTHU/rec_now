# coding=utf-8
''' 2021_12_02 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

from rec_now.util.param_normalizer import wrap_as_list


class StarDenseLayer(keras.layers.Dense):
    """STAR Topology Fully-Connected Network.

    将dense layer的网络参数，和个性化的网络参数相乘，作为dense层的参数，从而增加网络的个性化水平。

    Reference:
        [One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction]
        (https://arxiv.org/abs/2101.11427)

    Note:
        尽管计算量与相同输入输出的Dense层相近，但中间计算过程会生成batch size倍大小的kernel，
        所以大batch时，使用量会显著增加。

    Symbols:
        B: batch size
        D: 输入的维度
        U: 输出的维度

    Example:
        batch_size = 2
        units_in = 3
        units_out = 5
        max_scene_size = 100

        # 生成参数表
        starnet_param_size = StarDenseLayer.get_starnet_param_size(units_in, units_out)
        starnet_kernel_initializer = StarDenseLayer.get_starnet_kernel_initializer()
        initial_value = starnet_kernel_initializer([max_scene_size, starnet_param_size], dtype=tf.float32)
        starnet_param_lookup_table = tf.Variable(initial_value=initial_value)

        # 构建一个batch
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        inputs_scene = tf.random.uniform([batch_size], minval=0, maxval=max_scene_size, dtype=tf.int32, seed=2)
        starnet_param_this_batch = tf.nn.embedding_lookup(starnet_param_lookup_table, inputs_scene)

        # run this batch
        layer = StarDenseLayer(units_out)
        outputs = layer(inputs, starnet_param_this_batch)
    """

    def __init__(self, units, **kwargs):
        """
        Args:
            units (int): 输出单元的数量
        """
        super().__init__(units, ** kwargs)

    @classmethod
    def get_starnet_param_size(cls, units_in, units_out):
        """计算star网络参数的参数量，用作外部通过embedding生成参数时的embedding dim.

        Args:
            units_in (int): 网络输入维度
            units_out (int): 网络输出维度

        Returns:
            (int): star网络的总参数量
        """
        return units_in * units_out + units_out

    @classmethod
    def get_starnet_kernel_initializer(cls):
        """生成starnet的kernel的初始化器.

        由于分场景的kernel是和主干的kernel相乘，因而，分场景的kernel应该用1初始化.

        Returns:
            全1初始化器的实例.
        """
        return tf.ones_initializer()

    @classmethod
    def get_starnet_bias_initializer(cls):
        """生成starnet的bias的初始化器.

        Returns:
            全0初始化器的实例.
        """
        return tf.zeros_initializer()

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        self.units_in = input_shape[-1]
        return super().build(input_shape)

    def _reshape_starnet_param(self, net_param):
        """对star网络参数进行reshape.

        Args:
            net_param (tf.Tensor)): 个性化网络的参数，形状为(B, D*U+U)

        Returns:
            (tf.Tensor): 个性化网络的kernel参数，形状为(B, D, U)
            (tf.Tensor): 个性化网络的bias参数，形状为(B, 1, U)
        """
        dim_in = self.units_in
        dim_out = self.units
        kernel = net_param[:, :dim_in * dim_out]
        kernel = tf.reshape(kernel, [-1, dim_in, dim_out])
        bias = net_param[:, dim_in * dim_out:]
        bias = tf.reshape(bias, [-1, 1, dim_out])
        return kernel, bias

    def call(self, inputs, starnet_param_list):
        """计算该dense层的输出.

        输出 = matmul(inputs, sum(kernels)) + bias. 其中kernels由self.kernel和starnet_param_list参数叠加得到

        Args:
            inputs (tf.Tensor): 输入，形状为(B, D)
            starnet_param_list (List(tf.Tensor) or tf.Tensor): 个性化网络参数list，其中元素形状为(B, D*U+U)
            starnet_weight (float, optional): 个性化网络参数的权重

        Returns:
            (tf.Tensor): dense层的输出，形状为(B, U)
        """
        kernel_list = []
        bias_list = []

        starnet_param_list = wrap_as_list(starnet_param_list)
        for net_param in starnet_param_list:
            kernel, bias = self._reshape_starnet_param(net_param)  # (B, D, U), (B, 1, U)
            kernel_list.append(kernel)
            bias_list.append(bias)

        kernel_final = tf.expand_dims(self.kernel, axis=0)  # (1, D, U)
        for kernel in kernel_list:
            kernel_final = kernel_final * kernel  # (B, D, U)

        num_starnet = len(starnet_param_list)
        if num_starnet > 1:
            bias_final = tf.add_n(bias_list)   # (B, 1, U)
        else:
            bias_final = bias_list[0]
        if self.bias is not None:
            bias_final = bias_final + self.bias  # (B, 1, U)

        # 由于starnet的kernel是用全1初始化的，而bias需要用全0初始化。
        # 但是，实际使用时，kenral和bias放在一个变量中，并统一用1进行初始化的。
        # 因而，这里我们统一对每个bias进行-1。
        bias_final = bias_final + tf.constant(-num_starnet, dtype=bias_final.dtype)

        inputs = tf.expand_dims(inputs, axis=1)  # (B, 1, D)
        outputs = tf.matmul(inputs, kernel_final)  # (B, 1, U)
        outputs = tf.add(outputs, bias_final)  # (B, 1, U)
        outputs = tf.squeeze(outputs, axis=1)  # (B, U)

        outputs = self.activation(outputs)  # (B, U)
        return outputs


class ParasiticStarDenseLayer(keras.layers.Layer):
    """在已有dense层的基础上，叠加一组寄生的kernel和bias.

    寄生的kernel和原有kernel相乘作为该层的kernel，寄生的bias和原有的bias相加作为该层的bias。
    该层的作用和StarDesneLayer相似，不同之处在于，每一个场景需要生成一个该层的实例。
    相对于StarDesneLayer的优点在于，该层不会生成batch size倍大小的中间变量，因而可以节省内存。

    Symbols:
        B: batch size
        D: 输入的维度
        U: 输出的维度
    """

    def __init__(self, kernel=None, bias=None, dense_layer=None, activation=None,
                 parasitic_kernel_initializer='Ones', num_groups=1, **kwargs):
        """
        Args:
            kernel (tf.Tensor, optional): dense层的kernel，形状为(D, U)
            bias (tf.Tensor, optional): dense层的bias，形状为(U,)
            dense_layer (keras.layers.Dense): 需要寄生的dense层，和kernel、bias只能设置一个
            activation (str or callable, optional): 激活函数
            parasitic_kernel_initializer (str or initializer): 寄生kernel的初始化器. Defaults to 'Ones'.
            num_groups (int, optional): 寄生kernel的组数. Defaults to 1.
        """
        if dense_layer is not None:
            self.dense_layer = dense_layer
            if dense_layer.built:
                self.trunk_kernel = dense_layer.kernel
                self.trunk_bias = dense_layer.bias
        else:
            self.trunk_kernel = kernel
            self.trunk_bias = bias
            self.dense_layer = None
            if self.trunk_kernel is None:
                raise ValueError('kernel is None')
        self.parasitic_kernel_initializer = keras.initializers.get(parasitic_kernel_initializer)
        self.activation = keras.activations.get(activation)
        self.num_groups = num_groups
        super().__init__(**kwargs)

    def _build_dense_layer(self, input_shape):
        """生成被寄生的dense层的kernel和bias.
        """
        if hasattr(self, 'trunk_kernel'):
            return
        if not self.dense_layer.built:
            with tf.name_scope(self.dense_layer.name):
                self.dense_layer.build(input_shape)
        self.trunk_kernel = self.dense_layer.kernel
        self.trunk_bias = self.dense_layer.bias

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        if self.built:
            return

        self._build_dense_layer(input_shape)
        shape = tf.TensorShape([self.num_groups])
        self.parasitic_kernel = self.add_weight(
            'kernel',
            shape=shape.concatenate(self.trunk_kernel.shape),
            initializer=self.parasitic_kernel_initializer,
            dtype=self.trunk_kernel.dtype,
            trainable=True)
        if self.trunk_bias is not None:
            self.parasitic_bias = self.add_weight(
                'bias',
                shape=shape.concatenate(self.trunk_bias.shape),
                initializer=tf.zeros_initializer(),
                dtype=self.trunk_bias.dtype,
                trainable=True)
        else:
            self.parasitic_bias = None
        self.built = True

    def _only_use_trunk(self, group_idx):
        """是否只使用主干的参数.
        """
        if group_idx is None:
            return True
        if isinstance(group_idx, int) and group_idx < 0:
            return True
        return False

    def _get_kernel(self, group_idx, stop_trunk_grad):
        """获取最终的kernel.

        Args:
            group_idx (int, optional): 使用第几组寄生参数.
                                      如果该值为None，则只使用主干的参数.
            stop_trunk_grad (bool): 是否不回传梯度到主干的参数上.

        Returns:
            (tf.Tensor): 最终使用的kernel
        """
        kernel = self.trunk_kernel
        # 不回传梯度到主干
        if stop_trunk_grad:
            kernel = tf.stop_gradient(kernel)
        # 仅使用主干kernel
        if self._only_use_trunk(group_idx):
            return kernel
        # 使用寄生kernel
        kernel = kernel * self.parasitic_kernel[group_idx]
        return kernel

    def _get_bias(self, group_idx, stop_trunk_grad):
        """获取最终的bias.
        """
        bias = self.trunk_bias
        # 没有bias
        if bias is None:
            return None
        # 不回传梯度到主干
        if stop_trunk_grad:
            bias = tf.stop_gradient(bias)
        # 仅使用主干bias
        if self._only_use_trunk(group_idx):
            return bias
        # 使用寄生bias
        parasitic_bias = self.parasitic_bias[group_idx]
        bias = bias + parasitic_bias
        return bias

    def call(self, inputs, group_idx=0, stop_trunk_grad=False):
        """计算该dense层的输出.

        Args:
            inputs (tf.Tensor): 输入，形状为(B, D)
            group_idx (int, optional): 使用第几组寄生参数，Defaults to 0.
                                      如果该值为None，则只使用主干的参数.
            stop_trunk_grad (bool, optional): 是否不回传梯度到主干的参数上. Defaults to False.

        Returns:
            (tf.Tensor): dense层的输出，形状为(B, U)
        """
        kernel = self._get_kernel(group_idx, stop_trunk_grad)
        bias = self._get_bias(group_idx, stop_trunk_grad)

        outputs = tf.matmul(inputs, kernel)
        if bias is not None:
            outputs = outputs + bias
        outputs = self.activation(outputs)

        return outputs
