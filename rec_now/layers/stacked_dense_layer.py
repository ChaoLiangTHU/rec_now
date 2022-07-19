# coding=utf-8
''' 2021_10_15 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


from rec_now.util.param_normalizer import wrap_as_list
from rec_now.layers.star_dense_layer import ParasiticStarDenseLayer


class StackedDenseLayer(keras.layers.Dense):
    """ 带有个性化的网络参数的dense层.

    在dense layer的网络参数上，叠加个性化的网络参数，作为dense层的参数，从而增加网络的个性化水平。

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
        resnet_param_size = StackedDenseLayer.get_resnet_param_size(units_in, units_out)
        resnet_kernel_initializer = StackedDenseLayer.get_resnet_kernel_initializer()
        initial_value = resnet_kernel_initializer([max_scene_size, resnet_param_size], dtype=tf.float32)
        resnet_param_lookup_table = tf.Variable(initial_value=initial_value)

        # 构建一个batch
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        inputs_scene = tf.random.uniform([batch_size], minval=0, maxval=max_scene_size, dtype=tf.int32, seed=2)
        resnet_param_this_batch = tf.nn.embedding_lookup(resnet_param_lookup_table, inputs_scene)

        # run this batch
        layer = StackedDenseLayer(units_out)
        outputs = layer(inputs, resnet_param_this_batch)
    """

    def __init__(self, units, **kwargs):
        """
        Args:
            units (int): 输出单元的数量
        """
        super().__init__(units, ** kwargs)

    @classmethod
    def get_resnet_param_size(cls, units_in, units_out):
        """计算个性化网络参数的参数量，用作外部通过embedding生成参数时的embedding dim.

        Args:
            units_in (int): 网络输入维度
            units_out (int): 网络输出维度

        Returns:
            (int): 个性化网络的总参数量
        """
        return units_in * units_out + units_out

    @classmethod
    def get_resnet_kernel_initializer(cls):
        """生成resnet的kernel的初始化器.

        由于个性化的kernel是和主干的kernel相加，因而，分场景的kernel应该用0进行初始化.

        Returns:
            全0初始化器的实例.
        """
        return tf.zeros_initializer()

    @classmethod
    def get_resnet_bias_initializer(cls):
        """生成resnet的bias的初始化器.

        Returns:
            全0初始化器的实例.
        """
        return tf.zeros_initializer()

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        self.units_in = input_shape[-1]
        return super().build(input_shape)

    def _reshape_resnet_param(self, net_param):
        """对个性化的网络参数进行reshape.

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

    def call(self, inputs, resnet_param_list, resnet_weight=1.0):
        """计算该dense层的输出.

        输出 = matmul(inputs, sum(kernels)) + bias. 其中kernels由self.kernel和resnet_param_list参数叠加得到

        Args:
            inputs (tf.Tensor): 输入，形状为(B, D)
            resnet_param_list (List(tf.Tensor) or tf.Tensor): 个性化网络参数list，其中元素形状为(B, D*U+U)
            resnet_weight (float, optional): 个性化网络参数的权重

        Returns:
            (tf.Tensor): dense层的输出，形状为(B, U)
        """
        kernel_list = []
        bias_list = []

        resnet_param_list = wrap_as_list(resnet_param_list)
        for net_param in resnet_param_list:
            kernel, bias = self._reshape_resnet_param(net_param)  # (B, D, U), (B, 1, U)
            kernel_list.append(kernel)
            bias_list.append(bias)
        kernel_final = tf.add_n(kernel_list)  # (B, D, U)
        bias_final = tf.add_n(bias_list)  # (B, 1, U)

        if resnet_weight != 1.0:
            kernel_final = resnet_weight * kernel_final
            bias_final = resnet_weight * bias_final

        if self.kernel is not None:
            kernel_final = kernel_final + tf.expand_dims(self.kernel, axis=0)
        if self.bias is not None:
            bias_final = bias_final + self.bias

        inputs = tf.expand_dims(inputs, axis=1)  # (B, 1, D)
        outputs = tf.matmul(inputs, kernel_final)  # (B, 1, U)
        outputs = tf.add(outputs, bias_final)  # (B, 1, U)
        outputs = tf.squeeze(outputs, axis=1)  # (B, U)

        outputs = self.activation(outputs)  # (B, U)
        return outputs


class ParasiticStackedDenseLayer(ParasiticStarDenseLayer):
    """在已有dense层的基础上，叠加一组寄生的kernel和bias.

    寄生的kernel和原有kernel相加作为该层的kernel，寄生的bias和原有的bias相加作为该层的bias。
    该层的作用和StackedDenseLayer相似，不同之处在于，每一个场景需要生成一个该层的实例。
    相对于StackedDenseLayer的优点在于，该层不会生成batch size倍大小的中间变量，因而可以节省内存。

    与父类ParasiticStarDenseLayer的不同在于，寄生的kernel和原有的kernel是相加，而不是相乘。
    因而，寄生的kernel的初始化值一般为0，而不是1。
    """

    def __init__(self, kernel=None, bias=None, dense_layer=None, activation=None,
                 parasitic_kernel_initializer='Zeros', num_groups=1, **kwargs):
        """[summary]

        Args:
            kernel (tf.Tensor, optional): dense层的kernel，形状为(D, U)
            bias (tf.Tensor, optional): dense层的bias，形状为(U,)
            dense_layer (keras.layers.Dense): 需要寄生的dense层，和kernel、bias只能设置一个
            activation (str or callable, optional): 激活函数
            parasitic_kernel_initializer (str or initializer): 寄生kernel的初始化器. Defaults to 'Zeros'.
            num_groups (int, optional): 寄生kernel的组数. Defaults to 1.
        """
        super().__init__(kernel=kernel, bias=bias, dense_layer=dense_layer, activation=activation,
                         parasitic_kernel_initializer=parasitic_kernel_initializer,
                         num_groups=num_groups, **kwargs)

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
        if group_idx is None:
            return kernel
        # 使用寄生kernel
        kernel = kernel + self.parasitic_kernel[group_idx]
        return kernel
