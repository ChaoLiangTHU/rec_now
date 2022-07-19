# coding=utf-8
''' 2021_12_01 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

from rec_now.rec_block.embedding_wise_weight import gather_embedding_element_wise_weight


class SENETLayer(keras.layers.Dense):
    """Squeeze-Excitation network (SENET) layer.

    与原始论文中各个fields的embeddings的维度必须相同，而我们的实现方案不要求各个embeddings的维度相同.

    Reference:
        [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction]
        (https://arxiv.org/abs/1905.09433)

    Symbols:
        B: batch size
        F: filed 数量
        Df: 第f个field的维度
        total_dim: sum(Df)
    """

    def __init__(self, reduction_ratio, activation_inner='tanh', activation_outer='tanh', **kwargs):
        """
        Args:
            reduction_ratio (float): 压缩比例
            activation_inner (str, optional): SENET的内层激活函数
            activation_outer (str, optional): SENET的外层激活函数
            其他参数复用自父类keras.layers.Dense
        """
        super().__init__(0, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.activation_inner = keras.activations.get(activation_inner)
        self.activation_outer = keras.activations.get(activation_outer)
        self.input_spec = None

    def _build_senet(self):
        """生成SENET
        """
        name = f"{self.name}/senet"
        model = keras.models.Sequential(name=name)

        dnn_dims = [self.middle_dim, self.num_field]
        dnn_activations = [self.activation_inner, self.activation_outer]

        for idx, (dim, activation) in enumerate(zip(dnn_dims, dnn_activations)):
            layer = keras.layers.Dense(dim,
                                       activation=activation,
                                       use_bias=self.use_bias,
                                       kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer,
                                       kernel_regularizer=self.kernel_regularizer,
                                       bias_regularizer=self.bias_regularizer,
                                       activity_regularizer=self.activity_regularizer,
                                       kernel_constraint=self.kernel_constraint,
                                       bias_constraint=self.bias_constraint,
                                       name=f'{name}/dense_{idx}')
            model.add(layer)
        return model

    def _get_middle_dim(self):
        """获取中间层的维度
        """
        middle_dim = round(self.num_field * self.reduction_ratio)
        middle_dim = max(middle_dim, 1)  # 中间层维度必须大于0
        return middle_dim

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        if not isinstance(input_shape, list):
            input_shape = [input_shape]

        self.num_field = len(input_shape)
        self.total_dim = 0  # 所有维度的总和
        self.pos_idx = []  # 每个维度对于与SENET输出的下标
        for field_idx, shape in enumerate(input_shape):
            dim = int(shape[-1])
            self.total_dim += dim
            self.pos_idx.extend([field_idx] * dim)

        self.middle_dim = self._get_middle_dim()  # 中间层的维度
        self.senet = self._build_senet()
        self.built = True

    def call(self, inputs):
        """计算SENET.

        Args:
            inputs (List[tf.Tensor]): 该层的输入，其中每个元素为一个field的embedding，形状为(B, Df)

        Returns:
            (tf.Tensor): SENET的输出，形状为(B, total_dim)
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        # squeeze inputs
        squeezed_inputs = []
        for input in inputs:
            sequeezed_input = tf.reduce_mean(input, axis=-1, keepdims=True)
            squeezed_inputs.append(sequeezed_input)

        # apply senet
        senet_input = tf.concat(squeezed_inputs, axis=-1)  # (B, F)
        senet_output = self.senet(senet_input)  # (B, F)
        elementwise_weights = gather_embedding_element_wise_weight(senet_output, self.pos_idx)  # (B, total_dim)

        concated_inputs = tf.concat(inputs, axis=-1)  # (B, total_dim)
        outputs = concated_inputs * elementwise_weights  # (B, total_dim)

        return outputs
