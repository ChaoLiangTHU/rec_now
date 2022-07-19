# coding=utf-8
''' 2021_10_15 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class CINLayer(keras.layers.Layer):
    """xDeepFM中的Compressed Interaction Network (CIN)层.

    Reference:
        [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems]
        (https://arxiv.org/abs/1803.05170)


    Symbols:
        B: batch size
        D: 输入embedding的维度
        F: field的数量
        Hs: hidden layer channel size的list. (Hs[0]=F)
    """

    def __init__(self, hidden_sizes, embedding_dim=-1, initializer='glorot_uniform',
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """
        Args:
            hidden_sizes (List[int]): 隐层的通道数, 简写为Hs
            embedding_dim (int): embedding的维度，简写为D。在call的输入不为list时，需要提前设置该项
            其他参数意义同keras.layers.Layer
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.hidden_sizes = hidden_sizes
        self.embedding_dim = embedding_dim
        self.initializer = keras.initializers.get(initializer)

    def _extend_hidden_sizes(self):
        """将input的field数量和隐层channel数量拼成一个list.
        """
        return [self.num_field] + self.hidden_sizes

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        if(isinstance(input_shape, list)):
            self.num_field = len(input_shape)
            self.embedding_dim = int(input_shape[0][-1])
        else:
            if self.embedding_dim <= 0:
                raise ValueError('embedding_dim shall bigger than 0 when inputs is not a list of embeddings.')
            self.num_field = int(int(input_shape[0][-1]) / self. embedding_dim)

        extended_hidden_sizes = self._extend_hidden_sizes()
        self.idx2weight = {}
        for layer_idx in range(1, len(extended_hidden_sizes)):
            num_channel = extended_hidden_sizes[layer_idx]
            num_prev_channel = extended_hidden_sizes[layer_idx - 1]
            shape = [1, 1, num_channel, num_prev_channel * self.num_field]
            weight = self.add_weight(
                'weight_of_layer%s' % layer_idx,
                shape=shape,
                initializer=self.initializer,
                dtype=self.dtype,
                trainable=True)
            self.idx2weight[layer_idx] = weight

        return super().build(input_shape)

    def call(self, inputs, output_input=True, sum_channel=True):
        """计算inputs的CIN.

        Args:
            inputs (tf.Tensor or List[tf.Tensor]): 输入矩阵，形状为(B, F, D)
                                                    或长度为F的list，其中元素为形状为(B,D)
            output_input (bool, optional): 输出是否包含输入
            sum_channel (bool, optional): 是否将更层的channel维度相加。如果为False，则channel维进行concat

        Returns:
            (tf.Tensor): CIN的结果
                        sum_channel=True时，形状为(B, D)
                        sum_channel=False，output_input=True时，形状为(B, sum(Hs)*D)
                        sum_channel=False，output_input=False时，形状为(B, D+sum(Hs)*D)
        """

        if(isinstance(inputs, list)):
            emb = tf.concat(inputs, axis=1)  # (B, F*D)
        else:
            emb = inputs  # (B, F*D)

        emb_dim = self.embedding_dim
        num_field = self.num_field

        layer0 = tf.reshape(emb, [-1, num_field, emb_dim])  # (B, F, D)
        layer0_tranpose = tf.transpose(layer0, [0, 2, 1])  # (B, D, F), 3维张量的第1维和第2维对换
        layers = [layer0_tranpose]

        extended_hidden_sizes = self._extend_hidden_sizes()
        for layer_idx in range(1, len(extended_hidden_sizes)):
            prev_hidden = layers[-1]  # (B, D, Hs[layer_idx-1])，对于第1层, Hs[1-1] = Hs[0] = F
            hidden = tf.einsum('bdf,bdh->bdfh', layer0_tranpose, prev_hidden)  # (B, D, F, Hs[layer_idx-1])
            num_prev_channel = extended_hidden_sizes[layer_idx - 1]
            hidden_shape = [-1, emb_dim, num_field * num_prev_channel, 1]
            hidden = tf.reshape(hidden, hidden_shape)  # (B, D, F*Hs[layer_idx-1], 1)
            weight = self.idx2weight[layer_idx]  # (1, 1, Hs[layer_idx], F*Hs[layer_idx-1])
            hidden = tf.matmul(weight, hidden)  # (B, D, Hs[layer_idx], 1)
            hidden = tf.squeeze(hidden, axis=-1)  # (B, D, Hs[layer_idx])
            layers.append(hidden)

        if not output_input:
            layers = layers[1:]

        output = tf.concat(layers, axis=-1)  # (B, D, sum(Hs))
        if sum_channel:
            output = tf.reduce_sum(output, axis=-1, keepdims=False)  # (B, D)
        else:
            output = tf.transpose(output, perm=[0, 2, 1])  # (B, sum(Hs), D), 3维张量的第1维和第2维对换
            merged_dim = output.shape[1] * output.shape[2]
            output = tf.reshape(output, [-1, merged_dim])  # (B, sum(Hs)*D)
        return output
