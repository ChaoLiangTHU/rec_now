# coding=utf-8
''' 2022_01_06 lcreg163@163.com

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow import keras


def _get_slice_size(dynamic_shape, length, axis, rank):
    """将动态shape (通过tf.shape得到)的第axis维替换为length.
    """
    size = [dynamic_shape[i] if i != axis else length for i in range(rank)]
    return size


def _get_pad_size(num_to_pad, axis, rank):
    """获取tf.pad函数所需的paddings参数.

    Args:
        num_to_pad (tf.Tensor): 第axis维需要在最后补充的值的数量.
        axis (int): 需要进行pad的维度.
        rank (int): 需要pad的tensor的总的维度.

    Returns:
        (list): tf.pad函数所需的paddings参数.
    """
    pad_size = [[0, 0] if i != axis else [0, num_to_pad] for i in range(rank)]
    return pad_size


@tf.function
def pad_or_truncate(tensor, length, axis=-1, constant_values=0):
    """将tensor的第axis维的维度截断或补全到维度为length.

    如果tensor的第axis维长度大于length，则进行截断.
    如果tensor的第axis维长度小于length，则在最后补constant_values.
    如果tensor的第axis维长度等于length，则返回tensor.

    该方法主要用于动态地将用户的稀疏特征（比如点击历史），归一化到相同的长度.
    比如，对于形状为(batch_size, None, embedding_dim)的tensor，
    pad_or_truncate(tensor, length, axis=1)的输出的形状为(batch_size, length, embedding_dim)，
    该输出可以用于需要定长输入的场景，比如transformer.

    Args:
        tensor (tf.Tensor): 要进行截断或填充的tensor.
        length (int): 截断或补全后的长度.
        axis (int, optional): 对哪一维度进行截断或补全. Defaults to -1.
        constant_values (int, optional): 补全的值. Defaults to 0.

    Returns:
        (tf.Tensor): 截断或补全后的tensor.
    """
    length = int(length)
    shape = tf.shape(tensor)
    rank = tensor.shape.rank
    origin_length = shape[axis]
    axis = axis % rank  # 保证axis非负

    if length < origin_length:
        begin = [0] * rank
        size = _get_slice_size(shape, length, axis, rank)
        result = tf.slice(tensor, begin, size)
    elif length > origin_length:
        num_to_pad = length - origin_length
        pad_size = _get_pad_size(num_to_pad, axis, rank)
        result = tf.pad(tensor, pad_size, constant_values=constant_values)
    else:
        result = tensor
    static_shape = tensor.shape.as_list()
    static_shape[axis] = length
    result = tf.ensure_shape(result, static_shape)
    return result


class FixLengthLayer(keras.layers.Layer):
    """将tensor的第axis维的维度截断或补全到维度为length.
    """

    def __init__(self, length, axis, constant_values=0, **kwargs):
        """
        Args:
            length (int): 截断或补全后的长度.
            axis (int, optional): 对哪一维度进行截断或补全. Defaults to -1.
            constant_values (int, optional): 补全的值. Defaults to 0.
        """
        super().__init__(**kwargs)
        self.length = length
        self.axis = axis
        self.constant_values = constant_values

    def call(self, inputs):
        """将inputs的第axis维的维度截断或补全到维度为length.

        如果inputs的第axis维长度大于length，则进行截断.
        如果inputs的第axis维长度小于length，则在最后补constant_values.
        如果inputs的第axis维长度等于length，则返回tensor.

        Args:
            inputs (tf.Tensor): 要进行截断或填充的tensor.

        Returns:
            (tf.Tensor): 截断或补全后的tensor.
        """
        return pad_or_truncate(inputs, self.length, self.axis, self.constant_values)
