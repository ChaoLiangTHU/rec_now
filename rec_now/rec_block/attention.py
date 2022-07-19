# coding=utf-8
''' 2021_11_01 lcreg163@163.com
注意力机制相关的函数
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def attention_by_dot_product(user_emb, doc_emb, filter_neg=False):
    """采用dot product 计算的attention.

    Symbols:
        B: batch size
        L: 用户特征的长度(比如用户有L个tag)
        D: embedding 的维度

    Args:
        user_emb (tf.Tensor): 用户特征的embedding，形状为(B, L, D)
        doc_emb (tf.Tensor): 物品的embedding，形状为(B, D)

    Returns:
        attn_mat (tf.Tensor): attention的结果，形状为(B, D)
        attn_score_sum (tf.Tensor): attention的分数，形状为(B, 1)
    """
    doc_emb_extended = tf.expand_dims(doc_emb, axis=1)
    attn_score = user_emb * doc_emb_extended  # (B, L, D)
    attn_score = tf.reduce_sum(attn_score, axis=2, keepdims=True)  # (B, L, 1), 对3维张量的第2维聚合
    if filter_neg:
        attn_score = tf.maximum(attn_score, 0.0)
    attn = user_emb * attn_score  # (B, L, D)
    attn_mat = tf.reduce_sum(attn, axis=1, keepdims=False)  # (B, D)

    attn_score = tf.squeeze(attn_score, axis=2)  # (B, L)
    attn_score_sum = tf.reduce_sum(attn_score, axis=1, keepdims=True)  # (B, 1)
    return attn_mat, attn_score_sum


def attention_by_dnn(user_emb, doc_emb, dnn_dims, dnn_activation='relu', dnn_name='din'):
    """采用DNN 计算的attention.

    Symbols:
        B: batch size
        L: 用户特征的长度(比如用户有L个tag)
        D: embedding 的维度

    Args:
        user_emb (tf.Tensor): 用户特征的embedding，形状为(B, L, D)
        doc_emb (tf.Tensor): 物品的embedding，形状为(B, D)
        dnn_dims (list of int): DNN的各层的维度，最后一维必须为1
        dnn_activation (str or func): DNN的激活函数
        dnn_name (str): DNN的名称

    Returns:
        attn_mat (tf.Tensor): attention的结果，形状为(B, D)
        attn_score_sum (tf.Tensor): attention的分数，形状为(B, 1)
        model (keras model): DIN权重模型
    """
    doc_emb_extended = tf.expand_dims(doc_emb, axis=1)  # (B, 1, D)
    doc_emb_tiled = tf.tile(doc_emb_extended, [1, tf.shape(user_emb)[1], 1])  # (B, L, D)
    dnn_input = tf.concat([user_emb, doc_emb_tiled], axis=-1)  # (B, L, 2*D)

    if dnn_dims[-1] != 1:
        dnn_dims.append(1)

    model = tf.keras.Sequential(name=dnn_name)
    for i, dim in enumerate(dnn_dims):
        name = f'layer{i}'
        activation_func = dnn_activation if i < len(dnn_dims) - 1 else None
        layer = tf.keras.layers.Dense(dim, activation=activation_func, name=name)
        model.add(layer)
    attn_score = model(dnn_input)  # (B, L, 1)
    attn_score = tf.sigmoid(attn_score)  # (B, L, 1)

    attn = user_emb * attn_score  # (B, L, D)
    attn_mat = tf.reduce_sum(attn, axis=1, keepdims=False)  # (B, D)

    attn_score = tf.squeeze(attn_score, axis=2)  # (B, L)
    attn_score_sum = tf.reduce_sum(attn_score, axis=1, keepdims=True)  # (B, 1)
    return attn_mat, attn_score_sum, model
