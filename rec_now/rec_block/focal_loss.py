# coding=utf-8
''' 2021_10_20 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def focal_crossentropy_loss(
    labels,
    logits,
    alpha=0.25,
    gamma=2.0,
    stop_weight_gradient=False,
    return_mean=True
):
    """ 误差函数focal loss的实现。
    focal loss出自论文https://arxiv.org/pdf/1708.02002.pdf
    对于非均衡样本分类问题比较有效。

    Symbols:
        B: batch size

    Args:
        labels (tf.Tensor): 样本标签，元素为1或0，形状为(B,)
        logits (tf.Tensor): 模型输出的logits，形状为(B,)
        alpha (float, optional): 正样本权重。负样本权重为1-alpha。设为None则正负样本权重均为1。
        gamma (float, optional): 样本权重，误差**gamma作为第二项样本权重。设为None则不启用
        stop_weight_gradient (bool, optional): 是否对gamma调制的权重不回传梯度
        return_mean (bool, optional): 是否返回loss的均值

    Raises:
        ValueError: alpha参数设置不正确
        ValueError: gamma参数设置不正确

    Returns:
        tf.Tensor: focal loss. return_mean 为 True 时，为一个标量。
                               return_mean 为 False 时，形状为 (B,)
    """
    if alpha and (alpha <= 0.0 or alpha >= 1.0):
        raise ValueError("Value of alpha should be greater than zero and less than one.")
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    focal_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    if alpha:
        alpha = tf.cast(alpha, dtype=labels.dtype)
        alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)
        focal_loss = alpha_factor * focal_loss

    if gamma:
        pred_prob = tf.sigmoid(logits)
        pred_sim = (labels * pred_prob) + ((1 - labels) * (1 - pred_prob))
        gamma = tf.cast(gamma, dtype=labels.dtype)
        modulating_factor = tf.pow((1.0 - pred_sim), gamma)
        if stop_weight_gradient:
            modulating_factor = tf.stop_gradient(modulating_factor)
        focal_loss = modulating_factor * focal_loss

    if return_mean:
        focal_loss = tf.reduce_mean(focal_loss)
    return focal_loss
