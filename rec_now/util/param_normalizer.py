# coding=utf-8
''' 2021_12_03 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def wrap_as_list(inputs):
    """将inputs调整为一个list.

    如果inputs是一个list，则直接返回inputs.
    如果inputs不是一个list，则将其封装进一个list.

    Args:
        inputs (Any): 输入参数

    Returns:
        (list): [description]
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return inputs
