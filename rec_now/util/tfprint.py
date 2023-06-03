# coding=utf-8
''' 2021_09_16 lcreg163@163.com

对tensorflow 1.x 版本的tf.Print的封装.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
if tf.__version__ >= '2.0.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


def tfprint(tensor, description, do_print=True, first_n=2000, summarize=8):
    """对tf.Print的封装，增加了打印变量shape的功能，并通过do_print参数控制是否输出，方便在正式脚本中屏蔽tf.Print语句.

    Args:
        tensor (tf.Tensor): 要打印的tensor
        description (str): 文字说明
        do_print (bool, optional): 是否打印输入. Defaults to True.
        first_n (int, optional): 至多打印多少次. Defaults to 2000.
        summarize (int, optional): 每次打印输入的多少个元素. Defaults to 8.

    Returns:
        (tf.Tensor): 要打印的tensor，该值必须加入到计算图中才能进行打印
    """
    # 不进行打印的情况
    if not do_print:
        return tensor

    # 打印tensor的情况
    print_list = [tf.shape(tensor), "  :  ", tensor]
    return tf.Print(tensor, print_list, description, first_n, summarize=summarize)


def _append_one_var(varlist, var, is_last_var, reshape_func):
    """将一个待打印的var添加到varlist的尾部.
    """
    if isinstance(var, str):
        varlist.append(var)
    else:
        varlist.extend([tf.shape(var), "  :  ", reshape_func(var)])
        if not is_last_var:
            varlist.append(";  ")


def tfprintlist(tensor, description, des_var_list=None, reshape=True, do_print=True, first_n=200000, summarize=8):
    """对tf.Print的封装，打印输入tensor或des_var_list中的变量的值和shape.

    Args:
        tensor (tf.Tensor): 要打印的tensor
        description (str): 文字说明
        des_var_list ([type], optional): 如果为None，则打印x；如果为list，则打印其中的每个tensor的值和shape. Defaults to None.
        reshape (bool, optional): 是否把要打印的tensor变成1维的进行打印. Defaults to True.
        do_print (bool, optional): 是否打印输入. Defaults to True.
        first_n (int, optional): 至多打印多少次. Defaults to 200000.
        summarize (int, optional): 每次打印输入的多少个元素. Defaults to 8.

    Returns:
        (tf.Tensor): 要打印的tensor，该值必须加入到计算图中才能进行打印
    """
    # 不进行打印的情况
    if not do_print:
        return tensor

    # 将输入按需变换为1维的
    def reshape_func(input):
        if not reshape:
            return input
        else:
            return tf.reshape(input, [-1])

    # 只打印tensor的情况
    if des_var_list is None:
        varlist = [tf.shape(tensor), "  :  ", reshape_func(tensor)]
        return tf.Print(tensor, varlist, description, first_n, summarize=summarize)

    # 打印一组张量的情况
    varlist = []
    for idx, var in enumerate(des_var_list):
        is_last_var = idx >= len(des_var_list)-1
        _append_one_var(varlist, var, is_last_var, reshape_func)
    return tf.Print(tensor, varlist, description, first_n, summarize=summarize)


def tfprint_minmax(tensor, description, do_print=True, first_n=200000, summarize=8):
    """对tf.Print的封装，打印输入张量的形状和最小、最大值.

    Args:
        tensor (tf.Tensor): 要打印的tensor
        description (str): 文字说明
        do_print (bool, optional): 是否打印输入. Defaults to True.
        first_n (int, optional): 至多打印多少次. Defaults to 200000.
        summarize (int, optional): 每次打印输入的多少个元素. Defaults to 8.

    Returns:
        (tf.Tensor): 要打印的tensor，该值必须加入到计算图中才能进行打印
    """
    # 不进行打印的情况
    if not do_print:
        return tensor

    # 打印tensor及其最小最大值的情况
    min_val = tf.reduce_min(tensor, axis=None, keepdims=False)
    max_val = tf.reduce_max(tensor, axis=None, keepdims=False)
    print_list = [tf.shape(tensor), "  :  minmax: ", min_val, max_val]
    return tf.Print(tensor, print_list, description, first_n, summarize=summarize)
