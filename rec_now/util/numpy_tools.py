# coding=utf-8
''' 2021_11_01 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def calc_sum_of_abs_diff(arr1, arr2):
    """计算两个ndarray的差的绝对值之和.

    Args:
        arr1 (ndarray like): 待对比的ndarray 1
        arr2 (ndarray like): 待对比的ndarray 2, shape要和arr1相同

    Returns:
        total_abs_diff (float): arr1和arr2的的差的绝对值之和
    """
    arr1 = np.array(arr1, dtype=np.float64)
    arr2 = np.array(arr2, dtype=np.float64)
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    total_abs_diff = np.sum(abs_diff)
    return total_abs_diff


def all_equal(arr1, arr2):
    """计算两个ndarray的差的元素是否完全相同.

    Args:
        arr1 (ndarray like): 待对比的ndarray 1
        arr2 (ndarray like): 待对比的ndarray 2, shape要和arr1相同

    Returns:
        total_abs_diff (float): arr1和arr2的的差的绝对值之和
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return np.alltrue(arr1 == arr2)
