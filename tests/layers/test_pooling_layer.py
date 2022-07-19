# coding=utf-8
''' 2021_11_29 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.pooling_layer import PoolingLayer


class TestPoolingLayer(unittest.TestCase):
    def test_pooling_layer(self):
        inputs = tf.constant([[1, 2, 3],
                              [10, 11, 12]], dtype=tf.float32)

        pooling_layer = PoolingLayer(axis=0, keepdims=True, combiner='sum')
        result = pooling_layer(inputs)
        expected_result = [[11, 13, 15]]
        diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(diff, 0.0, delta=1E-5)

    def test_pooling_layer_axis1(self):
        inputs = tf.constant([[1, 2, 3],
                              [10, 11, 12]], dtype=tf.float32)

        pooling_layer = PoolingLayer(axis=1, keepdims=False, combiner='sum')
        result = pooling_layer(inputs)
        expected_result = [6, 33]
        diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
