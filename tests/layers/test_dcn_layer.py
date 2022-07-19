# coding=utf-8
''' 2021_11_29 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.dcn_layer import DCNLayer


class TestDCNLayer(unittest.TestCase):
    def test_dcn_layer(self):
        tf.random.set_seed(1)
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

        degree_of_cross = 3
        dcn_layer = DCNLayer(degree_of_cross)

        result = dcn_layer(inputs)

        expected_result = [[-3.0642402, -6.1284804, -9.19272],
                           [-140.33298, -175.41623, -210.49947]]
        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
