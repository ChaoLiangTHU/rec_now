# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.mmoe_layer import MMOELayer


class TestMMOELayer(unittest.TestCase):
    def test_mmoe_layer(self):
        tf.random.set_seed(1)
        num_task = 2
        num_experts = 4
        dnn_dims = [8, 3]
        merge_output = False
        inputs = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7], [4, 5, 6, 7]], dtype=tf.float32)

        mmoe_layer = MMOELayer(num_task, num_experts, dnn_dims, name="MMoE")
        result = mmoe_layer(inputs, merge_output)

        expected_result = [[[0.14612462, 0.44929513, -0.78639925],
                            [0.4357101, 1.235869, -1.852003],
                            [0.4357101, 1.235869, -1.852003]],

                           [[0.3218293, -0.01983448, -0.7208969],
                            [0.8113805, -0.34114015, -1.272174],
                            [0.8113805, -0.34114015, -1.272174]]]

        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
