# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.ple_layer import PLELayer


class TestPLELayer(unittest.TestCase):
    def test_ple_layer(self):
        tf.random.set_seed(1)

        num_task = 2
        num_shared_task = 1
        list_of_dnn_dims = [[2, 3], [2, 3], [3, 2]]
        list_of_num_experts_per_task = [4, 3, 2]
        inputs = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=tf.float32)

        ple_layer = PLELayer(num_task, list_of_dnn_dims, list_of_num_experts_per_task, num_shared_task, name="PLE")
        task1_output, task2_output = ple_layer(inputs)

        expected_task1_output = [[-0.00202228, 0.03448214],
                                 [0.00163524, 0.13148618]]
        output_diff = calc_sum_of_abs_diff(task1_output, expected_task1_output)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

        expected_task2_output = [[-0.00116823, 0.01959552],
                                 [0.00839254, 0.04192837]]
        output_diff2 = calc_sum_of_abs_diff(task2_output, expected_task2_output)
        self.assertAlmostEqual(output_diff2, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
