# coding=utf-8
''' 2022_01_07 lcreg163@163.com

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.fix_length_layer import FixLengthLayer


class TestFixLengthLayer(unittest.TestCase):
    def setUp(self) -> None:
        num_to_keep = 2
        self.fix_length_layer = FixLengthLayer(length=num_to_keep, axis=-1, name='FixLengthLayer')

    def test_truncate(self):
        tensor_to_truncate = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        truancated_tensor = self.fix_length_layer(tensor_to_truncate)
        expected_result = [[1, 2], [4, 5]]
        output_diff = calc_sum_of_abs_diff(truancated_tensor, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_pad(self):
        tensor_to_pad = tf.constant([[1], [2]], dtype=tf.float32)
        padded_tensor = self.fix_length_layer(tensor_to_pad)
        expected_result = [[1, 0], [2, 0]]
        output_diff = calc_sum_of_abs_diff(padded_tensor, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_same_length(self):
        tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        result_tensor = self.fix_length_layer(tensor)
        expected_result = [[1, 2], [3, 4]]
        output_diff = calc_sum_of_abs_diff(result_tensor, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
