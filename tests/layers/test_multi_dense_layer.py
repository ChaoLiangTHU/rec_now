# coding=utf-8
''' 2021_11_30 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.multi_dense_layer import MultiDenseLayer


class TestMultiDenseLayer(unittest.TestCase):
    def test_multi_dense_layer_2d(self):
        """测试输入为2维的情形
        """
        tf.random.set_seed(1)
        batch_size = 2
        num_dnn = 3
        dim_in = 4
        dim_out = 1
        input_shape = [batch_size, dim_in]
        input_value = tf.random_normal_initializer()(shape=input_shape, dtype=tf.float32)
        input = tf.Variable(initial_value=input_value, dtype=tf.float32)
        dense_layer = MultiDenseLayer(dim_out, num_dnn)
        result = dense_layer(input)
        expected_result = [[[-0.03584324], [-0.004118]],
                           [[-0.03538369], [-0.00670193]],
                           [[0.0174423], [-0.01468011]]]
        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_multi_dense_layer_3d(self):
        """测试输入为3维的情形
        """
        tf.random.set_seed(1)
        batch_size = 2
        num_dnn = 3
        dim_in = 4
        dim_out = 1
        input_shape = [num_dnn, batch_size, dim_in]
        input_value = tf.random_normal_initializer()(shape=input_shape, dtype=tf.float32)
        input = tf.Variable(initial_value=input_value, dtype=tf.float32)
        dense_layer = MultiDenseLayer(dim_out, num_dnn)
        result = dense_layer(input)
        expected_result = [[[-0.03584324], [-0.004118]],
                           [[-0.0016742], [0.00547126]],
                           [[0.00360582], [-0.03530192]]]
        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_multi_dense_layer_wrong_3d(self):
        """测试输入为3维，但与设置的DNN数量不同的情形
        """
        tf.random.set_seed(1)
        batch_size = 2
        num_dnn = 3
        dim_in = 4
        dim_out = 1
        wrong_input_dnn_num = num_dnn + 1
        input_shape = [wrong_input_dnn_num, batch_size, dim_in]
        input_value = tf.random_normal_initializer()(shape=input_shape, dtype=tf.float32)
        input = tf.Variable(initial_value=input_value, dtype=tf.float32)
        dense_layer = MultiDenseLayer(dim_out, num_dnn)
        with self.assertRaises(InvalidArgumentError) as context:
            dense_layer(input)
            exception_str = str(context.exception)
            self.assertTrue('[4,2,4] vs. [3,4,1]' in exception_str)


if __name__ == '__main__':
    unittest.main()
