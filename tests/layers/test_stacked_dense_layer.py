# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf
from tensorflow import keras


from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.stacked_dense_layer import StackedDenseLayer
from rec_now.layers.stacked_dense_layer import ParasiticStackedDenseLayer


class TestStackedDenseLayer(unittest.TestCase):
    def test_stacked_dense_layer(self):
        tf.random.set_seed(1)
        batch_size = 2
        units_in = 3
        units_out = 5
        max_scene_size = 100

        # 生成参数表
        resnet_param_size = StackedDenseLayer.get_resnet_param_size(units_in, units_out)
        resnet_kernel_initializer = StackedDenseLayer.get_resnet_kernel_initializer()
        initial_value = resnet_kernel_initializer([max_scene_size, resnet_param_size], dtype=tf.float32)
        resnet_param_lookup_table = tf.Variable(initial_value=initial_value)

        # 构建一个batch
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        inputs_scene = tf.random.uniform([batch_size], minval=0, maxval=max_scene_size, dtype=tf.int32, seed=2)
        resnet_param_this_batch = tf.nn.embedding_lookup(resnet_param_lookup_table, inputs_scene)

        # run this batch
        layer = StackedDenseLayer(units_out)
        outputs = layer(inputs, resnet_param_this_batch)

        # test result
        expected_result = [[-0.0108437, 0.06807042, 0.05824887, 0.01455763, -0.01269773],
                           [0.14119211, 0.8420988, 0.3796606, 0.27883598, 0.05301704]]
        output_diff = calc_sum_of_abs_diff(outputs, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


class TestParasiticStackedDenseLayer(unittest.TestCase):
    def test_parasitic_stacked_dense_layer(self):
        tf.random.set_seed(0)
        batch_size = 2
        units_in = 2
        units_out = 3

        raw_dense_layer = keras.layers.Dense(units_out)
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        parasitic_layer = ParasiticStackedDenseLayer(dense_layer=raw_dense_layer,
                                                     parasitic_kernel_initializer='Ones')
        outputs = parasitic_layer(inputs)
        # test result
        expected_result = [[0.41276646, 0.2785303, 0.75729215],
                           [1.1768938, 0.84114075, 1.7925735]]
        output_diff = calc_sum_of_abs_diff(outputs, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == "__main__":
    unittest.main()
