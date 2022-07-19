# coding=utf-8
''' 2021_11_29 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.can_layer import CANLayer


class TestCANLayer(unittest.TestCase):
    def test_has_non_zero(self):
        embedding = tf.constant([[[1, 2, 0], [0, 0, 0]],
                                 [[1, 0, 0], [0, 0, 0.1]]], dtype=tf.float32)
        result = CANLayer._has_non_zero(embedding, axis=-1, keepdims=True)
        expected_result = [[[True], [False]],
                           [[True], [True]]]
        diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(diff, 0.0, delta=1E-5)

    def test_can_layer(self):
        batch_size = 2
        n_emb_per_sample = 3
        emb_dim = 4
        dnn_dims = [4, 3, 2]
        use_bias = True

        tf.random.set_seed(1)
        value = tf.random_normal_initializer()(shape=[batch_size, n_emb_per_sample, emb_dim], dtype=tf.float32)
        embedding = tf.Variable(initial_value=value, dtype=tf.float32)

        size_dnn_param = CANLayer.get_dnn_param_size(emb_dim, dnn_dims, use_bias=use_bias)
        value = tf.random_normal_initializer()(shape=[batch_size, size_dnn_param], dtype=tf.float32)
        dnn_params = tf.Variable(value, dtype=tf.float32)

        can_layer = CANLayer(dnn_dims=dnn_dims,
                             use_bias=use_bias,
                             mask_all_zero_embedding=True)
        can_output = can_layer(embedding, dnn_params)

        expected_result = [[0.06818546, 0.12346052],
                           [0.11372094, 0.23575373]]

        output_diff = calc_sum_of_abs_diff(can_output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
