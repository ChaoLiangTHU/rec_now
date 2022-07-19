# coding=utf-8
''' 2021_11_29 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.senet_layer import SENETLayer


class TestSENETLayer(unittest.TestCase):
    def test_senet_layer(self):
        tf.random.set_seed(1)
        batch_size = 2
        num_field = 3
        embedding_dims = [i + 1 for i in range(num_field)]
        initializer = tf.random_normal_initializer()

        embeddings = []
        for dim in embedding_dims:
            initial_value = initializer((batch_size, dim), dtype=tf.float32)
            embedding = tf.Variable(initial_value=initial_value, dtype=tf.float32)
            embeddings.append(embedding)

        reduction_ratio = 0.3
        senet_layer = SENETLayer(reduction_ratio)
        result = senet_layer(embeddings)

        expected_result = [[-0.00147045, -0.00081997, 0.00221329, 0.00113419, 0.00100974, -0.00180815],
                           [-0.00311359, -0.00019354, 0.00409976, -0.00334057, 0.00116946, 0.00371958]]
        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
