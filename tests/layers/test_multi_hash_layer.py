# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.multi_hash_layer import MultiHashLayer


class TestMultiHashLayer(unittest.TestCase):
    def test_multi_hash_layer(self):
        tf.random.set_seed(1)

        embedding_dim = 2

        num_bins = 10
        num_hash = 3

        inputs = [['Aa', 'Bb'], ['Cc', 'Dd'], ['Ee', 'Ff']]
        inputs = tf.constant(inputs)

        multi_hash_layer = MultiHashLayer(num_bins=num_bins, embedding_dim=embedding_dim,
                                          num_hash=num_hash, embeddings_initializer=tf.random_normal_initializer(), name='age')
        output = multi_hash_layer(inputs, combiner='concat')
        expected_result = [[[-0.05506101, 0.07728758, -0.01676805, -0.05213338,
                             0.03821649, -0.04180959],
                            [-0.01598444, 0.01866628, -0.03418445, 0.03368045,
                             -0.08921313, -0.02610026]],

                           [[0.0191822, -0.04398289, -0.03821107, -0.05186243,
                             0.05728074, 0.01030363],
                            [-0.01741772, -0.01682349, 0.05045691, 0.0618127,
                               0.01563057, 0.04971462]],

                           [[-0.01598444, 0.01866628, -0.01676805, -0.05213338,
                             0.03821649, -0.04180959],
                            [-0.05506101, 0.07728758, 0.05045691, 0.0618127,
                             -0.00986082, 0.02690467]]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

        output = multi_hash_layer(inputs, combiner='sum')
        expected_result = [[[-0.03361257, -0.01665538],
                            [-0.13938202, 0.02624647]],

                           [[0.03825187, -0.0855417],
                            [0.04866976, 0.09470383]],

                           [[0.005464, -0.07527669],
                            [-0.01446492, 0.16600496]]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_multi_hash_layer_test_pooling(self):
        tf.random.set_seed(1)

        embedding_dim = 2

        num_bins = 10
        num_hash = 3

        inputs = [['Aa', 'Bb'], ['Cc', 'Dd'], ['Ee', 'Ff']]
        inputs = tf.constant(inputs)
        weights = tf.ones_like(inputs, dtype=tf.float32) * 0.5

        multi_hash_layer = MultiHashLayer(num_bins=num_bins, embedding_dim=embedding_dim,
                                          num_hash=num_hash, embeddings_initializer=tf.random_normal_initializer(), name='age')
        output = multi_hash_layer.get_pooling(inputs, weights)

        expected_result = [[-0.08649729, 0.00479554],
                           [0.04346082, 0.00458107],
                           [-0.00450046, 0.04536413]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_multi_hash_layer_no_emb(self):
        tf.random.set_seed(1)

        embedding_dim = -1
        num_bins = 1000
        num_hash = 3

        inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
        inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)

        multi_hash_layer = MultiHashLayer(num_bins=num_bins, embedding_dim=embedding_dim,
                                          num_hash=num_hash, embeddings_initializer=tf.random_normal_initializer())
        output = multi_hash_layer(inputs, combiner='concat')
        expected_result = [[809, 954, 690, 178, 168, 578],
                           [859, 941, 233, 230, 311, 20],
                           [9, 228, 330, 245, 394, 369],
                           [374, 713, 248, 70, 185, 525],
                           [472, 521, 568, 664, 41, 462],
                           [621, 123, 902, 156, 860, 822],
                           [621, 63, 659, 926, 792, 165]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
