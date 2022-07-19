# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.inner_pnn_layer import InnerPNNLayer


class TestInnerPNNLayer(unittest.TestCase):
    def test_inner_pnn_layer(self):
        tf.random.set_seed(1)

        batch_size = 4
        embedding_dim = 2
        num_field = 3

        embeddings = []
        for field_idx in range(num_field):
            embedding = tf.random.normal([batch_size, embedding_dim], mean=float(field_idx),
                                         dtype=tf.float32, seed=field_idx)
            embeddings.append(embedding)
        ipnn_layer = InnerPNNLayer(name='InnerPNNLayer')
        result = ipnn_layer(embeddings)

        expected_result = [[2.3125873, 1.9014311, 10.842302],
                           [1.5337583, 2.2996788, 13.424303],
                           [-0.799806, -8.9576435, 1.9083395],
                           [0.07315224, 1.5779386, 4.7213144]]

        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
