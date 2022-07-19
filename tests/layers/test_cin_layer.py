# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.cin_layer import CINLayer


class TestCINLayer(unittest.TestCase):
    def test_cin_layer(self):
        tf.random.set_seed(1)

        batch_size = 2
        embedding_dim = 3
        embeddings_size = 5
        num_field = 10

        features = {}
        embedding_tables = {}
        embeddings = []
        for i in range(num_field):
            features[i] = tf.constant([idx % embeddings_size for idx in range(batch_size)], dtype=tf.int64)
            values = tf.random_normal_initializer()(shape=[embeddings_size, embedding_dim], dtype=tf.float32)
            embedding_tables[i] = tf.Variable(initial_value=values, name='emb%s' % i)
            embeddings.append(tf.nn.embedding_lookup(embedding_tables[i], features[i]))

        cin = CINLayer([2, 1], name="CIN")
        cin_output = cin(embeddings, output_input=True, sum_channel=True)

        expected_result = [[0.00754739, 0.15105832, 0.23957989],
                           [0.03649065, -0.12252036, -0.02113211]]

        output_diff = calc_sum_of_abs_diff(cin_output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
