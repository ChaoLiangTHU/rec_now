# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.fm_layer import FMLayer


class TestFMLayer(unittest.TestCase):
    def test_fm_layer(self):
        tf.random.set_seed(1)

        batch_size = 2
        embedding_dim = 5
        num_field = 7

        embeddings = []
        for field_idx in range(num_field):
            embedding = tf.random.uniform([batch_size, embedding_dim], maxval=1.0,
                                          dtype=tf.float32, seed=field_idx)
            embeddings.append(embedding)

        fm = FMLayer(name='fm')
        fm_output = fm(embeddings)
        expected_result = [[17.120375],
                           [32.70206]]

        output_diff = calc_sum_of_abs_diff(fm_output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
