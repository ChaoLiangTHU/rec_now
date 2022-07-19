# coding=utf-8
''' 2021_09_27 lcreg163@163.com
将各个embedding的权重变为各个embedding的各个dimension上的权重
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.rec_block.embedding_wise_weight import gather_embedding_element_wise_weight


class TestGatherEmbeddingElementWiseWeight(unittest.TestCase):
    def test_gather_embedding_element_wise_weight(self):
        tf.config.run_functions_eagerly(True)
        pos_idx = [0, 1, 1, 2, 2, 2]  # 每个sample有三个embedding，三个embedding的维度分别为1,2,3
        num_embedding = max(pos_idx) + 1
        batch_size = 4
        embedding_weights = tf.range(num_embedding * batch_size) + 10
        embedding_weights = tf.reshape(embedding_weights, [batch_size, num_embedding])
        embedding_weights = tf.cast(embedding_weights, tf.float32)
        element_wise_weight = gather_embedding_element_wise_weight(embedding_weights, pos_idx)

        expected_result = [[10, 11, 11, 12, 12, 12],
                           [13, 14, 14, 15, 15, 15],
                           [16, 17, 17, 18, 18, 18],
                           [19, 20, 20, 21, 21, 21]]
        expected_result = tf.constant(expected_result, dtype=tf.float32)
        elements_diff = (element_wise_weight - expected_result).numpy().any()
        self.assertFalse(elements_diff)


if __name__ == "__main__":
    unittest.main()
