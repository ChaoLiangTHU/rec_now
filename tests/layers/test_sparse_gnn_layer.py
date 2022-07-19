# coding=utf-8
''' 2022_01_18 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.sparse_gnn_layer import SparseGNNLayer


class TestSparseGNNLayerLayer(unittest.TestCase):
    def test_sparse_gnn_layer(self):
        tf.random.set_seed(1)
        batch_size = 2
        num_fields = 3
        dim_in = 4
        input_shape = [batch_size, num_fields, dim_in]
        input_value = tf.random_normal_initializer()(shape=input_shape, dtype=tf.float32)
        input = tf.Variable(initial_value=input_value, dtype=tf.float32)

        fields = [0, 1, 2]
        field2neighbors = {
            0: [2],
            1: [2, 0]
        }

        gnn_layer = SparseGNNLayer(fields=fields, field2neighbors=field2neighbors,
                                   num_layers=3, share_weights_between_layers=False,
                                   activation='tanh')
        expected_result = [[[-0.06320834, -0.08645303, -0.02788609],
                            [0.06614043, -0.03772061, -0.03598052],
                            [0.00979297, -0.00018102, -0.031269],
                            [-0.0545692, -0.03511687, -0.03570567]],

                           [[-0.01362271, 0.06267785, 0.01263753],
                            [-0.00716899, 0.00449803, 0.03215501],
                            [0.04116546, 0.02203641, 0.10609905],
                            [0.04295323, 0.02166578, -0.04118742]]]

        # input 为3维的情况
        result = gnn_layer(input, transpose_outputs=False, flattern_outputs=False)
        output_diff = calc_sum_of_abs_diff(result, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

        # input 为2维的情况
        input_2d = tf.reshape(input, [batch_size, -1])
        result2 = gnn_layer(input_2d, transpose_outputs=False, flattern_outputs=False)
        output_diff = calc_sum_of_abs_diff(result2, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

        # input 为embedding list的情况
        input_list = tf.split(input, num_fields, axis=1)
        input_list = [tf.squeeze(x, axis=1) for x in input_list]
        result3 = gnn_layer(input_list, transpose_outputs=False, flattern_outputs=False)
        output_diff = calc_sum_of_abs_diff(result3, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_list_of_edge_to_neighbors(self):
        list_of_edges = [(1, 2), (1, 3), (2, 3)]

        # 测试有向图模式
        edges = SparseGNNLayer.list_of_edge_to_neighbors(list_of_edges, directed=True)
        expected_result = {1: set([2, 3]),
                           2: set([3])}
        self.assertCountEqual(edges.keys(), expected_result.keys())
        for node in edges.keys():
            self.assertCountEqual(edges[node], expected_result[node])

        # 测试无向图模式
        edges = SparseGNNLayer.list_of_edge_to_neighbors(list_of_edges, directed=False)
        expected_result = {1: set([2, 3]),
                           2: set([1, 3]),
                           3: set([1, 2])}
        self.assertCountEqual(edges.keys(), expected_result.keys())
        for node in edges.keys():
            self.assertCountEqual(edges[node], expected_result[node])


if __name__ == '__main__':
    unittest.main()
