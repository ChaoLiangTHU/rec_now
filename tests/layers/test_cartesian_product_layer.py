# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff, all_equal
from rec_now.layers.cartesian_product_layer import CartesianProductLayer


class TestCartesianProductLayer(unittest.TestCase):
    def test_cartesian_product_layer(self):
        tf.random.set_seed(1)
        input1 = [['A', 'B'], ['C', 'D']]
        input2 = [['a', 'b', 'c'], ['d', 'e', 'f']]
        input3 = [[1, 2], [3, 4]]

        layer = CartesianProductLayer(separator='-')

        inputs = [input1, input2, input3]
        inputs = [tf.convert_to_tensor(x) for x in inputs]

        # check result without invalid_pattern_list
        output = layer(inputs)
        expected_result = np.array([[b'A-a-1', b'A-a-2', b'A-b-1', b'A-b-2', b'A-c-1', b'A-c-2',
                                   b'B-a-1', b'B-a-2', b'B-b-1', b'B-b-2', b'B-c-1', b'B-c-2'],
                                    [b'C-d-3', b'C-d-4', b'C-e-3', b'C-e-4', b'C-f-3', b'C-f-4',
                                     b'D-d-3', b'D-d-4', b'D-e-3', b'D-e-4', b'D-f-3', b'D-f-4']])
        # self.assertTrue(all_equal(output, expected_result))

        # check result with invalid_pattern_list
        output = layer(inputs, invalid_pattern_list=['A', 'f', 'None'], default_result_str='')
        expected_result = np.array([[b'', b'', b'', b'', b'', b'',
                                     b'B-a-1', b'B-a-2', b'B-b-1', b'B-b-2', b'B-c-1', b'B-c-2'],
                                    [b'C-d-3', b'C-d-4', b'C-e-3', b'C-e-4', b'', b'',
                                     b'D-d-3', b'D-d-4', b'D-e-3', b'D-e-4', b'', b'']])
        self.assertTrue(all_equal(output, expected_result))

    def test_cartesian_product_layer_broadcast_to_batch(self):
        tf.random.set_seed(1)
        input1 = [['A', 'B']]  # user feature, batch_size is 1
        input2 = 'a'  # user feature, batch_size is 1
        input3 = [[1, 2], [3, 4]]  # non-user feature, 1st dim is true batch size 2

        layer = CartesianProductLayer(separator='-')

        inputs = [input1, input2, input3]
        inputs = [tf.convert_to_tensor(x) for x in inputs]

        output = layer(inputs)
        expected_result = [[b'A-a-1', b'A-a-2', b'B-a-1', b'B-a-2'],
                           [b'A-a-3', b'A-a-4', b'B-a-3', b'B-a-4']]
        self.assertTrue(all_equal(output, expected_result))

    def test_cartesian_product_layer_digits(self):
        tf.random.set_seed(1)
        input1 = [[[1], [2]], [[3], [4]]]
        input2 = [5, 6]
        input3 = [[7, 8], [9, 0]]

        layer = CartesianProductLayer(separator='')

        inputs = [input1, input2, input3]
        inputs = [tf.convert_to_tensor(x) for x in inputs]
        output = layer(inputs)

        output = tf.strings.to_number(output, out_type=tf.float32)

        expected_result = np.array([[157, 158, 257, 258],
                                    [369, 360, 469, 460]],
                                   dtype=np.float32)
        r = calc_sum_of_abs_diff(output, expected_result)
        self.assertEqual(r, 0)

    def test_cartesian_product_layer_replace_invalid_patterns(self):
        input = ["A1a-na", "B1b-", "-C1c", "na-D1d"]

        pts = []  # patterns list
        ers = []  # expected_results list
        pts.append([None, "na"])
        ers.append([b'', b'B1b-', b'-C1c', b'na-D1d'])

        pts.append([None, ""])
        ers.append([b'A1a-na', b'', b'-C1c', b'na-D1d'])

        pts.append(["", None])
        ers.append([b'A1a-na', b'B1b-', b'', b'na-D1d'])

        pts.append(["na", None])
        ers.append([b'A1a-na', b'B1b-', b'-C1c', b''])

        pts.append(["A1a|na", None])  # or condition
        ers.append([b'', b'B1b-', b'-C1c', b''])

        pts.append(["|na", None])  # or condition, one is empty string
        ers.append([b'A1a-na', b'B1b-', b'', b''])

        pts.append(["A1a|na", ""])  # or condition, two pattern
        ers.append([b'', b'', b'-C1c', b''])

        self.assertTrue(len(pts) == len(ers))
        for pt, er in zip(pts, ers):
            patterns = CartesianProductLayer._gen_invalid_pattern_conditions(pt, separator='-')
            r = input
            for p in patterns:
                r = tf.strings.regex_replace(r, p, "", replace_global=False)
            self.assertTrue(all_equal(r, er))


if __name__ == '__main__':
    unittest.main()
