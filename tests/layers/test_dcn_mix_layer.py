# coding=utf-8
''' 2021_11_01 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.dcn_mix_layer import DCNMixLayer


class TestDCNMixLayer(unittest.TestCase):
    def test_dcn_mix_layer(self):
        """测试不过滤attention score为负值的vec时的情况
        """
        tf.random.set_seed(10)
        input = tf.constant([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]], dtype=tf.float32)

        dcn_mix_layer = DCNMixLayer(dim_sub_space=3, num_layer=2, num_expert=4)
        dcn_mix_output = dcn_mix_layer(input)

        true_output = [[-0.00718718, -0.05909997, 0.04065184, 0.06140723, -0.05879733],
                       [-0.35002837, -1.4658885, 1.1511558, 1.3849638, -1.1614282]]

        output_diff = calc_sum_of_abs_diff(dcn_mix_output, true_output)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
