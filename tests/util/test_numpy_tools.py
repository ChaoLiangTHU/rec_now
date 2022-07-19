# coding=utf-8
''' 2021_11_01 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import unittest

from rec_now.util.numpy_tools import calc_sum_of_abs_diff


class TestNumpyTools(unittest.TestCase):
    def test_calc_sum_of_abs_diff(self):
        arr1 = [1., 2.]
        arr2 = [1.1, 2.]
        result = calc_sum_of_abs_diff(arr1, arr2)
        self.assertAlmostEqual(result, 0.1, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
