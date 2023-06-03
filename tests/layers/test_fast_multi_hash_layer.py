# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import time
import numpy as np

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.multi_hash_layer import FastMultiHashLayer, MultiHashLayer


class TestMultiHashLayer(unittest.TestCase):
    def test_fast_multi_hash_layer(self):
        tf.random.set_seed(1)

        embedding_dim = 2

        num_bins = 10
        num_hash = 3

        inputs = [['Aa', 'Bb'], ['Cc', 'Dd'], ['Ee', 'Ff']]
        inputs = tf.constant(inputs)

        multi_hash_layer = FastMultiHashLayer(num_bins=num_bins, embedding_dim=embedding_dim,
                                              num_hash=num_hash, embeddings_initializer=tf.random_normal_initializer(), name='age')
        output = multi_hash_layer(inputs, combiner='concat')

        expected_result = [[-0.03129962, -0.0357513, -0.03292613, -0.04916694, 0.00508573,
                            -0.05960856, -0.05506101, 0.07728758, -0.07800285, -0.00789563,
                            -0.00520009, -0.03755732],
                           [0.06398293, -0.00107379, 0.06124375, 0.00293248, 0.00845301,
                            0.05227221, 0.00439039, -0.01016302, 0.01944189, -0.05186224,
                            0.05924026, -0.01769078],
                           [0.00439039, -0.01016302, -0.03292613, -0.04916694, 0.00508573,
                            -0.05960856, -0.06123361, -0.04905606, 0.01944189, -0.05186224,
                            0.04603096, -0.01844464]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

        output = multi_hash_layer(inputs, combiner='sum')
        expected_result = [[[-0.05914002, -0.1445268],
                            [-0.13826394, 0.03183464]],

                           [[0.1336797, 0.0541309],
                            [0.08307254, -0.07971604]],

                           [[-0.02345001, -0.11893852],
                            [0.00423924, -0.11936294]]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_fast_multi_hash_layer_test_pooling(self):
        tf.random.set_seed(1)

        embedding_dim = 2

        num_bins = 10
        num_hash = 3

        inputs = [['Aa', 'Bb'], ['Cc', 'Dd'], ['Ee', 'Ff']]
        inputs = tf.constant(inputs)
        weights = tf.ones_like(inputs, dtype=tf.float32) * 0.5

        multi_hash_layer = FastMultiHashLayer(num_bins=num_bins, embedding_dim=embedding_dim,
                                              num_hash=num_hash, embeddings_initializer=tf.random_normal_initializer(), name='age')
        output = multi_hash_layer.get_pooling(inputs, weights)

        expected_result = [[-0.09870198, -0.05634608],
                           [0.10837612, -0.01279257],
                           [-0.00960538, -0.11915073]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_fast_multi_hash_layer_no_emb(self):
        tf.random.set_seed(1)

        embedding_dim = -1
        num_bins = 1000
        num_hash = 3

        inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
        inputs = tf.convert_to_tensor(inputs, dtype=tf.int64)

        multi_hash_layer = FastMultiHashLayer(num_bins=num_bins, embedding_dim=embedding_dim,
                                              num_hash=num_hash, embeddings_initializer=tf.random_normal_initializer())
        output = multi_hash_layer(inputs, combiner='concat')
        expected_result = [[849, 759, 690, 178, 168, 578],
                           [921, 543, 233, 230, 311, 20],
                           [971, 487, 330, 245, 394, 369],
                           [88, 627, 248, 70, 185, 525],
                           [85, 862, 568, 664, 41, 462],
                           [439, 888, 902, 156, 860, 822],
                           [843, 665, 659, 926, 792, 165]]

        output_diff = calc_sum_of_abs_diff(output, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_fast_multi_hash_layer_speed(self):
        return True
        num_bins = 25600
        num_hash = 2
        v = tf.Variable(np.random.random((num_bins * num_hash, 8)), shape=[num_bins * num_hash, 8], dtype=tf.float32)
        n = 10000
        rset = set()
        t = time.time()
        for i in range(n):
            input = [i] * 256
            input = tf.convert_to_tensor(input, dtype=tf.int32)
            input = tf.reshape(input, [-1, 1])
            if input.dtype.is_integer:
                input = tf.as_string(input)

            total_shift = 62
            one_size = total_shift // num_hash
            bit_mask = (1 << one_size) - 1
            total_bins = 1 << total_shift
            if False:
                hashed = tf.strings.to_hash_bucket_fast(input, total_bins)
                # hashed1 = tf.
                hashed1 = tf.bitwise.bitwise_and(hashed, bit_mask) % num_bins
                hashed2 = tf.bitwise.right_shift(hashed, one_size) % num_bins
                # output = bin(hashed.numpy())
                # print('output:', output, type(int(hashed.numpy())))
                # s1 = bin(hashed1.numpy())
                # s2 = bin(hashed2.numpy())
                # print(s2 + s1[2:] == output)
                # print(hashed1.numpy(), hashed2.numpy())
            else:
                hashed1 = tf.strings.to_hash_bucket_strong(input, num_bins, [1, 1])
                hashed2 = tf.strings.to_hash_bucket_strong(input, num_bins, [2, 2]) + num_bins
                # hashed1 = tf.strings.to_hash_bucket_fast(input, num_bins)
            # print(hashed1)
            total = tf.concat([hashed1, hashed2], axis=-1)
            # # print(total)
            r = tf.nn.embedding_lookup(v, total)
            # print(tf.nn.embedding_lookup(v, total))
            # r1 = tf.nn.embedding_lookup(v, hashed1)
            # r2 = tf.nn.embedding_lookup(v, hashed2)
            # rset.add((int(hashed1.numpy()), int(hashed2.numpy())))
        # print(len('0b11111110011111000001011001111101111011101001001001000010010101'))
        # print(bin(bit_mask))
        print("#collision:", n - len(rset))
        t = time.time() - t
        print(f'use: {t}s')


if __name__ == '__main__':
    unittest.main()
