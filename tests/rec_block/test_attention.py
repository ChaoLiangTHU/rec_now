# coding=utf-8
''' 2021_11_01 lcreg163@163.com
注意力机制相关的函数
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.rec_block.attention import attention_by_dot_product
from rec_now.rec_block.attention import attention_by_dnn


class TestAttention(unittest.TestCase):
    def test_attention_by_dot_product(self):
        """测试不过滤attention score为负值的vec时的情况
        """
        user1_emb = [[0.1, 0.2], [-0.1, -0.2]]
        user2_emb = [[0.3, 0.4], [-0.3, -0.4]]
        user_emb = tf.constant([user1_emb, user2_emb], dtype=tf.float32)
        doc_emb = [[0.1, 0.2], [0.3, 0.4]]
        doc_emb = tf.constant(doc_emb, dtype=tf.float32)
        attn_mat, attn_score = attention_by_dot_product(user_emb, doc_emb, filter_neg=False)

        true_attn_mat = [[0.01, 0.02], [0.15, 0.2]]
        true_attn_score = [[0.], [0.]]

        attn_mat_diff = calc_sum_of_abs_diff(attn_mat, true_attn_mat)
        self.assertAlmostEqual(attn_mat_diff, 0.0, delta=1E-5)

        attn_score_diff = calc_sum_of_abs_diff(attn_score, true_attn_score)
        self.assertAlmostEqual(attn_score_diff, 0.0, delta=1E-5)

    def test_attention_by_dot_product_case2(self):
        """测试过滤attention score为负值的vec时的情况
        """
        user1_emb = [[0.1, 0.2], [-0.1, -0.2]]
        user2_emb = [[0.3, 0.4], [-0.3, -0.4]]
        user_emb = tf.constant([user1_emb, user2_emb], dtype=tf.float32)
        doc_emb = [[0.1, 0.2], [0.3, 0.4]]
        doc_emb = tf.constant(doc_emb, dtype=tf.float32)
        attn_mat, attn_score = attention_by_dot_product(user_emb, doc_emb, filter_neg=True)

        true_attn_mat = [[0.005, 0.01], [0.075, 0.1]]
        true_attn_score = [[0.05], [0.25]]

        attn_mat_diff = calc_sum_of_abs_diff(attn_mat, true_attn_mat)
        self.assertAlmostEqual(attn_mat_diff, 0.0, delta=1E-5)

        attn_score_diff = calc_sum_of_abs_diff(attn_score, true_attn_score)
        self.assertAlmostEqual(attn_score_diff, 0.0, delta=1E-5)

    def test_attention_by_dnn(self):
        """测试使用DNN生成attention权重
        """
        tf.random.set_seed(0)
        user1_emb = [[0.1, 0.2], [-0.1, -0.2]]
        user2_emb = [[0.3, 0.4], [-0.3, -0.4]]
        user_emb = tf.constant([user1_emb, user2_emb], dtype=tf.float32)
        doc_emb = [[0.1, 0.2], [0.3, 0.4]]
        doc_emb = tf.constant(doc_emb, dtype=tf.float32)

        attn_mat, attn_score, _ = attention_by_dnn(user_emb, doc_emb, dnn_dims=[32, 24, 1])

        true_attn_mat = [[0.00044473, 0.00088945], [0.00321232, 0.0042831]]
        true_attn_score = [[0.9462962], [0.8750266]]

        attn_mat_diff = calc_sum_of_abs_diff(attn_mat, true_attn_mat)
        self.assertAlmostEqual(attn_mat_diff, 0.0, delta=1E-5)

        attn_score_diff = calc_sum_of_abs_diff(attn_score, true_attn_score)
        self.assertAlmostEqual(attn_score_diff, 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
