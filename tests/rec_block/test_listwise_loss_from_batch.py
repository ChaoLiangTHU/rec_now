# coding=utf-8
''' 2021_06_04 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.rec_block.listwise_loss_from_batch import to_listwise_sample
from rec_now.rec_block.listwise_loss_from_batch import nan_to_zero
from rec_now.rec_block.listwise_loss_from_batch import listwise_loss_via_softmax_cross_entropy_with_logits


class TestListwiseLoss(unittest.TestCase):
    def test_listwise_loss(self):
        """测试batch内可以组成2个有效的list的情况.
        """
        tf.config.run_functions_eagerly(True)
        sample_group_idx_var = tf.transpose(tf.constant([[1, 1, 2, 1, 2, 2, 3, 4]], dtype=tf.float32))
        labels = tf.transpose(tf.constant([[1, 1, 1, 0, 0, 0, 1, 0]], dtype=tf.float32))
        logits = tf.transpose(tf.constant([[0.1, 0.01, 0.2, 0.001, 0.02, 0.002, 0.3, 0.4]], dtype=tf.float32))

        sample_mask, labels_for_softmax, logits_for_softmax = to_listwise_sample(sample_group_idx_var, labels, logits)
        n_valid_list = tf.shape(labels_for_softmax)[0]
        n_sample_per_valid_group = tf.reduce_mean(tf.reduce_sum(tf.cast(sample_mask, tf.float32), axis=-1))
        n_sample_per_valid_group = nan_to_zero(n_sample_per_valid_group)
        listwise_loss = listwise_loss_via_softmax_cross_entropy_with_logits(labels_for_softmax=labels_for_softmax,
                                                                            logits_for_softmax=logits_for_softmax)

        self.assertEquals(n_valid_list, 2)
        self.assertAlmostEqual(listwise_loss.numpy(), 1.0291535, delta=1e-4)

    def test_listwise_loss_case2(self):
        """测试batch内可以没有有效的list的情况.
        """
        tf.config.run_functions_eagerly(True)
        sample_group_idx_var = tf.transpose(tf.constant([[3, 4]], dtype=tf.float32))
        labels = tf.transpose(tf.constant([[1, 0]], dtype=tf.float32))
        logits = tf.transpose(tf.constant([[0.3, 0.4]], dtype=tf.float32))

        sample_mask, labels_for_softmax, logits_for_softmax = to_listwise_sample(sample_group_idx_var, labels, logits)
        n_valid_list = tf.shape(labels_for_softmax)[0]
        n_sample_per_valid_group = tf.reduce_mean(tf.reduce_sum(tf.cast(sample_mask, tf.float32), axis=-1))
        n_sample_per_valid_group = nan_to_zero(n_sample_per_valid_group)
        listwise_loss = listwise_loss_via_softmax_cross_entropy_with_logits(labels_for_softmax=labels_for_softmax,
                                                                            logits_for_softmax=logits_for_softmax)
        self.assertEquals(n_valid_list, 0)
        self.assertAlmostEqual(listwise_loss.numpy(), 0.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
