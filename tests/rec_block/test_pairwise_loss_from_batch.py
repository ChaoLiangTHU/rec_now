# coding=utf-8
''' 2021_06_04 lcreg163@163.com
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.rec_block.pairwise_loss_from_batch import occurance_power_weight
from rec_now.rec_block.pairwise_loss_from_batch import bpr_loss_func
from rec_now.rec_block.pairwise_loss_from_batch import pairwise_loss


class TestPairwiseLossFromBatch(unittest.TestCase):
    def test_occurance_power_weight(self):
        group_id = [1, 1, 2, 4, 4, 4]
        weights1 = occurance_power_weight(group_id, power=-1)
        result1 = [0.5, 0.5, 1., 0.33333334, 0.33333334, 0.33333334]

        weights2 = occurance_power_weight(group_id, power=2)
        result2 = [4., 4., 1., 9., 9., 9.]

        for e, r in zip(result1, weights1.numpy()):
            self.assertAlmostEqual(e, r, delta=0.0001)

        for e, r in zip(result2, weights2.numpy()):
            self.assertAlmostEqual(e, r, delta=0.0001)

    def test_pairwise_loss(self):
        sample_group_idx_var = tf.transpose(tf.constant([[1, 1, 2, 2, 2]], dtype=tf.float32))
        logits = tf.transpose(tf.constant([[0, 1, 2, 3, 4]], dtype=tf.float32))
        label = tf.transpose(tf.constant([[1.1, 0, 0, 1, 1]], dtype=tf.float32))

        def pairwise_loss_func(outputs_pos, outputs_neg, weights):
            return bpr_loss_func(outputs_pos, outputs_neg, weights, 1.0)

        # 测试不带样本mask的版本
        pairloss = pairwise_loss(logits,
                                 label,
                                 sample_group_idx_var,
                                 pairwise_loss_func,
                                 only_use_wrong_order_pair=False,
                                 click_occurance_power=-0.5)
        self.assertAlmostEqual(pairloss.numpy(), 0.5415076, delta=1e-4)

        # 测试自定义weight
        def _label_pair_to_weight_func(label_matrix, label_matrix_transpose, **kwargs):
            weights = tf.cast(label_matrix > label_matrix_transpose, dtype=tf.float32)
            return weights

        pairloss_with_weight = pairwise_loss(logits,
                                             label,
                                             sample_group_idx_var,
                                             pairwise_loss_func,
                                             only_use_wrong_order_pair=False,
                                             click_occurance_power=-0.5,
                                             label_pair_to_weight_func=_label_pair_to_weight_func
                                             )
        self.assertAlmostEqual(pairloss_with_weight.numpy(), 0.5415076, delta=1e-4)

        # 测试带样本mask的版本
        mask = tf.transpose(tf.constant([[True, True, False, False, False]], dtype=tf.bool))
        pairloss_with_sample_mask = pairwise_loss(logits,
                                                  label,
                                                  sample_group_idx_var,
                                                  pairwise_loss_func,
                                                  only_use_wrong_order_pair=False,
                                                  click_occurance_power=-0.5,
                                                  mask=mask)
        self.assertAlmostEqual(pairloss_with_sample_mask.numpy(), 1.3132617, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
