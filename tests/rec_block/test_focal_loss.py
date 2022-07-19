# coding=utf-8
''' 2021_10_20 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from rec_now.rec_block.focal_loss import focal_crossentropy_loss


class TestPairwiseLossFromBatch(unittest.TestCase):
    def test_focal_crossentropy(self):
        labels = tf.constant([1, 1, 0, 0], dtype=tf.float32)
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.constant([0.9, 0.8, 0.7, 0.6], dtype=tf.float32)
        logits = tf.reshape(logits, [-1, 1])

        focal_losses = focal_crossentropy_loss(labels, logits, alpha=None, gamma=None, return_mean=True)
        self.assertAlmostEqual(0.71323216, focal_losses.numpy(), delta=1E-5)
        focal_losses = focal_crossentropy_loss(labels, logits, alpha=0.25, gamma=None, return_mean=True)
        self.assertAlmostEqual(0.44589227, focal_losses.numpy(), delta=1E-5)
        focal_losses = focal_crossentropy_loss(labels, logits, alpha=None, gamma=1, return_mean=True)
        self.assertAlmostEqual(0.40516436, focal_losses.numpy(), delta=1E-5)


if __name__ == '__main__':
    unittest.main()
