# coding=utf-8
''' 2021_10_15 lcreg163@163.com

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf
from tensorflow import keras

from rec_now.util.numpy_tools import calc_sum_of_abs_diff
from rec_now.layers.star_dense_layer import StarDenseLayer
from rec_now.layers.star_dense_layer import ParasiticStarDenseLayer


class TestStarDenseLayer(unittest.TestCase):
    def test_star_dense_layer(self):
        tf.random.set_seed(1)
        batch_size = 2
        units_in = 3
        units_out = 5
        max_scene_size = 100

        # 生成参数表
        starnet_param_size = StarDenseLayer.get_starnet_param_size(units_in, units_out)
        starnet_kernel_initializer = StarDenseLayer.get_starnet_kernel_initializer()
        initial_value = starnet_kernel_initializer([max_scene_size, starnet_param_size], dtype=tf.float32)
        starnet_param_lookup_table = tf.Variable(initial_value=initial_value)

        # 构建一个batch
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        inputs_scene = tf.random.uniform([batch_size], minval=0, maxval=max_scene_size, dtype=tf.int32, seed=2)
        starnet_param_this_batch = tf.nn.embedding_lookup(starnet_param_lookup_table, inputs_scene)

        # run this batch
        layer = StarDenseLayer(units_out)
        outputs = layer(inputs, starnet_param_this_batch)

        # test result
        expected_result = [[-0.0108437, 0.06807042, 0.05824887, 0.01455763, -0.01269773],
                           [0.14119211, 0.8420988, 0.3796606, 0.27883598, 0.05301704]]
        output_diff = calc_sum_of_abs_diff(outputs, expected_result)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)


class TestParasiticStarDenseLayer(unittest.TestCase):
    def test_parasitic_star_dense_layer_with_multi_groups(self):
        tf.random.set_seed(1)
        batch_size = 2
        units_in = 3
        units_out = 4
        num_groups = 5

        dense_layer = keras.layers.Dense(units_out)
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        layer = ParasiticStarDenseLayer(dense_layer=dense_layer,
                                        parasitic_kernel_initializer='Ones',
                                        num_groups=num_groups)

        # test group 0
        group_idx0 = tf.constant(0, dtype=tf.int32)
        outputs0 = layer(inputs, group_idx0)
        expected_result0 = [[-0.02065258, 0.0599786, 0.04785775, 0.00602703],
                            [-0.24781615, 0.4868825, 0.78814316, 0.0116475]]
        output_diff = calc_sum_of_abs_diff(outputs0, expected_result0)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

        # test group 1
        group_idx1 = tf.constant(1, dtype=tf.int32)
        outputs1 = layer(inputs, group_idx1)
        expected_result1 = expected_result0
        output_diff = calc_sum_of_abs_diff(outputs1, expected_result1)
        self.assertAlmostEqual(output_diff, 0.0, delta=1E-5)

    def test_parasitic_star_dense_layer_gradients(self):
        tf.random.set_seed(1)
        batch_size = 2
        units_in = 3
        units_out = 1
        num_groups = 2
        num_train_loop = 3
        dense_layer = keras.layers.Dense(units_out)
        inputs = tf.random.uniform([batch_size, units_in], minval=0, maxval=1, dtype=tf.float32, seed=1)
        labels = tf.ones([batch_size, units_out])
        layer = ParasiticStarDenseLayer(dense_layer=dense_layer,
                                        parasitic_kernel_initializer='Ones',
                                        num_groups=num_groups)
        optimizer = tf.keras.optimizers.Adam(0.005)
        for _ in range(num_train_loop):
            with tf.GradientTape() as tape:
                predicts = layer(inputs, group_idx=1, stop_trunk_grad=True)
                loss = tf.reduce_sum(tf.square(predicts - labels), axis=1)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, layer.trainable_weights)
            optimizer.apply_gradients(zip(grads, layer.trainable_weights))

        expected_parastic_kernel = [[[1.], [1.], [1.]],
                                    [[0.9850103], [1.0149864], [1.0149883]]]
        parastic_kernel_diff = calc_sum_of_abs_diff(layer.parasitic_kernel.numpy(), expected_parastic_kernel)
        self.assertAlmostEqual(parastic_kernel_diff, 0.0, delta=1E-5)

        expected_parastic_bias = [[0.], [0.01499427]]
        parastic_bias_diff = calc_sum_of_abs_diff(layer.parasitic_bias.numpy(), expected_parastic_bias)
        self.assertAlmostEqual(parastic_bias_diff, 0.0, delta=1E-5)

        self.assertAlmostEqual(loss.numpy(), 0.56336486, delta=1E-5)


if __name__ == "__main__":
    unittest.main()
