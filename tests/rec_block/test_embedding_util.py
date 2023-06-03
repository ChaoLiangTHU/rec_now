import unittest

import tensorflow as tf

from rec_now.util.numpy_tools import calc_sum_of_abs_diff, all_equal
from rec_now.rec_block.embedding_util import (
    first_occurance_in_row, isin, mask_values, batch_segment_ids_of_targets,
    sparse_batch_segment_ids_of_targets, embedding_using_sparse_batch_segment_ids,
    embedding_single_slot,
    pool_slots, pool_single_slot, fetch_single_slot
)


class TestEmbeddingUtil(unittest.TestCase):
    def test_isin(self):
        mat = [[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]]
        target_values = [1, 3, 5, 7, 9]
        result = isin(mat, target_values).numpy()
        expected_result = [[False, True, False, True, False],
                           [True, False, True, False, True]]
        self.assertTrue(all_equal(result, expected_result))

    def test_mask_values(self):
        mat = [[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]]
        target_values = [1, 3, 5, 7, 9]
        result = mask_values(mat, target_values, padding_value=-1).numpy()
        expected_result = [[-1, 1, -1, 3, -1],
                           [5, -1, 7, -1, 9]]
        self.assertTrue(all_equal(result, expected_result))

    def test_first_occurance_in_row(self):
        mat = [[0, 1, 1, 2, 3, 3],
               [1, 3, 3, 2, 5, 5]]
        result = first_occurance_in_row(mat, padding_value=-1).numpy()
        expected_result = [[0, 1, -1, 2, 3, -1],
                           [1, 3, -1, 2, 5, -1]]
        self.assertTrue(all_equal(result, expected_result))

    def test_batch_segment_ids_of_targets(self):
        slots = [[0, 1, 1, 2, 3, 3],
                 [1, 3, 3, 2, 5, 5]]
        target_slots = [1, 3, 5]
        batch_segment_ids, num_rows, num_ids, num_segments = batch_segment_ids_of_targets(slots, target_slots)

        expected_batch_segment_ids = [[-1, 0, 0, -1, 1, 1],
                                      [3, 4, 4, -1, 5, 5]]
        self.assertTrue(all_equal(batch_segment_ids, expected_batch_segment_ids))

        self.assertTrue(num_rows == 2)
        self.assertTrue(num_ids == 3)
        self.assertTrue(num_segments == 6)

    def test_sparse_batch_segment_ids_of_targets(self):
        slots = [[0, 1, 1, 2, 3, 3],
                 [1, 3, 3, 2, 5, 5]]
        target_slots = [1, 3, 5]
        mask, sp_segment_ids, num_rows, num_ids, num_segments = sparse_batch_segment_ids_of_targets(slots, target_slots)

        expected_mask = [[False, True, True, False, True, True],
                         [True, True, True, False, True, True]]
        self.assertTrue(all_equal(mask, expected_mask))
        expected_sp_segment_ids = [0, 0, 1, 1, 3, 4, 4, 5, 5]
        self.assertTrue(all_equal(sp_segment_ids, expected_sp_segment_ids))

        self.assertTrue(num_rows == 2)
        self.assertTrue(num_ids == 3)
        self.assertTrue(num_segments == 6)

    def test_embedding_using_sparse_batch_segment_ids(self):
        # assume we have 4 slots and each have 10 different keys
        num_slots = 4
        num_keys_per_slot = 10

        # lookup table
        params = [[i, -i] for i in range(num_slots * num_keys_per_slot)]
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        def embedding_func(ids): return tf.nn.embedding_lookup(params, ids=ids)

        target_slots = [1, 3]

        ids = [[0, 10, 20, 30],
               [21, 30, 31, 1]]
        ids = tf.convert_to_tensor(ids)

        # assume ids = slot * 10 + index_in_slot
        slots = (tf.cast(ids, tf.float64) + 0.5) / 10.0
        slots = tf.cast(slots, tf.int32)

        weights = tf.cast(ids, dtype=tf.float32) * 10.0
        pooled_embedding = embedding_using_sparse_batch_segment_ids(
            embedding_func, slots, target_slots, ids, weights=weights)
        expected_results = [[[1000., -1000.],
                             [9000., -9000.]],

                            [[0., 0.],
                             [18610., -18610.]]]
        self.assertTrue(all_equal(pooled_embedding, expected_results))

        weights = None
        pooled_embedding = embedding_using_sparse_batch_segment_ids(
            embedding_func, slots, target_slots, ids, weights=weights)
        expected_results = [[[10., -10.],
                             [30., -30.]],

                            [[0., 0.],
                             [61., -61.]]],
        self.assertTrue(all_equal(pooled_embedding, expected_results))

    def test_embedding_single_slot(self):
        # assume we have 4 slots and each have 10 different keys
        num_slots = 4
        num_keys_per_slot = 10

        # lookup table
        params = [[i, -i] for i in range(num_slots * num_keys_per_slot)]
        params = tf.convert_to_tensor(params, dtype=tf.float32)
        def embedding_func(ids): return tf.nn.embedding_lookup(params, ids=ids)
        ids = [[0, 10, 10, 30],
               [21, 22, 31, 1]]
        ids = tf.convert_to_tensor(ids, dtype=tf.int64)
        # assume ids = slot * 10 + index_in_slot
        slots = (tf.cast(ids, tf.float64) + 0.5) / 10.0
        slots = tf.cast(slots, tf.int32)
        target_slot = 2
        weights = tf.cast(ids, dtype=tf.float32) * 10.0
        embedding_tensor, weights_tensor, mask_tensor = embedding_single_slot(
            embedding_func, slots, target_slot, ids, weights)
        expected_embedding_tensor = [[[0., 0.],
                                     [0., 0.]],

                                     [[21., -21.],
                                      [22., -22.]]]
        expected_weights_tensor = [[[0.],
                                   [0.]],

                                   [[210.],
                                    [220.]]]
        expected_mask_tensor = [[[False],
                                 [False]],

                                [[True],
                                 [True]]]

        self.assertAlmostEqual(calc_sum_of_abs_diff(embedding_tensor, expected_embedding_tensor), 0.0, delta=1E-5)
        self.assertAlmostEqual(calc_sum_of_abs_diff(weights_tensor, expected_weights_tensor), 0.0, delta=1E-5)
        self.assertTrue(all_equal(mask_tensor, expected_mask_tensor))

    def test_pool_slots(self):
        slots = [[1, 2, 3, 0, 0],
                 [2, 2, 4, 5, 0]]
        slots = tf.constant(slots)

        ids = [[0, 0, 0, 0, 0],
               [8, 0, 0, 0, 0]]  # for slot 2, we have keys 28, 20
        ids = slots * 10 + tf.constant(ids)

        weights = tf.cast(slots, dtype=tf.float32) * 0.1

        target_slots = [2, 3]
        pooled_ids, pooled_weights = pool_slots(
            slots, target_slots, ids, weights, drop_duplicate_slot=False)
        # drop_duplicate_slot=False, we get 20 (the mininum id) for slot 2 in 2nd row. we get 0 for slot 3 in 2nd row.
        expected_pooled_ids = [[20, 30],
                               [20, 0]]
        expected_pooled_weights = [[0.2, 0.3],
                                   [0.4, 0.]]
        self.assertTrue(all_equal(pooled_ids, expected_pooled_ids))
        self.assertAlmostEqual(calc_sum_of_abs_diff(pooled_weights, expected_pooled_weights), 0.0, delta=1E-5)

        pooled_ids, pooled_weights = pool_slots(slots, target_slots, ids, weights, drop_duplicate_slot=True)
        # drop_duplicate_slot=True, we get 28 (the first occurance) for slot 2 in 2nd row.
        expected_pooled_ids = [[20, 30],
                               [28, 0]]
        expected_pooled_weights = [[0.2, 0.3],
                                   [0.2, 0.]]
        self.assertTrue(all_equal(pooled_ids, expected_pooled_ids))
        self.assertAlmostEqual(calc_sum_of_abs_diff(pooled_weights, expected_pooled_weights), 0.0, delta=1E-5)

    def test_pool_single_slot(self):
        slots = [[1, 2, 3],
                 [2, 3, 4]]
        slots = tf.constant(slots)
        ids = slots * 10
        weights = tf.cast(slots, dtype=tf.float32) * 0.1
        target_slot = 2
        pooled_ids, pooled_weights = pool_single_slot(slots, target_slot, ids, weights)
        expected_pooled_ids = [[20],
                               [20]]
        expected_pooled_weights = [[0.2],
                                   [0.2]]
        self.assertTrue(all_equal(pooled_ids, expected_pooled_ids))
        self.assertAlmostEqual(calc_sum_of_abs_diff(pooled_weights, expected_pooled_weights), 0.0, delta=1E-5)

    def test_fetch_single_slot(self):
        ids = [[0, 10, 11, 30],
               [21, 22, 31, 1]]
        ids = tf.convert_to_tensor(ids, dtype=tf.int64)

        # assume ids = slot * 10 + index_in_slot
        slots = (tf.cast(ids, tf.float64) + 0.5) / 10.0
        slots = tf.cast(slots, tf.int32)

        target_slot = 1

        weights = tf.cast(ids, dtype=tf.float32) * 10.0

        # default id and weight are 0
        id, weight = fetch_single_slot(slots, target_slot, ids, weights, default_id=0, default_weight=0, ncols=None)
        expected_id = [[10, 11],
                       [0, 0]]
        expected_weight = [[100., 110.],
                           [0., 0.]]
        self.assertTrue(all_equal(id, expected_id))
        self.assertAlmostEqual(calc_sum_of_abs_diff(weight, expected_weight), 0.0, delta=1E-5)

        # change default id and weight
        id, weight = fetch_single_slot(slots, target_slot, ids, weights, default_id=10, default_weight=1, ncols=None)
        expected_id = [[10, 11],
                       [10, 10]]
        expected_weight = [[100., 110.],
                           [1., 1.]]
        self.assertTrue(all_equal(id, expected_id))
        self.assertAlmostEqual(calc_sum_of_abs_diff(weight, expected_weight), 0.0, delta=1E-5)


if __name__ == '__main__':
    unittest.main()
