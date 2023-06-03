# coding=utf-8
''' 2023_05_26 lcreg163@163.com

'''

import warnings
import numpy as np
import tensorflow as tf


def isin(values, target_values):
    """Like np.isin.

    Usage:
    >>> mat = [[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]]
    >>> target_values = [1, 3, 5, 7, 9]
    >>> print(isin(mat, target_values).numpy())
        [[False  True  False True  False]
         [True   False True  False True ]]
    Args:
        values (tf.Tensor): a Tensor
        target_values (list): target values to mask as True, other values in "values" are set to False

    Returns:
        (tf.Tensor): a tf.bool tensor, only positions corresponding to target_values are set to True.
    """
    values = tf.convert_to_tensor(values)
    keys_tensor = tf.constant(target_values, dtype=values.dtype)
    vals_tensor = tf.ones_like(keys_tensor, dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=0)
    r = table.lookup(values) > 0
    return r


def mask_values(values, target_values, padding_value=0):
    """Keep values in target values, and set other values to padding_value.

    Usage:
    >>> mat = [[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]]
    >>> target_values = [1, 3, 5, 7, 9]
    >>> print(mask_values(mat, target_values, padding_value=-1).numpy())
        [[-1  1 -1  3 -1]
         [ 5 -1  7 -1  9]]
    """
    mask = isin(values, target_values)
    return tf.where(mask, values, padding_value)


def first_occurance_in_row(mat, need_sort=False, padding_value=0):
    """Keep only the first occurance of a number in each row.

    Usage:
    >>> mat = [[0, 1, 1, 2, 3, 3],
               [1, 3, 3, 2, 5, 5]]
    >>> print(first_occurance_in_row(mat, padding_value=-1).numpy())
        [[ 0  1 -1  2  3 -1]
         [ 1  3 -1  2  5 -1]]

    Args:
        mat (tf.Tensor): 2D digital tensor
        need_sort (bool, optional): If same number are not adjacent in a row, need set to True. Defaults to False.
        padding_value (number, optional): Repeated numbers are set to padding_value. Defaults to 0.

    Returns:
        (tf.Tensor): A matrix with only the first repeated number keeped.
    """
    mat = tf.convert_to_tensor(mat)
    if mat.shape.rank != 2:
        raise ValueError(f"mat must be 2D tensor, get {mat.shape.rank}D tensor")
    if need_sort:
        mat = tf.sort(mat)
    right = mat[:, 1:]
    mask = mat[:, :-1] != mat[:, 1:]

    left = mat[:, 0:1]
    right = tf.where(mask, right, tf.constant(padding_value, dtype=right.dtype))
    result = tf.concat([left, right], axis=-1)
    return result


def batch_segment_ids_of_targets(slots, target_slots):
    """Get batch segment_ids for target slots in slots.

    If a slot is not in target_slots, its segment_id is set to -1.
    The results are used for later unsorted_segment_sum, unsorted_segment_mean, etc.

    Usage:
    >>> slots = [[0, 1, 1, 2, 3, 3],
                 [1, 3, 3, 2, 5, 5]]
    >>> target_slots = [1, 3, 5]
    >>> batch_segment_ids, num_rows, num_ids, num_segments = batch_segment_ids_of_targets(slots, target_slots)
    >>> print(batch_segment_ids.numpy())
        [[-1  0  0 -1  1  1]
         [ 3  4  4 -1  5  5]]
    >>> print(num_rows.numpy())
        2
    >>> print(num_ids)
        3
    >>> print(num_segments.numpy())
        6

    Args:
        slots (tf.Tensor): int or string 2D tensor.
        target_slots (list[int or string]): target slot to extract.

    Returns:
        batch_segment_ids (tf.Tensor): batch segment_ids.
        num_rows (tf.Tensor): shape is [], it's the batch_size of slots
        num_ids (int): len(target_slots), it's # of target slots in each row
        num_segments (tf.Tensor): it's the total number of segments, euqals num_rows*num_ids, shape is []
    """
    slots = tf.convert_to_tensor(slots)
    if not isinstance(target_slots, list):
        target_slots = list(target_slots)

    keys_tensor = tf.constant(target_slots, dtype=slots.dtype)
    vals_tensor = tf.constant(list(range(len(target_slots))), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1)
    segment_ids = table.lookup(slots)

    # make row-different segment slots
    num_rows = tf.shape(slots)[0]
    num_ids = len(target_slots)
    row_shift = num_ids * tf.reshape(tf.range(num_rows, dtype=segment_ids.dtype), [-1, 1])
    row_shift = row_shift * tf.cast(segment_ids >= 0, dtype=segment_ids.dtype)
    batch_segment_ids = segment_ids + row_shift
    num_segments = num_ids * num_rows
    return batch_segment_ids, num_rows, num_ids, num_segments


def sparse_batch_segment_ids_of_targets(slots, target_slots):
    """Get batch segment_ids for target slots in slots.

    The results are used for later unsorted_segment_sum, unsorted_segment_mean, etc.
    The difference between sparse_batch_segment_ids_of_targets and batch_segment_ids is that sparse_batch_segment_ids_of_targets
    return sparse batch segment ids, which may be faster.
    All returned values of sparse_batch_segment_ids_of_targets can be easied comupted using results of batch_segment_ids as below:
      mask = segment_ids >= 0
      sp_segment_ids = tf.boolean_mask(segment_ids, mask)

    Usage:
    >>> slots = [[0, 1, 1, 2, 3, 3],
                 [1, 3, 3, 2, 5, 5]]
    >>> target_slots = [1, 3, 5]
    >>> mask, sp_segment_ids, num_rows, num_ids, num_segments = sparse_batch_segment_ids_of_targets(slots, target_slots)
    >>> print(mask.numpy())
        [[False  True  True False  True  True]
         [ True  True  True False  True  True]]
    >>> print(sp_segment_ids.numpy())
        [0 0 1 1 3 4 4 5 5]
    >>> print(num_rows.numpy())
        2
    >>> print(num_ids)
        3
    >>> print(num_segments.numpy())
        6

    Args:
        slots (tf.Tensor): int or string 2D tensor.
        target_slots (list[int or string]): target slot to extract.

    Returns:
        mask (tf.Tensor): a tf.bool tensor, same shape as slots, corresponding values in target_slots are set True.
        sp_segment_ids (tf.Tensor): batch segment ids, 1D tensor, size is #true values in mask.
        num_rows (tf.Tensor): shape is [], it's the batch_size of slots
        num_ids (int): len(target_slots), it's # of target slots in each row
        num_segments (tf.Tensor): it's the total number of segments, euqals num_rows*num_ids, shape is []
    """
    slots = tf.convert_to_tensor(slots)
    if not isinstance(target_slots, list):
        target_slots = list(target_slots)

    keys_tensor = tf.constant(target_slots, dtype=slots.dtype)
    vals_tensor = tf.constant(list(range(len(target_slots))), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1)
    segment_ids = table.lookup(slots)

    mask = segment_ids >= 0

    num_rows = tf.shape(slots)[0]
    num_ids = len(target_slots)
    sp_segment_ids = tf.boolean_mask(segment_ids, mask)
    row_indice = tf.where(mask)[:, 0]
    row_indice = tf.cast(row_indice, dtype=sp_segment_ids.dtype)
    sp_segment_ids = row_indice * num_ids + sp_segment_ids
    num_segments = num_rows * num_ids
    return mask, sp_segment_ids, num_rows, num_ids, num_segments


def embedding_using_batch_segment_ids(embedding_func, slots, target_slots, ids, weights=None):
    """ Use faster version embedding_using_sparse_batch_segment_ids instead.
    """
    batch_segment_ids, num_rows, num_ids, num_segments = batch_segment_ids_of_targets(slots, target_slots)

    ids = tf.where(batch_segment_ids >= 0, ids, tf.zeros_like(ids, dtype=ids.dtype))
    unique_ids, flat_indices = tf.unique(tf.reshape(ids, [-1]))
    indices = tf.reshape(flat_indices, tf.shape(ids))

    unique_embeddings = embedding_func(unique_ids)
    embeddings = tf.gather(params=unique_embeddings, indices=indices)

    if weights is not None:
        embeddings = embeddings * tf.expand_dims(weights, -1)

    results = tf.math.unsorted_segment_sum(data=embeddings, segment_ids=batch_segment_ids, num_segments=num_segments)
    results = tf.reshape(results, [num_rows, num_ids, -1])
    return results


def embedding_using_sparse_batch_segment_ids_v1(embedding_func, slots, target_slots, ids, weights=None):
    """ Use faster version embedding_using_sparse_batch_segment_ids instead.
    """
    batch_segment_ids, num_rows, num_ids, num_segments = batch_segment_ids_of_targets(slots, target_slots)
    mask = batch_segment_ids >= 0
    sp_segment_ids = tf.boolean_mask(batch_segment_ids, mask)
    sp_ids = tf.boolean_mask(ids, mask)
    unique_ids, flat_indices = tf.unique(sp_ids)

    unique_embeddings = embedding_func(unique_ids)
    embeddings = tf.gather(params=unique_embeddings, indices=flat_indices)

    if weights is not None:
        sp_weights = tf.boolean_mask(weights, mask)
        embeddings = embeddings * tf.expand_dims(sp_weights, -1)

    results = tf.math.unsorted_segment_sum(data=embeddings, segment_ids=sp_segment_ids, num_segments=num_segments)
    results = tf.reshape(results, [num_rows, num_ids, -1])
    return results


def embedding_using_sparse_batch_segment_ids(embedding_func, slots, target_slots, ids, weights=None, method='sum', use_unique=True):
    """Embed ids and pooling by the same slot.

    Symbols:
        B: batch_size
        T: # target slots
        C: # colomns
        D: embeding dimension

    Usage:
    >>> # assume we have 4 slots and each have 10 different keys
    >>> num_slots = 4
    >>> num_keys_per_slot = 10

    >>> # lookup table
    >>> params = [[i, -i] for i in range(num_slots * num_keys_per_slot)]
    >>> params = tf.convert_to_tensor(params, dtype=tf.float32)
    >>> def embedding_func(ids): return tf.nn.embedding_lookup(params, ids=ids)

    >>> target_slots = [1, 3]

    >>> ids = [[0, 10, 20, 30],
               [21, 30, 31, 1]]
    >>> ids = tf.convert_to_tensor(ids)

    >>> # assume ids = slot * 10 + index_in_slot
    >>> slots = (tf.cast(ids, tf.float64) + 0.5) / 10.0
    >>> slots = tf.cast(slots, tf.int32)
    >>> print(slots.numpy())
        [[0 1 2 3]
        [2 3 3 0]]

    >>> weights = tf.cast(ids, dtype=tf.float32) * 10.0
    >>> pooled_embedding = embedding_using_sparse_batch_segment_ids(
    >>>     embedding_func, slots, target_slots, ids, weights=weights)
    >>> print(pooled_embedding.numpy())
        [[[  1000.  -1000.]
        [  9000.  -9000.]]

        [[     0.      0.]
        [ 18610. -18610.]]]

    >>> weights = None
    >>> pooled_embedding = embedding_using_sparse_batch_segment_ids(
    >>>     embedding_func, slots, target_slots, ids, weights=weights)
    >>> print(pooled_embedding.numpy())
        [[[ 10. -10.]
        [ 30. -30.]]

        [[  0.   0.]
        [ 61. -61.]]]

    Args:
        embedding_func (function): a tf function that map ids to embeddings.
        slots (tf.Tensor): slots of ids. shape is [B, C]
        target_slots (list[int]): slots to embed and pool.
        ids (tf.Tensor): ids look up embeddings. shape is [B, C]
        weights (tf.Tensor, optional): weigths of ids, shape is [B, C]. Defaults to None.
        method (str, optional): 'sum' for sum pooling and 'mean' for mean pooling. Defaults to 'sum'.
        use_unique (bool, optional): use unique to speed up if set to True. Defaults to True.

    Returns:
        pooled_embedding (tf.Tensor): _description_
    """
    mask, sp_segment_ids, num_rows, num_ids, num_segments = sparse_batch_segment_ids_of_targets(slots, target_slots)
    sp_ids = tf.boolean_mask(ids, mask)
    if use_unique:
        unique_ids, flat_indices = tf.unique(sp_ids)
        # # example of embedding_func
        # def embedding_func(ids): tf.nn.embedding_lookup(
        #     params=tf.random([100, 16], dtype=tf.float32), ids=unique_ids)
        unique_embeddings = embedding_func(unique_ids)
        embeddings = tf.gather(params=unique_embeddings, indices=flat_indices)
    else:
        embeddings = embedding_func(sp_ids)

    if weights is not None:
        sp_weights = tf.boolean_mask(weights, mask)
        embeddings = embeddings * tf.expand_dims(sp_weights, -1)

    if method == 'mean':
        results = tf.math.unsorted_segment_mean(data=embeddings, segment_ids=sp_segment_ids, num_segments=num_segments)
    elif method == 'sum':
        results = tf.math.unsorted_segment_sum(data=embeddings, segment_ids=sp_segment_ids, num_segments=num_segments)
    pooled_embedding = tf.reshape(results, [num_rows, num_ids, -1])  # (B, T, D)
    return pooled_embedding


def embedding_single_slot(embedding_func, slots, target_slot, ids, weights=None, default_weight=0, ncols=None, use_unique=True):
    """Get a slot's embedding without pooling.

    Usage:
    >>> # assume we have 4 slots and each have 10 different keys
    >>> num_slots = 4
    >>> num_keys_per_slot = 10
    >>> # lookup table
    >>> params = [[i, -i] for i in range(num_slots * num_keys_per_slot)]
    >>> params = tf.convert_to_tensor(params, dtype=tf.float32)
    >>> def embedding_func(ids): return tf.nn.embedding_lookup(params, ids=ids)
    >>> ids = [[0, 10, 10, 30],
    >>>        [21, 22, 31, 1]]
    >>> ids = tf.convert_to_tensor(ids, dtype=tf.int64)
    >>> # assume ids = slot * 10 + index_in_slot
    >>> slots = (tf.cast(ids, tf.float64) + 0.5) / 10.0
    >>> slots = tf.cast(slots, tf.int32)
    >>> target_slot = 2
    >>> weights = tf.cast(ids, dtype=tf.float32) * 10.0
    >>> embedding_tensor, weights_tensor, mask_tensor = embedding_single_slot(
    >>>     embedding_func, slots, target_slot, ids, weights)
    >>> print(embedding_tensor.numpy())
        [[[  0.   0.]
          [  0.   0.]]

         [[ 21. -21.]
          [ 22. -22.]]]
    >>> print(weights_tensor.numpy())
        [[[  0.]
          [  0.]]

         [[210.]
          [220.]]]
    >>> print(mask_tensor.numpy())
        [[[False]
          [False]]

         [[ True]
          [ True]]]

    Symbols:
        B: batch_size
        T: # target slots
        C: # colomns
        D: embeding dimension

    Args:
        embedding_func (function): a tf function that map ids to embeddings.
        slots (tf.Tensor): slots of ids. shape is [B, C]
        target_slot (int): slot to embed
        ids (tf.Tensor): ids look up embeddings. shape is [B, C]
        weights (tf.Tensor, optional): weigths of ids, shape is [B, C]. Defaults to None.
        default_weight (float, optional): default_weight for weights when a training sample do not have ncols target_slot. Defaults to 0.
        ncols (int, optional): ids of target_slot is padded (or truncated) to ncols. Defaults to None.
        use_unique (bool, optional): use unique to speed up if set to True. Defaults to True.

    Returns:
        embedding_tensor (tf.Tensor): embedding tensor, shape is [B, ncols, D]
        weights_tensor (tf.Tensor): weights, shape is [B, ncols, 1]
        mask_tensor (tf.Tensor): mask tensor, false for padding values, shape is [B, ncols, 1]
    """
    mask = slots == target_slot
    row_ids = tf.where(mask)[:, 0]
    nrows = tf.shape(slots)[0]

    sp_ids = tf.boolean_mask(ids, mask)

    if use_unique:
        unique_ids, flat_indices = tf.unique(sp_ids)
        unique_embeddings = embedding_func(unique_ids)
        embeddings = tf.gather(params=unique_embeddings, indices=flat_indices)
    else:
        embeddings = embedding_func(sp_ids)
    embedding_tensor = tf.RaggedTensor.from_value_rowids(embeddings, row_ids).to_tensor(
        default_value=0, shape=(nrows, ncols, embeddings.shape[-1]))
    embedding_tensor.set_shape((None, ncols, embeddings.shape[-1]))

    if weights is not None:
        sp_weights = tf.boolean_mask(weights, mask)
        weights_tensor = tf.RaggedTensor.from_value_rowids(sp_weights, row_ids)
        weights_tensor = weights_tensor.to_tensor(default_value=default_weight, shape=(nrows, ncols))
        weights_tensor.set_shape((None, ncols))
        weights_tensor = tf.expand_dims(weights_tensor, axis=-1)

    else:
        weights_tensor = None
    mask_tensor = tf.RaggedTensor.from_value_rowids(tf.ones_like(row_ids, dtype=tf.bool), row_ids)
    mask_tensor = mask_tensor.to_tensor(default_value=False, shape=(nrows, ncols))
    mask_tensor = tf.expand_dims(mask_tensor, axis=-1)
    return embedding_tensor, weights_tensor, mask_tensor


def pool_slots(slots, target_slots, ids=None, weights=None, method='sum', drop_duplicate_slot=False):
    """Fetch a list of slots's ids and weights at one time. ids and weights are pooled using mean and sum, respectively.

    Usage:
    >>> slots = [[1, 2, 3, 0, 0],
                 [2, 2, 4, 5, 0]]
    >>> slots = tf.constant(slots)
    >>> ids = [[0, 0, 0, 0, 0],
               [8, 0, 0, 0, 0]]  # for slot 2, we have keys 28, 20
    >>> ids = slots * 10 + tf.constant(ids)
    >>> weights = tf.cast(slots, dtype=tf.float32) * 0.1

    >>> target_slots = [2, 3]
    >>> pooled_ids, pooled_weights = pool_slots(slots, target_slots, ids, weights, drop_duplicate_slot=False)
    >>> # drop_duplicate_slot=False, we get 20 (the mininum id) for slot 2 in 2nd row. we get 0 for slot 3 in 2nd row.
    >>> print(pooled_ids.numpy())
        [[20 30]
         [20  0]]
    >>> print(pooled_weights.numpy())
        [[0.2 0.3]
         [0.4 0. ]]

    >>> pooled_ids, pooled_weights = pool_slots(slots, target_slots, ids, weights, drop_duplicate_slot=True)
    >>> # drop_duplicate_slot=True, we get 28 (the first occurance) for slot 2 in 2nd row.
    >>> print(pooled_ids.numpy())
        [[20 30]
         [28  0]]

    Args:
        slots (tf.Tensor): slot matrix
        target_slots (list[int]): slots to pool
        ids (tf.Tensor, optional): ids to pool, same shape as slots. Defaults to None.
        weights (tf.Tensor, optional): weights to pool, same shape as slots. Defaults to None.
        method (str, optional): pooling weights method, 'sum' or 'mean'. Defaults to 'sum'.
        drop_duplicate_slot (bool, optional): For a given slot, whether only use the first occurance of that slot in a sample.
                                             Defaults to False.
    Returns:
        pooled_ids(tf.Tensor or None):  shape is [B, len(target_slots)], B is batch_size
        pooled_weights(tf.Tensor or None): shape is [B, len(target_slots)], B is batch_size
    """
    target_slots = list(target_slots)

    slots = tf.convert_to_tensor(slots)
    if slots.shape.rank == 1:
        slots = tf.reshape(slots, [1, -1])
    if slots.shape.rank != 2:
        raise ValueError(f"only support 2 (or 1) dimentional slots, get {slots.shape.rank}")

    batch_segment_ids, num_rows, num_ids, num_segments = batch_segment_ids_of_targets(slots, target_slots)

    def _pooling_func(values, batch_segment_ids, method='sum', need_deduplicate=False):
        if values is None:
            return None

        if need_deduplicate:
            batch_segment_ids = first_occurance_in_row(batch_segment_ids, need_sort=False, padding_value=-1)

        if method == 'min0':
            results = tf.math.unsorted_segment_min(values, batch_segment_ids, num_segments=num_segments)
            results = tf.where(results != results.dtype.max, results, 0)
        elif method == 'mean':
            results = tf.math.unsorted_segment_mean(values, batch_segment_ids, num_segments=num_segments)
        elif method == 'sum':
            results = tf.math.unsorted_segment_sum(values, batch_segment_ids, num_segments=num_segments)
        else:
            raise ValueError(f"not support '{method}'")
        results = tf.reshape(results, [num_rows, num_ids])
        return results
    pooled_ids = _pooling_func(ids, batch_segment_ids, 'min0', drop_duplicate_slot)
    pooled_weights = _pooling_func(weights, batch_segment_ids, method, drop_duplicate_slot)
    return pooled_ids, pooled_weights


def pool_single_slot(slots, target_slot, ids=None, weights=None):
    """Pool a slot's key & value, only work for slot that occur exactly once a sample.

    Usage:
    >>> slots = [[1, 2, 3],
                 [2, 3, 4]]
    >>> slots = tf.constant(slots)
    >>> ids = slots * 10
    >>> weights = tf.cast(slots, dtype=tf.float32) * 0.1
    >>> target_slot = 2
    >>> pooled_ids, pooled_weights = pool_single_slot(slots, target_slot, ids, weights)
    >>> print(pooled_ids.numpy())
        [[20],
         [20]]
    >>> print(pooled_weights.numpy())
        [[0.2],
         [0.2]]

    Args:
        slots (tf.Tensor): slot matrix
        target_slots (int): slot to pool
        ids (tf.Tensor, optional): ids to pool, same shape as slots. Defaults to None.
        weights (tf.Tensor, optional): weights to pool, same shape as slots. Defaults to None.

    Returns:
        key (tf.Tensor): shape is [None, 1]
        value (tf.Tensor): shape is [None, 1]
    """
    warnings.warn("pool_single_slot only work for slot that occur exactly once a sample, use fetch_single_slot instead")
    mask = tf.equal(slots, target_slot)

    def fetch_func(values):
        if values is None:
            return None
        masked_values = tf.boolean_mask(values, mask)
        return tf.reshape(masked_values, [-1, 1])
    return fetch_func(ids), fetch_func(weights)


def fetch_single_slot(slots, target_slot, ids=None, weights=None, default_id=0, default_weight=0, ncols=None):
    """Fetch a slot's ids and weights and pad (or truncate) to ncols.

    Usage:
    >>> ids = [[0, 10],
    >>>        [10, 20],
    >>>        [20, 21]]
    >>> ids = tf.convert_to_tensor(ids, dtype=tf.int64)
    >>> # assume ids = slot * 10 + index_in_slot
    >>> slots = (tf.cast(ids, tf.float64) + 0.5) / 10.0
    >>> slots = tf.cast(slots, tf.int32)
    >>> target_slot = 2
    >>> weights = tf.cast(ids, dtype=tf.float32) * 10.0
    >>> slot_ids, slot_weights = fetch_single_slot(slots, target_slot, ids, weights, default_id=0, ncols=None)
    >>> print(slot_ids.numpy())
        [[ 0  0]
         [20  0]
         [20 21]]
    >>> print(slot_weights.numpy())
        [[  0.   0.]
         [200.   0.]
         [200. 210.]]

    Symbols:
        nrows: batch size of slots, ids, and weights

    Args:
        slots (tf.Tensor): slot matrix
        target_slots (int): slot to fetch
        ids (tf.Tensor, optional): all ids' matrix, same shape as slots. Defaults to None.
        weights (tf.Tensor, optional): all weights' matrix, same shape as slots. Defaults to None.
        default_id (int, optional): default_id for ids when a training sample do not have ncols target_slot. Defaults to 0.
        default_weight (float, optional): default_weight for weights when a training sample do not have ncols target_slot. Defaults to 0.
        ncols (int, optional): Result matrix is padded (or truncated) to (nrows, ncols). Defaults to None.

    Returns:
        target_ids (tf.Tensor): target ids fetched form ids when ids is not None. Shape is (nrows, ncols)
        target_weights (tf.Tensor): target weights fetched form weights when ids is not None. Shape is (nrows, ncols)
    """
    slots = tf.convert_to_tensor(slots)
    mask = slots == target_slot
    row_ids = tf.where(mask)[:, 0]
    nrows = tf.shape(slots)[0]

    def _fetch_func(values, default_value):
        if values is None:
            return None
        sp_values = tf.boolean_mask(values, mask)
        # not set nrows here (nrows must be of type tf.int64 for from_value_rowids), set shape while to_tensor
        ragged_target_values = tf.RaggedTensor.from_value_rowids(sp_values, value_rowids=row_ids, nrows=None)
        tensor = ragged_target_values.to_tensor(default_value=default_value, shape=[nrows, ncols])
        tensor.set_shape((None, ncols))
        return tensor
    return _fetch_func(ids, default_id), _fetch_func(weights, default_weight)


