# coding=utf-8
''' 20221101 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class CartesianProductLayer(tf.keras.layers.Layer):
    """CartesianProduct of inputs, each input is cast to string and then do cartesian product.

    Support replace result with default_result_str, if one of the inputs' content matches the corresponding pattern in invalid_pattern_list.

    Symbols:
        B: batch size
        L1: shape of inputs1 may be [B, L1] or [B] (L1=1), or [B, L01,L02, ... L0n] (L1=L01*L02*...*L0n) or [1, L1]
        L2: shaple of inputs2 may be [B, L2] or [B]
        ...
        Ln: shaple of inputs_n may be [B, Ln] or [B]
        D: L1*L2*...*Ln, output shape is [B, L1*L2*...*Ln]
    """

    def __init__(self, separator='-',
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """

        Args:
            num_bins (list[int]): _description_
            hash_seeds (list[int]): seeds of hash functions
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.separator = separator

    def build(self, input_shape=None):
        """generate variables used by this layer.
        """
        self.built = True
        return super().build(input_shape)

    def _get_target_dtype(self, inputs):
        TARGET_DTYPES = [tf.int32, tf.int64, tf.string]
        target_index = 0
        for input in inputs:
            dtype = input.dtype
            idx = TARGET_DTYPES.index(dtype)
            if idx > target_index:
                target_index = idx
        return TARGET_DTYPES[target_index]

    @staticmethod
    def _get_size_except_1st_dim(tensor: tf.Tensor):
        s = tensor.shape.as_list()
        ds = None
        r = 1
        for idx, d in enumerate(s):
            if idx == 0:
                continue
            if d is None or d < 0:
                if ds is None:
                    ds = tf.shape(s)
                d = ds[idx]
            r *= d
        return r

    @staticmethod
    def _get_dim(tensor: tf.Tensor, idx: int = -1):
        s = tensor.shape.as_list()
        if len(s) == 0:
            return 1
        d = s[idx]
        if isinstance(d, int) and d > 0:
            return d
        return tf.shape(tensor)[idx]

    @staticmethod
    def _prod(l: list):
        r = l[0]
        for x in l[1:]:
            r *= x
        return r

    @staticmethod
    def _batch_size_is_one(tensor: tf.Tensor):
        s = tensor.shape.as_list()
        if len(s) >= 1:
            d = s[0]
            return isinstance(d, int) and d == 1
        else:
            return True

    @staticmethod
    def _gen_invalid_pattern_conditions(invalid_pattern_list, separator="-"):
        patterns = []
        if invalid_pattern_list is None:
            return patterns

        # see (https://github.com/google/re2/wiki/Syntax) for the meaning of regex patterns
        p_all = ".*"  # any sequence
        p_begin = "^"  # start of str
        p_end = "$"  # end of str
        # gen base pattern
        p = [p_all] * len(invalid_pattern_list)

        for i, s in enumerate(invalid_pattern_list):
            if s is None:
                continue
            cp = [x for x in p]
            cp[i] = "(%s)" % s
            patterns.append(p_begin + separator.join(cp) + p_end)
        return patterns

    def call(self, inputs, invalid_pattern_list=None, default_result_str=""):
        """

        Args:
            inputs (list[tf.Tensor]): tensor list.
            invalid_pattern_list (list[str]): same length with inputs,
                ["", "na"] means when either the 1st tensor is "" or the 2nd tensor is "na",
                    the result str will be replaced with default_result_str.
                [None, "na"] means when the 2nd tensor is "na",
                    the result str will be replaced with default_result_str.
                None means do not filter.
            default_result_str: str to replace matched patterns

        Returns:
            tf.Tensor, dtype=tf.string
        """
        # reshape & cast to string
        str_inputs = []
        set_batch_size_is_one = set()
        for i, input in enumerate(inputs):
            if self._batch_size_is_one(input):  # [], [1, -1, ..., -1]
                set_batch_size_is_one.add(i)
                input = tf.reshape(input, [1, -1])  # [1, Li]
            elif len(input.shape) != 2:
                dim = self._get_size_except_1st_dim(input)
                input = tf.reshape(input, [-1, dim])  # [B, Li]

            if input.dtype == tf.string:
                str_inputs.append(input)
            else:
                str_inputs.append(tf.strings.as_string(input))

        if len(set_batch_size_is_one) == len(inputs):
            set_batch_size_is_one.clear()
        else:
            for idx, input in enumerate(str_inputs):
                if idx not in set_batch_size_is_one:
                    batch_size = self._get_dim(input, 0)
                    break

        dim_list = [self._get_dim(input, -1) for input in str_inputs]

        tiled_inputs = []
        for idx, input in enumerate(str_inputs):
            new_shape = [-1]
            tiles = [1] if idx not in set_batch_size_is_one else [batch_size]
            for i in range(len(str_inputs)):
                dim = dim_list[i]
                if idx == i:
                    new_shape.append(dim)
                    tiles.append(1)
                else:
                    new_shape.append(1)
                    tiles.append(dim)
            input = tf.reshape(input, new_shape)
            input = tf.tile(input, tiles)  # [B, L1,L2,...,Ln]
            input = tf.reshape(input, [batch_size, -1])  # [B, L1*L2*...*Ln]
            tiled_inputs.append(input)

        result = tf.strings.join(tiled_inputs, separator=self.separator)  # [B, L1*L2*...*Ln]

        # replace invalid result_str with default_result_str
        if invalid_pattern_list is not None and len(invalid_pattern_list) != len(inputs):
            raise ValueError('length not equal:%s v.s %s' % (len(invalid_pattern_list), len(inputs)))
        for pattern in self._gen_invalid_pattern_conditions(invalid_pattern_list, self.separator):
            result = tf.strings.regex_replace(result, pattern, default_result_str,
                                              replace_global=False)  # [B, L1*L2*...*Ln]

        return result  # [B, L1*L2*...*Ln]
