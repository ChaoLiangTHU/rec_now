# coding=utf-8
''' 20221101 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MultiHashLayer(tf.keras.layers.Layer):
    """Hash values with multi hash layers that use different seeds and then do embedding (optional).

    Symbols:
        B: batch size
        L: shape of inputs may be [B, L] or [B]
        D: output dimension of embeddings
        Nh: number of hash func
    """

    def __init__(self, num_bins, embedding_dim=-1, num_hash=2, salts=1, embeddings_initializer=None,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """_summary_

        Args:
            num_bins (int): _description_
            embedding_dim (int, optional): >0: do embedding with dimesion embedding_dim. <0: not use embedding. Defaults to -1.
            num_hash (int, optional): how many hash functions to use. Defaults to 2.
            salts (int, optional): salts for each hash function. Defaults to 1.
            embeddings_initializer (_type_, optional): tf initializer for embeddings. Defaults to None.
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.num_hash = num_hash
        if isinstance(salts, int):
            self.salts = [salts + i for i in range(num_hash)]
        else:
            self.salts = list(salts)
        while len(self.salts) < num_hash:
            self.salts.append(self.salts[-1] + 1)

        if self.embedding_dim > 0:
            if embeddings_initializer is None:
                embeddings_initializer = tf.keras.initializers.RandomUniform(minval=-1E-4, maxval=1E-4)
            self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)

    def build(self, input_shape=None):
        """generate variables used by this layer.
        """
        if self.built:
            return super().build(input_shape)
        self.hash_layers = []
        self.embedding_layers = []
        for i in range(self.num_hash):
            try:
                layer = tf.keras.layers.Hashing(self.num_bins, salt=(self.salts[i], self.salts[i]))
            except:
                from tensorflow.python.keras.layers.preprocessing.hashing import Hashing

                layer = Hashing(self.num_bins, salt=(self.salts[i], self.salts[i]))
            self.hash_layers.append(layer)
            if self.embedding_dim > 0:
                with tf.name_scope(self.name):
                    embedding_layer = tf.keras.layers.Embedding(
                        self.num_bins, self.embedding_dim, embeddings_initializer=self.embeddings_initializer, name=self.name + "_embedding")
                    embedding_layer.build(None)
                self.embedding_layers.append(embedding_layer)
        self.built = True
        return super().build(input_shape)

    def call(self, inputs, combiner='sum'):
        """hash and embedding.

        Args:
            inputs (tf.Tensor): input tf tensor.
            combiner (str, optional): how to deal with multi outputs, can be 'concat', 'sum', 'mean', None. Defaults to 'concat'.

        Returns:
            tf.Tensor
        """

        outputs = []
        if inputs.dtype.is_integer:
            inputs = tf.strings.as_string(inputs)
        for i, hash_layer in enumerate(self.hash_layers):
            output = hash_layer(inputs)
            if self.embedding_dim > 0:
                output = self.embedding_layers[i](output)
            outputs.append(output)

        if len(outputs) == 1:
            return outputs[-1]  # [B] or [B, L] or [B, L, D]
        elif combiner == 'concat':
            return tf.concat(outputs, axis=-1)  # [B, Nh] or [B, L, Nh] or [B, Nh*D] or [B, L, Nh*D]
        elif combiner == 'sum' and self.embedding_dim > 0:
            return tf.math.add_n(outputs)  # [B, D] or [B, L, D]
        elif combiner == 'mean' and self.embedding_dim > 0:
            return tf.math.add_n(outputs) * (1.0 / len(outputs))  # [B, D] or [B, L, D]
        else:
            return outputs  # [[B, D] or [B, L, D]] * Nh

    def get(self, inputs):
        if not self.built:
            self.build()
        res = self(inputs, combiner="sum")  # [B, D] or [B, L, D]
        return res

    def get_pooling(self, keys, weights=None, name=None):
        emb = self.get(keys)  # [B, L, D]
        if weights is not None:
            weights = tf.expand_dims(weights, -1, name=name)
            emb = weights * emb
        if len(emb.shape) > 2:
            axis = list(range(1, len(emb.shape) - 1))
            res = tf.reduce_sum(emb, axis=axis, keepdims=False)  # [B, D]
        else:
            res = emb
        return res


class FastMultiHashLayer(tf.keras.layers.Layer):
    """Hash values with multi hash layers that use different seeds and then do embedding (optional).

    Symbols:
        B: batch size
        L: shape of inputs may be [B, L] or [B]
        D: output dimension of embeddings
        Nh: number of hash func
    """

    def __init__(self, num_bins, embedding_dim=-1, num_hash=2, salts=1, embeddings_initializer=None,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        """_summary_

        Args:
            num_bins (int): _description_
            embedding_dim (int, optional): >0: do embedding with dimesion embedding_dim. <0: not use embedding. Defaults to -1.
            num_hash (int, optional): how many hash functions to use. Defaults to 2.
            salts (int, optional): salts for each hash function. Defaults to 1.
            embeddings_initializer (_type_, optional): tf initializer for embeddings. Defaults to None.
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.num_hash = num_hash
        if isinstance(salts, int):
            self.salts = [salts + i for i in range(num_hash)]
        else:
            self.salts = list(salts)
        while len(self.salts) < num_hash:
            self.salts.append(self.salts[-1] + 1)

        if self.embedding_dim > 0:
            if embeddings_initializer is None:
                embeddings_initializer = tf.keras.initializers.RandomUniform(minval=-1E-4, maxval=1E-4)
            self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)

    def build(self, input_shape=None):
        """generate variables used by this layer.
        """
        if self.built:
            return super().build(input_shape)
        self.hash_layers = []
        for i in range(self.num_hash):
            salt = None if i == 0 else [self.salts[i], self.salts[i]]
            try:
                layer = tf.keras.layers.Hashing(self.num_bins, salt=salt)
            except:
                from tensorflow.python.keras.layers.preprocessing.hashing import Hashing

                layer = Hashing(self.num_bins, salt=salt)
            self.hash_layers.append(layer)
        if self.embedding_dim > 0:
            with tf.name_scope(self.name):
                embedding_layer = tf.keras.layers.Embedding(
                    self.num_bins * self.num_hash, self.embedding_dim, embeddings_initializer=self.embeddings_initializer, name=self.name + "_embedding")
                embedding_layer.build(None)
            self.embedding_layer = embedding_layer
        self.built = True
        return super().build(input_shape)

    @staticmethod
    def _get_size_except_batch_size(tensor: tf.Tensor):
        s = tensor.shape.as_list()
        r = 1
        for x in s[1:]:
            r *= x
        return r

    def call(self, inputs, combiner='sum'):
        """hash and embedding.

        Args:
            inputs (tf.Tensor): input tf tensor.
            combiner (str, optional): how to deal with multi outputs, can be 'concat', 'sum', 'mean', None. Defaults to 'concat'.

        Returns:
            tf.Tensor
        """
        if inputs.dtype.is_integer:
            inputs = tf.strings.as_string(inputs)
        outputs = []
        for i, hash_layer in enumerate(self.hash_layers):
            output = hash_layer(inputs)  # [B] or [B, L]
            if self.embedding_dim > 0:
                if i > 0:
                    output += (i * self.num_bins)
                output = tf.expand_dims(output, -1)  # [B, 1] or [B, L, 1]
            outputs.append(output)
        outputs = tf.concat(outputs, axis=-1)  # [B, num_hash] or [B, L, num_hash]
        if self.embedding_dim > 0:
            outputs = self.embedding_layer(outputs)  # [B, num_hash, D] or [B, L, num_hash, D]
        if combiner == 'concat':
            outputs = tf.reshape(outputs, [-1, self._get_size_except_batch_size(outputs)])
        elif combiner == 'sum' and self.embedding_dim > 0:
            return tf.math.reduce_sum(outputs, axis=-2, keepdims=False)  # [B, D] or [B, L, D]
        elif combiner == 'mean' and self.embedding_dim > 0:
            return tf.math.reduce_mean(outputs, axis=-2, keepdims=False)  # [B, D] or [B, L, D]
        return outputs  # [B, num_hash, D] or [B, L, num_hash, D]

    def get(self, inputs):
        if not self.built:
            self.build()
        res = self(inputs, combiner="sum")  # [B, D] or [B, L, D]
        return res

    def get_pooling(self, keys, weights=None, name=None):
        emb = self.get(keys)  # [B, L, D]
        if weights is not None:
            weights = tf.expand_dims(weights, -1, name=name)
            emb = weights * emb
        if len(emb.shape) > 2:
            axis = list(range(1, len(emb.shape) - 1))
            res = tf.reduce_sum(emb, axis=axis, keepdims=False)  # [B, D]
        else:
            res = emb
        return res
