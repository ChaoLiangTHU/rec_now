# coding=utf-8
''' 2022_01_14 lcreg163@163.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
from tensorflow import keras


DEFAULT_NEIGHBOR_INITIAL_WEIGHT = 0.1  # 初始的邻节点权重


class SparseGNNLayer(keras.layers.Layer):
    """基于与周围节点加权求和进行图卷积的特征交叉层.

    该实现要求人工指定图中各节点的邻节点(有向边)，然后使用图卷积进行图运算.
    可以使用多层GNN进行图卷积，各层GNN使用的邻节点关系是相同的，但可以训练得到不同的权重.

    Example:
        # 构造batch_size为2，有4个fields的输入
        user_id = [[1], [2]]
        user_age = [[10], [20]]
        doc_id = [[1], [3]]
        doc_subject = [[0], [1]]

        embedding_table_size = 100
        embedding_dim = 3
        embeddings = []
        for feature in [user_id, user_age, doc_id, doc_subject]:
            embedding_layer = keras.layers.Embedding(embedding_table_size, embedding_dim, input_length=1)
            feature = tf.constant(feature, dtype=tf.int64)
            embedding = embedding_layer(feature)
            embedding = tf.squeeze(embedding, axis=1)
            embeddings.append(embedding)

        fields = ['user_id', 'user_age', 'doc_id', 'doc_subject']

        # 图邻节点列表，边为有向边， 比如doc_subject在user_id的邻节点中，但user_id不在doc_subject的邻节点中
        field2neighbors = {
            'user_id': ['doc_id', 'doc_subject'],
            'user_age': ['doc_subject'],
            'doc_subject': ['user_age']
        }

        gnn_layer = SparseGNNLayer(fields=fields, field2neighbors=field2neighbors,
                                num_layers=3, share_weights_between_layers=False,
                                activation='tanh')
        outputs = gnn_layer(embeddings)
        print(outputs)

    Symbols:
        B: batch size
        D: 输入维度
        F: fields的数量
    """

    def __init__(self, fields, field2neighbors,
                 weights_initializer=tf.constant_initializer(DEFAULT_NEIGHBOR_INITIAL_WEIGHT),
                 num_layers=1, share_weights_between_layers=True, activation='tanh', **kwargs):
        """
        Args:
            fields (List): 长度为B的list，其中每个元素代表一个field (类型不限)
            field2neighbors (dict): field到其neighbors的映射，每组neighbors为一个list或set
            weights_initializer (初始化器, optional): 邻节点权重的初始化器. Defaults to tf.constant_initializer(0.1).
            num_layers (int, optional): 使用多少层的GNN. Defaults to 1.
            share_weights_between_layers (bool, optional): 各层的GNN是否使用相同的权重. Defaults to True.
            activation (str or callable, optional): 各层GNN的激活函数. Defaults to 'tanh'.
        """
        super().__init__(**kwargs)
        self.fields = fields
        self.field2neighbors = self._normalize_neighbors(field2neighbors)
        self.field2idx = {field: idx for idx, field in enumerate(fields)}
        self.weights_initializer = weights_initializer
        self.num_layers = num_layers
        self.share_weights_between_layers = share_weights_between_layers
        self.activation = keras.activations.get(activation)
        self._check_fields()
        self._check_field2neighbors()

    def _normalize_neighbors(self, field2neighbors):
        """将field2neighbors归一化为dict形式.
        """
        if isinstance(field2neighbors, (list, set)):
            return SparseGNNLayer.list_of_edge_to_neighbors(field2neighbors)
        if not isinstance(field2neighbors, dict):
            raise TypeError('field2neighbors must be one of `list of pairs`, `set of pairs`, '
                            f'`dict of neighbors`, but get {type(field2neighbors)}')
        return field2neighbors

    def _check_fields(self):
        """检查fields中是否有重复值.
        """
        set_fields = set(self.fields)
        if len(set_fields) != len(self.fields):
            num_duplicated_fields = len(self.fields) - len(set_fields)
            raise ValueError(f'{num_duplicated_fields} duplicated fields in fields.')

    def _check_field2neighbors(self):
        """检查field2neighbors中的field是否全在fields中.
        """
        set_fields = set(self.fields)
        for field, neighbors in self.field2neighbors.items():
            if field not in set_fields:
                raise ValueError(f'field `{field}` in field2neighbors but not in fields.')
            for neighbor in neighbors:
                if neighbor not in set_fields:
                    raise ValueError(f'field `{neighbor}` in field2neighbors but not in fields.')

    def _num_edges(self):
        """统计一层GNN的总边数(即总的邻节点个数).
        """
        num_edges = 0
        for neighbors in self.field2neighbors.values():
            num_edges += len(neighbors)
        return num_edges

    def _generate_indices(self):
        """将dict型的邻节点表示，转换为tf.SparseTensor所需的下标.
        """
        indices = []
        for idx, field in enumerate(self.fields):
            neighbors = self.field2neighbors.get(field, [])
            for neighbor in neighbors:
                neighbor_idx = self.field2idx[neighbor]
                indices.append([neighbor_idx, idx])
        indices = sorted(indices)
        indices = tf.constant(indices, dtype=tf.int64)
        return indices

    def _num_sets_of_gnn_weights(self):
        """有多少组GNN参数.
        """
        if self.share_weights_between_layers:
            return 1
        return self.num_layers

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        if self.built:
            return
        num_nodes = len(self.fields)
        num_edges = self._num_edges()
        num_sets_of_gnn_weights = self._num_sets_of_gnn_weights()
        self.gnn_weights = [self.add_weight(name=f'weights_{idx}', shape=[num_edges], dtype=self.dtype,
                                            initializer=self.weights_initializer, trainable=self.trainable)
                            for idx in range(num_sets_of_gnn_weights)
                            ]
        self.dense_shape = tf.constant([num_nodes, num_nodes], dtype=tf.int64)
        self.indices = self._generate_indices()
        self.sparse_gnn_weights = [tf.SparseTensor(self.indices, weights, self.dense_shape)
                                   for weights in self.gnn_weights]
        self.built = True

    def _convert_2d_tensor_to_3d(self, inputs, num_nodes):
        """如果inputs为形状为(B, F*D)的二维tensor，将其变为(B, F, D)大小的tensor.
        """
        if len(inputs.shape) == 2:
            all_dim = inputs.shape[-1]
            if all_dim % num_nodes != 0:
                raise ValueError(f'can not determine embedding_dim! '
                                 f'{all_dim} can not be divided by {num_nodes}.')
            dim = inputs.shape[-1] // num_nodes
            inputs = tf.reshape(inputs, [-1, num_nodes, dim])  # (B, F, D)
        return inputs

    def _transpose_inputs(self, inputs, num_nodes):
        """如果inputs形状为 (B, F, D)，将其转置为(B, D, F)大小的tensor.
        """
        if inputs.shape[1] == num_nodes:
            if inputs.shape[1] == inputs.shape[2]:
                logging.warning(f'WARNING: #fields and embedding_dim are both {inputs.shape[1]}, '
                                'treat the input as (B, F, D) format.')
            inputs = tf.transpose(inputs, perm=[0, 2, 1])  # (B, D, F)
        return inputs

    def _normalize_inputs(self, inputs):
        """将inputs的形状统一为(B, D, F).
        """
        num_nodes = len(self.fields)

        # 如果inputs为embeddings的list，将其合成(B, F*D)大小的tensor
        if isinstance(inputs, list):
            inputs = tf.concat(inputs, axis=-1)  # (B, F*D)

        inputs = self._convert_2d_tensor_to_3d(inputs, num_nodes)  # (B, F, D)
        inputs = self._transpose_inputs(inputs, num_nodes)  # (B, F, D)
        return inputs

    def _transpose_and_flattern_outputs(self, outputs, transpose=False, flattern=False):
        """根据参数，决定是否调整输出中F维和D维的顺序，并从3维转换为2维.
        """
        if transpose:
            outputs = tf.transpose(outputs, perm=[0, 2, 1])
        if flattern:
            num_field_mul_dim = int(outputs.shape[1]) * int(outputs.shape[2])
            outputs = tf.reshape(outputs, [-1, num_field_mul_dim])
        return outputs

    def call(self, inputs, return_all_layers=False, transpose_outputs=True, flattern_outputs=True):
        """计算GNN的输出.

        Args:
            inputs (tf.Tensor): 形状为(B, F, D)或(B, D, F)或(B, F*D)，也可以为一个长度为F的list，其中每个tensor形状为(B, D)
                                注意，如果输入为3维的，且F=D，则认为输入的形状为(B, F, D)
            return_all_layers (bool, optional): 是否返回所以GNN层的输出. Defaults to False.
            transpose_outputs (bool, optional): 是否将输出的形状从(B, D, F)调整为(B, F, D). Defaults to True.
            flattern_outputs (bool, optional): 是否将输出的形状从三维调整为两维. Defaults to True.

        Returns:
            return_all_layers 为False时，返回tf.Tensor，为最后一层GNN的输出. 默认参数情况下，形状为 (B, F*D)
            return_all_layers 为True时，返回List(tf.Tensor)，为各层GNN的输出.
        """
        all_outputs = []
        outputs = self._normalize_inputs(inputs)  # (B, D, F)
        for i in range(self.num_layers):
            sparse_gnn_weights = self.sparse_gnn_weights[i % len(self.sparse_gnn_weights)]
            dense_weights = tf.sparse.to_dense(sparse_gnn_weights)  # (F, F)
            conved = tf.matmul(outputs, dense_weights)  # (B, D, F)
            outputs = outputs + conved  # (B, D, F)
            outputs = self.activation(outputs)  # (B, D, F)
            all_outputs.append(outputs)
        if return_all_layers:
            all_outputs = [self._transpose_and_flattern_outputs(x, transpose_outputs, flattern_outputs)
                           for x in all_outputs]
            return all_outputs
        else:
            outputs = self._transpose_and_flattern_outputs(outputs, transpose_outputs, flattern_outputs)
            return outputs

    @staticmethod
    def list_of_edge_to_neighbors(list_of_edge, directed=True):
        """将邻节点list，转换dict形式，供该类的构造函数使用.

        Args:
            list_of_edge (List): 邻节点list，其中每个元素为一个2元tuple: (node_to, node_from), 在GNN中, node_to会聚合node_from的信息.
            directed (bool, optional): list_of_edge中的pair所代表的邻节点关系，是否为有向的. Defaults to True.

        Returns:
            (dict): node到其neighbors的映射，每组neighbors为一个set
        """
        field2neighbors = {}

        def add_pair(node_to, node_from):
            if node_to not in field2neighbors:
                field2neighbors[node_to] = set()
            field2neighbors[node_to].add(node_from)

        for pair in list_of_edge:
            node_to = pair[0]
            node_from = pair[1]
            add_pair(node_to, node_from)
            if not directed:
                add_pair(node_from, node_to)

        return field2neighbors
