# coding=utf-8
''' 2021_11_29 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

from rec_now.layers.multi_dense_layer import MultiDenseLayer


class MMOELayer(keras.layers.Dense):
    """Multi-gate Mixture-of-Experts (MMoE) layer.

    Reference:
        [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts]
        (https://dl.acm.org/doi/10.1145/3219819.3220007)

    Symbols:
        B: batch size
        D: 输入维度
        N: experts的数量
        T: task的数量
        dim_out: 单个expert的输出维度
    """

    def __init__(self, num_task, num_experts, dnn_dims, **kwargs):
        """
        Args:
            num_task (int): task数量
            num_experts (int): experts数量
            dnn_dims (List[int]): DNN各层的输出单元数
        """
        super().__init__(0, **kwargs)
        self.num_task = num_task
        self.num_experts = num_experts
        self.dnn_dims = dnn_dims

    def _get_gate_name(self, layer_name, task_name):
        """获取一个gate的名称.
        """
        return f'{self.name}/ple_gate_{layer_name}/task_{task_name}'

    def _build_gates(self):
        """生成一个gate.

        Args:
            units (int): 输出单元数量
            layer_name (str): 该层的名称
            task_name (str): 任务名

        Returns:
            (keras.models.Sequential): gate模型
        """
        name = f"{self.name}/gates"
        model = keras.models.Sequential(name=name)
        layer = MultiDenseLayer(self.num_experts, self.num_task, name=f'{name}/MultiDenseLayer')
        model.add(layer)
        model.add(keras.layers.Softmax())
        return model

    def _build_experts(self):
        """生成num_experts个experts.
        """
        name = f"{self.name}/experts"
        model = keras.models.Sequential(name=name)
        dnn_dims = self.dnn_dims
        num_experts = self.num_experts
        for layer_idx, dim in enumerate(dnn_dims):
            is_last_layer = layer_idx == len(dnn_dims) - 1
            activation = None if is_last_layer else self.activation
            expert = MultiDenseLayer(dim, num_experts,
                                     use_bias=self.use_bias,
                                     activation=activation,
                                     kernel_constraint=self.kernel_constraint,
                                     bias_constraint=self.bias_constraint,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_regularizer=self.bias_regularizer,
                                     activity_regularizer=self.activity_regularizer,
                                     kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer,
                                     name=f'{name}/MultiDenseLayer_{layer_idx}')
            model.add(expert)
        return model

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """

        self.dnn_experts = self._build_experts()
        self.gates = self._build_gates()
        self.built = True

    def call(self, inputs, merge_output=True):
        """计算MMoE.

        Args:
            inputs (tf.Tensor): 该层的输入，形状为 (B, D)
            merge_output (bool, optional): 是否将所有task的结果合并到一个张量中. Defaults to True.

        Returns:
            MMoE的结果.
                如果merge_output = True: 返回一个形状为(T, B, dim_out)的张量
                如果merge_output = False: 返回num_task个形状为(B, dim_out)的张量
        """

        experts_output = self.dnn_experts(inputs)  # (N, B, dim_out)
        experts_output = tf.expand_dims(experts_output, axis=0)  # (1, N, B, dim_out)

        gates_output = self.gates(inputs)  # (T, B, N)
        gates_output = tf.transpose(gates_output, perm=(0, 2, 1))  # (T, N, B)
        gates_output = tf.expand_dims(gates_output, axis=-1)  # (T, N, B, 1)

        output = experts_output * gates_output  # (T, N, B, dim_out)
        output = tf.reduce_sum(output, axis=1, keepdims=False)  # (T, B, dim_out)

        if merge_output:
            return output  # (T, B, dim_out)

        outputs = []
        for task_idx in range(self.num_task):
            task_output = output[task_idx, :, :]  # (B, dim_out)
            outputs.append(task_output)
        return outputs  # [(B, dim_out)] * T
