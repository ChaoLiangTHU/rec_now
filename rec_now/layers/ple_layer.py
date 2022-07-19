# coding=utf-8
''' 2021_11_29 lcreg163@163.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
from tensorflow import keras

from rec_now.layers.multi_dense_layer import MultiDenseLayer


class PLELayer(keras.layers.Dense):
    """Progressive Layered Extraction (PLE) layer.

    Reference:
        [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model
         for Personalized Recommendations]
        (https://dl.acm.org/doi/abs/10.1145/3383313.3412236)

    Symbols:
        B: batch size
        D: 输入维度
        N: 与当前任务有关的experts的数量，意义随当前任务和当前层数改变
        dim_out: DNN的输出维度，与具体DNN相关
    """

    def __init__(self, num_task, list_of_dnn_dims, list_of_num_experts_per_task, num_shared_task=1, **kwargs):
        """
        Args:
            num_task (int): task数量
            list_of_dnn_dims (List[List[int]] or List[int]): 每层的DNN experts的维度。每层中各个experts的维度相同
            list_of_num_experts_per_task (List[List[int]] or List[int] or int): 每层中各个任务的experts数量.
                                                            长度与list_of_dnn_dims相同.
                                                            也可以为一个int，表示各层、各任务experts数量均相同
            num_shared_task (int, optional): 共享任务的数量. Defaults to 1.
            其他参数参见父类keras.layers.Dense
        """
        if not isinstance(list_of_dnn_dims, list):
            raise TypeError('`list_of_dnn_dims` must be a list or list[list]')
        super().__init__(0, **kwargs)
        self.num_task = num_task
        self.num_shared_task = num_shared_task
        self.num_total_task = num_task + num_shared_task
        self.list_of_dnn_dims, self.list_of_num_experts_per_task, self.is_shared_tasks, self.task_names = \
            self._get_normalized_params(num_task, num_shared_task, list_of_dnn_dims, list_of_num_experts_per_task)

    @classmethod
    def _extend_int_list(cls, list_or_int, size_extend):
        """将一个list或int扩展为长度为size_extend的list，不足的补输入的最后一个元素.

        Args:
            list_or_int (List[int] or int): 待扩展的参数
            size_extend (int): 扩展后的长度

        Raises:
            TypeError: 输入参数类型不是int也不是list时
            ValueError: 输入为空list时

        Returns:
            (List[int]): [description]
        """
        if not isinstance(list_or_int, (int, list)):
            raise TypeError('`list_or_int` must be of type `int` or `list of int`, '
                            'but got `%s`' % type(list_or_int))

        if isinstance(list_or_int, int):
            list_or_int = [list_or_int]

        if not list_or_int:
            raise ValueError('list can not be empty')

        list_or_int = copy.copy(list_or_int)
        while len(list_or_int) < size_extend:
            list_or_int.append(list_or_int[-1])

        return list_or_int

    @classmethod
    def _get_normalized_params(cls, num_task, num_shared_task, list_of_dnn_dims, list_of_num_experts_per_task):
        """将参数归一化到标准形式.

        Args:
            num_task (int): task数量
            num_shared_task (int): 共享任务的数量
            list_of_dnn_dims (List[List[int]] or List[int]): 每层的DNN experts的参数
            list_of_num_experts_per_task (List[List[int]] or List[int] or int): 每层中各个任务的experts数量

        Raises:
            TypeError: list_of_dnn_dims类型错误

        Returns:
            list_of_dnn_dims (List[List[int]]): 每层的DNN experts的参数
            list_of_num_experts_per_task (List[List[int]]):每层中各个任务的experts数量，共享任务的排在最前
            is_shared_tasks (List[int]): 各个任务是否为共享任务
            task_names (List[str]): 各个任务的名字
        """
        num_total_task = num_task + num_shared_task
        num_layer = len(list_of_dnn_dims)
        list_of_num_experts_per_task = cls._extend_int_list(list_of_num_experts_per_task, num_layer)
        list_of_num_experts_per_task = [cls._extend_int_list(num_experts, num_total_task)
                                        for num_experts in list_of_num_experts_per_task]
        list_of_dnn_dims = [cls._extend_int_list(dim, 1) for dim in list_of_dnn_dims]
        is_shared_tasks = [True] * num_shared_task
        task_names = [f'shared_{task_idx}' for task_idx in range(num_task)]
        for task_idx in range(num_task):
            is_shared_tasks.append(False)
            task_names.append(f'special_{task_idx}')

        return list_of_dnn_dims, list_of_num_experts_per_task, is_shared_tasks, task_names

    def _get_dnn_name(self, layer_name, task_name):
        """获取一个DNN的名称.
        """
        return f'{self.name}/ple_layer_{layer_name}/task_{task_name}'

    def _get_gate_name(self, layer_name, task_name):
        """获取一个gate的名称.
        """
        return f'{self.name}/ple_gate_{layer_name}/task_{task_name}'

    def _build_one_dnn(self, dnn_dims, num_experts, layer_name, task_name):
        """生成一个DNN.

        Args:
            dnn_dims (List[int]): DNN各层的输出单元数
            num_experts (int): expert的数量
            layer_name (str): 该层的名称
            task_name (str): 任务名

        Returns:
            (keras.models.Sequential): DNN模型
        """
        name = self._get_dnn_name(layer_name, task_name)
        model = keras.models.Sequential(name=name)
        for idx, dim in enumerate(dnn_dims):
            is_last_layer = idx == len(dnn_dims) - 1
            activation = None if is_last_layer else self.activation
            layer = MultiDenseLayer(dim, num_experts,
                                    activation=activation,
                                    use_bias=self.use_bias,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    bias_regularizer=self.bias_regularizer,
                                    activity_regularizer=self.activity_regularizer,
                                    kernel_constraint=self.kernel_constraint,
                                    bias_constraint=self.bias_constraint,
                                    name=f'{name}/MultiDenseLayer_{idx}')
            model.add(layer)

        return model

    def _build_one_gate(self, units, layer_name, task_name):
        """生成一个gate.

        Args:
            units (int): 输出单元数量
            layer_name (str): 该层的名称
            task_name (str): 任务名

        Returns:
            (keras.models.Sequential): gate模型
        """
        name = self._get_gate_name(layer_name, task_name)
        model = keras.models.Sequential(name=name)
        layer = keras.layers.Dense(units, name=f'{name}/dense')
        model.add(layer)
        model.add(keras.layers.Softmax())
        return model

    def _get_experts_num(self, num_experts_per_task):
        """获取总的任务数和共享任务数.
        """
        num_total_experts = 0
        num_shared_experts = 0
        for is_shared_task, num_experts in zip(self.is_shared_tasks, num_experts_per_task):
            num_total_experts += num_experts
            if is_shared_task:
                num_shared_experts += num_experts

        return num_total_experts, num_shared_experts

    def _build_one_layer(self, layer_idx, dnn_dims, num_experts_per_task, num_layer):
        """生成一层PLE的参数.
        """
        is_last_layer = layer_idx == num_layer - 1
        layer_dnns = []
        layer_gates = []
        num_total_experts, num_shared_experts = self._get_experts_num(num_experts_per_task)

        task_params = zip(self.is_shared_tasks, self.task_names, num_experts_per_task)
        for is_shared_task, task_name, num_experts in task_params:
            layer_dnns.append(self._build_one_dnn(dnn_dims, num_experts, layer_idx, task_name))
            if is_shared_task and is_last_layer:
                layer_gates.append(None)
            else:
                gate_output_dim = num_experts + num_shared_experts if not is_shared_task else num_total_experts
                layer_gates.append(self._build_one_gate(gate_output_dim, layer_idx, task_name))
        return layer_dnns, layer_gates

    def build(self, input_shape):
        """生成该层所需的Variable (如有需要).
        """
        num_layer = len(self.list_of_dnn_dims)

        self.dnns = []
        self.gates = []
        layer_params = zip(range(num_layer), self.list_of_dnn_dims, self.list_of_num_experts_per_task)
        for layer_idx, dnn_dims, num_experts_per_task in layer_params:
            layer_dnns, layer_gates = self._build_one_layer(layer_idx, dnn_dims, num_experts_per_task, num_layer)
            self.dnns.append(layer_dnns)
            self.gates.append(layer_gates)

        self.built = True

    def _get_input(self, last_layer_outputs, task_idx, is_shared_task):
        """获取某个任务的输入(由该任务专有输入和共享任务的输入组成).

        Args:
            last_layer_outputs (List[tf.Tensor]): 上一层的输出
            task_idx (int): 任务下标
            is_shared_task (bool): 任务是否为共享任务
        """
        if is_shared_task:
            return tf.concat(last_layer_outputs, axis=-1)
        is_shared_tasks = self.is_shared_tasks
        inputs = [last_layer_outputs[task_idx]]
        for last_output, is_shared_task in zip(last_layer_outputs, is_shared_tasks):
            if is_shared_task:
                inputs.append(last_output)

        return tf.concat(inputs, axis=-1)

    def _get_gate_input(self, dnn_outputs, task_idx, is_shared_task):
        """获取某个gate的输入(由该任务专有输入和共享任务的输入组成).

        Args:
            dnn_outputs (List[tf.Tensor]): 该层DNN的输出
            task_idx (int): 任务下标
            is_shared_task (bool): 任务是否为共享任务

        Returns:
            (tf.Tensor): 聚合后的输出
        """
        if is_shared_task:
            return tf.concat(dnn_outputs, axis=0)
        is_shared_tasks = self.is_shared_tasks
        inputs = [dnn_outputs[task_idx]]
        for last_output, is_shared_task in zip(dnn_outputs, is_shared_tasks):
            if is_shared_task:
                inputs.append(last_output)

        return tf.concat(inputs, axis=0)

    def _apply_dnns(self, is_first_layer, inputs, outputs, layer_dnns):
        """调用该层的DNN，计算该层的输出.
        """
        dnn_outputs = []
        task_inputs = []
        dnn_params = zip(range(self.num_total_task), self.is_shared_tasks, layer_dnns)
        for task_idx, is_shared_task, dnn in dnn_params:
            last_layer_outputs = None if is_first_layer else outputs[-1]
            dnn_input = inputs if is_first_layer else self._get_input(last_layer_outputs, task_idx, is_shared_task)
            task_inputs.append(dnn_input)
            dnn_output = dnn(dnn_input)  # (N, B, dim_out)
            dnn_outputs.append(dnn_output)

        return dnn_outputs, task_inputs

    def _apply_gates(self, is_last_layer, dnn_outputs, task_inputs, layer_gates):
        """调用一层的gates函数.
        """
        gated_outputs = []
        gate_params = zip(range(self.num_total_task), self.is_shared_tasks, layer_gates)
        for task_idx, is_shared_task, gate in gate_params:
            if is_shared_task and is_last_layer:
                gated_outputs.append(None)
                continue
            gate_input = task_inputs[task_idx]
            gate_output = gate(gate_input)  # (B, N)
            gate_output = tf.transpose(gate_output, perm=[1, 0])  # (N, B)
            gate_output = tf.expand_dims(gate_output, axis=2)  # (N, B, 1)

            dnn_output = self._get_gate_input(dnn_outputs, task_idx, is_shared_task)  # (N, B, dim_out)
            dnn_output = dnn_output * gate_output
            dnn_output = tf.reduce_sum(dnn_output, axis=0, keepdims=False)  # (B, dim_out)
            gated_outputs.append(dnn_output)

        return gated_outputs

    def call(self, inputs):
        """计算PLE.

        Args:
            inputs (tf.Tensor): 该层的输入，形状为 (B, D)

        Returns:
            (List[tf.Tensor]): 各个任务(不含共享任务)的输出，长度为num_task, 其中元素形状为(B, dim_out)
        """
        num_layer = len(self.list_of_dnn_dims)
        outputs = []
        for layer_idx in range(num_layer):
            is_first_layer = layer_idx == 0
            is_last_layer = layer_idx == num_layer - 1
            layer_dnns = self.dnns[layer_idx]
            layer_gates = self.gates[layer_idx]

            # apply dnns and gates
            dnn_outputs, task_inputs = self._apply_dnns(is_first_layer, inputs, outputs, layer_dnns)
            gated_outputs = self._apply_gates(is_last_layer, dnn_outputs, task_inputs, layer_gates)

            outputs.append(gated_outputs)

        last_layer_outputs = outputs[-1]
        valid_outputs = [output for output in last_layer_outputs if output is not None]

        return valid_outputs  # [(B, dim_out)] * num_task
