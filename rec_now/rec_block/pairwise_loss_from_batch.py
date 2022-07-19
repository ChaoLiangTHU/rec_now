# coding=utf-8
''' 2021_06_04 lcreg163@163.com

从mini-batch中提取属于同一组的pair，用于计算pairwise loss.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


SMALL_POSIVITE_FLOAT = 1.0E-10


def _generate_pair_mask(sample_group_idx_var, only_upper_band=False):
    """ 通过每个样本的序号(如用户ID)，生成序号相同的样本的mask.

    Symbols:
        B: batch size

    Args:
        sample_group_idx_var: 样本所属的组，形状为(B,)，比如输入为[1,1,2,2,2]表示前两个样本属于一组，后三个样本属于一组
        only_upper_band: 是否只保留矩阵的上三角部分

    Returns:
        形状为(B, B)的bool矩阵，表示对应位置的sample的pair loss是否有效
    """

    batch_size = tf.size(sample_group_idx_var)
    sample_group_idx_var = tf.reshape(sample_group_idx_var, [-1, 1])

    group_idx_diff = sample_group_idx_var - tf.transpose(sample_group_idx_var)

    same_qid_mask = tf.cast(tf.equal(group_idx_diff, 0.0), dtype=tf.float32)
    pair_mask = same_qid_mask - tf.eye(batch_size)
    pair_mask = tf.cast(pair_mask, dtype=tf.bool)
    if only_upper_band:
        pair_mask = tf.linalg.band_part(pair_mask, 0, 1)
    return pair_mask


def generate_pair_mask(group_tensor_or_list, only_upper_band=False):
    """通过每个样本的序号(如用户ID)，生成序号相同的样本的mask.

    Symbols:
        B: batch size

    Example:
        B = 5, 样本分组group_tensor_or_list为[1, 1, 2, 2, 2]表示前两个样本属于一组，后三个样本属于一组
        当only_upper_band为False时，输出为:
            [  [False  True False False False]
            [ True False False False False]
            [False False False  True  True]
            [False False  True False  True]
            [False False  True  True False]  ]

    Args:
        group_tensor_or_list (tf.Tensor or List[tf.Tensor]): 样本所属的组，或所属组的list，每一个的形状均为(B,)
        only_upper_band (bool, optional): 是否只保留矩阵的上三角部分

    Returns:
        形状为(B, B)的bool矩阵，表示对应位置的sample的pair loss是否有效
    """
    if not isinstance(group_tensor_or_list, list):
        group_tensor_or_list = [group_tensor_or_list]
    pair_mask = None
    for group in group_tensor_or_list:
        one_pair_mask = _generate_pair_mask(group, only_upper_band)
        if pair_mask is None:
            pair_mask = one_pair_mask
        else:
            pair_mask = tf.logical_and(pair_mask, one_pair_mask)
    return pair_mask


def vec_to_matrix_pair(vec):
    """将一个vector进行列扩展、行扩展，得到两个矩阵.

    Symbols:
        B: batch size

    Args:
        vec (tf.Tensor): 形状为(B, 1)或(1, B)的向量

    Returns:
        mat (tf.Tensor): 形状为(B, B)的矩阵，通过对vec进行列扩展得到
        mat_T (tf.Tensor): mat的转置
    """
    vec = tf.reshape(vec, [-1, 1])
    tile_param = [1, tf.size(vec)]
    mat = tf.tile(vec, tile_param)
    return mat, tf.transpose(mat)


def bpr_loss_func(outputs_pos, outputs_neg, weights=None, factor=1.0, reduce_mean=True):
    """BRP loss.

    用cross entropy loss来处理pair。将正负样本的logits差视为logit, 送入cross entropy函数中

    Reference:
        [BPR: Bayesian Personalized Ranking from Implicit Feedback]
        (https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

    Symbols:
        B: batch size

    Args:
        outputs_pos (tf.Tensor): 正样本的logits，形状为(B, 1)
        outputs_neg (tf.Tensor): 负样本的logits，形状为(B, 1)
        weights (tf.Tensor, optional): 样本权重，形状为(B, 1)
        factor (float, optional): 调节因子(温度系数的倒数)

    Returns:
        (tf.Tensor): BRP loss, ，形状为(,)
    """
    logits = outputs_pos - outputs_neg
    if factor != 1.0:
        logits = logits * factor
    labels = tf.ones_like(logits)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    if weights is not None:
        losses = losses * weights
    loss = tf.reduce_sum(losses)
    if reduce_mean:
        loss = loss / (tf.cast(tf.size(losses), tf.float32) + SMALL_POSIVITE_FLOAT)
    return loss


def occurance_power_weight(group_id, power=0.0):
    """分组样本数的power次方作为样本权重.

    Example:
        group_id = [1, 2, 2]
        occurance_power_weight(group_id, power=1.0) = [1.0, 2.0, 2.0] # 同组的样本越多，权重越大
        occurance_power_weight(group_id, power=0.0) = [1.0, 1.0, 1.0] # 所有样本等权重相同
        occurance_power_weight(group_id, power=-1.0) = [1.0, 0.5, 0.5] # 同组的样本越多，权重越小

    Args:
        group_id (tf.Tensor): 样本的group_id
        power (float, optional): 分组样本数的

    Returns:
        (tf.Tensor): 样本权重
    """
    _, idx, count = tf.unique_with_counts(group_id)
    weights = tf.cast(count, tf.float32)
    if power != 1.0:
        weights = tf.pow(weights, power)
    weights = tf.gather(weights, idx)
    return weights


def _apply_sample_mask(pair_mask, mask):
    """只标记mask为True的样本的pair为有效pair.

    Symbols:
        B: batch size

    Args:
        pair_mask (tf.Tensor): 可以组pair的样本的mask，形状为(B, B)
        mask (tf.Tensor): 有效样本的mask，形状为(B,)

    Returns:
        (tf.Tensor): 更新后的可以组pair的样本的mask
    """
    if mask is None:
        return pair_mask
    mask_matrix, mask_matrix_transpose = vec_to_matrix_pair(mask)
    sample_mask = tf.logical_and(mask_matrix, mask_matrix_transpose)
    pair_mask = tf.logical_and(pair_mask, sample_mask)
    return pair_mask


def _calc_label_cond_and_weights(labels, label_pair_to_weight_func, **kwargs):
    """根据labels, 生成各个pair的权重矩阵.

    Args:
        labels (tf.Tenor): 各个样本的label
        label_pair_to_weight_func (callable): 输入为正样本label的Tensor矩阵和负样本label的Tensor矩阵，输出为weights的Tensor.
                                              默认为None，表示正样本大于负样本则计算pairwise loss，且权重均为1

    Returns:
        label_cond (tf.Tenor): 根据label生成的mask, tf.bool类型的矩阵, 其中为True的元素对应weights_mat中大于0的元素.
        weights_mat (tf.Tenor or None): 各个pair的权重.
    """
    label_matrix, label_matrix_transpose = vec_to_matrix_pair(labels)
    if label_pair_to_weight_func is None:
        label_cond = label_matrix > label_matrix_transpose
        weights_mat = None
    else:
        weights_mat = label_pair_to_weight_func(label_matrix, label_matrix_transpose, **kwargs)
        label_cond = weights_mat > 0
    return label_cond, weights_mat


def _apply_wrong_order_pair_mask(only_use_wrong_order_pair, outputs_matrix, outputs_matrix_transpose, pair_mask):
    """在原有pair mask的基础上, 增加错误order mask (如果only_use_wrong_order_pair为True).
    """
    if only_use_wrong_order_pair:
        output_cond = outputs_matrix < outputs_matrix_transpose
        pair_mask = tf.logical_and(pair_mask, output_cond)
    return pair_mask


def _apply_pair_mask(mat, pair_mask_flattened):
    """取出mat中的元素，mask为pair_mask_flattened.

    Args:
        mat (tf.Tenosr): 矩阵，元素总数和pair_mask_flattened相同.
        pair_mask_flattened (tf.Tenosr): 一维的Tensor，tf.bool型.
    Returns:
        masked_val (tf.Tensor): 经过pair_mask_flattened mask后的元素.
    """
    if mat is None:
        return None
    return tf.boolean_mask(tf.reshape(mat, [-1]), pair_mask_flattened)


def _merge_weights_by_mul(weights1, weights2):
    if weights1 is None:
        return weights2
    if weights2 is None:
        return weights1
    return weights1 * weights2


def pairwise_loss(outputs, labels, groups,
                  pairloss_func=bpr_loss_func,
                  only_use_wrong_order_pair=False,
                  return_num_pair=False,
                  click_occurance_power=0.0,
                  mask=None,
                  label_pair_to_weight_func=None,
                  **kwargs
                  ):
    """ 计算pairwise loss.

    Args:
        outputs: 各个样本的输出(logits或者最终输出)
        labels: 各个样本的label
        groups: 各个样本所属的group，可以是tf.Tensor 或 tf.Tensor的list(表示多组条件，各组条件进行and操作)。
                该参数为list时要求第一个为主分组(比如用户ID)，主分组用于计算同组下的数量weight(click_occurance_power不为0时).
        bpr_loss_func: 输入为pair中的正样本outputs和负样本outputs，输出为pairwise的值
        only_use_wrong_order_pair: 是否只使用逆序(负样本比正样本打分高)的样本
        click_occurance_power: 对于同一个group, 比如用户A，假设其有 3个pair（最终），则每个样本的权重为 3**click_occurance_power。
                                如果click_occurance_power=-1，则表示每个用户的所有pair权重和为1； 如果为0，则各个pair的权重均为1
        mask (tf.Tensor, optional): 形状和labels相同，代表该样本是否计算pairwise loss, 为True计算，为False不计算. 为None时表示所有样本有效.
        label_pair_to_weight_func: 输入为正样本label的Tensor矩阵和负样本label的Tensor矩阵，输出为weights的Tensor.
                                   默认为None，表示正样本大于负样本则计算pairwise loss，且权重均为1
    Returns:
        pairwise loss, 同bpr_loss_func的输出
    """
    pair_mask = generate_pair_mask(groups)
    pair_mask = _apply_sample_mask(pair_mask, mask)
    outputs_matrix, outputs_matrix_transpose = vec_to_matrix_pair(outputs)
    label_cond, weights_mat = _calc_label_cond_and_weights(labels, label_pair_to_weight_func, **kwargs)

    pair_mask = tf.logical_and(pair_mask, label_cond)
    pair_mask = _apply_wrong_order_pair_mask(only_use_wrong_order_pair,
                                             outputs_matrix, outputs_matrix_transpose,
                                             pair_mask)
    pair_mask_flattened = tf.reshape(pair_mask, [-1])
    pair_mask_flattened = tf.stop_gradient(pair_mask_flattened)

    weights = _apply_pair_mask(weights_mat, pair_mask_flattened)
    weights = _apply_occurance_weights(groups, click_occurance_power, pair_mask_flattened, weights)

    if weights is not None:
        weights = tf.stop_gradient(weights)

    outputs_pos = _apply_pair_mask(outputs_matrix, pair_mask_flattened)
    outputs_neg = _apply_pair_mask(outputs_matrix_transpose, pair_mask_flattened)
    loss = pairloss_func(outputs_pos, outputs_neg, weights)
    if return_num_pair:
        n_pair = tf.cast(tf.size(outputs_pos), tf.float32)
        return loss, n_pair
    else:
        return loss


def _apply_occurance_weights(groups, click_occurance_power, pair_mask_flattened, weights):
    """根据每组中最终的pair数量，以及click_occurance_power，生成各组pair的权重.
    """
    if click_occurance_power != 0.0:
        group = groups[0] if isinstance(groups, list) else groups
        group_matrix, _ = vec_to_matrix_pair(group)
        groups_pos = _apply_pair_mask(group_matrix, pair_mask_flattened)
        occurance_weights = occurance_power_weight(groups_pos, power=click_occurance_power)
        weights = _merge_weights_by_mul(weights, occurance_weights)
    return weights
