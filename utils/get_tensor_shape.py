'''
Author: hanyu
Date: 2021-06-28 10:26:54
LastEditTime: 2021-06-28 10:43:56
LastEditors: hanyu
Description: get tensor's shape utils
FilePath: /RL_Lab/utils/get_tensor_shape.py
'''

import tensorflow as tf


def get_tensor_shape(tensor: tf.Tensor) -> list:
    """get tensor's shape

    Args:
        tensor (tf.Tensor): Tensor

    Returns:
        list: shape
    """

    shape = tensor.shape.as_list()

    non_static_indexes = list()
    for (index, dim) in enumerate(shape):
        if dim is None:
            # tf.compat.v1.placeholder((-1, 4)) for example
            # return the shape=(None, 1)
            # It will get the shape only in session.run, with tf.shape()
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    raise ValueError
    # TODO fix the tensor dynamic shape
