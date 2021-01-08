'''
Author: hanyu
Date: 2021-01-08 09:19:56
LastEditTime: 2021-01-08 09:26:28
LastEditors: hanyu
Description: get shape from tensor
FilePath: /test_ppo/utils/get_shape.py
'''
from __future__ import absolute_import

import tensorflow as tf


def get_shape(tensor):
    shape = tensor.shape.as_list()

    non_static_indexes = list()
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dynamic_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dynamic_shape[index]
    return shape
