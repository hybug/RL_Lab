'''
Author: hanyu
Date: 2021-06-28 11:51:05
LastEditTime: 2021-06-28 12:13:49
LastEditors: hanyu
Description: utils convert logits into categorical distribution
FilePath: /RL_Lab/utils/draw_categorical_samples.py
'''

import tensorflow as tf
from utils.get_tensor_shape import get_tensor_shape


def draw_categorical_samples(logits: tf.Tensor) -> tf.Tensor:
    """convert logits tensor into categorical tensor

    Args:
        logits (tf.Tensor): logits tensor

    Returns:
        tf.Tensor: categorical tensor
    """

    shape = get_tensor_shape(logits)

    if len(shape) > 2:
        # the dimension like [-1, 1, 1, N] convert into [-1, N]
        logits = tf.reshape(logits, shape=[-1, shape[-1]])

    # Draws samples from a categorical distribution.
    samples = tf.random.categorical(logits, 1)
    # Reshape into [-1, 1]
    samples = tf.reshape(samples, shape=shape[:-1] + [1])
    return samples
