'''
Author: hanyu
Date: 2021-01-08 09:19:14
LastEditTime: 2021-01-18 12:54:27
LastEditors: hanyu
Description: categorical logits
FilePath: /test_ppo/infer/categorical.py
'''

import tensorflow as tf
from utils.get_shape import get_shape


def categorical(logits):
    '''
    description: draw samples from a categorical distribution which
                 come from logits.
    param {logits[Tensor]}
    return {list: The drawn samples of shape [batch_size, 1]}
    '''
    shape = get_shape(logits)
    if len(shape) > 2:
        logits = tf.reshape(logits, shape=[-1, shape[-1]])
    samples = tf.random.categorical(logits, 1)
    samples = tf.reshape(samples, shape=shape[:-1] + [1])
    return samples
