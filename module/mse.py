'''
Author: hanyu
Date: 2021-01-22 09:45:52
LastEditTime: 2021-01-22 09:48:29
LastEditors: hanyu
Description: mse
FilePath: /test_ppo/module/mse.py
'''

import tensorflow as tf


def mse(y_hat, y_target, clip=None, clip_center=None):
    '''
    description: get the mse value
    param {*}
    return {*}
    '''
    if clip is not None and clip_center is not None:
        clip_hat = clip_center + \
            tf.clip_by_value(y_hat - clip_center, -clip, clip)
        loss = 0.5 * tf.reduce_max(
            [(clip_hat - y_target) ** 2, (y_hat - y_target) ** 2], axis=0)
    else:
        loss = 0.5 * (y_hat - y_target) ** 2
    return loss
