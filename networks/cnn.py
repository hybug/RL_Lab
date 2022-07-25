'''
Author: hanyu
Date: 2022-07-19 16:06:08
LastEditTime: 2022-07-25 12:00:37
LastEditors: hanyu
Description: cnn
FilePath: /RL_Lab/networks/cnn.py
'''
import tensorflow as tf

from configs.config_base import PolicyParams


def cnn_simple(params: PolicyParams):
    model = tf.keras.Sequential(name='simple_cnn')
    for (filter, kernal, stride) in params.cnn_filters[:-1]:
        model.add(
            tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=kernal,
                strides=stride,
                padding='valid',
                activation=params.activation,
                kernel_initializer=params.kernel_initializer))
        model.add(tf.keras.layers.MaxPool2D(pool_size=params.pool_size))
    out_size, kernel, stride = params.cnn_filters[-1]
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(units=out_size,
                              activation=params.output_activation,
                              kernel_initializer=params.kernel_initializer))
    return model, out_size
