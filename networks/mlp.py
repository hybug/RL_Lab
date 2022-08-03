'''
Author: hanyu
Date: 2022-07-19 16:03:12
LastEditTime: 2022-08-03 15:15:03
LastEditors: hanyu
Description: mlp
FilePath: /RL_Lab/networks/mlp.py
'''
import tensorflow as tf
from configs.config_base import PolicyParams


def mlp(inputs, params: PolicyParams, name: str = "mlp"):
    last_layer = inputs

    for i, (filter) in enumerate(params.mlp_filters):
        last_layer = tf.keras.layers.Dense(
            units=filter,
            activation=params.activation,
            kernel_initializer=params.kernel_initializer,
            name=f"{name}_{i}")(last_layer)

    return last_layer
