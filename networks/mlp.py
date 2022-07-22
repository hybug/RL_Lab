'''
Author: hanyu
Date: 2022-07-19 16:03:12
LastEditTime: 2022-07-22 17:00:42
LastEditors: hanyu
Description: mlp
FilePath: /RL_Lab/networks/mlp.py
'''
import tensorflow as tf
from configs.config_base import PolicyParams


def mlp(params: PolicyParams, output_size: int, head_name: str = 'MLP'):
    model = tf.keras.Sequential(name=head_name)

    for idx, hidden_size in enumerate(params.mlp_filters):
        model.add(
            tf.keras.layers.Dense(units=hidden_size,
                                  activation=params.activation,
                                  kernel_initializer=params.kernel_initializer,
                                  bias_initializer=None,
                                  name=head_name + f'_hidden_{idx}'))
    # output_size = params.mlp_filters[-1]
    # if 'critic' in head_name:
    #     model.add(
    #         tf.keras.layers.Dense(units=1,
    #                               activation=params.activation,
    #                               name=head_name + 'output'))
    # else:
    model.add(
        tf.keras.layers.Dense(units=output_size,
                              activation=params.activation,
                              name=head_name + 'output'))
    return model
