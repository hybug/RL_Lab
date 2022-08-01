'''
Author: hanyu
Date: 2022-07-19 16:07:09
LastEditTime: 2022-07-28 16:32:13
LastEditors: hanyu
Description: actor critic framework
FilePath: /RL_Lab/networks/ac.py
'''

import tensorflow as tf
from configs.config_base import PolicyParams
from networks.cnn import cnn_simple, impala_cnn
from networks.mlp import mlp

from networks.network_utils import register_network


@register_network("impala_cnn_actor_critic")
def impala_cnn_actor_critic(params: PolicyParams,
                            name="impala_cnn_actor_critic"):
    inputs = tf.keras.layers.Input(shape=tuple(params.input_shape),
                                   name=f"{name}_obs")
    conv_out = impala_cnn(inputs, params)

    vf_lantent = tf.keras.layers.Flatten()(conv_out)
    lantent = tf.keras.layers.Flatten()(conv_out)

    logits_out = tf.keras.layers.Dense(units=params.act_size,
                                       name=name + "_logits_out")(lantent)
    value_out = tf.keras.layers.Dense(units=1,
                                      name=name + "_value_out")(vf_lantent)

    model = tf.keras.Model(inputs, [logits_out, value_out])

    def forward(inputs: dict):
        logits, value = model(inputs['obs'])
        return logits, value

    return {"forward": forward, "model": model}


@register_network('cnn_simple_actor_critic')
def cnn_simple_actor_critic(params: PolicyParams,
                            name='cnn_simple_actor_critic'):
    inputs = tf.keras.layers.Input(shape=tuple(params.input_shape),
                                   name=f"{name}_obs")
    conv_out = cnn_simple(inputs, params)

    logits_out = tf.keras.layers.Dense(units=params.act_size,
                                       name=name + "_logits_out")(conv_out)
    value_out = tf.keras.layers.Dense(units=1,
                                      name=name + "_value_out")(conv_out)

    model = tf.keras.Model(inputs, [logits_out, value_out])

    def forward(inputs: dict):
        logits, value = model(inputs["obs"])
        return logits, value

    return {"forward": forward, "model": model}


@register_network('mlp_simple_actor_critic')
def mlp_simple_actor_critic(params: PolicyParams,
                            name='mlp_simple_actor_critic'):
    inputs = tf.keras.layers.Input(shape=tuple(params.input_shape),
                                   name=f"{name}_obs")
    mlp_out = mlp(inputs, params)

    logits_out = tf.keras.layers.Dense(units=params.act_size,
                                       name=name + "_logits_out")(mlp_out)
    value_out = tf.keras.layers.Dense(units=1,
                                      name=name + "_value_out")(mlp_out)

    model = tf.keras.Model(inputs, [logits_out, value_out])

    def forward(inputs: dict):
        logits, value = model(inputs["obs"])
        return logits, value

    return {"forward": forward, "model": model}
