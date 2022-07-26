'''
Author: hanyu
Date: 2022-07-19 16:07:09
LastEditTime: 2022-07-26 16:12:55
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

    logits_out = tf.keras.layers.Dense(units=params.act_size,
                                       name=name + "_logits_out")(conv_out)
    value_out = tf.keras.layers.Dense(units=1,
                                      name=name + "_value_out")(conv_out)

    model = tf.keras.Model(inputs, [logits_out, value_out])

    def forward(inputs: dict):
        logits, value = model(inputs['obs'])
        return logits, value

    return {"forward": forward, "model": model}


@register_network('cnn_simple_actor_critic')
def cnn_simple_actor_critic(params: PolicyParams,
                            name='cnn_sample_actor_critic'):
    cnn, _ = cnn_simple(params=params)

    # Add CNN into the model
    _actor = tf.keras.Sequential(name='actor')
    _critic = tf.keras.Sequential(name='critic')
    _actor.add(cnn)
    _critic.add(cnn)

    # Concat the last MLP layer to actor
    _mlp_actor = mlp(params=params,
                     output_size=params.act_size,
                     head_name='actor_mlp')
    _actor.add(_mlp_actor)

    # Concat the last MLP layer to critic, the output_size is 1
    _mlp_critic = mlp(params=params, output_size=1, head_name='critic_mlp')
    _critic.add(_mlp_critic)

    _actor.build(input_shape=(None, ) + tuple(params.input_shape))
    _actor.summary()

    _critic.build(input_shape=(None, ) + tuple(params.input_shape))
    _critic.summary()

    def forward(input=None):
        logits = _actor(input['obs'])
        values = _critic(input['obs'])
        return logits, values

    return {"forward": forward, "trainable_networks": [_actor, _critic]}


@register_network('mlp_simple_actor_critic')
def mlp_simple_actor_critic(params: PolicyParams,
                            name='mlp_simple_actor_critic'):
    # mlp_simple = mlp(params=params)

    # # Add CNN into the model
    _actor = tf.keras.Sequential(name='actor')
    _critic = tf.keras.Sequential(name='critic')
    # _actor.add(mlp_simple)
    # _critic.add(mlp_simple)

    # Concat the last MLP layer to actor
    _mlp_actor = mlp(params=params,
                     output_size=params.act_size,
                     head_name='mlp_actor')
    _actor.add(_mlp_actor)

    # Concat the last MLP layer to critic, the output_size is 1
    _mlp_critic = mlp(params=params, output_size=1, head_name='mlp_critic')
    _critic.add(_mlp_critic)

    _actor.build(input_shape=(None, ) + tuple(params.input_shape))
    _actor.summary()

    _critic.build(input_shape=(None, ) + tuple(params.input_shape))
    _critic.summary()

    def forward(input=None):
        logits = _actor(input['obs'])
        values = _critic(input['obs'])
        return logits, values

    return {"forward": forward, "trainable_networks": [_actor, _critic]}