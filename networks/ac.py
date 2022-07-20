'''
Author: hanyu
Date: 2022-07-19 16:07:09
LastEditTime: 2022-07-19 16:09:14
LastEditors: hanyu
Description: actor critic framework
FilePath: /RL_Lab/networks/ac.py
'''

import tensorflow as tf
from alogrithm.ppo.config_ppo import PolicyParams
from networks.cnn import cnn_simple
from networks.mlp import mlp

from networks.network_utils import register_network


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
    _mlp_actor = mlp(params=params, head_name='actor_mlp')
    _actor.add(_mlp_actor)

    # Concat the last MLP layer to critic, the output_size is 1
    _mlp_critic = mlp(params=params, head_name='critic_mlp')
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
    _mlp_actor = mlp(params=params, head_name='mlp_actor')
    _actor.add(_mlp_actor)

    # Concat the last MLP layer to critic, the output_size is 1
    _mlp_critic = mlp(params=params, head_name='mlp_critic')
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