'''
Author: hanyu
Date: 2022-07-19 16:10:55
LastEditTime: 2022-07-19 16:12:03
LastEditors: hanyu
Description: categorical model
FilePath: /RL_Lab/models/categorical_model.py
'''
import numpy as np
import tensorflow as tf
from alogrithm.ppo.config_ppo import Params


class CategoricalModel(tf.keras.Model):
    """Categorical Model for Discrete Action Spaces
    """
    def __init__(
        self,
        network: dict = None,
        params: Params = None,
    ) -> None:
        super().__init__("CategoricalModel")

        self.act_size = params.act_size
        self.forward = network['forward']
        self.all_neworkds = network['trainable_networks']

    def compute_action_from_logits(self, logits):
        """Draw from random categorical distribution

        Args:
            logits (list): logits

        Returns:
            _type_: drawed action
        """
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    def logp(self, logits, action):
        logp_all = tf.nn.log_softmax(logits)
        one_hot = tf.one_hot(action, depth=self.act_size)
        logp = tf.reduce_sum(one_hot * logp_all, axis=-1)
        return logp

    def call(self, inputs):
        return self.forward(inputs)

    def get_action_logp_value(self, obs):
        logits, values = self.predict(obs)
        action = self.compute_action_from_logits(logits)
        logp_t = self.logp(logits, action)
        return np.squeeze(action), np.squeeze(logp_t), np.squeeze(values)

    def entropy(self, logits=None):
        a0 = logits - tf.reduce_sum(logits, axis=-1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis=-1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
        return entropy
