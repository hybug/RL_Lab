'''
Author: hanyu
Date: 2022-07-19 16:10:55
LastEditTime: 2022-08-26 11:38:35
LastEditors: hanyu
Description: categorical model
FilePath: /RL_Lab/models/categorical_model.py
'''
import numpy as np
import tensorflow as tf

from configs.config_base import Params
from models.model_base import TFModelBase


class CategoricalModel(TFModelBase):
    """Categorical Model for Discrete Action Spaces
    """
    def __init__(
        self,
        network: dict = None,
        params: Params = None,
    ) -> None:
        super().__init__("CategoricalModel")

        self.act_size = params.policy.act_size
        self.forward_func = network['forward']
        self.base_model = network['model']
        self.register_variables(self.base_model.variables)

    def compute_action_from_logits(self, logits):
        """Draw from random categorical distribution

        Args:
            logits (list): logits

        Returns:
            _type_: drawed action
        """
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        # u = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
        # return tf.math.argmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)

    def logp(self, logits, action):
        # logp_all = tf.nn.log_softmax(logits)
        # one_hot = tf.one_hot(action, depth=self.act_size)
        # logp = tf.reduce_sum(one_hot * logp_all, axis=-1)
        # return logp
        # print(action.shape, logits.shape)
        # print(action.dtype, logits.dtype)
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        # ?????????

        # if action.dtype in (tf.float16, tf.float32, tf.float64):
        #     action = tf.cast(action, tf.int32)
        # if action.dtype in (tf.uint8, tf.int16, tf.int32, tf.int64):
        #     x = tf.one_hot(action, depth=self.act_size)
        logp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(action, tf.int32), logits=logits)
        # logp = -tf.nn.softmax_cross_entropy_with_logits(labels=x,
        #                                                 logits=logits)
        return -logp

    def forward(self, inputs_dict: dict):
        return self.forward_func(inputs_dict)
        # return self.base_model(inputs_dict['obs'])

    def get_action_logp_value(self, obs):
        logits, values = self.forward(obs)
        action = self.compute_action_from_logits(logits)
        logp_t = self.logp(logits, action)
        return np.squeeze(action), np.squeeze(logp_t), np.squeeze(values), np.squeeze(logits)

    def entropy(self, logits=None):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis=-1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
        return entropy

    def kl(self, prev_logits, logits):
        a0 = prev_logits - tf.reduce_max(prev_logits, axis=1, keepdims=True)
        a1 = logits - tf.reduce_max(logits, axis=1, keepdims=True)
        ea0 = tf.math.exp(a0)
        ea1 = tf.math.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 *
                             (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)),
                             axis=1)
