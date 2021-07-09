'''
Author: hanyu
Date: 2021-06-22 12:46:08
LastEditTime: 2021-07-09 09:01:59
LastEditors: hanyu
Description: policy network of PPO
FilePath: /RL_Lab/examples/PPO_hungry_geese/ray_ppo/policy_model.py
'''
from utils.get_tensor_shape import get_tensor_shape
from utils.infer_utils.draw_categorical_samples import draw_categorical_samples
from ray_helper.miscellaneous import tf_model_ws


def wrap_model():
    """warpper of policy model
    """
    import tensorflow as tf
    from utils.get_tensor_shape import get_tensor_shape

    @tf_model_ws
    class PolicyModel(object):
        def __init__(self,
                     act_space,
                     scope='agent',
                     **kwargs) -> None:
            self.act_space = act_space

            self.s_t = kwargs.get('s')
            self.prev_r = kwargs.get('r')

            # TODO feature net

            # Actor Network
            self.current_action_logits = self.actor_net(
                self.s_t, scope + '_current_actor')
            # random sample according to logits
            # output shape: [N]
            self.current_action = tf.squeeze(
                draw_categorical_samples(self.current_action_logits), axis=-1)

            # Critic Network
            self.current_value = tf.critic_net(self.s_t, '_current_critic')

            advantage = kwargs.get('adv', None)
            if advantage is not None:
                # calculate the mean advantage
                self.old_current_value = kwargs.get('v_cur')
                self.ret = advantage + self.old_current_value

        def feature_net(self, s, prev_a, prev_r, scope='feature') -> tuple:
            """feature extract network

            Args:
                s (tensor): state tensor
                prev_a (tensor): previous action
                prev_r (tensor): previous reward
                scope (str, optional): op's scope. Defaults to 'feature'.

            Returns:
                tuple: extracted feature
            """
            # TODO
            pass

        def actor_net(self, s, scope):
            """actor network

            Args:
                s (Tensor): state tensor
                scope (str): name scope
            """
            input = s
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                input = tf.layers.dense(input,
                                        get_tensor_shape(s)[-1],
                                        tf.nn.rule,
                                        name='dense')
                act_logits = tf.layers.dense(
                    input,
                    self.act_space,
                    activation=None,
                    name='a_logits')
            return act_logits

        def critic_net(self, s, scope):
            """critic network

            Args:
                s (Tensor): state tensor
                scope (str): name scope
            """
            input = s
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                input = tf.layers.dense(input,
                                        get_tensor_shape(s)[-1],
                                        tf.nn.rule,
                                        name='dense')
                v_value = tf.layers.dense(
                    input,
                    self.act_space,
                    activation=None,
                    name='v_value')
            return v_value
