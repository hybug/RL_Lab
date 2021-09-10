'''
Author: hanyu
Date: 2021-06-22 12:46:08
LastEditTime: 2021-07-13 09:27:41
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
                # normalize the advantage
                self.old_current_value = kwargs.get('v_cur')
                self.ret = advantage + self.old_current_value

                self.a_t = kwargs.get('a')
                self.behavior_logits = kwargs.get('a_logits')
                self.r_t = kwargs.get('r')

                self.adv_mean = tf.reduce_mean(advantage, axis=[0, 1])
                advantage -= self.adv_mean
                self.adv_std = tf.math.sqrt(
                    tf.reduce_mean(advantage ** 2, axis=[0, 1]))
                self.advantage = advantage / tf.maximum(self.adv_std, 1e-12)

                self.slots = tf.cast(kwargs.get('slots'), tf.float32)

                # use reward prediction
                # TODO
                # use pixel reconstruction
                # TODO
                # use piexel control
                # TODO

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
            shape = get_tensor_shape(s)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                input = tf.reshape(s, [-1] + shape[-3:])
                filter = [16, 32, 32]
                kernel = [(3, 3), (3, 3), (5, 3)]
                stride = [(1, 2), (1, 2), (2, 1)]
                for i in range(len(filter)):
                    input = tf.layers.conv2d(
                        input,
                        filters=filter[i],
                        kernel_size=kernel[i][0],
                        strides=stride[i][0],
                        padding='valid',
                        activation=None,
                        name=f'conv_{i}'
                    )
                    input = tf.layers.max_pooling2d(
                        input,
                        pool_size=kernel[i][1],
                        strides=stride[i][1],
                        padding='valid',
                        name=f'maxpool_{i}'
                    )
                    input = self.residual_block()
                input = tf.nn.relu(input)

                new_shape = get_tensor_shape(input)
                image_feature = tf.reshape()

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

        @staticmethod
        def residual_block(input, scope):
            shape = get_tensor_shape(input)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                output = tf.nn.relu(input)
                output = tf.layers.conov2d(
                    output,
                    filters=shape[-1],
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=None,
                    name='conv0'
                )
                output = tf.nn.relu(output)
                output = tf.layers.conv2d(
                    output,
                    filters=shape[-1],
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=None,
                    name='conv1'
                )
                output = output + input
            return output
        
    return PolicyModel