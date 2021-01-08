'''
Author: hanyu
Date: 2021-01-06 10:13:41
LastEditTime: 2021-01-08 09:18:40
LastEditors: hanyu
Description: policy network of PPO
FilePath: /test_ppo/examples/PPO_super_mario_bros/policy_graph.py
'''
from ray_helper.miscellaneous import tf_model_ws


def warp_Model():
    '''
    description: warp the policy model
    param {*}
    return {Object: policy model}
    '''
    import tensorflow as tf
    from infer import categorical
    from utils import get_shape

    @tf_model_ws
    class Model(object):
        def __init__(self,
                     act_space,
                     rnn,
                     use_rmc,
                     use_hrnn,
                     use_reward_prediction,
                     after_rnn,
                     use_pixel_control,
                     user_pixel_reconstructin,
                     scope='agent',
                     **kwargs):
            self.act_space = act_space
            self.use_rmc = use_rmc
            self.use_hrnn = use_hrnn
            self.scope = scope

            self.s_t = kwargs.get('s')
            self.prev_actions = kwargs.get('prev_a')
            self.prev_r = kwargs.get('prev_r')
            self.state_in = kwargs.get('state_in')

            prev_a = tf.one_hot(self.prev_actions,
                                depth=act_space, dtype=tf.float32)

            self.feature, self.cnn_feature, self.image_feature, self.state_out = self.feature_net(
                self.s_t, rnn, prev_a, self.prev_r, self.state_in, scope + '_current_feature')

            if use_hrnn:
                # TODO
                pass

            self.current_act_logits = self.a_net(
                self.feature, scope + '_acurrent')
            self.current_act = tf.squeeze(
                categorical(self.current_act_logits), axis=-1)

    return Model
