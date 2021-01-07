'''
Author: hanyu
Date: 2021-01-06 10:13:41
LastEditTime: 2021-01-07 12:30:09
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
            self.use_hrnn = hrnn
            self.scope = scope

    return Model
