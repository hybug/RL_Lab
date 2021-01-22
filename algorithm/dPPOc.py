'''
Author: hanyu
Date: 2021-01-11 13:10:43
LastEditTime: 2021-01-22 09:43:36
LastEditors: hanyu
Description: distributed PPO clip algorithm
FilePath: /test_ppo/algorithm/dPPOc.py
'''

import tensorflow as tf
from module.ImportantSampling import IS_from_logits


def dPPOc(action, policy_logits, behavior_logits, advantage, clip):
    '''
    description: get the clipped ppo loss
    param {*}
    return {*}
    '''
    ratio = IS_from_logits(
        policy_logits=policy_logits,
        action=action,
        behavior_logits=behavior_logits)

    neg_loss = advantage * ratio
    if clip is not None:
        ratio_clip = tf.clip_by_value(ratio, 1.0 - clip, 1.0 + clip)
        neg_loss = tf.minimum(neg_loss, advantage * ratio_clip)
    loss = -neg_loss
    return loss
