'''
Author: hanyu
Date: 2021-01-22 09:44:38
LastEditTime: 2021-01-22 09:52:10
LastEditors: hanyu
Description: algorithm
FilePath: /test_ppo/algorithm/dPPOcC.py
'''

from collections import namedtuple
from algorithm.dPPOc import dPPOc
from module.mse import mse

PPOcCloss = namedtuple('PPOcCloss', ['p_loss', 'v_loss'])


def dPPOcC(act, policy_logits, behavior_logits, advantage, policy_clip, vf, vf_target, value_clip, old_vf):
    a_loss = dPPOc(action=act,
                   policy_logits=policy_logits,
                   behavior_logits=behavior_logits,
                   advantage=advantage,
                   clip=policy_clip)
    v_loss = mse(y_hat=vf,
                 y_target=vf_target,
                 clip=value_clip,
                 clip_center=old_vf)
    return PPOcCloss(a_loss, v_loss)
