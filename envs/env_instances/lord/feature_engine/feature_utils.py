'''
Author: hanyu
Date: 2021-08-16 08:32:00
LastEditTime: 2021-08-17 13:06:38
LastEditors: hanyu
Description: feature generator utils
FilePath: /LordHD-RL/env/feature_engine/feature_utils.py
'''
from envs.env_instances.lord.utils.basic_utils import InfoSet
from envs.env_instances.lord.feature_engine.policy_labels import policy_label, action2label


def gen_action_mask(infoset: InfoSet, am_version) -> list:
    def gen_action_mask_v0(infoset):
        action_mask = [0.0] * len(policy_label)
        for action in infoset.legal_actions:
            action_mask[action2label(action)] = 1
        return action_mask

    if am_version == 0:
        return gen_action_mask_v0(infoset)
