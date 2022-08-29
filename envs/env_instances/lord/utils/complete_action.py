'''
Author: hanyu
Date: 2021-08-09 09:18:01
LastEditTime: 2022-08-26 16:43:23
LastEditors: hanyu
Description: complete action with kicker network
FilePath: /RL_Lab/envs/env_instances/lord/utils/complete_action.py
'''
from envs.env_instances.lord.feature_engine.policy_labels import policy_label_reverse
from envs.env_instances.lord.sl_helper.sl_kick_helper import get_combo_with_kicker_NN
from envs.env_instances.lord.sl_helper.sl_feature_helper import infoset2json
from envs.env_instances.lord.utils.basic_utils import InfoSet, card_str_list_to_idx_list


def complete_action(action_label: int, infoset: InfoSet) -> list:
    action_pred = policy_label_reverse[action_label]
    if 'X' in action_pred:
        # Nedd kicker cards
        json_data = infoset2json(infoset)
        action_pred = get_combo_with_kicker_NN(action_pred, json_data)
        action_str_list = [c[0] if c in ['RJ', 'BJ'] else c[-1]
                           for c in action_pred.split(' ')]
        action_idx_list = card_str_list_to_idx_list(action_str_list)
    elif action_pred == 'Pass':
        action_idx_list = list()
    else:
        action_idx_list = card_str_list_to_idx_list(action_pred.split(' '))
    action_idx_list.sort()
    return action_idx_list
