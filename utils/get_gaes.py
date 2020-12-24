'''
Author: hanyu
Date: 2020-11-07 08:38:39
LastEditTime: 2020-11-07 08:46:22
LastEditors: hanyu
Description: get gae/advantage
FilePath: /test_ppo/utils/get_gaes.py
'''

import copy


def get_gaes(deltas, rewards, state_values, next_state_values, GAMMA, LAMBDA):
    assert (deltas is not None) or ((rewards is not None) and (
        state_values is not None) and (next_state_values is not None))

    if deltas is None:
        deltas = [r_t + GAMMA * next_v - v for r_t, next_v,
                  v in zip(rewards, next_state_values, state_values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
    return gaes, deltas
