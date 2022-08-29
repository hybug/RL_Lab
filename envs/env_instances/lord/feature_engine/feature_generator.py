'''
Author: hanyu
Date: 2021-08-11 08:10:17
LastEditTime: 2022-08-29 16:35:16
LastEditors: hanyu
Description: feature generator
FilePath: /RL_Lab/envs/env_instances/lord/feature_engine/feature_generator.py
'''
import numpy as np


def get_observation(infoset, obs_shape: list):
    def get_observation_v0(infoset):
        if infoset is None:
            return dict()
        from envs.env_instances.lord.sl_helper.sl_feature_helper import infoset2json, json2feature
        from envs.env_instances.lord.feature_engine.feature_utils import gen_action_mask

        # Generate state feature
        json_data = infoset2json(infoset)
        feature_data = json2feature(json_data)
        state_arr = np.array(feature_data).flatten().reshape(
            obs_shape[-1], 4, obs_shape[1]).transpose(1, 2, 0)

        # Generate action mask
        action_mask = gen_action_mask(
            infoset=infoset, am_version=0)  # TODO

        obs_dict = {"state": state_arr, "action_mask": action_mask}

        return obs_dict

    return get_observation_v0(infoset)
