'''
Author: hanyu
Date: 2022-08-26 16:51:24
LastEditTime: 2022-08-29 16:10:30
LastEditors: hanyu
Description: lord wrappers
FilePath: /RL_Lab/envs/wrappers/lord_wrappers.py
'''
import gym
from gym import spaces
import numpy as np

from envs.env_instances.lord.feature_engine.feature_generator import get_observation


class WarpObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_shape: list, act_shape: list):
        super().__init__(env)
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.observation_space = spaces.Dict({
            "obs":
            spaces.Box(low=0, high=1, shape=(obs_shape), dtype=np.float32),
            "am":
            spaces.Box(low=0, high=1, shape=(act_shape), dtype=np.float32)
        })

    def observation(self, infoset):
        return get_observation(infoset=infoset, obs_shape=self.obs_shape)


def wrap_lord(env, obs_shape=[4, 15, 13], act_shape=[309]):
    env = WarpObservation(env, obs_shape, act_shape)
    return env
