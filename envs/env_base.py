'''
Author: hanyu
Date: 2022-07-19 11:30:23
LastEditTime: 2022-07-20 15:04:54
LastEditors: hanyu
Description: env base
FilePath: /RL_Lab/envs/env_base.py
'''

from configs.config_base import EnvParams
from envs.game_envs.env_football.football import Football


def _init_env(env_params: EnvParams, seed: int, worker_id: int):

    if "football" in env_params.env_name:
        env_params.env_name = env_params.env_name.split('-')[-1]
        env = Football(env_params=env_params, worker_idx=worker_id)
        env_type = 'football'
    else:
        env = None
        env_type = None
    return env, env_type


class EnvBase():
    def __init__(self, env_params: EnvParams, worker_id: int = 0) -> None:

        # Restore variables from EnvParams
        self._env_name = env_params.env_name
        self._seed = env_params.seed
        self._is_frame_stacking = env_params.frame_stacking
        self._stack_size = env_params.frames_stack_size

        # Initialize some picture infos
        self._act_is_discrete = False
        self._obs_is_visual = False
        self._obs_is_vector = False
        self._obs_is_grayscale = False
        self._obs_shapes = dict()
        self._env, self._env_type = _init_env(env_params, self._seed,
                                              worker_id)

        if self._env_type == 'football':
            self._act_is_discrete = not self._env.is_act_continuous
            self._obs_is_visual = False
            self._obs_is_vector = True
            self._obs_is_grayscale = False

            self._act_shape = self._env.action_dim

    @property
    def env_info(self):
        return self._env.env_info

    def reset(self):
        pass

    def step(self):
        pass
