'''
Author: hanyu
Date: 2022-07-19 16:19:34
LastEditTime: 2022-07-22 17:03:10
LastEditors: hanyu
Description: environment
FilePath: /RL_Lab/envs/env.py
'''

from configs.config_base import EnvParams
from envs.env_base import EnvBase


class Env(EnvBase):
    def __init__(self, env_params: EnvParams, worker_id: int = 0) -> None:
        super().__init__(env_params, worker_id)

    def reset(self):
        """Reset the environment.

        Returns:
            np.ndarray: observation
        """
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()
