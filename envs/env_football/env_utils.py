'''
Author: hanyu
Date: 2022-07-19 16:21:21
LastEditTime: 2022-07-19 17:50:14
LastEditors: hanyu
Description: env utils
FilePath: /RL_Lab/envs/env_utils.py
'''

from configs.config_base import EnvParams
from envs.batched_env import BatchedEnv
from envs.env import Env


def create_batched_env(num_envs: int,
                       env_params: EnvParams,
                       start: int = 0) -> BatchedEnv:
    """Create batched environment

    Args:
        num_envs (int): number of envs

    Returns:
        list: n environments each running in its own process
    """
    env_funcs = [
        lambda i=i: create_single_env(env_params, i + start)
        for i in range(num_envs)
    ]
    return BatchedEnv(env_funcs)


def create_single_env(env_params: EnvParams, worker_idx: int) -> Env:
    env = Env(env_params=env_params, worker_id=worker_idx)
    return env
