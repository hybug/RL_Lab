'''
Author: hanyu
Date: 2022-07-29 16:04:22
LastEditTime: 2022-08-29 17:43:49
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/tests/test_single_env.py
'''
from random import choice
from envs.env_instances.lord.lord_env import LordEnv
from envs.wrappers.lord_wrappers import wrap_lord

env = LordEnv(silence_mode=False)
env = wrap_lord(env)

for _ in range(1):
    done = False
    obs = env.reset()
    while not done:
        am = obs['action_mask']
        a = choice([i for i in range(len(am)) if am[i] != 0])
        obs, reward, done, info = env.step(a)
        print(f'reward: {reward}')
