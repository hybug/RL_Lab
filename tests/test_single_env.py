'''
Author: hanyu
Date: 2022-07-29 16:04:22
LastEditTime: 2022-08-23 21:18:24
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/tests/test_single_env.py
'''
from random import choice
import gym
from envs.wrappers.atari_wrappers import is_atari, wrap_deepmind
# env = gym.make(f"ALE/{env_params.env_name}-v5")
env = gym.make("BeamRiderNoFrameskip-v4")
if is_atari(env):
    env = wrap_deepmind(env)
    env_type = 'gym-atari-deepmind'

done = False
env.reset()
while not done:
    a = choice(range(8))
    obs, reward, done, info = env.step(a)
    print(reward)