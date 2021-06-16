'''
Author: hanyu
Date: 2021-06-09 09:23:32
LastEditTime: 2021-06-16 12:43:18
LastEditors: hanyu
Description: test env
FilePath: /test_ppo/examples/PPO_hungry_geese/rllib_training/test_env.py
'''

from examples.PPO_hungry_geese.rllib_training.env import warp_env
import random

Env = warp_env()
env = Env(debug=True)

obs = env.reset(True)
is_terminal = False
while not is_terminal:
    actions = {idx: random.choice([0, 1, 2, 3])
               for idx in range(env.num_agents)}
    obs, reward, is_terminal, info = env.step(actions)
