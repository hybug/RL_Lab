'''
Author: hanyu
Date: 2021-06-09 09:23:32
LastEditTime: 2021-06-21 03:37:20
LastEditors: hanyu
Description: test env
FilePath: /test_ppo/examples/PPO_hungry_geese/rllib_ppo/test_env.py
'''

from examples.PPO_hungry_geese.rllib_ppo.env import warp_env
import random

Env = warp_env()
env = Env(debug=True)

for _ in range(100):
    obs = env.reset(True)
    is_terminal = {'__all__': False}
    while not is_terminal['__all__']:
        actions = {f'geese_{idx}': random.choice([0, 1, 2, 3])
                   for idx in range(len(obs))}
        obs, reward, is_terminal, info = env.step(actions)
