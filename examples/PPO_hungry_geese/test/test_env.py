'''
Author: hanyu
Date: 2021-06-09 09:23:32
LastEditTime: 2021-06-16 09:08:09
LastEditors: hanyu
Description: test env
FilePath: /test_ppo/examples/PPO_hungry_geese/test/test_env.py
'''

from examples.PPO_hungry_geese.env import _warp_env
import random

Env = _warp_env()
env = Env(debug=True)

obs = env.reset(True)
is_terminal = False
while not is_terminal:
    actions = {idx: random.choice([0, 1, 2, 3])
               for idx in range(env.num_agents)}
    obs, reward, is_terminal, info = env.step(actions,
                                              [0] * env.num_agents,
                                              [0] * env.num_agents)
