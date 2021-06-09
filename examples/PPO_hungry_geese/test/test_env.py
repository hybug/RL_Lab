'''
Author: hanyu
Date: 2021-06-09 09:23:32
LastEditTime: 2021-06-09 09:32:14
LastEditors: hanyu
Description: test env
FilePath: /test_ppo/examples/PPO_hungry_geese/test/test_env.py
'''

from examples.PPO_hungry_geese.env import _warp_env

Env = _warp_env()
env = Env()