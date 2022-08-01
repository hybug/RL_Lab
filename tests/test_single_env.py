'''
Author: hanyu
Date: 2022-07-29 16:04:22
LastEditTime: 2022-07-29 18:26:05
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/tests/test_single_env.py
'''
from random import choice
import gfootball.env as football_env
import gfootball
from config import BASEDIR

env = football_env.create_environment(env_name="11_vs_11_easy_stochastic",
                                      stacked=True,
                                      rewards="scoring,checkpoints",
                                      write_goal_dumps=False,
                                      logdir=BASEDIR + "/logs/game_log/",
                                    #   write_full_episode_dumps=True,
                                      render=False,
                                      dump_frequency=0)

obs = env.reset()
for i in range(10):
    action = choice(range(19))
    # if i < 2:
    #     action = 5

    obs, rew, done, info = env.step(action)
    print(action, rew)