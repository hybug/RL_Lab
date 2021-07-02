'''
Author: hanyu
Date: 2021-07-01 12:02:12
LastEditTime: 2021-07-02 08:36:10
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/examples/PPO_hungry_geese/test/test_submission.py
'''
from kaggle_environments import make
from examples.PPO_hungry_geese.rllib_ppo.submit.submission import agent

env = make('hungry_geese', debug=False)
obs = env.reset(1)

action = agent(obs[0]['observation'], None)
print(action)
