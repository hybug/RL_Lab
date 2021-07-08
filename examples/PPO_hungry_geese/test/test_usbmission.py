'''
Author: hanyu
Date: 2021-07-08 03:44:22
LastEditTime: 2021-07-08 03:45:37
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/examples/PPO_hungry_geese/test/test_usbmission.py
'''

from kaggle_environments import make
env = make("hungry_geese", debug=True)

env.reset(2)
env.run(['/home/jj/workspace/hanyu/RL_Lab/examples/PPO_hungry_geese/rllib_ppo/submit/submission_test.py',
         '/home/jj/workspace/hanyu/RL_Lab/examples/PPO_hungry_geese/rllib_ppo/submit/submission_test.py'])
env.render(mode="ipython", width=800, height=700)
