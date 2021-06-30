'''
Author: hanyu
Date: 2021-06-30 12:37:47
LastEditTime: 2021-06-30 13:04:05
LastEditors: hanyu
Description: get submit model
FilePath: /RL_Lab/examples/PPO_hungry_geese/rllib_ppo/submit/get_submit_model.py
'''

import pickle
import bz2
import base64

model_path = '/home/jj/workspace/hanyu/RL_Lab/result/hungry_geese/rllib-hungrygeese-agent_4-gamma_0.99-reuse_3/PPO_HungryGeeseEnv_215af_00000_0_2021-06-28_08-37-53/checkpoint_001000/checkpoint-1000'

extra_data = pickle.load(open(model_path, 'rb'))
objs = pickle.loads(extra_data['worker'])

policy_0_prams = base64.b64encode(bz2.compress(pickle.dumps(objs['state']['policy_0'])))


print()