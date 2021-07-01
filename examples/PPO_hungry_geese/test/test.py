'''
Author: hanyu
Date: 2021-07-01 12:02:12
LastEditTime: 2021-07-01 12:47:48
LastEditors: hanyu
Description: test
FilePath: /RL_Lab/examples/PPO_hungry_geese/test/test.py
'''
import pickle
import bz2
import base64



# policy_0_prams = base64.b64encode(bz2.compress(pickle.dumps([1])))

# ws = pickle.loads(bz2.decompress(base64.b64decode(policy_0_prams)))

b = b'QlpoOTFBWSZTWWdDTT4AAARXwGgAAAEACAACIAAgAEAAIAAiAZPUIMmI5BrYeLuSKcKEgzoaafA='
print(pickle.loads(bz2.decompress(base64.b64decode(b))))

obses = list()

def agent(obs_dict, config_dict):
    obses.append(obs_dict)
    state_list = convert_input(obses)
    output = model.forward({"obs": state_list})
    logits = output[0].numpy()[0]
    actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    return actions[np.argmax(p)]
