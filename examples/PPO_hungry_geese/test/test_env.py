'''
Author: hanyu
Date: 2021-06-09 09:23:32
LastEditTime: 2021-07-02 09:36:08
LastEditors: hanyu
Description: test env
FilePath: /RL_Lab/examples/PPO_hungry_geese/test/test_env.py
'''
import random

import ray
import ray.rllib.agents.ppo as ppo
from examples.PPO_hungry_geese.rllib_ppo.env import warp_env
from examples.PPO_hungry_geese.rllib_ppo.MaskedResNetClass import \
    MaskedResidualNetwork
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

Env = warp_env()
env = Env(debug=True)
ray.init()

def env_creator(env_config):
    HungryGeeseEnv = warp_env()
    return HungryGeeseEnv()


def policy_mapping_fn(agent_id):
    pol_id = random.choice(policy_ids)
    return pol_id


def gen_policy(i):
    env = env_creator({})
    config = {
        "model": {
            "custom_model": ["MaskedResNet1", "MaskedResNet2"][i % 2],
        },
        "gamma": random.choice([0.95, 0.99]),
    }
    return (None, env.observation_space, env.action_space, config)


policies = {
    f'policy_{i}': gen_policy(i) for i in range(4)
}
policy_ids = list(policies.keys())


register_env("HungryGeeseEnv", env_creator)

ModelCatalog.register_custom_model(
    "MaskedResNet1", MaskedResidualNetwork)
ModelCatalog.register_custom_model(
    "MaskedResNet2", MaskedResidualNetwork)
model_config = {
    'env': 'HungryGeeseEnv',
    "num_workers": 0,
    'model': {
        'custom_model_config': {'conv_activation': 'relu',
                                'conv_filters': [(64, 1, 1, 4), (128, 1, 1, 3), (256, 1, 1, 2), (128, (7, 11), 1, 1)],
                                'no_final_linear': False,
                                'vf_share_layers': False,
                                'rows': 7,
                                'columns': 11,
                                'channels': 17}
    },
    'multiagent': {
        'policies': policies,
        'policy_mapping_fn': policy_mapping_fn
    }
}


model_path = '/home/jj/workspace/hanyu/RL_Lab/result/hungry_geese/rllib-hungrygeese-agent_4-gamma_0.99-reuse_3/PPO_HungryGeeseEnv_1c317_00000_0_2021-06-30_07-09-28/checkpoint_002000/checkpoint-2000'
trainer = ppo.PPOTrainer(config=model_config)
trainer.restore(model_path)

obs = env.reset(True)
is_terminal = False
while not is_terminal:
    action = trainer.compute_action(obs, explore=False)
    obs, reward, is_terminal, info = env.step({0: action})
