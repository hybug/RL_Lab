'''
Author: hanyu
Date: 2021-06-09 09:23:32
LastEditTime: 2021-07-08 06:08:11
LastEditors: hanyu
Description: test env
FilePath: /RL_Lab/examples/PPO_hungry_geese/test/test_env.py
'''
import random
import numpy as np
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

label = ['NORTH', 'EAST', 'SOUTH', 'WEST']

model_path = '/home/jj/workspace/hanyu/RL_Lab/result/hungry_geese/rllib-hungrygeese-agent_4-gamma_0.99-reuse_3/PPO_HungryGeeseEnv_1c317_00000_0_2021-06-30_07-09-28/checkpoint_002000/checkpoint-2000'
trainer = ppo.PPOTrainer(config=model_config)
trainer.restore(model_path)

NUM_AGENTS = 4



def convert_input(obses):
    state = np.zeros((4 * NUM_AGENTS + 1, 7 * 11), dtype=np.float32)
    obs = obses[-1]
    agent_idx = obs['index']

    # set the player position
    for idx, geese_pos in enumerate(obs['geese']):
        pin = 0
        # the head position of the geeses[all agents']
        for pos in geese_pos[:1]:
            state[pin + (idx - agent_idx) % NUM_AGENTS, pos] = 1
        pin += NUM_AGENTS
        # the tail position of the geeses[all agents']
        for pos in geese_pos[-1:]:
            state[pin + (idx - agent_idx) % NUM_AGENTS, pos] = 1
        pin += NUM_AGENTS
        # the whole position of the geeses[all agents']
        for pos in geese_pos:
            state[pin + (idx - agent_idx) % NUM_AGENTS, pos] = 1
    pin += NUM_AGENTS
    # set teh previous head position
    if len(obses) > 1:
        prev_obs = obses[-2]
        for idx, geese_pos in enumerate(prev_obs['geese']):
            for pos in geese_pos[:1]:
                state[pin + (idx - agent_idx) %
                      NUM_AGENTS, pos] = 1

    pin += NUM_AGENTS
    # set food position
    for pos in obs['food']:
        state[pin, pos] = 1
    obs = state.reshape(-1, 7, 11).transpose(1, 2, 0)
    return obs


obs = env.reset(True)
is_terminal = False
agent_obses = list()
while not is_terminal:
    action_dict = dict()
    for agent_idx in range(4):
        agent_obses.append(env.info_list[0]['observation'])
        if f'policy_{agent_idx}' in obs.keys():
            action = trainer.compute_action(
                obs[f'policy_{agent_idx}'], explore=False, policy_id=f'policy_{agent_idx}')
            action_0 = label[action]
            from examples.PPO_hungry_geese.rllib_ppo.submit.submission import model
            action_1 = label[np.argmax(model.forward(
                {"obs": [obs[f'policy_{agent_idx}'].tolist()]})[0].numpy())]
            from examples.PPO_hungry_geese.rllib_ppo.submit.submission import agent
            from examples.PPO_hungry_geese.rllib_ppo.submit.submission import convert_input
            action_3 = agent(env.info_list[0]['observation'], None)

            action_dict[f'policy_{agent_idx}'] = action
    obs, reward, is_terminal, info = env.step(action_dict)
    is_terminal = is_terminal['__all__']
