'''
Author: hanyu
Date: 2021-06-09 07:18:28
LastEditTime: 2021-06-09 13:05:39
LastEditors: hanyu
Description: environment
FilePath: /test_ppo/examples/PPO_hungry_geese/env.py
'''
import numpy as np
from enum import Enum, auto
from collections import namedtuple

Seg = namedtuple('Seg', ['s', 'a', 'a_logits',
                         'r', 'gaes', 'v_cur', 'state_in'])


class CellState(Enum):
    EMPTY = 0
    FOOD = auto()
    HEAD = auto()
    BODY = auto()
    TAIL = auto()
    MY_HEAD = auto()
    MY_BODY = auto()
    MY_TAIL = auto()
    ANY_GOOSE = auto()


def _warp_env():
    import gym
    from kaggle_environments.envs.hungry_geese.hungry_geese import Action
    from kaggle_environments import evaluate, make, utils

    class HungryGeeseEnv(gym.Env):

        def __init__(self, opponent=['Random'], debug=False):
            """init the game environments

            Args:
                opponent (list, optional): opponent policy. Defaults to ['Random'].
                debug (bool, optional): debug flag. Defaults to False.
            """
            super(HungryGeeseEnv, self).__init__()

            self.debug = debug
            self.num_previous_observations = 1
            self.num_agents = 2
            self.actions = [a for a in Action]
            self.num_channels = self.num_agents * 4 + 1

            # init the environment
            self.env = make('hungry_geese', debug=self.debug)
            self.rows = self.env.configuration.rows
            self.columns = self.env.configuration.columns
            self.hunger_rate = self.env.configuration.hunger_rate
            self.min_food = self.env.configuration.min_food
            self.trianer = self.env.Train([None, *opponent])

            # init the action space & obs space
            self.action_space = gym.spaces.Discrete(len(self.actions))
            self.observation_space = gym.spaces.Box(low=CellState.EMPTY.value,
                                                    high=CellState.ANY_GOOSE.value,
                                                    shape=(self.num_previous_observations + 1,
                                                           self.rows,
                                                           self.columns), dtype=np.uint8)

            # reset the segment buffer
            self._reset_segment()

        def reset(self, force=False):
            if self.is_terminal or force:
                # [{
                #     'action': 'NORTH',
                #     'reward': 0,
                #     'info': {},
                #     'observation': {
                #         'remainingOverageTime': 60,
                #         'step': 0,
                #         'geese': [[63], [12], [74], [48]],
                #         'food': [11, 0],
                #         'index': 0
                #     },
                #     'status': 'ACTIVE'}, ...]
                info_list = self.env.reset(self.num_agents)
                state_list = list()
                for agent_idx, info_dict in enumerate(info_list):
                    obs_dict = info_dict.get('observation')
                    s_t = self.convert_info_dict(obs_dict, agent_idx)
                    state_list.append(s_t)
                return state_list

        def convert_obs_dict(self, obs_dict, agent_idx: int):
            state = np.zeros((self.num_channels, self.rows *
                              self.columns), dtype=np.float32)
            # set the player position
            for idx, geese_pos in enumerate(obs_dict['geese']):
                # the head position of the geeses[all agents']
                for pos in geese_pos[:1]:
                    state[0 + (idx - agent_idx) % self.num_agents, pos] = 1
                # the tail position of the geeses[all agents']
                for pos in geese_pos[-1:]:
                    state[4 + (idx - agent_idx) % self.num_agents, pos] = 1
                # the whole position of the geeses[all agents']
                for pos in geese_pos:
                    state[8 + (idx - agent_idx) % self.num_agents, pos] = 1
            # set teh previous head position
            if len(self.s[agent_idx]) > 1:
                prev_obs = self.obs[agent_idx][-1]
                for idx, geese_pos in enumerate(prev_obs['geese']):
                    for pos in geese_pos[:1]:
                        state[12 + (idx - agent_idx) %
                              self.num_agents, pos] = 1

            # set food position
            for pos in obs_dict['food']:
                state[16, pos] = 1
            return state.reshape(-1, 7, 11)

        def _reset_segment(self):
            """reset the segment
            """
            self.s = [[] for _ in range(self.num_agents)]
            self.a = [[] for _ in range(self.num_agents)]
            self.a_logits = [[] for _ in range(self.num_agents)]
            self.r = [[] for _ in range(self.num_agents)]

            self.v_cur = [[] for _ in range(self.num_agents)]

            self.obs = [[] for _ in range(self.num_agents)]

        def _set_segment(self, s, a, a_logits, r, v_cur, obs_dict, agent_idx):
            self.s[agent_idx].append(s)
            self.a[agent_idx].append(a)
            self.a_logits[agent_idx].append(a_logits)
            self.r[agent_idx].append(r)
            self.v_cur[agent_idx].append(v_cur)
            self.obs[agent_idx].append(obs_dict)

        def step(self):
            pass

    return HungryGeeseEnv
