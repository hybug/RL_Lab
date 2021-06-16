'''
Author: hanyu
Date: 2021-06-09 07:18:28
LastEditTime: 2021-06-16 12:59:04
LastEditors: hanyu
Description: environment
FilePath: /test_ppo/examples/PPO_hungry_geese/rllib_training/env.py
'''
from os import stat
import numpy as np
from enum import Enum, auto
from collections import namedtuple
from utils.get_gaes import get_gaes
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import config

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


def warp_env():
    import gym
    from kaggle_environments.envs.hungry_geese.hungry_geese import Action
    from kaggle_environments import evaluate, make, utils

    class HungryGeeseEnv(MultiAgentEnv):

        def __init__(self, worker_idx=-1, debug=False):
            """init the game environments

            Args:
                opponent (list, optional): opponent policy. Defaults to ['Random'].
                debug (bool, optional): debug flag. Defaults to False.
            """
            super(HungryGeeseEnv, self).__init__()

            self.debug = debug
            self.worker_idx = worker_idx
            self.num_previous_observations = 1
            self.num_agents = 4
            self.actions_label = [a for a in Action]
            self.num_channels = self.num_agents * 4 + 1
            self.agents = [f'geese_{i}' for i in range(self.num_agents)]

            # init the environment
            self.env = make('hungry_geese', debug=self.debug)
            self.rows = self.env.configuration.rows
            self.columns = self.env.configuration.columns
            self.hunger_rate = self.env.configuration.hunger_rate
            self.min_food = self.env.configuration.min_food
            # self.trianer = self.env.Train([None, *opponent])

            # init the action space & obs space
            self.action_space = gym.spaces.Discrete(len(self.actions_label))
            self.observation_space = gym.spaces.Box(low=CellState.EMPTY.value,
                                                    high=CellState.ANY_GOOSE.value,
                                                    shape=(self.num_previous_observations + 1,
                                                           self.rows,
                                                           self.columns), dtype=np.uint8)

            # reset the segment buffer
            self._reset_segment()
            self.is_terminal = False

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
                state_list = self._extract_obs_info(info_list)
                self.turn = 0

                # Display
                self.display_state()
                return {self.agents[i]: state for i, state in enumerate(state_list)}
            else:
                config.logging.error(
                    'The game is not over...set force True to reset the env.')

        def step(self, actions: dict):
            """step the actions dict and get the return

            Args:
                actions (dict): actions dict by {agent_idx: action_num}

            Returns:
                tuple: (observation, reward, terminal, info)
            """
            info_list = self.env.step(
                [self.actions_label[a].name for _, a in actions.items()])
            self.display_action(actions)
            self.turn += 1

            # extract state info
            state_list = self._extract_obs_info(info_list)
            # extract reward info
            reward_list = self._extract_reward_info(info_list)
            print(reward_list, info_list[0]['observation']['food'])
            # extract terminal flag info
            status_list = self._extract_status_info(info_list)
            self.is_terminal = ([True] * self.num_agents == status_list)

            # set segments
            # for agent_idx in range(self.num_agents):
            #     self._set_segment(state_list[agent_idx],
            #                       actions[agent_idx],
            #                       a_logits[agent_idx],
            #                       reward_list[agent_idx],
            #                       v_cur[agent_idx],
            #                       agent_idx)
            # get segments

            # display
            self.display_state()
            obs = {self.agents[i]: state for i, state in enumerate(state_list)}
            reward = {self.agents[i]: reward for i, reward in enumerate(reward_list)}
            terminal = {self.agents[i]: done for i, done in enumerate(status_list)}
            terminal['__all__'] = self.is_terminal
            return obs, reward, terminal, {}

        def get_history(self, force=False):
            if self.is_terminal or force:
                if self.is_terminal:
                    # using Generallized Advantage Estimator estimate advantage
                    gaes, _ = get_gaes(
                        None, self.r, self.v_cur, self.v_cur[1:], [0], 0.99, 0.95)
                    seg = Seg(self.s, self.a, self.a_logits, self.r,
                              gaes, self.v_cur, self.state_in)
                    return self.postprocess(seg)

        def _extract_obs_info(self, info_list: list) -> list:
            """extract observation info from info_list

            Args:
                info_list (list): info list returned from step/reset

            Returns:
                list: state_list
            """
            state_list = list()
            obs_dict = info_list[0].get('observation')
            for agent_idx, _ in enumerate(info_list):
                s_t = self.convert_obs_dict(obs_dict, agent_idx)
                state_list.append(s_t)
            return state_list

        def _extract_reward_info(self, info_list: list) -> list:
            """extract reward_info and return the training-reward according reward function

            Args:
                info_list (list): info list returned by step() or reset()

            Returns:
                list: reward list using for training
            """
            reward_list = list()
            for agent_idx, info_dict in enumerate(info_list):
                r_raw = info_dict['reward']
                if r_raw == 201 + 100 * (self.turn - 1):
                    r_t = -0.1
                elif r_raw == 201 + 100 * (self.turn - 2):
                    r_t = -1
                    assert info_dict['status'] == 'DONE'
                else:
                    r_t = 1
                reward_list.append(r_t)
            return reward_list

        def _extract_status_info(self, info_list: list) -> list:
            status_list = list()
            for agent_idx, info_dict in enumerate(info_list):
                if info_dict['status'] == 'DONE':
                    status_list.append(True)
                elif info_dict['status'] == 'ACTIVE':
                    status_list.append(False)
            return status_list

        def convert_obs_dict(self, obs_dict, agent_idx: int):
            state = np.zeros((self.num_channels, self.rows *
                              self.columns), dtype=np.float32)
            # set the player position
            for idx, geese_pos in enumerate(obs_dict['geese']):
                pin = 0
                # the head position of the geeses[all agents']
                for pos in geese_pos[:1]:
                    state[pin + (idx - agent_idx) % self.num_agents, pos] = 1
                pin += self.num_agents
                # the tail position of the geeses[all agents']
                for pos in geese_pos[-1:]:
                    state[pin + (idx - agent_idx) % self.num_agents, pos] = 1
                pin += self.num_agents
                # the whole position of the geeses[all agents']
                for pos in geese_pos:
                    state[pin + (idx - agent_idx) % self.num_agents, pos] = 1
            pin += self.num_agents
            # set teh previous head position
            if len(self.s[agent_idx]) > 1:
                prev_obs = self.obs[agent_idx][-1]
                for idx, geese_pos in enumerate(prev_obs['geese']):
                    for pos in geese_pos[:1]:
                        state[pin + (idx - agent_idx) %
                              self.num_agents, pos] = 1

            pin += self.num_agents
            # set food position
            for pos in obs_dict['food']:
                state[pin, pos] = 1
            return state.reshape(-1, self.rows, self.columns).transpose(1, 2, 0)

        def _reset_segment(self):
            """reset the segment
            """
            self.s = [[] for _ in range(self.num_agents)]
            self.a = [[] for _ in range(self.num_agents)]
            self.a_logits = [[] for _ in range(self.num_agents)]
            self.r = [[] for _ in range(self.num_agents)]

            self.v_cur = [[] for _ in range(self.num_agents)]

            self.obs = [[] for _ in range(self.num_agents)]

        def _set_segment(self, s, a, a_logits, r, v_cur, agent_idx):
            self.s[agent_idx].append(s)
            self.a[agent_idx].append(a)
            self.a_logits[agent_idx].append(a_logits)
            self.r[agent_idx].append(r)
            self.v_cur[agent_idx].append(v_cur)
            # self.obs[agent_idx].append(obs_dict)

        def display_state(self, force=False):
            if self.debug or force:
                print("=" * (self.columns) +
                      f" Turn {self.turn} " + "=" * (self.columns))
                print(self.env.render(mode='ansi'))

        def display_action(self, actions: dict, force=False):
            if self.debug or force:
                DIRECT_DICT = {'SOUTH': 'DOWN', 'NORTH': 'UP',
                               'EAST': 'RIGHT', 'WEST': 'LEFT'}
                for agent_idx, a in actions.items():
                    direction = self.actions_label[a].name
                    print(
                        f'Agent P{agent_idx} action: {direction}({DIRECT_DICT[direction]})')

    return HungryGeeseEnv
