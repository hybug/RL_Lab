'''
Author: hanyu
Date: 2021-06-09 07:18:28
LastEditTime: 2021-07-22 09:30:32
LastEditors: hanyu
Description: environment
FilePath: /RL_Lab/examples/PPO_hungry_geese/rllib_ppo/env.py
'''
from logging import info
import numpy as np
from enum import Enum, auto
from collections import namedtuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv

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
    from kaggle_environments import make

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
            self.agents = [f'policy_{i}' for i in range(self.num_agents)]

            # init the environment
            self.env = make('hungry_geese', debug=self.debug)
            self.rows = self.env.configuration.rows
            self.columns = self.env.configuration.columns
            self.hunger_rate = self.env.configuration.hunger_rate
            self.min_food = self.env.configuration.min_food
            self.info_list = None
            # self.trianer = self.env.Train([None, *opponent])

            # init the action space & obs space
            self.action_space = gym.spaces.Discrete(len(self.actions_label))
            self.observation_space = gym.spaces.Box(low=CellState.EMPTY.value,
                                                    high=CellState.ANY_GOOSE.value,
                                                    shape=(self.rows,
                                                           self.columns,
                                                           self.num_channels), dtype=np.uint8)

            # reset the segment buffer
            self._reset_segment()

        def reset(self, force=False):
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
            self.info_list = info_list
            self.geese_length = [0 for _ in range(self.num_agents)]
            self.turn = 0
            # alive for False, dead for True
            self.terminal_status_list = [False] * self.num_agents
            self.last_reward = [101] * self.num_agents
            self.is_terminal = False

            state_list = self._extract_obs_info(info_list)

            # Display
            self.display_state()
            return {self.agents[i]: state for i, state in enumerate(state_list)}

        def step(self, actions: dict):
            """step the actions dict and get the return

            Args:
                actions (dict): actions dict by {agent_idx: action_num}

            Returns:
                tuple: (observation, reward, terminal, info)
            """
            def __complete_actions(actions):
                ret = dict()
                for idx in range(self.num_agents):
                    if f'policy_{idx}' in actions.keys():
                        ret[f'policy_{idx}'] = actions[f'policy_{idx}']
                    else:
                        ret[f'policy_{idx}'] = -1
                return ret

            info_list = self.env.step(
                [self.actions_label[a].name for _, a in __complete_actions(actions).items()])
            self.info_list = info_list
            for agent_idx in range(self.num_agents):
                if self.info_list[0]['observation']['geese'][agent_idx]:
                    self.geese_length[agent_idx] = len(self.info_list[0]['observation']['geese'][agent_idx])
            self.display_action(actions)
            self.turn += 1

            # extract state info
            state_list = self._extract_obs_info(info_list)
            # extract reward info
            reward_list = self._extract_reward_info(info_list)
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
            # extract multi-agents' return
            obs = {self.agents[i]: state for i, state in enumerate(
                state_list) if state is not None}
            reward = {self.agents[i]: reward for i,
                      reward in enumerate(reward_list) if reward is not None}
            terminal = {self.agents[i]: done for i,
                        done in enumerate(status_list)}
            terminal['__all__'] = self.is_terminal

            # update agents' status & last reward
            self.terminal_status_list = status_list
            self.last_reward = [info_list[i]['reward']
                                for i in range(self.num_agents)]
            # print('return info: ', obs.keys(), reward, terminal)
            # print('raw reward: ', self.last_reward)
            # print('reward:', list(reward.values()))
            # print('terminal: ', self.terminal_status_list)
            assert obs.keys() == reward.keys()
            return obs, reward, terminal, {}


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
                if not self.terminal_status_list[agent_idx]:
                    state_list.append(s_t)
                else:
                    state_list.append(None)
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
                if info_list[agent_idx]['status'] == 'DONE':
                    # dead or collision
                    r_t = -1
                else:
                    if r_raw % 100 - self.last_reward[agent_idx] % 100 == 1:
                        r_t = 1
                    elif r_raw % 100 - self.last_reward[agent_idx] % 100 == 0:
                        r_t = -0.1
                    else:
                        assert r_raw % 100 - self.last_reward[agent_idx] % 100 == -1
                        r_t = -0.5

                if not self.terminal_status_list[agent_idx]:
                    reward_list.append(r_t)
                else:
                    reward_list.append(None)
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
            input_dict = dict()
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
            obs = state.reshape(-1, self.rows, self.columns).transpose(1, 2, 0)

            # # avail actions
            # avail_actions = [0] * len(self.actions_label)
            input_dict['obs'] = obs
            return obs

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
