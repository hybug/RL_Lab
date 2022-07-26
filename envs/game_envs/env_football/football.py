# -*- coding:utf-8  -*-
# Time  : 2020/12/28 16:33
# Author: Yahui Cui
import copy
import json

import gfootball.env as football_env
import numpy as np
from configs.config_base import EnvParams
from envs.game_envs.env_football.obs_interfaces.observation import DictObservation
from envs.game_envs.env_football.simulators.game import Game
from envs.game_envs.env_football.utils.box import Box
from envs.game_envs.env_football.utils.discrete import Discrete
from get_logger import BASEDIR


class Football(Game, DictObservation):
    def __init__(self, env_params: EnvParams, worker_idx: int):
        conf = {
            "n_player": env_params.n_player,
            "is_obs_continuous": env_params.is_obs_continuous,
            "is_act_continuous": env_params.is_act_continuous,
            "game_name": env_params.env_name,
            "agent_nums": env_params.agent_nums,
            "obs_type": env_params.obs_type,
            "max_step": env_params.max_step,
            "act_box": env_params.act_box
        }
        super().__init__(conf['n_player'], conf['is_obs_continuous'],
                         conf['is_act_continuous'], conf['game_name'],
                         conf['agent_nums'], conf['obs_type'])

        self._worker_idx = worker_idx
        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        self.env_core = football_env.create_environment(
            env_name=conf["game_name"],
            stacked=False,
            representation='extracted',
            rewards="scoring",
            logdir=BASEDIR + '/logs/football_log/',
            write_goal_dumps=False,
            write_full_episode_dumps=True,
            render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=self.agent_nums[0],
            number_of_right_players_agent_controls=self.agent_nums[1])

        self.load_action_space(conf)
        obs_list = self.env_core.reset()
        self.won = {}
        self.joint_action_space = self.set_action_space()
        self.current_state = self.get_sorted_next_state(obs_list)
        self.all_observes = self.current_state
        self.n_return_temp = [0.0] * self.n_player
        self.n_return = [0.0] * self.n_player

        self.action_dim = self.get_action_dim()
        self.init_info = None

    def load_action_space(self, conf):
        if "act_box" in conf:
            input_action = json.loads(conf["act_box"]) if isinstance(
                conf["act_box"], str) else conf["act_box"]
            # print(input_action)
            if self.is_act_continuous:
                if ("high" not in input_action) or (
                        "low" not in input_action) or ("shape"
                                                       not in input_action):
                    raise Exception(
                        "act_box in continuous case must have fields low, high, shape"
                    )
                shape = tuple(input_action["shape"])
                self.env_core.action_space = Box(input_action["low"],
                                                 input_action["high"], shape,
                                                 np.float32)
            else:
                if "discrete_n" not in input_action:
                    raise Exception(
                        "act_box in discrete case must have field discrete_n")
                discrete_n = int(input_action["discrete_n"])
                self.env_core.action_space = Discrete(discrete_n)

    def step(self, action):
        # action = self.decode(joint_action)
        info = None
        info_before = self.step_before_info()
        next_state, reward, self.done, info_after = self.get_next_state(action)
        sorted_next_state = self.get_sorted_next_state(next_state)
        self.current_state = sorted_next_state
        self.all_observes = self.current_state
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        # reward = self.get_reward(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, None

    def close(self):
        self.env_core.close()

    def decode(self, joint_action):
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()
        joint_action_decode = []
        for action in joint_action:
            joint_action_decode.append(action[0].index(1))
        return joint_action_decode

    def get_next_state(self, action):
        observation, reward, done, info = self.env_core.step(action)
        return observation, reward, done, info

    # def get_reward(self, reward):
    #     r = [0] * self.n_player
    #     for i in range(self.n_player):
    #         r[i] = reward[i]
    #         self.n_return_temp[i] += r[i]

    #     # left n_return
    #     self.n_return[0] = self.n_return_temp[0]
    #     # right n_return
    #     self.n_return[self.agent_nums[0]] = self.n_return_temp[
    #         self.agent_nums[0]]

    #     return r

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        # if self.done:
        #     self.env_core.close()

        return self.done

    def set_action_space(self):
        action_space = [[self.env_core.action_space]
                        for _ in range(self.n_player)]
        return action_space

    def check_win(self):
        left_sum = self.n_return[0]
        right_sum = self.n_return[self.agent_nums[0]]
        if left_sum > right_sum:
            return '0'
        elif left_sum < right_sum:
            return '1'
        else:
            return '-1'

    def reset(self):
        obs_list = self.get_sorted_next_state(self.env_core.reset())
        self.step_cnt = 0
        self.done = False
        self.current_state = obs_list
        self.all_observes = self.current_state
        return self.all_observes, [0] * self.n_player, self.is_terminal(), []

    def get_action_dim(self):
        action_dim = 1
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space)):
            action_dim *= self.joint_action_space[i][0].n

        return action_dim

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_render_data(self, current_state):
        return []

    def get_dict_observation(self, current_state, player_id, info_before):
        ob = current_state[player_id]
        ob['controlled_idx'] = player_id
        return ob

    def get_sorted_next_state(self, next_state):
        # left_team = next_state[:self.agent_nums[0]]
        # right_team = next_state[self.agent_nums[0]:]
        # left_team = sorted(left_team, key=lambda keys: keys['active'])
        # right_team = sorted(right_team, key=lambda keys: keys['active'])

        # new_state = []
        # index = 0
        # for item in left_team:
        #     each = copy.deepcopy(item)
        #     each["controlled_player_index"] = index
        #     new_state.append(each)
        #     index += 1

        # for item in right_team:
        #     each = copy.deepcopy(item)
        #     each["controlled_player_index"] = index
        #     new_state.append(each)
        #     index += 1

        # return new_state
        return next_state


if __name__ == "__main__":

    params = {
        "n_player": 22,
        "max_step": 3000,
        "env_name": "11_vs_11_stochastic",
        "is_obs_continuous": False,
        "is_act_continuous": False,
        "agent_nums": [11, 11],
        "obs_type": ["dict", "dict"],
        "act_box": {
            "discrete_n": 19
        }
    }
    env_params = EnvParams(**params)

    football = Football(env_params, 0)
    print()
