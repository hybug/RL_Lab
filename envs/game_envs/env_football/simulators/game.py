'''
Author: hanyu
Date: 2022-03-24 15:35:41
LastEditTime: 2022-03-28 15:07:53
LastEditors: hanyu
Description: 
FilePath: /gfootball_rl/env_football/simulators/game.py
'''
# -*- coding:utf-8  -*-
# 作者：zruizhi
# 创建时间： 2020/7/10 10:24 上午
# 描述：
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EnvInfo():
    n_player: int
    is_obs_continuous: bool
    is_act_continuous: bool
    game_name: str
    agent_nums: int
    obs_type: int


class Game(ABC):
    def __init__(self, n_player, is_obs_continuous, is_act_continuous,
                 game_name, agent_nums, obs_type):
        self.n_player = n_player
        self.current_state = None
        self.all_observes = None
        self.is_obs_continuous = is_obs_continuous
        self.is_act_continuous = is_act_continuous
        self.game_name = game_name
        self.agent_nums = agent_nums
        self.obs_type = obs_type

        self.env_info = EnvInfo(n_player=n_player,
                                is_obs_continuous=is_obs_continuous,
                                is_act_continuous=is_act_continuous,
                                game_name=game_name,
                                agent_nums=agent_nums,
                                obs_type=obs_type)

    def get_config(self, player_id):
        raise NotImplementedError

    def get_render_data(self, current_state):
        return current_state

    def set_current_state(self, current_state):
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        raise NotImplementedError

    def get_next_state(self, all_action):
        raise NotImplementedError

    def get_reward(self, all_action):
        raise NotImplementedError

    @abstractmethod
    def step(self, all_action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def set_action_space(self):
        raise NotImplementedError
