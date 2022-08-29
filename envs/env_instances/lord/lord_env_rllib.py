'''
Author: hanyu
Date: 2021-08-09 06:45:03
LastEditTime: 2022-08-26 16:42:52
LastEditors: hanyu
Description: lord game environment
FilePath: /RL_Lab/envs/env_instances/lord/lord_env_rllib.py
'''
import gym
from envs.env_instances.lord.feature_engine.feature_generator import \
    get_observation
from envs.env_instances.lord.feature_engine.policy_labels import policy_label
from envs.env_instances.lord.lord_game import LordGame
from envs.env_instances.lord.utils.basic_utils import (
    InfoSet, dispatch_card_lib_random)
from envs.env_instances.lord.utils.complete_action import complete_action


class LordEnv(gym.Env):
    def __init__(self, agent, env_config, silence_mode=True) -> None:
        """init the lord game environment

        Args:
            agents (list): Agent object list
        """
        self.rl_position = env_config['rl_position']
        # self.players = dict()
        # for pos in ['landlord', 'peasant_down', 'peasant_up']:
        #     self.players[pos] = agents[pos]
        self.sl_player = agent
        self.env_config = env_config

        self._env = LordGame(silence_mode=silence_mode)

        self.observation_space = gym.spaces.Dict({'state': gym.spaces.Box(low=0,
                                                                          high=1,
                                                                          shape=([4,
                                                                                  self.env_config.getint(
                                                                                      'card_species'),
                                                                                  self.env_config.getint('num_channels')])),
                                                  'action_mask': gym.spaces.Box(low=0, high=1, shape=([len(policy_label)]))})
        self.action_space = gym.spaces.Discrete(len(policy_label))

        self.infoset = None

    def reset(self) -> None:
        """reset the game-env
        """
        # Reset the member variables
        self._env.reset()

        # Dispatch the player's hand
        card_play_data = dispatch_card_lib_random()
        for key in card_play_data:
            card_play_data[key].sort()

        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        # 等待动作轮询到rl agent
        self.auto_step()

        return get_observation(self.infoset,
                               env_config=self.env_config)

    def step(self, action: int):
        action = complete_action(action, self.infoset)

        assert action in self.infoset.legal_actions

        # Game core step action
        self._env.step(action)
        # Update the infoset from game core
        self.infoset = self._game_infoset

        # Auto step
        self.auto_step()

        # Process return's variables
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = get_observation(self.infoset,
                                  env_config=self.env_config)
        else:
            obs = get_observation(self.infoset,
                                  env_config=self.env_config)
        return obs, reward, done, {}

    # def step(self, action: int):
    #     while True:
    #         action = complete_action(action, self.infoset)

    #         assert action in self.infoset.legal_actions

    #         # Game core step action
    #         self._env.step(action)
    #         # Update the infoset from game core
    #         self.infoset = self._game_infoset

    #         # Auto step
    #         self.auto_step()

    #         # Process return's variables
    #         done = False
    #         reward = 0.0
    #         if self._game_over:
    #             done = True
    #             reward = self._get_reward()
    #             obs = get_observation(self.infoset,
    #                                   env_config=self.env_config)
    #         else:
    #             obs = get_observation(self.infoset,
    #                                   env_config=self.env_config)

    #         if self.sl_player.gen_action(self.infoset) != []:
    #             return obs, reward, done, {}

    # def auto_step(self):
    #     while not (self._game_over or self.infoset.player_position == self.rl_position):
    #         assert self.infoset.player_position != self.rl_position
    #         # sl player compute action
    #         action = self.sl_player.gen_action(self.infoset)
    #         assert action in self.infoset.legal_actions
    #         # Step action
    #         self._env.step(action)
    #         self.infoset = self._game_infoset

    def auto_step(self):
        while not (self._game_over or self.infoset.player_position == self.rl_position):
            assert self.infoset.player_position != self.rl_position
            # sl player compute action
            action = self.sl_player.gen_action(self.infoset)
            assert action in self.infoset.legal_actions
            # Step action
            self._env.step(action)
            self.infoset = self._game_infoset

    def _get_reward(self) -> int:
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            return 2.0 * bomb_num
        else:
            return -2.0 * bomb_num

    @property
    def _game_infoset(self) -> InfoSet:
        """Return game core infoset

        Returns:
            InfoSet: game infoset
        """
        return self._env.game_infoset

    @property
    def _game_over(self) -> bool:
        """Return the game_over flag

        Returns:
            bool: game over or not
        """
        return self._env.game_over

    @property
    def _game_winner(self):
        return self._env.get_winner()

    @property
    def _game_bomb_num(self):
        return self._env.get_bomb_num()
