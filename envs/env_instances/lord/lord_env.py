'''
Author: hanyu
Date: 2021-08-09 06:45:03
LastEditTime: 2022-08-29 17:41:52
LastEditors: hanyu
Description: lord game environment
FilePath: /RL_Lab/envs/env_instances/lord/lord_env.py
'''
import gym
# from envs.env_instances.lord.feature_engine.feature_generator import \
#     get_observation
from envs.env_instances.lord.lord_game import LordGame
from envs.env_instances.lord.utils.basic_utils import (InfoSet, dispatch_card)
from envs.env_instances.lord.utils.complete_action import complete_action


class LordEnv(gym.Env):
    def __init__(self, silence_mode=True) -> None:
        """init the lord game environment

        Args:
            agents (list): Agent object list
        """

        self._env = LordGame(silence_mode=silence_mode)

        self.infoset = None

    def reset(self) -> None:
        """reset the game-env
        """
        # Reset the member variables
        self._env.reset()

        # Dispatch the player's hand
        card_play_data = dispatch_card()
        for key in card_play_data:
            card_play_data[key].sort()

        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        # return get_observation(self.infoset)
        return self.infoset

    def step(self, action: int):
        action_str = complete_action(action, self.infoset)

        assert action_str in self.infoset.legal_actions

        # Game core step action string
        self._env.step(action_str)
        # Update the infoset from game core
        self.infoset = self._game_infoset

        # Process return's variables
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            # obs = get_observation(None)
            obs = None

        else:
            # obs = get_observation(self.infoset)
            obs = self.infoset
        return obs, reward, done, {}

    def _get_reward(self) -> int:
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            return 2.0 * (bomb_num + 1)
        else:
            return -1.0 * (bomb_num + 1)

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
