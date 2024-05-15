'''
Author: hanyu
Date: 2024-05-15 11:17:15
LastEditTime: 2024-05-15 17:52:47
LastEditors: hanyu
Description: mcts impl
FilePath: /RL_Lab/alogrithm/alphazero/mcts_env.py
'''
from copy import deepcopy
import numpy as np
from typing import List
import torch
from envs.env_gomoku.env import Board


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    def __init__(self, parent: 'TreeNode', prior_p: float, c_puct: float) -> None:
        self._parent = parent
        self._children = dict()
        self._n_visit = 0
        self._Q = 0
        self._U = 0
        self._P = prior_p
        self.c_puct = c_puct

    def expand(self, action_priors: List):
        for action, prob in enumerate(action_priors):
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self):
        """Select max U + Q
        """
        action = max(self._children.items(), key=lambda x: x[1].get_value(self.c_puct))
        return action

    def update(self, leaf_value: float):
        """Update node values from leaf evaluation

        Args:
            leaf_value (float): the value of subtree evaluation from current player's perspective
        """
        # Update visit count
        self._n_visit += 1
        # Average value calculation
        self._Q = 1.0 * (leaf_value - self._Q) / self._n_visit

    def update_recursive(self, leaf_value: float):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """uct calculation

        Args:
            c_puct (_type_): controlling param
        """
        self._U = (c_puct * self._P * np.sqrt(self._parent._n_visit) / (self._n_visit + 1))
        return self._Q + self._U

    def is_leaf(self):
        return self._children == dict()

    def is_parent(self):
        return self._parent is None


class MCTSTree:
    def __init__(self, c_puct: float = 5, rollout_times: int = 2000) -> None:
        self._root = TreeNode(None, 1.0, c_puct)
        self._c_puct = c_puct
        self.rollout_times = rollout_times

    def search_root(self, state: Board, temperature: float = 1e-3) -> tuple:

        for n in range(self.rollout_times):
            state_copy = deepcopy(state)
            self.search(state_copy)

        action_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*action_visits)
        act_probs = softmax(1.0 / temperature * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def search(self, state: Board, policy: 'torch.nn.Module') -> None:
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedy select next action
            action, node = node.select()
            state.step_a_move(action)

        action_probs, leaf_value = policy(state)  # TODO

        game_is_over, winner = state.game_end()
        if not game_is_over:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        # Update the tree
        node.update_recursive(-leaf_value)
