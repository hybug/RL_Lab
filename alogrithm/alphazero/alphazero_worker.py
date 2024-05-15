from alogrithm.alphazero.mcts_env import MCTSTree
from envs.env_gomoku.env import GomokuEnv


class Worker:
    def __init__(self) -> None:
        self.env = GomokuEnv()
        self.buffer = None
        self.mcts_tree = MCTSTree()
        self.model = 