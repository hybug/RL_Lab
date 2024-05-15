'''
Author: hanyu
Date: 2024-05-15 10:29:55
LastEditTime: 2024-05-15 21:40:54
LastEditors: hanyu
Description: gomoku env
FilePath: /RL_Lab/envs/env_gomoku/env.py
'''

from random import choice, randint
from typing import Tuple
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [0, 1]  # player1 and player2
        self.rewards = {0: 0, 1: 0}

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.legal_moves = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.rewards = {0: 0, 1: 0}

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def step_a_move(self, move):
        self.states[move] = self.current_player
        self.legal_moves.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.legal_moves))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            for p in self.players:
                if p == winner:
                    self.rewards[p] = 1
                else:
                    self.rewards[p] = -1
            return True, winner
        elif not len(self.legal_moves):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class GomokuEnv(object):

    def __init__(self, debug_mode: bool = False, **kwargs):
        self.board = Board(**kwargs)
        self.debug_mode = debug_mode

    def show_board(self, board: Board):
        """Draw the board and show game info"""
        width = board.width
        height = board.height
        player1 = board.players[0]
        player2 = board.players[1]

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def reset(self) -> Tuple:
        self.start_player = randint(0, 1)
        self.board.init_board(self.start_player)
        if self.debug_mode:
            self.show_board(self.board)
        return self.board.current_state(), False, self.board.rewards

    def step(self, action: int) -> Tuple:

        self.board.step_a_move(action)
        game_is_over, winner = self.board.game_end()
        if self.debug_mode:
            self.show_board(self.board)
            if game_is_over:
                print(f"Game is over | Winner: {winner} | Reward: {self.board.rewards}")

        return self.board.current_state(), game_is_over, self.board.rewards


def _test_random_selfplay():
    env = Env(debug_mode=True)

    for _ in range(10):
        _, game_is_over, _ = env.reset()
        while not game_is_over:
            action = choice(env.board.legal_moves)
            _, game_is_over, _ = env.step(action)


if __name__ == "__main__":
    _test_random_selfplay()
