'''
Author: hanyu
Date: 2021-08-09 06:59:11
LastEditTime: 2022-08-29 16:35:04
LastEditors: hanyu
Description: lord game core
FilePath: /RL_Lab/envs/env_instances/lord/lord_game.py
'''
from copy import deepcopy

from envs.env_instances.lord.rule_engine import move_detector as md
from envs.env_instances.lord.rule_engine import move_selector as ms
from envs.env_instances.lord.rule_engine.move_generator import MovesGener
from envs.env_instances.lord.utils import basic_utils, rule_utils
from envs.env_instances.lord.utils.basic_utils import (
    InfoSet, card_idx_list_to_str_list, get_the_next_turn)
from envs.env_instances.lord.utils.rule_utils import BOMBS


class LordGame(object):

    def __init__(self, silence_mode: True) -> None:
        """Init lord game core

        Args:
            silence_mode (True): display mode
        """

        self._reset_init_info()
        self.silence_mode = silence_mode

    def _reset_init_info(self) -> None:
        """reset the initial variable
        """
        self.card_play_action_seq = list()
        self.public_cards = list()
        self.game_over = False

        self.stepping_player_position = None
        self.player_utility_dict = None
        self.scores = {'landlord': 0,
                       'peasant_down': 0,
                       'peasant_up': 0}
        self.num_wins = {'landlord': 0,
                         'peasant_down': 0,
                         'peasant_up': 0}

        self.last_move_dict = {'landlord': [],
                               'peasant_down': [],
                               'peasant_up': []}
        self.played_cards = {'landlord': [],
                             'peasant_down': [],
                             'peasant_up': []}

        self.last_move = list()
        self.last_two_move = list()

        self.info_sets = {'landlord': InfoSet('landlord'),
                          'peasant_down': InfoSet('peasant_down'),
                          'peasant_up': InfoSet('peasant_up')}
        self.bomb_num = 0
        self.last_pid = 'landlord'

    def reset(self):
        """reset game core except self.players
        """
        self._reset_init_info()

    def step(self, action: list) -> None:

        # Update the last non-pass player's id
        if len(action) > 0:
            self.last_pid = self.stepping_player_position

        if action in BOMBS:
            # Update bomb number
            self.bomb_num += 1

        # Update last_move_dict
        self.last_move_dict[self.stepping_player_position] = action.copy()

        # Update card_play_action_seq
        self.card_play_action_seq.append(action)
        # Update stepping player's hand
        self.update_stepping_player_hand_cards(action)

        # Update played_cards
        self.played_cards[self.stepping_player_position] += action

        self.check_game_over()
        if not self.game_over:
            # Turn to next player
            self.get_stepping_player_position()
            # Update game infoset
            self.game_infoset = self.get_infoset()
        self.display(self.game_infoset)

    def check_game_over(self) -> None:
        """Check game is over or not
        """
        if len(self.info_sets['landlord'].player_hand_cards) == 0 or \
            len(self.info_sets['peasant_down'].player_hand_cards) == 0 or \
                len(self.info_sets['peasant_up'].player_hand_cards) == 0:
            self.compute_player_utility()
            self.update_scores()
            self.game_over = True

    def update_stepping_player_hand_cards(self, action: str) -> None:
        """Update the stepping player's hand according to the action string

        Args:
            action (list): action list without suit
        """
        if action:
            for card in action:
                self.info_sets[self.stepping_player_position].player_hand_cards.remove(
                    card)
            self.info_sets[self.stepping_player_position].player_hand_cards.sort()

    def compute_player_utility(self):
        """Compute player's score utility
        """
        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            # The winner is landlord
            self.player_utility_dict = {'landlord': 2,
                                        'peasant_down': -1,
                                        'peasant_up': -1}
        else:
            assert len(self.info_sets['peasant_down'].player_hand_cards) == 0 or\
                len(self.info_sets['peasant_up'].player_hand_cards) == 0
            # The winner is peasant
            self.player_utility_dict = {'landlord': -2,
                                        'peasant_down': 1,
                                        'peasant_up': 1}

    def update_scores(self) -> None:
        """Update the final scores
        """
        assert self.player_utility_dict
        for pos, utility in self.player_utility_dict.items():
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                self.scores[pos] += utility * \
                    self.compute_bomb_multi_num(self.bomb_num)
            else:
                self.scores[pos] += utility * \
                    self.compute_bomb_multi_num(self.bomb_num)

    def get_winner(self) -> str:
        """Get the winner position

        Returns:
            str: the winner position
        """
        return self.winner

    def get_bomb_num(self) -> int:
        """Get the bomb number

        Returns:
            int: the bomb number
        """
        return self.bomb_num

    def compute_bomb_multi_num(self, bomb_num: int) -> int:
        """Compute multi number in scores counting

        Args:
            bomb_num (int): bomb number

        Returns:
            int: multi number
        """
        ret = 1
        if bomb_num == 0:
            return ret
        else:
            return ret * 2 * bomb_num

    def card_play_init(self, card_play_data: dict) -> None:
        """Initialize the card infomation according to the card_play_data

        Args:
            card_play_data (dict): card play infomation
        """
        self.info_sets['landlord'].player_hand_cards = \
            card_play_data['landlord']
        self.info_sets['peasant_down'].player_hand_cards = \
            card_play_data['peasant_down']
        self.info_sets['peasant_up'].player_hand_cards = \
            card_play_data['peasant_up']
        self.public_cards = card_play_data['public_cards']
        self.get_stepping_player_position()
        self.game_infoset = self.get_infoset()

    def get_stepping_player_position(self) -> str:
        """Get the stepping player's position

        Returns:
            str: stepping player's position
        """
        if self.stepping_player_position is None:
            # Landlord stepping first
            self.stepping_player_position = 'landlord'
        else:
            self.stepping_player_position = basic_utils.get_the_next_turn(
                self.stepping_player_position)
        return self.stepping_player_position

    def get_infoset(self) -> InfoSet:
        """get the infoset

        Returns:
            InfoSet: returned infoset
        """
        self.info_sets[
            self.stepping_player_position].last_pid = self.last_pid

        self.info_sets[
            self.stepping_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.stepping_player_position].bomb_num = self.bomb_num

        self.info_sets[
            self.stepping_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.stepping_player_position].last_two_moves = self.get_last_two_moves()

        self.info_sets[
            self.stepping_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.stepping_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in basic_utils.STEP_ORDER}

        self.info_sets[self.stepping_player_position].other_hand_cards = []
        for pos in basic_utils.STEP_ORDER:
            if pos != self.stepping_player_position:
                self.info_sets[
                    self.stepping_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.stepping_player_position].played_cards = \
            self.played_cards
        self.info_sets[self.stepping_player_position].public_cards = \
            self.public_cards
        self.info_sets[self.stepping_player_position].public_cards = \
            self.public_cards
        self.info_sets[self.stepping_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.stepping_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in basic_utils.STEP_ORDER}
        self.info_sets[self.stepping_player_position]._game_over = self.game_over
        return deepcopy(self.info_sets[self.stepping_player_position])

    def get_last_move(self) -> list:
        """Get the last move lsit

        Returns:
            list: the last move
        """
        last_move = list()
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                # Last move is pass
                last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]
        return last_move

    def get_last_two_moves(self) -> list:
        """Get the last two moves,
        order: [stepN-1, stepN]

        Returns:
            list: the last two moves
        """
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_legal_card_play_actions(self):
        """get the legal actions
        """
        mg = MovesGener(
            self.info_sets[self.stepping_player_position].player_hand_cards)

        action_sequence = self.card_play_action_seq

        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == rule_utils.TYPE_0_PASS:
            moves = mg.gen_moves()

        elif rival_move_type == rule_utils.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == rule_utils.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = ms.filter_type_13_4_2(all_moves, rival_move)

        elif rival_move_type == rule_utils.TYPE_14_4_22:
            all_moves = mg.gen_type_14_4_22()
            moves = ms.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [rule_utils.TYPE_0_PASS,
                                   rule_utils.TYPE_4_BOMB, rule_utils.TYPE_5_KING_BOMB]:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def display(self, infoset):
        if not self.silence_mode:
            print('=' * 48)
            print('Public Cards: ' +
                  ' '.join(card_idx_list_to_str_list(infoset.public_cards)))
            print(f'Last valid player: {infoset.last_pid}'
                  f' | Last valid move: {" ".join(card_idx_list_to_str_list(infoset.last_move))}')
            if infoset.card_play_action_seq:
                last_print_move = ' '.join(card_idx_list_to_str_list(
                    infoset.card_play_action_seq[-1])) if infoset.card_play_action_seq[-1] else 'Pass'
                last_print_player = get_the_next_turn(
                    get_the_next_turn(infoset.player_position))
                print(
                    f'Current player: {last_print_player} | Current move: {last_print_move}')

            turn_flag = {'landlord': '', 'peasant_down': '', 'peasant_up': ''}
            turn_flag[infoset.player_position] = '<=='
            print(
                f"Landlord: {' '.join(card_idx_list_to_str_list(infoset.all_handcards['landlord']))}"
                f" [{len(infoset.all_handcards['landlord'])}] {turn_flag['landlord']}")
            print(
                f"PeasantD: {' '.join(card_idx_list_to_str_list(infoset.all_handcards['peasant_down']))}"
                f" [{len(infoset.all_handcards['peasant_down'])}] {turn_flag['peasant_down']}")
            print(
                f"PeasantU: {' '.join(card_idx_list_to_str_list(infoset.all_handcards['peasant_up']))}"
                f" [{len(infoset.all_handcards['peasant_up'])}] {turn_flag['peasant_up']}")
            if self.game_over:
                print(f'Winner: {self.winner}')
                print(f'Scores: {self.scores}')
            print()
