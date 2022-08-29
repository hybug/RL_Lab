'''
Author: hanyu
Date: 2021-08-09 06:57:23
LastEditTime: 2022-08-29 16:49:07
LastEditors: hanyu
Description: basic utils
FilePath: /RL_Lab/envs/env_instances/lord/utils/basic_utils.py
'''
import os
import config
import random

STEP_ORDER = ['landlord', 'peasant_down', 'peasant_up']


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict = None
        # The three landload cards. A list.
        self.public_cards = None
        self.public_cards = None
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the postions
        self.last_move_dict = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict.
        self.all_handcards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
        # The number of bombs played so far
        self.bomb_num = None
        # Game status
        self._game_over = False


def get_the_next_turn(position: str) -> str:
    """get the next turn position

    Args:
        position (str): position string

    Returns:
        str: next stepping poisition
    """

    return STEP_ORDER[(STEP_ORDER.index(position) + 1) % len(STEP_ORDER)]


def cardstr2idx(card_str: str) -> int:
    """convert card string into index

    Args:
        card_str (str): card string format

    Returns:
        int: card index format
    """
    CardStr2Idx = {
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        'T': 10,
        'J': 11,
        'Q': 12,
        'K': 13,
        'A': 14,
        '2': 17,
        'B': 20,
        'R': 30
    }
    return CardStr2Idx[card_str]


def card_str_list_to_idx_list(card_str_list: list) -> list:
    return [cardstr2idx(card_str) for card_str in card_str_list]


def cardidx2str(card_idx: int) -> str:
    """convert card index into string

    Args:
        card_idx (int): card index format

    Returns:
        str: card index format
    """
    CardIdx2Str = {
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: 'T',
        11: 'J',
        12: 'Q',
        13: 'K',
        14: 'A',
        17: '2',
        20: 'B',
        30: 'R'
    }
    return CardIdx2Str[card_idx]


def card_idx_list_to_str_list(card_idx_list: list) -> list:
    """convert card index list into card string list

    Args:
        card_idx_list (list): card index list

    Returns:
        list: card str list
    """
    return [cardidx2str(card_idx) for card_idx in card_idx_list]


def sort_card_str_list(card_str_list: list) -> list:
    """Sort card list with rank str

    Args:
        card_str_list (list): card str rank list

    Returns:
        list: sorted card list
    """
    card_idx_list = card_str_list_to_idx_list(card_str_list)
    card_idx_list.sort()
    return card_idx_list_to_str_list(card_idx_list)


def dispatch_card():
    CardIdx2Str = {
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: 'T',
        11: 'J',
        12: 'Q',
        13: 'K',
        14: 'A',
        17: '2',
        20: 'B',
        30: 'R'
    }
    all_card_idxs = sorted(list(CardIdx2Str.keys())[:-2] * 4 + [20, 30])
    random.shuffle(all_card_idxs)

    ret = dict()
    ret["landlord"] = all_card_idxs[:20]
    ret["peasant_down"] = all_card_idxs[20:37]
    ret["peasant_up"] = all_card_idxs[37:]
    ret["public_cards"] = all_card_idxs[:3]
    return ret


def dispatch_card_lib_random():
    import random
    import gzip
    card_lib_path = config.BASEDIR + '/card_libraray_zipped/'
    all_files_name_list = os.listdir(card_lib_path)

    def get_file_data_with_readline_generator(path):
        with gzip.open(path, 'rt') as f:
            while True:
                data = f.readline()
                if data:
                    yield data
                else:
                    return

    line_num = random.randint(1, 9999)
    line_generator = get_file_data_with_readline_generator(
        card_lib_path + random.choice(all_files_name_list))
    for _ in range(line_num):
        line = next(line_generator).rstrip()

    ret = dict()
    hands_list = line.split(' ')
    for index, h_s in enumerate(hands_list):
        if len(h_s) == 20:
            lord_idx = index
    ret['landlord'] = card_str_list_to_idx_list(list(hands_list[lord_idx]))
    ret['peasant_down'] = card_str_list_to_idx_list(
        list(hands_list[(lord_idx + 1) % 3]))
    ret['peasant_up'] = card_str_list_to_idx_list(
        list(hands_list[(lord_idx + 2) % 3]))
    ret['public_cards'] = ret['landlord'][-3:]

    return ret


def dispatch_card_lib_ordinal(index: int):
    import gzip
    card_lib_path = config.BASEDIR + '/card_libraray_zipped/'

    def get_file_data_with_readline_generator(path):
        with gzip.open(path, 'rt') as f:
            while True:
                data = f.readline()
                if data:
                    yield data
                else:
                    return

    file_num = index // 10000
    line_num = index % 10000
    selected_file_path = card_lib_path + f'/card_lib_{file_num}.gz'
    line_generator = get_file_data_with_readline_generator(selected_file_path)
    for _ in range(line_num):
        line = next(line_generator).rstrip()

    ret = dict()
    hands_list = line.split(' ')
    ret['landlord'] = card_str_list_to_idx_list(list(hands_list[0]))
    ret['peasant_down'] = card_str_list_to_idx_list(list(hands_list[1]))
    ret['peasant_up'] = card_str_list_to_idx_list(list(hands_list[2]))
    ret['public_cards'] = card_str_list_to_idx_list(list(hands_list[3]))

    return ret


if __name__ == "__main__":
    dispatch_card()