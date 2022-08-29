'''
Author: hanyu
Date: 2021-08-12 03:32:45
LastEditTime: 2021-08-12 07:30:50
LastEditors: hanyu
Description: sl utils
FilePath: /LordHD-RL/env/sl_helper/sl_utils.py
'''
# from core import rule, ranks, game as game_core, Card

# all_cards = game_core.Deck().cards
# first_combo = rule.Combo('First', 0, '', 0)
# all_legal_moves = rule.get_legal_moves(first_combo, all_cards)
import random
suits = ['S', 'H', 'D', 'C']
ranks = ['RJ', 'BJ', '2', 'A', 'K', 'Q', 'J',
         'T', '9', '8', '7', '6', '5', '4', '3']
# ranks = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2', 'BJ', 'RJ']
levels = {'3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
          'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13, 'BJ': 14, 'RJ': 15}


class Card(object):
    def __init__(self, str_card: str = '', rank: str = '', suit: str = None) -> None:
        """Card
        BJ, RJ
        :param rank: str:  3, 4, 5, 6, 7, 8, 9, T(Ten), J(Jack), Q(Queen ), K(King ), A(Ace ), 2, BJ(BlackJoker), RJ(RedJoker)
        :param suit: str='': Spade (S), Diamond (D), Club (C), Heart (H)
                             Suits are irrelevant.

        """
        c = str_card.strip(' ').upper()

        if suit is None:
            if len(c) != 2:
                raise ValueError(f"Card [{c}] Invalid rank")
            if c == 'BJ' or c == 'RJ':
                suit = ''
                rank = c
                level = levels[c]
            else:
                suit = c[0]
                rank = c[1]
                level = levels[rank]
        else:
            level = levels[rank]

        if rank not in ranks:
            raise ValueError(f"Card [{c}] Invalid rank")
        if suit and suit not in suits:
            raise ValueError(f"Card [{c}] Invalid suit")

        self.rank = rank
        self.level = level
        self.suit = suit

    def __eq__(self, o: object) -> bool:
        return self.suit == o.suit and self.rank == o.rank

    def __str__(self) -> str:
        """
        命令行交互时显示，或是格式化输出时增加!r进行标识 \n
        >>> c=Card(str_cart="HT")
        >>> print(f"{c!r}")
        Card(suit='H',rank='T',level=8)
        SUITS = {
            'Spade': '♠',
            'Diamond': '♦',
            'Club': '♣',
            'Heart': '♥',
            '': ''
        }
        """
        if self.suit == 'S':
            suit = '♠'
        elif self.suit == 'D':
            suit = '♦'
        elif self.suit == 'C':
            suit = '♣'
        elif self.suit == 'H':
            suit = '♥'
        else:
            suit = ''
        return f"{suit}{self.rank}"

    def __repr__(self) -> str:
        return f"{self.suit}{self.rank}"

    def __hash__(self) -> int:
        return (ord(self.suit) << 8) + self.level if self.suit else self.level


class Deck(object):
    def __init__(self) -> None:
        """Create one pack of playing cards.

        :return deck -> list:

        * The level of rank 2 is special.

        * The suit of jokers are none.

        """
        self.cards = list()

        for suit in suits:
            for rank in ranks[2:]:
                self.cards.append(Card(rank=rank, suit=suit))
        self.cards.append(Card(rank='BJ', suit=''))
        self.cards.append(Card(rank='RJ', suit=''))

        for _ in range(3):  # Shuffle 3 times.
            random.shuffle(self.cards)
        assert len(self.cards) == 54


def get_down_up_hands(pos: int, seqs: list, lord_pos: int, pub_cards: list):
    assert len(pub_cards) == 2 and pub_cards[0] != ''
    lord_hand_num = 17 + (len(pub_cards[0].split(' ')))
    peasant_hand_num = 17

    _, down_index, up_index = pos, (pos + 1) % 3, (pos + 2) % 3
    down_played_cards = []
    up_played_cards = []
    me_played_cards = []
    for index, move in enumerate(seqs):
        if move == 'Pass':
            continue
        if (lord_pos + index) % 3 == down_index:
            down_played_cards.extend(move.split(' '))
        elif (lord_pos + index) % 3 == up_index:
            up_played_cards.extend(move.split(' '))
        else:
            me_played_cards.extend(move.split(' '))

    down_num = lord_hand_num - len(down_played_cards) \
        if down_index == lord_pos else peasant_hand_num - len(down_played_cards)
    up_num = lord_hand_num - len(up_played_cards) \
        if up_index == lord_pos else peasant_hand_num - len(up_played_cards)
    return down_num, up_num, me_played_cards, down_played_cards, up_played_cards


def num_2_plane(hand_num: int) -> str:
    s100 = "111111111111111111111111111111111111111111111111111111111111\n" \
           "000000000000000000000000000000000000000000000000000000000000"
    s010 = "000000000000000000000000000000000000000000000000000000000000\n" \
           "111111111111111111111111111111111111111111111111111111111111"
    if hand_num == 1:
        return s100
    elif hand_num == 2:
        return s010
    else:
        return


def num_3_plane_short(hand_num: int) -> str:
    s1000 = "111111111111111000000000000000000000000000000000000000000000"

    s0100 = "000000000000000111111111111111000000000000000000000000000000"

    s0010 = "000000000000000000000000000000111111111111111000000000000000"

    if hand_num == 1:
        return s1000
    elif hand_num == 2:
        return s0100
    elif hand_num > 2:
        return s0010


def num_8_plane_short(hand_num: int) -> str:
    s10000000 = "111111111111111000000000000000000000000000000000000000000000\n" \
                "000000000000000000000000000000000000000000000000000000000000"

    s01000000 = "000000000000000111111111111111000000000000000000000000000000\n" \
                "000000000000000000000000000000000000000000000000000000000000"

    s00100000 = "000000000000000000000000000000111111111111111000000000000000\n" \
                "000000000000000000000000000000000000000000000000000000000000"

    s00010000 = "000000000000000000000000000000000000000000000111111111111111\n" \
                "000000000000000000000000000000000000000000000000000000000000"

    s00001000 = "000000000000000000000000000000000000000000000000000000000000\n" \
                "111111111111111000000000000000000000000000000000000000000000"

    s00000100 = "000000000000000000000000000000000000000000000000000000000000\n" \
                "000000000000000111111111111111000000000000000000000000000000"

    s00000010 = "000000000000000000000000000000000000000000000000000000000000\n" \
                "000000000000000000000000000000111111111111111000000000000000"

    s00000001 = "000000000000000000000000000000000000000000000000000000000000\n" \
                "000000000000000000000000000000000000000000000111111111111111"

    if hand_num == 1:
        return s10000000
    elif hand_num == 2:
        return s01000000
    elif hand_num == 3:
        return s00100000
    elif hand_num == 4:
        return s00010000
    elif hand_num == 5:
        return s00001000
    elif hand_num == 6:
        return s00000100
    elif hand_num == 7:
        return s00000010
    else:
        return s00000001


def num_3_plane_list(hand_num: int) -> list:
    tensor1 = [float(
        one) for one in '111111111111111111111111111111111111111111111111111111111111']
    tensor0 = [float(
        zero) for zero in '000000000000000000000000000000000000000000000000000000000000']
    t100 = [tensor1, tensor0, tensor0]
    t010 = [tensor0, tensor1, tensor0]
    t001 = [tensor0, tensor0, tensor1]
    t000 = [tensor0, tensor0, tensor0]
    if hand_num == 0:
        return t000
    elif hand_num == 1:
        return t100
    elif hand_num == 2:
        return t010
    elif hand_num > 2:
        return t001


def rank_to_bool_str(rank_list: list) -> str:
    """Convert ranks of cards to 4 * 15 maps with boolean values.
    Example:

    :param rank_list: str: RJ 2 K K T 9 6 6 5 5 4 4 4 4 3 3 3 # 17
    :return: str: 1 0 1 0 1 0 0 1 1 0 0 1 1 1 1
                  0 0 0 0 1 0 0 0 0 0 0 1 1 1 1
                  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
                  0 0 0 0 0 0 0 0 0 0 0 0 0 1 0

    """
    ranks = ['RJ', 'BJ', '2', 'A', 'K', 'Q', 'J',
             'T', '9', '8', '7', '6', '5', '4', '3']
    if not rank_list:
        return '000000000000000000000000000000000000000000000000000000000000'
    bool_str = ''
    for _ in range(4):
        rank_list_copy = rank_list.copy()
        for rank in ranks:
            if rank in rank_list_copy:
                bool_str += '1'
                rank_list.remove(rank)
            else:
                bool_str += '0'

    return bool_str


def get_one_zero_str():
    one_list = []
    zero_list = []
    for _ in range(60):
        one_list.append('1')
        zero_list.append('0')
    one_str = ''.join(one_list)
    zero_str = ''.join(zero_list)
    return one_str, zero_str


def bids_2_lord_pos(bids: list) -> int:
    """
    convert bids to lord position
    """
    assert len(bids) != 0
    for bid_str in reversed(bids):
        assert len(bid_str.split(':')) == 2
        bid_player = int(bid_str.split(':')[0])
        bid_value = int(bid_str.split(':')[1])
        if bid_value != 0:
            return bid_player
    return -1


def switch_joker_from_gbgl(hand_str_gbgl: str) -> str:
    """
    swtich RJ, BJ to GB, GL
    """
    hand_str_rjbj = hand_str_gbgl.replace('GB', 'RJ').replace('GL', 'BJ')
    return hand_str_rjbj
