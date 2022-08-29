'''
Author: hanyu
Date: 2021-08-09 09:26:42
LastEditTime: 2021-08-11 08:11:50
LastEditors: hanyu
Description: rule utils
FilePath: /LordHD-RL/env/utils/rule_utils.py
'''
import itertools

# global parameters
MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

# action types
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
TYPE_5_KING_BOMB = 5
TYPE_6_3_1 = 6
TYPE_7_3_2 = 7
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11
TYPE_12_SERIAL_3_2 = 12
TYPE_13_4_2 = 13
TYPE_14_4_22 = 14
TYPE_15_WRONG = 15

# betting round action
PASS = 0
CALL = 1
RAISE = 2

# Bomb list
BOMBS = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [
             13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]


def select(cards, num):
    """return all possible results of selecting num cards from cards list

    Args:
        cards ([type]): cards list
        num ([type]): combination number

    Returns:
        [type]: all possible results
    """
    return [list(i) for i in itertools.combinations(cards, num)]
