'''
Author: hanyu
Date: 2021-08-11 13:27:28
LastEditTime: 2022-08-29 16:20:37
LastEditors: hanyu
Description: policy label
FilePath: /RL_Lab/envs/env_instances/lord/feature_engine/policy_labels.py
'''
import re
from envs.env_instances.lord.utils.basic_utils import card_idx_list_to_str_list

policy_label = {
    'R': 0, 'B': 1, '2': 2, 'A': 3, 'K': 4, 'Q': 5, 'J': 6, 'T': 7,
    '9': 8, '8': 9, '7': 10, '6': 11, '5': 12, '4': 13, '3': 14,

    '2 2': 15, 'A A': 16, 'K K': 17, 'Q Q': 18, 'J J': 19, 'T T': 20,
    '9 9': 21, '8 8': 22, '7 7': 23, '6 6': 24, '5 5': 25, '4 4': 26, '3 3': 27,

    '2 2 2': 28, 'A A A': 29, 'K K K': 30, 'Q Q Q': 31, 'J J J': 32, 'T T T': 33,
    '9 9 9': 34, '8 8 8': 35, '7 7 7': 36, '6 6 6': 37, '5 5 5': 38, '4 4 4': 39, '3 3 3': 40,

    '2 2 2 X': 41, 'A A A X': 42, 'K K K X': 43, 'Q Q Q X': 44, 'J J J X': 45, 'T T T X': 46,
    '9 9 9 X': 47, '8 8 8 X': 48, '7 7 7 X': 49, '6 6 6 X': 50, '5 5 5 X': 51, '4 4 4 X': 52,
    '3 3 3 X': 53,

    '2 2 2 XX': 54, 'A A A XX': 55, 'K K K XX': 56, 'Q Q Q XX': 57, 'J J J XX': 58, 'T T T XX': 59,
    '9 9 9 XX': 60, '8 8 8 XX': 61, '7 7 7 XX': 62, '6 6 6 XX': 63, '5 5 5 XX': 64, '4 4 4 XX': 65,
    '3 3 3 XX': 66,

    'A K Q J T 9 8 7 6 5 4 3': 67, 'A K Q J T 9 8 7 6 5 4': 68, 'A K Q J T 9 8 7 6 5': 69,
    'A K Q J T 9 8 7 6': 70, 'A K Q J T 9 8 7': 71, 'A K Q J T 9 8': 72, 'A K Q J T 9': 73,
    'A K Q J T': 74,
    'K Q J T 9 8 7 6 5 4 3': 75, 'K Q J T 9 8 7 6 5 4': 76, 'K Q J T 9 8 7 6 5': 77, 'K Q J T 9 8 7 6': 78,
    'K Q J T 9 8 7': 79, 'K Q J T 9 8': 80, 'K Q J T 9': 81,
    'Q J T 9 8 7 6 5 4 3': 82, 'Q J T 9 8 7 6 5 4': 83, 'Q J T 9 8 7 6 5': 84, 'Q J T 9 8 7 6': 85,
    'Q J T 9 8 7': 86, 'Q J T 9 8': 87,
    'J T 9 8 7 6 5 4 3': 88, 'J T 9 8 7 6 5 4': 89, 'J T 9 8 7 6 5': 90, 'J T 9 8 7 6': 91,
    'J T 9 8 7': 92,
    'T 9 8 7 6 5 4 3': 93, 'T 9 8 7 6 5 4': 94, 'T 9 8 7 6 5': 95, 'T 9 8 7 6': 96,
    '9 8 7 6 5 4 3': 97, '9 8 7 6 5 4': 98, '9 8 7 6 5': 99,
    '8 7 6 5 4 3': 100, '8 7 6 5 4': 101,
    '7 6 5 4 3': 102,

    'A A K K Q Q J J T T 9 9 8 8 7 7 6 6 5 5': 103, 'A A K K Q Q J J T T 9 9 8 8 7 7 6 6': 104,
    'A A K K Q Q J J T T 9 9 8 8 7 7': 105, 'A A K K Q Q J J T T 9 9 8 8': 106, 'A A K K Q Q J J T T 9 9': 107,
    'A A K K Q Q J J T T': 108, 'A A K K Q Q J J': 109, 'A A K K Q Q': 110,
    'K K Q Q J J T T 9 9 8 8 7 7 6 6 5 5 4 4': 111, 'K K Q Q J J T T 9 9 8 8 7 7 6 6 5 5': 112,
    'K K Q Q J J T T 9 9 8 8 7 7 6 6': 113, 'K K Q Q J J T T 9 9 8 8 7 7': 114, 'K K Q Q J J T T 9 9 8 8': 115,
    'K K Q Q J J T T 9 9': 116, 'K K Q Q J J T T': 117, 'K K Q Q J J': 118,
    'Q Q J J T T 9 9 8 8 7 7 6 6 5 5 4 4 3 3': 119, 'Q Q J J T T 9 9 8 8 7 7 6 6 5 5 4 4': 120,
    'Q Q J J T T 9 9 8 8 7 7 6 6 5 5': 121, 'Q Q J J T T 9 9 8 8 7 7 6 6': 122, 'Q Q J J T T 9 9 8 8 7 7': 123,
    'Q Q J J T T 9 9 8 8': 124, 'Q Q J J T T 9 9': 125, 'Q Q J J T T': 126,
    'J J T T 9 9 8 8 7 7 6 6 5 5 4 4 3 3': 127, 'J J T T 9 9 8 8 7 7 6 6 5 5 4 4': 128,
    'J J T T 9 9 8 8 7 7 6 6 5 5': 129, 'J J T T 9 9 8 8 7 7 6 6': 130, 'J J T T 9 9 8 8 7 7': 131,
    'J J T T 9 9 8 8': 132, 'J J T T 9 9': 133,
    'T T 9 9 8 8 7 7 6 6 5 5 4 4 3 3': 134, 'T T 9 9 8 8 7 7 6 6 5 5 4 4': 135, 'T T 9 9 8 8 7 7 6 6 5 5': 136,
    'T T 9 9 8 8 7 7 6 6': 137, 'T T 9 9 8 8 7 7': 138, 'T T 9 9 8 8': 139,
    '9 9 8 8 7 7 6 6 5 5 4 4 3 3': 140, '9 9 8 8 7 7 6 6 5 5 4 4': 141, '9 9 8 8 7 7 6 6 5 5': 142,
    '9 9 8 8 7 7 6 6': 143, '9 9 8 8 7 7': 144,
    '8 8 7 7 6 6 5 5 4 4 3 3': 145, '8 8 7 7 6 6 5 5 4 4': 146, '8 8 7 7 6 6 5 5': 147, '8 8 7 7 6 6': 148,
    '7 7 6 6 5 5 4 4 3 3': 149, '7 7 6 6 5 5 4 4': 150, '7 7 6 6 5 5': 151,
    '6 6 5 5 4 4 3 3': 152, '6 6 5 5 4 4': 153,
    '5 5 4 4 3 3': 154,

    'A A A K K K Q Q Q J J J T T T 9 9 9': 155, 'A A A K K K Q Q Q J J J T T T': 156,
    'A A A K K K Q Q Q J J J': 157, 'A A A K K K Q Q Q': 158, 'A A A K K K': 159,
    'K K K Q Q Q J J J T T T 9 9 9 8 8 8': 160, 'K K K Q Q Q J J J T T T 9 9 9': 161,
    'K K K Q Q Q J J J T T T': 162, 'K K K Q Q Q J J J': 163, 'K K K Q Q Q': 164,
    'Q Q Q J J J T T T 9 9 9 8 8 8 7 7 7': 165, 'Q Q Q J J J T T T 9 9 9 8 8 8': 166,
    'Q Q Q J J J T T T 9 9 9': 167, 'Q Q Q J J J T T T': 168, 'Q Q Q J J J': 169,
    'J J J T T T 9 9 9 8 8 8 7 7 7 6 6 6': 170, 'J J J T T T 9 9 9 8 8 8 7 7 7': 171,
    'J J J T T T 9 9 9 8 8 8': 172, 'J J J T T T 9 9 9': 173, 'J J J T T T': 174,
    'T T T 9 9 9 8 8 8 7 7 7 6 6 6 5 5 5': 175, 'T T T 9 9 9 8 8 8 7 7 7 6 6 6': 176,
    'T T T 9 9 9 8 8 8 7 7 7': 177, 'T T T 9 9 9 8 8 8': 178, 'T T T 9 9 9': 179,
    '9 9 9 8 8 8 7 7 7 6 6 6 5 5 5 4 4 4': 180, '9 9 9 8 8 8 7 7 7 6 6 6 5 5 5': 181,
    '9 9 9 8 8 8 7 7 7 6 6 6': 182, '9 9 9 8 8 8 7 7 7': 183, '9 9 9 8 8 8': 184,
    '8 8 8 7 7 7 6 6 6 5 5 5 4 4 4 3 3 3': 185, '8 8 8 7 7 7 6 6 6 5 5 5 4 4 4': 186,
    '8 8 8 7 7 7 6 6 6 5 5 5': 187, '8 8 8 7 7 7 6 6 6': 188, '8 8 8 7 7 7': 189,
    '7 7 7 6 6 6 5 5 5 4 4 4 3 3 3': 190, '7 7 7 6 6 6 5 5 5 4 4 4': 191,
    '7 7 7 6 6 6 5 5 5': 192, '7 7 7 6 6 6': 193,
    '6 6 6 5 5 5 4 4 4 3 3 3': 194, '6 6 6 5 5 5 4 4 4': 195, '6 6 6 5 5 5': 196,
    '5 5 5 4 4 4 3 3 3': 197, '5 5 5 4 4 4': 198,
    '4 4 4 3 3 3': 199,

    'A A A K K K Q Q Q J J J T T T X': 200, 'A A A K K K Q Q Q J J J X': 201,
    'A A A K K K Q Q Q X': 202, 'A A A K K K X': 203, 'K K K Q Q Q J J J T T T 9 9 9 X': 204,
    'K K K Q Q Q J J J T T T X': 205, 'K K K Q Q Q J J J X': 206, 'K K K Q Q Q X': 207,
    'Q Q Q J J J T T T 9 9 9 8 8 8 X': 208, 'Q Q Q J J J T T T 9 9 9 X': 209,
    'Q Q Q J J J T T T X': 210, 'Q Q Q J J J X': 211,
    'J J J T T T 9 9 9 8 8 8 7 7 7 X': 212, 'J J J T T T 9 9 9 8 8 8 X': 213,
    'J J J T T T 9 9 9 X': 214, 'J J J T T T X': 215,
    'T T T 9 9 9 8 8 8 7 7 7 6 6 6 X': 216, 'T T T 9 9 9 8 8 8 7 7 7 X': 217,
    'T T T 9 9 9 8 8 8 X': 218, 'T T T 9 9 9 X': 219,
    '9 9 9 8 8 8 7 7 7 6 6 6 5 5 5 X': 220, '9 9 9 8 8 8 7 7 7 6 6 6 X': 221,
    '9 9 9 8 8 8 7 7 7 X': 222, '9 9 9 8 8 8 X': 223,
    '8 8 8 7 7 7 6 6 6 5 5 5 4 4 4 X': 224, '8 8 8 7 7 7 6 6 6 5 5 5 X': 225,
    '8 8 8 7 7 7 6 6 6 X': 226, '8 8 8 7 7 7 X': 227,
    '7 7 7 6 6 6 5 5 5 4 4 4 3 3 3 X': 228, '7 7 7 6 6 6 5 5 5 4 4 4 X': 229,
    '7 7 7 6 6 6 5 5 5 X': 230, '7 7 7 6 6 6 X': 231,
    '6 6 6 5 5 5 4 4 4 3 3 3 X': 232, '6 6 6 5 5 5 4 4 4 X': 233, '6 6 6 5 5 5 X': 234,
    '5 5 5 4 4 4 3 3 3 X': 235, '5 5 5 4 4 4 X': 236,
    '4 4 4 3 3 3 X': 237,

    'A A A K K K Q Q Q J J J XX': 238, 'A A A K K K Q Q Q XX': 239, 'A A A K K K XX': 240,
    'K K K Q Q Q J J J T T T XX': 241, 'K K K Q Q Q J J J XX': 242, 'K K K Q Q Q XX': 243,
    'Q Q Q J J J T T T 9 9 9 XX': 244, 'Q Q Q J J J T T T XX': 245, 'Q Q Q J J J XX': 246,
    'J J J T T T 9 9 9 8 8 8 XX': 247, 'J J J T T T 9 9 9 XX': 248, 'J J J T T T XX': 249,
    'T T T 9 9 9 8 8 8 7 7 7 XX': 250, 'T T T 9 9 9 8 8 8 XX': 251, 'T T T 9 9 9 XX': 252,
    '9 9 9 8 8 8 7 7 7 6 6 6 XX': 253, '9 9 9 8 8 8 7 7 7 XX': 254, '9 9 9 8 8 8 XX': 255,
    '8 8 8 7 7 7 6 6 6 5 5 5 XX': 256, '8 8 8 7 7 7 6 6 6 XX': 257, '8 8 8 7 7 7 XX': 258,
    '7 7 7 6 6 6 5 5 5 4 4 4 XX': 259, '7 7 7 6 6 6 5 5 5 XX': 260, '7 7 7 6 6 6 XX': 261,
    '6 6 6 5 5 5 4 4 4 3 3 3 XX': 262, '6 6 6 5 5 5 4 4 4 XX': 263, '6 6 6 5 5 5 XX': 264,
    '5 5 5 4 4 4 3 3 3 XX': 265, '5 5 5 4 4 4 XX': 266,
    '4 4 4 3 3 3 XX': 267,

    '2 2 2 2 X': 268, 'A A A A X': 269, 'K K K K X': 270, 'Q Q Q Q X': 271, 'J J J J X': 272,
    'T T T T X': 273, '9 9 9 9 X': 274, '8 8 8 8 X': 275, '7 7 7 7 X': 276, '6 6 6 6 X': 277,
    '5 5 5 5 X': 278, '4 4 4 4 X': 279, '3 3 3 3 X': 280,

    '2 2 2 2 XX': 281, 'A A A A XX': 282, 'K K K K XX': 283, 'Q Q Q Q XX': 284, 'J J J J XX': 285,
    'T T T T XX': 286, '9 9 9 9 XX': 287, '8 8 8 8 XX': 288, '7 7 7 7 XX': 289, '6 6 6 6 XX': 290,
    '5 5 5 5 XX': 291, '4 4 4 4 XX': 292, '3 3 3 3 XX': 293,

    '2 2 2 2': 294, 'A A A A': 295, 'K K K K': 296, 'Q Q Q Q': 297, 'J J J J': 298,
    'T T T T': 299, '9 9 9 9': 300, '8 8 8 8': 301, '7 7 7 7': 302, '6 6 6 6': 303,
    '5 5 5 5': 304, '4 4 4 4': 305, '3 3 3 3': 306,

    'R B': 307,

    'Pass': 308,
}

policy_label_reverse = {0: 'R', 1: 'B', 2: '2', 3: 'A', 4: 'K', 5: 'Q', 6: 'J',
                        7: 'T', 8: '9', 9: '8', 10: '7', 11: '6', 12: '5', 13: '4', 14: '3', 15: '2 2',
                        16: 'A A', 17: 'K K', 18: 'Q Q', 19: 'J J', 20: 'T T', 21: '9 9', 22: '8 8',
                        23: '7 7', 24: '6 6', 25: '5 5', 26: '4 4', 27: '3 3', 28: '2 2 2',
                        29: 'A A A', 30: 'K K K', 31: 'Q Q Q', 32: 'J J J', 33: 'T T T', 34: '9 9 9',
                        35: '8 8 8', 36: '7 7 7', 37: '6 6 6', 38: '5 5 5', 39: '4 4 4', 40: '3 3 3',
                        41: '2 2 2 X', 42: 'A A A X', 43: 'K K K X', 44: 'Q Q Q X', 45: 'J J J X',
                        46: 'T T T X', 47: '9 9 9 X', 48: '8 8 8 X', 49: '7 7 7 X', 50: '6 6 6 X',
                        51: '5 5 5 X', 52: '4 4 4 X', 53: '3 3 3 X', 54: '2 2 2 XX', 55: 'A A A XX',
                        56: 'K K K XX', 57: 'Q Q Q XX', 58: 'J J J XX', 59: 'T T T XX', 60: '9 9 9 XX',
                        61: '8 8 8 XX', 62: '7 7 7 XX', 63: '6 6 6 XX', 64: '5 5 5 XX', 65: '4 4 4 XX',
                        66: '3 3 3 XX', 67: 'A K Q J T 9 8 7 6 5 4 3', 68: 'A K Q J T 9 8 7 6 5 4',
                        69: 'A K Q J T 9 8 7 6 5', 70: 'A K Q J T 9 8 7 6', 71: 'A K Q J T 9 8 7',
                        72: 'A K Q J T 9 8', 73: 'A K Q J T 9', 74: 'A K Q J T', 75: 'K Q J T 9 8 7 6 5 4 3',
                        76: 'K Q J T 9 8 7 6 5 4', 77: 'K Q J T 9 8 7 6 5', 78: 'K Q J T 9 8 7 6',
                        79: 'K Q J T 9 8 7', 80: 'K Q J T 9 8', 81: 'K Q J T 9', 82: 'Q J T 9 8 7 6 5 4 3',
                        83: 'Q J T 9 8 7 6 5 4', 84: 'Q J T 9 8 7 6 5', 85: 'Q J T 9 8 7 6', 86: 'Q J T 9 8 7',
                        87: 'Q J T 9 8', 88: 'J T 9 8 7 6 5 4 3', 89: 'J T 9 8 7 6 5 4', 90: 'J T 9 8 7 6 5',
                        91: 'J T 9 8 7 6', 92: 'J T 9 8 7', 93: 'T 9 8 7 6 5 4 3', 94: 'T 9 8 7 6 5 4',
                        95: 'T 9 8 7 6 5', 96: 'T 9 8 7 6', 97: '9 8 7 6 5 4 3', 98: '9 8 7 6 5 4',
                        99: '9 8 7 6 5', 100: '8 7 6 5 4 3', 101: '8 7 6 5 4', 102: '7 6 5 4 3',
                        103: 'A A K K Q Q J J T T 9 9 8 8 7 7 6 6 5 5', 104: 'A A K K Q Q J J T T 9 9 8 8 7 7 6 6',
                        105: 'A A K K Q Q J J T T 9 9 8 8 7 7', 106: 'A A K K Q Q J J T T 9 9 8 8',
                        107: 'A A K K Q Q J J T T 9 9', 108: 'A A K K Q Q J J T T', 109: 'A A K K Q Q J J',
                        110: 'A A K K Q Q', 111: 'K K Q Q J J T T 9 9 8 8 7 7 6 6 5 5 4 4', 112: 'K K Q Q J J T T 9 9 8 8 7 7 6 6 5 5',
                        113: 'K K Q Q J J T T 9 9 8 8 7 7 6 6', 114: 'K K Q Q J J T T 9 9 8 8 7 7',
                        115: 'K K Q Q J J T T 9 9 8 8', 116: 'K K Q Q J J T T 9 9', 117: 'K K Q Q J J T T',
                        118: 'K K Q Q J J', 119: 'Q Q J J T T 9 9 8 8 7 7 6 6 5 5 4 4 3 3', 120: 'Q Q J J T T 9 9 8 8 7 7 6 6 5 5 4 4',
                        121: 'Q Q J J T T 9 9 8 8 7 7 6 6 5 5', 122: 'Q Q J J T T 9 9 8 8 7 7 6 6', 123: 'Q Q J J T T 9 9 8 8 7 7',
                        124: 'Q Q J J T T 9 9 8 8', 125: 'Q Q J J T T 9 9', 126: 'Q Q J J T T',
                        127: 'J J T T 9 9 8 8 7 7 6 6 5 5 4 4 3 3', 128: 'J J T T 9 9 8 8 7 7 6 6 5 5 4 4',
                        129: 'J J T T 9 9 8 8 7 7 6 6 5 5', 130: 'J J T T 9 9 8 8 7 7 6 6', 131: 'J J T T 9 9 8 8 7 7',
                        132: 'J J T T 9 9 8 8', 133: 'J J T T 9 9', 134: 'T T 9 9 8 8 7 7 6 6 5 5 4 4 3 3',
                        135: 'T T 9 9 8 8 7 7 6 6 5 5 4 4',
                        136: 'T T 9 9 8 8 7 7 6 6 5 5', 137: 'T T 9 9 8 8 7 7 6 6', 138: 'T T 9 9 8 8 7 7',
                        139: 'T T 9 9 8 8', 140: '9 9 8 8 7 7 6 6 5 5 4 4 3 3', 141: '9 9 8 8 7 7 6 6 5 5 4 4',
                        142: '9 9 8 8 7 7 6 6 5 5', 143: '9 9 8 8 7 7 6 6', 144: '9 9 8 8 7 7',
                        145: '8 8 7 7 6 6 5 5 4 4 3 3', 146: '8 8 7 7 6 6 5 5 4 4', 147: '8 8 7 7 6 6 5 5',
                        148: '8 8 7 7 6 6', 149: '7 7 6 6 5 5 4 4 3 3', 150: '7 7 6 6 5 5 4 4', 151: '7 7 6 6 5 5',
                        152: '6 6 5 5 4 4 3 3', 153: '6 6 5 5 4 4', 154: '5 5 4 4 3 3',
                        155: 'A A A K K K Q Q Q J J J T T T 9 9 9', 156: 'A A A K K K Q Q Q J J J T T T',
                        157: 'A A A K K K Q Q Q J J J', 158: 'A A A K K K Q Q Q', 159: 'A A A K K K',
                        160: 'K K K Q Q Q J J J T T T 9 9 9 8 8 8', 161: 'K K K Q Q Q J J J T T T 9 9 9',
                        162: 'K K K Q Q Q J J J T T T', 163: 'K K K Q Q Q J J J', 164: 'K K K Q Q Q',
                        165: 'Q Q Q J J J T T T 9 9 9 8 8 8 7 7 7', 166: 'Q Q Q J J J T T T 9 9 9 8 8 8',
                        167: 'Q Q Q J J J T T T 9 9 9',
                        168: 'Q Q Q J J J T T T', 169: 'Q Q Q J J J', 170: 'J J J T T T 9 9 9 8 8 8 7 7 7 6 6 6',
                        171: 'J J J T T T 9 9 9 8 8 8 7 7 7', 172: 'J J J T T T 9 9 9 8 8 8', 173: 'J J J T T T 9 9 9',
                        174: 'J J J T T T', 175: 'T T T 9 9 9 8 8 8 7 7 7 6 6 6 5 5 5', 176: 'T T T 9 9 9 8 8 8 7 7 7 6 6 6',
                        177: 'T T T 9 9 9 8 8 8 7 7 7', 178: 'T T T 9 9 9 8 8 8', 179: 'T T T 9 9 9',
                        180: '9 9 9 8 8 8 7 7 7 6 6 6 5 5 5 4 4 4', 181: '9 9 9 8 8 8 7 7 7 6 6 6 5 5 5',
                        182: '9 9 9 8 8 8 7 7 7 6 6 6', 183: '9 9 9 8 8 8 7 7 7', 184: '9 9 9 8 8 8',
                        185: '8 8 8 7 7 7 6 6 6 5 5 5 4 4 4 3 3 3', 186: '8 8 8 7 7 7 6 6 6 5 5 5 4 4 4',
                        187: '8 8 8 7 7 7 6 6 6 5 5 5', 188: '8 8 8 7 7 7 6 6 6', 189: '8 8 8 7 7 7',
                        190: '7 7 7 6 6 6 5 5 5 4 4 4 3 3 3', 191: '7 7 7 6 6 6 5 5 5 4 4 4', 192: '7 7 7 6 6 6 5 5 5',
                        193: '7 7 7 6 6 6', 194: '6 6 6 5 5 5 4 4 4 3 3 3', 195: '6 6 6 5 5 5 4 4 4', 196: '6 6 6 5 5 5',
                        197: '5 5 5 4 4 4 3 3 3', 198: '5 5 5 4 4 4', 199: '4 4 4 3 3 3', 200: 'A A A K K K Q Q Q J J J T T T X',
                        201: 'A A A K K K Q Q Q J J J X', 202: 'A A A K K K Q Q Q X', 203: 'A A A K K K X',
                        204: 'K K K Q Q Q J J J T T T 9 9 9 X', 205: 'K K K Q Q Q J J J T T T X', 206: 'K K K Q Q Q J J J X',
                        207: 'K K K Q Q Q X', 208: 'Q Q Q J J J T T T 9 9 9 8 8 8 X', 209: 'Q Q Q J J J T T T 9 9 9 X',
                        210: 'Q Q Q J J J T T T X', 211: 'Q Q Q J J J X', 212: 'J J J T T T 9 9 9 8 8 8 7 7 7 X',
                        213: 'J J J T T T 9 9 9 8 8 8 X', 214: 'J J J T T T 9 9 9 X', 215: 'J J J T T T X',
                        216: 'T T T 9 9 9 8 8 8 7 7 7 6 6 6 X', 217: 'T T T 9 9 9 8 8 8 7 7 7 X', 218: 'T T T 9 9 9 8 8 8 X',
                        219: 'T T T 9 9 9 X', 220: '9 9 9 8 8 8 7 7 7 6 6 6 5 5 5 X', 221: '9 9 9 8 8 8 7 7 7 6 6 6 X',
                        222: '9 9 9 8 8 8 7 7 7 X', 223: '9 9 9 8 8 8 X', 224: '8 8 8 7 7 7 6 6 6 5 5 5 4 4 4 X',
                        225: '8 8 8 7 7 7 6 6 6 5 5 5 X', 226: '8 8 8 7 7 7 6 6 6 X', 227: '8 8 8 7 7 7 X',
                        228: '7 7 7 6 6 6 5 5 5 4 4 4 3 3 3 X', 229: '7 7 7 6 6 6 5 5 5 4 4 4 X',
                        230: '7 7 7 6 6 6 5 5 5 X', 231: '7 7 7 6 6 6 X', 232: '6 6 6 5 5 5 4 4 4 3 3 3 X',
                        233: '6 6 6 5 5 5 4 4 4 X', 234: '6 6 6 5 5 5 X', 235: '5 5 5 4 4 4 3 3 3 X',
                        236: '5 5 5 4 4 4 X', 237: '4 4 4 3 3 3 X', 238: 'A A A K K K Q Q Q J J J XX',
                        239: 'A A A K K K Q Q Q XX', 240: 'A A A K K K XX', 241: 'K K K Q Q Q J J J T T T XX',
                        242: 'K K K Q Q Q J J J XX', 243: 'K K K Q Q Q XX', 244: 'Q Q Q J J J T T T 9 9 9 XX',
                        245: 'Q Q Q J J J T T T XX', 246: 'Q Q Q J J J XX', 247: 'J J J T T T 9 9 9 8 8 8 XX',
                        248: 'J J J T T T 9 9 9 XX', 249: 'J J J T T T XX', 250: 'T T T 9 9 9 8 8 8 7 7 7 XX',
                        251: 'T T T 9 9 9 8 8 8 XX', 252: 'T T T 9 9 9 XX', 253: '9 9 9 8 8 8 7 7 7 6 6 6 XX',
                        254: '9 9 9 8 8 8 7 7 7 XX', 255: '9 9 9 8 8 8 XX', 256: '8 8 8 7 7 7 6 6 6 5 5 5 XX',
                        257: '8 8 8 7 7 7 6 6 6 XX', 258: '8 8 8 7 7 7 XX', 259: '7 7 7 6 6 6 5 5 5 4 4 4 XX',
                        260: '7 7 7 6 6 6 5 5 5 XX', 261: '7 7 7 6 6 6 XX', 262: '6 6 6 5 5 5 4 4 4 3 3 3 XX',
                        263: '6 6 6 5 5 5 4 4 4 XX', 264: '6 6 6 5 5 5 XX', 265: '5 5 5 4 4 4 3 3 3 XX',
                        266: '5 5 5 4 4 4 XX', 267: '4 4 4 3 3 3 XX', 268: '2 2 2 2 X', 269: 'A A A A X',
                        270: 'K K K K X', 271: 'Q Q Q Q X', 272: 'J J J J X', 273: 'T T T T X', 274: '9 9 9 9 X',
                        275: '8 8 8 8 X', 276: '7 7 7 7 X', 277: '6 6 6 6 X', 278: '5 5 5 5 X', 279: '4 4 4 4 X',
                        280: '3 3 3 3 X', 281: '2 2 2 2 XX', 282: 'A A A A XX', 283: 'K K K K XX', 284: 'Q Q Q Q XX',
                        285: 'J J J J XX', 286: 'T T T T XX', 287: '9 9 9 9 XX', 288: '8 8 8 8 XX', 289: '7 7 7 7 XX',
                        290: '6 6 6 6 XX', 291: '5 5 5 5 XX', 292: '4 4 4 4 XX', 293: '3 3 3 3 XX', 294: '2 2 2 2',
                        295: 'A A A A', 296: 'K K K K', 297: 'Q Q Q Q', 298: 'J J J J', 299: 'T T T T',
                        300: '9 9 9 9', 301: '8 8 8 8', 302: '7 7 7 7', 303: '6 6 6 6', 304: '5 5 5 5',
                        305: '4 4 4 4', 306: '3 3 3 3', 307: 'R B', 308: 'Pass'}


def match_label(action_str: str) -> int:
    label = ''
    if re.search(r'2 2 2', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '41'
    elif re.search(r'A A A', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '42'
    elif re.search(r'K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '43'
    elif re.search(r'Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '44'
    elif re.search(r'J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '45'
    elif re.search(r'T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '46'
    elif re.search(r'9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '47'
    elif re.search(r'8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '48'
    elif re.search(r'7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '49'
    elif re.search(r'6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '50'
    elif re.search(r'5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '51'
    elif re.search(r'4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '52'
    elif re.search(r'3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 4:
        label = '53'

    elif re.search(r'2 2 2', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '54'
    elif re.search(r'A A A', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '55'
    elif re.search(r'K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '56'
    elif re.search(r'Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '57'
    elif re.search(r'J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '58'
    elif re.search(r'T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '59'
    elif re.search(r'9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '60'
    elif re.search(r'8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '61'
    elif re.search(r'7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '62'
    elif re.search(r'6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '63'
    elif re.search(r'5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '64'
    elif re.search(r'4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '65'
    elif re.search(r'3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 5:
        label = '66'

    elif re.search(r'2 2 2 2', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '268'
    elif re.search(r'A A A A', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '269'
    elif re.search(r'K K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '270'
    elif re.search(r'Q Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '271'
    elif re.search(r'J J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '272'
    elif re.search(r'T T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '273'
    elif re.search(r'9 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '274'
    elif re.search(r'8 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '275'
    elif re.search(r'7 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '276'
    elif re.search(r'6 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '277'
    elif re.search(r'5 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '278'
    elif re.search(r'4 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '279'
    elif re.search(r'3 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '280'

    elif re.search(r'2 2 2 2', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '281'
    elif re.search(r'A A A A', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '282'
    elif re.search(r'K K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '283'
    elif re.search(r'Q Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '284'
    elif re.search(r'J J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '285'
    elif re.search(r'T T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '286'
    elif re.search(r'9 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '287'
    elif re.search(r'8 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '288'
    elif re.search(r'7 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '289'
    elif re.search(r'6 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '290'
    elif re.search(r'5 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '291'
    elif re.search(r'4 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '292'
    elif re.search(r'3 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '293'

    elif re.search(r'2 2 2 2', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '268'
    elif re.search(r'A A A A', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '269'
    elif re.search(r'K K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '270'
    elif re.search(r'Q Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '271'
    elif re.search(r'J J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '272'
    elif re.search(r'T T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '273'
    elif re.search(r'9 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '274'
    elif re.search(r'8 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '275'
    elif re.search(r'7 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '276'
    elif re.search(r'6 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '277'
    elif re.search(r'5 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '278'
    elif re.search(r'4 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '279'
    elif re.search(r'3 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 6:
        label = '280'

    elif re.search(r'A A A K K K Q Q Q J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '200'
    elif re.search(r'A A A K K K Q Q Q J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '201'
    elif re.search(r'A A A K K K Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '202'
    elif re.search(r'A A A K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '203'
    elif re.search(r'K K K Q Q Q J J J T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '204'
    elif re.search(r'K K K Q Q Q J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '205'
    elif re.search(r'K K K Q Q Q J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '206'
    elif re.search(r'K K K Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '207'
    elif re.search(r'Q Q Q J J J T T T 9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '208'
    elif re.search(r'Q Q Q J J J T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '209'
    elif re.search(r'Q Q Q J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '210'
    elif re.search(r'Q Q Q J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '211'
    elif re.search(r'J J J T T T 9 9 9 8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '212'
    elif re.search(r'J J J T T T 9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '213'
    elif re.search(r'J J J T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '214'
    elif re.search(r'J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '215'
    elif re.search(r'T T T 9 9 9 8 8 8 7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '216'
    elif re.search(r'T T T 9 9 9 8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '217'
    elif re.search(r'T T T 9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '218'
    elif re.search(r'T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '219'
    elif re.search(r'9 9 9 8 8 8 7 7 7 6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '220'
    elif re.search(r'9 9 9 8 8 8 7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '221'
    elif re.search(r'9 9 9 8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '222'
    elif re.search(r'9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '223'
    elif re.search(r'8 8 8 7 7 7 6 6 6 5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '224'
    elif re.search(r'8 8 8 7 7 7 6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '225'
    elif re.search(r'8 8 8 7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '226'
    elif re.search(r'8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '227'
    elif re.search(r'7 7 7 6 6 6 5 5 5 4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '228'
    elif re.search(r'7 7 7 6 6 6 5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '229'
    elif re.search(r'7 7 7 6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '230'
    elif re.search(r'7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '231'
    elif re.search(r'6 6 6 5 5 5 4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 16:
        label = '232'
    elif re.search(r'6 6 6 5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '233'
    elif re.search(r'6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '234'
    elif re.search(r'5 5 5 4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 12:
        label = '235'
    elif re.search(r'5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '236'
    elif re.search(r'4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 8:
        label = '237'

    elif re.search(r'A A A K K K Q Q Q J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '238'
    elif re.search(r'A A A K K K Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '239'
    elif re.search(r'A A A K K K', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '240'
    elif re.search(r'K K K Q Q Q J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '241'
    elif re.search(r'K K K Q Q Q J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '242'
    elif re.search(r'K K K Q Q Q', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '243'
    elif re.search(r'Q Q Q J J J T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '244'
    elif re.search(r'Q Q Q J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '245'
    elif re.search(r'Q Q Q J J J', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '246'
    elif re.search(r'J J J T T T 9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '247'
    elif re.search(r'J J J T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '248'
    elif re.search(r'J J J T T T', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '249'
    elif re.search(r'T T T 9 9 9 8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '250'
    elif re.search(r'T T T 9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '251'
    elif re.search(r'T T T 9 9 9', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '252'
    elif re.search(r'9 9 9 8 8 8 7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '253'
    elif re.search(r'9 9 9 8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '254'
    elif re.search(r'9 9 9 8 8 8', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '255'
    elif re.search(r'8 8 8 7 7 7 6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '256'
    elif re.search(r'8 8 8 7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '257'
    elif re.search(r'8 8 8 7 7 7', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '258'
    elif re.search(r'7 7 7 6 6 6 5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '259'
    elif re.search(r'7 7 7 6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '260'
    elif re.search(r'7 7 7 6 6 6', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '261'
    elif re.search(r'6 6 6 5 5 5 4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 20:
        label = '262'
    elif re.search(r'6 6 6 5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '263'
    elif re.search(r'6 6 6 5 5 5', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '264'
    elif re.search(r'5 5 5 4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 15:
        label = '265'
    elif re.search(r'5 5 5 4 4 4', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '266'
    elif re.search(r'4 4 4 3 3 3', action_str, re.M | re.I) and \
            len(action_str.split()) == 10:
        label = '267'
    else:
        raise ValueError(f"action str: {action_str} cant match")
    return int(label)


def action2label(action_idx_list: list) -> int:
    """Convert action string into label

    Args:
        action_idx_list (list): action index list

    Returns:
        int: label
    """
    action_idx_list_copy = sorted(action_idx_list, reverse=True)
    action_str_list = card_idx_list_to_str_list(action_idx_list_copy)
    action_str = ' '.join(action_str_list)

    if action_str in policy_label.keys():
        label = policy_label[action_str]
    elif action_str == '':
        assert action_str_list == list()
        label = policy_label['Pass']
    else:
        label = match_label(action_str)
    return label
