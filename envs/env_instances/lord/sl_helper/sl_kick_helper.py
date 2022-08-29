'''
Author: hanyu
Date: 2021-08-12 06:48:37
LastEditTime: 2022-08-29 15:20:24
LastEditors: hanyu
Description: sl kick helper
FilePath: /RL_Lab/envs/env_instances/lord/sl_helper/sl_kick_helper.py
'''
import operator
from ray.rllib.utils.framework import try_import_tf
# import tensorflow as tf
from envs.env_instances.lord.sl_helper import sl_utils, sl_labels
import config
import time

tf1, tf, tfv = try_import_tf()
# kick_predict_fn = tf.compat.v1.contrib.predictor.from_saved_model(
#     config.BASEDIR + '/env/sl_helper/sl_model/kick_1_3')
# kick_g = tf1.Graph()
# with tf1.Session(graph=kick_g) as sess:
#     with kick_g.as_default():
#         kick_predict_fn = tf1.saved_model.load(sess, ['serve'],
#             config.BASEDIR + '/env/sl_helper/sl_model/kick_1_3')
# with tf1.Session() as sess:
kick_predict_fn = tf.saved_model.load(config.BASEDIR +
                                      '/envs/env_instances/lord/sl_helper/sl_model/kick_1_3')


def do_predict_kick(features: list, pred_fn):

    features_dict = {}

    for i in range(len(features)):
        fea = features[i]
        features_dict[f'feature{i}'] = tf1.constant([fea])
    with tf1.Session() as sess:
        predictions = kick_predict_fn.signatures['serving_default'](
            **features_dict)
        sess.run(tf1.global_variables_initializer())
        sess.run(tf1.get_collection("saved_model_initializers"))
        time1 = time.time()
        ret = sess.run(predictions['probabilities'])
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++',
              time.time() - time1)
    # predictions = pred_fn.signatures['serving_default'](**features_dict)
    return ret[0]


def get_combo_with_kicker_NN(move_with_X: str, json_data) -> str:
    hand = json_data['hand']
    seqs = json_data['seqs']
    lord_pos = json_data['lord_pos']
    pub_cards = json_data['pub_cards']
    pos = json_data['pos']
    levels = {
        '3': 1,
        '4': 2,
        '5': 3,
        '6': 4,
        '7': 5,
        '8': 6,
        '9': 7,
        'T': 8,
        'J': 9,
        'Q': 10,
        'K': 11,
        'A': 12,
        '2': 13,
        'BJ': 14,
        'RJ': 15
    }

    move_list_with_X = move_with_X.split(' ')
    hand_rank_list = [
        card_str if card_str in ['RJ', 'BJ'] else card_str[-1]
        for card_str in hand.split(' ')
    ]

    # split kick major and kick part & gen feature
    kick_major_list = move_list_with_X[:len(move_list_with_X) - 1]
    kick_X = move_list_with_X[-1]

    # cal major number and part length
    major_num = len(kick_major_list) // 3
    part_length = len(kick_major_list) // major_num

    kick_cards_list = []
    remove_part = []
    # 带牌主体部分小于2时，直接通过NN输出，否则通过NN循环预测输出
    if major_num <= 2:
        part_kick_cards = gen_kick_part(pos, hand_rank_list, seqs, lord_pos,
                                        pub_cards, kick_major_list, kick_X,
                                        None, kick_major_list)
        next_move_list = []
        next_move_list.extend(kick_major_list)
        next_move_list.extend(part_kick_cards)
        next_move_list = sorted(next_move_list,
                                key=lambda x: levels[x],
                                reverse=True)
        next_move = ' '.join(next_move_list)
        return next_move
    else:
        for index in range(major_num):
            part_kick_major_list = kick_major_list[index * part_length:index *
                                                   part_length + part_length]
            part_kick_cards = gen_kick_part(pos, hand_rank_list, seqs,
                                            lord_pos, pub_cards,
                                            part_kick_major_list, kick_X, None,
                                            kick_major_list)
            kick_cards_list.extend(part_kick_cards)

            # 去掉上次预测的major part & kick part
            remove_part.extend(part_kick_cards)
            remove_part.extend(
                kick_major_list[index * part_length:index * part_length +
                                part_length])
            for remove_card in remove_part:
                if remove_card in hand_rank_list:
                    hand_rank_list.remove(remove_card)

        next_move_list = []
        next_move_list.extend(kick_major_list)
        next_move_list.extend(kick_cards_list)
        next_move_list = sorted(next_move_list,
                                key=lambda x: levels[x],
                                reverse=True)
        next_move = ' '.join(next_move_list)
        return next_move


def gen_kick_part(pos: int, hand_rank_list: list, seqs: list, lord_pos: int,
                  pub_cards: list, kick_major_list: list, kick_X: list,
                  kick_predict_fn, total_kick_major_list) -> list:
    one_str, zero_str = sl_utils.get_one_zero_str()
    kick_features = []

    # gen feature list
    # Hand
    hand_rank_list_copy = hand_rank_list.copy()
    for card_rank in kick_major_list:
        if card_rank in hand_rank_list_copy:
            hand_rank_list_copy.remove(card_rank)
    hand_bool_str = sl_utils.rank_to_bool_str(hand_rank_list_copy)
    hand_bool_list = [float(char) for char in hand_bool_str.strip()]

    # Role
    if pos == lord_pos:
        # landlord
        role_bool_list = sl_utils.num_3_plane_list(1)
    elif pos == (lord_pos + 1) % 3:
        # Down
        role_bool_list = sl_utils.num_3_plane_list(2)
    elif pos == (lord_pos + 2) % 3:
        # Up
        role_bool_list = sl_utils.num_3_plane_list(3)

    # The cards number of down player and up player
    down_num, up_num, me_played_cards, down_played_cards, up_played_cards = sl_utils.get_down_up_hands(
        pos, seqs, lord_pos, pub_cards)
    hand_num_bool_list = sl_utils.num_3_plane_list(
        down_num) + sl_utils.num_3_plane_list(up_num)
    assert len(hand_num_bool_list) == 6

    # Left cards and played card
    played_cards_str_list = []
    played_cards_str_list.extend(me_played_cards)
    played_cards_str_list.extend(down_played_cards)
    played_cards_str_list.extend(up_played_cards)
    if pub_cards[1] != '':
        played_cards_str_list.extend(pub_cards[1].split(' '))
    deck = sl_utils.Deck()
    deck_str_list = [repr(c) for c in deck.cards]
    played_cards_copy = played_cards_str_list.copy()
    for card_str in played_cards_copy:
        deck_str_list.remove(card_str)

    # played_cards_rank_list = [card_str if card_str in ['RJ', "BJ"] else card_str[-1] for card_str in
    #                           played_cards_str_list]
    left_cards_rank_list = [
        card_str if card_str in ['RJ', 'BJ'] else card_str[-1]
        for card_str in deck_str_list
    ]

    # played_cards_bool_str = sl_utils.rank_to_bool_plane(played_cards_rank_list)
    left_cards_bool_str = sl_utils.rank_to_bool_str(left_cards_rank_list)
    left_cards_bool_list = [
        float(char) for char in left_cards_bool_str.strip()
    ]
    assert len(left_cards_bool_list) == 4 * 15

    # 带牌主体特征： 三带OR四带 & 单飞OR双飞
    if len(kick_major_list) == 3:
        major_num = 1
        major_type_bool_str = sl_utils.num_2_plane(1)
    elif len(kick_major_list) == 4:
        major_num = 1
        major_type_bool_str = sl_utils.num_2_plane(2)
    elif len(kick_major_list) == 6:
        major_num = 2
        major_type_bool_str = sl_utils.num_2_plane(1)
    major_num_bool_str = sl_utils.num_2_plane(major_num)

    # kick type
    assert kick_X in ['X', 'XX']
    kick_type = len(kick_X)

    if kick_type == 1 and len(kick_major_list) == 3:  # 带牌张数 = 1
        kick_num_bool_str = one_str + '\n' + zero_str + '\n' + zero_str
        kick_num = 1
    elif kick_type == 1 and len(kick_major_list) in [4, 6]:  # 带牌张数 = 2
        kick_num_bool_str = zero_str + '\n' + one_str + '\n' + zero_str
        kick_num = 2
    elif kick_type == 2 and len(kick_major_list) == 3:  # 带牌张数 = 2
        kick_num_bool_str = zero_str + '\n' + one_str + '\n' + zero_str
        kick_num = 2
    elif kick_type == 2 and len(kick_major_list) in [4, 6]:  # 带牌张数 = 4
        kick_num_bool_str = zero_str + '\n' + zero_str + '\n' + one_str
        kick_num = 4
    else:
        kick_num = -1
    assert kick_num != -1

    major_type_bool_list = []
    for plane_str in major_type_bool_str.split('\n'):
        plane = []
        for char in [float(char) for char in plane_str.strip()]:
            plane.append(char)
        assert len(plane) == 4 * 15
        major_type_bool_list.append(plane)
    assert len(major_type_bool_list) == 2

    major_num_bool_list = []
    for plane_str in major_num_bool_str.split('\n'):
        plane = []
        for char in [float(char) for char in plane_str.strip()]:
            plane.append(char)
        assert len(plane) == 4 * 15
        major_num_bool_list.append(plane)
    assert len(major_num_bool_list) == 2

    kick_num_bool_list = []
    for plane_str in kick_num_bool_str.split('\n'):
        plane = []
        for char in [float(char) for char in plane_str.strip()]:
            plane.append(char)
        assert len(plane) == 4 * 15
        kick_num_bool_list.append(plane)
    assert len(kick_num_bool_list) == 3

    kick_features.append(hand_bool_list)
    kick_features.extend(role_bool_list)
    kick_features.extend(hand_num_bool_list)
    kick_features.append(left_cards_bool_list)
    kick_features.extend(major_type_bool_list)
    kick_features.extend(major_num_bool_list)
    kick_features.extend(kick_num_bool_list)

    # do kick predict
    prob_list = do_predict_kick(kick_features, kick_predict_fn)

    label_dict = {i: prob_list[i] for i in range(len(prob_list))}
    label_dict_list_sorted = list(
        sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True))
    counter_kick_label = {
        value: key
        for key, value in sl_labels.kick_label.items()
    }
    for cnt in range(len(label_dict_list_sorted)):
        label = label_dict_list_sorted[cnt][0]
        kick_cards = counter_kick_label.get(str(label), '')
        if not judge_card_inhand(kick_cards, hand_rank_list, kick_major_list,
                                 total_kick_major_list):
            continue
        if judge_bomb_inkick(kick_cards):
            continue
        if len(kick_cards.split(' ')) != kick_num:
            continue
        else:
            return kick_cards.split(' ')
    return []


def judge_card_inhand(kick_cards: str, hand_rank_list: list,
                      major_rank_list: list,
                      total_major_rank_list: list) -> bool:
    kickcards_rank_list = kick_cards.split(' ')
    hand_rank_list_copy = hand_rank_list.copy()
    for card_rank in kickcards_rank_list:
        if card_rank in hand_rank_list_copy and card_rank not in major_rank_list and card_rank not in total_major_rank_list:
            hand_rank_list_copy.remove(card_rank)
        else:
            return False

    return True


def judge_bomb_inkick(kick_cards: str) -> bool:
    kickcards_rank_list = kick_cards.split(' ')
    if 'RJ' in kickcards_rank_list and 'BJ' in kickcards_rank_list:
        return True
    count_dict = {}
    for card_rank in kickcards_rank_list:
        if card_rank not in count_dict.keys():
            count_dict[card_rank] = 1
        else:
            count_dict[card_rank] += 1
            if count_dict[card_rank] == 4:
                return True
    return False


def judge_legal_kick_length(kick_cards: str, kick_major_list: list,
                            kick_type: int) -> bool:
    policy_kick_length = len(kick_cards.split(' '))
    if len(kick_major_list) % 3 == 0:
        legal_kick_length = kick_type * (len(kick_major_list) / 3)
        if len(set(kick_cards.split(' '))) != (len(kick_major_list) / 3):
            return False
    elif len(kick_major_list) % 4 == 0:
        legal_kick_length = kick_type * (len(kick_major_list) / 4) * 2
        if len(set(kick_cards.split(' '))) != 2 * (len(kick_major_list) / 4):
            return False
    return True if policy_kick_length == legal_kick_length else False
