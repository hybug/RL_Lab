'''
Author: hanyu
Date: 2021-08-12 03:22:27
LastEditTime: 2022-08-29 16:49:28
LastEditors: hanyu
Description: sl feature engine helper
FilePath: /RL_Lab/envs/env_instances/lord/sl_helper/sl_feature_helper.py
'''
from envs.env_instances.lord.sl_helper import sl_utils as utils


def infoset2json(infoset):
    EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                        8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                        13: 'K', 14: 'A', 17: '2', 20: 'BJ', 30: 'RJ'}

    def _convert_hand_idx_list(hand_idx_list):
        # print(hand_idx_list)
        suit = ['H', 'S', 'C', 'D']
        hand_str_list = [EnvCard2RealCard[c_idx] for c_idx in hand_idx_list]
        hand_str_list_with_suit = list()
        for c_str in hand_str_list:
            if c_str not in ['RJ', 'BJ']:
                for s in suit:
                    if f'{s}{c_str}' in card_set:
                        hand_str_list_with_suit.append(f'{s}{c_str}')
                        # print(f'1: remove {s}{c_str}')
                        card_set.remove(f'{s}{c_str}')
                        break
            else:
                hand_str_list_with_suit.append(c_str)
                # print(f'3: remove {c_str}')
                card_set.remove(c_str)
        # if not hand_str_list_with_suit:
        #     print()
        return hand_str_list_with_suit

    req_data = dict()
    card_set = ['RJ', 'BJ', 'H2', 'S2', 'C2', 'D2', 'HA', 'SA', 'CA',
                'DA', 'HK', 'SK', 'CK', 'DK', 'HQ', 'SQ', 'CQ', 'DQ', 'HJ', 'SJ',
                'CJ', 'DJ', 'HT', 'ST', 'CT', 'DT', 'H9', 'S9', 'C9', 'D9', 'H8',
                'S8', 'C8', 'D8', 'H7', 'S7', 'C7', 'D7', 'H6', 'S6', 'C6', 'D6',
                'H5', 'S5', 'C5', 'D5', 'H4', 'S4', 'C4', 'D4', 'H3', 'S3', 'C3', 'D3']

    if len(infoset.public_cards) == 3:
        public_cards = ' '.join(
            _convert_hand_idx_list(infoset.public_cards))
    card_set.extend(public_cards.split(' '))
    # else:
    #     for c in public_cards.split(' '):
    #         print(f'2: remove {c}')
    #         card_set.remove(c)
    req_data['pub_cards'] = [public_cards, '']

    pos_dict = {'landlord': 0, 'peasant_down': 1, 'peasant_up': 2}
    pos = pos_dict[infoset.player_position]
    req_data['pos'] = pos

    hand_idx_list = infoset.all_handcards[infoset.player_position]
    req_data['hand'] = ' '.join(_convert_hand_idx_list(hand_idx_list))

    seq = list()
    for c_idx_seq in infoset.card_play_action_seq:
        if c_idx_seq:
            seq.append(' '.join(_convert_hand_idx_list(c_idx_seq)))
        else:
            seq.append('Pass')
    req_data['seqs'] = seq

    req_data['bids'] = ["0:1", "1:0", "2:0"]
    req_data['lord_pos'] = utils.bids_2_lord_pos(req_data['bids'])

    return req_data


def json2feature(json_data):
    pos = json_data['pos']
    hand = json_data['hand']
    seqs = json_data['seqs']
    lord_pos = utils.bids_2_lord_pos(json_data['bids'])
    pub_cards = json_data['pub_cards']

    # 当前手牌特征
    hand_list = hand.split(' ')
    hand_rank_list = [card_str if card_str in ['RJ', 'BJ']
                      else card_str[-1] for card_str in hand_list]
    hand_bool_str = utils.rank_to_bool_str(hand_rank_list)

    # 当前玩家角色特征
    if pos == lord_pos:
        role_bool_str = utils.num_3_plane_short(1)
    elif pos == (lord_pos + 1) % 3:
        role_bool_str = utils.num_3_plane_short(2)
    else:
        role_bool_str = utils.num_3_plane_short(3)

    # 上下家手牌数特征
    down_num, up_num, me_played_cards, \
        down_played_cards, up_played_cards = utils.get_down_up_hands(pos, seqs, lord_pos,
                                                                     pub_cards)
    hand_num_bool_str = utils.num_8_plane_short(down_num)
    hand_num_bool_str += '\n'
    hand_num_bool_str += utils.num_8_plane_short(up_num)

    # 底牌信息特征
    assert len(pub_cards) == 2 and pub_cards[0] != ''
    selected_cards_str_list = pub_cards[0].split(' ')
    abandoned_cards_str_list = pub_cards[1].split(
        ' ') if pub_cards[1] != '' else []
    selected_cards_rank_list = [card_str if card_str in ['RJ', 'BJ'] else card_str[-1]
                                for card_str in selected_cards_str_list]
    if abandoned_cards_str_list:
        abandoned_cards_rank_list = [card_str if card_str in ['RJ', 'BJ'] else card_str[-1]
                                     for card_str in abandoned_cards_str_list]
    else:
        abandoned_cards_rank_list = []
    selected_cards_bool_str = utils.rank_to_bool_str(selected_cards_rank_list)
    abandoned_cards_bool_str = utils.rank_to_bool_str(
        abandoned_cards_rank_list)

    # 剩余牌特征
    played_cards_str_list = []
    played_cards_str_list.extend(me_played_cards)
    played_cards_str_list.extend(down_played_cards)
    played_cards_str_list.extend(up_played_cards)
    deck = utils.Deck()
    deck_str_list = [repr(c) for c in deck.cards]
    played_cards_copy = played_cards_str_list.copy()
    for card_str in played_cards_copy:
        deck_str_list.remove(card_str)

    # 额外去掉当前玩家手牌
    hand_copy = hand.split(' ').copy()
    for card_str in hand_copy:
        deck_str_list.remove(card_str)
    left_cards_rank_list = [card_str if card_str in ['RJ', 'BJ'] else card_str[-1]
                            for card_str in deck_str_list]
    # 分割炸弹与其他牌
    left_bomb_rank_list = []
    cnt_dict = {}
    for card_rank in left_cards_rank_list:
        if card_rank not in cnt_dict.keys():
            cnt_dict[card_rank] = 1
        else:
            cnt_dict[card_rank] += 1
    for key, value in cnt_dict.items():
        if value == 4:
            for _ in range(4):
                left_bomb_rank_list.append(key)
                left_cards_rank_list.remove(key)
    left_cards_bool_str = utils.rank_to_bool_str(left_cards_rank_list)
    left_bomb_bool_str = utils.rank_to_bool_str(left_bomb_rank_list)

    # 出牌序列特征
    round_list = []
    count = 0
    for i in range(len(seqs), 0, -1):
        move_list = seqs[i - 1]
        move_rank_list = []
        for card_str in move_list.split(' '):
            if card_str in ['Pass', 'RJ', 'BJ']:
                move_rank_list.append(card_str)
            else:
                move_rank_list.append(card_str[-1])
        move_bool_str = utils.rank_to_bool_str(move_rank_list)
        round_list.append(move_bool_str)
        count += 1
        if len(round_list) == 3:
            break
    one_str, zero_str = utils.get_one_zero_str()
    if len(round_list) < 3:
        for _ in range(len(round_list), 3, 1):
            round_list.append(zero_str)
    round_list = list(reversed(round_list))
    assert len(round_list) == 3

    # 将str转成list
    game_state = []
    hand_bool_list = []
    for char in [float(char) for char in hand_bool_str.strip()]:
        hand_bool_list.append(char)
    assert len(hand_bool_list) == 4 * 15

    role_bool_list = []
    for char in [float(char) for char in role_bool_str.strip()]:
        role_bool_list.append(char)
    assert len(role_bool_list) == 4 * 15

    hand_num_bool_list = []
    for plane_str in hand_num_bool_str.split('\n'):
        plane = []
        for char in [float(char) for char in plane_str.strip()]:
            plane.append(char)
        assert len(plane) == 4 * 15
        hand_num_bool_list.append(plane)
    assert len(hand_num_bool_list) == 4

    left_cards_bool_list = []
    for char in [float(char) for char in left_cards_bool_str.strip()]:
        left_cards_bool_list.append(char)
    assert len(left_cards_bool_list) == 4 * 15

    left_bomb_bool_list = []
    for char in [float(char) for char in left_bomb_bool_str.strip()]:
        left_bomb_bool_list.append(char)
    assert len(left_bomb_bool_list) == 4 * 15

    selected_cards_bool_list = []
    for char in [float(char) for char in selected_cards_bool_str.strip()]:
        selected_cards_bool_list.append(char)
    assert len(selected_cards_bool_list) == 4 * 15

    abandoned_cards_bool_list = []
    for char in [float(char) for char in abandoned_cards_bool_str.strip()]:
        abandoned_cards_bool_list.append(char)
    assert len(abandoned_cards_bool_list) == 4 * 15

    seq_bool_list = []
    for plane_str in round_list:
        plane = []
        for char in [float(char) for char in plane_str.strip()]:
            plane.append(char)
        assert len(plane) == 4 * 15
        seq_bool_list.append(plane)
    assert len(seq_bool_list) == 3

    game_state.append(hand_bool_list)
    game_state.append(role_bool_list)
    game_state.extend(hand_num_bool_list)
    game_state.append(left_cards_bool_list)
    game_state.append(left_bomb_bool_list)
    game_state.extend(seq_bool_list)
    game_state.append(selected_cards_bool_list)
    game_state.append(abandoned_cards_bool_list)
    return game_state
