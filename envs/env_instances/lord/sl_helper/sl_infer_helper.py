'''
Author: hanyu
Date: 2021-08-12 03:42:04
LastEditTime: 2022-08-29 15:18:25
LastEditors: hanyu
Description: sl infer helper
FilePath: /RL_Lab/envs/env_instances/lord/sl_helper/sl_infer_helper.py
'''
import heapq
import operator
import re

import config
from envs.env_instances.lord.sl_helper import sl_labels
from envs.env_instances.lord.sl_helper.sl_kick_helper import get_combo_with_kicker_NN
from envs.env_instances.lord.utils.basic_utils import card_str_list_to_idx_list, sort_card_str_list


def sl_policy_infer(infoset, json_data, feature, policy_model):
    prob_list = policy_model.session.run(policy_model.prob, feed_dict={
                                         policy_model.game_state: feature})
    prob_list = [float(prob) for prob in prob_list[0]]
    label_dict = {i: prob_list[i] for i in range(len(prob_list))}
    label_dict_list = list(
        sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True))
    counter_policy_label = {value: key for key,
                            value in sl_labels.policy_label.items()}

    # 预测结果 从大到小 堆排序 取309个数值的索引
    prob_sorted_list = list(
        map(prob_list.index, heapq.nlargest(309, prob_list)))

    # 添加note
    note_list = []
    for label, prob in label_dict_list:
        note_list.append(round(prob, 3))
        if len(note_list) > 5:
            break
    note = str(note_list)

    for i in range(len(prob_sorted_list)):
        if i >= 10:
            config.logging.error('There is error in policy infer')
        label = prob_sorted_list[i]
        policy_next_move = counter_policy_label[str(label)]

        next_move_rank_list = []
        if policy_next_move != 'Pass':
            if re.search(r'X', ' '.join(policy_next_move), re.M | re.I):
                # Need kicker cards
                policy_next_move = get_combo_with_kicker_NN(
                    policy_next_move, json_data)

            for rank in policy_next_move.split(' '):
                if rank == 'RJ' or rank == 'BJ':
                    next_move_rank_list.append(rank[0])
                else:
                    next_move_rank_list.append(rank[-1])

            next_move_idx_list = card_str_list_to_idx_list(next_move_rank_list)
            next_move_idx_list.sort()
            if next_move_idx_list not in infoset.legal_actions:
                continue

            # if re.search(r'X', ' '.join(policy_next_move), re.M | re.I):  # Need kicker cards
            #     # Using MC roll out to get kick part
            #     # is_legal, next_combo = rule.get_combo_with_kicker(
            #     #     current_combo, policy_next_move, hand_cards, legal_moves)

            #     # Using kick NN to get kick part
            #     next_move = kick.get_combo_with_kicker_NN(policy_next_move, pos, hand, seqs, lord_pos, pub_cards,
            #                                               kick_predict_fn)
            #     next_combo = rule.search_rank_combo(
            #         next_move.split(' '), legal_moves)
            #     is_legal = False if next_combo.content == '' else True

            # else:
            #     next_combo = rule.search_rank_combo(
            #         next_move_rank_list, legal_moves)
            #     is_legal = False if next_combo.content == '' else True

            # if not is_legal:
            #     i += 1
            #     continue

        else:
            # if not legal_moves['Pass']:
            #     continue
            # else:
            #     next_combo = rule.Combo('Pass', 0, 'Pass', 0)
            if [] not in infoset.legal_actions:
                continue
            else:
                return [], note

        next_move_str_list = sort_card_str_list(next_move_rank_list)

        return card_str_list_to_idx_list(next_move_str_list), note
