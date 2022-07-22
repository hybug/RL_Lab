'''
Author: hanyu
Date: 2022-07-19 16:50:23
LastEditTime: 2022-07-20 15:17:30
LastEditors: hanyu
Description: policy base
FilePath: /RL_Lab/policy/policy_base.py
'''


from configs.config_base import Params


class PolicyBase():
    def __init__(self, params: Params) -> None:
        self.params = params
        self.lr = self.params.policy.lr
        self.train_iter = self.params.policy.train_iters
        self.clip_ratio = self.params.policy.clip_ratio
        self.clip_grads = self.params.policy.clip_grads
        self.target_kl = self.params.policy.target_kl
        self.ent_coef = self.params.policy.ent_coef
        self.v_coef = self.params.policy.v_coef
