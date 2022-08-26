'''
Author: hanyu
Date: 2022-07-19 16:50:23
LastEditTime: 2022-08-05 11:10:35
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
        self.clip_param = self.params.policy.clip_param
        self.clip_grads = self.params.policy.clip_grads
        self.target_kl = self.params.policy.target_kl
        self.ent_coef = self.params.policy.ent_coef
        self.vf_loss_coef = self.params.policy.vf_loss_coef
        self.vf_clip_param = self.params.policy.vf_clip_param
        self.kl_coef = self.params.policy.kl_coef
        self.kl_target = self.params.policy.target_kl
