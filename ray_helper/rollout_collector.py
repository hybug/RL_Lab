'''
Author: hanyu
Date: 2020-12-24 11:48:45
LastEditTime: 2020-12-24 12:37:59
LastEditors: hanyu
Description: 
FilePath: /test_ppo/ray_helper/rollout_collector.py
'''
# coding: utf-8

import itertools
import os
import time
import ray
import logging
import pickle
from queue import Queue
from threading import Thread

class RolloutCollector:
    """
    just like a dataloader, helper Policy Optimizer for data fetching
    """

    def __init__(self, server_nums, ps=None, policy_evaluator_build_func=None, **kwargs):

        server_nums = int(server_nums)
        '''
        if kwargs['cpu_per_actor'] != -1:
            cpu_p_a = int(kwargs['cpu_per_actor'])
            self._inf_servers = [Evaluator.options(num_cpus=cpu_p_a).remote(kwargs) for _ in range(server_nums)]
        else:
        '''
        # get pramas policy_evaluator need
        (model_kwargs, build_evaluator_model, env_kwargs, build_env) = policy_evaluator_build_func(kwargs)
        from ray_helper.policy_evaluator import Evaluator
        # put ParamServer into kwargs
        kwargs['ps'] = ps
        
        # 
        self._inf_servers = [
            
        ]