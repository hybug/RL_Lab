'''
Author: hanyu
Date: 2020-12-24 11:48:45
LastEditTime: 2020-12-28 13:25:53
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
        (model_kwargs, build_evaluator_model, env_kwargs,
         build_env) = policy_evaluator_build_func(kwargs)
        from ray_helper.policy_evaluator import Evaluator
        # put ParamServer into kwargs
        kwargs['ps'] = ps

        # ray.remote()(class object) same as @ray.remote() decorator
        self._inf_servers = [
            ray.remote(
                num_cpus=kwargs['cpu_per_actor']
            )(Evaluator).remote(
                model_func=build_evaluator_model,
                model_kwargs=model_kwargs,
                envs_func=build_env,
                env_kwargs=env_kwargs, kwargs=kwargs
            ) for _ in range(server_nums)
        ]
        print('inf_erver start!')
        self._data_gen = self._run(**kwargs)
        self.get_one_sample = lambda: self.retrieval_sample(
            next(self._run(num_returns=1, timeout=None))
        )

    def _run(self, **kwargs):
        '''
        description: run the _inf_server in remote worker
        param {
            num_returns:
            timeout:
            kwargs:
            }
        return {*}
        '''
        working_obj_ids = []
        # flags about the worker whether idle
        worker_flags = [True for _ in range(len(self._inf_servers))]
        # time clock of every remote _inf_servers
        worker_tics = [time.time() for _ in range(len(self._inf_servers))]
        # object id to index
        objid2_idx = {}
        while True:
            for idx, flag in enumerate(worker_flags):
                if flag:
                    # if the remote _inf_servers is idle
                    # get the remote _inf_server object
                    server = self._inf_servers[idx]
                    # get the remote sample() object id
                    obj_id = server.sample.remote()
                    working_obj_ids.append(obj_id)
                    # after take out the remote _inf_server, set the remote worker False
                    worker_flags[idx] = False
                    # generate the dict of {obj_id: idx}
                    objid2_idx[obj_id] = idx

            # TODO