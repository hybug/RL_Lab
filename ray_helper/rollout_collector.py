'''
Author: hanyu
Date: 2020-12-24 11:48:45
LastEditTime: 2020-12-29 13:57:48
LastEditors: hanyu
Description: Rollout Collector
FilePath: /test_ppo/ray_helper/rollout_collector.py
'''

import itertools
import os
import time
import ray
import logging
import pickle
from queue import Queue
from threading import Thread


def flatten(list_of_list):
    '''
    description: combinate the list of iterable and return A Iterable
    '''
    return itertools.chain.from_iterable(list_of_list)


def ray_get_and_free(object_ids):
    '''
    description: Call ray.get and then queue the object ids for deletion.
                 This function should be used whenever possible in RLlib, to optimize
                 memory usage. The only exception is when an object_id is shared among
                 multiple readers.
    param {object_ids(ObjectID|List[ObjectID]): Objects ids to fetch and free}
    return {The result of ray.get(object_ids): the object_ids ref objects}
    '''
    free_decay_time = 10.0
    max_free_queue_size = 100

    global _LAST_FREE_TIME
    global _TO_FREE

    # get the object_ids' ref objects using ray.get()
    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _TO_FREE.extend(object_ids)

    # free the object of object_ids' object ref
    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_TO_FREE) > max_free_queue_size or now - _LAST_FREE_TIME > free_decay_time):
        # free a list of IDs from the in-process and plasma object stores.
        # Examples:
        #     >>> x_id = f.remote()
        #     >>> ray.get(x_id)  # wait for x to be created first
        #     >>> free([x_id])  # unpin & delete x globally
        ray.internal.free(_TO_FREE)

        _TO_FREE = []
        _LAST_FREE_TIME = now

    return result


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
        # get the data generator(_inf_server.sample().remote())
        self._data_gen = self._run(**kwargs)
        self.get_one_sample = lambda: self.retrieval_sample(
            next(self._run(num_returns=1, timeout=None))
        )

    def _run(self, **kwargs):
        '''
        description: run the _inf_server in remote worker
        param {
            num_returns: The number of object refs that should be returned.
            timeout: The maximum amount of time in seconds to wait before returning.
            kwargs:
            }
        return {A list of ready object refs(data generator, _inf_server.sample().remote())}
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

            # using ray.wait return the ready working object id
            # and the working object id which not ready
            ready_ids, working_obj_ids = ray.wait(working_obj_ids,
                                                  num_returns=kwargs['num_returns'],
                                                  timeout=kwargs['timeout'])

            for _id in ready_ids:
                iidx = objid2_idx[_id]
                worker_flags[iidx] = True
                # refresh the worker tics
                ddur = time.time() - worker_tics[iidx]
                worker_tics[iidx] = time.time()
                print(f'iidx: {iidx}, dur: {ddur}')
                objid2_idx.pop(_id)

            yield ready_ids

    def __next__(self):
        '''
        description: the data generator
        param {*}
        return {*}
        '''
        return next(self._data_gen)

    @staticmethod
    def retrieval_sample(ready_ids):
        '''
        description: retireval the samples from ready_id obejcts
        param {ready_ids: ready _inf_server ids}
        return {all_segs(samples) from ready_ids object ref(_inf_server)}
        '''

        # get the segs samples using ray.get() &
        # free the ready_ids from the in-process and plasma object stores
        try:
            all_segs = ray_get_and_free(ready_ids)
        except ray.exceptions.UnreconstructableError as e:
            all_segs = []
            logging.info(str(e))
        except ray.exceptions.RayError as e:
            all_segs = []
            logging.info(str(e))

        res = []
        # zip the ready_id(_inf_server id) and the all_segs(samples)
        for idx, (obj_id, segs) in enumerate(zip(ready_ids, all_segs)):
            # get the total segs iterable
            segs = flatten(segs)
            for idx1, seg in enumerate(segs):
                res.append(seg)
        return res


def fetch_one_structure(small_data_collector, cache_struct_path, is_head):
    '''
    description: fetch one sturcture from cache_struct_path,
                 if cache_struc_path not exists, dump from small_data_collector
                 into pkl files.
    param {
        small_data_collector: rollout collector.
        cache_struct_path: the path of cached structure.
        is_head: for distributed training, if local rank not equal zero,
                 just wait for result.
    }
    return {pkl structure(segs sample) from cache_struct_path or
            small_data_collector}
    '''
    sleep_time = 15
    if is_head:
        if os.path.exists(cache_struct_path):
            with open(cache_struct_path, 'rb') as f:
                structure = pickle.load(f)
        else:
            while True:
                segs = small_data_collector.get_one_sample()
                if len(segs) > 0:
                    seg = segs[0]
                    if seg is not None:
                        structure = seg
                        if structure is not None:
                            with open(cache_struct_path, 'wb') as f:
                                pickle.dump(structure, f)
                            del small_data_collector
                            break
                logging.warning(f'NO DATA, SLEEP {sleep_time} seconds!')
                time.sleep(sleep_time)
    else:
        while True:
            if os.path.exists(cache_struct_path):
                with open(cache_struct_path, 'rb') as f:
                    structure = pickle.load(f)
                    if structure is not None:
                        break
            else:
                time.sleep(sleep_time)
    return structure
