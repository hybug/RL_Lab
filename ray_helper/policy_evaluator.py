'''
Author: hanyu
Date: 2020-12-24 11:49:39
LastEditTime: 2020-12-24 12:27:14
LastEditors: hanyu
Description: warp Env and Model, so it is nearly independent with Algorithm
FilePath: /test_ppo/ray_helper/policy_evaluator.py
'''

import ray
import random


class Evaluator:

    def __init__(self, model_func, model_kwargs, envs_func, env_kwargs, kwargs):
        import tensorflow as tf
        # init model and env
        self.model = model_func(model_kwargs)
        self.envs = envs_func(env_kwargs)

        # init checkpioints
        self.ckpt = None
        self.ckpt_dir = kwargs['ckpt_dir']
        # load ckpt preriod from ParamsServer
        self.load_ckpt_period = int(kwargs['load_ckpt_period'])

        # init ParamServer & Session
        self.ps = kwargs['ps']
        self.sess = tf.Session()
        self._init_policy_graph_param(self.sess)

        # segs data generator
        self._data_g = self._ont_inf_step()

    def _init_policy_graph_param(self, sess):
        '''
        description: init policy graph param, from ckpt or from ps
        param {sess: tf sess}
        return {None}
        '''
        import tensorflow as tf
        # get the checkpoint proto from the "checkpoint file"
        # CheckpointState proto with model_checkpoint_path and all_model_checkpoint_paths
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # if the CheckpointState is available
            # init tensorflow saver
            self.saver = tf.train.Saver(
                max_to_keep=None, keep_checkpoint_every_n_hours=6)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # if None, run global_variables_initializer
            sess.run(tf.global_variables_initializer())

        # get init hashing from ParamServer using ray.get() from ray.Actor
        print('getting init hashing from ps')
        self.ckpt = ray.get(self.ps.get_hashing.remote())

    def _one_inf_step(self):
        '''
        description: generator of envs' one-inf-step
        param {*}
        return {*}
        '''
        model = self.model
        sess = self.sess

        step_cnt = 0
        while True:
            if step_cnt % self.load_ckpt_period == 0:
                # load ckpt from ParamServer
                ckpt_hashing = ray.get(self.ps.get_hashing.remote())
                if self.ckpt != ckpt_hashing:
                    # init ckpt hashing not equal curr ckpt hashing
                    # pull params in ParamServer
                    ws = ray.get(self.ps.pull.remote())
                    if ws is not None:
                        # set the new params into model
                        self.model.set_ws(self.sess, ws)
                        print(
                            f'using new ckpt before: {self.ckpt} after: {ckpt_hashing}')
                        # replace the ckpt hashing into the newest
                        self.ckpt = ckpt_hashing
                    else:
                        print('ws from ParamServer is NULL!')

            # env step and get segs
            segs = self.envs.step(sess, model)

            if segs:
                random.shuffle(segs)
                while segs:
                    # split segs into segs_return & segs_left
                    # TODO
                    segs_return, segs = segs[:32], segs[32:]
                    yield segs_return
                step_cnt += 1

    def sample(self):
        '''
        description: generate the segs data from _one_inf_step
        param {*}
        return {*}
        '''
        buffer = [next(self._data_g)]
        return buffer
