'''
Author: hanyu
Date: 2021-08-12 03:02:11
LastEditTime: 2022-08-29 15:19:47
LastEditors: hanyu
Description: wrap model
FilePath: /RL_Lab/envs/env_instances/lord/sl_helper/wrap_model.py
'''


def wrap_sl_policy_net():
    import tensorflow.compat.v1 as tf
    from envs.env_instances.lord.sl_helper.sl_policy_cls import Policy
    from config import BASEDIR

    graph_policy = tf.Graph()
    sess_policy = tf.Session(graph=graph_policy)
    with sess_policy.as_default():
        with graph_policy.as_default():
            policy_model = Policy()
            path = BASEDIR + '/env/sl_helper/sl_model/policy-1-780000'
            policy_model.saver.restore(policy_model.session, path)
            # print(f'restore from {path}')
    return policy_model
