'''
Author: hanyu
Date: 2022-07-19 16:08:23
LastEditTime: 2022-08-01 18:08:41
LastEditors: hanyu
Description: network utils
FilePath: /RL_Lab/networks/network_utils.py
'''
nn_mapping = dict()


def register_network(name: str):
    def decorator(func):
        nn_mapping[name] = func
        return func

    return decorator


def nn_builder(name: str) -> callable:
    if callable(name):
        return name
    elif name in nn_mapping:
        return nn_mapping[name]
    else:
        raise ValueError('Unknown network name: {}'.format(name))

