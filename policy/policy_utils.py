'''
Author: hanyu
Date: 2022-08-01 18:09:59
LastEditTime: 2022-08-25 20:35:16
LastEditors: hanyu
Description: policy utils
FilePath: /RL_Lab/policy/policy_utils.py
'''
import numpy as np


def explained_variance(y: np.array, y_pred: np.array) -> float:
    """Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    Args:
        y_pred (_type_): y pred
        y (_type_): y

    Returns:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y)
    diff_var = np.var(y - y_pred)
    # return np.nan if var_y == 0 else 1 - np.var(y - y_pred) / var_y
    return max(-1, 1 - (diff_var / var_y))
    # _, y_var = tf.nn.moments(y, axes=[0])
    # _, diff_var = tf.nn.moments(y - y_pred, axes=[0])
    # return tf.maximum(-1.0, 1 - (diff_var / y_var))
