'''
Author: hanyu
Date: 2022-08-01 18:09:59
LastEditTime: 2022-08-01 20:41:47
LastEditors: hanyu
Description: policy utils
FilePath: /RL_Lab/policy/policy_utils.py
'''
import numpy as np


def explained_variance(y_pred: np.array, y: np.array) -> float:
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
    return np.nan if var_y == 0 else 1 - np.var(y - y_pred) / var_y
