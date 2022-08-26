'''
Author: hanyu
Date: 2022-08-04 18:07:05
LastEditTime: 2022-08-05 17:33:07
LastEditors: hanyu
Description: postprocess
FilePath: /RL_Lab/postprocess/postprocess.py
'''

from policy.sample_batch import SampleBatch
import numpy as np


class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"


def compute_gae_from_sample_batch(rollout: SampleBatch,
                                  last_value: float,
                                  last_done: bool,
                                  gamma: float = 0.9,
                                  lam: float = 1.0) -> SampleBatch:
    # rets = np.zeros_like(rollout[SampleBatch.REWARDS])
    advs = np.zeros_like(rollout[SampleBatch.REWARDS])
    steps_per_epoch = rollout.count

    last_gae_lam = 0
    for t in reversed(range(steps_per_epoch)):
        if t == steps_per_epoch - 1:
            next_non_terminal = 1 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1 - rollout[SampleBatch.DONES][t + 1]
            next_value = rollout[SampleBatch.VF_PREDS][t + 1]

        delta = rollout[SampleBatch.REWARDS][
            t] + gamma * next_value * next_non_terminal - rollout[
                SampleBatch.VF_PREDS][t]
        advs[
            t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam

    rollout[
        Postprocessing.VALUE_TARGETS] = advs + rollout[SampleBatch.VF_PREDS]
    rollout[Postprocessing.ADVANTAGES] = (advs - advs.mean()) / (advs.std() +
                                                                 1e-8)
    return rollout
